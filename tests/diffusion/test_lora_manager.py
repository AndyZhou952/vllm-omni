# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.layers.linear import LinearBase

from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager


@dataclass
class _DummyLoRAConfig:
    fully_sharded_loras: bool = False


class _DummyQuantMethod:
    def __init__(self, weight: torch.Tensor):
        self._weight = weight

    def apply(self, _base_layer, x: torch.Tensor, bias: torch.Tensor | None):
        y = x @ self._weight.t()
        if bias is not None:
            y = y + bias
        return y


class _DummyLoRALayer:
    def __init__(self, n_slices: int, output_slices: tuple[int, ...]):
        self.n_slices = n_slices
        self.output_slices = output_slices
        self.set_calls: list[tuple[list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]] = []
        self.reset_calls: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1


class _FakeLinearBase(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def test_diffusion_base_linear_apply_multi_slice():
    # Build a fake diffusion LoRA layer with 2 slices and rank=2.
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 3
    out_slices = (2, 1)
    out_dim = sum(out_slices)
    rank = 2

    # Base weight: identity-ish mapping to make base output easy to reason about.
    base_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    # Allocate stacked weights: (max_loras=1, 1, rank, in_dim) and (1, 1, out_slice, rank)
    a0 = torch.zeros((1, 1, rank, in_dim))
    b0 = torch.zeros((1, 1, out_slices[0], rank))
    a1 = torch.zeros((1, 1, rank, in_dim))
    b1 = torch.zeros((1, 1, out_slices[1], rank))

    # Slice 0: delta0 = (x @ A0.T) @ B0.T
    A0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    a0[0, 0, :, :] = A0
    b0[0, 0, :, :] = B0

    # Slice 1: delta1 = (x @ A1.T) @ B1.T
    A1 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # (2, 3)
    B1 = torch.tensor([[2.0, 0.0]])  # (1, 2)
    a1[0, 0, :, :] = A1
    b1[0, 0, :, :] = B1

    layer.lora_a_stacked = (a0, a1)
    layer.lora_b_stacked = (b0, b1)
    layer.output_slices = out_slices

    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = layer.apply(x)

    # Base output is identity: [1,2,3]
    base_out = x @ base_weight.t()
    # delta0:
    # (x @ A0.T) = [1,2]
    # [1,2] @ B0.T = [1,2]
    delta0 = torch.tensor([[1.0, 2.0]])
    # delta1:
    # (x @ A1.T) = [3,1]
    # [3,1] @ B1.T = [6]
    delta1 = torch.tensor([[6.0]])
    expected = torch.cat([base_out[:, :2] + delta0, base_out[:, 2:3] + delta1], dim=-1)
    assert torch.allclose(out, expected)


def test_lora_manager_supported_modules_are_stable_with_wrapped_layers(monkeypatch):
    # Simulate a pipeline that already contains LoRA wrappers where the original
    # LinearBase is nested under ".base_layer".
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = _DummyBaseLayerWithLoRA(_FakeLinearBase())

    # vLLM helper would see only the nested LinearBase and yield "base_layer".
    assert get_supported_lora_modules(pipeline) == ["base_layer"]

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=1,
    )

    assert "foo" in manager._supported_lora_modules
    assert "base_layer" not in manager._supported_lora_modules


def test_lora_manager_replace_layers_does_not_rewrap_base_layer(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        if isinstance(layer, _FakeLinearBase):
            return _DummyBaseLayerWithLoRA(layer)
        return layer

    replace_calls: list[str] = []

    def _fake_replace_submodule(root: torch.nn.Module, module_name: str, submodule: torch.nn.Module):
        replace_calls.append(module_name)
        setattr(root, module_name, submodule)

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(manager_mod, "replace_submodule", _fake_replace_submodule)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()

    manager._replace_layers_with_lora(peft_helper)
    manager._replace_layers_with_lora(peft_helper)

    # Only the top-level layer should have been replaced; nested ".base_layer"
    # must be skipped to avoid nesting LoRA wrappers.
    assert replace_calls == ["foo"]


def test_lora_manager_activates_fused_lora_on_packed_layer():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=1,
    )

    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(2, 1, 1))
    manager._lora_modules = {"transformer.blocks.0.attn.to_qkv": packed_layer}

    rank = 2
    A = torch.ones((rank, 4))
    B = torch.arange(0, sum(packed_layer.output_slices) * rank, dtype=torch.float32).view(-1, rank)
    lora = LoRALayerWeights(
        module_name="transformer.blocks.0.attn.to_qkv",
        rank=rank,
        lora_alpha=rank,
        lora_a=A,
        lora_b=B,
    )
    manager._registered_adapters = {7: type("LM", (), {"id": 7, "loras": {"transformer.blocks.0.attn.to_qkv": lora}, "get_lora": lambda self, k: self.loras.get(k)})()}
    manager._adapter_scales = {7: 0.5}

    manager._activate_adapter(7)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
    assert isinstance(lora_a_list, list)
    assert isinstance(lora_b_list, list)
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    assert all(torch.allclose(a, A) for a in lora_a_list)
    # B should be split into 3 slices and scaled.
    b0, b1, b2 = lora_b_list
    assert b0.shape[0] == 2 and b1.shape[0] == 1 and b2.shape[0] == 1
    assert torch.allclose(torch.cat([b0, b1, b2], dim=0), B * 0.5)


def test_lora_manager_activates_packed_lora_from_sublayers():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=1,
    )

    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(2, 1, 1))
    manager._lora_modules = {"transformer.blocks.0.attn.to_qkv": packed_layer}

    rank = 2
    loras: dict[str, LoRALayerWeights] = {}
    for name, out_dim in zip(["to_q", "to_k", "to_v"], [2, 1, 1]):
        loras[f"transformer.blocks.0.attn.{name}"] = LoRALayerWeights(
            module_name=f"transformer.blocks.0.attn.{name}",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)) * (1 if name == "to_q" else 2),
            lora_b=torch.ones((out_dim, rank)) * (3 if name == "to_q" else 4),
        )

    manager._registered_adapters = {1: type("LM", (), {"id": 1, "loras": loras, "get_lora": lambda self, k: self.loras.get(k)})()}
    manager._adapter_scales = {1: 2.0}

    manager._activate_adapter(1)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
    assert isinstance(lora_a_list, list)
    assert isinstance(lora_b_list, list)
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    # Scale should apply to B only.
    assert torch.allclose(lora_b_list[0], torch.ones((2, rank)) * 3 * 2.0)
    assert torch.allclose(lora_b_list[1], torch.ones((1, rank)) * 4 * 2.0)
    assert torch.allclose(lora_b_list[2], torch.ones((1, rank)) * 4 * 2.0)
