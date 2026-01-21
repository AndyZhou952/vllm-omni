# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.linear import LinearBase

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager


def test_lora_manager_set_active_adapter_end_to_end(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _ToyLinear(LinearBase):
        def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x @ self.weight.t()
            if self.bias is not None:
                y = y + self.bias
            return y

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer
            self.n_slices = 1
            self._lora_a: torch.Tensor | list[torch.Tensor | None] | None = None
            self._lora_b: torch.Tensor | list[torch.Tensor | None] | None = None

        def set_lora(self, index: int, lora_a, lora_b):
            assert index == 0
            self._lora_a = lora_a
            self._lora_b = lora_b

        def reset_lora(self, index: int):
            assert index == 0
            self._lora_a = None
            self._lora_b = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base_layer(x)
            if self._lora_a is None or self._lora_b is None:
                return out

            A = self._lora_a[0] if isinstance(self._lora_a, list) else self._lora_a
            B = self._lora_b[0] if isinstance(self._lora_b, list) else self._lora_b
            if A is None or B is None:
                return out
            return out + (x @ A.t()) @ B.t()

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        if isinstance(layer, _ToyLinear):
            return _DummyBaseLayerWithLoRA(layer)
        return layer

    class _DummyLoRAModel:
        def __init__(self, adapter_id: int, loras: dict[str, LoRALayerWeights]):
            self.id = adapter_id
            self.loras = loras

        def get_lora(self, module_name: str):
            return self.loras.get(module_name)

    def _fake_load_adapter(self: DiffusionLoRAManager, _lora_request: LoRARequest):
        rank = 1
        lora_a = torch.tensor([[1.0, 0.0, 0.0]])
        lora_b = torch.tensor([[1.0], [0.0], [0.0]])
        lora = LoRALayerWeights(
            module_name="transformer.foo",
            rank=rank,
            lora_alpha=rank,
            lora_a=lora_a,
            lora_b=lora_b,
        )
        lora_model = _DummyLoRAModel(adapter_id=11, loras={"transformer.foo": lora})
        peft_helper = type("_PH", (), {"r": rank, "lora_alpha": rank, "target_modules": ["foo"]})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)
    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", _fake_load_adapter)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    base_weight = torch.eye(3)
    pipeline.transformer.foo = _ToyLinear(weight=base_weight)

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    x = torch.tensor([[1.0, 2.0, 3.0]])
    base_out = pipeline.transformer.foo(x)

    req = LoRARequest(lora_name="test", lora_int_id=11, lora_path="/tmp/test")
    manager.set_active_adapter(req, lora_scale=2.0)

    assert isinstance(pipeline.transformer.foo, _DummyBaseLayerWithLoRA)
    out = pipeline.transformer.foo(x)

    expected = base_out + torch.tensor([[2.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)

    manager.set_active_adapter(None)
    out_no_lora = pipeline.transformer.foo(x)
    assert torch.allclose(out_no_lora, base_out)
