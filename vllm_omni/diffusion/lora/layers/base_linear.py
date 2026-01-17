# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


class DiffusionBaseLinearLayerWithLoRA(BaseLinearLayerWithLoRA):
    """
    Diffusion-specific base that overrides apply() to use direct torch matmul
    instead of punica_wrapper.

    punica_wrapper is used to hold multiple LoRA slots and slices efficiently.

    This matches the semantics of PunicaWrapperGPU.add_lora_linear():
    - Shrink: buffer = (x @ lora_a.T)
    - Expand: y += buffer @ lora_b.T

    All other functionality (weight management, TP slicing, forward logic)
    is inherited from vLLM's BaseLinearLayerWithLoRA.
    """

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        This matches the exact computation in PunicaWrapperGPU.add_lora_linear()
        for the single-LoRA case. For packed projections (e.g. fused QKV), we
        apply LoRA per-slice using `output_slices`.
        """
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        if not hasattr(self, "lora_a_stacked") or not hasattr(self, "lora_b_stacked"):
            return output
        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output

        # In fully-sharded LoRA mode, vLLM uses an all-gather between shrink and
        # expand for ColumnParallelLinear variants. This diffusion path doesn't
        # implement that communication yet.
        if getattr(self, "lora_config", None) is not None:
            if self.lora_config.fully_sharded_loras and self.tp_size > 1:
                raise NotImplementedError(
                    "Diffusion LoRA apply() does not support fully_sharded_loras "
                    "with tensor parallelism yet."
                )

        original_shape = output.shape
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = output.reshape(-1, output.shape[-1])

        output_slices = getattr(self, "output_slices", None)
        if output_slices is None:
            # Fallback: infer slice sizes from the allocated tensors.
            output_slices = tuple(lora_b.shape[2] for lora_b in self.lora_b_stacked)

        if len(output_slices) != len(self.lora_a_stacked) or len(output_slices) != len(self.lora_b_stacked):
            raise RuntimeError(
                "LoRA slice metadata mismatch: "
                f"output_slices={len(output_slices)}, "
                f"lora_a_stacked={len(self.lora_a_stacked)}, "
                f"lora_b_stacked={len(self.lora_b_stacked)}"
            )

        offset = 0
        for slice_idx, slice_size in enumerate(output_slices):
            A = self.lora_a_stacked[slice_idx][0, 0, :, :]  # (rank, in_dim)
            B = self.lora_b_stacked[slice_idx][0, 0, :, :]  # (out_dim, rank)

            if A.numel() == 0 or B.numel() == 0:
                offset += slice_size
                continue

            # LoRA shrink & expand as in add_lora_linear():
            #   buffer = (x @ A.T)
            #   y += buffer @ B.T
            delta = (x_flat @ A.t()) @ B.t()
            y_flat[:, offset : offset + slice_size] = y_flat[:, offset : offset + slice_size] + delta
            offset += slice_size

        return y_flat.view(original_shape)
