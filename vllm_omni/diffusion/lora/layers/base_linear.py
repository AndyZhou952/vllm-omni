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
        for the single-slice, single-LoRA case.
        """
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        original_shape = output.shape
        x_flat, y_flat = x, output
        if x.ndim == 3 and output.ndim == 3:
            x_flat = x.view(-1, x.shape[-1])
            y_flat = output.view(-1, output.shape[-1])

        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output

        A = self.lora_a_stacked[0][0, 0, :, :]  # (rank, in_dim)
        B = self.lora_b_stacked[0][0, 0, :, :]  # (out_dim, rank)

        if A.numel() == 0 or B.numel() == 0:
            return output

        # LoRA shrink & expand as in add_lora_linear()
        delta = (x_flat @ A.t()) @ B.t()
        y_flat = y_flat + delta

        if x.ndim == 3 and output.ndim == 3:
            output = y_flat.view(original_shape)
        else:
            output = y_flat

        return output
