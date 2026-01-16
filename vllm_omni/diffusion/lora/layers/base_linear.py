# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


class DiffusionBaseLinearLayerWithLoRA(BaseLinearLayerWithLoRA):
    """
    Diffusion-specific base that overrides apply() to use direct torch matmul
    instead of punica_wrapper.

    Matches the semantics of PunicaWrapperGPU.add_lora_linear():
    - Shrink: buffer = (x @ lora_a.T) * scale
    - Expand: y += buffer @ lora_b.T

    All other functionality (weight management, TP slicing, forward logic)
    is inherited from vLLM's BaseLinearLayerWithLoRA.
    """

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        This matches the exact computation in PunicaWrapperGPU.add_lora_linear()
        for the single-slice, single-LoRA case (diffusion use case).
        """
        # 1. Base layer forward (same as vLLM)
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # 2. Flatten batch/seq dims if 3D (same as vLLM BaseLinearLayerWithLoRA.apply)
        original_shape = output.shape
        x_flat, y_flat = x, output
        if x.ndim == 3 and output.ndim == 3:
            x_flat = x.view(-1, x.shape[-1])
            y_flat = output.view(-1, output.shape[-1])

        # 3. Early return if no LoRA weights
        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output

        # 4. Extract LoRA weights (single slot, single slice for diffusion)
        # lora_a_stacked[0] shape: (max_loras=1, 1, rank, in_dim)
        # lora_b_stacked[0] shape: (max_loras=1, 1, out_dim, rank)
        A = self.lora_a_stacked[0][0, 0, :, :]  # (rank, in_dim)
        B = self.lora_b_stacked[0][0, 0, :, :]  # (out_dim, rank)

        if A.numel() == 0 or B.numel() == 0:
            return output

        # 5. Compute LoRA delta: Δy = (x @ A.T) @ B.T * scale
        # This matches punica_wrapper.add_lora_linear() semantics:
        #   buffer = (x @ lora_a.T) * scale  [shrink]
        #   y += buffer @ lora_b.T           [expand]
        scale = 1.0  # Can be parameterized if needed
        delta = (x_flat @ A.t()) @ B.t() * scale
        y_flat = y_flat + delta

        # 6. Reshape back if needed
        if x.ndim == 3 and output.ndim == 3:
            output = y_flat.view(original_shape)
        else:
            output = y_flat

        return output
