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

    def _resolve_output_slices(
        self,
        num_slices: int,
        out_dim: int,
    ) -> list[tuple[int, int]] | None:
        output_splits = None
        for attr in (
            "output_splits",
            "output_sizes",
            "split_sizes",
            "output_partition_sizes",
        ):
            output_splits = getattr(self, attr, None)
            if output_splits is None:
                output_splits = getattr(self.base_layer, attr, None)
            if output_splits is not None:
                break

        if output_splits is not None:
            if isinstance(output_splits, torch.Tensor):
                output_splits = output_splits.tolist()
            output_splits = list(output_splits)
        else:
            num_heads = getattr(self, "num_heads", None)
            if num_heads is None:
                num_heads = getattr(self.base_layer, "num_heads", None)
            num_kv_heads = getattr(self, "num_kv_heads", None)
            if num_kv_heads is None:
                num_kv_heads = getattr(self.base_layer, "num_kv_heads", None)
            head_dim = getattr(self, "head_dim", None)
            if head_dim is None:
                head_dim = getattr(self, "head_size", None)
            if head_dim is None:
                head_dim = getattr(self.base_layer, "head_dim", None)
            if head_dim is None:
                head_dim = getattr(self.base_layer, "head_size", None)

            if (num_slices == 3 and num_heads is not None and num_kv_heads is not None
                    and head_dim is not None):
                q_size = int(num_heads) * int(head_dim)
                kv_size = int(num_kv_heads) * int(head_dim)
                output_splits = [q_size, kv_size, kv_size]
            elif out_dim % num_slices == 0:
                output_splits = [out_dim // num_slices] * num_slices

        if output_splits is None:
            return None

        if len(output_splits) != num_slices or sum(output_splits) != out_dim:
            return None

        ranges = []
        start = 0
        for size in output_splits:
            end = start + int(size)
            ranges.append((start, end))
            start = end
        return ranges

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        This matches the computation in PunicaWrapperGPU.add_lora_linear()
        and applies all available LoRA slices when present.
        """
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        original_shape = output.shape
        x_flat, y_flat = x, output
        if x.ndim == 3 and output.ndim == 3:
            x_flat = x.view(-1, x.shape[-1])
            y_flat = output.view(-1, output.shape[-1])

        lora_a_stacked = self.lora_a_stacked
        lora_b_stacked = self.lora_b_stacked
        if isinstance(lora_a_stacked, torch.Tensor):
            lora_a_stacked = [lora_a_stacked]
        if isinstance(lora_b_stacked, torch.Tensor):
            lora_b_stacked = [lora_b_stacked]

        if not lora_a_stacked or not lora_b_stacked:
            return output

        out_dim = y_flat.shape[-1]
        for lora_a_group, lora_b_group in zip(lora_a_stacked, lora_b_stacked):
            if lora_a_group.numel() == 0 or lora_b_group.numel() == 0:
                continue
            if lora_a_group.dim() == 2:
                lora_a_group = lora_a_group.unsqueeze(0).unsqueeze(0)
            elif lora_a_group.dim() == 3:
                lora_a_group = lora_a_group.unsqueeze(0)
            if lora_b_group.dim() == 2:
                lora_b_group = lora_b_group.unsqueeze(0).unsqueeze(0)
            elif lora_b_group.dim() == 3:
                lora_b_group = lora_b_group.unsqueeze(0)

            num_loras, num_slices = lora_a_group.shape[0], lora_a_group.shape[1]
            slice_ranges = self._resolve_output_slices(num_slices, out_dim)
            for lora_idx in range(num_loras):
                for slice_idx in range(num_slices):
                    A = lora_a_group[lora_idx, slice_idx]  # (rank, in_dim)
                    B = lora_b_group[lora_idx, slice_idx]  # (out_dim, rank)

                    if A.numel() == 0 or B.numel() == 0:
                        continue

                    # LoRA shrink & expand as in add_lora_linear()
                    delta = (x_flat @ A.t()) @ B.t()
                    if delta.shape[-1] == out_dim:
                        y_flat = y_flat + delta
                        continue
                    if slice_ranges is None:
                        continue
                    start, end = slice_ranges[slice_idx]
                    if delta.shape[-1] != end - start:
                        continue
                    y_flat[:, start:end] = y_flat[:, start:end] + delta

        if x.ndim == 3 and output.ndim == 3:
            output = y_flat.view(original_shape)
        else:
            output = y_flat

        return output
