# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import List

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_omni.config.lora import LoRAConfig

from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)

def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """ from vllm/lora/model_manager.py _match_target_modules, helper function
    """
    import regex as re

    return any(
        re.match(r".*\.{target_module}$".format(target_module=target_module), module_name)
        or target_module == module_name
        for target_module in target_modules
    )

def _expand_expected_modules_for_merged_projections(
        supported_modules: set[str]
) -> set[str]:
    expanded = set(supported_modules)

    # known patterns: merged projections accept their separate counterparts
    if "add_kv_proj" in supported_modules:
        expanded.update(["add_k_proj", "add_v_proj", "add_q_proj"])
    if "to_qkv" in supported_modules:
        expanded.update(["to_q", "to_k", "to_v"])

    return expanded

def from_layer_diffusion(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: List[str],
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    """
    Diffusion-specific layer replacement. similar to vLLM's `from_layer`
    """
    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
    ]

    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance = lora_cls(layer)  # type: ignore[arg-type]
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance

    return layer
