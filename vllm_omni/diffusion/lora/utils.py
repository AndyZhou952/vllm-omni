# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import regex as re
from typing import TYPE_CHECKING

from torch import nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def iter_pipeline_components(pipeline: nn.Module):
    for comp_name in ["text_encoder", "unet", "transformer"]:
        if hasattr(pipeline, comp_name):
            yield comp_name, getattr(pipeline, comp_name)


def match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """ from vllm/lora/model_manager.py _match_target_modules
    """
    return any(
        re.match(r".*\.{target_module}$".format(target_module=target_module), module_name)
        or target_module == module_name
        for target_module in target_modules
    )
