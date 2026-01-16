"""
Configuration module for vLLM-Omni.
"""

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.lora import LoRAConfig

__all__ = [
    "OmniModelConfig",
    "LoRAConfig",
]
