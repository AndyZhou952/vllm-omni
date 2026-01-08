# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PEFT-based LoRA manager for diffusion pipelines.

This manager leverages diffusers' built-in PEFT support and manual PEFT integration
for custom pipelines. Supports both PEFT format (with adapter_config.json) and
safetensors-only adapters, multi-adapter composition, and custom kernel support.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel

logger = init_logger(__name__)

class DiffusionLoRAManager:
    """Per-worker LoRA cache and injector using PEFT."""
    def __init__(
            self,
            pipeline: nn.Module,
            device: torch.device,
            *,
            dtype: torch.dtype,
            max_cached_adapters: int = 1, # to be consistent w/ vLLM limit
            allowed_dirs: list[str] | None = None,
    ):
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else []
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.max_cached_adapters = max_cached_adapters

    # public API
    def set_active_adapter(
            self,
            lora_req: LoRARequest | list[LoRARequest] | None,
    ) -> None:
        """
        Set active adapter(s). Supports single or multi-adpater composition.
        """
        if lora_req is None:
            self._disable_lora()
            return

        if not isinstance(lora_req, list):
            lora_req = [lora_req]

        adapter_names = []
        for req in lora_req:
            try:
                name = self._ensure_loaded(req)
                adapter_names.append(name)
            except Exception as e:
                logger.warning(f"Failed to load adapter: {e}")
                continue

        if not adapter_names:
            logger.wearning("No adapters were successfully loaded")
            return

        self._set_adapters(adapter_names)

    def _ensure_loaded(
            self,
            lora_req: LoRARequest,
    ) -> str:
        """
        Ensure adapter is loaded, return adapter name.
        """
        adapter_name = lora_req.lora_name or f"adapter_{lora_req.lora_int_id}"

        if adapter_name in self.cache:
            # Update last used time and move to end (LRU)
            self.cache.move_to_end(adapter_name)
            self.cache[adapter_name]["last_used"] = time.time()
            logger.debug("LoRA cache hit: %s", adapter_name)
            return adapter_name

        lora_path = self._validate_path(lora_req.lora_path)
        if lora_path is None:
            raise ValueError(f"LoRA path not in whitelist or does not exist: {lora_req.lora_path}")

        self._load_adapter(lora_path, adapter_name, lora_req)

        self.cache[adapter_name] = {
            "path": lora_path,
            "scale": getattr(lora_req, "scale", 1.0),
            "last_used": time.time(),
        }

        self._evict_if_needed()
        logger.info("LoRA adapter loaded: %s from %s", adapter_name, lora_path)
        return adapter_name

    def _load_adapter(
            self,
            lora_path: str,
            adapter_name: str,
            lora_req: LoRARequest
    ) -> None:
        """
        Load adapter using PEFT.
        vllm-omni pipelines use custom kernels
        """
        try:
            from peft import PeftModel, get_peft_model
        except ImportError:
            raise ImportError("peft is not installed which is required for LoRA support.")

        # load adapter weights
        if os.path.isdir(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            peft_config = self._load_peft_config(lora_path)  # peft FT style w/ adapter_config.json
        else:
            if os.path.isdir(lora_path):
                safetensors_files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                if not safetensors_files:
                    raise ValueError(f"No .safetensors files found in {lora_path}")
                if len(safetensors_files) > 1:
                    logger.warning(
                        f"Multiple .safetensors files found in {lora_path}, using {safetensors_files[0]}")
                safetensors_path = os.path.join(lora_path, safetensors_files[0])
            else:
                safetensors_path = lora_path
            peft_config = self._infer_peft_config(safetensors_path)

        for comp_name, component in self._iter_components(): # text_encoder, unet, transformers ex. vae, autoencoder
            # peft wrapping
            if not isinstance(component, PeftModel):
                component = get_peft_model(component, peft_config)
                setattr(self.pipeline, comp_name, component)

            if os.path.isdir(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
                component.load_adapter(lora_path, adapter_name=adapter_name)
            else:
                component.add_adapter(adapter_name, peft_config)
                self._load_safetensors_to_component(component, safetensors_path, adapter_name)

        scale = getattr(lora_req, "scale", 1.0)
        if scale != 1.0:
            self._apply_scale(adapter_name, scale)

    def _register_vllm_custom_module(
            self,
            config: LoraConfig
    ) -> None:
        """
        Register vLLM custom kernel mappings for PEFT
        """
        try:
            from vllm.model_executor.layers.linear import (
                QKVParallelLinear,
                ColumnParallelLinear,
                MergedColumnParallelLinear,
                RowParallelLinear,
                ReplicatedLinear,
            )
            from vllm.lora.layers.column_parallel_linear import (
                QKVParallelLinearWithLoRA,
                ColumnParallelLinearWithLoRA,
                MergedColumnParallelLinearWithLoRA,
            )
            from vllm.lora.layers.row_parallel_linear import (
                RowParallelLinearWithLoRA,
            )
            from vllm.lora.layers.replicated_linear import (
                ReplicatedLinearWithLoRA,
            )

            config._register_custom_module({
                QKVParallelLinear: QKVParallelLinearWithLoRA,
                ColumnParallelLinear: ColumnParallelLinearWithLoRA,
                MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
                RowParallelLinear: RowParallelLinearWithLoRA,
                ReplicatedLinear: ReplicatedLinearWithLoRA,
            })
            logger.debug("Registered vLLM custom kernel mappings for PEFT")
        except ImportError:
            logger.warning("Cannot import vLLM custom kernels, consider checking version compatibility if your pipeline uses custom kernels.")

    def _load_peft_config(
            self,
            lora_dir: str
    ) -> LoraConfig:
        """
        load peft config from adapter_config.json
        """
        import json
        config_path = os.path.join(lora_dir, "adapter_config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = LoraConfig.from_dict(config_dict)

        self._register_vllm_custom_module(config)
        return config

    # TODO - andy: check this separately.
    # TODO: move to helpers
    def infer_peft_config(
            self,
            lora_path: str
    ) -> LoraConfig:
        """
        infer lora config from safetensors without adapter_config
        naming convension:
        - Diffusers/Kohya: lora_down/lora_up
        - PEFT: lora_A/lora_B
        - Embeddings: lora_embedding_A/lora_embedding_B
        """
        from safetensors import safe_open

        with safe_open(lora_path, framework="pt") as f:
            raise NotImplementedError

    #  TODO - andy: check
    def _load_safetensors_to_component(
            self,
            component: PeftModel,
            lora_path: str,
            adapter_name: str,
    ) -> None:
        """
        Load safetensors weights into a PEFT component.
        """
        raise NotImplementedError

    # helpers

    def _iter_components(self):
        """Iterate over pipeline components that support LoRA.
        """
        for comp_name in ["text_encoder", "unet", "transformer"]:
            if hasattr(self.pipeline, comp_name):
                yield comp_name, getattr(self.pipeline, comp_name)

    def _iter_peft_components(self):
        """Iterate over PEFT-wrapped components.
        """
        for comp_name, component in self._iter_components():
            if isinstance(component, PeftModel):
                yield comp_name, component

    # adapters management
    def _manage_adapters(self, action: str, adapter_names: list[str] | None = None) -> None:
        """
        unified adapter management (set/disable).
        """
        for _, component in self._iter_peft_components():
            if action == "set":
                adapters = adapter_names[0] if len(adapter_names) == 1 else adapter_names
                component.set_adapter(adapters)
            elif action == "disable":
                component.disable_adapters()

    def _set_adapters(self, adapter_names: list[str]) -> None:
        """Set active adapters
        """
        self._manage_adapters("set", adapter_names)

    def _disable_lora(self) -> None:
        """Disable all LoRA adapters."""
        self._manage_adapters("disable")

    def _apply_scale(self, adapter_name: str, scale: float) -> None:
        """apply scale factor to adapter weights.
        """
        if scale == 1.0:
            return

        for _, component in self._iter_peft_components():
            for name, module in component.named_modules():
                if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                    module.lora_A[adapter_name].weight.data *= scale
                    module.lora_B[adapter_name].weight.data *= scale

    # cache management
    def _evict_if_needed(self) -> None:
        """Evict least recently used adapters if over capacity.

        Uses simple LRU eviction: when cache exceeds max_cached_adapters,
        removes the oldest (least recently used) adapter.
        """
        while len(self.cache) > self.max_cached_adapters:
            oldest_name, _ = self.cache.popitem(last=False)
            self._unload_adapter(oldest_name)
            logger.info(
                "Evicted LoRA adapter %s to keep within cache limit (%d adapters)",
                oldest_name,
                self.max_cached_adapters,
            )

    def _unload_adapter(self, adapter_name: str) -> None:
        """Unload adapter from pipeline.
        """
        if hasattr(self.pipeline, "delete_adapters"):
            self.pipeline.delete_adapters([adapter_name])
        else:
            # manual: remove from components
            for comp_name, component in self._iter_peft_components():
                try:
                    component.delete_adapter(adapter_name)
                except Exception as e:
                    logger.warning(
                        "Failed to delete adapter %s from %s: %s",
                        adapter_name,
                        comp_name,
                        e,
                    )

    def _validate_path(self, path: str | None) -> str | None:
        """Validate path against whitelist & normalize request
        """
        if not path:
            return None

        real_path = os.path.realpath(path)

        # Check if path exists
        if not os.path.exists(real_path):
            logger.warning("LoRA path does not exist: %s", real_path)
            return None

        # Check whitelist
        if self.allowed_dirs:
            if not any(
                    real_path.startswith(root + os.sep) or real_path == root
                    for root in self.allowed_dirs
            ):
                logger.warning("LoRA path not in whitelist: %s", real_path)
                return None

        return real_path

