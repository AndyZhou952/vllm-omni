# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LoRA manager for diffusion pipelines.

Requires PEFT format adapters with adapter_config.json.
Uses vLLM's custom LoRA layers (QKVParallelLinearWithLoRA, etc.) for efficient inference.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_supported_lora_modules

from vllm_omni.diffusion.lora.utils import (
    iter_pipeline_components,
    match_target_modules,
)

if TYPE_CHECKING:
    from vllm.lora.models import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.config.lora import LoRAConfig
    from vllm.lora.lora_weights import LoRALayerWeights

logger = init_logger(__name__)


class DiffusionLoRAManager:
    """Per-worker LoRA cache and injector."""

    def __init__(
            self,
            pipeline: nn.Module,
            device: torch.device,
            *,
            dtype: torch.dtype,
            max_cached_adapters: int = 1,
    ) -> None:
        """Initialize the LoRA manager.

        Args:
            pipeline: The diffusion pipeline (may be custom, not necessarily DiffusionPipeline)
            device: Device to load adapters on
            dtype: Data type for adapter weights
            max_cached_adapters: Maximum number of adapters to cache (LRU eviction)
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self._registered_adapters: OrderedDict[int, dict] = OrderedDict()
        self.max_cached_adapters = max_cached_adapters
        self.modules: dict[str, BaseLayerWithLoRA] = {}

    # === Public API ===

    def set_active_adapter(
            self,
            lora_req: LoRARequest | list[LoRARequest] | None,
    ) -> None:
        """Set active adapter(s).
        """
        if lora_req is None:
            self.remove_all_adapters()
            return

        if not isinstance(lora_req, list):
            lora_req = [lora_req]

        adapter_ids = []
        for req in lora_req:
            try:
                adapter_id = self._ensure_loaded(req)
                adapter_ids.append(adapter_id)
            except Exception as exc:
                logger.warning("Failed to load LoRA adapter: %s", exc)
                continue

        if not adapter_ids:
            logger.warning("No adapters were successfully loaded")
            return

        if adapter_ids:
            self._activate_adapter(adapter_ids[0] if len(adapter_ids) == 1 else adapter_ids)

    # === Loading & Caching ===

    def _ensure_loaded(self, lora_req: LoRARequest) -> int:
        """Validate and ensure adapter is loaded, return adapter ID.
        """
        adapter_id = lora_req.lora_int_id

        # check cache
        if adapter_id in self._registered_adapters:
            self._registered_adapters.move_to_end(adapter_id)
            self._registered_adapters[adapter_id]["last_used"] = time.time()
            logger.debug("LoRA cache hit: %s (id: %d)", lora_req.lora_name or f"adapter_{adapter_id}", adapter_id)
            return adapter_id

        from vllm.lora.utils import get_adapter_absolute_path
        lora_path = get_adapter_absolute_path(lora_req.lora_path)
        self._registered_adapters[adapter_id] = {
            "path": lora_path,
            "last_used": time.time(),
            "name": lora_req.lora_name or f"adapter_{adapter_id}",
        }

        self._load_adapter(lora_path, adapter_id, lora_req)

        self.remove_oldest_adapter()
        adapter_name = lora_req.lora_name or f"adapter_{adapter_id}"
        logger.info("LoRA adapter loaded: %s (id: %d) from %s", adapter_name, adapter_id, lora_path)
        return adapter_id

    def _load_adapter(self, lora_path: str, adapter_id: int, lora_req: LoRARequest) -> None:
        """Load adapter using vLLM's approach.
        Requires PEFT format with adapter_config.json.
        Reuses vLLM's LoRA loading functions and layer replacement mechanism.
        """
        expected_modules = set(get_supported_lora_modules(self.pipeline))

        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # not needed for diffusion
        )

        lora_config = LoRAConfig(
            max_lora_rank=512,
            max_loras=self.max_cached_adapters,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
        )
        peft_helper.validate_legal(lora_config)

        lora_model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=expected_modules,
            peft_helper=peft_helper,
            lora_model_id=lora_req.lora_int_id,
            device="cpu",
            dtype=self.dtype,
        )

        # Replace layers only if not already replaced (first adapter load)
        if not self.modules:
            self._replace_layers_with_lora(peft_helper, lora_config)

        # Store lora_model in dict entry (which is guaranteed to exist now)
        self._registered_adapters[adapter_id]["lora_model"] = lora_model

    # === Layer Management ===

    def _replace_layers_with_lora(self, peft_helper: PEFTHelper, lora_config: LoRAConfig) -> None:
        """Replace matching layers with LoRA versions.

        This is called once during first adapter load. Component separation
        is unique to diffusion pipelines (text_encoder, unet, transformer).
        """
        from vllm.lora.utils import from_layer, replace_submodule

        target_modules = peft_helper.target_modules
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        # Process each component separately
        for comp_name, component in iter_pipeline_components(self.pipeline):
            for module_name, module in component.named_modules():
                # module_name is relative to component (e.g., "layers.0.attn.to_q")

                # Check if this module matches target_modules
                # target_modules may be component-prefixed (e.g., "unet.to_q")
                # or component-agnostic (e.g., "to_q")
                matches = False
                for target in target_modules:
                    # Remove component prefix from target if present
                    if target.startswith(f"{comp_name}."):
                        target_suffix = target[len(comp_name) + 1:]
                        if match_target_modules(module_name, [target_suffix]):
                            matches = True
                            break
                    elif match_target_modules(module_name, [target]):
                        matches = True
                        break

                if not matches:
                    continue

                # Store with component prefix to match PEFT format
                module_path = f"{comp_name}.{module_name}"
                if module_path in self.modules:
                    continue

                new_module = from_layer(
                    module,
                    max_loras=self.max_cached_adapters,
                    lora_config=lora_config,
                    packed_modules_list=[],
                    model_config=None,
                )

                if new_module != module and isinstance(new_module, BaseLayerWithLoRA):
                    replace_submodule(component, module_name, new_module)
                    self.modules[module_path] = new_module
                    logger.debug(f"Replaced {module_path} with LoRA version")

    def _get_lora_layer_weights(
            self, lora_model: LoRAModel, module_path: str
    ) -> LoRALayerWeights | None:
        """Get LoRA weights for a module path.
        """
        if lora_model.check_lora_name(module_path):
            return lora_model.get_lora(module_path)

        # Extract component and relative module name
        parts = module_path.split(".", 1)
        if len(parts) == 2:
            comp_name, module_name = parts
            # Try without component prefix (some PEFT adapters may store this way)
            if lora_model.check_lora_name(module_name):
                return lora_model.get_lora(module_name)

        return None

    def _apply_lora_weights(self, lora_model: LoRAModel) -> None:
        """Apply LoRA weights from LoRAModel to replaced modules.
        """
        for module_path, module in self.modules.items():
            if not isinstance(module, BaseLayerWithLoRA):
                continue

            lora_weights = self._get_lora_layer_weights(lora_model, module_path)

            if lora_weights:
                module.set_lora(0, lora_weights.lora_a, lora_weights.lora_b)
                logger.debug(f"Applied LoRA weights to {module_path}")
            else:
                # Reset if no weights found
                module.reset_lora(0)
                logger.debug(f"No LoRA weights found for {module_path}, reset")

    # === Adapter Activation ===

    def _activate_adapter(self, adapter_id: int | list[int]) -> None:
        """Activate adapter(s) by loading weights into LoRA modules.
        """
        if isinstance(adapter_id, list):
            # Multi-adapter composition not yet supported
            adapter_id = adapter_id[0]
            logger.warning("Multi-adapter composition not yet supported, using first adapter")

        # adapter_id is guaranteed to be in _registered_adapters (ensured by _ensure_loaded)
        lora_model = self._registered_adapters[adapter_id].get("lora_model")
        if lora_model is None:
            adapter_name = self._registered_adapters[adapter_id].get("name", f"adapter_{adapter_id}")
            logger.warning(f"LoRA model not found for adapter {adapter_name} (id: {adapter_id})")
            return

        self._apply_lora_weights(lora_model)

        adapter_name = self._registered_adapters[adapter_id].get("name", f"adapter_{adapter_id}")
        logger.debug(f"Activated adapter: {adapter_name} (id: {adapter_id})")

    def remove_all_adapters(self) -> None:
        """Remove all adapters from cache and reset LoRA modules.
        """
        # Reset LoRA weights (set to zero or disable)
        for module_path, module in self._iter_lora_modules():
            if isinstance(module, BaseLayerWithLoRA):
                module.reset_lora(0)

        self._registered_adapters.clear()

        logger.debug("Removed all adapters")

    # === Cache Management ===

    def remove_oldest_adapter(self) -> None:
        """Remove the oldest adapter from cache (LRU eviction).
        """
        while len(self._registered_adapters) > self.max_cached_adapters:
            oldest_id, cache_data = self._registered_adapters.popitem(last=False)
            oldest_name = cache_data.get("name", f"adapter_{oldest_id}")
            self._unload_adapter(oldest_id, oldest_name)
            logger.info(
                "Evicted LoRA adapter %s (id: %d) to keep within cache limit (%d adapters)",
                oldest_name,
                oldest_id,
                self.max_cached_adapters,
            )

    def _unload_adapter(self, adapter_id: int, adapter_name: str | None = None) -> None:
        # Reset LoRA modules (they remain replaced, just weights cleared)
        for module_path, module in self._iter_lora_modules():
            if isinstance(module, BaseLayerWithLoRA):
                module.reset_lora(0)

        if adapter_name is None:
            # Try to get name from cache if still available, otherwise use default format
            cache_entry = self._registered_adapters.get(adapter_id)
            adapter_name = cache_entry.get("name", f"adapter_{adapter_id}") if cache_entry else f"adapter_{adapter_id}"
        logger.debug(f"Unloaded adapter: {adapter_name} (id: {adapter_id})")

    # === Utilities ===

    def _iter_lora_modules(self):
        for module_path, module in self.modules.items():
            if isinstance(module, BaseLayerWithLoRA):
                yield module_path, module
