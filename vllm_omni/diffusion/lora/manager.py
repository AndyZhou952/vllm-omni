# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import (
    get_adapter_absolute_path,
    get_supported_lora_modules,
    from_layer,
    replace_submodule,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


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


class DiffusionLoRAManager:
    """Manager for LoRA adapters in diffusion models.

    Reuses vLLM's LoRA infrastructure, adapted for diffusion pipelines.
    Uses LRU cache management similar to LRUCacheLoRAModelManager.
    """

    def __init__(
            self,
            pipeline: nn.Module,
            device: torch.device,
            dtype: torch.dtype,
            max_cached_adapters: int = 1,
            static_lora_path: str | None = None,
            static_lora_scale: float = 1.0,
    ):
        """
        Initialize the DiffusionLoRAManager.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype

        # LRU-style cache management
        self.max_cached_adapters = max_cached_adapters  # max_cpu_loras
        self._registered_adapters: dict[int, LoRAModel] = {}  # adapter_id -> LoRAModel
        self._active_adapter_id: int | None = None
        self._adapter_scales: dict[int, float] = {}  # adapter_id -> external scale

        # LRU cache tracking (adapter_id -> last_used_time)
        self._adapter_access_order: OrderedDict[int, float] = OrderedDict()
        # Pinned adapters are not evicted
        self._pinned_adapters: set[int] = set()

        # track replaced modules
        # key: full module name (component.module.path); value: LoRA layer
        self._lora_modules: dict[str, BaseLayerWithLoRA] = {}

        # create punica wrapper for LoRA computation
        # estimate max_num_batched_tokens from pipeline scheduler config
        max_image_seq_len = 4096  # default
        if hasattr(pipeline, "scheduler") and hasattr(pipeline.scheduler, "config"):
            scheduler_config = pipeline.scheduler.config
            if isinstance(scheduler_config, dict):
                max_image_seq_len = scheduler_config.get("max_image_seq_len", 4096)
            elif hasattr(scheduler_config, "get"):
                max_image_seq_len = scheduler_config.get("max_image_seq_len", 4096)

        max_num_batched_tokens = math.ceil(max_image_seq_len / 8) * 8

        logger.info(
            "Initializing DiffusionLoRAManager: device=%s, dtype=%s, max_cached_adapters=%d, "
            "max_num_batched_tokens=%d, static_lora_path=%s",
            device, dtype, max_cached_adapters, max_num_batched_tokens, static_lora_path
        )

        self.punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens=max_num_batched_tokens,
            max_batches=1,  # single request
            device=self.device,
            max_loras=1,  # single lora
        )

        if static_lora_path is not None:
            logger.info("Loading static LoRA from %s with scale %.2f", static_lora_path, static_lora_scale)
            static_request = LoRARequest(
                lora_name = 'static',
                lora_int_id = 1,
                lora_path = static_lora_path,
            )
            self.set_active_adapter(static_request, static_lora_scale)

    def set_active_adapter(self, lora_request: LoRARequest | None, lora_scale: float = 1.0) -> None:
        """Set the active LoRA adapter for the pipeline.

        Args:
            lora_request: The LoRA request, or None to deactivate all adapters.
            lora_scale: The external scale for the LoRA adapter.
        """
        if lora_request is None:
            logger.debug("No lora_request provided, deactivating all LoRA adapters")
            self._deactivate_all_adapters()
            return

        adapter_id = lora_request.lora_int_id
        logger.debug(
            "Setting active adapter: id=%d, name=%s, path=%s, scale=%.2f, cache_size=%d/%d",
            adapter_id, lora_request.lora_name, lora_request.lora_path, lora_scale,
            len(self._registered_adapters), self.max_cached_adapters
        )
        if adapter_id not in self._registered_adapters:
            logger.info("Loading new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)
            self.add_lora(lora_request, lora_scale)
        else:
            logger.debug("Adapter %d already loaded, activating", adapter_id)

            # update access order
            self._adapter_scales[adapter_id] = lora_scale
            self._adapter_access_order[adapter_id] = time.time()
            self._adapter_access_order.move_to_end(adapter_id)

        self._activate_adapter(adapter_id)

    def _load_adapter(
            self,
            lora_request: LoRARequest,
    ) -> tuple[LoRAModel, PEFTHelper]:

        supported_lora_modules = set(get_supported_lora_modules(self.pipeline))
        expected_lora_modules = _expand_expected_modules_for_merged_projections(
            supported_lora_modules
        )
        logger.debug("Supported LoRA modules: %s", expected_lora_modules)

        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)

        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # no need in diffusion
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
        )

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r, peft_helper.lora_alpha, peft_helper.target_modules
        )

        lora_model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",  # consistent w/ vllm's behavior
            dtype=self.dtype,
            model_vocab_size=None,
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
            weights_mapper=None,
        )

        logger.info(
            "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
            lora_model.id, len(lora_model.loras), list(lora_model.loras.keys())
        )

        for lora in lora_model.loras.values():
            lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

        return lora_model, peft_helper

    def _replace_layers_with_lora(self, peft_helper: PEFTHelper) -> None:
        target_modules = peft_helper.target_modules
        if not isinstance(target_modules, list):
            target_modules = [target_modules]

        # dummy lora config
        lora_config = LoRAConfig(
            max_lora_rank=peft_helper.r,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        if hasattr(self.pipeline, "transformer"):
            for module_name, module in self.pipeline.transformer.named_modules(remove_duplicate=False):
                full_module_name = f"transformer.{module_name}"
                if not _match_target_modules(module_name, target_modules):
                    continue

                if full_module_name in self._lora_modules:
                    logger.debug("Layer %s already replaced, skipping", full_module_name)
                    continue

                lora_layer = from_layer(
                    layer=module,
                    max_loras=1,
                    lora_config=lora_config,
                    packed_modules_list=[],
                    model_config=None,
                )

                if lora_layer is not module and isinstance(lora_layer, BaseLayerWithLoRA):
                    replace_submodule(self.pipeline.transformer, module_name, lora_layer)
                    self._lora_modules[full_module_name] = lora_layer
                    lora_layer.set_mapping(self.punica_wrapper)
                    logger.debug("Replaced layer: %s -> %s", full_module_name, type(lora_layer).__name__)

    def _activate_adapter(self, adapter_id: int) -> None:
        if self._active_adapter_id == adapter_id:
            logger.debug("Adapter %d already active, skipping", adapter_id)
            return

        logger.info("Activating adapter: id=%d", adapter_id)
        lora_model = self._registered_adapters[adapter_id]

        # activate weights in each LoRA layer
        for full_module_name, lora_layer in self._lora_modules.items():
            # try full name first
            lora_weights = lora_model.get_lora(full_module_name)
            # fallbacks
            if lora_weights is None:
                # relative name
                component_relative_name = full_module_name.split(".", 1)[
                    -1] if "." in full_module_name else full_module_name
                lora_weights = lora_model.get_lora(component_relative_name)
            if lora_weights is None:
                # try just the suffix
                module_suffix = full_module_name.split(".")[-1]
                lora_weights = lora_model.get_lora(module_suffix)

            if lora_weights is None:
                # Reset if no LoRA for this module
                lora_layer.reset_lora(0)
                continue

            scale = self._adapter_scales.get(adapter_id, 1.0)
            scaled_lora_b = lora_weights.lora_b * scale
            lora_layer.set_lora(index=0, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)
            logger.debug(
                "Activated LoRA for %s: lora_a shape=%s, lora_b shape=%s, scale=%.2f",
                full_module_name, lora_weights.lora_a.shape, lora_weights.lora_b.shape, scale
            )

        self._active_adapter_id = adapter_id

    def _deactivate_all_adapters(self) -> None:
        logger.info("Deactivating all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._active_adapter_id = None
        logger.debug("All adapters deactivated")

    def _evict_if_needed(self) -> None:
        while len(self._registered_adapters) > self.max_cached_adapters:
            # Pick LRU among non-pinned adapters
            evict_candidates = [aid for aid in self._adapter_access_order.keys() if aid not in self._pinned_adapters]
            if not evict_candidates:
                logger.warning(
                    "Cache full (%d) but all adapters are pinned; cannot evict. "
                    "Increase max_cached_adapters or unpin adapters.",
                    self.max_cached_adapters,
                )
                break

            lru_adapter_id = evict_candidates[0]
            logger.info(
                "Evicting LRU adapter: id=%d (cache: %d/%d)",
                lru_adapter_id,
                len(self._registered_adapters),
                self.max_cached_adapters,
            )
            self.remove_adapter(lru_adapter_id)

    def add_lora(self, lora_request: LoRARequest, lora_scale: float = 1.0) -> bool:
        """
        Add a new adapter to the cache without activating it.
        """
        adapter_id = lora_request.lora_int_id

        if adapter_id in self._registered_adapters:
            logger.debug("Adapter %d already registered, skipping", adapter_id)
            return False

        logger.info("Adding new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)
        lora_model, peft_helper = self._load_adapter(lora_request)
        self._registered_adapters[adapter_id] = lora_model
        self._adapter_scales[adapter_id] = lora_scale

        self._replace_layers_with_lora(peft_helper)

        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)

        # evict if cache full
        self._evict_if_needed()

        logger.debug("Adapter %d added, cache size: %d/%d",
                     adapter_id, len(self._registered_adapters), self.max_cached_adapters)
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        """
        Remove an adapter from the cache.
        """
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot remove", adapter_id)
            return False

        logger.info("Removing adapter: id=%d", adapter_id)
        if self._active_adapter_id == adapter_id:
            self._deactivate_all_adapters()

        del self._registered_adapters[adapter_id]
        self._adapter_scales.pop(adapter_id, None)
        self._adapter_access_order.pop(adapter_id, None)
        self._pinned_adapters.discard(adapter_id)
        logger.debug("Adapter %d removed, cache size: %d/%d",
                     adapter_id, len(self._registered_adapters), self.max_cached_adapters)
        return True

    def list_adapters(self) -> list[int]:
        """Return list of registered adapter ids."""
        return list(self._registered_adapters.keys())

    def pin_adapter(self, adapter_id: int) -> bool:
        """Mark an adapter as pinned so it will not be evicted."""
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot pin", adapter_id)
            return False
        self._pinned_adapters.add(adapter_id)
        # Touch access order so it is most recently used
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)
        logger.info("Pinned adapter id=%d (won't be evicted)", adapter_id)
        return True
