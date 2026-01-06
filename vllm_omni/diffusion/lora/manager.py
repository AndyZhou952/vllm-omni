import os
import threading
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from vllm.lora.request import LoRARequest
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from peft import PeftModel, get_peft_model
    from peft.config import PeftConfig
    from peft.utils import WEIGHTS_NAME

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library is not available. LoRA support will not work.")

class DiffusionLoRAManager:
    # TODO - andy
    # ref: https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/lora/worker_manager.py
    # add/remove/list/apply adapters
    # update_lora_weights
    # GPU workflow: lora manager update globally -> caching -> pipeline infer
    # no kv cache
    # threading?
    def __init__(
            self,
            max_loras: int = 1,
            max_lora_rank: int = 16,
            pipeline: nn.Module | None = None,
    ):
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT library is required for LoRA support.")

        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.pipeline = pipeline

        self._lora_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        self._lock = threading.Lock()

        self._active_adpater_id: int | None = None

        self._original_transformer: nn.Module | None = None # store original transformer reference before wrapping
        self._peft_model: PeftModel | None = None

        if pipeline is not None:
            self._initialize_peft_wrapper()

    def _initialize_peft_wrapper(self) -> None:
        if self.pipeline is None:
            return

        # find transformer component in pipeline
        if not hasattr(self.pipeline, "transformer"):
            logger.warning("Pipeline does not have a 'transformer' attribute. LoRA may not work correctly.")
            return

        self._original_transformer = self.pipeline.transformer
        # dynamic adapter loading - on demand
        logger.info("LoRA manager initialized for pipeline transformer")

    def _validate_lora_path(self):
        raise NotImplementedError

    def _load_adapter_from_disk(
            self,
            lora_request: LoRARequest
    ) -> PeftModel:
        self._validate_lora_path(lora_request.lora_local_path)

        try:
            peft_config = PeftConfig.from_pretrained(lora_request.lora_local_path)
        except Exception as e:
            raise ValueError(f"Failed to laod LoRA config from {lora_request.lora_local_path}: {e}")

        # rank validation
        if hasattr(peft_config, "r") and peft_config.r > self.max_lora_rank:
            raise ValueError(f"LoRA rank {peft_config.r} exceeds maximum allowed rank {self.max_lora_rank}")

        # load the adpater
        if self._peft_model is None and self._original_transformer is not None:
            try:
                self._peft_model = PeftModel.from_pretrained(
                    self._original_transformer,
                    lora_request.lora_local_path,
                    adapter_name = lora_request.lora_name,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA adapter from {lora_request.lora_local_path}: {e}")
        elif self._peft_model is not None:
            # additional adapter: add to existing PeftModel
            try:
                self._peft_model.load_adapter(lora_request.lora_local_path, adapter_name=lora_request.lora_name)
            except Exception as e:
                raise RuntimeError(f"Failed to add LoRA adapter {lora_request.lora_name}: {e}")
        else:
            raise RuntimeError("Cannot load adapter: transformer not initialized")

        return self._peft_model

    def add_apaters(
            self,
            lora_request: LoRARequest
    ) -> dict[str, Any]:
        with self._lock:
            lora_id = lora_request.lora_int_id
            if lora_id in self._lora_cache: # check if already loaded
                # reactive: move to end (most recently used)
                adapter_info = self._lora_cache.pop(lora_id)
                self._lora_cache[lora_id] = adapter_info
                logger.info(f"Reactived LoRA adapter {lora_request.lora_name} to {lora_id}")

                # set as active adapter:
                if self._peft_model is not None:
                    try:
                        self._peft_model.set_adapter(lora_request.lora_name)
                        self._active_adpater_id = lora_id
                    except Exception as e:
                        logger.error(f"Failed to set LoRA adapter {lora_request.lora_name}: {e}")
                        return {"status": "error", "error": str(e)}
                return {"status": "success", "lora_id": lora_id, "action": "reactivated"}

            if len(self._lora_cache) >= self.max_loras: # check if needed to evict an adapter
                evicted_id, evicted_info = self._lora_cache.popitem(last=False)
                logger.info(f"Evicted LoRA adapter {evicted_info['name']} from {evicted_id} to make room")

                # unload from model if it's the active one
                if evicted_id == self._active_adpater_id:
                    adapter_name = evicted_info["name"]
                    if self._peft_model is not None:
                        try:
                            self._peft_model.disable_adapters()
                            # adapter still in PEFT but disabled
                        except Exception as e:
                            logger.wearning(f"Failed to disable LoRA adapter {adapter_name}: {e}")
                    self._active_adpater_id = None

            # load new adapter
            try:
                peft_model = self._load_adapter_from_disk(lora_request)
                adapter_info = {
                    "name": lora_request.lora_name,
                    "path": lora_request.lora_local_path,
                    "lora_id": lora_id,
                    "peft_model": peft_model,
                }
                self._lora_cache[lora_id] = adapter_info

                if self._peft_model is not None:
                    self._peft_model.set_adapter(lora_request.lora_name)
                    self._active_adpater_id = lora_id

                logger.info(f"Loaded LoRA adapter {lora_request.lora_name} to {lora_id}")
                return {"status": "success", "lora_id": lora_id, "action": "loaded"}
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter {lora_request.lora_name}: {e}")
                return {"status": "error", "error": str(e)}

    def remove_adapters(
            self,
            lora_id: int
    ) -> dict[str, Any]:
        with self._lock:
            if lora_id not in self._lora_cache:
                return {"status": "error", "error": f"LoRA adapter {lora_id} not found"}

            adapter_info = self._lora_cache.pop(lora_id)

            # unload if active
            if lora_id == self._active_adpater_id:
                self._unload_adapter_from_model(lora_id)
                self._active_adpater_id = None

            logger.info(f"Removed LoRA adapter {adapter_info['name']} from {lora_id}")
            return {"status": "success", "lora_id": lora_id}

    def _unload_adapter_from_model(self, lora_id: int) -> None:


        adapter_info = self._lora_cache.get(lora_id)
        if adapter_info is None:
            return

        adapter_name = adapter_info["name"]
        try:
            self._peft_model.disable_adapters()
            # adapter still in PEFT but disabled
        except Exception as e:
            logger.warning(f"Failed to disable LoRA adapter {adapter_name}: {e}")

    def list_adapters(self) -> list[dict[str, Any]]:
        with self._lock:
            adapters = []
            for lora_id, adapter_info in self._lora_cache.items():
                adapters.append(
                    {
                        "lora_id": lora_id,
                        "name": adapter_info["name"],
                        "path": adapter_info["path"],
                        "active": lora_id == self._active_adpater_id,
                    }
                )
            return adapters

    def _apply_adapters(
            self,
            lora_requests: list[LoRARequest]
    ) -> dict[str, Any]:
        # status: success
        # apply requested LoRAs, ensuring they don't exceed_max_loras
        if not lora_requests:
            if self._peft_model is not None and self._active_adpater_id is not None:
                self._peft_model.disable_adapters()
                self._active_adpater_id = None
            return {"status": "success"}

        for lora_request in lora_requests:
            result = self.add_apaters(lora_request)
            if result.get('status') != "success":
                return result # not sccessful

        # set adapters
        raise NotImplementedError # set multiple adapters?

        return {"status": "success"}

    def update_lora_weights(
            self,
            lora_id: int,
            weights: dict[str, torch.Tensor]
    ) -> dict[str, Any]: # RL use case
        with self._lock:
            if lora_id not in self._lora_cache:
                return {"status": "error", "error": f"LoRA adapter {lora_id} not found"}

            adapter_info = self._lora_cache[lora_id]
            adapter_name = adapter_info["name"]

            if self._peft_model is None:
                return {"status": "error", "error": "PEFT model not initialized"}

            try:
                for param_name, weight_tensor in weights.items():
                    # ref: https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py#L57
                    if hasattr(self._peft_model, "get_peft_model_state_dict"):
                        state_dict = self._peft_model.get_peft_model_state_dict(adapter_name=adapter_name)
                        if param_name in state_dict:
                            state_dict[param_name].copy_(weight_tensor)
                        else:
                            logger.warning(f"Parameter {param_name} not found in LoRA adapter {adapter_name}")
                    else:
                        logger.warning("PEFT model does not have get_peft_model_state_dict method")
                        return {"status": "error", "error": "PEFT model does not have get_peft_model_state_dict method"}

                logger.info(f"Updated weights for LoRA adapter {adapter_name} id {lora_id}")
                return {"status": "success", "lora_id": lora_id}
            except Exception as e:
                logger.error(f"Failed to update weights for LoRA adapter {adapter_name}: {e}")
                return {"status": "error", "error": str(e)}