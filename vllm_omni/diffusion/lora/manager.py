class DiffusionLoRAManager:
    # TODO - andy
    # ref: https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/lora/worker_manager.py
    # add/remove/list/apply adapters
    # update_lora_weights
    # GPU workflow: lora manager update globally -> caching -> pipeline infer
    # no kv cache
    # threading?
    def __init__(self):
        raise NotImplementedError

    def add_apaters(self):
        raise NotImplementedError

    def remove_adapters(self):
        raise NotImplementedError

    def list_adapters(self):
        raise NotImplementedError

    def _apply_adapters(self):
        # status: success
        raise NotImplementedError

    def update_lora_weights(self):
        raise NotImplementedError
