# Online LoRA Inference (Diffusion)

This example shows how to use **per-request LoRA** with vLLM-Omni diffusion models via the OpenAI-compatible Chat Completions API.

> Note: The LoRA adapter path must be readable on the **server** machine (usually a local path or a mounted directory).

## Start Server

```bash
# Pick a diffusion model (examples)
# export MODEL=stabilityai/stable-diffusion-3.5-medium
# export MODEL=Qwen/Qwen-Image

bash run_server.sh
```

## Call API (curl)

```bash
# Required: local LoRA folder on the server
export LORA_PATH=/path/to/lora_adapter

# Optional
export SERVER=http://localhost:8091
export PROMPT="A piece of cheesecake"
export LORA_NAME=my_lora
export LORA_SCALE=1.0
export LORA_INT_ID=1

bash run_curl_lora_inference.sh
```

## Call API (Python)

```bash
python openai_chat_client.py \
  --prompt "A piece of cheesecake" \
  --lora-path /path/to/lora_adapter \
  --lora-name my_lora \
  --lora-scale 1.0 \
  --output output.png
```

## LoRA Format

LoRA adapters should be in PEFT format, for example:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

