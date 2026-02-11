# LoRA Inference(Diffusion)

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/lora_inference>.


This example shows how to use **per-request LoRA** with vLLM-Omni diffusion models via:
- OpenAI-compatible Chat Completions API (`/v1/chat/completions`)
- OpenAI-compatible Images API (`/v1/images/generations`)

> Note: The LoRA adapter path must be readable on the **server** machine (usually a local path or a mounted directory).
> Note: Image verification scripts in this folder use `/v1/images/generations` with request body field `lora`.

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
# Optional: if omitted, the server derives a stable id from LORA_PATH.
# export LORA_INT_ID=123

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

## Verify LoRA effect (`/v1/images/generations`)

Run a deterministic scale sweep (same prompt + same seed):

```bash
export SERVER=http://localhost:8091
export LORA_PATH=/path/to/lora_adapter
export PROMPT="MIA_char, standing in a new york city"
bash run_curl_lora_verify_images.sh
```

Then compute quantitative comparisons:

```bash
python verify_lora_online.py \
  --server "$SERVER" \
  --lora-path "$LORA_PATH" \
  --prompt "$PROMPT" \
  --out-dir lora_verify_output
```

Interpretation:
- `baseline ~= lora_scale_0` is expected.
- `lora_scale_0 != lora_scale_1` indicates LoRA is actually applied.
- If `lora_scale_0 ~= lora_scale_1`, LoRA is likely loaded but ineffective (target-module mismatch, per-layer reset, shape mismatch, or near-zero effective weights).

## LoRA Format

LoRA adapters should be in PEFT format, for example:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/lora_inference/openai_chat_client.py"
    ``````
??? abstract "run_curl_lora_inference.sh"
    ``````sh
    --8<-- "examples/online_serving/lora_inference/run_curl_lora_inference.sh"
    ``````
??? abstract "run_curl_lora_verify_images.sh"
    ``````sh
    --8<-- "examples/online_serving/lora_inference/run_curl_lora_verify_images.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/lora_inference/run_server.sh"
    ``````
??? abstract "verify_lora_online.py"
    ``````py
    --8<-- "examples/online_serving/lora_inference/verify_lora_online.py"
    ``````
