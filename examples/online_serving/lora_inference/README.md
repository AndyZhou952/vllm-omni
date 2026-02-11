# LoRA Inference(Diffusion)

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
- `baseline_post_wrap ~= lora_scale_0` is expected.
- `lora_scale_0 != lora_scale_1` indicates LoRA is actually applied.
- If `lora_scale_0 ~= lora_scale_1`, LoRA is likely loaded but ineffective.
- If `baseline_pre != baseline_post_wrap`, this can happen after first LoRA request because LoRA wrappers are inserted once and runtime state changes; use `baseline_post_wrap` as the reference baseline.

## Troubleshooting with server logs

Use these LoRA diagnostics from server logs:

- `LoRA replacement summary`:
  - `replaced=0` or very low means adapter target modules are not mapping to runtime layers.
  - High `target_filtered` means module-name mismatch against `target_modules`.
- `LoRA activation summary`:
  - `activated_layers=0` + high `reset_layers` means adapter is loaded but not effectively applied.
  - `shape_mismatch_skips>0` means LoRA tensor shapes do not fit packed layer expectations.
- Request trace:
  - `LoRA request parsed for ...`
  - `Worker ... LoRA before activation ...`
  - `Setting active adapter ...`
  Confirm that name/path/id/scale are consistent from API parse to worker activation.

## LoRA Format

LoRA adapters should be in PEFT format, for example:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```
