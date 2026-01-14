# LoRA Inference Examples

This directory contains examples for using LoRA (Low-Rank Adaptation) adapters with vLLM-omni diffusion models for offline inference.
The example uses the  `stabilityai/stable-diffusion-3.5-medium` as the default model, but you can replace it with other models in vLLM-omni.

## Overview

Similar to vLLM, vLLM-omni uses a unified LoRA handling mechanism:

- **Pre-loaded LoRA**: Loaded at initialization via `--lora-path` (pre-loaded into cache)
- **Per-request LoRA**: Loaded on-demand. In the example, the LoRA is loaded via `--lora-request-path` in each request

Both approaches use the same underlying mechanism - all LoRA adapters are handled uniformly through `set_active_adapter()`. If no LoRA request is provided in a request, all adapters are deactivated.

## Prerequisites

- A compatible diffusion model (default: `stabilityai/stable-diffusion-3.5-medium`)
- LoRA adapter files in PEFT format
- vLLM-omni installed

## Usage

### Pre-loaded LoRA (via --lora-path)

Load a LoRA adapter at initialization. This adapter is pre-loaded into the cache and can be activated by requests:

```bash
python -m examples.offline_inference.lora_inference.lora_inference \
    --prompt "A piece of cheesecake" \
    --lora-path /path/to/your/lora/adapter \
    --lora-scale 1.0 \
    --num_inference_steps 50 \
    --height 1024 \
    --width 1024 \
    --output output_preloaded.png
```

**Note**: When using `--lora-path`, the adapter is loaded at init time with ID 1. To use it in the request, you still need to provide `--lora-request-id 1` (or it will be deactivated).

### Per-request LoRA (via --lora-request-path)

Load a LoRA adapter on-demand for each request:

```bash
python -m examples.offline_inference.lora_inference.lora_inference \
    --prompt "A piece of cheesecake" \
    --lora-request-path /path/to/your/lora/adapter \
    --lora-request-id 1 \
    --lora-scale 1.0 \
    --num_inference_steps 50 \
    --height 1024 \
    --width 1024 \
    --output output_per_request.png
```

### No LoRA

If no LoRA request is provided, we will use the base model without any LoRA adapters:

```bash
python -m examples.offline_inference.lora_inference.lora_inference \
    --prompt "A piece of cheesecake" \
    --num_inference_steps 50 \
    --height 1024 \
    --width 1024 \
    --output output_no_lora.png
```

## Parameters

### LoRA Parameters

- `--lora-path`: Path to LoRA adapter to pre-load at initialization (loads into cache with ID 1)
- `--lora-request-path`: Path to LoRA adapter for per-request loading
- `--lora-request-id`: Integer ID for the LoRA adapter (required for per-request LoRA). If not provided and `--lora-request-path` is set, will use hash of path.
- `--lora-scale`: Scale factor for LoRA weights (default: 1.0). Higher values increase the influence of the LoRA adapter.

### Standard Parameters

- `--prompt`: Text prompt for image generation (required)
- `--seed`: Random seed for reproducibility (default: 42)
- `--height`: Image height in pixels (default: 1024)
- `--width`: Image width in pixels (default: 1024)
- `--num_inference_steps`: Number of denoising steps (default: 50)
- `--output`: Output file path (default: `lora_output.png`)

## How LoRA Works

All LoRA adapters are handled uniformly:

1. **Initialization**: If `--lora-path` is provided, the adapter is loaded into cache with ID 1
2. **Per-request**: If `--lora-request-path` is provided, the adapter is loaded/activated for that request
3. **No LoRA**: If no LoRA request is provided (`req.lora_request` is None), all adapters are deactivated

The system uses LRU cache management - adapters are cached and evicted when the cache is full (unless pinned).

## LoRA Adapter Format

LoRA adapters must be in PEFT (Parameter-Efficient Fine-Tuning) format. A typical LoRA adapter directory structure:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```
