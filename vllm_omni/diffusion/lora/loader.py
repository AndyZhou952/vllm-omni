# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA loading utilities for diffusion models."""

import os
from typing import Any

import torch
from safetensors.torch import load_file
from vllm.logger import init_logger

from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_qwen_lora_to_diffusers,
    _convert_non_diffusers_wan_lora_to_diffusers,
    _convert_non_diffusers_z_image_lora_to_diffusers,
)

logger = init_logger(__name__)


def _detect_lora_format(state_dict: dict[str, Any]) -> str:
    """Detect the format of LoRA weights."""
    keys = list(state_dict.keys())

    # lightx2v distilled format markers
    has_alpha = any(k.endswith(".alpha") for k in keys)
    has_lora_unet = any(k.startswith("lora_unet_") for k in keys)
    has_diffusion_model = any(k.startswith("diffusion_model.") for k in keys)
    has_lora_down = any(".lora_down.weight" in k for k in keys)

    if has_alpha or has_lora_unet or has_diffusion_model or has_lora_down:
        return "non_diffusers"

    # PEFT/diffusers format (already has lora_A/lora_B)
    if any(".lora_A.weight" in k or ".lora_B.weight" in k for k in keys):
        return "peft"

    return "unknown"


def _get_lora_converter(model_type: str):
    """Get the appropriate converter function for the model type."""
    converters = {
        "qwen_image": _convert_non_diffusers_qwen_lora_to_diffusers,
        "wan": _convert_non_diffusers_wan_lora_to_diffusers,
        "z_image": _convert_non_diffusers_z_image_lora_to_diffusers,
    }
    return converters.get(model_type)


def _load_state_dict(lora_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA state dict from file or directory."""
    if lora_path.endswith(".safetensors"):
        return load_file(lora_path)

    if os.path.isdir(lora_path):
        # Find safetensors file in directory
        for filename in os.listdir(lora_path):
            if filename.endswith(".safetensors"):
                return load_file(os.path.join(lora_path, filename))
        raise FileNotFoundError(f"No .safetensors file found in {lora_path}")

    # Try HuggingFace download
    from huggingface_hub import hf_hub_download

    try:
        # Try adapter_model.safetensors first (PEFT format)
        path = hf_hub_download(lora_path, "adapter_model.safetensors")
        return load_file(path)
    except Exception:
        # Try to find any safetensors file in the repo
        from huggingface_hub import list_repo_files

        files = list_repo_files(lora_path)
        safetensors_files = [f for f in files if f.endswith(".safetensors")]
        if safetensors_files:
            path = hf_hub_download(lora_path, safetensors_files[0])
            return load_file(path)
        raise FileNotFoundError(f"No .safetensors file found in {lora_path}")


def _get_base_layer_key(lora_key: str) -> str:
    """Extract the base layer key from a LoRA key.

    Example: 'transformer.blocks.0.attn.to_q.lora_A.weight'
             -> 'transformer.blocks.0.attn.to_q.weight'
    """
    return lora_key.replace(".lora_A.weight", ".weight").replace(
        ".lora_B.weight", ".weight"
    )


def _find_matching_param(
    model: torch.nn.Module, target_key: str
) -> tuple[str, torch.nn.Parameter] | None:
    """Find a model parameter that matches the target key.

    Handles prefix variations like 'transformer.' being present or not.
    Uses dict-based lookup for O(1) direct/prefix matching before fallback.
    """
    # Build parameter dict once for efficient lookups
    param_dict = dict(model.named_parameters())

    # 1. Direct match (O(1) lookup)
    if target_key in param_dict:
        return target_key, param_dict[target_key]

    # 2. Try without 'transformer.' prefix (O(1) lookup)
    stripped = target_key.removeprefix("transformer.")
    if stripped in param_dict:
        return stripped, param_dict[stripped]

    # 3. Suffix match fallback (O(n) - only when needed)
    for name, param in param_dict.items():
        if name.endswith(stripped) or stripped.endswith(name):
            return name, param

    return None


def _apply_lora_to_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> int:
    """Apply LoRA weights directly to model parameters.

    For each LoRA pair (lora_A, lora_B), computes:
        delta_W = lora_B @ lora_A
    and adds it to the base weight.

    Returns:
        Number of LoRA layers successfully applied.
    """
    applied_count = 0
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}

    # Group lora_A and lora_B weights by base layer
    for key, value in state_dict.items():
        if ".lora_A.weight" in key:
            base_key = _get_base_layer_key(key)
            lora_pairs.setdefault(base_key, {})["lora_A"] = value
        elif ".lora_B.weight" in key:
            base_key = _get_base_layer_key(key)
            lora_pairs.setdefault(base_key, {})["lora_B"] = value

    # Apply each LoRA pair to the corresponding base weight
    for base_key, lora_weights in lora_pairs.items():
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            logger.warning("Incomplete LoRA pair for %s, skipping", base_key)
            continue

        lora_a = lora_weights["lora_A"]
        lora_b = lora_weights["lora_B"]

        # Find the matching parameter in the model
        match = _find_matching_param(model, base_key)
        if match is None:
            logger.warning("No matching parameter found for %s", base_key)
            continue

        param_name, param = match

        # Compute delta: delta_W = lora_B @ lora_A
        # lora_A shape: (rank, in_features)
        # lora_B shape: (out_features, rank)
        # delta shape: (out_features, in_features)
        with torch.no_grad():
            delta = lora_b.to(param.device, param.dtype) @ lora_a.to(
                param.device, param.dtype
            )

            # Handle shape mismatches (e.g., packed QKV weights)
            if delta.shape != param.shape:
                logger.debug(
                    "Shape mismatch for %s: delta %s vs param %s",
                    param_name,
                    delta.shape,
                    param.shape,
                )
                # For packed weights like qkv, the LoRA might be for a subset
                if len(delta.shape) == 2 and len(param.shape) == 2:
                    if delta.shape[1] == param.shape[1]:
                        # Same input dim, different output - likely packed
                        # Apply to the first portion that matches
                        out_dim = delta.shape[0]
                        param.data[:out_dim, :] += delta
                        applied_count += 1
                        logger.debug("Applied LoRA to partial weight %s", param_name)
                        continue
                logger.warning(
                    "Cannot apply LoRA to %s due to shape mismatch", param_name
                )
                continue

            param.data += delta
            applied_count += 1
            logger.debug("Applied LoRA to %s", param_name)

    return applied_count


def _apply_lora_with_peft(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    adapter_name: str = "default",
) -> torch.nn.Module:
    """Apply LoRA using PEFT's adapter system (for non-fused LoRA)."""
    from peft import LoraConfig, inject_adapter_in_model

    # Infer rank and target modules from state dict
    rank = None
    target_modules = set()

    for key, value in state_dict.items():
        if ".lora_A.weight" in key:
            rank = value.shape[0]
            # Extract target module name (last component before .lora_A)
            base_key = key.replace(".lora_A.weight", "")
            parts = base_key.split(".")
            target_modules.add(parts[-1])

    if rank is None:
        raise ValueError("Could not infer LoRA rank from state dict")

    logger.info("LoRA rank: %d, target modules: %s", rank, list(target_modules))

    # Create LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,  # Alpha already applied during conversion
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )

    # Inject adapter into model
    model = inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)

    # Build a mapping from original keys to PEFT keys
    # PEFT adds 'base_model.model.' prefix and changes structure
    peft_state_dict = {}
    model_param_names = {name for name, _ in model.named_parameters()}

    for key, value in state_dict.items():
        # Try to find matching PEFT parameter name
        matched = False
        for param_name in model_param_names:
            # Check if the param_name contains the essential parts of our key
            key_parts = key.replace("transformer.", "").split(".")
            if all(part in param_name for part in key_parts[-4:]):
                peft_state_dict[param_name] = value
                matched = True
                break

        if not matched:
            # Try direct mapping with base_model.model prefix
            peft_key = f"base_model.model.{key}"
            if peft_key in model_param_names:
                peft_state_dict[peft_key] = value
            else:
                logger.debug("Could not map LoRA key: %s", key)

    # Load weights
    missing, unexpected = model.load_state_dict(peft_state_dict, strict=False)
    if missing:
        logger.debug("Missing keys during LoRA load: %s", missing[:5])
    if unexpected:
        logger.debug("Unexpected keys during LoRA load: %s", unexpected[:5])

    return model


def load_lora_weights(
    transformer: torch.nn.Module,
    lora_path: str,
    model_type: str,
    adapter_name: str = "default",
    fuse: bool = False,
) -> torch.nn.Module:
    """
    Load LoRA weights into a transformer model.

    Supports both PEFT format and distilled formats (e.g., lightx2v).

    Args:
        transformer: The transformer model to inject LoRA into
        lora_path: Path to LoRA weights (file, directory, or HuggingFace ID)
        model_type: Model type for format conversion ("qwen_image", "wan", "z_image")
        adapter_name: Name for the adapter (used when fuse=False)
        fuse: If True, directly apply LoRA delta to base weights (recommended
              for distilled LoRAs like lightx2v). If False, use PEFT adapters.

    Returns:
        The transformer with LoRA applied
    """
    logger.info("Loading LoRA weights from %s", lora_path)

    state_dict = _load_state_dict(lora_path)
    lora_format = _detect_lora_format(state_dict)
    logger.info("Detected LoRA format: %s", lora_format)

    # Convert non-diffusers format to diffusers/PEFT format
    if lora_format == "non_diffusers":
        converter = _get_lora_converter(model_type)
        if converter is None:
            raise ValueError(
                f"No LoRA converter available for model type: {model_type}"
            )
        logger.info("Converting to diffusers format using %s converter", model_type)
        state_dict = converter(state_dict)
    elif lora_format == "unknown":
        raise ValueError(f"Unknown LoRA format in {lora_path}")

    # Log debug info about the converted state dict
    lora_keys = [k for k in state_dict.keys() if "lora_A" in k]
    logger.debug(
        "Converted state dict has %d LoRA layers, example keys: %s",
        len(lora_keys),
        lora_keys[:3] if lora_keys else "none",
    )

    if fuse:
        # Directly apply LoRA delta to base weights (recommended for distilled)
        logger.info("Fusing LoRA weights directly into base model")
        applied = _apply_lora_to_model(transformer, state_dict)
        logger.info("Successfully applied %d LoRA layers", applied)
    else:
        # Use PEFT adapter system
        logger.info("Applying LoRA via PEFT adapter system")
        transformer = _apply_lora_with_peft(transformer, state_dict, adapter_name)
        logger.info("LoRA adapter '%s' injected successfully", adapter_name)

    return transformer
