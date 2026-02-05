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
    """
    param_dict = dict(model.named_parameters())

    if target_key in param_dict:
        return target_key, param_dict[target_key]

    stripped = target_key.removeprefix("transformer.")
    if stripped in param_dict:
        return stripped, param_dict[stripped]

    for name, param in param_dict.items():
        if name.endswith(stripped) or stripped.endswith(name):
            return name, param

    return None


# Mapping from packed layer suffix to its constituent sublayers
PACKED_MODULES_MAPPING = {
    "to_qkv": ["to_q", "to_k", "to_v"],
    "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
}

# Reverse mapping: sublayer -> (packed_layer, index)
SUBLAYER_TO_PACKED = {}
for packed, sublayers in PACKED_MODULES_MAPPING.items():
    for idx, sub in enumerate(sublayers):
        SUBLAYER_TO_PACKED[sub] = (packed, idx, len(sublayers))


def _get_packed_layer_key(base_key: str) -> tuple[str, int, int] | None:
    """Check if a base key corresponds to a sublayer of a packed layer.

    Returns (packed_key, slice_index, num_slices) or None.
    """
    # Extract the layer suffix (e.g., "to_q" from "transformer_blocks.0.attn.to_q.weight")
    parts = base_key.replace(".weight", "").split(".")
    if not parts:
        return None

    suffix = parts[-1]
    if suffix in SUBLAYER_TO_PACKED:
        packed_suffix, idx, num_slices = SUBLAYER_TO_PACKED[suffix]
        # Rebuild the key with packed suffix
        packed_key = ".".join(parts[:-1] + [packed_suffix]) + ".weight"
        return packed_key, idx, num_slices

    return None


def _apply_lora_to_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> int:
    """Apply LoRA weights directly to model parameters.

    For each LoRA pair (lora_A, lora_B), computes:
        delta_W = lora_B @ lora_A
    and adds it to the base weight.

    Handles packed layers (e.g., to_qkv = [to_q, to_k, to_v]) by:
    1. Grouping sublayer LoRAs together
    2. Computing individual deltas
    3. Concatenating and applying to the packed weight

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

    # Separate direct matches from packed layer sublayers
    direct_pairs: dict[str, dict[str, torch.Tensor]] = {}
    packed_sublayers: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    param_dict = dict(model.named_parameters())

    for base_key, lora_weights in lora_pairs.items():
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            logger.warning("Incomplete LoRA pair for %s, skipping", base_key)
            continue

        # Check if this is a direct match
        match = _find_matching_param(model, base_key)
        if match is not None:
            direct_pairs[base_key] = lora_weights
            continue

        # Check if this is a sublayer of a packed layer
        packed_info = _get_packed_layer_key(base_key)
        if packed_info is not None:
            packed_key, slice_idx, num_slices = packed_info
            # Verify the packed layer exists
            packed_match = _find_matching_param(model, packed_key)
            if packed_match is not None:
                actual_packed_key = packed_match[0]
                packed_sublayers.setdefault(actual_packed_key, {})[slice_idx] = lora_weights
                logger.debug(
                    "Grouped %s as slice %d of packed layer %s",
                    base_key, slice_idx, actual_packed_key
                )
                continue

        logger.warning("No matching parameter found for %s", base_key)

    # Apply direct LoRA pairs
    for base_key, lora_weights in direct_pairs.items():
        lora_a = lora_weights["lora_A"]
        lora_b = lora_weights["lora_B"]

        match = _find_matching_param(model, base_key)
        if match is None:
            continue

        param_name, param = match

        with torch.no_grad():
            delta = lora_b.to(param.device, param.dtype) @ lora_a.to(
                param.device, param.dtype
            )

            if delta.shape != param.shape:
                logger.debug(
                    "Shape mismatch for %s: delta %s vs param %s",
                    param_name, delta.shape, param.shape,
                )
                if len(delta.shape) == 2 and len(param.shape) == 2:
                    if delta.shape[1] == param.shape[1]:
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

    # Apply packed layer LoRAs
    for packed_key, sublayer_dict in packed_sublayers.items():
        param = param_dict[packed_key]

        # Determine expected number of slices from the packed module mapping
        num_slices = len(sublayer_dict)  # default
        for packed_name, sublayers in PACKED_MODULES_MAPPING.items():
            if packed_name in packed_key:
                num_slices = len(sublayers)
                break

        # Compute deltas for available slices
        deltas: list[torch.Tensor | None] = [None] * num_slices
        slice_out_dims: list[int] = []

        for slice_idx, lora_weights in sublayer_dict.items():
            lora_a = lora_weights["lora_A"]
            lora_b = lora_weights["lora_B"]
            delta = lora_b.to(param.device, param.dtype) @ lora_a.to(
                param.device, param.dtype
            )
            deltas[slice_idx] = delta
            slice_out_dims.append(delta.shape[0])

        if not slice_out_dims:
            logger.warning("No valid slices for packed layer %s", packed_key)
            continue

        # For packed layers, each slice should have the same output dim
        # Total output = num_slices * slice_out_dim
        slice_out_dim = slice_out_dims[0]
        expected_total = num_slices * slice_out_dim

        with torch.no_grad():
            if param.shape[0] == expected_total:
                # Apply each slice to its portion of the packed weight
                for slice_idx, delta in enumerate(deltas):
                    if delta is not None:
                        start = slice_idx * slice_out_dim
                        end = start + slice_out_dim
                        param.data[start:end, :] += delta
                        applied_count += 1
                        logger.debug(
                            "Applied LoRA slice %d to packed layer %s [%d:%d]",
                            slice_idx, packed_key, start, end
                        )
            else:
                # Fallback: concatenate available deltas and apply
                valid_deltas = [d for d in deltas if d is not None]
                if valid_deltas:
                    combined = torch.cat(valid_deltas, dim=0)
                    if combined.shape[0] <= param.shape[0] and combined.shape[1] == param.shape[1]:
                        param.data[:combined.shape[0], :] += combined
                        applied_count += len(valid_deltas)
                        logger.debug(
                            "Applied %d LoRA slices to packed layer %s (partial)",
                            len(valid_deltas), packed_key
                        )
                    else:
                        logger.warning(
                            "Shape mismatch for packed layer %s: combined %s vs param %s",
                            packed_key, combined.shape, param.shape
                        )

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

    lora_keys = [k for k in state_dict.keys() if "lora_A" in k]
    logger.info(
        "Converted state dict has %d LoRA layers, example keys: %s",
        len(lora_keys),
        lora_keys[:3] if lora_keys else "none",
    )

    # Debug: show model parameter names for comparison
    model_params = list(dict(transformer.named_parameters()).keys())
    logger.info(
        "Model has %d parameters, example keys: %s",
        len(model_params),
        model_params[:3] if model_params else "none",
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
