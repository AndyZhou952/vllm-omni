#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Offline LoRA scale sweep for diffusion models.

Generates 5 deterministic images for the same prompt:
  1) no_lora
  2) lora_scale_0
  3) lora_scale_1
  4) lora_scale_5
  5) lora_scale_50
"""

import argparse
import hashlib
from pathlib import Path

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def _extract_first_image(output) -> object:
    images = None
    if hasattr(output, "images") and output.images:
        images = output.images
    elif hasattr(output, "request_output") and output.request_output:
        req_out = output.request_output
        if isinstance(req_out, list) and len(req_out) > 0:
            req_out = req_out[0]
        if hasattr(req_out, "images") and req_out.images:
            images = req_out.images
    if not images:
        raise ValueError("No images found in offline output")
    return images[0]


def _mse(img_a, img_b) -> float:
    if img_a.size != img_b.size:
        raise ValueError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    a = img_a.convert("RGB").tobytes()
    b = img_b.convert("RGB").tobytes()
    if len(a) != len(b):
        raise ValueError(f"Image byte-length mismatch: {len(a)} vs {len(b)}")
    sq_sum = 0
    for va, vb in zip(a, b):
        d = va - vb
        sq_sum += d * d
    return sq_sum / len(a)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_single(
    omni: Omni,
    prompt: str,
    *,
    seed: int,
    width: int,
    height: int,
    steps: int,
    lora_request: LoRARequest | None,
    lora_scale: float,
):
    sampling_params = OmniDiffusionSamplingParams(
        width=width,
        height=height,
        seed=seed,
        num_inference_steps=steps,
    )
    if lora_request is not None:
        sampling_params.lora_request = lora_request
        sampling_params.lora_scale = lora_scale

    outputs = omni.generate(prompt, sampling_params)
    if not outputs:
        raise ValueError("No output generated from omni.generate()")
    first_output = outputs[0] if isinstance(outputs, list) else outputs
    return _extract_first_image(first_output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline deterministic LoRA scale sweep")
    parser.add_argument("--model", default="/home/andy/model/stable-diffusion-3.5-medium", help="Model path")
    parser.add_argument(
        "--lora-path",
        default="/home/andy/model/SD3.5M-FlowGRPO-GenEval",
        help="LoRA adapter folder (PEFT format)",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=15, help="num_inference_steps")
    parser.add_argument("--out-dir", default="offline_lora_sweep_output", help="Output directory")
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.2,
        help="Threshold for near-identical image comparison",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_id = stable_lora_int_id(args.lora_path)
    lora_name = Path(args.lora_path).name or "lora_adapter"
    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=lora_id,
        lora_path=args.lora_path,
    )

    print(f"Model       : {args.model}")
    print(f"LoRA path   : {args.lora_path}")
    print(f"LoRA name/id: {lora_name}/{lora_id}")
    print(f"Prompt      : {args.prompt}")
    print(f"Output dir  : {out_dir}")

    # Do not preload LoRA at init; use per-request activation for each run.
    omni = Omni(model=args.model)

    runs: list[tuple[str, LoRARequest | None, float]] = [
        ("no_lora", None, 1.0),
        ("lora_scale_0", lora_request, 0.0),
        ("lora_scale_1", lora_request, 1.0),
        ("lora_scale_5", lora_request, 5.0),
        ("lora_scale_50", lora_request, 50.0),
    ]

    generated = {}
    for run_name, req, scale in runs:
        img = _generate_single(
            omni,
            args.prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            lora_request=req,
            lora_scale=scale,
        )
        out_path = out_dir / f"{run_name}.png"
        img.save(out_path)
        generated[run_name] = (img, out_path)
        print(f"Saved {run_name}: {out_path} sha256={_sha256(out_path)}")

    print("")
    baseline = generated["no_lora"][0]
    l0 = generated["lora_scale_0"][0]
    l1 = generated["lora_scale_1"][0]
    l5 = generated["lora_scale_5"][0]
    l50 = generated["lora_scale_50"][0]
    print("MSE diagnostics (against no_lora):")
    print(f"  no_lora vs lora_scale_0  : {_mse(baseline, l0):.6f}")
    print(f"  no_lora vs lora_scale_1  : {_mse(baseline, l1):.6f}")
    print(f"  no_lora vs lora_scale_5  : {_mse(baseline, l5):.6f}")
    print(f"  no_lora vs lora_scale_50 : {_mse(baseline, l50):.6f}")
    print("MSE diagnostics (between LoRA scales):")
    print(f"  lora_scale_0 vs lora_scale_1  : {_mse(l0, l1):.6f}")
    print(f"  lora_scale_1 vs lora_scale_5  : {_mse(l1, l5):.6f}")
    print(f"  lora_scale_5 vs lora_scale_50 : {_mse(l5, l50):.6f}")

    if _mse(l0, l1) <= args.mse_threshold and _mse(l1, l5) <= args.mse_threshold:
        print("WARN: LoRA scales appear to have little effect (0/1/5 very similar).")
    else:
        print("PASS: LoRA scale sweep shows measurable differences.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
