#!/usr/bin/env python3
"""
Verify LoRA influence for /v1/images/generations with deterministic A/B runs.

Outputs:
  - baseline.png (no LoRA)
  - lora_scale_0.png
  - lora_scale_05.png
  - lora_scale_1.png
And prints quantitative differences.
"""

import argparse
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def generate_image(
    *,
    server: str,
    prompt: str,
    size: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    lora_payload: dict | None,
) -> Image.Image:
    payload = {
        "prompt": prompt,
        "n": 1,
        "size": size,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    if lora_payload is not None:
        payload["lora"] = lora_payload

    response = requests.post(
        f"{server.rstrip('/')}/v1/images/generations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    response.raise_for_status()
    b64 = response.json()["data"][0]["b64_json"]
    image_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def mse(img_a: Image.Image, img_b: Image.Image) -> float:
    if img_a.size != img_b.size:
        raise ValueError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    a = img_a.tobytes()
    b = img_b.tobytes()
    if len(a) != len(b):
        raise ValueError(f"Image byte-length mismatch: {len(a)} vs {len(b)}")
    sq_sum = 0
    for va, vb in zip(a, b):
        d = va - vb
        sq_sum += d * d
    return sq_sum / len(a)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic LoRA verification for /v1/images/generations")
    parser.add_argument("--server", default="http://localhost:1997", help="OpenAI-compatible server URL")
    parser.add_argument("--prompt", default="MIA_char, standing in a new york city", help="Text prompt")
    parser.add_argument("--size", default="1024x1024", help="Image size, e.g. 1024x1024")
    parser.add_argument("--steps", type=int, default=15, help="num_inference_steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lora-path",
        default="/home/andy/model/SD3.5M-FlowGRPO-GenEval",
        help="Server-local LoRA path",
    )
    parser.add_argument("--lora-name", default="SD3.5M-FlowGRPO-GenEval", help="LoRA adapter name")
    parser.add_argument("--out-dir", default="lora_verify_output", help="Output directory")
    parser.add_argument("--mse-threshold", type=float, default=0.2, help="Threshold for near-identical images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = generate_image(
        server=args.server,
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_payload=None,
    )
    l0 = generate_image(
        server=args.server,
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_payload={
            "name": args.lora_name,
            "local_path": args.lora_path,
            "scale": 0.0,
        },
    )
    l05 = generate_image(
        server=args.server,
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_payload={
            "name": args.lora_name,
            "local_path": args.lora_path,
            "scale": 0.5,
        },
    )
    l1 = generate_image(
        server=args.server,
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_payload={
            "name": args.lora_name,
            "local_path": args.lora_path,
            "scale": 1.0,
        },
    )

    base_path = out_dir / "baseline.png"
    l0_path = out_dir / "lora_scale_0.png"
    l05_path = out_dir / "lora_scale_05.png"
    l1_path = out_dir / "lora_scale_1.png"
    base.save(base_path)
    l0.save(l0_path)
    l05.save(l05_path)
    l1.save(l1_path)

    mse_base_l0 = mse(base, l0)
    mse_base_l1 = mse(base, l1)
    mse_l0_l1 = mse(l0, l1)

    print("Saved images:")
    print(f"  {base_path} sha256={sha256(base_path)}")
    print(f"  {l0_path} sha256={sha256(l0_path)}")
    print(f"  {l05_path} sha256={sha256(l05_path)}")
    print(f"  {l1_path} sha256={sha256(l1_path)}")
    print("")
    print("MSE metrics:")
    print(f"  baseline vs lora_scale_0  : {mse_base_l0:.6f}")
    print(f"  baseline vs lora_scale_1  : {mse_base_l1:.6f}")
    print(f"  lora_scale_0 vs scale_1   : {mse_l0_l1:.6f}")

    near_identical_base_l0 = mse_base_l0 <= args.mse_threshold
    has_lora_effect = mse_l0_l1 > args.mse_threshold

    print("")
    if near_identical_base_l0:
        print("CHECK PASS: baseline ~= scale=0.0 (expected).")
    else:
        print("CHECK WARN: baseline != scale=0.0 (possible nondeterminism or pipeline variance).")

    if has_lora_effect:
        print("CHECK PASS: scale=0.0 != scale=1.0 (LoRA likely applied).")
        return 0

    print("CHECK FAIL: scale=0.0 ~= scale=1.0 (LoRA likely not effectively applied).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
