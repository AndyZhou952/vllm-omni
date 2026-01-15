# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

from vllm_omni.lora.request import LoRARequest
from vllm_omni.entrypoints.omni import Omni

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with LoRA adapters.")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium", help="Text prompt for image generation.")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lora_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to static LoRA adapter (loaded at initialization). "
             "If provided, this LoRA will be active for all requests.",
    )
    parser.add_argument(
        "--lora-request-path",
        type=str,
        default=None,
        help="Path to LoRA adapter for per-request loading (dynamic LoRA). "
             "Requires --lora-request-id to be set.",
    )
    parser.add_argument(
        "--lora-request-id",
        type=int,
        default=None,
        help="Integer ID for the LoRA adapter (for dynamic LoRA). "
             "If not provided and --lora-request-path is set, will use hash of path.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = args.model

    omni_kwargs = {}

    if args.lora_path:
        omni_kwargs["lora_path"] = args.lora_path
        print(f"Using static LoRA from: {args.lora_path}")

    omni = Omni(model=model, **omni_kwargs)

    lora_request = None
    if args.lora_request_path:
        if args.lora_request_id is None:
            lora_request_id = abs(hash(args.lora_request_path)) % (2 ** 30)
        else:
            lora_request_id = args.lora_request_id

        lora_name = Path(args.lora_request_path).stem
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_request_id,
            lora_path=args.lora_request_path,
        )
        print(f"Using per-request LoRA: name={lora_name}, id={lora_request_id}, scale={args.lora_scale}")
    elif args.lora_path:
        # pre-loaded LoRA
        lora_request = LoRARequest(
            lora_name="preloaded",
            lora_int_id=1,
            lora_path=args.lora_path,
        )

    gen_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
    }

    if lora_request:
        gen_kwargs["lora_request"] = lora_request
        gen_kwargs["lora_scale"] = args.lora_scale

    outputs = omni.generate(**gen_kwargs)

    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")

    if isinstance(outputs, list):
        first_output = outputs[0]
    else:
        first_output = outputs

    images = None
    if hasattr(first_output, "images") and first_output.images:
        images = first_output.images
    elif hasattr(first_output, "request_output") and first_output.request_output:
        req_out = first_output.request_output
        if isinstance(req_out, list) and len(req_out) > 0:
            req_out = req_out[0]
        if hasattr(req_out, "images") and req_out.images:
            images = req_out.images

    if not images:
        raise ValueError("No images found in request_output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "lora_output"
    if len(images) <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
