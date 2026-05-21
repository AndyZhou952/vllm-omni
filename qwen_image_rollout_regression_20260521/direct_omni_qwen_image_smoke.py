#!/usr/bin/env python3
"""Direct vLLM-Omni Qwen-Image smoke test.

This must live in a real file because vLLM-Omni launches spawned worker
processes that need to re-import the main module.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path or HF id for Qwen-Image.")
    parser.add_argument(
        "--output-image",
        default=None,
        help="Optional path where the first generated image should be saved.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("constructing omni", flush=True)
    omni = Omni(
        model=args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("generating", flush=True)
    start = time.perf_counter()
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
    )
    outputs = omni.generate(
        {
            "prompt": "A small red cube on a white table, product photo, sharp focus.",
            "negative_prompt": "blurry, low quality, distorted text, watermark",
        },
        sampling_params,
    )
    print("elapsed", time.perf_counter() - start, flush=True)
    images = outputs[0].request_output.images
    print("images", len(images), flush=True)
    if images and args.output_image:
        output_image = Path(args.output_image)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_image)
        print("saved image", flush=True)


if __name__ == "__main__":
    main()
