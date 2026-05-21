#!/usr/bin/env python3
"""Benchmark verl-omni Qwen-Image rollout through vLLM-Omni.

This script isolates the FlowGRPO rollout generation path without running the
full trainer. It intentionally supports three modes:

* ``serialized``: fair cross-version baseline. Runs N separate ``generate``
  calls. Use this for 0.18.0 vs 0.20/PR3639 comparisons.
* ``batched``: 0.20/PR3639-only B=N path. Calls ``generate_batched`` when the
  installed verl-omni server provides it.
* ``continuous``: vLLM-Omni step-wise continuous batching candidate. This is
  separate from verl's B=N ``enable_batched_diffusion`` path.

Run after activating the intended env. If running outside the ``verl-omni``
checkout, set ``PYTHONPATH=/path/to/verl-omni``.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_PROMPT = (
    "A clean product photo of a white ceramic coffee mug on a wooden desk with "
    "a printed label that reads \"FLOWGRPO TEST\" in large black letters, soft "
    "studio lighting, sharp focus, realistic shadows."
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted text, unreadable letters, watermark, noisy image"
)
QWEN_IMAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN_IMAGE_PROMPT_DROP_TOKENS = 34


def _json_or_none(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    return json.loads(value)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_range(value: str) -> list[int]:
    parsed = _parse_int_list(value)
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("expected START,END")
    return parsed


def _runtime_imports():
    try:
        import ray
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from verl.utils.tokenizer import normalize_token_ids
        from verl.workers.rollout.replica import RolloutMode

        from verl_omni.workers.rollout.replica import DiffusionOutput
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
            vLLMOmniHttpServer,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Activate the intended verl-omni / "
            "vLLM-Omni environment before running this benchmark. "
            f"Original error: {exc}"
        ) from exc

    return {
        "ray": ray,
        "torch": torch,
        "OmegaConf": OmegaConf,
        "AutoTokenizer": AutoTokenizer,
        "normalize_token_ids": normalize_token_ids,
        "RolloutMode": RolloutMode,
        "DiffusionOutput": DiffusionOutput,
        "vLLMOmniHttpServer": vLLMOmniHttpServer,
    }


def _tokenize_prompt(tokenizer: Any, normalize_token_ids: Any, text: str) -> list[int]:
    token_ids = tokenizer(QWEN_IMAGE_PROMPT_TEMPLATE.format(text)).input_ids
    token_ids = normalize_token_ids(token_ids)
    if len(token_ids) <= QWEN_IMAGE_PROMPT_DROP_TOKENS:
        raise ValueError(
            f"Prompt is too short after Qwen-Image templating ({len(token_ids)} tokens). "
            "Qwen-Image drops the first 34 template tokens; use a longer prompt."
        )
    return token_ids


def _maybe_tokenize_negative(
    tokenizer: Any,
    normalize_token_ids: Any,
    text: str | None,
) -> list[int] | None:
    if text is None or text == "":
        return None
    token_ids = normalize_token_ids(tokenizer(QWEN_IMAGE_PROMPT_TEMPLATE.format(text)).input_ids)
    if len(token_ids) <= QWEN_IMAGE_PROMPT_DROP_TOKENS:
        raise ValueError(
            f"Negative prompt is too short after Qwen-Image templating ({len(token_ids)} tokens). "
            "Qwen-Image drops the first 34 template tokens; use a longer negative prompt or disable CFG."
        )
    return token_ids


def _build_sampling_params(args: argparse.Namespace, *, n: int, seed: int) -> dict[str, Any]:
    return {
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "height": args.height,
        "width": args.width,
        "max_sequence_length": args.max_sequence_length,
        "logprobs": args.logprobs,
        "seed": seed,
        "noise_level": args.noise_level,
        "sde_type": args.sde_type,
        "sde_window_size": args.sde_window_size,
        "sde_window_range": args.sde_window_range,
        "num_outputs_per_prompt": n,
    }


def _build_rollout_config(args: argparse.Namespace, OmegaConf: Any, *, n: int) -> Any:
    engine_kwargs = {
        "step_execution": args.step_execution,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.enable_stage_profiler:
        engine_kwargs["enable_diffusion_pipeline_profiler"] = True
    profiler_config = _json_or_none(args.profiler_config_json)
    if profiler_config is not None:
        engine_kwargs["profiler_config"] = profiler_config

    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "load_format": args.load_format,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": not args.log_stats,
            "calculate_log_probs": args.logprobs,
            "n": n,
            "external_lib": args.external_lib,
            "engine_kwargs": {"vllm_omni": engine_kwargs},
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "max_sequence_length": args.max_sequence_length,
            },
            "algo": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionRolloutAlgoConfig",
                "noise_level": args.noise_level,
                "sde_type": args.sde_type,
                "sde_window_size": args.sde_window_size,
                "sde_window_range": args.sde_window_range,
            },
        }
    )


def _build_model_config(args: argparse.Namespace, OmegaConf: Any) -> Any:
    tokenizer_path = args.tokenizer_path or str(Path(args.model) / "tokenizer")
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": args.model,
            "tokenizer_path": tokenizer_path,
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": args.algorithm,
        }
    )


def _summarize_output(output: Any, *, expect_logprobs: bool) -> dict[str, Any]:
    if isinstance(output, list):
        items = output
    else:
        items = [output]

    summary: dict[str, Any] = {"num_outputs": len(items), "logprobs_present": None}
    logprobs_present = []
    shapes = []
    stop_reasons = []
    for item in items:
        diffusion_output = getattr(item, "diffusion_output", None)
        log_probs = getattr(item, "log_probs", None)
        stop_reasons.append(getattr(item, "stop_reason", None))
        logprobs_present.append(log_probs is not None)
        shape = tuple(getattr(diffusion_output, "shape", ())) if diffusion_output is not None else None
        shapes.append(shape)
    summary["shapes"] = [list(shape) if shape is not None else None for shape in shapes]
    summary["stop_reasons"] = stop_reasons
    summary["logprobs_present"] = logprobs_present
    if expect_logprobs and not all(logprobs_present):
        raise RuntimeError(f"Expected logprobs for every output, got {logprobs_present}")
    return summary


def _call_generate(
    ray: Any,
    server: Any,
    *,
    prompt_ids: list[int],
    negative_prompt_ids: list[int] | None,
    sampling_params: dict[str, Any],
    n: int,
    timeout_s: int,
) -> list[Any]:
    outputs = []
    per_sample_params = dict(sampling_params)
    per_sample_params["num_outputs_per_prompt"] = 1
    for idx in range(n):
        outputs.append(
            ray.get(
                server.generate.remote(
                    prompt_ids=prompt_ids,
                    negative_prompt_ids=negative_prompt_ids,
                    sampling_params={**per_sample_params, "seed": sampling_params["seed"] + idx},
                    request_id=f"serialized_{idx}_{uuid4().hex[:8]}",
                ),
                timeout=timeout_s,
            )
        )
    return outputs


def _call_generate_batched(
    ray: Any,
    server: Any,
    server_class: type,
    *,
    prompt_ids: list[int],
    negative_prompt_ids: list[int] | None,
    sampling_params: dict[str, Any],
    n: int,
    timeout_s: int,
) -> Any:
    if not hasattr(server_class, "generate_batched"):
        raise RuntimeError("Installed vLLMOmniHttpServer has no generate_batched method.")

    method = getattr(server, "generate_batched")
    signature = inspect.signature(getattr(server_class, "generate_batched"))
    kwargs = {
        "prompt_ids": prompt_ids,
        "sampling_params": sampling_params,
        "request_id": f"batched_{uuid4().hex[:8]}",
        "num_outputs_per_prompt": n,
    }
    if "negative_prompt_ids" in signature.parameters:
        kwargs["negative_prompt_ids"] = negative_prompt_ids
    return ray.get(method.remote(**kwargs), timeout=timeout_s)


def _sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_once(
    ray: Any,
    torch: Any,
    server: Any,
    server_class: type,
    args: argparse.Namespace,
    *,
    mode: str,
    prompt_ids: list[int],
    negative_prompt_ids: list[int] | None,
    n: int,
    iteration: int,
    measured: bool,
) -> dict[str, Any]:
    sampling_params = _build_sampling_params(args, n=n, seed=args.seed + iteration * 1000)
    _sync_cuda(torch)
    start = time.perf_counter()
    if mode == "serialized" or mode == "continuous":
        output = _call_generate(
            ray,
            server,
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            sampling_params=sampling_params,
            n=n,
            timeout_s=args.timeout_s,
        )
    elif mode == "batched":
        output = _call_generate_batched(
            ray,
            server,
            server_class,
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            sampling_params=sampling_params,
            n=n,
            timeout_s=args.timeout_s,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    _sync_cuda(torch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_summary = _summarize_output(output, expect_logprobs=args.logprobs)
    return {
        "mode": mode,
        "n": n,
        "iteration": iteration,
        "measured": measured,
        "elapsed_ms": elapsed_ms,
        "ms_per_output": elapsed_ms / max(n, 1),
        "output": output_summary,
    }


def _make_benchmark_server_class(base_cls: type) -> type:
    class BenchmarkVLLMOmniHttpServer(base_cls):  # type: ignore[misc, valid-type]
        async def start_omni_profile(self, profile_prefix: str | None = None):
            if not hasattr(self, "engine") or self.engine is None:
                raise RuntimeError("Engine is not initialized.")
            return await self.engine.start_profile(profile_prefix=profile_prefix)

        async def stop_omni_profile(self):
            if not hasattr(self, "engine") or self.engine is None:
                raise RuntimeError("Engine is not initialized.")
            return await self.engine.stop_profile()

    return BenchmarkVLLMOmniHttpServer


def _launch_server(imports: dict[str, Any], args: argparse.Namespace, *, n: int) -> tuple[Any, type]:
    ray = imports["ray"]
    OmegaConf = imports["OmegaConf"]
    RolloutMode = imports["RolloutMode"]
    base_server_cls = imports["vLLMOmniHttpServer"]
    server_cls = _make_benchmark_server_class(base_server_cls)

    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": args.nccl_debug,
        "VLLM_LOGGING_LEVEL": args.vllm_logging_level,
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "NCCL_CUMEM_ENABLE": "0",
    }
    if args.cuda_visible_devices:
        env_vars["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": env_vars}, ignore_reinit_error=True)

    rollout_cfg = _build_rollout_config(args, OmegaConf, n=n)
    model_cfg = _build_model_config(args, OmegaConf)
    ServerCls = ray.remote(server_cls)
    server = ServerCls.options(runtime_env={"env_vars": env_vars}, max_concurrency=10).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=args.gpus_per_node,
        nnodes=1,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    ray.get(server.launch_server.remote(), timeout=args.launch_timeout_s)
    return server, server_cls


def _aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in results:
        if not row["measured"]:
            continue
        grouped.setdefault((row["mode"], row["n"]), []).append(row["elapsed_ms"])

    summary = []
    for (mode, n), values in sorted(grouped.items()):
        summary.append(
            {
                "mode": mode,
                "n": n,
                "repeats": len(values),
                "mean_ms": statistics.mean(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "mean_ms_per_output": statistics.mean(values) / max(n, 1),
            }
        )
    return summary


async def _profile_request_if_needed(
    ray: Any,
    server: Any,
    enabled: bool,
    prefix: str,
    func,
) -> Any:
    if not enabled:
        return func()
    ray.get(server.start_omni_profile.remote(profile_prefix=prefix))
    try:
        return func()
    finally:
        ray.get(server.stop_omni_profile.remote())


def run(args: argparse.Namespace) -> dict[str, Any]:
    imports = _runtime_imports()
    ray = imports["ray"]
    torch = imports["torch"]
    AutoTokenizer = imports["AutoTokenizer"]
    normalize_token_ids = imports["normalize_token_ids"]

    tokenizer_path = args.tokenizer_path or str(Path(args.model) / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_ids = _tokenize_prompt(tokenizer, normalize_token_ids, args.prompt)
    negative_prompt_ids = _maybe_tokenize_negative(tokenizer, normalize_token_ids, args.negative_prompt)

    all_results: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    modes = args.modes
    for mode in modes:
        if mode == "continuous" and not args.step_execution:
            unsupported.append(
                {
                    "mode": mode,
                    "reason": "continuous mode requires --step-execution",
                }
            )
            continue

        for n in args.n_values:
            server, server_cls = _launch_server(imports, args, n=n)
            try:
                if mode == "batched" and not hasattr(server_cls, "generate_batched"):
                    unsupported.append(
                        {
                            "mode": mode,
                            "n": n,
                            "reason": "installed vLLMOmniHttpServer has no generate_batched method",
                        }
                    )
                    continue

                for iteration in range(args.warmups):
                    all_results.append(
                        _time_once(
                            ray,
                            torch,
                            server,
                            server_cls,
                            args,
                            mode=mode,
                            prompt_ids=prompt_ids,
                            negative_prompt_ids=negative_prompt_ids,
                            n=n,
                            iteration=iteration,
                            measured=False,
                        )
                    )

                for repeat in range(args.repeats):
                    iteration = args.warmups + repeat

                    def timed_request():
                        return _time_once(
                            ray,
                            torch,
                            server,
                            server_cls,
                            args,
                            mode=mode,
                            prompt_ids=prompt_ids,
                            negative_prompt_ids=negative_prompt_ids,
                            n=n,
                            iteration=iteration,
                            measured=True,
                        )

                    profile_this = args.profile_once and repeat == 0
                    result = asyncio.run(
                        _profile_request_if_needed(
                            ray,
                            server,
                            profile_this,
                            f"qwen_flowgrpo_{mode}_n{n}",
                            timed_request,
                        )
                    )
                    all_results.append(result)
                    print(
                        json.dumps(
                            {
                                "mode": result["mode"],
                                "n": result["n"],
                                "iteration": result["iteration"],
                                "elapsed_ms": result["elapsed_ms"],
                                "ms_per_output": result["ms_per_output"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                ray.kill(server, no_restart=True)

    metadata = {
        "model": args.model,
        "tokenizer_path": tokenizer_path,
        "n_values": args.n_values,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "logprobs": args.logprobs,
        "step_execution": args.step_execution,
        "max_num_seqs": args.max_num_seqs,
        "cuda_visible_devices": args.cuda_visible_devices,
        "python": os.sys.executable,
    }
    report = {
        "metadata": metadata,
        "results": all_results,
        "summary": _aggregate(all_results),
        "unsupported": unsupported,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "unsupported": unsupported}, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen-Image"))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--algorithm", default="flow_grpo")
    parser.add_argument("--external-lib", default="verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--n-values", type=_parse_int_list, default=[1, 4, 16])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["serialized", "batched", "continuous"],
        default=["serialized"],
        help="serialized is the fair cross-version baseline; batched is 0.20/PR3639-only.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--noise-level", type=float, default=1.2)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--sde-window-size", type=int, default=2)
    parser.add_argument("--sde-window-range", type=_parse_range, default=[0, 5])
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--no-logprobs", dest="logprobs", action="store_false")
    parser.set_defaults(logprobs=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--launch-timeout-s", type=int, default=1800)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=1058)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--enable-stage-profiler", action="store_true")
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--profiler-config-json", default=None)
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--output-json", default="outputs/qwen_flowgrpo_rollout_bench.json")
    parser.add_argument("--nccl-debug", default="WARN")
    parser.add_argument("--vllm-logging-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
