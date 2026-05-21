# Qwen-Image FlowGRPO Rollout Regression

## Executive Summary

Both target stacks are runnable with the packaged scripts: baseline `vllm/vllm-omni 0.18.0` and latest `vllm 0.21.0+cu129` with editable `vllm-omni` main. The key regression is in TP=4 serialized Qwen-Image FlowGRPO rollout.

After one warmup request, latest main is still slower for serialized `n=1`: `2,447 ms/output` vs `1,465 ms/output` on 0.18 (`+67%`). The matched profiler points to TP all-reduce overhead from Qwen-Image's all-reduce-heavy dual-stream transformer blocks, not attention or MLP math kernels.

## Repro Package

- Experiment script: `bench_qwen_image_flowgrpo_rollout.py`
- Direct vLLM-Omni smoke script: `direct_omni_qwen_image_smoke.py`
- Portable reproduction runner: `reproduce_commands.sh`
- Raw benchmark JSONs: `raw_results/`
- Matched warmed torch-profiler traces: `profiles/`

To reproduce on another machine, set `MODEL=/path/to/Qwen-Image` and run `./reproduce_commands.sh` from this folder. The script runs the local `.py` files in this package, clones the required repos into `work/`, creates isolated envs, and writes regenerated outputs to `reproduced_outputs/`.

## Setup Tested

- Hardware: 4 x NVIDIA L20X, driver `570.133.20`, CUDA driver/runtime `12.9`.
- Baseline: `torch 2.10.0+cu128`, `vllm 0.18.0`, `vllm-omni 0.18.0`.
- Latest: `torch 2.11.0+cu129`, `vllm 0.21.0+cu129`, editable `vllm-omni 0.1.dev1666+gda5361879`.
- Workload: Qwen-Image FlowGRPO custom rollout, TP=4, 512x512, 10 steps, `true_cfg_scale=4.0`, logprobs enabled.

## Fair Comparison

This is the apples-to-apples serialized comparison because `0.18.0` does not support the verl PR83 B=N batched rollout path.

| Measurement | 0.18 serialized | Latest serialized | Delta |
|---|---:|---:|---:|
| First request, `n=1` | 11,668 ms/output | 14,646 ms/output | +25.5% |
| Warmed steady state, `n=1`, mean of 3 | 1,465 ms/output | 2,447 ms/output | +67.1% |
| Warmed torch-profiler run, `n=1` | 2,553 ms/output | 3,621 ms/output | +41.8% |

The profiler run adds overhead and is diagnostic only. The steady-state benchmark row is the best current latency claim.

The larger `n` matrix below was a one-shot run without explicit warmup, so use it as rollout-shape context rather than the final steady-state number:

| Samples per prompt | 0.18 serialized ms/output | Latest serialized ms/output | Delta |
|---:|---:|---:|---:|
| 1 | 11,668.20 | 14,645.51 | +25.5% |
| 4 | 3,657.88 | 5,610.05 | +53.4% |
| 16 | 2,083.49 | 3,123.61 | +49.9% |

## Latest-Only Modes

| Samples per prompt | Latest serialized ms/output | PR83 B=N batched ms/output | Step-execution sequential ms/output |
|---:|---:|---:|---:|
| 1 | 14,645.51 | 14,591.93 | 14,751.99 |
| 4 | 5,610.05 | 6,500.58 | 5,714.49 |
| 16 | 3,123.61 | 1,275.42 | 3,195.98 |

Interpretation:

- Latest serialized is slower than 0.18 serialized after warmup, so the regression exists independently of PR83 B=N batching.
- PR83 B=N batching is not helpful at `n=4` in this one-shot run, but it is strongly beneficial at `n=16`.
- The step-execution run here only validates that the mode is runnable. The benchmark submits N requests serially, so this is not a true concurrent continuous-batching throughput measurement.

## What Is Working

- Direct latest-main `Omni` works with a negative prompt and `true_cfg_scale=4.0`.
- The custom FlowGRPO rollout works on latest main through the packaged benchmark script.
- TP=4 serialized, batched, and step-execution modes all complete.
- Logprobs are present in all measured outputs.

## Why Latest Is Slower

The warmed A/B profiler points to tensor-parallel communication. The Qwen-Image transformer has 60 dual-stream blocks. For each denoising step and each CFG branch, every block performs four TP `RowParallelLinear` reductions:

- image attention output: `QwenImageCrossAttention.to_out`
- text attention output: `QwenImageCrossAttention.to_add_out`
- image MLP output: `QwenImageTransformerBlock.img_mlp`
- text MLP output: `QwenImageTransformerBlock.txt_mlp`

With 10 steps and true CFG, this is `60 blocks x 10 steps x 2 CFG passes x 4 reductions = 4,800 NCCL all-reduce kernels per rank`. The profiler shows exactly 4,800 NCCL all-reduce kernel calls per rank.

Matched warmed profiler evidence:

| Signal | 0.18 warmed profile | Latest warmed profile | Interpretation |
|---|---:|---:|---|
| Profiler wall time | 2,553 ms | 3,621 ms | +1,068 ms diagnostic delta |
| Max `record_param_comms` self-CUDA | 2,014 ms | 3,011 ms | +997 ms, nearly the full profiler delta |
| Max trace `nccl:all_reduce` annotation | 2,192 ms | 3,256 ms | +1,064 ms, same scale as latency delta |
| NCCL kernel count | 4,800/rank | 4,800/rank | Count is unchanged; per-collective cost is worse |
| FlashAttention kernel total | ~39 ms/rank | ~39 ms/rank | Attention is not the regression |
| Largest GEMM buckets | ~170-230 ms/rank | ~170-230 ms/rank | MLP/linear math is not the regression |

This does not prove a single changed line yet, but it narrows the problem to the TP all-reduce path exercised by Qwen-Image's `RowParallelLinear` layers: `vllm.model_executor.layers.linear.RowParallelLinear.forward()` calls `tensor_model_parallel_all_reduce()`, which dispatches into vLLM's CUDA/NCCL communicator. The Qwen model code creates a very high-frequency all-reduce workload; latest vLLM/vLLM-Omni pays about 1 ms more aggregate communication time per request on this workload.

## Actionable Next Steps

1. Treat `RowParallelLinear` all-reduce as the primary code suspect. Validate by running the same warmed `n=1` test with TP=1 and TP=2; if the gap collapses, the regression is confirmed as communication-path, not model math.
2. Test CFG parallel (`CFG=2 x TP=2` on four GPUs) so the positive/negative CFG branches no longer run as two sequential TP=4 passes. This should reduce the sequential 4,800-all-reduce critical path.
3. Prototype a Qwen-specific communication reduction for the small text stream: avoid TP all-reduce for `to_add_out` / `txt_mlp` or keep text stream replicated, then A/B latency and memory. The text sequence is small, so those reductions are likely poor tradeoffs.
4. Keep PR83 B=N batching for high-sample rollout; latest `n=16` batched reached `1,275 ms/output` in the one-shot matrix.
5. After the communication fix/config is chosen, rerun `--warmups 1 --repeats 3` for `n=1,4,16` and a full RL rollout trace.

## Bottom Line

Latest main is runnable, but steady-state serialized TP=4 rollout is still slower than 0.18. The current best explanation is not a bad attention/MLP kernel; it is the high-frequency TP all-reduce path in Qwen-Image dual-stream blocks, where latest spends about 1s more communication time per profiled `n=1` request. The nearest code targets are Qwen-Image `RowParallelLinear` outputs and vLLM's tensor-parallel all-reduce dispatch.
