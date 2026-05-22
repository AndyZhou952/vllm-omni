# Qwen-Image FlowGRPO Regression — Tech Lead Summary

## Problem we set out to solve

**Production symptom:** After moving from **vLLM-Omni 0.18.0** to **latest main + vLLM 0.21**, Qwen-Image **FlowGRPO RL rollout** (verl, TP=4, one sample per prompt) is **~60% slower** in steady state.

| What we measured first | 0.18 | Latest | Regression |
|---|---:|---:|---:|
| verl FlowGRPO rollout, warmed `n=1`, **TP=4** (headline) | **1,465 ms/output** | **2,375 ms/output** | **+62%** |

**Workload:** `/mnt/models/hub/Qwen-Image`, 512×512, 10 denoise steps, `true_cfg_scale=4.0`, logprobs on, serialized rollout.

**Open questions we then tested:**

1. Is this only **multi-GPU NCCL**, or also single-GPU?
2. Is it only the **verl + custom FlowGRPO pipeline**, or also **stock** `Omni` image generation?
3. What does the **profiler** show — math, compile, or communication?

---

## How to read the “layers” in this doc

We use **layers** as depth of diagnosis, not separate products:

| Layer | Meaning | Question it answers |
|---|---|---|
| **Layer 1 — E2E latency** | Wall-clock time of a full generate, after warmup | “What does RL / serving pay per image?” |
| **Layer 2 — Profiler breakdown** | Where GPU time goes inside the diffusion worker (`pipeline_forward`, NCCL, `CompiledFxGraph`, GEMM) | “*Why* is it slower — comms, kernels, or missing fusion?” |
| **Layer 3 — Path & scope** | *Which code path* is timed (stock `Omni` vs verl FlowGRPO vs bare `Omni` smoke) | “Is the upgrade bad everywhere, or only on the RL stack?” |

Each layer adds evidence; **root cause** at the end combines all three.

---

## Experiments and data

### Experiment A — verl FlowGRPO rollout (production-shaped RL path)

**What:** `bench_qwen_image_flowgrpo_rollout.py` → Ray `vLLMOmniHttpServer` + custom `QwenImagePipelineWithLogProb` + logprobs.

**Layer 1 — warmed `n=1` (mean of 3 repeats after 1 warmup):**

| TP | GPUs | 0.18 ms/out | Latest ms/out | Latest vs 0.18 |
|---:|---|---:|---:|---:|
| **1** | 1 | **1,089** | **1,732** | **+59%** |
| **2** | 2 | **1,516** | **2,463** | **+63%** |
| **4** | 4 | **1,465** | **2,375** | **+62%** |

**What this experiment reveals:**

- Regression is **real after warmup** (not a cold-start artifact).
- Gap is **almost flat from TP=1 → TP=4 (~59–63%)** → **cannot** be explained by NCCL alone (TP=1 has no cross-GPU tensor parallel).
- **CFG parallel (2×TP=2)** does **not** help on latest (2,552 ms vs 2,375 ms at TP=4).

**Mitigation on latest (same bench):** PR83 **B=N batched** at `n=16` → **305 ms/out** (warmed); **does not** fix `n=1`.

Sources: `raw_results/*_warm1_repeat3_serialized_n1_tp*.json`, `report_new.md`.

---

### Experiment B — Stock `Omni` offline (no custom pipeline, no verl)

**What:** `compare_omni_offline.py` with `--pipeline-mode default` → `Omni(model=...)` only. No `custom_pipeline_args`; vLLM-Omni loads built-in **`QwenImagePipeline`** from its registry (same idea as `omni.generate(prompt)` in docs).

**Layer 1 — single GPU (TP=1), warmed:**

| Path | 0.18 | Latest | Latest vs 0.18 |
|---|---:|---:|---:|
| **Stock `Omni` default** | **1,914 ms** | **2,209 ms** | **+15%** |
| verl FlowGRPO rollout (from Exp A) | **1,089** | **1,732** | **+59%** |

**Layer 2 — profiler, stock `Omni` TP=1 (rank 0, one profiled step):**

| Signal | 0.18 | Latest |
|---|---:|---:|
| `pipeline_forward` | 1.99 s | 2.53 s |
| **`CompiledFxGraph`** | **1.70 s** | **0** |
| `aten::addmm` (eager GEMM) | 0.27 s | 0.54 s |
| NCCL | **0** | **0** |

**What this experiment reveals:**

- The **upgrade also slows stock image inference**, but **much less** (~**+15%**) than the **verl FlowGRPO** path (~**+59%**).
- On **both** versions, verl FlowGRPO rollout is **faster** than offline stock `Omni` at TP=1 (0.18: 1,089 vs 1,914 ms; latest: 1,732 vs 2,209 ms) → the **custom pipeline class is not intrinsically slower**; the RL server path is more optimized for this workload.
- Stock latest shows the **same profiler signature** as FlowGRPO: **missing `CompiledFxGraph`**, more eager GEMM → issue is **vLLM 0.21 / latest worker behavior**, not “custom pipeline only.”

**TP=4 stock `Omni`:** wall times noisy on 0.18 (one **13 s** outlier on first measured repeat); steady repeats ~2.5–3.1 s (0.18) vs ~2.5–3.5 s (latest). Use **verl TP=4** for a clean production TP=4 headline.

Sources: `compare.md`, `raw_results/omni_*_default_*.json`, `profiles/compare_offline/`.

---

### Experiment C — Profiler & controls on the FlowGRPO / verl path

**Layer 2 — FlowGRPO verl rollout, TP=1 (no NCCL):**

| Signal | 0.18 | Latest |
|---|---:|---:|
| `pipeline_forward` | 1.76 s | 2.49 s |
| **`CompiledFxGraph`** | **1.38 s** | **0** |
| `aten::addmm` | 0.12 s | 0.54 s |
| NCCL | 0 | 0 |

**Layer 2 — FlowGRPO verl rollout, TP=4:**

| Signal | 0.18 | Latest |
|---|---:|---:|
| NCCL all-reduce **calls** / rank | 4,800 | 4,800 (unchanged) |
| NCCL time (straggler ranks) | ~1.6–2.0 s | **~2.9 s** |
| **Per-call** all-reduce | ~**330 µs** | ~**610 µs** (~**2×**) |
| **`CompiledFxGraph`** | present | **missing** |

**Layer 3 — bare `Omni` smoke (no verl, no logprobs, TP=1):**

| Stack | Wall time |
|---|---:|
| 0.18 `Omni.generate` | 2.00 s |
| Latest `Omni.generate` | **1.84 s** (faster) |

**What these experiments reveal:**

- **~640 ms** of the TP=1 FlowGRPO gap matches **lost kernel fusion** (Factor A), not NCCL.
- **~900–1,000 ms** at TP=4 matches **2× slower per all-reduce** at the **same 4,800 calls** (Factor B), not more collectives.
- Raw `torch.distributed` NCCL microbench is **identical** 0.18 vs 0.21 (~0.03 ms) → NCCL library is fine; cost is in **vLLM/Omni TP dispatch** (`record_param_comms`, small tensors, 4,800× per request).
- Bare `Omni` without verl/logprobs is **not** slower on latest → the **large RL regression** is tied to **verl rollout + logprob path + latest worker**, on top of a smaller stock regression.

**Remediation knobs tried (latest TP=1):** `enforce_eager`, `ir_enable_torch_wrap: false`, custom AR env — **no** fix; `enforce_eager` was slightly **faster**, so “disable compile” is not the answer.

Sources: `report_new.md`, `raw_results/profiler_analysis_summary.txt`, `raw_results/latest_tp1_*.json`.

---

## Issues ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Cold start only | **Ruled out** | Warmed steady state still +59–62% |
| NCCL-only regression | **Ruled out** | TP=1 still +59%, zero NCCL in profile |
| Attention / MLP kernels regressed | **Ruled out** | FA/GEMM totals small and stable |
| All Qwen inference broken on latest | **Ruled out** | Stock `Omni` only ~+15%; bare smoke faster |
| Custom pipeline inherently slow | **Ruled out** | verl FlowGRPO faster than stock `Omni` on both versions |
| CFG parallel fixes latency | **Ruled out** | Latest CFG 2×2 worse than TP=4 |
| Raw NCCL driver regression | **Ruled out** | Microbench identical |

---

## Root cause (validated)

Two factors stack; both appear on **stock `Omni`** and **verl FlowGRPO**, but the **RL path amplifies** the hit (~59% vs ~15% at TP=1).

| Factor | What broke | Where you see it | Approx. cost |
|---|---|---|---|
| **A — Compile / fusion regression** | Latest runs **eager `aten::addmm`**; 0.18 runs large **`CompiledFxGraph`** regions (`torch.compile` logs success but fusion does not show up in trace) | TP=1 & TP=4; stock `Omni` + verl FlowGRPO profiles | **~600–700 ms** at TP=1 |
| **B — TP all-reduce latency regression** | Same **4,800** `RowParallelLinear` NCCL calls per request; **~2× slower per call** on latest worker ranks | TP≥2 only (FlowGRPO TP=4 production) | **~900–1,000 ms** at TP=4 |

**Why RL sees ~60% but stock `Omni` only ~15%:** Same underlying Factors A+B, but verl FlowGRPO adds stack-specific overhead on latest (Ray server, logprob/SDE subgraph, vLLM 0.21 IR `forward_context` vs 0.18). Stock path still proves the **upgrade regression is not custom-pipeline-exclusive**.

**Nearest code targets:** (1) Restore `CompiledFxGraph` / regional compile on latest diffusion worker; (2) Bisect vLLM 0.21 `tensor_model_parallel_all_reduce` / `record_param_comms` for Qwen dual-stream blocks.

---

## Key results (at a glance)

| Result | Value |
|---|---|
| **Production regression (verl FlowGRPO, TP=4, `n=1`)** | **1,465 → 2,375 ms (+62%)** |
| **Same gap at TP=1 (no NCCL)** | **+59%** → not comms-only |
| **Stock `Omni` upgrade hit (TP=1)** | **+15%** → upgrade hurts all inference, RL worse |
| **Best RL mitigation today (`n=16` batched)** | **305 ms/out** on latest |
| **Root cause** | **A:** missing compile fusion **+ B:** 2× NCCL per-call at TP=4 |

---

## Actions

### Ship / plan now (RL)

- **`n > 1`:** Use **PR83 B=N batched** rollout (~**305 ms/out** warmed at `n=16`).
- **`n = 1`:** Budget **~1.7–2.4 s/output** on latest until Factors A+B are fixed; do **not** rely on CFG parallel for speed.

### Fix (engineering order)

1. **Factor A** — Restore **`CompiledFxGraph`** on latest for default + FlowGRPO workers (IR/compile path, logprob graph interaction). **Success check:** TP=1 verl profile shows `CompiledFxGraph` again; TP=1 latency toward **~1.1 s** class vs 0.18.
2. **Factor B** — Cut per all-reduce latency toward **~330 µs** at TP=4 (vLLM TP path, not raw NCCL tuning). **Success check:** TP=4 verl **~1.5 s** class vs 0.18 **~1.5 s**.
3. **Re-run** `run_tp_cfg_sweep.sh`, `run_compare_no_tp.sh` (stock + verl, single GPU), `run_compare_with_tp.sh` (TP=4 only).
4. **Optional:** Text-stream AR skip (TP≥2); fix `Omni(custom_pipeline)` offline image output (currently fails).

### Reproduce

| Goal | Command / doc |
|---|---|
| Full investigation detail | `report_new.md` |
| Stock vs FlowGRPO scope | `compare.md` |
| No-TP A/B (1 GPU) | `bash run_compare_no_tp.sh` |
| TP=4 A/B (4 GPU) | `bash run_compare_with_tp.sh` |
| RL rollout matrix | `bash run_tp_cfg_sweep.sh` |
| Profiler | `bash run_profiler_followup.sh` |

**Artifacts:** `raw_results/`, `profiles/`, `reproduce_commands.sh`, `analyze_profiler.py`.
