#!/usr/bin/env bash
set -euo pipefail

# Self-contained reproduction runner for:
# Qwen-Image FlowGRPO rollout regression, vLLM-Omni 0.18.0 vs latest main.
#
# This script intentionally runs the experiment .py files from this results
# folder. It clones/checks out external repos into WORKDIR, creates envs, and
# writes all newly generated outputs under RESULTS_DIR by default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# User-configurable inputs
###############################################################################

# Point MODEL to a local Qwen-Image model directory on the target machine for
# offline/repeatable runs. A HuggingFace model id can also be used if the target
# machine has network/model-cache access.
MODEL="${MODEL:-Qwen/Qwen-Image}"

# Scratch space for cloned repos and Python envs.
WORKDIR="${WORKDIR:-${SCRIPT_DIR}/work}"

# Where reproduced logs / json / traces will be written.
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/reproduced_outputs}"

# Repos. Override these if using internal mirrors.
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VERL_OMNI_REPO="${VERL_OMNI_REPO:-https://github.com/volcengine/verl-omni.git}"
VERL_REPO="${VERL_REPO:-https://github.com/verl-project/verl.git}"

# Commits used in the investigation.
VLLM_OMNI_LATEST_REF="${VLLM_OMNI_LATEST_REF:-da536187}"
VERL_OMNI_PR83_REF="${VERL_OMNI_PR83_REF:-3a739f7fd9949738c7b78d520c62445af7713409}"
VERL_REF="${VERL_REF:-f81209acafef9b3d8b5023491951f4f114557c52}"

# Fetch the GitHub PR ref before checking out the pinned verl-omni commit. This
# keeps the script runnable even if the commit is not reachable from main.
VERL_OMNI_EXTRA_FETCH="${VERL_OMNI_EXTRA_FETCH:-pull/83/head:refs/remotes/origin/pr/83}"

# GPU topology used in the report. Override for a different machine.
CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

# This machine used CUDA 12.9. Override VLLM_021_WHEEL for a different CUDA
# wheel, or set INSTALL_VLLM_021_DEFAULT=1 to use `uv pip install vllm==0.21.0`.
VLLM_021_WHEEL="${VLLM_021_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl}"
INSTALL_VLLM_021_DEFAULT="${INSTALL_VLLM_021_DEFAULT:-0}"

# Set RUN_PROFILE=0 to skip the torch profiler trace.
RUN_PROFILE="${RUN_PROFILE:-1}"

###############################################################################
# Derived paths
###############################################################################

VLLM_OMNI_SRC="${WORKDIR}/repos/vllm-omni-main"
VERL_OMNI_SRC="${WORKDIR}/repos/verl-omni-pr83"
ENV018="${WORKDIR}/envs/v018"
ENV021="${WORKDIR}/envs/v021"
VERL_PKG="git+${VERL_REPO}@${VERL_REF}"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_qwen_image_flowgrpo_rollout.py"
DIRECT_SMOKE_SCRIPT="${SCRIPT_DIR}/direct_omni_qwen_image_smoke.py"

mkdir -p "${WORKDIR}/repos" "${WORKDIR}/envs" "${RESULTS_DIR}"

###############################################################################
# Helpers
###############################################################################

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

clone_or_update() {
  local repo_url="$1"
  local dest="$2"
  local ref="$3"
  local extra_fetch="${4:-}"

  if [ ! -d "${dest}/.git" ]; then
    git clone "${repo_url}" "${dest}"
  fi
  git -C "${dest}" fetch --all --tags
  if [ -n "${extra_fetch}" ]; then
    git -C "${dest}" fetch origin "${extra_fetch}" || true
  fi
  git -C "${dest}" checkout "${ref}"
}

ray_stop() {
  local py="$1"
  "${py}" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
}

print_versions() {
  local title="$1"
  local py="$2"
  "${py}" - <<PY
import importlib.metadata as m
print("${title}")
for p in ["torch", "vllm", "vllm-omni", "verl", "verl-omni", "diffusers", "transformers", "numpy", "ray"]:
    try:
        print(p, m.version(p))
    except Exception as exc:
        print(p, "MISSING", exc)
try:
    import torch
    print("torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch_check_failed", exc)
PY
}

common_args=(
  --model "${MODEL}"
  --height 512
  --width 512
  --num-inference-steps 10
  --true-cfg-scale 4.0
  --max-sequence-length 256
  --noise-level 1.2
  --sde-type sde
  --sde-window-size 2
  --sde-window-range 0,5
  --warmups 0
  --repeats 1
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_LIST}"
  --gpus-per-node "${GPUS_PER_NODE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

steady_args=(
  "${common_args[@]}"
  --warmups 1
  --repeats 3
)

profile_args=(
  "${common_args[@]}"
  --warmups 1
  --repeats 1
)

runtime_deps=(
  "Levenshtein"
  "accelerate"
  "cachetools"
  "codetiming"
  "datasets"
  "diffusers"
  "dill"
  "hydra-core"
  "numpy<2.0.0"
  "omegaconf"
  "packaging>=20.0"
  "pandas"
  "peft"
  "pyarrow>=19.0.0"
  "pybind11"
  "pylatexenc"
  "ray[default]>=2.41.0"
  "tensorboard"
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
  "torchdata"
  "transformers"
  "wandb"
)

###############################################################################
# 0. Preconditions
###############################################################################

require_cmd git
require_cmd uv

if [ ! -d "${MODEL}" ]; then
  echo "MODEL is not a local directory: ${MODEL}" >&2
  echo "Continuing; this must be a resolvable model id or present in the HF cache." >&2
fi

if [ ! -f "${BENCH_SCRIPT}" ] || [ ! -f "${DIRECT_SMOKE_SCRIPT}" ]; then
  echo "Missing local experiment scripts in ${SCRIPT_DIR}" >&2
  exit 1
fi

###############################################################################
# 1. Clone/check out repos
###############################################################################

clone_or_update "${VLLM_OMNI_REPO}" "${VLLM_OMNI_SRC}" "${VLLM_OMNI_LATEST_REF}"
clone_or_update "${VERL_OMNI_REPO}" "${VERL_OMNI_SRC}" "${VERL_OMNI_PR83_REF}" "${VERL_OMNI_EXTRA_FETCH}"

###############################################################################
# 2. Create environments
###############################################################################

if [ ! -x "${ENV018}/bin/python" ]; then
  uv venv --python 3.12 --seed "${ENV018}"
fi

# Baseline dependency stack.
# --no-deps on editable verl-omni is deliberate: the vLLM/vLLM-Omni stack is
# pinned first so resolver conflicts do not replace it.
uv pip install --python "${ENV018}/bin/python" "vllm==0.18.0" "vllm-omni==0.18.0"
uv pip install --python "${ENV018}/bin/python" --no-deps "${VERL_PKG}"
uv pip install --python "${ENV018}/bin/python" -e "${VERL_OMNI_SRC}" --no-deps
uv pip install --python "${ENV018}/bin/python" "${runtime_deps[@]}"

if [ ! -x "${ENV021}/bin/python" ]; then
  uv venv --python 3.12 --seed "${ENV021}"
fi

if [ "${INSTALL_VLLM_021_DEFAULT}" = "1" ]; then
  uv pip install --python "${ENV021}/bin/python" "vllm==0.21.0" --torch-backend=auto
else
  uv pip install --python "${ENV021}/bin/python" --force-reinstall --no-deps "${VLLM_021_WHEEL}"
fi
uv pip install --python "${ENV021}/bin/python" -e "${VLLM_OMNI_SRC}"
uv pip install --python "${ENV021}/bin/python" --no-deps "${VERL_PKG}"
uv pip install --python "${ENV021}/bin/python" -e "${VERL_OMNI_SRC}" --no-deps
uv pip install --python "${ENV021}/bin/python" "${runtime_deps[@]}"

###############################################################################
# 3. Record environment state
###############################################################################

print_versions "BASELINE_ENV" "${ENV018}/bin/python" | tee "${RESULTS_DIR}/versions_v018.txt"
print_versions "LATEST_ENV" "${ENV021}/bin/python" | tee "${RESULTS_DIR}/versions_latest.txt"

###############################################################################
# 4. Functional smoke
###############################################################################

ray_stop "${ENV021}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV021}/bin/python" "${DIRECT_SMOKE_SCRIPT}" \
  --model "${MODEL}" \
  --output-image "${RESULTS_DIR}/direct_omni_qwen_image_smoke.png" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  > "${RESULTS_DIR}/direct_omni_qwen_image_smoke.log" 2>&1

ray_stop "${ENV021}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV021}/bin/python" "${BENCH_SCRIPT}" \
  "${common_args[@]}" \
  --modes serialized \
  --n-values 1 \
  --output-json "${RESULTS_DIR}/latest_smoke_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/latest_smoke_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

###############################################################################
# 5. Warmed steady-state comparison used for the headline regression
###############################################################################

ray_stop "${ENV018}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV018}/bin/python" "${BENCH_SCRIPT}" \
  "${steady_args[@]}" \
  --modes serialized \
  --n-values 1 \
  --output-json "${RESULTS_DIR}/v018_warm1_repeat3_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/v018_warm1_repeat3_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

ray_stop "${ENV021}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV021}/bin/python" "${BENCH_SCRIPT}" \
  "${steady_args[@]}" \
  --modes serialized \
  --n-values 1 \
  --output-json "${RESULTS_DIR}/latest_warm1_repeat3_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/latest_warm1_repeat3_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

###############################################################################
# 6. One-shot shape matrix
###############################################################################

# Fair baseline: 0.18 serialized only.
ray_stop "${ENV018}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV018}/bin/python" "${BENCH_SCRIPT}" \
  "${common_args[@]}" \
  --modes serialized \
  --n-values 1,4,16 \
  --output-json "${RESULTS_DIR}/v018_matrix_serialized_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/v018_matrix_serialized_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

# Latest serialized plus PR83 B=N batched path.
ray_stop "${ENV021}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV021}/bin/python" "${BENCH_SCRIPT}" \
  "${common_args[@]}" \
  --modes serialized batched \
  --n-values 1,4,16 \
  --output-json "${RESULTS_DIR}/latest_matrix_serialized_batched_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/latest_matrix_serialized_batched_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

# Step-execution runnable check. This script sends N requests serially, so it is
# not a true concurrent continuous-batching throughput benchmark.
ray_stop "${ENV021}/bin/python"
PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
"${ENV021}/bin/python" "${BENCH_SCRIPT}" \
  "${common_args[@]}" \
  --modes continuous \
  --step-execution \
  --max-num-seqs 4 \
  --n-values 1,4,16 \
  --output-json "${RESULTS_DIR}/latest_matrix_continuous_tp${TENSOR_PARALLEL_SIZE}.json" \
  > "${RESULTS_DIR}/latest_matrix_continuous_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

###############################################################################
# 7. Optional matched warmed profiler capture
###############################################################################

if [ "${RUN_PROFILE}" = "1" ]; then
  PROFILE_DIR="${RESULTS_DIR}/profiles/v018_warm_serialized_n1_tp${TENSOR_PARALLEL_SIZE}"
  mkdir -p "${PROFILE_DIR}"
  ray_stop "${ENV018}/bin/python"
  PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
  "${ENV018}/bin/python" "${BENCH_SCRIPT}" \
    "${profile_args[@]}" \
    --modes serialized \
    --n-values 1 \
    --profiler-config-json "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"active_iterations\":1,\"max_iterations\":1}" \
    --profile-once \
    --output-json "${RESULTS_DIR}/v018_warm1_torchprof_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.json" \
    > "${RESULTS_DIR}/v018_warm1_torchprof_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1

  PROFILE_DIR="${RESULTS_DIR}/profiles/latest_warm_serialized_n1_tp${TENSOR_PARALLEL_SIZE}"
  mkdir -p "${PROFILE_DIR}"
  ray_stop "${ENV021}/bin/python"
  PYTHONPATH="${VERL_OMNI_SRC}:${PYTHONPATH:-}" \
  "${ENV021}/bin/python" "${BENCH_SCRIPT}" \
    "${profile_args[@]}" \
    --modes serialized \
    --n-values 1 \
    --profiler-config-json "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"active_iterations\":1,\"max_iterations\":1}" \
    --profile-once \
    --output-json "${RESULTS_DIR}/latest_warm1_torchprof_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.json" \
    > "${RESULTS_DIR}/latest_warm1_torchprof_serialized_n1_tp${TENSOR_PARALLEL_SIZE}.log" 2>&1
fi

echo "Done. Results written to: ${RESULTS_DIR}"
