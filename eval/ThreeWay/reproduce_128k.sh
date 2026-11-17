#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval/ThreeWay/reproduce_128k.sh [OUTPUT_DIR] [GPU]

The defaults reproduce the completed H100 128K four-arm experiment on this
machine. Every path and timeout can also be overridden with an environment
variable:

  FASTDEPLOY_PYTHON, VLLM_PYTHON
  FASTDEPLOY_MODEL, VLLM_MODEL, BTS_CONFIG
  OUTPUT, GPU, WAIT_FOR_GPU_SECONDS, GPU_POLL_SECONDS
  GPU_IDLE_STABILITY_SECONDS, WORKER_TIMEOUT_SECONDS
  MAX_CONTAMINATION_RETRIES_PER_ARM
  MAX_WORKER_FAILURE_RETRIES_PER_ARM
  GPU_RECOVERY_TIMEOUT_SECONDS

Examples:
  bash eval/ThreeWay/reproduce_128k.sh
  WAIT_FOR_GPU_SECONDS=604800 GPU=0 bash eval/ThreeWay/reproduce_128k.sh

At exit the script prints four-arm median measured times and throughputs, all
three speedup comparisons, their 95% lower bounds, the 1.5x target, total wall
time, and artifact paths.  During execution a live 20-worker tqdm bar shows the
current restart, arm, retry attempt, phase, elapsed time, and ETA.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 2 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

FASTDEPLOY_PYTHON="${FASTDEPLOY_PYTHON:-/data3/lsf_data/prism_artifacts/fastdeploy_20261031_deployment/venv/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data3/lsf_data/miniconda3/envs/llm/bin/python}"
FASTDEPLOY_MODEL="${FASTDEPLOY_MODEL:-/data3/lsf_data/prism_artifacts/fastdeploy_128k_acceptance_20261109/Qwen3-8B-YaRN4-paddleformers}"
VLLM_MODEL="${VLLM_MODEL:-/data3/lsf_data/prism_data/results/ruler/topk040_tail256_yarn4_20261104/Qwen3-8B-YaRN4}"
BTS_CONFIG="${BTS_CONFIG:-${REPO_ROOT}/integrations/fastdeploy/configs/topk040_tail256.json}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="/data3/lsf_data/prism_artifacts/bts_three_way_128k_repro_${RUN_STAMP}"
OUTPUT_DIR="${1:-${OUTPUT:-${DEFAULT_OUTPUT}}}"
GPU_SELECTOR="${2:-${GPU:-0}}"

WAIT_FOR_GPU_SECONDS="${WAIT_FOR_GPU_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
GPU_IDLE_STABILITY_SECONDS="${GPU_IDLE_STABILITY_SECONDS:-30}"
WORKER_TIMEOUT_SECONDS="${WORKER_TIMEOUT_SECONDS:-14400}"
MAX_CONTAMINATION_RETRIES_PER_ARM="${MAX_CONTAMINATION_RETRIES_PER_ARM:-0}"
MAX_WORKER_FAILURE_RETRIES_PER_ARM="${MAX_WORKER_FAILURE_RETRIES_PER_ARM:-0}"
GPU_RECOVERY_TIMEOUT_SECONDS="${GPU_RECOVERY_TIMEOUT_SECONDS:-0}"

for executable in "${FASTDEPLOY_PYTHON}" "${VLLM_PYTHON}"; do
  if [[ ! -x ${executable} ]]; then
    echo "error: Python executable is missing or not executable: ${executable}" >&2
    exit 2
  fi
done
for directory in "${FASTDEPLOY_MODEL}" "${VLLM_MODEL}"; do
  if [[ ! -d ${directory} ]]; then
    echo "error: model directory is missing: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f ${BTS_CONFIG} ]]; then
  echo "error: BTS config is missing: ${BTS_CONFIG}" >&2
  exit 2
fi
if [[ -e ${OUTPUT_DIR} ]]; then
  echo "error: output must not already exist: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT_DIR}")"

arguments=(
  --fastdeploy-python "${FASTDEPLOY_PYTHON}"
  --vllm-python "${VLLM_PYTHON}"
  --fastdeploy-model "${FASTDEPLOY_MODEL}"
  --vllm-model "${VLLM_MODEL}"
  --bts-config "${BTS_CONFIG}"
  --gpu "${GPU_SELECTOR}"
  --logical-model-sha256 1669659312e50c6ccd83b884ab01ca80d1036c0781d3a386b6b4851978bf648c
  --tokenizer-sha256 71f004fa22c6561b00dbc2053c38548221a386d902250d497f20faf26e9285cc
  --request-count 16
  --vocab-size 151936
  --gpu-memory-utilization 0.90
  --worker-timeout-seconds "${WORKER_TIMEOUT_SECONDS}"
  --wait-for-gpu-seconds "${WAIT_FOR_GPU_SECONDS}"
  --gpu-poll-seconds "${GPU_POLL_SECONDS}"
  --gpu-idle-stability-seconds "${GPU_IDLE_STABILITY_SECONDS}"
  --max-contamination-retries-per-arm "${MAX_CONTAMINATION_RETRIES_PER_ARM}"
  --max-worker-failure-retries-per-arm "${MAX_WORKER_FAILURE_RETRIES_PER_ARM}"
  --gpu-recovery-timeout-seconds "${GPU_RECOVERY_TIMEOUT_SECONDS}"
  --output "${OUTPUT_DIR}"
)

echo "repository:       ${REPO_ROOT}"
echo "GPU selector:     ${GPU_SELECTOR}"
echo "output:           ${OUTPUT_DIR}"
echo "FastDeploy Python: ${FASTDEPLOY_PYTHON}"
echo "vLLM Python:       ${VLLM_PYTHON}"
echo "wait for GPU:      ${WAIT_FOR_GPU_SECONDS}s"

exec "${VLLM_PYTHON}" -m eval.ThreeWay.execute "${arguments[@]}"
