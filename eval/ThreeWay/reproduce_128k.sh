#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval/ThreeWay/reproduce_128k.sh [OUTPUT_DIR] [GPU]

Run the H100 128K four-arm protocol with your own installed environments and
prepared model directories. These environment variables are required:

  FASTDEPLOY_PYTHON, VLLM_PYTHON
  FASTDEPLOY_MODEL, VLLM_MODEL

Optional overrides:

  BTS_CONFIG
  OUTPUT, GPU, WAIT_FOR_GPU_SECONDS, GPU_POLL_SECONDS
  GPU_IDLE_STABILITY_SECONDS, WORKER_TIMEOUT_SECONDS
  MAX_CONTAMINATION_RETRIES_PER_ARM
  MAX_WORKER_FAILURE_RETRIES_PER_ARM
  GPU_RECOVERY_TIMEOUT_SECONDS

Examples:
  export FASTDEPLOY_PYTHON=/absolute/path/to/fastdeploy/bin/python
  export VLLM_PYTHON=/absolute/path/to/vllm/bin/python
  export FASTDEPLOY_MODEL=/absolute/path/to/Qwen3-8B-YaRN4-paddleformers
  export VLLM_MODEL=/absolute/path/to/Qwen3-8B-YaRN4
  bash eval/ThreeWay/reproduce_128k.sh

The default output is a new directory under the repository's results/throughput/.
The fixed model/tokenizer hashes must still match the reference experiment;
overriding a path does not change the protocol's expected model contents.

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

for required in FASTDEPLOY_PYTHON VLLM_PYTHON FASTDEPLOY_MODEL VLLM_MODEL; do
  if [[ -z ${!required:-} ]]; then
    echo "error: set ${required} to a path on this machine; see --help" >&2
    exit 2
  fi
done
BTS_CONFIG="${BTS_CONFIG:-${REPO_ROOT}/integrations/fastdeploy/configs/topk040_tail256.json}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${REPO_ROOT}/results/throughput/four-arm-${RUN_STAMP}"
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
