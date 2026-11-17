#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}

args=("$@")

has_option() {
    local wanted=$1 item
    for item in "${args[@]}"; do
        [[ $item == "$wanted" || $item == "$wanted="* ]] && return 0
    done
    return 1
}

add_value() {
    has_option "$1" || args+=("$1" "$2")
}

add_flag() {
    has_option "$1" || args+=("$1")
}

add_value --model "Qwen/Qwen3-8B"
add_value --revision "b968826d9c46dd6066d109eabc6255188de91218"
add_value --attention bts
add_value --max-length 32768
add_value --batch-size 1
add_value --device cuda
add_value --dtype bfloat16
add_value --output "$repo_root/results/longbench/bts_topk040_tail256"
add_flag --apply-chat-template
add_flag --trust-remote-code
add_flag --formal

exec "$python_bin" "$repo_root/eval/LongBench/run.py" "${args[@]}"
