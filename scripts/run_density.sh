#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}

reference_root=${PRISM_REFERENCE_ROOT:-$(cd -- "$repo_root/.." && pwd)}
ruler_suite=${RULER_SUITE_ROOT:-$reference_root/results/ruler/topk040_tail256_official_protocol_20261105}
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

add_value --ruler-suite "$ruler_suite"
add_value --model "${DENSITY_MODEL:-Qwen/Qwen3-8B}"
add_value --revision "${DENSITY_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
add_value --attention bts
add_value --samples-per-task 10
add_value --tasks "niah_single_1,niah_multikey_1,ruler_fwe,ruler_qa_hotpot,niah_multiquery"
add_value --device cuda
add_value --output "$repo_root/results/density/bts_topk040_tail256_selected5_pilot10"
add_flag --trust-remote-code
add_flag --local-files-only

exec "$python_bin" "$repo_root/eval/Density/run.py" "${args[@]}"
