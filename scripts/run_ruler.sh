#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}

reference_root=${PRISM_REFERENCE_ROOT:-$(cd -- "$repo_root/.." && pwd)}
ruler_suite=${RULER_SUITE_ROOT:-$reference_root/results/ruler/topk040_tail256_official_protocol_20261105}
args=("$@")
default_tasks="niah_single_1,niah_multikey_1,ruler_fwe,ruler_qa_hotpot,niah_multiquery"

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
add_value --model "${RULER_MODEL:-$ruler_suite/Qwen3-8B-YaRN4}"
add_value --attention bts
tasks_overridden=false
if has_option --tasks; then
    tasks_overridden=true
fi
add_value --tasks "$default_tasks"
add_value --max-length 131072
add_value --batch-size 1
add_value --device cuda
add_value --dtype bfloat16
output_tag=selected5
if [[ $tasks_overridden == true ]]; then
    output_tag=custom_tasks
fi
add_value --output "$repo_root/results/ruler/bts_topk040_tail256_threshold095_${output_tag}"
add_flag --apply-chat-template
add_flag --trust-remote-code
add_flag --confirm-run-unsafe-code

exec "$python_bin" "$repo_root/eval/RULER/run.py" "${args[@]}"
