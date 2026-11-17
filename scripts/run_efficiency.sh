#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}

replay_root=${EFFICIENCY_REPLAY_ROOT:-/data3/lsf_data/prism_artifacts/efficiency_review_20261009_inputs}
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

has_input_source() {
    has_option --attention-inputs-dir || has_option --data-path || has_option --prompt
}

requested_layer=12
for ((index = 0; index < ${#args[@]}; index++)); do
    case ${args[$index]} in
        --layer)
            if ((index + 1 < ${#args[@]})); then
                requested_layer=${args[$((index + 1))]}
            fi
            ;;
        --layer=*) requested_layer=${args[$index]#--layer=} ;;
    esac
done

add_value --model "Qwen/Qwen3-8B"
add_value --revision "b968826d9c46dd6066d109eabc6255188de91218"
if ! has_input_source; then
    if [[ $requested_layer == 12 || -n ${EFFICIENCY_REPLAY_ROOT:-} ]]; then
        add_value --attention-inputs-dir "$replay_root"
    fi
fi
add_value --methods "minference,flexprefill,xattention,bts"
add_value --layer 12
add_value --lengths "32768,65536,131072"
add_value --warmup 3
add_value --runs 8
add_value --rounds 3
add_value --device "cuda:0"
output_mode=capture
has_option --attention-inputs-dir && output_mode=replay
output_layer=$requested_layer
[[ $output_layer =~ ^[0-9]+$ ]] || output_layer=custom
add_value --output "$repo_root/results/efficiency/bts_topk040_tail256_layer${output_layer}_${output_mode}.json"

exec "$python_bin" "$repo_root/eval/efficiency/real_layer.py" "${args[@]}"
