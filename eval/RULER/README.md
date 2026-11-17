# RULER

`scripts/run_ruler.sh` directly invokes `run.py` exactly once. Its default is a
non-formal five-task selection, in this order:

1. `niah_single_1`
2. `niah_multikey_1`
3. `ruler_fwe`
4. `ruler_qa_hotpot`
5. `niah_multiquery`

Each frozen leaf contains 100 samples at each of 32K, 64K, and 128K, so the
default run evaluates 1,500 samples. Base task IDs supplied through `--tasks`
are resolved to the suite's `official_fixed_*` leaves before lm-eval starts.

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --model /path/to/Qwen3-8B-YaRN4 \
  --output /external/results/ruler-bts \
  --ruler-suite "$RULER_SUITE_ROOT"
```

Select one or more tasks with a comma-separated argument:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --tasks ruler_fwe,niah_multiquery \
  --output /external/results/ruler-custom
```

The launcher does not enable `--formal` by default. The complete frozen group
still contains all 13 tasks and all three lengths (3,900 samples). Run that
strict protocol explicitly with a new, empty output directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --tasks ruler \
  --formal \
  --output /external/results/ruler-full-formal
```

`compare.py` remains a backwards-compatible wrapper for this complete formal
group and does not launch a methods-by-length worker matrix.

The suite path must contain `manifest.json`, `frozen_files_sha256.json`,
`data_inventory.json`, and `tasks/official_fixed_ruler.yaml`. Before model
loading, the runner verifies all three metadata hashes plus 54 selected files:
the group/scoring files, 13 task YAMLs, and every 32K/64K/128K JSONL. It then
loads `PATH/tasks` as an lm-eval external task directory, sets the model limit
to 131,072, and records the verified manifests in `runtime_config.json`.
The legacy `--methods` spelling is accepted for one method only; the legacy
`--lengths` option, when supplied, must name the complete
`32768,65536,131072` population.

For BTS on RULER, only the low- and high-frequency block-mass thresholds change
to 0.95. Token Top-P remains 0.93, the exact Top-K ratio remains 0.4, and the
dense tail remains Tail256. The effective profile is recorded in the runtime
metadata.

The checked complete 13-task historical reference for this exact 0.95
block-threshold profile is 82.2628205 at 32K, 75.1666667 at 64K, and
69.6769231 at 128K. These full-suite macro values are provenance only for a
selected-task run and are not presented as its aggregate. The raw result
is
`results/ruler/topk040_tail256_official_protocol_threshold095_20261105/outputs/results_2026-11-06T22-31-42.346281.json`
with SHA256
`cc333bc31b199b2ccc1f01a6e586d15d649a71d906020f146107f432960bd406`.
This is provenance for a previously completed run; a newly launched evaluation
must not be described as having reproduced it until it produces matching evidence.
The historical result is not bundled. If you have it, set `PRISM_REFERENCE_ROOT`
to the directory containing that `results/` tree to verify its hash.

`run.py` exposes the same single-run path directly. Create the required YaRN4
overlay with `python eval/common/prepare_yarn_model.py ...` before a formal run.
