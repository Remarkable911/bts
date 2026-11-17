# Baseline methods

This directory freezes the four comparison methods used by the BTS
development project:

| CLI value | Execution path | Frozen parameters |
|---|---|---|
| `flashattention` | Transformers `flash_attention_2` | FlashAttention 2.8.3.post1 |
| `flexprefill` | local Qwen3 forward adapter | gamma 0.95, tau 0.1, block 128, no min/max budget, GQA interleave off |
| `minference` | local memory-hardened Qwen3 adapter + MInference operator | vertical 512, slash 2048, adaptive budget off, block 128, Triton fallback |
| `xattention` | local Qwen3 adapter + Block-Sparse-Attention | stride 8, threshold 0.9, block 128, sink/recent kept |

The registry is lazy, and evaluation runs validate exact dependency versions
before model weights are loaded. Conflicting environment overrides are rejected
instead of silently changing a formal method.

Install the baseline stack in the same Python 3.10/CUDA environment:

```bash
python -m pip install packaging psutil ninja
python -m pip install --no-build-isolation \
  -r eval/requirements-baselines.txt
```

LongBench can select one method directly. The frozen RULER compatibility
launcher also accepts exactly one method and evaluates the complete 32K, 64K,
and 128K population in one invocation:

```bash
python eval/LongBench/run.py \
  --model /path/to/Qwen3-8B --revision IMMUTABLE_REVISION \
  --attention flashattention --max-length 32768 --batch-size 1 \
  --device cuda --dtype bfloat16 --apply-chat-template --formal \
  --output /external/results/longbench-flashattention
python eval/RULER/compare.py \
  --model /path/to/Qwen3-8B-YaRN4 \
  --revision IMMUTABLE_REVISION --methods bts \
  --ruler-suite /path/to/frozen-ruler-suite \
  --output-root /external/results/ruler-bts
```

The RULER launcher rejects method matrices and partial length populations. A
complete method run contains 3,900 samples; evaluating another method requires
a separate invocation and output directory. The driver supplies 100 samples
per task and length, batch size 1, BF16 CUDA, chat-template application,
unsafe-code confirmation, and `--formal`; GPU visibility remains controlled
externally by `CUDA_VISIBLE_DEVICES`.

`eval/RULER/run.py` remains the direct single-method worker when an isolated
baseline check is needed.

The XAttention kernel is not published on PyPI. The requirements file installs
the audited Block-Sparse-Attention Git revision recorded in `UPSTREAM.json`;
its installed distribution version is still checked as `0.0.2` before a real
run.

FlashAttention is not the legacy `dense` alias: the former explicitly asks
Transformers for `flash_attention_2`, while `dense` explicitly uses SDPA.

The XAttention source is preserved exactly as the development project handles
it. The upstream XAttention repository did not contain a top-level license at
the audited revision; this submission does not infer or assert one. Its
external Block-Sparse-Attention kernel is BSD-3-Clause and its license is
bundled. See `UPSTREAM.json` and `THIRD_PARTY_NOTICES.md`.
