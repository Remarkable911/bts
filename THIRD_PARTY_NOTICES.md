# Third-party notices

## Evaluation baselines

`baselines/` contains the Qwen3 comparison adapters used by the BTS
development project:

- FlashAttention is selected as an external `flash-attn==2.8.3.post1`
  dependency. FlashAttention is BSD-3-Clause; the license is included at
  `baselines/licenses/FLASH_ATTENTION_BSD_3_CLAUSE.txt`.
- The MInference adapter calls `minference==0.1.6.0` and includes a modified
  leak-free index conversion path. Microsoft MInference is MIT licensed; the
  license is included at `baselines/licenses/MINFERENCE_MIT.txt`.
- FlexPrefill's adapter is frozen from the BTS development repository. The
  Apache-2.0 license from the named ByteDance FlexPrefill upstream is included
  as a provenance reference. The adapter also follows code present in the
  XAttention research repository; that repository had no top-level license at
  the audited revision, so this submission does not infer or assert one.
- XAttention's adapter is likewise frozen from the BTS development
  repository. Its external Block-Sparse-Attention kernel is BSD-3-Clause; the
  kernel license is included at
  `baselines/licenses/BLOCK_SPARSE_ATTENTION_BSD_3_CLAUSE.txt`. The
  XAttention Python upstream had no top-level license at the audited revision.

Exact source hashes, upstream revisions, dependency versions, and frozen
parameters are recorded in `baselines/UPSTREAM.json`. No compiled baseline
`.so` is redistributed.

## EleutherAI lm-evaluation-harness

`eval/common/lm_eval_harness/` contains a compact source subset of
EleutherAI lm-evaluation-harness 0.4.9.2: the evaluation runtime and only the
LongBench, RULER, GSM8K and Minerva Math/MATH-500 task definitions used by this
submission. Except for the task-specific files noted below, this subset is
licensed under the MIT License. The upstream notice and full license text are
retained as `UPSTREAM.md` and `LICENSE.md` in that directory.

## NVIDIA RULER task utilities

The RULER utility files `cwe_utils.py`, `essays.py`, `fwe_utils.py`,
`prepare_niah.py`, `qa_utils.py`, and `vt_utils.py` retain NVIDIA's 2024
copyright notice and are licensed under Apache License 2.0. Their source
headers retain the notice and license reference. A complete copy of the
Apache License 2.0 text is included in this distribution at
`baselines/licenses/FLEXPREFILL_APACHE_2_0.txt`.

## LongBench metrics

`eval/common/lm_eval_harness/lm_eval/tasks/longbench/metrics.py`
retains the 2023 THU-KEG & Zhipu AI copyright notice and its MIT License text.

Task datasets and model weights are not redistributed by this submission.
