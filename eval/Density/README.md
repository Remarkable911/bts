# Density

`run.py` strictly replays the audited fixed-10 panel from the parent
official-protocol RULER suite. It does not rebuild RULER data or reconstruct a
chat prompt. For each selected class it reads source indices
`0,11,22,33,44,55,66,77,88,99` and takes the exact model string from the
official sample log's `arguments.gen_args_0.arg_0`. The 50 source rows, output
rows, prompt hashes, token counts, and parent raw-result hashes are locked by
`fixed10_input_manifest.json`.

The execution path matches the parent Density scripts: tokenize with pinned
base `Qwen/Qwen3-8B`, repeat the whole token row and truncate it to exactly
`131072`, run an unpatched dense full-model forward to capture the inputs of all
36 attention layers, and only then replay BTS attention once per captured
layer. Later layers therefore receive dense upstream hidden states. Density is
the arithmetic mean of the 36 final logical-block densities; no generation or
decode is performed.

Set the suite location for your machine before running:

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
```

The frozen suite and official output logs are not included in this repository.
For compatibility, omitting the variable retains the legacy lookup under
`$PRISM_REFERENCE_ROOT/results/ruler/topk040_tail256_official_protocol_20261105`,
where `PRISM_REFERENCE_ROOT` defaults to the parent of the checkout.

A real run fails before model loading if the suite, selected 128K JSONL files,
the five exact official output logs, or the bundled parent-plan manifest fail
their hashes. The validated identities and exact execution scope are recorded
in the runtime configuration written to the output directory.

```bash
bash scripts/run_density.sh \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --attention bts \
  --ruler-suite "$RULER_SUITE_ROOT" \
  --samples-per-task 10 \
  --output /external/results/density-bts
```

The suite's `Qwen3-8B-YaRN4` checkpoint was used only when the official prompt
logs were originally generated. It must not be used for Density capture: the
parent result used the pinned base checkpoint (`max_position_embeddings=40960`,
serialized `rope_scaling=null`). The launcher also verifies the base config and
tokenizer file hashes before loading model weights.

The BTS run keeps the frozen block-retention thresholds at `0.93/0.93`, token
ratio `0.4`, and dense tail `256`. The historical parent summary is recorded as
reference-only metadata:

| Task | Historical fixed-10 mean density | `≤10%` |
| --- | ---: | :---: |
| `niah_single_1` | 10.9104511% | no |
| `niah_multikey_1` | 7.4343236% | yes |
| `ruler_fwe` | 11.1898202% | no |
| `ruler_qa_hotpot` | 8.1821132% | yes |
| `niah_multiquery` | 7.4472449% | yes |

Those values come from
`work/density_multiclass_pilot10_topk040_summary_20261107/summary.json`
(SHA-256
`9b2885529829062fa23d1bf747a09dd317a5eb6ee9baed1b61bfe3cd0e2663ca`).
They are not prefilled as the result of a new run. Each run writes its own
measured summary, then compares all 50 unrounded sample means against the exact
parent raw artifacts. `formal_protocol_valid` can only become true when all 50
match (absolute tolerance `1e-15`) and the GPU exclusivity checks pass.
The parent directory has no full-100 Density result for `niah_single_1` or
`ruler_fwe`, so this launcher deliberately reproduces the common audited
fixed-10 panel instead of claiming an unsupported five-class full-100 result.

On successful completion the terminal prints a compact table built only from
the new run's five `task_results`: sample count, mean density, mean pruning,
density standard deviation, pruning range, and the per-class `<=10%` decision.
It also prints the 50-sample parent-match count, formal-protocol status, and
the saved `summary.json` path. The complete audit JSON remains on disk instead
of being dumped to the terminal, and no cross-class grand average is reported.

Generated sample records and summaries are runtime outputs and are never part
of the submission. See [`eval/README.md`](../README.md) for method-specific
density definitions and the formal result boundary.
