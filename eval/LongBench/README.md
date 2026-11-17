# LongBench

`run.py` fixes the shared lm-eval runtime to the bundled LongBench group. The
formal protocol expands to 21 tasks and 4,750 samples at a 32K limit.

```bash
bash scripts/run_longbench.sh \
  --model /path/to/Qwen3-8B \
  --attention bts \
  --max-length 32768 \
  --batch-size 1 \
  --apply-chat-template \
  --formal \
  --output /external/results/longbench-bts
```

LongBench keeps the BTS low/high block-mass thresholds at 0.93. Token Top-P is
0.93, the exact Top-K ratio is 0.4, and the dense tail is Tail256; the runner
records this effective profile in its runtime metadata.

The checked historical Top-K0.4 + Tail256 reference score is 40.8784696310.
The raw result is
`../results/longbench/topk040_tail256_20261031/results_2026-11-01T03-39-55.872698.json`
with SHA256
`9826abeecb30c7c6c1662361cf3bc0b0ec914f1dfa89bd0b2ed8042bc9522b81`.
This is provenance for a previously completed run, not a claim that a newly
launched evaluation reproduced the score.

The entry point does not accept a task or scenario override; LongBench and
RULER remain separate suites. Shared implementation lives in
`eval/common/lm_eval_runner.py`.
