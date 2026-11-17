# Bundled lm-evaluation-harness subset

- Project: EleutherAI lm-evaluation-harness
- Version: 0.4.9.2
- License: MIT; see `LICENSE.md`
- Included framework: the Python runtime required by `simple_evaluate`
- Included tasks: LongBench, RULER, GSM8K, Minerva Math and MATH-500
- Excluded: unrelated tasks, upstream tests, documentation, examples, build
  metadata, caches and development scripts

The task sources retain their own copyright and license notices where present.

Submission-local reproducibility patches pin the LongBench dataset revision to
`944462f8d7475c8c153be22f890a299fe30254ee`, the Paul Graham essay revision to
`792d672b77d1a67720f7e9820d646358fc18792e`, and the HotpotQA Hub fallback to
`1908d6afbbead072334abe2965f91bd2709910ab`.  The compact RULER download helper
also adds retryable, SHA-256-locked raw-data caching; these changes do not alter
task prompts or metrics.
