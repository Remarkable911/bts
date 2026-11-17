#!/usr/bin/env python3
"""Run the frozen LongBench evaluation suite."""

from __future__ import annotations

import sys
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUBMISSION_ROOT))

from eval.common import lm_eval_runner


SUITE = "longbench"


def main(argv: list[str] | None = None) -> int:
    """Run only the bundled LongBench group with the shared runtime."""
    return lm_eval_runner.main(argv, suite=SUITE)


if __name__ == "__main__":
    raise SystemExit(main())
