#!/usr/bin/env python3
"""Run one method on selected tasks from the frozen official RULER suite."""

from __future__ import annotations

import sys
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUBMISSION_ROOT))

from eval.common import lm_eval_runner


SUITE = "ruler"


def main(argv: list[str] | None = None) -> int:
    """Run selected frozen RULER tasks across their 32K/64K/128K samples."""
    return lm_eval_runner.main(argv, suite=SUITE)


if __name__ == "__main__":
    raise SystemExit(main())
