#!/usr/bin/env python3
"""Compatibility entry point for one frozen official RULER evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
WORKER = HERE / "run.py"
sys.path.insert(0, str(SUBMISSION_ROOT))

from baselines.registry import normalize_name
from eval.common import lm_eval_runner


DEFAULT_METHODS = ("bts",)
DEFAULT_LENGTHS = lm_eval_runner.OFFICIAL_RULER_LENGTHS
DEFAULT_RULER_SUITE = lm_eval_runner.DEFAULT_RULER_SUITE
SAMPLES_PER_TASK = lm_eval_runner.OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH
RULER_TASKS = 13
PROTOCOL = "official_fixed_ruler_single_runner_v1"


@dataclass(frozen=True)
class ComparisonCase:
    """Legacy plan shape retained for callers that inspect one runner command."""

    method: str
    length: int
    output: Path


def _methods(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("--methods must contain one method")
    try:
        normalized = tuple(normalize_name(item) for item in requested)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if len(set(normalized)) != len(normalized):
        raise argparse.ArgumentTypeError(
            "--methods contains duplicates after alias normalization"
        )
    return normalized


def _lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--lengths must be comma-separated integers"
        ) from error
    if not lengths:
        raise argparse.ArgumentTypeError("--lengths must contain at least one length")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("--lengths must not contain duplicates")
    unsupported = [length for length in lengths if length not in DEFAULT_LENGTHS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            "formal RULER lengths must be 32768, 65536, or 131072"
        )
    return lengths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one method once on the frozen official_fixed_ruler group; "
            "the run contains 32K, 64K, and 128K samples."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local Qwen3-8B YaRN4 path or Hugging Face model ID",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Pinned model revision; required for a real remote-model run",
    )
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output",
        required=True,
        type=Path,
        help="New or empty output directory for the single lm-eval run",
    )
    parser.add_argument(
        "--methods",
        "--attention",
        dest="methods",
        type=_methods,
        default=DEFAULT_METHODS,
        help="One attention method (legacy --methods name remains accepted)",
    )
    parser.add_argument(
        "--lengths",
        type=_lengths,
        default=DEFAULT_LENGTHS,
        help=(
            "Compatibility option; the fixed suite always requires the complete "
            "32768,65536,131072 population"
        ),
    )
    parser.add_argument(
        "--ruler-suite",
        type=Path,
        default=DEFAULT_RULER_SUITE,
        help="Path containing manifest.json and tasks/official_fixed_ruler.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def build_plan(
    output_root: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
) -> tuple[ComparisonCase, ...]:
    """Return the single compatibility case for the complete fixed suite."""
    if len(methods) != 1:
        raise ValueError("official RULER accepts exactly one method per invocation")
    if tuple(lengths) != DEFAULT_LENGTHS:
        raise ValueError(
            "official RULER must run 32768,65536,131072 together in one invocation"
        )
    return (
        ComparisonCase(
            method=normalize_name(methods[0]),
            length=131072,
            output=output_root.expanduser().resolve(),
        ),
    )


def worker_command(
    case: ComparisonCase,
    *,
    model: str,
    revision: str,
    device: str,
    trust_remote_code: bool,
    ruler_suite: Path = DEFAULT_RULER_SUITE,
) -> list[str]:
    """Build the one runner command without executing a worker matrix."""
    command = [
        sys.executable,
        str(WORKER),
        "--model",
        model,
        "--revision",
        revision,
        "--attention",
        case.method,
        "--max-length",
        "131072",
        "--ruler-suite",
        str(ruler_suite.expanduser().absolute()),
        "--tasks",
        lm_eval_runner.OFFICIAL_RULER_GROUP,
        "--ruler-samples-per-task",
        str(SAMPLES_PER_TASK),
        "--batch-size",
        "1",
        "--device",
        device,
        "--dtype",
        "bfloat16",
        "--random-seed",
        "0",
        "--numpy-random-seed",
        "1234",
        "--torch-random-seed",
        "1234",
        "--fewshot-random-seed",
        "1234",
        "--apply-chat-template",
        "--confirm-run-unsafe-code",
        "--formal",
        "--output",
        str(case.output),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not str(args.device).startswith("cuda"):
        parser.error("formal RULER requires a CUDA device")
    try:
        (case,) = build_plan(args.output, args.methods, args.lengths)
    except ValueError as error:
        parser.error(str(error))

    command = worker_command(
        case,
        model=args.model,
        revision=args.revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        ruler_suite=args.ruler_suite,
    )
    runner_argv = command[2:]
    # Deliberately call the shared runner once in this process. The former
    # methods-by-length subprocess matrix was not the frozen official protocol.
    return lm_eval_runner.main(runner_argv, suite="ruler")


if __name__ == "__main__":
    raise SystemExit(main())
