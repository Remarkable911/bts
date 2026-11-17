#!/usr/bin/env python3
"""Render the final four-arm benchmark result as a compact terminal table."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from eval.ThreeWay import run as analysis


ARM_LABELS = {
    "fd_dense": "FastDeploy Dense (FA2)",
    "fd_bts": "FastDeploy BTS",
    "vllm_dense": "vLLM Dense (FA2)",
    "vllm_bts": "vLLM BTS",
}

COMPARISON_ROWS = (
    ("FastDeploy BTS / FastDeploy Dense", "fd_bts_over_fd_dense"),
    ("vLLM BTS / vLLM Dense", "vllm_bts_over_vllm_dense"),
    ("FastDeploy BTS / vLLM Dense", "fd_bts_over_vllm_dense"),
)


class SummaryError(RuntimeError):
    """The persisted output cannot support a trustworthy terminal summary."""


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"{path} must contain a JSON object")
    return value


def _arm_median_metrics(
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    throughput_medians: dict[str, float] = {}
    time_medians: dict[str, float] = {}
    for arm in analysis.ARMS:
        path = output_dir / f"{arm}.json"
        report = analysis.load_arm_report(path, expected_arm=arm)
        measured_seconds = [
            float(row["measured_seconds"]) for row in report["restarts"]
        ]
        throughputs = [
            int(row["total_input_tokens"]) / float(row["measured_seconds"])
            for row in report["restarts"]
        ]
        time_medians[arm] = statistics.median(measured_seconds)
        throughput_medians[arm] = statistics.median(throughputs)
    return throughput_medians, time_medians


def _controller_wall_seconds(output_dir: Path) -> float | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _load_json_object(manifest_path)
    started_at = manifest.get("started_at")
    locks = manifest.get("formal_run_locks")
    finished_at = locks.get("released_at") if isinstance(locks, dict) else None
    if finished_at is None:
        finished_at = manifest.get("completed_at")
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        return None
    try:
        seconds = (
            datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (TypeError, ValueError) as error:
        raise SummaryError(
            f"invalid controller timestamps in {manifest_path}: {error}"
        ) from error
    if not math.isfinite(seconds) or seconds < 0:
        raise SummaryError(f"invalid controller wall time in {manifest_path}")
    return seconds


def _format_duration(seconds: float) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    fields = []
    if days >= 1:
        fields.append(f"{int(days)}d")
    if days >= 1 or hours >= 1:
        fields.append(f"{int(hours)}h")
    if days >= 1 or hours >= 1 or minutes >= 1:
        fields.append(f"{int(minutes)}m")
    fields.append(f"{remaining_seconds:.1f}s")
    return " ".join(fields)


def _validated_comparisons(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    if result.get("schema_version") != analysis.ANALYSIS_SCHEMA_VERSION:
        raise SummaryError(
            "unexpected result schema: "
            f"{result.get('schema_version')!r}; expected "
            f"{analysis.ANALYSIS_SCHEMA_VERSION!r}"
        )
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, dict):
        raise SummaryError("result.comparisons must be an object")

    passed = []
    for _, key in COMPARISON_ROWS:
        row = comparisons.get(key)
        if not isinstance(row, dict):
            raise SummaryError(f"result is missing comparison {key!r}")
        for field in (
            "median_throughput_speedup",
            "bootstrap_one_sided_95pct_lower_bound",
            "target_speedup",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise SummaryError(f"{key}.{field} must be finite and numeric")
        if row["target_speedup"] != analysis.FORMAL_TARGET_SPEEDUP:
            raise SummaryError(
                f"{key}.target_speedup={row['target_speedup']!r}; expected "
                f"{analysis.FORMAL_TARGET_SPEEDUP!r}"
            )
        if not isinstance(row.get("target_passed"), bool):
            raise SummaryError(f"{key}.target_passed must be boolean")
        expected_pass = (
            row["bootstrap_one_sided_95pct_lower_bound"]
            >= row["target_speedup"]
        )
        if row["target_passed"] != expected_pass:
            raise SummaryError(
                f"{key}.target_passed conflicts with its 95% lower bound"
            )
        passed.append(row["target_passed"])

    expected_status = "PASS" if all(passed) else "FAIL"
    status = result.get("overall_status")
    if status != expected_status:
        raise SummaryError(
            f"overall_status={status!r} conflicts with comparison results; "
            f"expected {expected_status!r}"
        )
    return comparisons, expected_status


def render_result_summary(
    *,
    result: Mapping[str, Any],
    arm_medians: Mapping[str, float],
    arm_median_seconds: Mapping[str, float],
    output_dir: Path,
    controller_wall_seconds: float | None = None,
) -> str:
    comparisons, _status = _validated_comparisons(result)
    missing_arms = [arm for arm in analysis.ARMS if arm not in arm_medians]
    if missing_arms:
        raise SummaryError(f"missing arm throughput medians: {missing_arms}")
    missing_times = [arm for arm in analysis.ARMS if arm not in arm_median_seconds]
    if missing_times:
        raise SummaryError(f"missing arm time medians: {missing_times}")

    separator = "=" * 86
    lines = [
        separator,
        "BTS 128K THROUGHPUT RESULT SUMMARY",
        separator,
        "",
        "Four-arm medians (5 fresh-process restarts)",
        (
            f"{'Arm':<34} {'Measured time':>17} "
            f"{'Input throughput':>24}"
        ),
        f"{'-' * 34} {'-' * 17} {'-' * 24}",
    ]
    for arm in analysis.ARMS:
        lines.append(
            f"{ARM_LABELS[arm]:<34} "
            f"{arm_median_seconds[arm]:>13,.3f} s "
            f"{arm_medians[arm]:>17,.3f} tok/s"
        )

    lines.extend(
        [
            "",
            "Speedup comparisons",
            (
                f"{'Comparison':<39} {'Median':>9} {'95% lower':>11} "
                f"{'Target':>9}"
            ),
            (
                f"{'-' * 39} {'-' * 9} {'-' * 11} {'-' * 9}"
            ),
        ]
    )
    for label, key in COMPARISON_ROWS:
        row = comparisons[key]
        lines.append(
            f"{label:<39} "
            f"{row['median_throughput_speedup']:>8.3f}x "
            f"{row['bootstrap_one_sided_95pct_lower_bound']:>10.3f}x "
            f"{row['target_speedup']:>8.3f}x"
        )

    result_path = output_dir / "three-comparison.json"
    if controller_wall_seconds is not None:
        lines.extend(
            [
                "",
                "Controller wall time (GPU wait and cleanup included): "
                f"{_format_duration(controller_wall_seconds)}",
            ]
        )
    lines.extend(
        [
            "",
            f"Output directory: {output_dir}",
            f"Result JSON: {result_path}",
            separator,
        ]
    )
    return "\n".join(lines)


def _single_line(value: Any, *, limit: int = 500) -> str:
    rendered = " ".join(str(value).split())
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def render_failed_run_summary(
    *, output_dir: Path, controller_exit_code: int
) -> str:
    manifest_path = output_dir / "manifest.json"
    result_path = output_dir / "three-comparison.json"
    title = (
        "BTS 128K CONTROLLER ERROR AFTER FINAL ANALYSIS"
        if result_path.is_file()
        else "BTS 128K EXPERIMENT DID NOT PRODUCE A FINAL RESULT"
    )
    lines = [
        "=" * 86,
        title,
        "=" * 86,
        f"Controller exit code: {controller_exit_code}",
        f"Output directory: {output_dir}",
    ]
    if manifest_path.is_file():
        manifest = _load_json_object(manifest_path)
        lines.append(f"Manifest status: {_single_line(manifest.get('status', 'unknown'))}")
        completed = manifest.get("completed_arms")
        if isinstance(completed, list):
            lines.append(f"Completed arm/restart workers: {len(completed)} / 20")
        if manifest.get("failure_type") is not None:
            lines.append(f"Failure type: {_single_line(manifest['failure_type'])}")
        if manifest.get("failure") is not None:
            lines.append(f"Failure: {_single_line(manifest['failure'])}")
        lines.append(f"Manifest: {manifest_path}")
    else:
        lines.append(f"Manifest not found: {manifest_path}")
    if result_path.is_file():
        result_line = (
            "Candidate result exists but is not accepted because the controller "
            f"failed: {result_path}"
        )
    else:
        result_line = f"Final result not found: {result_path}"
    lines.extend(
        [
            result_line,
            "=" * 86,
        ]
    )
    return "\n".join(lines)


def render_dry_run_summary() -> str:
    separator = "=" * 86
    return "\n".join(
        [
            separator,
            "BTS 128K DRY RUN COMPLETED",
            "Static execution plan validated; no GPU benchmark was run and no "
            "measurement result was generated.",
            separator,
        ]
    )


def render_failed_dry_run_summary(*, controller_exit_code: int) -> str:
    separator = "=" * 86
    return "\n".join(
        [
            separator,
            "BTS 128K DRY RUN DID NOT COMPLETE",
            f"Controller exit code: {controller_exit_code}",
            "No GPU benchmark was run and no measurement result was generated.",
            separator,
        ]
    )


def render_run_summary(
    *, output_dir: Path, controller_exit_code: int, dry_run: bool = False
) -> str:
    """Render the final terminal block after the controller has cleaned up."""

    if dry_run:
        if controller_exit_code == 0:
            return render_dry_run_summary()
        return render_failed_dry_run_summary(
            controller_exit_code=controller_exit_code
        )
    absolute_output = output_dir.absolute()
    result_path = absolute_output / "three-comparison.json"
    if controller_exit_code not in (0, 2):
        return render_failed_run_summary(
            output_dir=absolute_output,
            controller_exit_code=controller_exit_code,
        )
    if not result_path.is_file():
        raise SummaryError(
            f"controller exited {controller_exit_code} but final result is missing: "
            f"{result_path}"
        )
    result = _load_json_object(result_path)
    recomputed = analysis.analyze_paths(
        {arm: absolute_output / f"{arm}.json" for arm in analysis.ARMS}
    )
    if result != recomputed:
        raise SummaryError(
            f"persisted result does not match recomputation from arm reports: "
            f"{result_path}"
        )
    expected_exit_code = 0 if result.get("overall_status") == "PASS" else 2
    if controller_exit_code != expected_exit_code:
        raise SummaryError(
            f"controller exit code {controller_exit_code} conflicts with "
            f"overall_status={result.get('overall_status')!r}"
        )
    arm_medians, arm_median_seconds = _arm_median_metrics(absolute_output)
    return render_result_summary(
        result=result,
        arm_medians=arm_medians,
        arm_median_seconds=arm_median_seconds,
        output_dir=absolute_output,
        controller_wall_seconds=_controller_wall_seconds(absolute_output),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the compact terminal summary for a ThreeWay run"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--controller-exit-code", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(
            render_run_summary(
                output_dir=args.output_dir,
                controller_exit_code=args.controller_exit_code,
                dry_run=args.dry_run,
            )
        )
        return 0
    except (SummaryError, analysis.ProtocolError) as error:
        print(f"terminal summary error: {error}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
