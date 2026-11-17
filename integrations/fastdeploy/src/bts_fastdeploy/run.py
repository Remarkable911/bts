"""Offline generation with enforced compatibility settings and worker evidence."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid

from . import METHOD_DISPLAY_NAME
from .config import MAX_SUPPORTED_SEQS


def _required_fastdeploy_environment():
    """Return process settings required before FastDeploy 2.5 is imported."""
    existing = [p for p in os.environ.get("FD_PLUGINS", "").split(",") if p]
    return {
        "BTS_FASTDEPLOY_ENABLE": "1",
        "FD_ATTENTION_BACKEND": "FLASH_ATTN",
        "FLAGS_flash_attn_version": "2",
        # FastDeploy 2.5 otherwise turns chunked prefill back on automatically
        # for CUDA mixed-mode engines, even when EngineArgs receives False.
        "FD_DISABLE_CHUNKED_PREFILL": "1",
        "FD_PLUGINS": ",".join(dict.fromkeys([*existing, "bts"])),
    }


def _apply_fastdeploy_environment(values):
    """Apply and verify settings before any FastDeploy import or worker spawn."""
    os.environ.update(values)
    mismatched = {
        name: os.environ.get(name)
        for name, value in values.items()
        if os.environ.get(name) != value
    }
    if mismatched:
        raise RuntimeError(f"Could not enforce FastDeploy process environment: {mismatched}")


def _prefill_layer_calls(audits):
    """Aggregate a validated per-layer prefill snapshot across workers."""
    result = {}
    for audit in audits:
        layers = audit.get("layers")
        if not isinstance(layers, dict):
            raise RuntimeError("Worker audit has no per-layer BTS counts")
        worker_total = 0
        for raw_layer, count in layers.items():
            try:
                layer = int(raw_layer)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Worker audit has invalid layer id: {raw_layer!r}"
                ) from error
            if str(layer) != str(raw_layer) or layer < 0:
                raise RuntimeError(f"Worker audit has invalid layer id: {raw_layer!r}")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise RuntimeError(
                    f"Worker audit has invalid call count for layer {layer}: {count!r}"
                )
            result[layer] = result.get(layer, 0) + count
            worker_total += count
        if audit.get("prefill_calls") != worker_total:
            raise RuntimeError(
                "Worker audit prefill total does not match its per-layer counts"
            )
    return result


def _generation_layer_delta(before, after, num_hidden_layers):
    """Require this generation to execute BTS in every and only model layer."""
    if (
        not isinstance(num_hidden_layers, int)
        or isinstance(num_hidden_layers, bool)
        or num_hidden_layers <= 0
    ):
        raise RuntimeError("Model num_hidden_layers must be a positive integer")
    layers = set(before) | set(after)
    delta = {layer: after.get(layer, 0) - before.get(layer, 0) for layer in layers}
    if any(count < 0 for count in delta.values()):
        raise RuntimeError("Worker audit per-layer counts decreased during generation")
    executed = {layer for layer, count in delta.items() if count > 0}
    expected = set(range(num_hidden_layers))
    if executed != expected:
        missing = sorted(expected - executed)
        unexpected = sorted(executed - expected)
        raise RuntimeError(
            "Generation did not prove exact BTS layer coverage: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {str(layer): delta[layer] for layer in sorted(expected)}


def _validate_worker_provenance(audits, launcher_provenance):
    """Reject audits produced by packages other than the active launcher package."""
    from .environment import validate_package_provenance

    if not audits:
        raise RuntimeError("Generation produced no BTS worker audit")
    expected = {
        name: launcher_provenance[name]
        for name in ("bts_file", "bts_fastdeploy_file")
    }
    for audit in audits:
        actual = validate_package_provenance(audit.get("package_provenance"))
        for name, path in expected.items():
            if actual[name] != path:
                raise RuntimeError(
                    "Worker package provenance differs from launcher: "
                    f"{name}={actual[name]!r}, launcher={path!r}"
                )


def _serialize_generations(outputs):
    """Normalize FastDeploy request outputs into JSON-serializable records.

    FastDeploy 2.5 exposes ``RequestOutput.outputs`` as one
    ``CompletionOutput`` object.  Some adjacent API versions expose a sequence,
    so handle both shapes explicitly without attempting to iterate the 2.5
    object (whose mapping-style ``__getitem__`` is not a sequence protocol).
    """
    generations = []
    for request_output in outputs:
        completions = request_output.outputs
        if completions is None:
            continue
        if not isinstance(completions, (list, tuple)):
            completions = (completions,)
        for completion in completions:
            token_ids = completion.token_ids
            generations.append(
                {
                    "text": completion.text,
                    "token_ids": [] if token_ids is None else list(token_ids),
                }
            )
    return generations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3 text model directory")
    parser.add_argument("--config", help="Existing runtime_config.json or a partial runtime object")
    parser.add_argument("--prompt", default="Explain sparse attention in one sentence.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Repeat the selected prompt this many times in one offline generate call",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Resolve config without importing Paddle/FastDeploy or using GPU")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.max_model_len <= args.max_new_tokens:
        parser.error("Require 0 < max-new-tokens < max-model-len")
    if not 1 <= args.max_num_seqs <= MAX_SUPPORTED_SEQS:
        parser.error(f"max-num-seqs must be in [1, {MAX_SUPPORTED_SEQS}]")
    if not 1 <= args.num_prompts <= args.max_num_seqs:
        parser.error("num-prompts must be in [1, max-num-seqs]")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")

    from .config import config_digest, load_config
    from .environment import package_provenance
    config = load_config(args.config)
    launcher_provenance = package_provenance()
    model = Path(args.model).resolve()
    model_info = json.loads((model / "config.json").read_text())
    if model_info.get("model_type") != "qwen3":
        parser.error("Initial FastDeploy support is limited to Qwen3 text models")
    engine_options = dict(
        model=str(model), tensor_parallel_size=1, data_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=False, enable_prefix_caching=False,
        graph_optimization_config={"use_cudagraph": False, "graph_opt_level": 0},
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
    )
    fastdeploy_environment = _required_fastdeploy_environment()
    report = dict(status="dry_run" if args.dry_run else "starting",
                  method_display_name=METHOD_DISPLAY_NAME,
                  runtime_config=asdict(config), config_sha256=config_digest(config),
                  engine_options=engine_options,
                  fastdeploy_environment=fastdeploy_environment,
                  launcher_package_provenance=launcher_provenance)
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output.resolve() / "runtime_config.json"
    runtime_path.write_text(json.dumps(asdict(config), indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    audit_dir = args.output.resolve() / ("workers-" + uuid.uuid4().hex)
    # These settings reach every worker before the FastDeploy import/spawn.
    _apply_fastdeploy_environment(fastdeploy_environment)
    os.environ["BTS_FASTDEPLOY_CONFIG"] = str(runtime_path)
    os.environ["BTS_FASTDEPLOY_AUDIT_DIR"] = str(audit_dir)
    os.environ["FD_LOG_DIR"] = str(args.output.resolve() / "fastdeploy_logs")
    Path(os.environ["FD_LOG_DIR"]).mkdir(parents=True,exist_ok=True)
    if os.environ.get("FD_DISABLE_CHUNKED_PREFILL") != "1":
        raise RuntimeError("FastDeploy chunked prefill guard was not preserved")

    from .environment import prepare_workers
    if prepare_workers() != launcher_provenance:
        raise RuntimeError("Launcher package provenance changed before worker spawn")
    from .audit import read_audits
    from fastdeploy import LLM, SamplingParams
    from .plugin import register
    register()
    llm = LLM(**engine_options)
    try:
        before_audits = read_audits(audit_dir)
        before_layers = _prefill_layer_calls(before_audits)
        prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
        started = time.perf_counter()
        prompts = [prompt] * args.num_prompts
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=args.max_new_tokens),
        )
        elapsed = time.perf_counter() - started
        audits = read_audits(audit_dir)
        after_layers = _prefill_layer_calls(audits)
        generation_layer_calls = _generation_layer_delta(
            before_layers, after_layers, model_info.get("num_hidden_layers")
        )
        _validate_worker_provenance(audits, launcher_provenance)
        delta = sum(generation_layer_calls.values())
        generations = _serialize_generations(outputs)
    finally:
        # Version-scoped FastDeploy 2.5 shutdown; do not leave validation workers
        # reserving GPU memory when the generation process finishes or fails.
        llm.llm_engine._exit_sub_services()
    if len(generations) != args.num_prompts or not all(
        g["token_ids"] for g in generations
    ):
        raise RuntimeError(
            "FastDeploy returned the wrong number of completions or no generated tokens"
        )
    report.update(status="passed", elapsed_seconds=elapsed, generation_prefill_calls=delta,
                  generation_layer_calls=generation_layer_calls,
                  workers=audits, generations=generations)
    (args.output / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
