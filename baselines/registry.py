"""Lazy, auditable registry for BTS and its formal comparison baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, MutableMapping


BASELINES_ROOT = Path(__file__).resolve().parent
SUBMISSION_ROOT = BASELINES_ROOT.parent


@dataclass(frozen=True)
class Dependency:
    distribution: str
    version: str
    import_name: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    strategy: str
    module_name: str | None = None
    forward_name: str | None = None
    collector_name: str | None = None
    model_args: tuple[tuple[str, Any], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    source_files: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ("qwen3",)


SPECS: dict[str, MethodSpec] = {
    "bts": MethodSpec(
        name="bts",
        display_name="BTS",
        strategy="bts_patch",
        source_files=("bts/bts.py", "bts/config.py"),
        model_types=("qwen3", "llama"),
    ),
    "dense": MethodSpec(
        name="dense",
        display_name="Dense SDPA",
        strategy="native",
        model_args=(("attn_implementation", "sdpa"),),
        model_types=("qwen3", "llama"),
    ),
    "flashattention": MethodSpec(
        name="flashattention",
        display_name="FlashAttention-2",
        strategy="native",
        module_name="baselines.FlashAttention",
        model_args=(("attn_implementation", "flash_attention_2"),),
        dependencies=(Dependency("flash-attn", "2.8.3.post1", "flash_attn"),),
        source_files=("baselines/FlashAttention.py",),
        model_types=("qwen3", "llama"),
    ),
    "flexprefill": MethodSpec(
        name="flexprefill",
        display_name="FlexPrefill",
        strategy="forward_patch",
        module_name="baselines.FlexPrefill",
        forward_name="flexprefill_attention_forward",
        collector_name="STAT_COLLECTOR",
        environment=(
            ("FLEX_PREFILL_GAMMA", "0.95"),
            ("FLEX_PREFILL_TAU", "0.1"),
            ("FLEX_PREFILL_MIN_BUDGET", ""),
            ("FLEX_PREFILL_MAX_BUDGET", ""),
            ("GQA_INTERLEAVE", "false"),
            ("BLOCK_SIZE", "128"),
            ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ),
        dependencies=(
            Dependency("flash-attn", "2.8.3.post1", "flash_attn"),
            Dependency("einops", "0.8.2", "einops"),
        ),
        source_files=("baselines/FlexPrefill.py",),
    ),
    "minference": MethodSpec(
        name="minference",
        display_name="MInference-local (vertical=512, slash=2048)",
        strategy="forward_patch",
        module_name="baselines.Minference",
        forward_name="minference_attention_forward",
        collector_name="STAT_COLLECTOR",
        environment=(
            ("MINFERENCE_VERTICAL_SIZE", "512"),
            ("MINFERENCE_SLASH_SIZE", "2048"),
            ("MINFERENCE_ADAPTIVE_BUDGET", ""),
            ("BLOCK_SIZE", "128"),
            ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ),
        dependencies=(Dependency("minference", "0.1.6.0", "minference"),),
        source_files=(
            "baselines/Minference.py",
            "baselines/minference_index.py",
            "baselines/minference_runtime.py",
        ),
    ),
    "xattention": MethodSpec(
        name="xattention",
        display_name="XAttention",
        strategy="forward_patch",
        module_name="baselines.XAttention",
        forward_name="xattn_attention_forward",
        collector_name="STAT_COLLECTOR",
        environment=(
            ("XATTN_STRIDE", "8"),
            ("XATTN_THRESHOLD", "0.9"),
            ("BLOCK_SIZE", "128"),
        ),
        dependencies=(
            Dependency("block-sparse-attn", "0.0.2", "block_sparse_attn"),
        ),
        source_files=("baselines/XAttention.py",),
    ),
}

BASELINE_NAMES = ("flashattention", "flexprefill", "minference", "xattention")
ATTENTION_NAMES = tuple(SPECS)
ALIASES = {
    "flash-attention": "flashattention",
    "flash_attention": "flashattention",
    "flashattention2": "flashattention",
    "fa2": "flashattention",
    "flex-prefill": "flexprefill",
    "xattn": "xattention",
    "x-attention": "xattention",
}

DENSITY_DEFINITIONS = {
    "bts": (
        "final packed 128-token logical blocks divided by full causal "
        "logical-block slots; per-sample density is the arithmetic mean over "
        "all attention layers"
    ),
    "dense": "full causal attention; logical density is defined as 1.0",
    "flashattention": (
        "full causal FlashAttention-2; logical density is defined as 1.0"
    ),
    "flexprefill": (
        "selected FlexPrefill 128-token logical block slots divided by full "
        "causal logical-block slots"
    ),
    "minference": (
        "semantic 128-token vertical/slash mask density; this is not an exact "
        "executed-kernel FLOP density for MInference's 64-token kernel blocks"
    ),
    "xattention": (
        "selected XAttention 128-token logical block slots divided by full "
        "causal logical-block slots"
    ),
}


@dataclass
class PreparedAttention:
    spec: MethodSpec
    model_args: dict[str, Any]
    patched_attention_classes: int
    collector: Any
    contract: dict[str, Any]
    module: Any = None


def normalize_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in SPECS:
        expected = ", ".join(ATTENTION_NAMES)
        raise ValueError(f"unknown attention method {name!r}; expected one of: {expected}")
    return normalized


def get_spec(name: str) -> MethodSpec:
    return SPECS[normalize_name(name)]


def configure_environment(
    name: str,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set frozen knobs, rejecting inherited values that change the method."""
    spec = get_spec(name)
    target = os.environ if environ is None else environ
    configured: dict[str, str] = {}
    for key, value in spec.environment:
        if key in target and target[key] != value:
            raise RuntimeError(
                f"{spec.display_name} requires {key}={value!r}, "
                f"but the environment contains {target[key]!r}"
            )
        target[key] = value
        configured[key] = value
    return configured


def _dependency_row(dependency: Dependency) -> dict[str, Any]:
    try:
        installed = importlib.metadata.version(dependency.distribution)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    try:
        importable = importlib.util.find_spec(dependency.import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        importable = False
    return {
        **asdict(dependency),
        "installed_version": installed,
        "importable": importable,
        "version_matches": installed == dependency.version,
    }


def dependency_report(name: str, *, strict: bool = False) -> list[dict[str, Any]]:
    rows = [_dependency_row(item) for item in get_spec(name).dependencies]
    if strict:
        failed = [
            row
            for row in rows
            if not row["importable"] or not row["version_matches"]
        ]
        if failed:
            details = "; ".join(
                f"{row['distribution']} required={row['version']} "
                f"installed={row['installed_version']} importable={row['importable']}"
                for row in failed
            )
            raise RuntimeError(
                f"baseline dependency preflight failed: {details}; "
                "install eval/requirements-baselines.txt"
            )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_report(name: str) -> dict[str, Any]:
    spec = get_spec(name)
    files: dict[str, str] = {}
    for relative in spec.source_files:
        path = SUBMISSION_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"baseline source file is missing: {path}")
        files[relative] = _sha256(path)
    expected = None
    if spec.name in BASELINE_NAMES:
        upstream = json.loads((BASELINES_ROOT / "UPSTREAM.json").read_text("utf-8"))
        expected = upstream.get("methods", {}).get(spec.name, {}).get(
            "submission_source_sha256"
        )
        if expected is None:
            raise RuntimeError(
                f"{spec.display_name} has no frozen submission source hashes"
            )
    if expected is not None and files != expected:
        changed = sorted(
            relative
            for relative in set(files) | set(expected)
            if files.get(relative) != expected.get(relative)
        )
        raise RuntimeError(
            f"{spec.display_name} frozen source hash mismatch: {', '.join(changed)}"
        )
    manifest = hashlib.sha256(
        "\n".join(f"{key}  {files[key]}" for key in sorted(files)).encode("utf-8")
    ).hexdigest()
    return {
        "files": files,
        "manifest_sha256": manifest,
        "frozen_source_verified": expected is not None,
    }


def method_contract(name: str) -> dict[str, Any]:
    spec = get_spec(name)
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "strategy": spec.strategy,
        "model_types": list(spec.model_types),
        "model_args": dict(spec.model_args),
        "frozen_environment": dict(spec.environment),
        "dependencies": dependency_report(spec.name),
        "source": source_report(spec.name),
    }


def _import_and_preflight(spec: MethodSpec):
    # Import the local adapter first.  In particular, MInference deliberately
    # bypasses its package __init__, which imports optional vLLM/retrieval
    # stacks unrelated to this baseline and may fail on a different CUDA ABI.
    module = importlib.import_module(spec.module_name) if spec.module_name else None
    preflight = getattr(module, "preflight_backend", None)
    if callable(preflight):
        observed = preflight()
        if observed.get("callable") is not True:
            raise RuntimeError(f"{spec.display_name} backend is not callable")
    if spec.name == "xattention" and not callable(
        getattr(module, "block_sparse_attn_func", None)
    ):
        raise RuntimeError("XAttention block_sparse_attn_func is unavailable")
    return module


def prepare_attention(
    name: str,
    model_id: str,
    *,
    model_type: str | None = None,
    revision: str | None = None,
) -> PreparedAttention:
    """Preflight and patch one method before any model weights are loaded."""
    spec = get_spec(name)
    if model_type is None:
        from bts.utils.patch import resolve_model_type

        model_type = resolve_model_type(model_id, revision=revision)
    model_type = str(model_type).lower().replace("-", "_")
    if model_type not in spec.model_types:
        raise RuntimeError(
            f"{spec.display_name} does not support model_type={model_type!r}; "
            f"allowed model types: {', '.join(spec.model_types)}"
        )
    configured = configure_environment(spec.name)
    base_contract = method_contract(spec.name)
    dependencies = dependency_report(spec.name, strict=True)
    module = None
    collector = None
    patched = 0

    if spec.strategy == "bts_patch":
        from bts import describe, patch_model
        from bts.bts import STAT_COLLECTOR

        patched = patch_model(model_id, revision=revision)
        collector = STAT_COLLECTOR
        method_details: dict[str, Any] = describe()
    elif spec.strategy == "forward_patch":
        module = _import_and_preflight(spec)
        forward = getattr(module, spec.forward_name)
        from bts.utils.patch import apply_patch

        patched = apply_patch(
            forward,
            model_id,
            revision=revision,
            allowed_model_types=spec.model_types,
        )
        collector = getattr(module, spec.collector_name)
        method_details = {
            "forward": f"{spec.module_name}.{spec.forward_name}",
            "collector": f"{spec.module_name}.{spec.collector_name}",
        }
    else:
        module = _import_and_preflight(spec) if spec.module_name else None
        method_details = {"native_model_backend": dict(spec.model_args)}

    if spec.strategy.endswith("patch") and patched <= 0:
        raise RuntimeError(f"{spec.display_name} patched no attention classes")

    contract = {
        **base_contract,
        "dependencies": dependencies,
        "effective_environment": configured,
        "patched_attention_classes": patched,
        "method_details": method_details,
    }
    return PreparedAttention(
        spec=spec,
        model_args=dict(spec.model_args),
        patched_attention_classes=patched,
        collector=collector,
        contract=contract,
        module=module,
    )


def install_runtime_hooks(prepared: PreparedAttention) -> dict[str, Any]:
    if prepared.spec.name != "minference":
        return {"installed": False, "reason": "not_required"}
    from .minference_runtime import install_hflm_cache_lifecycle

    return install_hflm_cache_lifecycle()


def load_forward(name: str) -> tuple[Callable[..., Any], Any, dict[str, Any]]:
    """Load a patched baseline forward for module-level benchmarks."""
    spec = get_spec(name)
    if spec.strategy != "forward_patch":
        raise ValueError(f"{spec.display_name} has no standalone patched forward")
    configure_environment(spec.name)
    contract = method_contract(spec.name)
    dependency_report(spec.name, strict=True)
    module = _import_and_preflight(spec)
    return (
        getattr(module, spec.forward_name),
        getattr(module, spec.collector_name),
        contract,
    )
