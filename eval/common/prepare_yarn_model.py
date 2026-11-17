#!/usr/bin/env python3
"""Create a reproducible Qwen3 YaRN model overlay for 128K evaluation.

The overlay contains a generated ``config.json`` and links to the immutable
files in a local model snapshot.  It never copies or modifies the source
checkpoint.  A manifest records the exact source, revision, configuration
hashes, and every linked file so an evaluator can audit the overlay before a
long run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


DEFAULT_TARGET_LENGTH = 131072
DEFAULT_ORIGINAL_LENGTH = 32768


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_yarn_config(
    source: dict[str, Any],
    *,
    target_length: int = DEFAULT_TARGET_LENGTH,
    original_length: int = DEFAULT_ORIGINAL_LENGTH,
) -> dict[str, Any]:
    """Return the frozen 128K YaRN configuration without mutating ``source``."""
    if source.get("model_type") != "qwen3":
        raise ValueError("the 128K overlay supports Qwen3 text checkpoints only")
    if int(source.get("head_dim", 0)) != 128:
        raise ValueError("the frozen BTS submission requires head_dim=128")
    if target_length <= 0 or original_length <= 0:
        raise ValueError("target and original lengths must be positive")
    if target_length % original_length:
        raise ValueError("target length must be an integer multiple of original length")
    if target_length > 131072:
        raise ValueError("the frozen BTS submission is validated only through 131072 tokens")

    result = dict(source)
    result["max_position_embeddings"] = target_length
    result["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": float(target_length / original_length),
        "original_max_position_embeddings": original_length,
    }
    return result


def resolve_snapshot(model: str, revision: str | None, local_files_only: bool) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()


def resolved_snapshot_revision(source_dir: Path) -> str | None:
    """Recover the immutable Hub commit from a standard snapshot path."""
    candidate = source_dir.resolve().name
    if len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate):
        return candidate
    return None


def validate_weight_index(source_dir: Path) -> list[str]:
    """Return all referenced safetensor shards, rejecting incomplete snapshots."""
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"model snapshot has no weight index: {index_path}")
    index = load_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"weight index has no non-empty weight_map: {index_path}")
    shard_values = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in shard_values):
        raise ValueError(f"weight index contains an invalid shard name: {index_path}")
    shards = sorted(set(shard_values))
    missing = [name for name in shards if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "model snapshot is missing indexed shard(s): " + ", ".join(missing)
        )
    return shards


def _install_file(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:  # Defensive guard for direct callers; argparse validates CLI input.
        raise ValueError(f"unsupported link mode: {mode}")


def create_overlay(
    source_dir: Path,
    output_dir: Path,
    *,
    model: str,
    revision: str | None,
    target_length: int,
    original_length: int,
    link_mode: str,
) -> dict[str, Any]:
    """Create one new overlay atomically and return its manifest."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    source_config_path = source_dir / "config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"model snapshot has no config.json: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")

    source_config_bytes = source_config_path.read_bytes()
    indexed_shards = validate_weight_index(source_dir)
    generated_config = build_yarn_config(
        json.loads(source_config_bytes),
        target_length=target_length,
        original_length=original_length,
    )
    generated_config_bytes = canonical_json_bytes(generated_config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    installed: list[dict[str, Any]] = []
    try:
        (temporary / "config.json").write_bytes(generated_config_bytes)
        for source in sorted(source_dir.iterdir(), key=lambda item: item.name):
            if source.name == "config.json" or not source.is_file():
                continue
            destination = temporary / source.name
            _install_file(source, destination, link_mode)
            installed.append(
                {
                    "name": source.name,
                    "source": str(source.resolve()),
                    "size": source.stat().st_size,
                    "mode": link_mode,
                }
            )

        required = ("model.safetensors.index.json", "tokenizer_config.json")
        missing = [name for name in required if not (temporary / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "model snapshot is incomplete; missing " + ", ".join(missing)
            )

        manifest = {
            "schema_version": 1,
            "protocol": "qwen3_yarn_overlay_for_bts_128k_v1",
            "model": model,
            "requested_revision": revision,
            "resolved_revision": resolved_snapshot_revision(source_dir),
            "source_snapshot": str(source_dir),
            "source_config_sha256": sha256_bytes(source_config_bytes),
            "generated_config_sha256": sha256_bytes(generated_config_bytes),
            "target_length": target_length,
            "original_max_position_embeddings": original_length,
            "rope_scaling": generated_config["rope_scaling"],
            "indexed_weight_shards": indexed_shards,
            "files": installed,
        }
        (temporary / "bts_yarn_manifest.json").write_bytes(
            canonical_json_bytes(manifest)
        )
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="Local Qwen3 snapshot or HF model ID")
    result.add_argument(
        "--revision",
        help="Pinned Hugging Face model revision; strongly recommended for remote IDs",
    )
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--target-length", type=int, default=DEFAULT_TARGET_LENGTH)
    result.add_argument("--original-length", type=int, default=DEFAULT_ORIGINAL_LENGTH)
    result.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How checkpoint files are placed in the overlay",
    )
    result.add_argument("--local-files-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if not Path(args.model).expanduser().is_dir() and args.revision in (None, "main"):
        argument_parser.error(
            "remote model IDs require an immutable commit in --revision"
        )
    source = resolve_snapshot(args.model, args.revision, args.local_files_only)
    manifest = create_overlay(
        source,
        args.output,
        model=args.model,
        revision=args.revision,
        target_length=args.target_length,
        original_length=args.original_length,
        link_mode=args.link_mode,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
