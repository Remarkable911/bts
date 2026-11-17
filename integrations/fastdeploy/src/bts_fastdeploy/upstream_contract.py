"""Fail-closed provenance for the approved FastDeploy/Paddle runtime.

The throughput claim compares BTS with an unmodified FastDeploy dense path.
Package versions alone do not prove that property: an in-place edited
``site-packages`` tree can retain the same distribution metadata.  This module
therefore hashes the actual files used by the controller, launcher and GPU
executor and checks them against a submitted allow-list.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


LOCK_PATH = Path(__file__).with_name("approved_upstream_sources.json")

RUNTIME_MODULES = {
    "fastdeploy.engine.args_utils": (
        "fastdeploy-gpu",
        "fastdeploy/engine/args_utils.py",
    ),
    "fastdeploy.model_executor.layers.attention.attention": (
        "fastdeploy-gpu",
        "fastdeploy/model_executor/layers/attention/attention.py",
    ),
    "fastdeploy.model_executor.layers.attention.attention_selecter": (
        "fastdeploy-gpu",
        "fastdeploy/model_executor/layers/attention/attention_selecter.py",
    ),
    "fastdeploy.model_executor.layers.attention.flash_attn_backend": (
        "fastdeploy-gpu",
        "fastdeploy/model_executor/layers/attention/flash_attn_backend.py",
    ),
    "fastdeploy.model_executor.models.paddleformers.base": (
        "fastdeploy-gpu",
        "fastdeploy/model_executor/models/paddleformers/base.py",
    ),
    "paddleformers.nn.attention.interface": (
        "paddleformers",
        "paddleformers/nn/attention/interface.py",
    ),
    "paddleformers.transformers.modeling_rope_utils": (
        "paddleformers",
        "paddleformers/transformers/modeling_rope_utils.py",
    ),
    "paddleformers.transformers.qwen3.configuration": (
        "paddleformers",
        "paddleformers/transformers/qwen3/configuration.py",
    ),
    "paddleformers.transformers.qwen3.modeling": (
        "paddleformers",
        "paddleformers/transformers/qwen3/modeling.py",
    ),
    "paddle.nn.functional.flash_attention": (
        "paddlepaddle-gpu",
        "paddle/nn/functional/flash_attention.py",
    ),
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _named_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"approved upstream contract requires one class {name!r}")
    return matches[0]


def _class_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"approved upstream contract requires one {cls.name}.{name} method"
        )
    return matches[0]


def _validate_fastdeploy_semantics(files: dict[str, dict[str, str]]) -> list[str]:
    backend_key = "fastdeploy/model_executor/layers/attention/flash_attn_backend.py"
    backend_path = Path(files[backend_key]["path"])
    backend_tree = ast.parse(backend_path.read_text(encoding="utf-8"))
    backend_cls = _named_class(backend_tree, "FlashAttentionBackend")
    forward = _class_method(backend_cls, "forward_mixed")
    expected_parameters = (
        "self",
        "q",
        "k",
        "v",
        "qkv",
        "compressed_kv",
        "k_pe",
        "layer",
        "forward_meta",
    )
    actual_parameters = tuple(argument.arg for argument in forward.args.args)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            "FastDeploy FlashAttentionBackend.forward_mixed signature changed: "
            f"{actual_parameters}"
        )
    named_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    call_names = [node.func.id for node in named_calls]
    required_calls = (
        "gqa_rope_write_cache",
        "flash_attn_func",
        "append_attention",
        "merge_prefill_decode_output",
    )
    for name in required_calls:
        if call_names.count(name) != 1:
            raise RuntimeError(
                f"FastDeploy forward_mixed requires exactly one {name} call; "
                f"found {call_names.count(name)}"
            )
    flash_call = next(node for node in named_calls if node.func.id == "flash_attn_func")
    parents = {
        child: parent
        for parent in ast.walk(forward)
        for child in ast.iter_child_nodes(parent)
    }
    flash_parent = parents.get(flash_call)
    if not (
        isinstance(flash_parent, ast.Subscript)
        and isinstance(flash_parent.slice, ast.Constant)
        and flash_parent.slice.value == 0
    ):
        raise RuntimeError("FastDeploy no longer consumes only attention output element 0")

    args_key = "fastdeploy/engine/args_utils.py"
    args_tree = ast.parse(Path(files[args_key]["path"]).read_text(encoding="utf-8"))
    args_cls = _named_class(args_tree, "EngineArgs")
    fields = {
        node.target.id
        for node in args_cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    required_fields = {
        "model",
        "tensor_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "graph_optimization_config",
        "gpu_memory_utilization",
        "disable_custom_all_reduce",
    }
    if missing := required_fields - fields:
        raise RuntimeError(f"FastDeploy EngineArgs fields changed: missing={sorted(missing)}")
    create_engine = _class_method(args_cls, "create_engine_config")
    disable_reads = [
        node
        for node in ast.walk(create_engine)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "envs"
        and node.attr == "FD_DISABLE_CHUNKED_PREFILL"
    ]
    if len(disable_reads) != 1:
        raise RuntimeError(
            "FastDeploy chunked-prefill opt-out contract changed: "
            f"found {len(disable_reads)} environment reads"
        )
    disables_chunking = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "enable_chunked_prefill"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(create_engine)
    )
    if not disables_chunking:
        raise RuntimeError("FastDeploy no longer honors the chunked-prefill opt-out")

    required_fragments = {
        "fastdeploy/model_executor/models/paddleformers/base.py": (
            'ALL_ATTENTION_FUNCTIONS._global_mapping["fastdeploy_append"]',
            'self.paddleformers_config._attn_implementation = "fastdeploy_append"',
            "forward_meta.rope_already_applied = True",
            "self_attn.forward(qkv=qkv, forward_meta=forward_meta)",
        ),
        "fastdeploy/model_executor/layers/attention/attention.py": (
            "forward_meta.attn_backend.forward",
        ),
        "fastdeploy/model_executor/layers/attention/attention_selecter.py": (
            "envs.FD_ATTENTION_BACKEND",
            "current_platform.get_attention_backend_cls",
        ),
        "paddleformers/transformers/qwen3/modeling.py": (
            "apply_rotary_pos_emb",
            "ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]",
            "self.q_norm",
            "self.k_norm",
            "ROPE_INIT_FUNCTIONS[self.rope_type]",
        ),
        "paddleformers/transformers/qwen3/configuration.py": (
            "standardize_rope_params",
            "rope_config_validation",
        ),
        "paddleformers/transformers/modeling_rope_utils.py": (
            '"yarn": _compute_yarn_parameters',
        ),
        "paddleformers/nn/attention/interface.py": (
            "ALL_ATTENTION_FUNCTIONS",
            "_global_mapping",
        ),
        "paddle/nn/functional/flash_attention.py": (
            "flash_attention_v3_varlen",
            "flash_attn_unpadded",
        ),
    }
    for relative_path, fragments in required_fragments.items():
        text = Path(files[relative_path]["path"]).read_text(encoding="utf-8")
        ast.parse(text)
        missing = [fragment for fragment in fragments if fragment not in text]
        if missing:
            raise RuntimeError(
                f"approved upstream semantics changed in {relative_path}: "
                f"missing={missing}"
            )
    return [
        "fastdeploy_native_flashattention_forward_mixed_v2_5",
        "fastdeploy_chunked_prefill_disable_guard_v2_5",
        "fastdeploy_paddleformers_attention_bridge_v2_5",
        "paddleformers_qwen3_rope_then_fastdeploy_attention_v1_1_1",
        "paddleformers_yarn4_configuration_v1_1_1",
        "paddle_native_flashattention_wrapper_v3_3_1",
    ]


def dependency_source_contract() -> dict[str, Any]:
    """Hash actual installed files and require the approved runtime exactly."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    packages: dict[str, Any] = {}
    all_files: dict[str, dict[str, str]] = {}
    for distribution_name, expected in lock["packages"].items():
        distribution = importlib.metadata.distribution(distribution_name)
        if distribution.version != expected["version"]:
            raise RuntimeError(
                f"approved {distribution_name} version is {expected['version']}; "
                f"found {distribution.version}"
            )
        package_files = {}
        for relative_path, approved_sha256 in expected["files"].items():
            path = Path(distribution.locate_file(relative_path)).resolve()
            if not path.is_file():
                raise RuntimeError(f"approved upstream source is missing: {path}")
            actual_sha256 = _file_sha256(path)
            if actual_sha256 != approved_sha256:
                raise RuntimeError(
                    "installed upstream source differs from the approved deployment: "
                    f"{distribution_name}:{relative_path}, "
                    f"expected={approved_sha256}, actual={actual_sha256}"
                )
            entry = {
                "path": str(path),
                "sha256": actual_sha256,
            }
            package_files[relative_path] = entry
            all_files[relative_path] = entry
        packages[distribution_name] = {
            "version": distribution.version,
            "files": package_files,
        }
    semantic_contracts = _validate_fastdeploy_semantics(all_files)
    result = {
        "schema_version": "bts_fastdeploy_dependency_source_contract_v1",
        "approved_lock_sha256": _file_sha256(LOCK_PATH),
        "packages": packages,
        "semantic_contracts": semantic_contracts,
    }
    result["sha256"] = _canonical_sha256(result)
    return result


def validate_runtime_dependency_imports(contract: dict[str, Any]) -> dict[str, str]:
    """Prove runtime modules resolve to the already-hashed approved files."""
    result = {}
    for module_name, (distribution_name, relative_path) in RUNTIME_MODULES.items():
        module = importlib.import_module(module_name)
        actual = Path(module.__file__).resolve()
        expected = Path(
            contract["packages"][distribution_name]["files"][relative_path]["path"]
        ).resolve()
        if actual != expected:
            raise RuntimeError(
                f"runtime module {module_name} is shadowed: expected={expected}, "
                f"actual={actual}"
            )
        result[module_name] = str(actual)
    return result
