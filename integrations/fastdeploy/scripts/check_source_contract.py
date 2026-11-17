"""Check the installed FastDeploy source without importing Paddle or using GPU.

This checks the actual release's integration seam. It does not replace an
end-to-end inference test.
"""

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    from bts_fastdeploy.upstream_contract import dependency_source_contract

    dependency_contract = dependency_source_contract()
    dist = importlib.metadata.distribution("fastdeploy-gpu")
    if dist.version.split("+")[0] != "2.5.0":
        raise RuntimeError(f"Unsupported FastDeploy version: {dist.version}")
    backend = Path(dist.locate_file("fastdeploy/model_executor/layers/attention/flash_attn_backend.py"))
    text = backend.read_text()
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "FlashAttentionBackend")
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "forward_mixed")
    calls = [n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    names = [n.func.id for n in calls]
    for name in ("flash_attn_func","gqa_rope_write_cache","append_attention","merge_prefill_decode_output"):
        if names.count(name) != 1:
            raise RuntimeError(f"Unexpected upstream calls: {name} occurs {names.count(name)} times")
    flash = next(n for n in calls if n.func.id == "flash_attn_func")
    parents = {child:parent for parent in ast.walk(method) for child in ast.iter_child_nodes(parent)}
    parent = parents[flash]
    if not (isinstance(parent,ast.Subscript) and isinstance(parent.slice,ast.Constant) and parent.slice.value == 0):
        raise RuntimeError("FastDeploy now consumes more than the attention output")

    args_path = Path(dist.locate_file("fastdeploy/engine/args_utils.py"))
    args_tree = ast.parse(args_path.read_text())
    args_cls = next(n for n in args_tree.body if isinstance(n,ast.ClassDef) and n.name == "EngineArgs")
    fields = {n.target.id for n in args_cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
    required = {"model","tensor_parallel_size","data_parallel_size","max_model_len","max_num_seqs",
                "max_num_batched_tokens","enable_chunked_prefill","enable_prefix_caching",
                "graph_optimization_config","gpu_memory_utilization","disable_custom_all_reduce"}
    if required - fields:
        raise RuntimeError(f"Unsupported launcher fields: {required - fields}")
    create_engine = next(
        n for n in args_cls.body
        if isinstance(n,ast.FunctionDef) and n.name == "create_engine_config"
    )
    disable_guard = [
        n for n in ast.walk(create_engine)
        if isinstance(n,ast.Attribute)
        and isinstance(n.value,ast.Name)
        and n.value.id == "envs"
        and n.attr == "FD_DISABLE_CHUNKED_PREFILL"
    ]
    if len(disable_guard) != 1:
        raise RuntimeError(
            "FastDeploy 2.5 chunked-prefill opt-out contract changed: "
            f"found {len(disable_guard)} FD_DISABLE_CHUNKED_PREFILL reads"
        )
    disables_chunking = any(
        isinstance(n,ast.Assign)
        and any(
            isinstance(target,ast.Attribute)
            and isinstance(target.value,ast.Name)
            and target.value.id == "self"
            and target.attr == "enable_chunked_prefill"
            for target in n.targets
        )
        and isinstance(n.value,ast.Constant)
        and n.value.value is False
        for n in ast.walk(create_engine)
    )
    if not disables_chunking:
        raise RuntimeError("FastDeploy no longer disables chunked prefill through the guarded branch")
    # Compile the genuine upstream method. Verify the private-globals technique
    # against release bytecode without instantiating a GPU backend.
    from bts_fastdeploy.backend import clone_prefill_method
    isolated = ast.Module(body=[method],type_ignores=[])
    for node in ast.walk(method):
        if isinstance(node,ast.arg):
            node.annotation = None
        if isinstance(node,ast.FunctionDef):
            node.returns = None
    namespace = {}
    exec(compile(ast.fix_missing_locations(isolated),str(backend),"exec"),namespace)
    original = namespace["forward_mixed"]
    clone = clone_prefill_method(original,lambda *args,**kwargs: None)
    assert clone.__code__ is original.__code__
    assert clone.__globals__ is not original.__globals__
    print(json.dumps(dict(status="passed", fastdeploy=dist.version, file=str(backend),
                         source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         dependency_contract_sha256=dependency_contract["sha256"],
                         approved_dependency_files=sum(
                             len(package["files"])
                             for package in dependency_contract["packages"].values()
                         ),
                         chunked_prefill_guard="FD_DISABLE_CHUNKED_PREFILL=1",
                         check="source contract and genuine method cloning; no GPU execution"),indent=2))


if __name__ == "__main__":
    main()
