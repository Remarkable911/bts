"""Configuration boundary for the single frozen BTS submission method.

The FastDeploy adapter deliberately has no copy of the research configuration
matrix. A configuration file can assert some or all of the frozen settings,
but it cannot select another algorithm. Reports always serialize the complete
canonical description returned by :func:`bts.describe`.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path


MAX_SUPPORTED_SEQS = 16


@dataclass(frozen=True)
class FrozenBTSConfig:
    """Serializable description of the immutable submitted algorithm."""

    block_size: int
    block_selection: str
    calibrate: bool
    force_sink: bool
    force_recent: bool
    token_selection: str
    token_ratio: float
    token_score: str
    query_block_size: int
    query_dtype: str
    query_mean_dtype: str
    token_score_dtype: str
    dense_tail_tokens: int
    attention: str
    low_freq_dim: int
    high_freq_dim: int
    low_freq_threshold: float
    high_freq_threshold: float


def _canonical_config():
    # Import lazily: importing/installing the optional adapter must not import
    # Torch, Transformers, Paddle, FastDeploy, or activate any patches.
    from bts import describe

    description = describe()
    expected = {field.name for field in fields(FrozenBTSConfig)}
    if set(description) != expected:
        raise RuntimeError(
            "The installed BTS public configuration contract changed: "
            f"expected {sorted(expected)}, found {sorted(description)}"
        )
    return FrozenBTSConfig(**description)


# Compatibility aliases for files written by the former research adapter.
# Values are assertions only; the adapter exposes no algorithm switch.
_LEGACY_SECTIONS = {
    "block": {
        "block_size": "block_size",
        "calibrate": "calibrate",
        "force_sink": "force_sink",
        "force_recent": "force_recent",
        "low_freq_dim": "low_freq_dim",
        "high_freq_dim": "high_freq_dim",
        "low_freq_threshold": "low_freq_threshold",
        "high_freq_threshold": "high_freq_threshold",
    },
    "token": {
        "token_select": "token_selection",
        "token_ratio": "token_ratio",
        "score_mode": "token_score",
        "query_block_size": "query_block_size",
        "dense_tail_tokens": "dense_tail_tokens",
    },
}


def _check_value(name, actual, expected, errors):
    if actual != expected:
        errors.append(f"{name}={actual!r} (frozen value is {expected!r})")


def _validate_file_assertions(data, canonical):
    if not isinstance(data, dict):
        raise ValueError("Expected a BTS configuration object")
    expected = asdict(canonical)
    errors = []
    unknown = []
    for name, value in data.items():
        if name in expected:
            _check_value(name, value, expected[name], errors)
            continue
        if name in _LEGACY_SECTIONS:
            if not isinstance(value, dict):
                errors.append(f"{name} must be an object")
                continue
            aliases = _LEGACY_SECTIONS[name]
            for nested_name, nested_value in value.items():
                target = aliases.get(nested_name)
                dotted = f"{name}.{nested_name}"
                if target is None:
                    unknown.append(dotted)
                else:
                    _check_value(dotted, nested_value, expected[target], errors)
            continue
        # This one legacy flag exactly states the behavior of the frozen path.
        if name == "token_sparse":
            _check_value(name, value, True, errors)
            continue
        unknown.append(name)
    if unknown:
        errors.append("unsupported fields: " + ", ".join(sorted(unknown)))
    if errors:
        raise ValueError(
            "FastDeploy accepts only the frozen BTS algorithm: " + "; ".join(errors)
        )


def load_config(path=None):
    """Load the canonical algorithm description and validate optional assertions."""

    config = _canonical_config()
    if path is None:
        return config
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a BTS configuration object")
    if "runtime_config" in data:
        if len(data) != 1:
            raise ValueError("runtime_config must be the only top-level configuration field")
        data = data["runtime_config"]
    elif "algorithm" in data:
        if len(data) != 1:
            raise ValueError("algorithm must be the only top-level configuration field")
        data = data["algorithm"]
    _validate_file_assertions(data, config)
    return config


def validate_algorithm(config):
    """Reject stale research configurations before any attention execution."""

    if not isinstance(config, FrozenBTSConfig):
        raise TypeError("Expected FrozenBTSConfig from bts_fastdeploy.load_config")
    canonical = _canonical_config()
    if config != canonical:
        actual = asdict(config)
        expected = asdict(canonical)
        differences = [
            f"{name}={actual[name]!r} (frozen value is {expected[name]!r})"
            for name in expected
            if actual[name] != expected[name]
        ]
        raise ValueError(
            "FastDeploy accepts only the frozen BTS algorithm: " + "; ".join(differences)
        )


def config_digest(config):
    validate_algorithm(config)
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_engine(fd_config):
    """Fail before cache writes or model execution for unimplemented semantics."""
    errors = []
    cache = fd_config.cache_config
    parallel = fd_config.parallel_config
    graph = fd_config.graph_opt_config
    model = fd_config.model_config
    for name in ("enable_chunked_prefill", "enable_prefix_caching"):
        if getattr(cache, name, False):
            errors.append(name)
    if graph.use_cudagraph or graph.graph_opt_level != 0:
        errors.append("CUDA Graph/static graph (use eager execution)")
    if parallel.tensor_parallel_size != 1 or parallel.data_parallel_size != 1:
        errors.append("TP/DP other than 1")
    if getattr(parallel, "pd_disaggregation_mode", "None") not in ("mixed", "None", None):
        errors.append("prefill/decode disaggregation")
    if fd_config.speculative_config.method is not None:
        errors.append("speculative decoding")
    hf = getattr(model, "hf_config", None)
    model_type = getattr(hf, "model_type", None) or getattr(model, "model_type", None)
    if model_type != "qwen3":
        errors.append(f"model_type={model_type!r} (initial support is Qwen3 text)")
    head_dim = getattr(hf, "head_dim", None) or getattr(model, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(hf, "hidden_size", None) or getattr(model, "hidden_size", None)
        num_heads = getattr(hf, "num_attention_heads", None) or getattr(
            model, "num_attention_heads", None
        )
        if hidden_size is not None and num_heads:
            if hidden_size % num_heads:
                errors.append("hidden_size is not divisible by num_attention_heads")
            else:
                head_dim = hidden_size // num_heads
    if head_dim is not None and head_dim != 128:
        errors.append(f"head_dim={head_dim!r} (frozen BTS requires 128)")
    if not getattr(model, "causal", True):
        errors.append("non-causal attention")
    if getattr(hf, "use_sliding_window", False) or getattr(model, "use_sliding_window", False):
        errors.append("sliding window attention")
    max_num_seqs = fd_config.scheduler_config.max_num_seqs
    if (
        isinstance(max_num_seqs, bool)
        or not isinstance(max_num_seqs, int)
        or not 1 <= max_num_seqs <= MAX_SUPPORTED_SEQS
    ):
        errors.append(
            f"max_num_seqs={max_num_seqs!r} "
            f"(validated range is 1..{MAX_SUPPORTED_SEQS})"
        )
    if (
        getattr(model, "is_quantized", False)
        or getattr(model, "quantization", None)
        or getattr(model, "quantization_config", None)
        or getattr(hf, "quantization_config", None)
    ):
        errors.append("quantized models")
    if str(getattr(model, "dtype", "bfloat16")) not in ("bfloat16", "paddle.bfloat16"):
        errors.append("model dtype other than bfloat16")
    if errors:
        raise ValueError("BTS FastDeploy unsupported configuration: " + ", ".join(errors))
