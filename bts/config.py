"""Frozen submission configuration.

The submission intentionally exposes one algorithm, not a matrix of experimental
switches. These constants reproduce the frozen method: dual-band BTS at mass
threshold 0.93, candidate-relative exact Top-K0.4 with mean-128 token scores,
Tail256 candidate preservation, and the fixed-prefix packed-attention kernel
with an internal generic correctness fallback.
"""

BLOCK_SIZE = 128
TOKEN_RATIO = 0.4
DENSE_TAIL_TOKENS = 256

LOW_FREQ_DIM = 96
HIGH_FREQ_DIM = 64
LOW_FREQ_THRESHOLD = 0.93
HIGH_FREQ_THRESHOLD = 0.93

FORCE_SINK = True
FORCE_RECENT = True


def describe() -> dict[str, int | float | str]:
    """Return the immutable method settings recorded with results."""
    return {
        "block_size": BLOCK_SIZE,
        "block_selection": "dual_band_probability_mass",
        "calibrate": True,
        "force_sink": FORCE_SINK,
        "force_recent": FORCE_RECENT,
        "token_selection": "block_count",
        "token_ratio": TOKEN_RATIO,
        "token_score": "qk_block_mean",
        "query_block_size": BLOCK_SIZE,
        "query_dtype": "bfloat16",
        "query_mean_dtype": "bfloat16_cached",
        "token_score_dtype": "float32",
        "dense_tail_tokens": DENSE_TAIL_TOKENS,
        "attention": "fixed_prefix_packed_default_with_generic_fallback",
        "low_freq_dim": LOW_FREQ_DIM,
        "high_freq_dim": HIGH_FREQ_DIM,
        "low_freq_threshold": LOW_FREQ_THRESHOLD,
        "high_freq_threshold": HIGH_FREQ_THRESHOLD,
    }
