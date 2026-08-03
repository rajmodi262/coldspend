"""Causal layer: regression discontinuity at the industry alarm threshold."""

from .rd import (
    RECOMMENDED_BANDWIDTH,
    RDResult,
    bootstrap_ci,
    diagnostics,
    fuzzy_rd,
    local_truth,
    mse_optimal_bandwidth,
    report,
)

__all__ = [
    "RDResult", "RECOMMENDED_BANDWIDTH", "bootstrap_ci", "diagnostics", "fuzzy_rd",
    "local_truth", "mse_optimal_bandwidth", "report",
]
