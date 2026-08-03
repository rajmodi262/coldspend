"""Causal layer: regression discontinuity at the industry alarm threshold."""

from .rd import RDResult, bootstrap_ci, diagnostics, fuzzy_rd, local_truth, report

__all__ = ["RDResult", "bootstrap_ci", "diagnostics", "fuzzy_rd", "local_truth", "report"]
