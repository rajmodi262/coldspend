"""External validation: checking the simulator against real-world records."""

from .labels import (
    LabelStats,
    StorageSpec,
    fetch_labels,
    parse_allowance,
    parse_range,
    summarise,
)
from .openfda import (
    FDAFailureMix,
    classify,
    compare_to_simulation,
    failure_mix,
    fetch_events,
)

__all__ = [
    "FDAFailureMix", "LabelStats", "StorageSpec", "classify", "compare_to_simulation",
    "failure_mix", "fetch_events", "fetch_labels", "parse_allowance", "parse_range",
    "summarise",
]
