"""External validation: checking the simulator against real-world records."""

from .openfda import (
    FDAFailureMix,
    classify,
    compare_to_simulation,
    failure_mix,
    fetch_events,
)

__all__ = [
    "FDAFailureMix", "classify", "compare_to_simulation", "failure_mix", "fetch_events",
]
