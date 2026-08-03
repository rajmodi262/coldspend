"""External validation against FDA's own recall record.

WHY THIS MODULE IS THE ANSWER TO "IT'S ALL SIMULATED"
-----------------------------------------------------
Being *driven* by real data is weaker than being *checked against* it. This is
pattern-oriented modelling: pick a pattern in the real world that the simulator
was never fitted to, and report how it compares — whether or not it matches.

THE DENOMINATOR TRAP, WHICH MATTERS MORE THAN THE RESULT
--------------------------------------------------------
openFDA tells you what share of *drug recalls* were caused by temperature. The
simulator tells you what share of *shipments* were destroyed. Those have
different denominators and comparing them directly would be meaningless — it
would look quantitative and be nonsense.

What IS comparable is the COMPOSITION of failures. Among things that went wrong
thermally, how many went wrong by being too hot versus too cold? Both sources
can answer that, and neither was fitted to the other.

AND THE DEDUPLICATION TRAP
--------------------------
openFDA returns one record per PRODUCT, not per event: 619 records collapse to
78 distinct `event_id`s, an inflation of roughly 7.9x. A single distributor
recall of many SKUs would otherwise dominate every statistic computed from it.
Always deduplicate on `event_id` before counting anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FDAFailureMix", "fetch_events", "classify", "failure_mix", "compare_to_simulation"]

ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
SEARCH = (
    'reason_for_recall:("temperature" OR "storage" OR "excursion" '
    'OR "refrigerated" OR "frozen")'
)
CACHE = Path("data/cache/openfda/enforcement_temperature.json")

# Classifying free text is crude, so the categories are deliberately coarse and
# "unspecified" is REPORTED rather than forced into a bucket. Roughly a third of
# real recall reasons say only "product held outside labeled storage conditions",
# which is genuinely silent on direction; pretending otherwise would manufacture
# a result.
#
# NOTE "abuse" and "excursion" are NOT heat indicators, though the phrasing
# tempts you. FDA uses "Temperature Abuse" for both directions — one recall in
# this very dataset reads "Temperature Abuse: product samples were stored at
# temperatures below 32* F". Treating them as heat inflated the heat share and
# drove the measured freeze share to zero.
_HEAT = re.compile(
    r"\b(heat|hot|high temperature|elevated temperature|exceed(ed|ing)?|"
    r"above (the )?(labell?ed|recommended|acceptable|required)|room temperature)\b",
    re.I,
)
_COLD = re.compile(
    r"\b(frozen|freez\w*|froze|sub-?freezing|sub-?zero|too cold|cold storage|"
    r"below[- ](the )?(32|recommended|labell?ed|acceptable|required|freezing))\b",
    re.I,
)


def fetch_events(force: bool = False, cache: Path = CACHE) -> list[dict]:
    """Temperature/storage-related drug recalls, ONE ROW PER EVENT.

    Cached to disk and committed, for the same reason the weather cache is: a
    figure that silently changes when FDA reindexes is not reproducible.
    """
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    import requests

    rows: list[dict] = []
    for skip in range(0, 2000, 100):
        r = requests.get(
            ENFORCEMENT_URL, params={"search": SEARCH, "limit": 100, "skip": skip}, timeout=60
        )
        if r.status_code != 200:
            break
        batch = r.json().get("results", [])
        if not batch:
            break
        rows.extend(batch)

    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r.get("event_id", r.get("recall_number", "")), r)
    events = list(seen.values())

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(events, indent=1), encoding="utf-8")
    return events


def classify(reason: str) -> str:
    """heat | freeze | both | unspecified."""
    h, c = bool(_HEAT.search(reason or "")), bool(_COLD.search(reason or ""))
    if h and c:
        return "both"
    if h:
        return "heat"
    if c:
        return "freeze"
    return "unspecified"


@dataclass
class FDAFailureMix:
    n_events: int
    counts: dict[str, int] = field(default_factory=dict)
    n_records: int = 0

    @property
    def inflation(self) -> float:
        """How many product rows FDA returns per actual event."""
        return self.n_records / max(self.n_events, 1)

    @property
    def n_classifiable(self) -> int:
        return (self.counts.get("heat", 0) + self.counts.get("both", 0)
                + self.counts.get("freeze", 0))

    @property
    def n_freeze(self) -> int:
        return self.counts.get("freeze", 0) + self.counts.get("both", 0)

    @property
    def freeze_share(self) -> float:
        """Freeze as a share of DIRECTIONALLY CLASSIFIABLE failures.

        'unspecified' is excluded rather than assumed, because a recall reason
        that says only 'product held outside labeled storage conditions' is
        genuinely silent on direction — and 60% of them are.
        """
        return self.n_freeze / max(self.n_classifiable, 1)

    @property
    def freeze_share_ci(self) -> tuple[float, float]:
        """Clopper-Pearson 95% interval.

        Reported because the point estimate rests on FOUR events. Quoting 12.9%
        without saying the interval runs from about 4% to 30% would imply a
        precision this evidence does not have, and would make any comparison
        against it look far more decisive than it is.
        """
        from scipy.stats import beta

        k, n = self.n_freeze, self.n_classifiable
        if n == 0:
            return (0.0, 1.0)
        lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        return (lo, hi)


def failure_mix(events: list[dict] | None = None) -> FDAFailureMix:
    events = events if events is not None else fetch_events()
    counts: dict[str, int] = {}
    for e in events:
        k = classify(e.get("reason_for_recall", ""))
        counts[k] = counts.get(k, 0) + 1
    return FDAFailureMix(n_events=len(events), counts=counts)


def compare_to_simulation(df, fda: FDAFailureMix | None = None) -> dict[str, float]:
    """Simulated failure composition against FDA's, on the one axis both measure.

    The simulator's freeze share is computed the same way: freezing events as a
    fraction of all thermal failures, heat or cold.
    """
    fda = fda or failure_mix()

    heat = int((df["excursion"] == 1).sum())
    freeze = int((df["freeze_degree_h"] > 0).sum())
    sim_share = freeze / max(heat + freeze, 1)

    lo, hi = fda.freeze_share_ci
    return {
        "fda_freeze_share": fda.freeze_share,
        "fda_ci_lo": lo,
        "fda_ci_hi": hi,
        "sim_freeze_share": sim_share,
        "fda_events": float(fda.n_events),
        "fda_classifiable": float(fda.n_classifiable),
        "sim_shipments": float(len(df)),
        "inside_interval": float(lo <= sim_share <= hi),
    }


STORAGE_NOT_TRANSIT = """\
Every freeze-caused recall in this dataset is a STORAGE failure, not a transit
failure: product held below 32 F in a distribution centre, exposed to
subfreezing temperatures in a warehouse, or crystallised after cold storage.

The simulator models TRANSIT ONLY. It has no warehouse stage, so it structurally
cannot produce the mechanism behind the entire real freeze record. That is a
scope limitation of the model, found by this comparison rather than assumed —
and it is a better outcome than agreement would have been, because agreement
would have told us nothing we did not already believe."""
