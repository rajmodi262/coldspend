"""Real storage conditions and excursion allowances, from FDA drug labels.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
The product catalogue's storage ranges were invented — plausible, but invented.
77,324 openFDA drug labels carry machine-readable `storage_and_handling` text,
so they need not be.

More importantly, the labels state **pre-approved excursion allowances** in their
own words:

    "store at 25 C; excursions permitted between 15-30 C"

That phrase is the foundation of this project's GMP claim. The argument is not
"software decides disposition" — it is "stop a shipment spending stability
budget it does not have, so the excursion that arrives falls INSIDE a
pre-approved allowance rather than outside one". Until now that rested on a
single NHS reference. It now rests on the regulator's own label corpus.

A PARSING TRAP WORTH KNOWING
----------------------------
Labels do not write degrees consistently. The same corpus uses U+00B0 (`°`),
U+00BA (`º`), U+2070 (superscript zero), a bare `o`, and sometimes nothing at
all. A regex expecting `°C` finds nothing — literally zero matches in 500
labels, which is how this was discovered. The character class below covers all
of them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

__all__ = ["StorageSpec", "LabelStats", "fetch_labels", "parse_range",
           "parse_allowance", "summarise"]

LABEL_URL = "https://api.fda.gov/drug/label.json"
CACHE = Path("data/cache/openfda/labels_storage.json")

# Every degree-like character the corpus actually uses. See the module docstring.
_DEG = "[°º⁰·o*]?"
_RANGE = re.compile(
    r"(-?\d{1,3})\s*" + _DEG + r"\s*C?\s*(?:to|-|and|–)\s*(-?\d{1,3})\s*" + _DEG + r"\s*C",
    re.I,
)
_ALLOWANCE = re.compile(
    r"excursions?\s+(?:are\s+)?permitted[^.]{0,80}?"
    r"(-?\d{1,3})\s*" + _DEG + r"\s*C?\s*(?:to|-|and|–)\s*(-?\d{1,3})\s*" + _DEG + r"\s*C",
    re.I,
)


@dataclass(frozen=True)
class StorageSpec:
    low_c: float
    high_c: float
    excursion_low_c: float | None = None
    excursion_high_c: float | None = None

    @property
    def has_allowance(self) -> bool:
        return self.excursion_low_c is not None

    @property
    def headroom_c(self) -> float:
        """How far above the labelled maximum an excursion is still permitted.

        This is the quantity the whole decision layer is aiming at: not "did we
        breach" but "how much permitted headroom is left".
        """
        if self.excursion_high_c is None:
            return 0.0
        return max(0.0, self.excursion_high_c - self.high_c)


def parse_range(text: str) -> tuple[int, int] | None:
    for a, b in _RANGE.findall(text or ""):
        try:
            lo, hi = int(a), int(b)
        except ValueError:
            continue
        if -90 <= lo < hi <= 60:
            return lo, hi
    return None


def parse_allowance(text: str) -> tuple[int, int] | None:
    m = _ALLOWANCE.search(text or "")
    if not m:
        return None
    try:
        lo, hi = int(m.group(1)), int(m.group(2))
    except ValueError:
        return None
    return (lo, hi) if -90 <= lo < hi <= 60 else None


def fetch_labels(n: int = 1000, force: bool = False, cache: Path = CACHE) -> list[str]:
    """Storage-and-handling text from real drug labels. Cached and committed."""
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    import requests

    out: list[str] = []
    for skip in range(0, n, 100):
        r = requests.get(
            LABEL_URL,
            params={"search": "_exists_:storage_and_handling", "limit": 100, "skip": skip},
            timeout=60,
        )
        if r.status_code != 200:
            break
        for res in r.json().get("results", []):
            out.append(re.sub(r"\s+", " ", " ".join(res.get("storage_and_handling") or [])))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


@dataclass
class LabelStats:
    n_labels: int
    ranges: Counter
    n_with_allowance: int
    mean_headroom_c: float

    @property
    def n_ranges(self) -> int:
        return sum(self.ranges.values())

    @property
    def refrigerated_share(self) -> float:
        """Share of labelled ranges that are 2-8 C.

        Worth knowing and slightly counter-intuitive: cold chain is a small
        MINORITY of drug labels by count. It matters because of what is in it,
        not how much of it there is.
        """
        cold = sum(c for (lo, hi), c in self.ranges.items() if lo <= 2 and hi <= 8)
        return cold / max(self.n_ranges, 1)


def summarise(texts: list[str] | None = None) -> LabelStats:
    texts = texts if texts is not None else fetch_labels()
    ranges: Counter = Counter()
    allow = 0
    headroom: list[float] = []

    for t in texts:
        r = parse_range(t)
        if r:
            ranges[r] += 1
        a = parse_allowance(t)
        if a and r:
            allow += 1
            headroom.append(max(0.0, a[1] - r[1]))

    return LabelStats(
        n_labels=len(texts),
        ranges=ranges,
        n_with_allowance=allow,
        mean_headroom_c=float(sum(headroom) / len(headroom)) if headroom else 0.0,
    )
