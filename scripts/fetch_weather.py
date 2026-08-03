"""Download and cache one year of real hourly reanalysis per airport.

Eighteen requests for the whole network. The cache is COMMITTED: reproducibility
depends on it, and it means the site builds with no network at all.

    python scripts/fetch_weather.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coldspend.sim import AIRPORTS, ReanalysisClimate  # noqa: E402


def main() -> None:
    clim = ReanalysisClimate()
    print(f"fetching {len(AIRPORTS)} airports into {clim.cache_dir} ...")
    for i, (code, a) in enumerate(sorted(AIRPORTS.items()), 1):
        t0 = time.time()
        arr = clim.fetch(a.lat, a.lon)
        print(f"  {i:2d}/{len(AIRPORTS)}  {code}  {a.name:<16} "
              f"{arr.size:5d}h  {arr.min():6.1f}..{arr.max():5.1f} C  "
              f"mean {arr.mean():5.1f}  ({time.time() - t0:.1f}s)")
    total = sum(p.stat().st_size for p in clim.cache_dir.glob("*.npz"))
    print(f"cache: {total / 1024:.0f} KB across {len(list(clim.cache_dir.glob('*.npz')))} files")


if __name__ == "__main__":
    main()
