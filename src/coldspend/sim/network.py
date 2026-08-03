"""Lane network — real airports, real pharma corridors.

Coordinates are genuine IATA airport positions. The lanes are real
pharmaceutical freight corridors: Hyderabad and Mumbai are India's pharma export
gateways, Brussels and Frankfurt are Europe's pharma air hubs (both IATA CEIV
Pharma certified), Basel/Zurich serve the Swiss originators, and Memphis is the
US express hub.

The HUB matters more than the endpoints. A hub is where a shipment is on the
ground, reachable, and where an intervention is physically possible — so hubs are
the decision epochs. Interventions cannot happen mid-flight, which is not a
modelling convenience but the actual operational constraint.

This is a curated set, not the full OurAirports database. Expanding it is a
30-line change; a wider network adds breadth, not insight, and Phase 1 is not
where breadth pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

__all__ = ["Airport", "Lane", "AIRPORTS", "LANES", "great_circle_km", "by_iata"]


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    lat: float
    lon: float


AIRPORTS: dict[str, Airport] = {
    a.iata: a
    for a in (
        Airport("BOM", "Mumbai", 19.0887, 72.8679),
        Airport("HYD", "Hyderabad", 17.2403, 78.4294),
        Airport("DEL", "Delhi", 28.5562, 77.1000),
        Airport("DXB", "Dubai", 25.2532, 55.3657),
        Airport("DOH", "Doha", 25.2731, 51.6080),
        Airport("FRA", "Frankfurt", 50.0379, 8.5622),
        Airport("BRU", "Brussels", 50.9014, 4.4844),
        Airport("AMS", "Amsterdam", 52.3105, 4.7683),
        Airport("LHR", "London Heathrow", 51.4700, -0.4543),
        Airport("ZRH", "Zurich", 47.4647, 8.5492),
        Airport("JFK", "New York JFK", 40.6413, -73.7781),
        Airport("ORD", "Chicago O'Hare", 41.9742, -87.9073),
        Airport("MEM", "Memphis", 35.0424, -89.9767),
        Airport("SIN", "Singapore", 1.3644, 103.9915),
        Airport("PVG", "Shanghai Pudong", 31.1443, 121.8083),
        Airport("NRT", "Tokyo Narita", 35.7720, 140.3929),
        Airport("GRU", "Sao Paulo", -23.4356, -46.4731),
        Airport("JNB", "Johannesburg", -26.1392, 28.2460),
    )
}


@dataclass(frozen=True)
class Lane:
    origin: str
    hub: str
    dest: str
    """origin -> hub -> dest. The hub is the decision epoch."""

    @property
    def code(self) -> str:
        return f"{self.origin}-{self.hub}-{self.dest}"

    def legs(self) -> tuple[tuple[str, str], ...]:
        return ((self.origin, self.hub), (self.hub, self.dest))


LANES: tuple[Lane, ...] = (
    Lane("BOM", "DXB", "FRA"),    # the monsoon lane — the one the deck singles out
    Lane("HYD", "DXB", "BRU"),    # India pharma -> Europe's CEIV hub
    Lane("HYD", "DOH", "LHR"),
    Lane("DEL", "DXB", "JFK"),
    Lane("BOM", "DOH", "AMS"),
    Lane("SIN", "DXB", "BRU"),
    Lane("PVG", "NRT", "ORD"),
    Lane("ZRH", "FRA", "JFK"),    # Swiss originator -> US
    Lane("BRU", "JFK", "MEM"),
    Lane("FRA", "DXB", "SIN"),
    Lane("JNB", "DXB", "FRA"),
    Lane("GRU", "JFK", "AMS"),
)


def by_iata(code: str) -> Airport:
    try:
        return AIRPORTS[code]
    except KeyError:
        raise KeyError(f"unknown airport {code!r}") from None


def great_circle_km(a: Airport, b: Airport) -> float:
    """Haversine distance. Used to set flight duration, so it feeds transit time
    and therefore exposure — not decoration."""
    lat1, lon1, lat2, lon2 = map(radians, (a.lat, a.lon, b.lat, b.lon))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def flight_hours(a: Airport, b: Airport, cruise_kmh: float = 850.0) -> float:
    """Block time: great-circle at cruise speed, plus 45 min for taxi and climb."""
    return great_circle_km(a, b) / cruise_kmh + 0.75
