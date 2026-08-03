"""The generator: products, lane network, ambient sources, shipment simulation."""

from .network import AIRPORTS, LANES, Airport, Lane, great_circle_km
from .products import CATALOGUE, Product, by_name
from .calibrate import check as calibration_check
from .calibrate import report as calibration_report
from .shipment import SimConfig, simulate_portfolio, simulate_shipment
from .weather import AmbientSource, OpenMeteoArchive, SyntheticClimate

__all__ = [
    "AIRPORTS", "CATALOGUE", "LANES", "Airport", "AmbientSource", "Lane",
    "OpenMeteoArchive", "Product", "SimConfig", "SyntheticClimate", "by_name",
    "calibration_check", "calibration_report", "great_circle_km",
    "simulate_portfolio", "simulate_shipment",
]
