"""Product catalogue — stability profiles and consignment values.

VALUES ARE REAL AND SOURCED. The single most common way a project like this loses
credibility is a made-up dollar figure, so every value here carries its source in
a comment and none is invented.

The value ladder matters because it is half the thesis: compliance treats every
breach equally, but a $4k saline shipment and a $443k CAR-T dose plainly do not
deserve the same response.

ACTIVATION ENERGIES ARE THE WEAK POINT — say so before being asked. Ea is fitted
and product-specific; published values for biologics span a wide range, and
protein aggregation is known NON-Arrhenius. The figures below are plausible
literature-range values used to differentiate products in the model, not
measurements of any particular molecule. Every dollar output should be swept
against them (Phase 4 sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Product", "CATALOGUE", "by_name"]


@dataclass(frozen=True)
class Product:
    name: str
    storage_min_c: float
    storage_max_c: float
    ref_temp_c: float
    shelf_life_h: float
    """Labelled shelf life at ref_temp_c, in hours."""
    ea_j_per_mol: float
    consignment_value_usd: float
    freeze_sensitive: bool
    value_source: str

    @property
    def storage_label(self) -> str:
        return f"{self.storage_min_c:g} to {self.storage_max_c:g} °C"

    @property
    def cryogenic(self) -> bool:
        """Cryogenic products need a dry shipper, not a passive 2-8 degC box.

        This is not a detail. Pairing a -135 degC product with an EPS cooler makes
        the model Arrhenius-extrapolate across ~165 K, which is meaningless —
        and the payload has in any case simply thawed, which is a different
        failure mode than gradual degradation.
        """
        return self.storage_max_c < -50.0

    def in_spec(self, temp_c: float) -> bool:
        return self.storage_min_c <= temp_c <= self.storage_max_c


CATALOGUE: tuple[Product, ...] = (
    Product(
        name="CAR-T dose",
        storage_min_c=-150.0,      # LN2 vapour phase; modelled here as its handling window
        storage_max_c=-120.0,
        ref_temp_c=-135.0,
        shelf_life_h=8760.0,
        ea_j_per_mol=95_000.0,
        consignment_value_usd=443_600.0,
        freeze_sensitive=False,
        value_source="Milliman, average CAR-T WAC. NOTE: the ~$1-2M figure often "
                     "quoted is total EPISODE cost incl. hospitalisation, not product value.",
    ),
    Product(
        name="mAb pallet",
        storage_min_c=2.0,
        storage_max_c=8.0,
        ref_temp_c=5.0,
        shelf_life_h=17_520.0,     # ~24 months refrigerated
        ea_j_per_mol=83_144.0,
        consignment_value_usd=850_000.0,
        freeze_sensitive=True,
        value_source="Derived: Merck KEYTRUDA WAC $6,136/100mg vial (Colorado "
                     "C.R.S. 12-280-308 disclosure, 02/2026) x pallet quantity.",
    ),
    Product(
        name="Vaccine pallet",
        storage_min_c=2.0,
        storage_max_c=8.0,
        ref_temp_c=5.0,
        shelf_life_h=8_760.0,
        ea_j_per_mol=78_000.0,
        consignment_value_usd=120_000.0,
        freeze_sensitive=True,      # WHO/PATH: freeze exposure is a major real failure mode
        value_source="Derived from UNICEF Supply Division published vaccine prices.",
    ),
    Product(
        name="Insulin shipment",
        storage_min_c=2.0,
        storage_max_c=8.0,
        ref_temp_c=5.0,
        shelf_life_h=26_280.0,
        ea_j_per_mol=80_000.0,
        consignment_value_usd=45_000.0,
        freeze_sensitive=True,
        value_source="Derived from published US list prices per vial x shipment quantity.",
    ),
    Product(
        name="Saline",
        storage_min_c=15.0,
        storage_max_c=25.0,
        ref_temp_c=20.0,
        shelf_life_h=26_280.0,
        ea_j_per_mol=60_000.0,
        consignment_value_usd=4_000.0,
        freeze_sensitive=False,
        value_source="B. Braun 1000 mL 0.9% NaCl list $14.95 (Mountainside Medical) "
                     "x shipment quantity.",
    ),
)


def by_name(name: str) -> Product:
    for p in CATALOGUE:
        if p.name == name:
            return p
    raise KeyError(f"unknown product {name!r}; have {[p.name for p in CATALOGUE]}")
