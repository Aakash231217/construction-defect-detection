"""Quantity-based repair cost estimation.

This module implements the practical civil-engineering approach to repair
costing that a site engineer or quantity surveyor would follow:

    1. Quantity estimation
       Based on the detected defect dimensions and severity, the module
       calculates the repair quantity in the appropriate unit:
         - Cracks ............ length (running metre) of crack to seal/inject
         - Spalling .......... area (sq m) of concrete to break out and patch
         - Honeycombing ...... area (sq m) and/or volume (cum) to grout/fill
         - Exposed rebar ..... area (sq m) of cover restoration

       The quantity depends on severity because the repair method changes:
         - Minor crack   -> surface sealant over the crack length
         - Moderate crack-> epoxy injection along the full crack length
         - Severe crack  -> injection + crack stitching (longer effective length)
         - Spalling      -> break-out depth increases with severity, so volume
                            grows even if surface area stays the same.

    2. Rate analysis
       Each repair item has a rate broken up into:
         - Material rate  (epoxy, mortar, sealant, steel, etc.)
         - Labour rate    (per unit of work)
         - Equipment rate (tools, pumps, scaffolding, etc.)
       The rates are indicative planning rates in INR, based on typical
       market/CPWD-style ranges.  They must be replaced with current local
       rates before billing.

    3. Cost = Quantity x Rate
       The total item cost is computed as quantity multiplied by the composite
       rate, and the breakup (material / labour / equipment) is shown so the
       professor can see exactly how the cost was derived.

All quantities are derived from the detected bounding-box dimensions when a
mm-per-pixel scale is available.  Without a scale the module falls back to an
area-ratio-based estimate and clearly flags it as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.defect_taxonomy import normalise_defect


class QuantityUnit(str, Enum):
    """Units in which a repair quantity can be expressed."""

    RUNNING_METRE = "running metre (rmt)"
    SQUARE_METRE = "sq m"
    CUBIC_METRE = "cum"
    KILOGRAM = "kg"
    NUMBERS = "nos"


@dataclass(frozen=True)
class RateBreakup:
    """Rate analysis for one repair item, in INR per unit quantity."""

    material_rate: float
    labour_rate: float
    equipment_rate: float

    @property
    def composite_rate(self) -> float:
        """Total rate per unit = material + labour + equipment."""
        return self.material_rate + self.labour_rate + self.equipment_rate

    def to_dict(self) -> dict[str, float]:
        return {
            "material_rate": self.material_rate,
            "labour_rate": self.labour_rate,
            "equipment_rate": self.equipment_rate,
            "composite_rate": self.composite_rate,
        }


@dataclass(frozen=True)
class QuantityEstimate:
    """Estimated repair quantity for a detected defect."""

    value: float
    unit: QuantityUnit
    description: str

    def formatted(self) -> str:
        return f"{self.value:.2f} {self.unit.value}"


@dataclass(frozen=True)
class CostEstimate:
    """Full cost estimate following Quantity x Rate = Cost."""

    quantity: QuantityEstimate
    rate: RateBreakup
    material_cost: float
    labour_cost: float
    equipment_cost: float
    total_cost: float
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "quantity": self.quantity.formatted(),
            "quantity_value": self.quantity.value,
            "quantity_unit": self.quantity.unit.value,
            "quantity_description": self.quantity.description,
            "material_rate": self.rate.material_rate,
            "labour_rate": self.rate.labour_rate,
            "equipment_rate": self.rate.equipment_rate,
            "composite_rate": self.rate.composite_rate,
            "material_cost": self.material_cost,
            "labour_cost": self.labour_cost,
            "equipment_cost": self.equipment_cost,
            "total_cost": self.total_cost,
            "notes": self.notes,
        }

    def formatted_summary(self) -> str:
        """Human-readable one-line cost summary for tables/reports."""
        return (
            f"Qty: {self.quantity.formatted()} | "
            f"Rate: INR {self.rate.composite_rate:.0f}/{self.quantity.unit.value} | "
            f"Cost: INR {self.total_cost:.0f}"
        )

    def formatted_detail(self) -> str:
        """Multi-line detailed breakup for reports."""
        return (
            f"Quantity: {self.quantity.formatted()} ({self.quantity.description})\n"
            f"Rate analysis (per {self.quantity.unit.value}):\n"
            f"  Material:   INR {self.rate.material_rate:.0f}\n"
            f"  Labour:     INR {self.rate.labour_rate:.0f}\n"
            f"  Equipment:  INR {self.rate.equipment_rate:.0f}\n"
            f"  Composite:  INR {self.rate.composite_rate:.0f}\n"
            f"Cost breakup:\n"
            f"  Material cost:   INR {self.material_cost:.0f}\n"
            f"  Labour cost:     INR {self.labour_cost:.0f}\n"
            f"  Equipment cost:  INR {self.equipment_cost:.0f}\n"
            f"  Total cost:      INR {self.total_cost:.0f}"
        )


# ---------------------------------------------------------------------------
# Rate analysis tables (indicative planning rates, INR per unit)
# ---------------------------------------------------------------------------
# These rates are indicative ranges based on typical Indian market / CPWD-style
# rate analysis.  They MUST be updated with current local rates before any
# billing or contractual use.
#
# Rate = Material + Labour + Equipment  (per unit of work)
# ---------------------------------------------------------------------------

# Crack repair rates per running metre
CRACK_RATES: dict[str, RateBreakup] = {
    # Minor: acrylic/PU sealant, surface application
    "minor": RateBreakup(
        material_rate=80.0,    # sealant cartridge
        labour_rate=50.0,      # surface prep + application
        equipment_rate=20.0,   # cleaning tools
    ),
    # Moderate: epoxy injection along crack length
    "moderate": RateBreakup(
        material_rate=250.0,   # epoxy injection resin + packers
        labour_rate=150.0,     # drilling, injection, finishing
        equipment_rate=100.0,  # injection pump, drill
    ),
    # Severe: structural epoxy injection + crack stitching
    "severe": RateBreakup(
        material_rate=500.0,   # high-grade epoxy + stitching bars/plates
        labour_rate=300.0,     # skilled repair + cause investigation
        equipment_rate=200.0,  # pump, drilling, scaffolding
    ),
    # Critical: engineer-designed strengthening (indicative)
    "critical": RateBreakup(
        material_rate=800.0,
        labour_rate=500.0,
        equipment_rate=400.0,
    ),
}

# Spalling repair rates per square metre
SPALLING_RATES: dict[str, RateBreakup] = {
    "minor": RateBreakup(
        material_rate=400.0,   # polymer-modified mortar
        labour_rate=250.0,     # surface prep + patching
        equipment_rate=150.0,  # hand tools
    ),
    "moderate": RateBreakup(
        material_rate=700.0,   # mortar + steel treatment
        labour_rate=400.0,     # break out, clean steel, reinstate
        equipment_rate=250.0,  # breaker, tools
    ),
    "severe": RateBreakup(
        material_rate=1200.0,  # micro-concrete/shotcrete + steel
        labour_rate=600.0,     # deep repair, formwork
        equipment_rate=500.0,  # shotcrete pump, formwork
    ),
    "critical": RateBreakup(
        material_rate=2000.0,  # jacketing/section restoration
        labour_rate=800.0,
        equipment_rate=700.0,
    ),
}

# Honeycombing repair rates per square metre
HONEYCOMBING_RATES: dict[str, RateBreakup] = {
    "minor": RateBreakup(
        material_rate=350.0,   # non-shrink mortar/grout
        labour_rate=200.0,
        equipment_rate=100.0,
    ),
    "moderate": RateBreakup(
        material_rate=600.0,   # pressure grout + mortar
        labour_rate=350.0,
        equipment_rate=250.0,  # grout pump
    ),
    "severe": RateBreakup(
        material_rate=1000.0,  # micro-concrete/shotcrete
        labour_rate=500.0,
        equipment_rate=450.0,
    ),
    "critical": RateBreakup(
        material_rate=1800.0,
        labour_rate=700.0,
        equipment_rate=600.0,
    ),
}

# Exposed reinforcement repair rates per square metre
EXPOSED_REBAR_RATES: dict[str, RateBreakup] = {
    "severe": RateBreakup(
        material_rate=900.0,   # mortar + corrosion inhibitor + steel treatment
        labour_rate=500.0,
        equipment_rate=300.0,
    ),
    "critical": RateBreakup(
        material_rate=1500.0,  # steel replacement + section restoration
        labour_rate=700.0,
        equipment_rate=500.0,
    ),
}

# Stepped masonry crack repair rates per running metre.  Severity is floored at
# Moderate for this defect (settlement indicator), so "minor" is defensive only.
STAIRSTEP_CRACK_RATES: dict[str, RateBreakup] = {
    "minor": RateBreakup(
        material_rate=200.0,   # repointing mortar + tell-tale
        labour_rate=350.0,
        equipment_rate=150.0,
    ),
    "moderate": RateBreakup(
        material_rate=335.0,   # repointing mortar, monitors, consumables
        labour_rate=508.0,     # raking out and repointing bed joints
        equipment_rate=218.0,  # grinder + access
    ),
    "severe": RateBreakup(
        material_rate=1200.0,  # helical stitching bars + mortar + epoxy
        labour_rate=883.0,     # slot cutting, bedding bars, repointing
        equipment_rate=650.0,  # drilling, grinder, scaffolding
    ),
    "critical": RateBreakup(
        material_rate=1600.0,
        labour_rate=1100.0,
        equipment_rate=850.0,
    ),
}

# ---------------------------------------------------------------------------
# Non-structural (durability / finish) defect rates, per square metre
# ---------------------------------------------------------------------------
# These are deliberately an order of magnitude below the concrete-repair rates.
# Quoting a repaint or a stain wash at spalling rates was the single largest
# source of cost error in the reports, so each finish defect now carries its own
# table instead of falling through to a structural one.

# Efflorescence / salt leaching (severity capped at Moderate)
EFFLORESCENCE_RATES: dict[str, RateBreakup] = {
    "negligible": RateBreakup(60.0, 60.0, 15.0),
    "minor": RateBreakup(110.0, 90.0, 20.0),     # wash + breathable repellent
    "moderate": RateBreakup(380.0, 190.0, 75.0),  # + crystalline waterproofing
    "severe": RateBreakup(380.0, 190.0, 75.0),
    "critical": RateBreakup(380.0, 190.0, 75.0),
}

# Active water seepage / damp patches (severity capped at Severe)
SEEPAGE_RATES: dict[str, RateBreakup] = {
    "negligible": RateBreakup(250.0, 150.0, 45.0),
    "minor": RateBreakup(400.0, 200.0, 60.0),     # local seal + coating
    "moderate": RateBreakup(550.0, 235.0, 75.0),  # crystalline waterproofing
    "severe": RateBreakup(1700.0, 510.0, 555.0),  # PU injection + full system
    "critical": RateBreakup(1700.0, 510.0, 555.0),
}

# Peeling / flaking paint (cosmetic; severity capped at Moderate)
PAINT_RATES: dict[str, RateBreakup] = {
    "negligible": RateBreakup(40.0, 20.0, 5.0),
    "minor": RateBreakup(72.0, 35.0, 10.0),     # scrape, spot prime, touch up
    "moderate": RateBreakup(200.0, 100.0, 45.0),  # strip back, prime, 2 coats
    "severe": RateBreakup(200.0, 100.0, 45.0),
    "critical": RateBreakup(200.0, 100.0, 45.0),
}

# Rust / corrosion staining (severity capped at Moderate; an indicator defect)
RUST_STAIN_RATES: dict[str, RateBreakup] = {
    "negligible": RateBreakup(50.0, 40.0, 10.0),
    "minor": RateBreakup(85.0, 65.0, 17.0),       # clean, seal, log for review
    "moderate": RateBreakup(305.0, 260.0, 375.0),  # + cover/half-cell survey
    "severe": RateBreakup(305.0, 260.0, 375.0),
    "critical": RateBreakup(305.0, 260.0, 375.0),
}


def _severity_key(level: str) -> str:
    return level.strip().lower()


def _get_crack_rate(level: str) -> RateBreakup:
    return CRACK_RATES.get(_severity_key(level), CRACK_RATES["moderate"])


def _get_spalling_rate(level: str) -> RateBreakup:
    return SPALLING_RATES.get(_severity_key(level), SPALLING_RATES["moderate"])


def _get_honeycombing_rate(level: str) -> RateBreakup:
    return HONEYCOMBING_RATES.get(_severity_key(level), HONEYCOMBING_RATES["moderate"])


def _get_exposed_rebar_rate(level: str) -> RateBreakup:
    return EXPOSED_REBAR_RATES.get(_severity_key(level), EXPOSED_REBAR_RATES["severe"])


def _get_stairstep_crack_rate(level: str) -> RateBreakup:
    return STAIRSTEP_CRACK_RATES.get(_severity_key(level), STAIRSTEP_CRACK_RATES["moderate"])


# Area-measured non-structural defects share one lookup shape.
NON_STRUCTURAL_RATE_TABLES: dict[str, dict[str, RateBreakup]] = {
    "efflorescence": EFFLORESCENCE_RATES,
    "water_seepage": SEEPAGE_RATES,
    "peeling_paint": PAINT_RATES,
    "rust_staining": RUST_STAIN_RATES,
}

# Allowance used when the detected class has no rate analysis behind it. Deliberately
# a clean-and-inspect figure, not a repair rate: the system does not know what the
# repair is, so it must not price one.
UNRECOGNISED_DEFECT_RATE = RateBreakup(
    material_rate=50.0,
    labour_rate=120.0,
    equipment_rate=30.0,
)

NON_STRUCTURAL_NOTES: dict[str, str] = {
    "efflorescence": (
        "Rate covers salt removal, breathable repellent and (from Moderate) "
        "crystalline waterproofing. Stopping the water path is what prevents "
        "recurrence; treating the face alone is not a durable repair."
    ),
    "water_seepage": (
        "Rate covers leak-path sealing and waterproofing. Drainage correction "
        "or tanking design, if required, is a separate engineer-designed item."
    ),
    "peeling_paint": (
        "Finish repair only: surface preparation, primer and finish coats. "
        "No structural content. If the substrate is damp, treat that first as "
        "a separate item or the coating will fail again."
    ),
    "rust_staining": (
        "Rate covers stain cleaning and (from Moderate) a cover-meter / "
        "half-cell survey to locate the corroding steel. If the survey finds "
        "section loss, re-cost the area as exposed reinforcement."
    ),
}


# ---------------------------------------------------------------------------
# Quantity estimation functions
# ---------------------------------------------------------------------------

def estimate_crack_quantity(
    crack_length_mm: float,
    severity_level: str,
) -> QuantityEstimate:
    """Estimate repair quantity for a crack based on its length and severity.

    The base quantity is the crack length in running metres.  For severe and
    critical cracks, an effective-length factor is applied because crack
    stitching and structural repair extend beyond the visible crack (drilling
    ports, stitching bars on either side, etc.).

    Parameters
    ----------
    crack_length_mm
        Length of the crack in millimetres (from bounding box or measurement).
    severity_level
        Severity label: Minor, Moderate, Severe, Critical.
    """
    base_length_m = crack_length_mm / 1000.0

    # Effective length factor: severe/critical repairs extend beyond the crack
    effective_factor = {
        "minor": 1.0,
        "moderate": 1.0,
        "severe": 1.2,      # stitching bars extend on either side
        "critical": 1.5,    # strengthening zone is wider
    }.get(_severity_key(severity_level), 1.0)

    effective_length = base_length_m * effective_factor

    desc = (
        f"Crack length {base_length_m:.2f} m"
        + (f" x {effective_factor} effective factor = {effective_length:.2f} m"
           if effective_factor > 1.0
           else "")
    )

    return QuantityEstimate(
        value=effective_length,
        unit=QuantityUnit.RUNNING_METRE,
        description=desc,
    )


def estimate_spalling_quantity(
    area_sq_m: float,
    severity_level: str,
    depth_mm: float | None = None,
) -> QuantityEstimate:
    """Estimate repair quantity for spalling based on area and severity.

    For minor/moderate spalling the quantity is the surface area in sq m.
    For severe/critical spalling, if depth is known, the quantity is expressed
    as volume (cum) = area x depth, because the repair involves removing and
    reinstating concrete to a significant depth.

    Parameters
    ----------
    area_sq_m
        Affected surface area in square metres.
    severity_level
        Severity label.
    depth_mm
        Optional measured spall depth in millimetres.
    """
    level = _severity_key(severity_level)

    if depth_mm is not None and depth_mm > 0 and level in {"severe", "critical"}:
        depth_m = depth_mm / 1000.0
        volume = area_sq_m * depth_m
        return QuantityEstimate(
            value=volume,
            unit=QuantityUnit.CUBIC_METRE,
            description=f"Area {area_sq_m:.2f} sq m x depth {depth_mm:.0f} mm = {volume:.3f} cum",
        )

    # Default: area-based quantity
    return QuantityEstimate(
        value=area_sq_m,
        unit=QuantityUnit.SQUARE_METRE,
        description=f"Spalled area {area_sq_m:.2f} sq m",
    )


def estimate_honeycombing_quantity(
    area_sq_m: float,
    severity_level: str,
    depth_mm: float | None = None,
) -> QuantityEstimate:
    """Estimate repair quantity for honeycombing.

    Similar to spalling: area for minor/moderate, volume for severe/critical
    when depth is known.
    """
    level = _severity_key(severity_level)

    if depth_mm is not None and depth_mm > 0 and level in {"severe", "critical"}:
        depth_m = depth_mm / 1000.0
        volume = area_sq_m * depth_m
        return QuantityEstimate(
            value=volume,
            unit=QuantityUnit.CUBIC_METRE,
            description=f"Area {area_sq_m:.2f} sq m x depth {depth_mm:.0f} mm = {volume:.3f} cum",
        )

    return QuantityEstimate(
        value=area_sq_m,
        unit=QuantityUnit.SQUARE_METRE,
        description=f"Honeycombed area {area_sq_m:.2f} sq m",
    )


def estimate_exposed_rebar_quantity(
    area_sq_m: float,
    severity_level: str,
    cover_mm: float = 40.0,
) -> QuantityEstimate:
    """Estimate repair quantity for exposed reinforcement.

    The quantity is the area of cover restoration in sq m.  For critical cases,
    the effective area is increased because the full bar length must be exposed
    and treated, not just the visible patch.
    """
    level = _severity_key(severity_level)
    effective_factor = 1.5 if level == "critical" else 1.0
    effective_area = area_sq_m * effective_factor

    desc = (
        f"Exposed area {area_sq_m:.2f} sq m"
        + (f" x {effective_factor} (full bar exposure) = {effective_area:.2f} sq m"
           if effective_factor > 1.0
           else "")
        + f"; cover to restore ~{cover_mm:.0f} mm"
    )

    return QuantityEstimate(
        value=effective_area,
        unit=QuantityUnit.SQUARE_METRE,
        description=desc,
    )


# ---------------------------------------------------------------------------
# Cost estimation: Quantity x Rate = Cost
# ---------------------------------------------------------------------------

def _compute_cost(quantity: QuantityEstimate, rate: RateBreakup, notes: str = "") -> CostEstimate:
    """Compute cost breakup from quantity and rate.

    Cost = Quantity x Rate
    Material cost  = Quantity x Material rate
    Labour cost    = Quantity x Labour rate
    Equipment cost = Quantity x Equipment rate
    Total cost     = Quantity x Composite rate
    """
    qty = quantity.value
    return CostEstimate(
        quantity=quantity,
        rate=rate,
        material_cost=qty * rate.material_rate,
        labour_cost=qty * rate.labour_rate,
        equipment_cost=qty * rate.equipment_rate,
        total_cost=qty * rate.composite_rate,
        notes=notes,
    )


def estimate_crack_cost(
    crack_length_mm: float,
    severity_level: str,
) -> CostEstimate:
    """Full cost estimate for crack repair: quantity x rate = cost."""
    quantity = estimate_crack_quantity(crack_length_mm, severity_level)
    rate = _get_crack_rate(severity_level)
    return _compute_cost(
        quantity,
        rate,
        notes="Rate includes sealant/epoxy material, skilled labour, and injection/cleaning equipment.",
    )


def estimate_spalling_cost(
    area_sq_m: float,
    severity_level: str,
    depth_mm: float | None = None,
) -> CostEstimate:
    """Full cost estimate for spalling repair: quantity x rate = cost."""
    quantity = estimate_spalling_quantity(area_sq_m, severity_level, depth_mm)
    rate = _get_spalling_rate(severity_level)
    return _compute_cost(
        quantity,
        rate,
        notes="Rate includes repair mortar/micro-concrete, labour for break-out and reinstatement, and equipment.",
    )


def estimate_honeycombing_cost(
    area_sq_m: float,
    severity_level: str,
    depth_mm: float | None = None,
) -> CostEstimate:
    """Full cost estimate for honeycombing repair: quantity x rate = cost."""
    quantity = estimate_honeycombing_quantity(area_sq_m, severity_level, depth_mm)
    rate = _get_honeycombing_rate(severity_level)
    return _compute_cost(
        quantity,
        rate,
        notes="Rate includes grout/mortar material, labour for chipping and grouting, and pump equipment.",
    )


def estimate_exposed_rebar_cost(
    area_sq_m: float,
    severity_level: str,
    cover_mm: float = 40.0,
) -> CostEstimate:
    """Full cost estimate for exposed reinforcement repair: quantity x rate = cost."""
    quantity = estimate_exposed_rebar_quantity(area_sq_m, severity_level, cover_mm)
    rate = _get_exposed_rebar_rate(severity_level)
    return _compute_cost(
        quantity,
        rate,
        notes="Rate includes corrosion inhibitor, repair mortar, steel treatment, labour, and equipment.",
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def estimate_repair_cost(
    defect_class: str,
    severity_level: str,
    box_width_px: float,
    box_height_px: float,
    image_width_px: float,
    image_height_px: float,
    mm_per_pixel: float | None = None,
    depth_mm: float | None = None,
    cover_mm: float = 40.0,
) -> CostEstimate:
    """Unified cost estimator for any detected defect.

    This is the main entry point called from the detection pipeline.  It:
      1. Converts pixel dimensions to real-world measurements (if scale available).
      2. Estimates the repair quantity in the appropriate unit.
      3. Looks up the rate analysis for the defect type and severity.
      4. Computes Cost = Quantity x Rate with full breakup.

    Parameters
    ----------
    defect_class
        Detected defect type: crack, spalling, honeycombing, exposed_rebar, etc.
    severity_level
        Severity label: Minor, Moderate, Severe, Critical.
    box_width_px, box_height_px
        Detection bounding box dimensions in pixels.
    image_width_px, image_height_px
        Full image dimensions in pixels.
    mm_per_pixel
        Scale reference.  When provided, quantities use real-world mm/m.
        When None, quantities fall back to area-ratio-based estimates.
    depth_mm
        Optional measured defect depth (for spalling/honeycombing).
    cover_mm
        Nominal reinforcement cover (default 40 mm per IS 456).

    Returns
    -------
    CostEstimate
        Full cost estimate with quantity, rate analysis, and cost breakup.
    """
    key = normalise_defect(defect_class)
    has_scale = mm_per_pixel is not None and mm_per_pixel > 0

    def area_sq_m() -> float:
        """Affected surface area in sq m, from scale when available."""
        if has_scale:
            return (box_width_px * mm_per_pixel) * (box_height_px * mm_per_pixel) / 1_000_000.0
        area_ratio = (box_width_px * box_height_px) / (image_width_px * image_height_px)
        # Without a scale reference, assume the frame captures ~2 m x 1.5 m.
        return area_ratio * 3.0

    def length_m() -> float:
        """Defect run length in metres, from scale when available."""
        if has_scale:
            return max(box_width_px, box_height_px) * mm_per_pixel / 1000.0
        area_ratio = (box_width_px * box_height_px) / (image_width_px * image_height_px)
        # Assume the frame captures ~2 m width.
        return (area_ratio ** 0.5) * 2.0

    # --- Crack ---
    if key == "crack":
        return estimate_crack_cost(length_m() * 1000.0, severity_level)

    # --- Stepped masonry crack (measured along its run, like any crack) ---
    if key == "stairstep_crack":
        quantity = estimate_crack_quantity(length_m() * 1000.0, severity_level)
        return _compute_cost(
            quantity,
            _get_stairstep_crack_rate(severity_level),
            notes=(
                "Masonry fabric repair only: bed-joint repointing, and helical "
                "stitching from Severe. Foundation underpinning or drainage "
                "correction is a separate engineer-designed item and is NOT "
                "included in this rate."
            ),
        )

    # --- Spalling ---
    if key == "spalling":
        return estimate_spalling_cost(area_sq_m(), severity_level, depth_mm)

    # --- Honeycombing ---
    if key == "honeycombing":
        return estimate_honeycombing_cost(area_sq_m(), severity_level, depth_mm)

    # --- Mold / dampness ---
    if key == "mold":
        return estimate_honeycombing_cost(area_sq_m(), severity_level, depth_mm)

    # --- Exposed reinforcement ---
    if key == "exposed_reinforcement":
        return estimate_exposed_rebar_cost(area_sq_m(), severity_level, cover_mm)

    # --- Non-structural durability / finish defects ---
    if key in NON_STRUCTURAL_RATE_TABLES:
        table = NON_STRUCTURAL_RATE_TABLES[key]
        area = area_sq_m()
        quantity = QuantityEstimate(
            value=area,
            unit=QuantityUnit.SQUARE_METRE,
            description=f"Affected area {area:.2f} sq m ({key.replace('_', ' ')})",
        )
        rate = table.get(_severity_key(severity_level), table["moderate"])
        return _compute_cost(quantity, rate, notes=NON_STRUCTURAL_NOTES[key])

    # --- Unrecognised defect ---
    # Falling back to a concrete-repair rate here would quote structural repair
    # prices for something the system cannot identify.  Return a make-good and
    # inspection allowance instead, and say plainly that the type is unknown.
    area = area_sq_m()
    quantity = QuantityEstimate(
        value=area,
        unit=QuantityUnit.SQUARE_METRE,
        description=f"Affected area {area:.2f} sq m (defect type not recognised)",
    )
    return _compute_cost(
        quantity,
        UNRECOGNISED_DEFECT_RATE,
        notes=(
            f"'{defect_class}' is not in the supported defect vocabulary, so no "
            "repair method or rate analysis applies. The figure shown is a "
            "make-good and inspection allowance only. A site engineer must "
            "identify the defect and specify the repair before this is costed."
        ),
    )


# ---------------------------------------------------------------------------
# Repair time (duration in working days)
# ---------------------------------------------------------------------------
# Duration is derived the way a site engineer would: labour man-days (from the
# BOQ norms x measured quantity) divided by the crew size, plus a curing /
# mobilisation allowance that grows with severity. When BOQ man-days are not
# available, a severity-band fallback is used. All values are working days.

_CURE_DAYS = {"minor": 0.5, "moderate": 1.0, "severe": 2.0, "critical": 3.0}
_FALLBACK_DAYS = {"minor": 1, "moderate": 2, "severe": 5, "critical": 14}


def estimate_repair_days(
    boq_breakup: dict | None,
    severity_level: str,
    crew_size: int = 2,
) -> int:
    """Estimate repair duration in working days.

    duration = (labour man-days / crew size) + curing/mobilisation allowance

    Parameters
    ----------
    boq_breakup
        The norms-based BOQ dict (may contain labour line quantities in man-days).
    severity_level
        Minor / Moderate / Severe / Critical.
    crew_size
        Number of workers on the repair crew (default 2).
    """
    level = _severity_key(severity_level)
    labour_mandays = 0.0
    if boq_breakup and boq_breakup.get("norms_found"):
        for line in boq_breakup.get("lines", []):
            if line.get("category") == "labour":
                labour_mandays += float(line.get("quantity", 0.0))

    cure = _CURE_DAYS.get(level, 1.0)
    if labour_mandays > 0:
        total = labour_mandays / max(1, crew_size) + cure
    else:
        total = float(_FALLBACK_DAYS.get(level, 3))
    return max(1, round(total))
