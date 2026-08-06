"""Engineering-based defect severity grading.

The grading here follows recognised civil-engineering references rather than an
arbitrary area cut-off:

* Cracks ........... width-based grading per ACI 224R-01 (Table 4.1 tolerable
                     crack widths by exposure) and IS 456:2000 (Cl. 35.3.2,
                     0.3 mm general limit).
* Exposed rebar .... loss of cover always implies active/imminent corrosion, so
                     it is graded "Severe" minimum and "Critical" once the
                     exposure is extensive (ICRI 310.1 / ACI 562 intent).
* Spalling ......... graded by affected surface area and, when depth relative to
                     cover is known, by depth (ICRI 310.1 / Concrete Society
                     TR54). Depth governs over area when available.
* Honeycombing ..... graded by surface extent and depth of voids (IS 456 Cl. 12,
                     ACI 309).

When a scale reference (mm-per-pixel) is supplied, grading uses real-world
millimetre measurements, which is the professional path. Without a scale the
module falls back to a surface-area ratio and clearly flags the result as an
approximation that needs manual measurement to confirm.

The returned result also includes indicative remedial measures, repair-cost
bands and repair-time bands for planning/reporting. These are budget-level
estimates only; final quantities and rates must be confirmed from site
measurements, access constraints, local labour/material rates and a structural
engineer's repair specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from src.boq import BoqEstimate, compute_boq
from src.cost_estimation import CostEstimate, estimate_repair_cost
from src.defect_taxonomy import normalise_defect


class Severity(IntEnum):
    """Ordered condition states so the worst criterion can be selected."""

    NEGLIGIBLE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return {
            Severity.NEGLIGIBLE: "Negligible",
            Severity.MINOR: "Minor",
            Severity.MODERATE: "Moderate",
            Severity.SEVERE: "Severe",
            Severity.CRITICAL: "Critical",
        }[self]


@dataclass(frozen=True)
class RemediationEstimate:
    remedial_measure: str
    repair_cost_estimate: str
    repair_time_estimate: str


# Recommended action per condition state (asset-management style guidance).
RECOMMENDED_ACTION: dict[Severity, str] = {
    Severity.NEGLIGIBLE: "Record only; no remedial action required.",
    Severity.MINOR: "Routine monitoring; cosmetic repair if/when convenient.",
    Severity.MODERATE: "Plan repair; investigate cause and seal against moisture ingress.",
    Severity.SEVERE: "Engage a structural engineer; protect/limit loads and schedule remediation.",
    Severity.CRITICAL: "Immediate structural engineering assessment; consider shoring or closure.",
}


GENERIC_REMEDIATION: dict[Severity, RemediationEstimate] = {
    Severity.NEGLIGIBLE: RemediationEstimate(
        "Record condition and continue normal inspection cycle.",
        "INR 0-100 per inspection item",
        "No repair time",
    ),
    Severity.MINOR: RemediationEstimate(
        "Clean surface and apply local cosmetic/protective repair as needed.",
        "INR 300-1,000 per small repair area",
        "Same day",
    ),
    Severity.MODERATE: RemediationEstimate(
        "Plan targeted repair after confirming cause and extent on site.",
        "INR 1,000-3,500 per repair area",
        "1-3 days",
    ),
    Severity.SEVERE: RemediationEstimate(
        "Engineer-led repair with load/access controls and durability treatment.",
        "INR 3,500-8,000+ per repair area",
        "3-7 days",
    ),
    Severity.CRITICAL: RemediationEstimate(
        "Immediate engineering assessment; provide temporary support before permanent repair.",
        "Engineer estimate required; often INR 8,000+ per repair area",
        "1-3 weeks or more",
    ),
}


REMEDIATION_GUIDE: dict[str, dict[Severity, RemediationEstimate]] = {
    "crack": {
        Severity.MINOR: RemediationEstimate(
            "Seal hairline crack with acrylic/PU sealant after cleaning.",
            "INR 150-400 per running metre",
            "Same day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Route-and-seal or epoxy inject the crack; control moisture ingress.",
            "INR 500-1,500 per running metre",
            "1-2 days",
        ),
        Severity.SEVERE: RemediationEstimate(
            "Structural epoxy injection or crack stitching with cause investigation.",
            "INR 1,500-4,000 per running metre",
            "2-5 days",
        ),
        Severity.CRITICAL: RemediationEstimate(
            "Provide temporary support if required, then engineer-designed strengthening/repair.",
            "Engineer estimate required; often INR 5,000+ per running metre",
            "1-2 weeks or more",
        ),
    },
    "spalling": {
        Severity.MINOR: RemediationEstimate(
            "Remove loose concrete and patch with polymer-modified repair mortar.",
            "INR 800-1,500 per sq m",
            "Same day to 1 day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Break out unsound concrete, clean/passivate steel and reinstate cover.",
            "INR 1,500-3,500 per sq m",
            "1-3 days",
        ),
        Severity.SEVERE: RemediationEstimate(
            "Deep patch repair with steel treatment/replacement and micro-concrete/shotcrete.",
            "INR 3,500-8,000 per sq m",
            "3-7 days",
        ),
        Severity.CRITICAL: RemediationEstimate(
            "Engineer-designed section restoration, jacketing or strengthening after making safe.",
            "Engineer estimate required; often INR 8,000+ per sq m",
            "1-3 weeks or more",
        ),
    },
    "honeycombing": {
        Severity.MINOR: RemediationEstimate(
            "Clean voids and fill with non-shrink repair mortar/grout.",
            "INR 700-1,500 per sq m",
            "Same day to 1 day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Chip weak concrete, pressure grout deeper voids and finish with repair mortar.",
            "INR 1,500-3,500 per sq m",
            "1-3 days",
        ),
        Severity.SEVERE: RemediationEstimate(
            "Remove defective concrete and rebuild section with micro-concrete or shotcrete.",
            "INR 3,500-7,500 per sq m",
            "3-7 days",
        ),
        Severity.CRITICAL: RemediationEstimate(
            "Engineer assessment for loss of section; strengthen or recast affected member zone.",
            "Engineer estimate required; often INR 7,500+ per sq m",
            "1-3 weeks or more",
        ),
    },
    "exposed_reinforcement": {
        Severity.SEVERE: RemediationEstimate(
            "Remove loose concrete, clean/passivate exposed steel and restore cover with repair mortar.",
            "INR 3,000-7,000 per sq m",
            "3-7 days",
        ),
        Severity.CRITICAL: RemediationEstimate(
            "Engineer-led corrosion repair: expose full bar length, replace/augment steel and restore section.",
            "Engineer estimate required; often INR 7,000+ per sq m",
            "1-3 weeks or more",
        ),
    },
    "stairstep_crack": {
        Severity.MODERATE: RemediationEstimate(
            "Install tell-tales and monitor movement; rake out and repoint mortar joints once movement is stable.",
            "INR 600-1,500 per running metre",
            "1-3 days",
        ),
        Severity.SEVERE: RemediationEstimate(
            "Investigate differential settlement; helical bed-joint stitching plus repointing after cause is treated.",
            "INR 1,500-4,000 per running metre",
            "3-7 days",
        ),
        Severity.CRITICAL: RemediationEstimate(
            "Foundation investigation and underpinning design before any fabric repair; monitor until movement stops.",
            "Engineer estimate required; often INR 5,000+ per running metre",
            "2-6 weeks or more",
        ),
    },
    "efflorescence": {
        Severity.MINOR: RemediationEstimate(
            "Dry-brush salt deposits, identify the moisture path and apply a breathable water repellent.",
            "INR 120-300 per sq m",
            "Same day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Remove salts, trace and stop the water ingress at source, then apply crystalline waterproofing.",
            "INR 350-900 per sq m",
            "1-3 days",
        ),
    },
    "water_seepage": {
        Severity.MINOR: RemediationEstimate(
            "Trace the source, seal the local entry point and apply surface waterproof coating.",
            "INR 250-600 per sq m",
            "Same day to 1 day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Cementitious crystalline waterproofing after tracing and sealing the leak path.",
            "INR 600-1,500 per sq m",
            "1-3 days",
        ),
        Severity.SEVERE: RemediationEstimate(
            "Injection grouting of the leak path plus a full waterproofing system; engineer input on drainage.",
            "INR 1,500-4,000 per sq m",
            "3-7 days",
        ),
    },
    "peeling_paint": {
        Severity.MINOR: RemediationEstimate(
            "Scrape loose paint, spot-prime and touch up the affected patch.",
            "INR 60-150 per sq m",
            "Same day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Scrape back to sound substrate, treat the underlying dampness, then primer plus two finish coats.",
            "INR 150-400 per sq m",
            "1-3 days",
        ),
    },
    "rust_staining": {
        Severity.MINOR: RemediationEstimate(
            "Clean the stain and record it; check cover depth at the next inspection cycle.",
            "INR 100-250 per sq m",
            "Same day",
        ),
        Severity.MODERATE: RemediationEstimate(
            "Cover-meter / half-cell survey to locate the corroding steel before any cosmetic treatment.",
            "INR 400-1,200 per sq m (including survey)",
            "1-3 days",
        ),
    },
}


# ---------------------------------------------------------------------------
# Condition-state limits by defect type
# ---------------------------------------------------------------------------
# Non-structural defects signal a durability or finish problem, not loss of
# section. Without a cap a large stain would be graded "Critical" purely on
# surface extent, which inflates the critical count and the repair budget.
SEVERITY_CAP: dict[str, Severity] = {
    "efflorescence": Severity.MODERATE,
    "peeling_paint": Severity.MODERATE,
    "rust_staining": Severity.MODERATE,
    "water_seepage": Severity.SEVERE,
}

# Defects that are diagnostic of a structural cause carry a minimum grade even
# when the visible patch is small: a stepped crack through masonry bed joints
# indicates differential settlement regardless of how short it is.
SEVERITY_FLOOR: dict[str, Severity] = {
    "stairstep_crack": Severity.MODERATE,
}

# Classes graded by crack width (ACI 224R / IS 456) when a scale is available.
CRACK_FAMILY = frozenset({"crack", "stairstep_crack"})


def _apply_limits(defect_key: str, severity: Severity) -> tuple[Severity, str]:
    """Clamp a grade to the defect type's condition-state limits.

    Returns the adjusted severity and a note explaining any adjustment (empty
    when the grade was already within limits).
    """
    floor = SEVERITY_FLOOR.get(defect_key)
    if floor is not None and severity < floor:
        return floor, (
            f" Graded at least {floor.label} because this defect type indicates a "
            "structural cause regardless of the visible extent."
        )
    cap = SEVERITY_CAP.get(defect_key)
    if cap is not None and severity > cap:
        return cap, (
            f" Capped at {cap.label}: this is a durability/finish defect, not loss "
            "of section. Grade the underlying cause separately if present."
        )
    return severity, ""


@dataclass(frozen=True)
class SeverityResult:
    level: str
    reason: str
    area_ratio: float
    score: int = 0
    standard: str = ""
    recommended_action: str = ""
    measured: str = ""
    remedial_measure: str = ""
    repair_cost_estimate: str = ""
    repair_time_estimate: str = ""
    # Quantity-based cost estimation (Quantity x Rate = Cost)
    cost_estimate: CostEstimate | None = None
    cost_breakup: dict[str, Any] = field(default_factory=dict)
    # Norms-based BOQ: material/labour/equipment norms and rates retrieved from
    # the RAG norms database; every line cost computed as quantity x rate.
    boq: BoqEstimate | None = None
    boq_breakup: dict[str, Any] = field(default_factory=dict)
@dataclass(frozen=True)
class CrackDimensions:
    """Real-world crack size, derived only when a scale reference is supplied.

    These come from the detected bounding box, so they are an upper bound on the
    real crack: the box is wider than the crack itself. True crack-width grading
    needs pixel-level segmentation, not a rectangle.
    """

    length_mm: float
    width_mm: float
    mm_per_pixel: float


MINOR_AREA_RATIO = 0.02
MODERATE_AREA_RATIO = 0.08

# ---------------------------------------------------------------------------
# Crack-width grading (millimetres) -- ACI 224R-01 / IS 456:2000
# ---------------------------------------------------------------------------
# ACI 224R-01 Table 4.1 tolerable crack widths and IS 456:2000 Cl. 35.3.2
# (0.3 mm general surface limit) define the engineering bands below. Width is
# the governing parameter for crack severity in practice.
CRACK_WIDTH_BANDS_MM: tuple[tuple[float, Severity], ...] = (
    (0.10, Severity.MINOR),      # < 0.1 mm: hairline, durability not affected
    (0.30, Severity.MODERATE),   # 0.1-0.3 mm: within IS 456 general limit, monitor
    (0.70, Severity.SEVERE),     # 0.3-0.7 mm: exceeds code limit, corrosion risk
)                                # > 0.7 mm: Critical (structural concern)

# ---------------------------------------------------------------------------
# Area-ratio fallback bands (fraction of image area) when no scale is provided.
# (minor_max, moderate_max, severe_max). Above severe_max => Critical.
# Defects implying material/volume loss are graded more conservatively.
# ---------------------------------------------------------------------------
DEFECT_AREA_BANDS: dict[str, tuple[float, float, float]] = {
    "crack": (0.02, 0.08, 0.20),
    "spalling": (0.015, 0.06, 0.15),
    "honeycombing": (0.015, 0.06, 0.15),
    "exposed_reinforcement": (0.01, 0.05, 0.12),
    "mold": (0.02, 0.08, 0.20),
    # Stepped masonry cracking: settlement indicator, so graded conservatively.
    "stairstep_crack": (0.01, 0.04, 0.12),
    # Surface staining needs to be extensive before it changes the condition
    # state, because the deposit itself carries no load.
    "efflorescence": (0.03, 0.12, 0.30),
    "peeling_paint": (0.05, 0.20, 0.40),
    "water_seepage": (0.02, 0.08, 0.20),
    # A small rust stain already implies steel is corroding behind the cover.
    "rust_staining": (0.01, 0.04, 0.10),
}
DEFAULT_AREA_BANDS = (MINOR_AREA_RATIO, MODERATE_AREA_RATIO, 0.20)


def _normalise(defect_class: str) -> str:
    return normalise_defect(defect_class)


def _remediation_key(defect_class: str) -> str:
    return normalise_defect(defect_class)


def estimate_remediation(defect_class: str, severity: Severity) -> RemediationEstimate:
    guide = REMEDIATION_GUIDE.get(_remediation_key(defect_class), {})
    return guide.get(severity, GENERIC_REMEDIATION[severity])


def _display_cost_breakup(cost_estimate: CostEstimate, boq: BoqEstimate) -> dict[str, Any]:
    """Return one reconciled estimate for user-facing outputs.

    The initial estimator is useful for selecting a repair quantity, but the
    BOQ is the authoritative displayed estimate because it derives every line
    from the retrieved norms and includes overheads and GST.
    """
    if not boq.norms_found or boq.work_quantity <= 0:
        return cost_estimate.to_dict()

    quantity = boq.work_quantity
    material_rate = boq.material_total / quantity
    labour_rate = boq.labour_total / quantity
    equipment_rate = boq.equipment_total / quantity
    base_rate = boq.subtotal / quantity
    final_rate = boq.grand_total / quantity
    return {
        "quantity": f"{quantity:.2f} {boq.work_unit}",
        "quantity_value": quantity,
        "quantity_unit": boq.work_unit,
        "quantity_description": "BOQ work quantity used to calculate all norms-based line items.",
        "material_rate": material_rate,
        "labour_rate": labour_rate,
        "equipment_rate": equipment_rate,
        "base_rate": base_rate,
        "composite_rate": final_rate,
        "material_cost": boq.material_total,
        "labour_cost": boq.labour_total,
        "equipment_cost": boq.equipment_total,
        "subtotal": boq.subtotal,
        "overheads": boq.overheads,
        "gst": boq.gst,
        "total_cost": boq.grand_total,
        "notes": "Norms-based BOQ total including overheads and GST. Every BOQ line equals quantity x rate.",
    }


def _thresholds_for(defect_class: str) -> tuple[float, float]:
    """Back-compat helper: (minor_max, moderate_max) area thresholds."""
    minor_max, moderate_max, _ = DEFECT_AREA_BANDS.get(_normalise(defect_class), DEFAULT_AREA_BANDS)
    return minor_max, moderate_max


def _grade_from_area(defect_class: str, area_ratio: float) -> Severity:
    minor_max, moderate_max, severe_max = DEFECT_AREA_BANDS.get(_normalise(defect_class), DEFAULT_AREA_BANDS)
    if area_ratio < minor_max:
        return Severity.MINOR
    if area_ratio < moderate_max:
        return Severity.MODERATE
    if area_ratio < severe_max:
        return Severity.SEVERE
    return Severity.CRITICAL


def _grade_crack_width(width_mm: float) -> Severity:
    """Grade a crack from its width in mm using ACI 224R / IS 456 bands."""
    for upper, severity in CRACK_WIDTH_BANDS_MM:
        if width_mm < upper:
            return severity
    return Severity.CRITICAL


def mm_per_pixel_from_reference(reference_size_mm: float, reference_size_px: float) -> float:
    """Convert a known real-world size and its pixel size into a mm-per-pixel scale.

    Provide a reference object of known size in the photo (e.g. a 100 mm marker that
    spans 250 px) to anchor pixel measurements to millimetres.
    """
    if reference_size_mm <= 0 or reference_size_px <= 0:
        raise ValueError("Reference sizes must be positive")
    return reference_size_mm / reference_size_px


def estimate_crack_dimensions(
    box_width: float,
    box_height: float,
    mm_per_pixel: float,
) -> CrackDimensions:
    """Estimate crack length and width in millimetres from the bounding box.

    Only meaningful when `mm_per_pixel` was derived from a scale reference in the
    same photo. The crack runs along the longer side of the box (length); the
    shorter side is treated as the (over-estimated) width.
    """
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive")

    length_px = max(box_width, box_height)
    width_px = min(box_width, box_height)
    return CrackDimensions(
        length_mm=length_px * mm_per_pixel,
        width_mm=width_px * mm_per_pixel,
        mm_per_pixel=mm_per_pixel,
    )


def _grade_exposed_reinforcement(area_ratio: float) -> tuple[Severity, str, str]:
    """Exposed steel = loss of cover => active corrosion risk by definition.

    Per ICRI 310.1 / ACI 562 intent, any exposed reinforcement is at least a
    serious durability defect; extensive exposure is critical.
    """
    severity = Severity.CRITICAL if area_ratio >= 0.05 else Severity.SEVERE
    reason = (
        "Reinforcement exposed: loss of concrete cover exposes steel to active "
        "corrosion. Carbonation/chloride ingress and section loss are likely."
    )
    return severity, reason, "ICRI 310.1 / ACI 562"


def _grade_spalling(area_ratio: float, depth_mm: float | None, cover_mm: float | None) -> tuple[Severity, str, str]:
    """Grade spalling by surface extent, escalated by depth relative to cover."""
    severity = _grade_from_area("spalling", area_ratio)
    reason = (
        f"Spalled/delaminated concrete over ~{area_ratio * 100:.1f}% of the "
        "captured surface; check for underlying corrosion and delamination."
    )
    standard = "ICRI 310.1 / Concrete Society TR54"
    if depth_mm is not None:
        # Depth governs: once spall depth reaches the reinforcement cover, steel
        # is effectively exposed and the defect is critical.
        if cover_mm and depth_mm >= cover_mm:
            severity = max(severity, Severity.CRITICAL)
            reason = (
                f"Spall depth ~{depth_mm:.0f} mm reaches the reinforcement cover "
                f"(~{cover_mm:.0f} mm): steel is effectively exposed."
            )
        elif depth_mm >= 25.0:
            severity = max(severity, Severity.SEVERE)
            reason = f"Deep spall (~{depth_mm:.0f} mm); significant section loss."
    return severity, reason, standard


def _grade_honeycombing(area_ratio: float, depth_mm: float | None) -> tuple[Severity, str, str]:
    """Grade honeycombing by surface extent, escalated by void depth."""
    severity = _grade_from_area("honeycombing", area_ratio)
    reason = (
        f"Honeycombing (voids/poor compaction) over ~{area_ratio * 100:.1f}% of "
        "the surface; reduces cover, strength and durability."
    )
    standard = "IS 456:2000 Cl. 12 / ACI 309"
    if depth_mm is not None and depth_mm >= 25.0:
        severity = max(severity, Severity.SEVERE)
        reason = f"Deep honeycombing (~{depth_mm:.0f} mm) into the section; structural concern."
    return severity, reason, standard


def estimate_severity(
    defect_class: str,
    box_width: float,
    box_height: float,
    image_width: float,
    image_height: float,
    mm_per_pixel: float | None = None,
    depth_mm: float | None = None,
    cover_mm: float | None = 40.0,
    exposure: str = "general",
) -> SeverityResult:
    """Grade a detected defect using civil-engineering standards.

    Parameters
    ----------
    defect_class
        Detected class, e.g. ``crack``, ``spalling``, ``honeycombing``,
        ``exposed_reinforcement``.
    box_width, box_height, image_width, image_height
        Detection bounding box and image size in pixels.
    mm_per_pixel
        Scale from a known reference in the photo (see
        :func:`mm_per_pixel_from_reference`). When provided, cracks are graded by
        real width (ACI 224R / IS 456) instead of by area ratio.
    depth_mm
        Optional measured depth (spalling/honeycombing) when available.
    cover_mm
        Nominal reinforcement cover used to decide when a spall exposes steel
        (IS 456 typically 40 mm for general exposure).
    exposure
        Exposure condition; reserved for future per-exposure crack limits.

    Notes
    -----
    Vision gives an upper-bound bounding box, not a true crack width, so even the
    millimetre path is conservative. Final assessment still needs site
    measurement (crack gauge, depth probe) and inspection standards.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    area_ratio = (box_width * box_height) / (image_width * image_height)
    key = _normalise(defect_class)
    defect_name = defect_class.replace("_", " ")
    measured = "area-ratio (no scale reference; approximate)"
    standard = "Area-ratio heuristic"

    if key == "exposed_reinforcement":
        severity, reason, standard = _grade_exposed_reinforcement(area_ratio)

    elif key == "spalling":
        severity, reason, standard = _grade_spalling(area_ratio, depth_mm, cover_mm)
        if depth_mm is not None:
            measured = f"depth {depth_mm:.0f} mm + area-ratio"

    elif key == "honeycombing":
        severity, reason, standard = _grade_honeycombing(area_ratio, depth_mm)
        if depth_mm is not None:
            measured = f"depth {depth_mm:.0f} mm + area-ratio"

    elif key in CRACK_FAMILY and mm_per_pixel:
        dims = estimate_crack_dimensions(box_width, box_height, mm_per_pixel)
        severity = _grade_crack_width(dims.width_mm)
        reason = (
            f"Crack width ~{dims.width_mm:.2f} mm (length ~{dims.length_mm:.0f} mm). "
            "Graded against ACI 224R / IS 456 0.3 mm limit. Width from the box is an "
            "upper bound; confirm with a crack gauge."
        )
        standard = "ACI 224R-01 / IS 456:2000 Cl. 35.3.2"
        measured = f"width {dims.width_mm:.2f} mm (scaled)"

    elif key == "rust_staining":
        severity = _grade_from_area(key, area_ratio)
        reason = (
            f"Rust/corrosion staining over ~{area_ratio * 100:.1f}% of the surface. "
            "Staining is leachate from reinforcement already corroding behind intact "
            "cover: an early-warning indicator, not yet section loss."
        )
        standard = "ICRI 310.1 (corrosion indicators) / IS 456:2000 Cl. 8"

    elif key == "efflorescence":
        severity = _grade_from_area(key, area_ratio)
        reason = (
            f"Efflorescence (salt/lime leaching) over ~{area_ratio * 100:.1f}% of the "
            "surface. Indicates water moving through the concrete and evaporating at "
            "the face; a durability and moisture-path issue."
        )
        standard = "IS 456:2000 Cl. 8 (durability) / BS 8102 intent"

    elif key == "water_seepage":
        severity = _grade_from_area(key, area_ratio)
        reason = (
            f"Active water seepage/damp patch over ~{area_ratio * 100:.1f}% of the "
            "surface. Sustained moisture drives corrosion of embedded steel and "
            "supports mould growth; treat the source, not just the face."
        )
        standard = "IS 456:2000 Cl. 8 / BS 8102 (water-resisting construction)"

    elif key == "peeling_paint":
        severity = _grade_from_area(key, area_ratio)
        reason = (
            f"Paint/finish delamination over ~{area_ratio * 100:.1f}% of the surface. "
            "A finish defect with no structural significance, though it commonly "
            "signals dampness in the substrate."
        )
        standard = "Finish/serviceability (non-structural)"

    else:
        severity = _grade_from_area(defect_class, area_ratio)
        reason = (
            f"{defect_name.capitalize()} covering ~{area_ratio * 100:.1f}% of the "
            f"captured surface ({severity.label.lower()} by surface extent)."
        )

    severity, limit_note = _apply_limits(key, severity)
    reason += limit_note

    remediation = estimate_remediation(defect_class, severity)

    # --- Quantity-based cost estimation (Quantity x Rate = Cost) ---
    cost_estimate = estimate_repair_cost(
        defect_class=defect_class,
        severity_level=severity.label,
        box_width_px=box_width,
        box_height_px=box_height,
        image_width_px=image_width,
        image_height_px=image_height,
        mm_per_pixel=mm_per_pixel,
        depth_mm=depth_mm,
        cover_mm=cover_mm if cover_mm is not None else 40.0,
    )

    # --- Norms-based BOQ (norms + rates retrieved, cost computed qty x rate) ---
    # The BOQ work quantity must match the norms' work unit: running metre for
    # cracks, surface area (sq m) for spalling/honeycombing/exposed steel.  The
    # cost module may express deep repairs in cum, so recompute the area here.
    if key in CRACK_FAMILY:
        boq_work_quantity = cost_estimate.quantity.value  # running metre
    elif mm_per_pixel:
        boq_work_quantity = (box_width * mm_per_pixel) * (box_height * mm_per_pixel) / 1_000_000.0
    else:
        boq_work_quantity = area_ratio * 3.0  # same fallback as cost estimation
    boq = compute_boq(
        defect_class=defect_class,
        severity_level=severity.label,
        work_quantity=boq_work_quantity,
    )
    display_cost_breakup = _display_cost_breakup(cost_estimate, boq)
    display_rate = display_cost_breakup["composite_rate"]
    display_total = display_cost_breakup["total_cost"]

    # Only claim a "BOQ total" when the norms database actually backed it. Without
    # a norms record the figure is a rate-table estimate with no overheads/GST, so
    # labelling it as a BOQ would overstate how the number was derived.
    if boq.norms_found:
        cost_line = (
            f"BOQ total: INR {display_total:.0f} | "
            f"Final rate: INR {display_rate:.0f}/{display_cost_breakup['quantity_unit']} "
            f"(includes overheads and GST)"
        )
    else:
        cost_line = (
            f"Indicative estimate: INR {display_total:.0f} | "
            f"Rate: INR {display_rate:.0f}/{display_cost_breakup['quantity_unit']} "
            f"(no norms record for this defect; excludes overheads and GST -- "
            f"engineer estimate required)"
        )

    return SeverityResult(
        level=severity.label,
        reason=reason,
        area_ratio=area_ratio,
        score=int(severity),
        standard=standard,
        recommended_action=RECOMMENDED_ACTION[severity],
        measured=measured,
        remedial_measure=remediation.remedial_measure,
        repair_cost_estimate=cost_line,
        repair_time_estimate=remediation.repair_time_estimate,
        cost_estimate=cost_estimate,
        cost_breakup=display_cost_breakup,
        boq=boq,
        boq_breakup=boq.to_dict(),
    )
