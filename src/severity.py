from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeverityResult:
    level: str
    reason: str
    area_ratio: float


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

# Per-defect area thresholds (minor_max, moderate_max) as a fraction of image area.
# Spalling and honeycombing are graded more conservatively because they imply
# material/volume loss, so smaller visible areas already warrant attention.
DEFECT_AREA_THRESHOLDS: dict[str, tuple[float, float]] = {
    "crack": (0.02, 0.08),
    "spalling": (0.015, 0.06),
    "honeycombing": (0.015, 0.06),
    "exposed_reinforcement": (0.01, 0.05),
}


def _thresholds_for(defect_class: str) -> tuple[float, float]:
    key = defect_class.strip().lower().replace(" ", "_")
    return DEFECT_AREA_THRESHOLDS.get(key, (MINOR_AREA_RATIO, MODERATE_AREA_RATIO))


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


def estimate_severity(defect_class: str, box_width: float, box_height: float, image_width: float, image_height: float) -> SeverityResult:
    """Estimate visual defect severity from the detected bounding-box area.

    This is a practical demo-level approximation. Final civil assessment should use
    manual measurements such as crack width, spalling depth, cavity depth, exposed
    reinforcement length, corrosion level, and site inspection standards.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    area_ratio = (box_width * box_height) / (image_width * image_height)
    defect_name = defect_class.replace("_", " ")
    minor_max, moderate_max = _thresholds_for(defect_class)

    if area_ratio < minor_max:
        return SeverityResult(
            level="Minor",
            reason=f"Small localized {defect_name} region detected.",
            area_ratio=area_ratio,
        )

    if area_ratio < moderate_max:
        return SeverityResult(
            level="Moderate",
            reason=f"Visible {defect_name} region covering a moderate surface area.",
            area_ratio=area_ratio,
        )

    return SeverityResult(
        level="Severe",
        reason=f"Large {defect_name} region detected; detailed structural inspection is recommended.",
        area_ratio=area_ratio,
    )
