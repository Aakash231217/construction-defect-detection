"""Shared defect-analysis pipeline used by both the Streamlit app and the report.

One image in -> graded detections + structural element out, using:

    1. the Roboflow detection model (src/roboflow_model.py) as the primary detector,
    2. an AI vision detector fallback (src/vision_fallback.py) when Roboflow finds
       nothing,
    3. optional expert/manual annotations (data/manual_detections.json),
    4. the severity engine (src/severity.py) to grade every detection, and
    5. an AI structural-element classifier (slab/wall/beam/column/staircase...).

Keeping this in one module means the deployed app and the PDF report always show
the same results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.roboflow_model import run_model, RoboflowModelError  # noqa: F401
from src.vision_fallback import detect_with_gpt, classify_structural_element
from src.severity import estimate_severity
from src.cost_estimation import estimate_repair_days
from src.defect_taxonomy import normalise_defect, preferred_label, same_family

ROOT = Path(__file__).resolve().parents[1]
MANUAL_PATH = ROOT / "data" / "manual_detections.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEV_ORDER = ["Critical", "Severe", "Moderate", "Minor", "Negligible"]

# Severity -> (fill, edge) hex for chips/UI, and RGB for drawn boxes.
SEV_COLORS = {
    "Negligible": ("#eef1f6", "#7a8698"),
    "Minor": ("#e7f7ee", "#2f9e63"),
    "Moderate": ("#fff3e0", "#e08a1e"),
    "Severe": ("#ffe7d6", "#e2691a"),
    "Critical": ("#ffe0e6", "#d83a52"),
}
SEV_RGB = {
    "Negligible": (122, 134, 152),
    "Minor": (47, 158, 99),
    "Moderate": (224, 138, 30),
    "Severe": (226, 105, 26),
    "Critical": (216, 58, 82),
}
SOURCE_HEX = {"Roboflow": "#3b74d4", "Manual review": "#1f97a8"}
DEFAULT_SOURCE_HEX = "#7a54d0"  # AI vision detector


def source_color(source: str) -> str:
    return SOURCE_HEX.get(source, DEFAULT_SOURCE_HEX)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    for attr in ("dict", "model_dump"):
        fn = getattr(result, attr, None)
        if callable(fn):
            return fn()
    return dict(result)  # last resort


def _prediction_bounds(prediction: dict[str, Any]) -> tuple[float, float, float, float]:
    width = float(prediction.get("width", 0.0))
    height = float(prediction.get("height", 0.0))
    center_x = float(prediction.get("x", 0.0))
    center_y = float(prediction.get("y", 0.0))
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def _prediction_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x1, left_y1, left_x2, left_y2 = _prediction_bounds(left)
    right_x1, right_y1, right_x2, right_y2 = _prediction_bounds(right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def normalise_detection_class(value: Any) -> str:
    return normalise_defect(value)


def merge_predictions(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Supplement primary detections, dropping overlaps within a defect family.

    Overlap suppression is by *family*, not by exact class name. One wall crack
    can come back as ``crack`` from the Roboflow model and ``stairstep_crack``
    from the vision fallback; the same damp patch can come back as both
    ``water_seepage`` and ``mold``. Keeping both would put two line items and
    two repair costs against one physical defect.

    When a family overlap is found the more specific diagnosis wins (see
    :func:`~src.defect_taxonomy.preferred_label`), because it carries the more
    useful remedy, and the surviving detection keeps the higher confidence of
    the two.
    """
    merged: list[dict[str, Any]] = []
    for prediction in primary:
        normalized = dict(prediction)
        normalized["class"] = normalise_detection_class(prediction.get("class"))
        merged.append(normalized)

    for candidate in secondary:
        normalized_candidate = dict(candidate)
        candidate_class = normalise_detection_class(candidate.get("class"))
        normalized_candidate["class"] = candidate_class

        overlap_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if same_family(existing.get("class", ""), candidate_class)
                and _prediction_iou(existing, normalized_candidate) >= iou_threshold
            ),
            None,
        )
        if overlap_index is None:
            merged.append(normalized_candidate)
            continue

        existing = merged[overlap_index]
        existing_class = str(existing.get("class", ""))
        if existing_class == candidate_class:
            continue  # exact duplicate: keep the primary detection unchanged

        kept_label = preferred_label(existing_class, candidate_class)
        existing_conf = float(existing.get("confidence", 0.0))
        candidate_conf = float(normalized_candidate.get("confidence", 0.0))
        winner = dict(existing if existing_conf >= candidate_conf else normalized_candidate)
        winner["class"] = kept_label
        winner["confidence"] = max(existing_conf, candidate_conf)
        winner["merged_from"] = sorted({existing_class, candidate_class} - {kept_label})
        merged[overlap_index] = winner

    return merged


def detect_roboflow(image_path: str | Path, conf: float) -> tuple[list[dict[str, Any]], int, int]:
    """Return (predictions, image_width, image_height) from the Roboflow model."""
    image_path = Path(image_path)
    raw = _as_dict(run_model(str(image_path)))
    image_block = raw.get("image", {}) or {}
    width = int(image_block.get("width", 0) or 0)
    height = int(image_block.get("height", 0) or 0)
    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size

    preds = []
    for p in raw.get("predictions", []) or []:
        if float(p.get("confidence", 0)) < conf:
            continue
        p["source"] = "Roboflow"
        preds.append(p)
    return preds, width, height


def load_manual(image_name: str, w: int, h: int) -> list[dict[str, Any]]:
    """Expert annotations for images the detectors miss (source = 'Manual review')."""
    if not MANUAL_PATH.exists():
        return []
    try:
        data = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    preds = []
    for item in data.get(image_name, []):
        box = item.get("box", {})
        try:
            x1, y1 = float(box["x_min"]) * w, float(box["y_min"]) * h
            x2, y2 = float(box["x_max"]) * w, float(box["y_max"]) * h
        except (KeyError, TypeError, ValueError):
            continue
        preds.append({
            "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
            "width": abs(x2 - x1), "height": abs(y2 - y1),
            "confidence": float(item.get("confidence", 1.0)),
            "class": str(item.get("defect_class", "defect")),
            "source": "Manual review",
        })
    return preds


def grade(preds: list[dict[str, Any]], w: int, h: int,
          mm_per_pixel: float | None = None) -> list[dict[str, Any]]:
    """Attach a severity result to every prediction (worst-first)."""
    graded = []
    for p in preds:
        bw, bh = float(p["width"]), float(p["height"])
        defect = str(p.get("class", "defect"))
        sev = estimate_severity(
            defect_class=defect, box_width=bw, box_height=bh,
            image_width=w, image_height=h, mm_per_pixel=mm_per_pixel,
        )
        cx, cy = float(p["x"]), float(p["y"])
        graded.append({
            "defect": defect,
            "confidence": float(p.get("confidence", 0.0)),
            "source": str(p.get("source", "Roboflow")),
            "box": (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
            "severity": sev.level,
            "score": int(getattr(sev, "score", 0)),
            "measured": sev.measured,
            "reason": sev.reason,
            "action": sev.recommended_action,
            "standard": sev.standard,
            "area_pct": sev.area_ratio * 100.0,
            # richer fields used by the app (remedy / cost / time / BOQ)
            "remedial_measure": sev.remedial_measure,
            "repair_time_estimate": sev.repair_time_estimate,
            "repair_days": estimate_repair_days(sev.boq_breakup, sev.level),
            "total_cost": float((sev.cost_breakup or {}).get("total_cost", 0.0)),
            "material_cost": float((sev.cost_breakup or {}).get("material_cost", 0.0)),
            "labour_cost": float((sev.cost_breakup or {}).get("labour_cost", 0.0)),
            "equipment_cost": float((sev.cost_breakup or {}).get("equipment_cost", 0.0)),
            "cost_breakup": sev.cost_breakup,
            "boq_breakup": sev.boq_breakup,
        })
    graded.sort(key=lambda d: (SEV_ORDER.index(d["severity"]) if d["severity"] in SEV_ORDER
                               else 99, -d["confidence"]))
    return graded


def annotate(image_path: str | Path, graded: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Draw severity-coloured boxes + numbered badges keyed to the detection list."""
    image_path, out_path = Path(image_path), Path(out_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    scale = max(img.width, img.height) / 900.0
    lw = max(3, int(4 * scale))
    fs = max(15, int(20 * scale))
    font = _font(fs)

    badge = max(26, int(34 * scale))
    for i, g in enumerate(graded, start=1):
        x1, y1, x2, y2 = g["box"]
        rgb = SEV_RGB.get(g["severity"], (216, 58, 82))
        draw.rectangle((x1, y1, x2, y2), outline=rgb + (255,), width=lw)
        bx = min(max(0, x1), img.width - badge)
        by = min(max(0, y1), img.height - badge)
        draw.rectangle((bx, by, bx + badge, by + badge), fill=rgb + (255,),
                       outline=(255, 255, 255, 255), width=max(2, int(2 * scale)))
        num = str(i)
        tb = draw.textbbox((0, 0), num, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text((bx + (badge - tw) / 2 - tb[0], by + (badge - th) / 2 - tb[1]),
                  num, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)
    return out_path


def classify_element(image_path: str | Path, *, model: str = "gpt-4o") -> dict[str, Any]:
    """Structural element (slab/wall/beam/column/staircase...) or 'unknown' on failure."""
    try:
        return classify_structural_element(image_path, model=model)
    except Exception:
        return {"element": "unknown", "confidence": 0.0, "reason": ""}


def analyze(
    image_path: str | Path,
    *,
    conf: float = 0.25,
    use_fallback: bool = True,
    do_element: bool = True,
    mm_per_pixel: float | None = None,
    gpt_model: str = "gpt-4o",
) -> dict[str, Any]:
    """Full analysis for one image.

    Returns ``{width, height, graded, element, roboflow_empty}``.
    """
    image_path = Path(image_path)
    roboflow_empty = False
    try:
        preds, w, h = detect_roboflow(image_path, conf)
    except Exception:
        preds, w, h = [], 0, 0
    roboflow_empty = not preds
    if use_fallback:
        try:
            secondary_preds, secondary_width, secondary_height = detect_with_gpt(
                image_path,
                model=gpt_model,
            )
            preds = merge_predictions(preds, secondary_preds)
            w = w or secondary_width
            h = h or secondary_height
        except Exception:
            pass
    if not w or not h:
        with Image.open(image_path) as im:
            w, h = im.size

    preds = list(preds) + load_manual(image_path.name, w, h)
    graded = grade(preds, w, h, mm_per_pixel=mm_per_pixel)

    element = classify_element(image_path, model=gpt_model) if do_element else \
        {"element": "unknown", "confidence": 0.0, "reason": ""}

    return {"width": w, "height": h, "graded": graded, "element": element,
            "roboflow_empty": roboflow_empty}
