"""Detection + Severity evidence report.

For every image in a folder this script:

    1. runs the LIVE Roboflow detector (src/roboflow_model.py),
    2. grades each detection with the severity engine (src/severity.py),
    3. draws severity-coloured bounding boxes on the photo, and
    4. lays out one card per image in a PDF (annotated photo + per-defect
       severity results), preceded by a summary page, and writes a CSV.

Detection + severity only (no RAG / cost), area-based severity (no scale card).

Run:
    python scripts/generate_detection_severity_report.py \
        --images report_images --out outputs/detection_severity_report.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.roboflow_model import run_model, RoboflowModelError  # noqa: E402
from src.severity import estimate_severity  # noqa: E402
from src.cost_estimation import estimate_repair_days  # noqa: E402
from src.vision_fallback import detect_with_gpt, classify_structural_element  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INK = "#1b2433"
MUTED = "#5b6675"

# Severity -> (fill, edge) for chips and boxes
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


def detect(image_path: Path, conf: float) -> tuple[list[dict[str, Any]], int, int]:
    """Return (predictions, image_width, image_height) from the Roboflow model."""
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


MANUAL_PATH = ROOT / "data" / "manual_detections.json"


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


def grade(preds: list[dict[str, Any]], w: int, h: int) -> list[dict[str, Any]]:
    """Attach a severity result to every prediction."""
    graded = []
    for p in preds:
        bw, bh = float(p["width"]), float(p["height"])
        defect = str(p.get("class", "defect"))
        sev = estimate_severity(
            defect_class=defect, box_width=bw, box_height=bh,
            image_width=w, image_height=h,
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
            "total_cost": float((sev.cost_breakup or {}).get("total_cost", 0.0)),
            "repair_days": estimate_repair_days(sev.boq_breakup, sev.level),
            "repair_band": sev.repair_time_estimate,
        })
    # worst first
    order = ["Critical", "Severe", "Moderate", "Minor", "Negligible"]
    graded.sort(key=lambda d: (order.index(d["severity"]) if d["severity"] in order else 99,
                               -d["confidence"]))
    return graded


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------
def annotate(image_path: Path, graded: list[dict[str, Any]], out_path: Path) -> Path:
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
        # numbered badge at the top-left corner (keys to the cards on the right)
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


# ---------------------------------------------------------------------------
# PDF drawing helpers
# ---------------------------------------------------------------------------
def _bg_axes(fig, title: str, subtitle: str):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 63.2), 100, 6.8, boxstyle="square,pad=0",
                                facecolor=INK, edgecolor="none", zorder=1))
    ax.text(3, 66.9, title, fontsize=19, fontweight="bold", color="white",
            va="center", zorder=2)
    ax.text(3, 64.4, subtitle, fontsize=10, color="#b9c4d6", va="center", zorder=2)
    ax.text(97, 65.6, "Construction Defect Detection  •  Detection + Severity",
            fontsize=8.5, color="#8492a8", va="center", ha="right", zorder=2)
    return ax


def _footer(ax, page_no: int, note: str) -> None:
    ax.plot([3, 97], [3.2, 3.2], color="#d5dbe6", lw=0.8, zorder=1)
    ax.text(3, 2.0, note, fontsize=7.6, color=MUTED, va="center")
    ax.text(97, 2.0, f"Page {page_no}", fontsize=8, color=MUTED, va="center", ha="right")


def _sev_chip(ax, x, y, severity, size=8.5):
    face, edge = SEV_COLORS.get(severity, SEV_COLORS["Critical"])
    ax.add_patch(FancyBboxPatch((x, y - 1.0), 12, 2.4,
                 boxstyle="round,pad=0.1,rounding_size=0.5",
                 facecolor=face, edgecolor=edge, linewidth=1.4, zorder=4))
    ax.text(x + 6, y + 0.2, severity.upper(), fontsize=size, fontweight="bold",
            color=edge, ha="center", va="center", zorder=5)


SEV_SCALE = ["Minor", "Moderate", "Severe", "Critical"]
SEV_SHORT = {"Minor": "Minor", "Moderate": "Mod.", "Severe": "Severe", "Critical": "Crit."}


def _sev_meter(ax, x, y, width, level):
    """4-segment severity scale with the active level filled and labelled."""
    n = len(SEV_SCALE)
    gap = 0.3
    seg = (width - gap * (n - 1)) / n
    for i, name in enumerate(SEV_SCALE):
        sx = x + i * (seg + gap)
        active = name == level
        face, edge = SEV_COLORS[name]
        ax.add_patch(FancyBboxPatch((sx, y), seg, 2.0,
                     boxstyle="round,pad=0.02,rounding_size=0.25",
                     facecolor=edge if active else "#eef1f6",
                     edgecolor=edge if active else "#cdd4de",
                     linewidth=1.2 if active else 0.8, zorder=4))
        ax.text(sx + seg / 2, y + 1.0, SEV_SHORT[name], fontsize=6.4,
                fontweight="bold" if active else "normal",
                color="white" if active else "#9aa4b2",
                ha="center", va="center", zorder=5)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------
def page_summary(pdf: PdfPages, records: list[dict[str, Any]], page_no: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = _bg_axes(fig, "Detection & Severity — Summary",
                  "Every processed image, its defect count and worst-case severity")

    total_def = sum(len(r["graded"]) for r in records)
    sev_tally: dict[str, int] = {}
    for r in records:
        for g in r["graded"]:
            sev_tally[g["severity"]] = sev_tally.get(g["severity"], 0) + 1

    # KPI tiles
    tiles = [("Images", str(len(records)), ("#e8f1ff", "#3b74d4")),
             ("Total detections", str(total_def), ("#f0eaff", "#7a54d0"))]
    for name in ["Critical", "Severe", "Moderate", "Minor"]:
        if sev_tally.get(name):
            tiles.append((name, str(sev_tally[name]), SEV_COLORS[name]))
    tw = 92 / len(tiles)
    for i, (label, value, (face, edge)) in enumerate(tiles):
        x = 4 + i * tw
        ax.add_patch(FancyBboxPatch((x, 52), tw - 2, 8,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3))
        ax.text(x + (tw - 2) / 2, 57.4, value, fontsize=18, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4)
        ax.text(x + (tw - 2) / 2, 53.6, label, fontsize=8.5, color=MUTED,
                ha="center", va="center", zorder=4)

    # Table
    cols = [("#", 4), ("Image", 26), ("Element", 12), ("Defects", 9),
            ("Detected classes", 19), ("Worst severity", 20)]
    x0, y = 4, 46
    cx = x0
    ax.add_patch(FancyBboxPatch((x0, y - 1), 90, 3, boxstyle="square,pad=0",
                 facecolor="#e8eef8", edgecolor="none", zorder=2))
    for name, w in cols:
        ax.text(cx + 0.5, y + 0.5, name, fontsize=9, fontweight="bold",
                color=INK, va="center", zorder=3)
        cx += w
    y -= 3.6
    pitch = min(3.2, (y - 5.0) / max(len(records), 1))
    order = ["Critical", "Severe", "Moderate", "Minor", "Negligible"]
    for idx, r in enumerate(records, start=1):
        graded = r["graded"]
        classes = ", ".join(sorted({g["defect"] for g in graded})) or "-"
        worst_g = min(graded, key=lambda g: order.index(g["severity"])
                      if g["severity"] in order else 99) if graded else None
        worst = worst_g["severity"] if worst_g else "-"
        extent = f"{worst_g['area_pct']:.1f}%" if worst_g else "-"
        if idx % 2 == 0:
            ax.add_patch(FancyBboxPatch((x0, y - 1.1), 90, 3.0, boxstyle="square,pad=0",
                         facecolor="#f6f8fc", edgecolor="none", zorder=1))
        element = (r.get("element") or {}).get("element", "unknown").title()
        cx = x0
        vals = [str(idx), r["name"], element, str(len(graded)), classes, ""]
        for (name, w), val in zip(cols, vals):
            if name == "Image":
                val = val if len(val) <= 30 else val[:27] + "…"
            if name == "Detected classes":
                val = val if len(val) <= 23 else val[:20] + "…"
            ax.text(cx + 0.5, y + 0.4, val, fontsize=8.2, color=MUTED,
                    va="center", zorder=3)
            cx += w
        if worst != "-":
            _sev_chip(ax, cx - 14.5, y + 0.4, worst, size=7.5)
        else:
            ax.text(cx + 0.5, y + 0.4, "no defect", fontsize=8, color=MUTED,
                    va="center", style="italic", zorder=3)
        y -= pitch
        if y < 4.5:
            break

    _footer(ax, page_no, "Detector: Roboflow model + AI vision detector fallback.  Structural element (slab/wall/beam/column/staircase) classified by the AI vision engine.")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image page
# ---------------------------------------------------------------------------
def page_image(pdf: PdfPages, record: dict[str, Any], index: int, total: int,
               page_no: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    disp_name = record["name"] if len(record["name"]) <= 40 else record["name"][:37] + "…"
    elem = (record.get("element") or {}).get("element", "unknown")
    ax = _bg_axes(fig, f"{index}/{total}   {disp_name}",
                  f"Structural element: {elem.title()}   •   {len(record['graded'])} detection(s)"
                  f"   •   {record['width']}×{record['height']} px   •   area-based severity (no scale)")

    # annotated image on the left
    img = plt.imread(str(record["annotated"]))
    iax = fig.add_axes([0.035, 0.10, 0.52, 0.74])
    iax.imshow(img)
    iax.axis("off")
    iax.set_title("Annotated detection (boxes coloured by severity)",
                  fontsize=9, color=MUTED)

    # right column: structural-element banner, then one block per detection
    rx = 60
    edict = record.get("element") or {}
    elem_label = str(edict.get("element", "unknown")).title()
    elem_conf = float(edict.get("confidence", 0.0) or 0.0)
    ax.add_patch(FancyBboxPatch((rx, 59.0), 37, 3.7,
                 boxstyle="round,pad=0.12,rounding_size=0.4",
                 facecolor="#eef4ff", edgecolor="#3b74d4", linewidth=1.5, zorder=3))
    ax.text(rx + 1.4, 61.6, "STRUCTURAL ELEMENT", fontsize=7.0, fontweight="bold",
            color="#3b74d4", va="center", zorder=4)
    ax.text(rx + 1.4, 60.0, elem_label, fontsize=12, fontweight="bold",
            color=INK, va="center", zorder=4)
    conf_txt = f"AI-classified · {elem_conf*100:.0f}%" if elem_conf else "AI-classified"
    ax.text(rx + 35.6, 60.6, conf_txt, fontsize=7.2, color="#5b6675",
            ha="right", va="center", style="italic", zorder=4)

    y = 58.4
    graded = record["graded"]
    if not graded:
        ax.add_patch(FancyBboxPatch((rx, 50), 37, 5, boxstyle="round,pad=0.2,rounding_size=0.4",
                     facecolor="#eef1f6", edgecolor="#7a8698", linewidth=1.4, zorder=3))
        ax.text(rx + 18.5, 52.5, "No defects detected above threshold",
                fontsize=9.5, color=MUTED, ha="center", va="center", zorder=4)
    n = len(graded)
    reason_max, action_max = (3, 2) if n <= 2 else (2, 1)
    for i, g in enumerate(graded, start=1):
        reason_lines = _wrap(g["reason"], 52)[:reason_max]
        action_lines = _wrap(g["action"], 52)[:action_max]
        # title(2.2) + metrics(2.0) + meter(3.0) + reason(1.7+lines) + action(1.7+lines) + pad
        block_h = 2.2 + 2.0 + 3.0 + (1.7 + 1.7 * len(reason_lines)) \
            + (1.7 + 1.7 * len(action_lines)) + 1.4
        if y - block_h < 4.5:
            ax.text(rx, y - 2, "… more detections in the CSV export", fontsize=8,
                    color=MUTED, style="italic", zorder=4)
            break
        face, edge = SEV_COLORS.get(g["severity"], SEV_COLORS["Critical"])
        ax.add_patch(FancyBboxPatch((rx, y - block_h), 37, block_h,
                     boxstyle="round,pad=0.15,rounding_size=0.4",
                     facecolor="white", edgecolor=edge, linewidth=1.5, zorder=3))
        ax.text(rx + 1.2, y - 1.8, f"{i}.  {g['defect']}", fontsize=10.5,
                fontweight="bold", color=INK, va="center", zorder=4)
        _sev_chip(ax, rx + 24, y - 1.8, g["severity"], size=7.5)
        # quantified "how much" line: severity score + affected extent + confidence
        ty = y - 4.2
        src_col = {"Roboflow": "#3b74d4", "Manual review": "#1f97a8"}.get(
            g["source"], "#7a54d0")
        ax.text(rx + 1.2, ty,
                f"Severity {g['score']}/4   •   extent {g['area_pct']:.1f}%"
                f"   •   conf {g['confidence']*100:.0f}%   •   ",
                fontsize=7.6, color=INK, va="center", zorder=4)
        ax.text(rx + 27.6, ty, f"via {g['source']}", fontsize=7.0, color=src_col,
                va="center", style="italic", fontweight="bold", zorder=4)
        # severity meter
        ty -= 2.8
        _sev_meter(ax, rx + 1.2, ty - 0.6, 34.6, g["severity"])
        ty -= 2.2
        ax.text(rx + 1.2, ty, "Reason:", fontsize=7.8, fontweight="bold",
                color=INK, va="center", zorder=4)
        for line in reason_lines:
            ty -= 1.7
            ax.text(rx + 1.2, ty, line, fontsize=7.6, color=MUTED, va="center", zorder=4)
        ty -= 2.0
        ax.text(rx + 1.2, ty, "Recommended action:", fontsize=7.8, fontweight="bold",
                color=edge, va="center", zorder=4)
        for line in action_lines:
            ty -= 1.7
            ax.text(rx + 1.2, ty, line, fontsize=7.6, color=MUTED, va="center", zorder=4)
        y -= block_h + 1.2

    srcs = sorted({g["source"] for g in graded}) or ["-"]
    note = f"Detector: {', '.join(srcs)}   |   Standard basis: {graded[0]['standard'] if graded else 'n/a'}"
    extras = []
    if "AI vision detector" in srcs:
        extras.append("AI vision detector used where the Roboflow model returned no detection")
    if "Manual review" in srcs:
        extras.append("Manual review = expert-confirmed annotation")
    if extras:
        note += "   |   " + "; ".join(extras) + "."
    _footer(ax, page_no, note)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "structural_element", "defect", "detection_source",
                    "confidence", "severity", "severity_score", "affected_extent_pct",
                    "est_cost_inr", "repair_time_days", "repair_time_band",
                    "measurement_basis", "standard", "reason", "recommended_action"])
        for r in records:
            element = (r.get("element") or {}).get("element", "unknown")
            for g in r["graded"]:
                w.writerow([r["name"], element, g["defect"], g["source"],
                            round(g["confidence"], 3), g["severity"], f"{g['score']}/4",
                            round(g["area_pct"], 2), round(g.get("total_cost", 0.0)),
                            g.get("repair_days", ""), g.get("repair_band", ""),
                            g["measured"], g["standard"], g["reason"], g["action"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="testimages",
                        help="Folder of images to process")
    parser.add_argument("--out", default="outputs/detection_severity_report.pdf",
                        help="Output PDF path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--no-gpt-fallback", action="store_true",
                        help="Disable the GPT-4o vision fallback for images the "
                             "Roboflow model leaves empty")
    parser.add_argument("--gpt-model", default="gpt-4o",
                        help="GPT vision model for the fallback detector")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the detections cache and call the detectors again")
    parser.add_argument("--no-element", action="store_true",
                        help="Skip AI structural-element classification")
    args = parser.parse_args()

    images_dir = (ROOT / args.images).resolve()
    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images found in {images_dir}")

    out_pdf = (ROOT / args.out).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    annot_dir = out_pdf.parent / "annotated"
    cache_path = out_pdf.with_name(out_pdf.stem + "_detections_cache.json")
    cache: dict[str, Any] = {}
    if cache_path.exists() and not args.refresh:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    records: list[dict[str, Any]] = []
    for path in files:
        if path.name in cache and not args.refresh:
            entry = cache[path.name]
            preds, w, h = entry["preds"], entry["width"], entry["height"]
            print(f"[cache]  {path.name} ...", flush=True)
        else:
            print(f"[detect] {path.name} ...", flush=True)
            try:
                preds, w, h = detect(path, args.conf)
            except (RoboflowModelError, Exception) as err:  # noqa: BLE001
                print(f"  ! Roboflow detection failed: {err}")
                preds, w, h = [], 0, 0
            if not preds and not args.no_gpt_fallback:
                print("  ... Roboflow empty; trying AI vision detector ...", flush=True)
                try:
                    preds, w, h = detect_with_gpt(path, model=args.gpt_model)
                except Exception as err:  # noqa: BLE001
                    print(f"  ! AI vision detector failed: {err}")
            if w == 0 or h == 0:
                with Image.open(path) as im:
                    w, h = im.size
            cache[path.name] = {"width": w, "height": h, "preds": preds}
        # structural element (slab/wall/beam/column/staircase) via the AI engine
        element = cache[path.name].get("element")
        if element is None and not args.no_element:
            try:
                element = classify_structural_element(path, model=args.gpt_model)
                print(f"  ~ element: {element['element']} "
                      f"({element['confidence']*100:.0f}%)")
            except Exception as err:  # noqa: BLE001
                print(f"  ! element classification failed: {err}")
                element = {"element": "unknown", "confidence": 0.0, "reason": ""}
            cache[path.name]["element"] = element
        element = element or {"element": "unknown", "confidence": 0.0, "reason": ""}
        preds = list(preds) + load_manual(path.name, w, h)
        graded = grade(preds, w, h)
        annotated = annotate(path, graded, annot_dir / f"{path.stem}_annotated.jpg")
        src = graded[0]["source"] if graded else "-"
        print(f"  -> {len(graded)} detection(s) [{src}]: "
              f"{', '.join(f'{g['defect']}/{g['severity']}' for g in graded) or 'none'}")
        records.append({"name": path.name, "width": w, "height": h,
                        "graded": graded, "annotated": annotated, "element": element})

    if not records:
        raise SystemExit("No images could be processed (see errors above).")

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    csv_path = out_pdf.with_suffix(".csv")
    write_csv(records, csv_path)

    with PdfPages(out_pdf) as pdf:
        page_summary(pdf, records, page_no=1)
        for i, r in enumerate(records, start=1):
            page_image(pdf, r, i, len(records), page_no=i + 1)
        info = pdf.infodict()
        info["Title"] = "Construction Defect Detection — Detection & Severity Report"

    print(f"\nWrote PDF : {out_pdf}")
    print(f"Wrote CSV : {csv_path}")
    print(f"Annotated : {annot_dir}")


if __name__ == "__main__":
    main()
