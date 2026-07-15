from __future__ import annotations

import argparse
import html
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.remedy_rag import RemedyQuery, generate_rag_remedy  # noqa: E402
from src.severity import estimate_severity, mm_per_pixel_from_reference  # noqa: E402


@dataclass(frozen=True)
class ManualCase:
    file_name: str
    title: str
    defect_class: str
    box: tuple[int, int, int, int]
    depth_mm: float | None = None


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _concrete_canvas(width: int = 1280, height: int = 820, seed: int = 1) -> Image.Image:
    random.seed(seed)
    image = Image.new("RGB", (width, height), (174, 174, 166))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            base = 172 + random.randint(-18, 18)
            stain = int(8 * (x / width) + 5 * (y / height))
            value = max(105, min(215, base - stain))
            pixels[x, y] = (value, value, max(95, value - random.randint(3, 12)))

    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(750):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, 4)
        shade = random.randint(60, 130)
        draw.ellipse((x, y, x + radius, y + radius), fill=(shade, shade, shade - 5, random.randint(60, 150)))

    for _ in range(16):
        x1 = random.randint(-100, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(180, 520)
        y2 = y1 + random.randint(-30, 30)
        draw.line((x1, y1, x2, y2), fill=(105, 105, 100, 35), width=random.randint(1, 3))

    return image.filter(ImageFilter.SMOOTH_MORE)


def _draw_scale_card(draw: ImageDraw.ImageDraw, x: int = 60, y: int = 700) -> None:
    draw.rectangle((x, y, x + 240, y + 54), fill=(245, 245, 235), outline=(40, 40, 35), width=2)
    draw.text((x + 14, y + 12), "100 mm scale", fill=(20, 20, 20), font=_font(18))
    draw.line((x + 15, y + 42, x + 215, y + 42), fill=(20, 20, 20), width=3)
    for tick in range(0, 201, 20):
        tick_height = 16 if tick % 100 == 0 else 10
        draw.line((x + 15 + tick, y + 42, x + 15 + tick, y + 42 - tick_height), fill=(20, 20, 20), width=2)


def _draw_crack(image: Image.Image) -> ManualCase:
    draw = ImageDraw.Draw(image, "RGBA")
    random.seed(24)
    points: list[tuple[int, int]] = []
    x, y = 260, 390
    for _ in range(14):
        points.append((x, y))
        x += random.randint(42, 64)
        y += random.randint(-16, 18)

    for offset, alpha, width in [(4, 70, 9), (2, 90, 6), (0, 235, 3)]:
        shifted = [(px, py + offset) for px, py in points]
        color = (28, 28, 24, alpha)
        draw.line(shifted, fill=color, width=width, joint="curve")

    for index in (4, 7, 10):
        px, py = points[index]
        branch = [(px, py), (px + random.randint(45, 90), py + random.choice([-1, 1]) * random.randint(35, 70))]
        draw.line(branch, fill=(30, 30, 26, 210), width=2)

    _draw_scale_card(draw)
    return ManualCase(
        file_name="realistic_crack.jpg",
        title="Realistic scaled crack",
        defect_class="crack",
        box=(250, 370, 1030, 415),
    )


def _draw_spalling(image: Image.Image) -> ManualCase:
    draw = ImageDraw.Draw(image, "RGBA")
    random.seed(31)
    polygon = [
        (455, 255), (560, 220), (710, 238), (805, 305), (835, 420),
        (760, 520), (610, 555), (485, 510), (415, 405), (420, 310),
    ]
    shadow = [(x + 7, y + 9) for x, y in polygon]
    draw.polygon(shadow, fill=(45, 40, 34, 80))
    draw.polygon(polygon, fill=(93, 83, 72, 230), outline=(55, 48, 40, 240))

    inner = [(500, 305), (610, 275), (735, 305), (785, 390), (715, 480), (570, 485), (485, 410)]
    draw.polygon(inner, fill=(66, 60, 53, 210))
    for _ in range(95):
        x = random.randint(455, 790)
        y = random.randint(275, 510)
        r = random.randint(2, 7)
        shade = random.randint(35, 100)
        draw.ellipse((x, y, x + r, y + r), fill=(shade, shade - 3, shade - 8, random.randint(90, 180)))

    _draw_scale_card(draw)
    return ManualCase(
        file_name="realistic_spalling.jpg",
        title="Realistic irregular spalling",
        defect_class="spalling",
        box=(410, 215, 845, 560),
        depth_mm=45.0,
    )


def _draw_rebar(image: Image.Image) -> ManualCase:
    draw = ImageDraw.Draw(image, "RGBA")
    random.seed(42)
    opening = [(350, 300), (470, 245), (820, 255), (930, 330), (900, 485), (740, 545), (420, 505), (320, 410)]
    draw.polygon([(x + 8, y + 10) for x, y in opening], fill=(45, 40, 34, 75))
    draw.polygon(opening, fill=(86, 76, 66, 235), outline=(50, 43, 36, 255))

    for y in (352, 438):
        draw.line((405, y + 4, 875, y + 4), fill=(40, 25, 18, 160), width=24)
        draw.line((405, y, 875, y), fill=(132, 64, 30, 255), width=18)
        draw.line((405, y - 5, 875, y - 5), fill=(184, 92, 42, 100), width=3)
        for x in range(420, 860, 42):
            draw.line((x, y - 10, x + 18, y + 10), fill=(70, 36, 20, 160), width=2)

    for _ in range(110):
        x = random.randint(345, 900)
        y = random.randint(275, 520)
        r = random.randint(2, 6)
        draw.ellipse((x, y, x + r, y + r), fill=(random.randint(45, 115), 50, 35, random.randint(75, 160)))

    _draw_scale_card(draw)
    return ManualCase(
        file_name="realistic_exposed_rebar.jpg",
        title="Realistic exposed reinforcement",
        defect_class="exposed_rebar",
        box=(315, 240, 940, 555),
    )


def _draw_annotation(image: Image.Image, case: ManualCase, output_path: Path) -> None:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    x1, y1, x2, y2 = case.box
    label = f"{case.defect_class.replace('_', ' ').title()} | manual annotation"
    draw.rectangle((x1, y1, x2, y2), outline=(235, 42, 42, 255), width=5)
    text_box = draw.textbbox((x1, max(8, y1 - 32)), label, font=_font(22))
    draw.rectangle(text_box, fill=(235, 42, 42, 235))
    draw.text((x1, max(8, y1 - 32)), label, fill=(255, 255, 255), font=_font(22))
    annotated.save(output_path, quality=92)


def _case_row(case: ManualCase, mm_per_pixel: float, use_openai: bool, openai_model: str) -> dict[str, Any]:
    x1, y1, x2, y2 = case.box
    severity = estimate_severity(
        defect_class=case.defect_class,
        box_width=x2 - x1,
        box_height=y2 - y1,
        image_width=1280,
        image_height=820,
        mm_per_pixel=mm_per_pixel,
        depth_mm=case.depth_mm,
    )
    rag = generate_rag_remedy(
        RemedyQuery(
            defect_class=case.defect_class,
            severity_level=severity.level,
            measured=severity.measured,
            reason=severity.reason,
            remedial_measure=severity.remedial_measure,
            repair_time_estimate=severity.repair_time_estimate,
            cost_breakup=severity.cost_breakup,
            boq_breakup=severity.boq_breakup,
        ),
        use_openai=use_openai,
        openai_model=openai_model,
    )
    return {
        "image": case.file_name,
        "defect": case.defect_class,
        "annotation": "Manual visual annotation for RAG demonstration",
        "severity": severity.level,
        "measurement_basis": severity.measured,
        "reason": severity.reason,
        "remedial_measure": severity.remedial_measure,
        "quantity": severity.cost_breakup.get("quantity", ""),
        "quantity_description": severity.cost_breakup.get("quantity_description", ""),
        "material_rate": severity.cost_breakup.get("material_rate", ""),
        "labour_rate": severity.cost_breakup.get("labour_rate", ""),
        "equipment_rate": severity.cost_breakup.get("equipment_rate", ""),
        "composite_rate": severity.cost_breakup.get("composite_rate", ""),
        "total_cost": round(float(severity.cost_breakup.get("total_cost", 0.0)), 2),
        "repair_time": severity.repair_time_estimate,
        "boq": severity.boq_breakup,
        "boq_table": severity.boq.formatted_table() if severity.boq and severity.boq.norms_found else "",
        "rag_used_openai": rag.used_llm,
        "rag_model": rag.model,
        "rag_sources": "; ".join(rag.sources),
        "rag_answer": rag.answer,
        "rag_error": rag.llm_error,
    }


def _write_html(rows: list[dict[str, Any]], output_dir: Path) -> None:
    image_cards = []
    for row in rows:
        image_cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(str(row['image']))}</h2>
              <img src="{html.escape(str(row['image']))}" alt="{html.escape(str(row['defect']))}">
              <p><strong>Defect:</strong> {html.escape(str(row['defect']))}</p>
              <p><strong>Annotation:</strong> {html.escape(str(row['annotation']))}</p>
            </section>
            """
        )

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['image']))}</td>"
            f"<td>{html.escape(str(row['defect']))}</td>"
            f"<td>{html.escape(str(row['severity']))}</td>"
            f"<td>{html.escape(str(row['measurement_basis']))}</td>"
            f"<td>{html.escape(str(row['quantity']))}</td>"
            f"<td>{html.escape(str(row['composite_rate']))}</td>"
            f"<td>{html.escape(str(row['total_cost']))}</td>"
            f"<td>{html.escape(str(row['repair_time']))}</td>"
            f"<td>{html.escape(str(row['rag_used_openai']))}</td>"
            "</tr>"
        )

    boq_sections = []
    for index, row in enumerate(rows, start=1):
        boq = row.get("boq") or {}
        if not boq.get("norms_found", False):
            continue
        line_rows = []
        for line in boq.get("lines", []):
            line_rows.append(
                "<tr>"
                f"<td>{html.escape(str(line['category']).title())}</td>"
                f"<td>{html.escape(str(line['description']))}</td>"
                f"<td>{html.escape(str(line['norm']))} {html.escape(str(line['norm_unit']))}</td>"
                f"<td>{html.escape(str(line['quantity']))} {html.escape(str(line['quantity_unit']))}</td>"
                f"<td>INR {html.escape(str(line['rate']))}</td>"
                f"<td>INR {html.escape(str(line['amount']))}</td>"
                "</tr>"
            )
        boq_sections.append(
            f"""
            <section class="card">
              <h2>{index}. BOQ - {html.escape(str(row['defect']).replace('_', ' ').title())} ({html.escape(str(row['severity']))})</h2>
              <p><strong>Remedy:</strong> {html.escape(str(boq.get('remedy', '')))}</p>
              <p><strong>Work quantity:</strong> {html.escape(str(boq.get('work_quantity', '')))} {html.escape(str(boq.get('work_unit', '')))}</p>
              <p><strong>Norms/rates source:</strong> {html.escape(str(boq.get('source', '')))}</p>
              <table>
                <thead><tr><th>Category</th><th>Item</th><th>Norm (per work unit)</th><th>Quantity = work qty x norm</th><th>Rate</th><th>Amount = qty x rate</th></tr></thead>
                <tbody>{''.join(line_rows)}</tbody>
              </table>
              <p>
                <strong>Material:</strong> INR {html.escape(str(boq.get('material_total', 0)))} |
                <strong>Labour:</strong> INR {html.escape(str(boq.get('labour_total', 0)))} |
                <strong>Equipment:</strong> INR {html.escape(str(boq.get('equipment_total', 0)))}
              </p>
              <p>
                <strong>Subtotal:</strong> INR {html.escape(str(boq.get('subtotal', 0)))} |
                <strong>Overheads (15%):</strong> INR {html.escape(str(boq.get('overheads', 0)))} |
                <strong>GST (18%):</strong> INR {html.escape(str(boq.get('gst', 0)))} |
                <strong>GRAND TOTAL:</strong> INR {html.escape(str(boq.get('grand_total', 0)))}
              </p>
              <p class="muted">Method statement: {html.escape(str(boq.get('method_steps', '')))}</p>
            </section>
            """
        )

    remedy_sections = []
    for index, row in enumerate(rows, start=1):
        remedy_sections.append(
            f"""
            <section class="card">
              <h2>{index}. {html.escape(str(row['defect']).replace('_', ' ').title())} RAG Remedy</h2>
              <p><strong>Sources:</strong> {html.escape(str(row['rag_sources']))}</p>
              <pre>{html.escape(str(row['rag_answer']))}</pre>
            </section>
            """
        )

    doc = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Realistic Manual RAG Defect Demo</title>
      <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #172033; background: #f4f6fa; }}
        h1 {{ margin-bottom: 4px; }}
        .muted {{ color: #536171; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
        .card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; margin: 18px 0; }}
        img {{ max-width: 100%; border: 1px solid #cbd3df; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th, td {{ border: 1px solid #d8dee9; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #e8eef8; text-align: left; }}
        pre {{ white-space: pre-wrap; background: #f1f4f9; border: 1px solid #d8dee9; padding: 12px; }}
      </style>
    </head>
    <body>
      <h1>Realistic Manual RAG Defect Demo</h1>
      <p class="muted">This report uses realistic images with manual annotations to demonstrate how the RAG module generates remedies from defect type, severity, measurement, quantity and cost data.</p>
      <p class="muted">Scale shown in each image: 100 mm card = 200 px, so mm_per_pixel = 0.50 mm/px.</p>
      <div class="grid">{''.join(image_cards)}</div>
      <section class="card">
        <h2>Measurement, Cost and RAG Summary</h2>
        <table>
          <thead><tr><th>Image</th><th>Defect</th><th>Severity</th><th>Measurement</th><th>Quantity</th><th>Composite Rate</th><th>Total Cost</th><th>Time</th><th>OpenAI Used</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </section>
      <h2>Bill of Quantities (norms and rates retrieved via RAG; cost = quantity x rate)</h2>
      {''.join(boq_sections)}
      <h2>RAG Remedy Plans</h2>
      {''.join(remedy_sections)}
    </body>
    </html>
    """
    (output_dir / "realistic-rag-report.html").write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic manual RAG visual demo images and report")
    parser.add_argument("--out", default="outputs/realistic-rag-demo", help="Output directory")
    parser.add_argument("--use-openai-rag", action="store_true", help="Use OpenAI for RAG remedy text")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    output_dir = ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    mm_per_pixel = mm_per_pixel_from_reference(reference_size_mm=100.0, reference_size_px=200.0)

    cases: list[ManualCase] = []
    for seed, drawer in [(12, _draw_crack), (18, _draw_spalling), (22, _draw_rebar)]:
        image = _concrete_canvas(seed=seed)
        case = drawer(image)
        _draw_annotation(image, case, output_dir / case.file_name)
        cases.append(case)

    rows = [_case_row(case, mm_per_pixel, args.use_openai_rag, args.openai_model) for case in cases]
    _write_html(rows, output_dir)

    print(f"Generated realistic visual RAG report: {output_dir / 'realistic-rag-report.html'}")
    print(f"Generated {len(rows)} realistic annotated images")


if __name__ == "__main__":
    main()