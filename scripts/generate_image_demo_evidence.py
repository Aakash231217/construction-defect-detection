from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.remedy_rag import RemedyQuery, generate_rag_remedy  # noqa: E402
from src.roboflow_model import run_model  # noqa: E402
from src.severity import estimate_severity, mm_per_pixel_from_reference  # noqa: E402


def _prediction_box(prediction: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(prediction.get("x", 0.0))
    y = float(prediction.get("y", 0.0))
    width = float(prediction.get("width", 0.0))
    height = float(prediction.get("height", 0.0))
    return x - width / 2, y - height / 2, x + width / 2, y + height / 2


def _draw_predictions(image_path: Path, predictions: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for prediction in predictions:
        x1, y1, x2, y2 = _prediction_box(prediction)
        label = f"{prediction.get('class', 'defect')} {float(prediction.get('confidence', 0.0)):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=4)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 40, 40))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _to_row(
    image_path: Path,
    prediction: dict[str, Any],
    image_width: float,
    image_height: float,
    mm_per_pixel: float | None,
    use_openai: bool,
    openai_model: str,
) -> dict[str, Any]:
    defect_class = str(prediction.get("class", "defect"))
    severity = estimate_severity(
        defect_class=defect_class,
        box_width=float(prediction.get("width", 0.0)),
        box_height=float(prediction.get("height", 0.0)),
        image_width=image_width,
        image_height=image_height,
        mm_per_pixel=mm_per_pixel,
    )
    rag = generate_rag_remedy(
        RemedyQuery(
            defect_class=defect_class,
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
        "image": image_path.name,
        "defect": defect_class,
        "confidence": round(float(prediction.get("confidence", 0.0)), 3),
        "severity": severity.level,
        "measurement_basis": severity.measured,
        "standard": severity.standard,
        "reason": severity.reason,
        "remedial_measure": severity.remedial_measure,
        "repair_quantity": severity.cost_breakup.get("quantity", ""),
        "quantity_description": severity.cost_breakup.get("quantity_description", ""),
        "material_rate": severity.cost_breakup.get("material_rate", ""),
        "labour_rate": severity.cost_breakup.get("labour_rate", ""),
        "equipment_rate": severity.cost_breakup.get("equipment_rate", ""),
        "composite_rate": severity.cost_breakup.get("composite_rate", ""),
        "final_boq_rate_incl_overheads_gst": severity.cost_breakup.get("composite_rate", ""),
        "total_repair_cost": round(float(severity.cost_breakup.get("total_cost", 0.0)), 2),
        "repair_time_estimate": severity.repair_time_estimate,
        "rag_used_openai": rag.used_llm,
        "rag_model": rag.model,
        "rag_sources": "; ".join(rag.sources),
        "rag_remedy_plan": rag.answer,
        "rag_error": rag.llm_error,
        "boq": severity.boq_breakup,
    }


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("No detections\n", encoding="utf-8")
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_html(rows: list[dict[str, Any]], annotated_images: list[Path], html_path: Path) -> None:
    cards = []
    for image_path in annotated_images:
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(image_path.name)}</h2>
              <img src="{html.escape(image_path.name)}" alt="Annotated detection image">
            </section>
            """
        )

    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['image']))}</td>"
            f"<td>{html.escape(str(row['defect']))}</td>"
            f"<td>{html.escape(str(row['confidence']))}</td>"
            f"<td>{html.escape(str(row['severity']))}</td>"
            f"<td>{html.escape(str(row['repair_quantity']))}</td>"
            f"<td>{html.escape(str(row['final_boq_rate_incl_overheads_gst']))}</td>"
            f"<td>{html.escape(str(row['total_repair_cost']))}</td>"
            f"<td>{html.escape(str(row['repair_time_estimate']))}</td>"
            f"<td>{html.escape(str(row['rag_sources']))}</td>"
            "</tr>"
        )

    remedy_sections = []
    for index, row in enumerate(rows, start=1):
        remedy_sections.append(
            f"""
            <section class="card">
              <h2>{index}. {html.escape(str(row['image']))} - {html.escape(str(row['defect']))}</h2>
              <p><strong>Severity:</strong> {html.escape(str(row['severity']))}</p>
              <p><strong>Measurement:</strong> {html.escape(str(row['measurement_basis']))}</p>
              <p><strong>Quantity:</strong> {html.escape(str(row['repair_quantity']))}</p>
              <p><strong>Cost:</strong> INR {html.escape(str(row['total_repair_cost']))}</p>
              <pre>{html.escape(str(row['rag_remedy_plan']))}</pre>
            </section>
            """
        )

        boq_sections = []
        for index, row in enumerate(rows, start=1):
                boq = row.get("boq") or {}
                if not boq.get("norms_found", False):
                        boq_sections.append(
                                f"""
                                <section class="card">
                                    <h2>{index}. BOQ - {html.escape(str(row['image']))}</h2>
                                    <p>No norms record was found for this detected class/severity. Engineer estimate required.</p>
                                </section>
                                """
                        )
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
                            <h2>{index}. BOQ - {html.escape(str(row['image']))} / {html.escape(str(row['defect']))}</h2>
                            <p><strong>Remedy:</strong> {html.escape(str(boq.get('remedy', '')))}</p>
                            <p><strong>Work quantity:</strong> {html.escape(str(boq.get('work_quantity', '')))} {html.escape(str(boq.get('work_unit', '')))}</p>
                            <p><strong>Norms/rates source:</strong> {html.escape(str(boq.get('source', '')))}</p>
                            <table>
                                <thead><tr><th>Category</th><th>Item</th><th>Norm</th><th>Quantity = Work Qty x Norm</th><th>Rate</th><th>Amount = Quantity x Rate</th></tr></thead>
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

    html_doc = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Construction Defect Image Demo Evidence</title>
      <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #172033; background: #f5f7fb; }}
        h1 {{ margin-bottom: 6px; }}
        .muted {{ color: #536171; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
        .card {{ background: #fff; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; margin: 18px 0; }}
        img {{ max-width: 100%; border: 1px solid #ccd3df; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; }}
        th, td {{ border: 1px solid #d8dee9; padding: 8px; vertical-align: top; font-size: 13px; }}
        th {{ background: #e8eef8; text-align: left; }}
        pre {{ white-space: pre-wrap; background: #f1f4f9; border: 1px solid #d8dee9; padding: 12px; }}
      </style>
    </head>
    <body>
      <h1>Construction Defect Image Demo Evidence</h1>
      <p class="muted">This report shows image-level detection evidence, severity, quantity-based cost estimate and RAG remedy generation.</p>
      <div class="grid">{''.join(cards)}</div>
      <section class="card">
        <h2>Detection and Cost Table</h2>
        <table>
          <thead>
            <tr>
              <th>Image</th><th>Defect</th><th>Confidence</th><th>Severity</th><th>Quantity</th><th>Final BOQ Rate (incl. OH/GST)</th><th>Final BOQ Total</th><th>Time</th><th>RAG Sources</th>
            </tr>
          </thead>
          <tbody>{''.join(row_html)}</tbody>
        </table>
      </section>
            <h2>Bill of Quantities (RAG norms/rates; cost = quantity x rate)</h2>
            {''.join(boq_sections)}
      <h2>RAG Remedy Plans</h2>
      {''.join(remedy_sections)}
    </body>
    </html>
    """
    html_path.write_text(html_doc, encoding="utf-8")


def generate_demo(
    image_paths: list[Path],
    output_dir: Path,
    mm_per_pixel: float | None,
    use_openai: bool,
    openai_model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    annotated_images: list[Path] = []

    for image_path in image_paths:
        result = run_model(str(image_path))
        if not isinstance(result, dict):
            continue
        predictions = [item for item in result.get("predictions", []) if isinstance(item, dict)]
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        _draw_predictions(image_path, predictions, annotated_path)
        annotated_images.append(annotated_path)

        image_block = result.get("image", {})
        image_width = float(image_block.get("width", Image.open(image_path).size[0]))
        image_height = float(image_block.get("height", Image.open(image_path).size[1]))
        for prediction in predictions:
            rows.append(_to_row(image_path, prediction, image_width, image_height, mm_per_pixel, use_openai, openai_model))

    _write_csv(rows, output_dir / "detections_with_rag.csv")
    _write_html(rows, annotated_images, output_dir / "image-demo-report.html")
    print(f"Generated {len(annotated_images)} annotated images")
    print(f"Generated {len(rows)} detection rows")
    print(f"Report: {output_dir / 'image-demo-report.html'}")
    print(f"CSV: {output_dir / 'detections_with_rag.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image demo evidence with detections, cost and RAG remedies")
    parser.add_argument("--images", nargs="*", default=["test-images/*.jpg"], help="Image files or glob patterns")
    parser.add_argument("--out", default="outputs/image-demo", help="Output directory")
    parser.add_argument("--ref-mm", type=float, default=None, help="Known reference size in mm")
    parser.add_argument("--ref-px", type=float, default=None, help="Reference size in image pixels")
    parser.add_argument("--use-openai-rag", action="store_true", help="Use OpenAI for RAG remedy text")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    image_paths: list[Path] = []
    for pattern in args.images:
        matches = sorted(ROOT.glob(pattern))
        image_paths.extend(matches if matches else [Path(pattern)])

    image_paths = [path if path.is_absolute() else ROOT / path for path in image_paths]
    image_paths = [path for path in image_paths if path.exists()]
    if not image_paths:
        raise SystemExit("No input images found")

    mm_per_pixel = None
    if args.ref_mm and args.ref_px:
        mm_per_pixel = mm_per_pixel_from_reference(args.ref_mm, args.ref_px)
        print(f"Scale: {mm_per_pixel:.4f} mm/px")
    else:
        print("No scale reference supplied; quantities are approximate from area ratio.")

    generate_demo(
        image_paths=image_paths,
        output_dir=ROOT / args.out,
        mm_per_pixel=mm_per_pixel,
        use_openai=args.use_openai_rag,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    main()