from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _safe(value: Any) -> str:
    return "" if value is None else str(value)


def _money(value: Any) -> str:
    try:
        return f"INR {float(value):,.2f}"
    except (TypeError, ValueError):
        return _safe(value)


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return _safe(value)


def _parse_boq(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def _html_escape_text(text: str) -> str:
    return (
        _safe(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _paragraph_from_markdown(line: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    line = line.strip()
    if not line:
        return Paragraph(" ", styles["body"])
    heading = re.match(r"^#{1,6}\s+(.*)$", line)
    if heading:
        return Paragraph(_html_escape_text(heading.group(1)), styles["h3"])
    if line.startswith("- "):
        return Paragraph("&#8226; " + _html_escape_text(line[2:]), styles["bullet"])
    numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
    if numbered:
        return Paragraph(f"{numbered.group(1)}. {_html_escape_text(numbered.group(2))}", styles["bullet"])
    return Paragraph(_html_escape_text(line), styles["body"])


def _image_flowable(path: Path, max_width: float, max_height: float) -> Image | Paragraph:
    if not path.exists():
        return Paragraph(f"Image missing: {_html_escape_text(path.name)}", _styles()["body"])
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12315a"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5b6b"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#12315a"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1d4e89"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#26364a"),
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=11.5,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontSize=8.3,
            leading=11.2,
            leftIndent=10,
            firstLineIndent=-6,
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=7,
            leading=9,
            spaceAfter=2,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontSize=6.8,
            leading=8.2,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.2,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "right": ParagraphStyle(
            "Right",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_RIGHT,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_html_escape_text(_safe(text)), style)


def _table(data: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, splitByRow=True)
    commands: list[tuple[Any, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d1dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12315a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row_index in range(1 if header else 0, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f5f7fb")))
    table.setStyle(TableStyle(commands))
    return table


def _summary_table(rows: list[dict[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [
            _p("Image", styles["table_header"]),
            _p("Defect", styles["table_header"]),
            _p("Conf.", styles["table_header"]),
            _p("Severity", styles["table_header"]),
            _p("Quantity", styles["table_header"]),
            _p("Total Cost", styles["table_header"]),
            _p("OpenAI", styles["table_header"]),
        ]
    ]
    for row in rows:
        data.append(
            [
                _p(row.get("image", ""), styles["table_cell"]),
                _p(row.get("defect", ""), styles["table_cell"]),
                _p(row.get("confidence", ""), styles["table_cell"]),
                _p(row.get("severity", ""), styles["table_cell"]),
                _p(row.get("repair_quantity", ""), styles["table_cell"]),
                _p(_money(row.get("total_repair_cost", "")), styles["table_cell"]),
                _p(f"{row.get('rag_used_openai', '')} {row.get('rag_model', '')}", styles["table_cell"]),
            ]
        )
    return _table(data, [33 * mm, 22 * mm, 15 * mm, 20 * mm, 36 * mm, 25 * mm, 30 * mm])


def _boq_table(boq: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    if not boq or not boq.get("norms_found", False):
        return [Paragraph("No norms-based BOQ available for this defect/severity.", styles["body"])]

    flowables: list[Any] = [
        Paragraph("BOQ / Rate Analysis", styles["h2"]),
        Paragraph(f"<b>Remedy:</b> {_html_escape_text(boq.get('remedy', ''))}", styles["body"]),
        Paragraph(
            f"<b>Work Quantity:</b> {_number(boq.get('work_quantity'), 3)} {_html_escape_text(boq.get('work_unit', ''))}",
            styles["body"],
        ),
        Paragraph(f"<b>Norms/Rate Source:</b> {_html_escape_text(boq.get('source', ''))}", styles["body"]),
        Spacer(1, 4),
    ]

    data: list[list[Any]] = [
        [
            _p("Cat.", styles["table_header"]),
            _p("Item", styles["table_header"]),
            _p("Norm", styles["table_header"]),
            _p("Qty", styles["table_header"]),
            _p("Rate", styles["table_header"]),
            _p("Amount", styles["table_header"]),
        ]
    ]
    for line in boq.get("lines", []):
        data.append(
            [
                _p(line.get("category", ""), styles["table_cell"]),
                _p(line.get("description", ""), styles["table_cell"]),
                _p(f"{line.get('norm', '')} {line.get('norm_unit', '')}", styles["table_cell"]),
                _p(f"{line.get('quantity', '')} {line.get('quantity_unit', '')}", styles["table_cell"]),
                _p(_money(line.get("rate", "")), styles["table_cell"]),
                _p(_money(line.get("amount", "")), styles["table_cell"]),
            ]
        )
    flowables.append(_table(data, [13 * mm, 45 * mm, 34 * mm, 25 * mm, 25 * mm, 29 * mm]))
    flowables.append(Spacer(1, 5))

    totals = [
        [_p("Material Total", styles["table_cell"]), _p(_money(boq.get("material_total")), styles["table_cell"])],
        [_p("Labour Total", styles["table_cell"]), _p(_money(boq.get("labour_total")), styles["table_cell"])],
        [_p("Equipment Total", styles["table_cell"]), _p(_money(boq.get("equipment_total")), styles["table_cell"])],
        [_p("Subtotal", styles["table_cell"]), _p(_money(boq.get("subtotal")), styles["table_cell"])],
        [_p("Overheads & Contingencies (15%)", styles["table_cell"]), _p(_money(boq.get("overheads")), styles["table_cell"])],
        [_p("GST (18%)", styles["table_cell"]), _p(_money(boq.get("gst")), styles["table_cell"])],
        [_p("GRAND TOTAL", styles["table_header"]), _p(_money(boq.get("grand_total")), styles["table_header"])],
    ]
    totals_table = _table(totals, [70 * mm, 45 * mm], header=False)
    totals_table.setStyle(TableStyle([("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#12315a"))]))
    flowables.append(totals_table)
    flowables.append(Spacer(1, 5))
    flowables.append(Paragraph(f"<b>Method Statement:</b> {_html_escape_text(boq.get('method_steps', ''))}", styles["body"]))
    return flowables


def _rag_flowables(text: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    flowables: list[Any] = [Paragraph("OpenAI RAG Remedy Plan", styles["h2"])]
    for line in _safe(text).splitlines():
        flowables.append(_paragraph_from_markdown(line, styles))
    return flowables


def _footer(canvas: Any, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(15 * mm, 9 * mm, "Construction Defect Detection - RAG + BOQ Report")
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(input_dir: Path, output_pdf: Path) -> None:
    csv_path = input_dir / "detections_with_rag.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
    )

    story: list[Any] = []
    story.append(Paragraph("Construction Defect Detection Report", styles["title"]))
    story.append(
        Paragraph(
            "End-to-end image evidence with detection outputs, severity, quantity estimation, norms-based BOQ, "
            "rate analysis, and OpenAI RAG remedy generation.",
            styles["subtitle"],
        )
    )
    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(_summary_table(rows, styles))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Note: where no physical scale reference is provided, image-derived quantities are preliminary and must be "
            "verified on site using scale card/ruler, crack gauge, depth probe, cover meter and engineer inspection.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    unique_images: list[str] = []
    for row in rows:
        image_name = row.get("image", "")
        if image_name and image_name not in unique_images:
            unique_images.append(image_name)

    story.append(Paragraph("Annotated Image Evidence", styles["h1"]))
    for image_name in unique_images:
        annotated = input_dir / f"{Path(image_name).stem}_annotated.jpg"
        story.append(Paragraph(image_name, styles["h2"]))
        story.append(_image_flowable(annotated, max_width=175 * mm, max_height=105 * mm))
        story.append(Spacer(1, 8))
    story.append(PageBreak())

    for index, row in enumerate(rows, start=1):
        boq = _parse_boq(row.get("boq", ""))
        story.append(Paragraph(f"Detection {index}: {row.get('defect', '').title()} - {row.get('image', '')}", styles["h1"]))
        detail_table = _table(
            [
                [_p("Field", styles["table_header"]), _p("Value", styles["table_header"])],
                [_p("Defect", styles["table_cell"]), _p(row.get("defect", ""), styles["table_cell"])],
                [_p("Confidence", styles["table_cell"]), _p(row.get("confidence", ""), styles["table_cell"])],
                [_p("Severity", styles["table_cell"]), _p(row.get("severity", ""), styles["table_cell"])],
                [_p("Measurement Basis", styles["table_cell"]), _p(row.get("measurement_basis", ""), styles["table_cell"])],
                [_p("Reason", styles["table_cell"]), _p(row.get("reason", ""), styles["table_cell"])],
                [_p("Repair Quantity", styles["table_cell"]), _p(row.get("repair_quantity", ""), styles["table_cell"])],
                [_p("Repair Time", styles["table_cell"]), _p(row.get("repair_time_estimate", ""), styles["table_cell"])],
                [_p("OpenAI RAG", styles["table_cell"]), _p(f"{row.get('rag_used_openai')} ({row.get('rag_model')})", styles["table_cell"])],
                [_p("Sources", styles["table_cell"]), _p(row.get("rag_sources", ""), styles["table_cell"])],
            ],
            [38 * mm, 135 * mm],
        )
        story.append(detail_table)
        story.append(Spacer(1, 8))
        story.extend(_boq_table(boq, styles))
        story.append(Spacer(1, 8))
        story.extend(_rag_flowables(row.get("rag_remedy_plan", ""), styles))
        if index != len(rows):
            story.append(PageBreak())

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a clean paginated PDF report from end-to-end detection CSV")
    parser.add_argument("--input-dir", default="outputs/testimages-end-to-end", help="Directory containing detections_with_rag.csv and annotated images")
    parser.add_argument("--output", default="outputs/testimages-end-to-end/construction-defect-rag-boq-report-clean.pdf", help="Output PDF path")
    args = parser.parse_args()
    build_pdf(Path(args.input_dir), Path(args.output))
    print(f"PDF_CREATED: {args.output}")


if __name__ == "__main__":
    main()