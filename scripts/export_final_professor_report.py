from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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


ROOT = Path(__file__).resolve().parents[1]


def _safe(value: Any) -> str:
    return "" if value is None else str(value)


def _escape(text: Any) -> str:
    return (
        _safe(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _money(value: Any) -> str:
    try:
        return f"INR {float(value):,.2f}"
    except (TypeError, ValueError):
        return _safe(value)


def _num(value: Any, digits: int = 3) -> str:
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


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12315a"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4c5d70"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#12315a"),
            spaceBefore=10,
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
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#26364a"),
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=8.7,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=11.5,
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
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontSize=6.8,
            leading=8.4,
        ),
        "head": ParagraphStyle(
            "Head",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.4,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escape(text), style)


def _add_box(drawing: Drawing, x: float, y: float, width: float, height: float, label: str) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            fillColor=colors.HexColor("#f5f9ff"),
            strokeColor=colors.HexColor("#1d4e89"),
            strokeWidth=1.2,
        )
    )
    drawing.add(
        String(
            x + width / 2,
            y + height / 2 + 2,
            label,
            fontName="Helvetica-Bold",
            fontSize=7.2,
            textAnchor="middle",
            fillColor=colors.HexColor("#12315a"),
        )
    )


def _architecture_diagram() -> Drawing:
    drawing = Drawing(540, 140)
    boxes = [
        (20, 50, 90, 40, "Image Input"),
        (140, 50, 95, 40, "Defect Detection"),
        (265, 50, 95, 40, "Severity Engine"),
        (390, 50, 95, 40, "Quantity & BOQ"),
        (515, 50, 95, 40, "Remedy Planning"),
    ]
    for box in boxes:
        _add_box(drawing, *box)
    for index in range(len(boxes) - 1):
        x1 = boxes[index][0] + boxes[index][2]
        y1 = boxes[index][1] + boxes[index][3] / 2
        x2 = boxes[index + 1][0]
        y2 = boxes[index + 1][1] + boxes[index + 1][3] / 2
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#2f6fdb"), strokeWidth=1.4))
    return drawing


def _rag_diagram() -> Drawing:
    drawing = Drawing(320, 160)
    boxes = [
        (20, 100, 95, 38, "Defect Context"),
        (135, 100, 95, 38, "Retrieved Knowledge"),
        (250, 100, 95, 38, "Grounded Prompt"),
        (135, 30, 95, 38, "Reasoning Layer"),
    ]
    for box in boxes:
        _add_box(drawing, *box)
    drawing.add(Line(115, 119, 135, 119, strokeColor=colors.HexColor("#2f6fdb"), strokeWidth=1.2))
    drawing.add(Line(230, 119, 250, 119, strokeColor=colors.HexColor("#2f6fdb"), strokeWidth=1.2))
    drawing.add(Line(182, 100, 182, 68, strokeColor=colors.HexColor("#2f6fdb"), strokeWidth=1.2))
    return drawing


def _md_flowables(text: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    items: list[Any] = []
    for line in _safe(text).splitlines():
        line = line.strip()
        if not line:
            items.append(Spacer(1, 2))
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            items.append(_p(heading.group(1), styles["h3"]))
        elif line.startswith("- "):
            items.append(Paragraph("&#8226; " + _escape(line[2:]), styles["bullet"]))
        else:
            items.append(_p(line, styles["body"]))
    return items


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
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12315a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    for row_index in range(1 if header else 0, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f5f7fb")))
    table.setStyle(TableStyle(commands))
    return table


def _image(path: Path, max_width: float, max_height: float) -> Any:
    if not path.exists():
        return Paragraph(f"Image missing: {_escape(path.name)}", _styles()["body"])
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def _footer(canvas: Any, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(14 * mm, 9 * mm, "Construction Defect Detection using YOLO + RAG + BOQ")
    canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _load_rows(input_dir: Path) -> list[dict[str, str]]:
    csv_path = input_dir / "detections_with_rag.csv"
    if not csv_path.exists():
        return []
    return list(csv.DictReader(csv_path.open(encoding="utf-8")))


def _overview(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("Project Overview", styles["h1"]))
    story.append(_p(
        "This project transforms defect images into an engineering-ready workflow for severity grading, quantity estimation, BOQ preparation, repair planning, and cost evaluation. "
        "The core idea is to connect visual inspection with a retrieval-based reasoning layer that uses engineering norms and repair guidance to produce structured remedial recommendations.",
        styles["body"],
    ))
    objectives = [
        "Detect visible construction defects such as cracks, spalling, mold/dampness and exposed reinforcement.",
        "Classify severity using engineering logic and standards such as ACI 224R, IS 456, ICRI 310.1 and ACI 562.",
        "Estimate quantities such as running metre, square metre and cubic metre using image dimensions and scale reference where available.",
        "Retrieve material, labour and equipment norms/rates through the knowledge layer and calculate BOQ cost using quantity x rate.",
        "Generate remedy plans through a retrieval-augmented AI reasoning engine grounded in engineering context and BOQ data.",
    ]
    for item in objectives:
        story.append(Paragraph("&#8226; " + _escape(item), styles["bullet"]))
    story.append(Spacer(1, 8))
    story.append(_p("System Architecture", styles["h2"]))
    story.append(_architecture_diagram())
    story.append(Spacer(1, 8))
    story.append(_p("RAG-Based Remedy Engine", styles["h2"]))
    story.append(_rag_diagram())
    story.append(_p(
        "The remedy engine combines detected defect context, severity output, repair norms, and engineering guidance into a grounded prompt. A generative reasoning layer then converts that structured context into a professional repair method statement and cost-aware recommendation.",
        styles["body"],
    ))


def _rag_documentation(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(PageBreak())
    story.append(_p("RAG System Documentation", styles["h1"]))
    story.append(_p(
        "The RAG subsystem is the knowledge-driven reasoning layer of the project. It uses retrieved engineering context to convert defect observations into structured remedial recommendations instead of relying on a purely generic language model response.",
        styles["body"],
    ))
    story.append(_p("Core Purpose", styles["h2"]))
    story.append(_p(
        "The RAG module bridges the gap between defect detection and engineering action. It collects defect severity, quantity and cost context, retrieves relevant repair guidance, and generates a professional repair plan that is grounded in the stored knowledge base.",
        styles["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(_p("Knowledge Sources", styles["h2"]))
    data = [
        [_p("Component", styles["head"]), _p("Role in the RAG Layer", styles["head"])],
        [_p("Defect context", styles["cell"]), _p("Provides the defect class, severity, measured values and repair quantity context.", styles["cell"])],
        [_p("Remedy knowledge base", styles["cell"]), _p("Stores repair guidance, method statements and defect-specific remediation notes.", styles["cell"])],
        [_p("Repair norms and rates", styles["cell"]), _p("Supplies material, labour and equipment norms and cost logic for BOQ preparation.", styles["cell"])],
        [_p("Reasoning layer", styles["cell"]), _p("Combines the retrieved evidence with the defect context and produces a grounded remedy plan.", styles["cell"])],
    ]
    story.append(_table(data, [44 * mm, 132 * mm]))
    story.append(Spacer(1, 8))
    story.append(_p("RAG Workflow", styles["h2"]))
    story.append(_rag_diagram())
    story.append(Spacer(1, 6))
    story.append(Paragraph("&#8226; " + _escape("The defect and severity engine provides the input context for the remedy module."), styles["bullet"]))
    story.append(Paragraph("&#8226; " + _escape("Relevant knowledge is retrieved from the repair knowledge base and norms database."), styles["bullet"]))
    story.append(Paragraph("&#8226; " + _escape("A grounded prompt is assembled using the retrieved evidence and the current defect state."), styles["bullet"]))
    story.append(Paragraph("&#8226; " + _escape("The reasoning layer produces a repair plan, method statement and cost-aware recommendation."), styles["bullet"]))
    story.append(Spacer(1, 8))
    story.append(_p("Why this approach is useful", styles["h2"]))
    story.append(Paragraph("&#8226; " + _escape("It makes the output more engineering-oriented and less generic than a standalone language model response."), styles["bullet"]))
    story.append(Paragraph("&#8226; " + _escape("It improves traceability because the remedy is linked to retrieved knowledge and not only to free-form generation."), styles["bullet"]))
    story.append(Paragraph("&#8226; " + _escape("It supports a professional workflow for preliminary repair planning, BOQ preparation and consultant review."), styles["bullet"]))
    story.append(Spacer(1, 6))
    story.append(_p("Current operational mode", styles["h2"]))
    story.append(_p(
        "In the current project, the RAG layer can run with a local deterministic fallback or with an external reasoning model when an API key is available. This makes it practical for both offline demo use and connected professional use.",
        styles["body"],
    ))


def _flowchart(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("Process Flowchart: Industry Process vs Proposed System", styles["h1"]))
    data = [
        [_p("Stage", styles["head"]), _p("Industry Process", styles["head"]), _p("Proposed Project Process", styles["head"])],
        [_p("1", styles["cell"]), _p("Site engineer visually identifies defect.", styles["cell"]), _p("YOLO/Roboflow detects defect from image or manual annotation is used for validation.", styles["cell"])],
        [_p("2", styles["cell"]), _p("Defect type is classified manually.", styles["cell"]), _p("Model predicts defect class such as crack, spall, mold or rebar.", styles["cell"])],
        [_p("3", styles["cell"]), _p("Tape/ruler/crack gauge/depth probe used for dimensions.", styles["cell"]), _p("Pixel dimensions are converted to real units using scale reference where available.", styles["cell"])],
        [_p("4", styles["cell"]), _p("Engineer decides severity based on width, area, depth and exposed steel.", styles["cell"]), _p("Severity is calculated using rule-based engineering thresholds.", styles["cell"])],
        [_p("5", styles["cell"]), _p("Repair method selected from experience/standards.", styles["cell"]), _p("The retrieval-based reasoning layer prepares a method statement from defect context and engineering references.", styles["cell"])],
        [_p("6", styles["cell"]), _p("Quantity surveyor prepares BOQ from norms and rates.", styles["cell"]), _p("The system retrieves norms and rates and computes BOQ line items with quantity x rate logic.", styles["cell"])],
        [_p("7", styles["cell"]), _p("Cost = quantity x rate, checked by engineer/contractor.", styles["cell"]), _p("Cost = measured quantity x retrieved norm x retrieved unit rate; final field validation remains essential.", styles["cell"])],
    ]
    story.append(_table(data, [18 * mm, 78 * mm, 78 * mm]))
    story.append(Spacer(1, 8))
    story.append(_p(
        "Flow sequence: Image capture -> defect detection -> dimension measurement -> severity classification -> RAG retrieval -> remedy selection -> BOQ preparation -> cost estimation -> industry validation.",
        styles["body"],
    ))


def _model_eval(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("Model Evaluation Summary from Roboflow Screenshot", styles["h1"]))
    data = [
        [_p("Model", styles["head"]), _p("Task / Type", styles["head"]), _p("mAP@50", styles["head"]), _p("Precision", styles["head"]), _p("Recall", styles["head"]), _p("F1", styles["head"])],
        [_p("Honeycomb Concrete 1", styles["cell"]), _p("YOLOv11 Instance Segmentation (Nano)", styles["cell"]), _p("99.5%", styles["cell"]), _p("100.0%", styles["cell"]), _p("99.5%", styles["cell"]), _p("99.7%", styles["cell"])],
        [_p("training-dataset 3", styles["cell"]), _p("Roboflow 3.0 Object Detection (Fast)", styles["cell"]), _p("77.9%", styles["cell"]), _p("83.8%", styles["cell"]), _p("73.3%", styles["cell"]), _p("78.2%", styles["cell"])],
        [_p("training-dataset 2", styles["cell"]), _p("YOLOv11 Object Detection (Fast)", styles["cell"]), _p("79.8%", styles["cell"]), _p("90.7%", styles["cell"]), _p("73.1%", styles["cell"]), _p("81.0%", styles["cell"])],
    ]
    story.append(_table(data, [45 * mm, 55 * mm, 20 * mm, 22 * mm, 20 * mm, 18 * mm]))
    story.append(_p(
        "The best visible validation result is Honeycomb Concrete 1 with 99.5% mAP@50, 100% precision, 99.5% recall and 99.7% F1-score. "
        "The object detection models show moderate performance around 78-80% mAP@50, suggesting that dataset improvement and additional labelled examples can improve recall.",
        styles["body"],
    ))


def _model_comparison(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("YOLOv11 vs YOLOv8 vs Faster R-CNN", styles["h1"]))
    data = [
        [_p("Parameter", styles["head"]), _p("YOLOv8", styles["head"]), _p("YOLOv11", styles["head"]), _p("Faster R-CNN", styles["head"])],
        [_p("Detector type", styles["cell"]), _p("One-stage", styles["cell"]), _p("One-stage", styles["cell"]), _p("Two-stage", styles["cell"])],
        [_p("Speed", styles["cell"]), _p("Fast", styles["cell"]), _p("Very fast", styles["cell"]), _p("Slower", styles["cell"])],
        [_p("Deployment", styles["cell"]), _p("Easy with Ultralytics", styles["cell"]), _p("Easy with Ultralytics/Roboflow", styles["cell"]), _p("More complex", styles["cell"])],
        [_p("Training cost", styles["cell"]), _p("Moderate", styles["cell"]), _p("Moderate", styles["cell"]), _p("High", styles["cell"])],
        [_p("Real-time use", styles["cell"]), _p("Good", styles["cell"]), _p("Best practical choice", styles["cell"]), _p("Limited", styles["cell"])],
        [_p("Use in project", styles["cell"]), _p("Baseline option", styles["cell"]), _p("Preferred detector", styles["cell"]), _p("Research comparison baseline", styles["cell"])],
    ]
    story.append(_table(data, [35 * mm, 45 * mm, 48 * mm, 48 * mm]))
    story.append(_p(
        "YOLOv11 is preferred because it provides real-time inference, simpler deployment and strong detection performance. Faster R-CNN can be discussed as a baseline, but it is less practical for a lightweight demo or real-time site workflow.",
        styles["body"],
    ))


def _dataset_eval_notes(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("Train/Validation/Test Split, Cross-Validation and Curves", styles["h1"]))
    rows = [
        [_p("Topic", styles["head"]), _p("Report Explanation", styles["head"])],
        [_p("Train/Validation/Test Split", styles["cell"]), _p("Dataset is normally divided into train, validation and test sets. Train learns model weights, validation monitors training, and test is used for final unbiased evaluation. Common split: 70/20/10 or 80/10/10.", styles["cell"])],
        [_p("Cross-Validation", styles["cell"]), _p("Not used in the current YOLO workflow because k-fold cross-validation requires multiple full training runs. It can be future work for stronger statistical validation.", styles["cell"])],
        [_p("Confusion Matrix", styles["cell"]), _p("Requires ground-truth labels and predictions. It shows correct detections on the diagonal and misclassifications off-diagonal. Export from Roboflow evaluation page or generate locally using labelled test data.", styles["cell"])],
        [_p("Precision-Recall Curve", styles["cell"]), _p("Shows precision-recall trade-off at different confidence thresholds. It is needed to decide whether the model should prioritize fewer false alarms or fewer missed defects.", styles["cell"])],
    ]
    story.append(_table(rows, [42 * mm, 135 * mm]))


def _summary_results(story: list[Any], rows: list[dict[str, str]], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("End-to-End Detection, Cost and RAG Results", styles["h1"]))
    data = [[_p("Image", styles["head"]), _p("Defect", styles["head"]), _p("Conf.", styles["head"]), _p("Severity", styles["head"]), _p("Quantity", styles["head"]), _p("Total Cost", styles["head"]), _p("OpenAI", styles["head"] )]]
    for row in rows:
        data.append([
            _p(row.get("image", ""), styles["cell"]),
            _p(row.get("defect", ""), styles["cell"]),
            _p(row.get("confidence", ""), styles["cell"]),
            _p(row.get("severity", ""), styles["cell"]),
            _p(row.get("repair_quantity", ""), styles["cell"]),
            _p(_money(row.get("total_repair_cost", "")), styles["cell"]),
            _p(f"{row.get('rag_used_openai')} {row.get('rag_model')}", styles["cell"]),
        ])
    story.append(_table(data, [34 * mm, 22 * mm, 15 * mm, 20 * mm, 36 * mm, 28 * mm, 30 * mm]))


def _image_evidence(story: list[Any], rows: list[dict[str, str]], input_dir: Path, styles: dict[str, ParagraphStyle]) -> None:
    story.append(PageBreak())
    story.append(_p("Annotated Image Evidence", styles["h1"]))
    image_names: list[str] = []
    for row in rows:
        name = row.get("image", "")
        if name and name not in image_names:
            image_names.append(name)
    for name in image_names:
        story.append(_p(name, styles["h2"]))
        story.append(_image(input_dir / f"{Path(name).stem}_annotated.jpg", 175 * mm, 100 * mm))
        story.append(Spacer(1, 8))


def _boq_section(story: list[Any], boq: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    if not boq or not boq.get("norms_found", False):
        story.append(_p("No norms-based BOQ available for this defect/severity.", styles["body"]))
        return
    story.append(_p("BOQ and Rate Analysis", styles["h2"]))
    story.append(_p(f"Remedy: {boq.get('remedy', '')}", styles["body"]))
    story.append(_p(f"Work quantity: {_num(boq.get('work_quantity'))} {boq.get('work_unit', '')}", styles["body"]))
    story.append(_p(f"Norms/rates source: {boq.get('source', '')}", styles["body"]))

    data = [[_p("Cat.", styles["head"]), _p("Item", styles["head"]), _p("Norm", styles["head"]), _p("Qty", styles["head"]), _p("Rate", styles["head"]), _p("Amount", styles["head"] )]]
    for line in boq.get("lines", []):
        data.append([
            _p(line.get("category", ""), styles["cell"]),
            _p(line.get("description", ""), styles["cell"]),
            _p(f"{line.get('norm', '')} {line.get('norm_unit', '')}", styles["cell"]),
            _p(f"{line.get('quantity', '')} {line.get('quantity_unit', '')}", styles["cell"]),
            _p(_money(line.get("rate", "")), styles["cell"]),
            _p(_money(line.get("amount", "")), styles["cell"]),
        ])
    story.append(_table(data, [12 * mm, 45 * mm, 34 * mm, 25 * mm, 25 * mm, 30 * mm]))
    totals = [
        [_p("Material", styles["cell"]), _p(_money(boq.get("material_total")), styles["cell"]), _p("Labour", styles["cell"]), _p(_money(boq.get("labour_total")), styles["cell"]), _p("Equipment", styles["cell"]), _p(_money(boq.get("equipment_total")), styles["cell"])],
        [_p("Subtotal", styles["cell"]), _p(_money(boq.get("subtotal")), styles["cell"]), _p("Overheads 15%", styles["cell"]), _p(_money(boq.get("overheads")), styles["cell"]), _p("GST 18%", styles["cell"]), _p(_money(boq.get("gst")), styles["cell"])],
        [_p("GRAND TOTAL", styles["head"]), _p(_money(boq.get("grand_total")), styles["head"]), _p("", styles["head"]), _p("", styles["head"]), _p("", styles["head"]), _p("", styles["head"])],
    ]
    total_table = _table(totals, [28 * mm, 30 * mm, 28 * mm, 30 * mm, 28 * mm, 30 * mm], header=False)
    total_table.setStyle(TableStyle([("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#12315a"))]))
    story.append(Spacer(1, 5))
    story.append(total_table)
    story.append(_p(f"Method statement: {boq.get('method_steps', '')}", styles["body"]))


def _detail_sections(story: list[Any], rows: list[dict[str, str]], styles: dict[str, ParagraphStyle]) -> None:
    for index, row in enumerate(rows, start=1):
        story.append(PageBreak())
        story.append(_p(f"Detailed Result {index}: {row.get('defect', '').title()} - {row.get('image', '')}", styles["h1"]))
        detail = [
            [_p("Field", styles["head"]), _p("Value", styles["head"])],
            [_p("Defect", styles["cell"]), _p(row.get("defect", ""), styles["cell"])],
            [_p("Confidence", styles["cell"]), _p(row.get("confidence", ""), styles["cell"])],
            [_p("Severity", styles["cell"]), _p(row.get("severity", ""), styles["cell"])],
            [_p("Measurement", styles["cell"]), _p(row.get("measurement_basis", ""), styles["cell"])],
            [_p("Reason", styles["cell"]), _p(row.get("reason", ""), styles["cell"])],
            [_p("Repair Quantity", styles["cell"]), _p(row.get("repair_quantity", ""), styles["cell"])],
            [_p("Repair Time", styles["cell"]), _p(row.get("repair_time_estimate", ""), styles["cell"])],
            [_p("RAG Model", styles["cell"]), _p(f"{row.get('rag_used_openai')} ({row.get('rag_model')})", styles["cell"])],
            [_p("Sources", styles["cell"]), _p(row.get("rag_sources", ""), styles["cell"])],
        ]
        story.append(_table(detail, [36 * mm, 140 * mm]))
        story.append(Spacer(1, 8))
        _boq_section(story, _parse_boq(row.get("boq", "")), styles)
        story.append(Spacer(1, 8))
        story.append(_p("OpenAI RAG Remedy Plan", styles["h2"]))
        story.extend(_md_flowables(row.get("rag_remedy_plan", ""), styles))


def _manual_validation(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(PageBreak())
    story.append(_p("Manual and Industry Validation Plan", styles["h1"]))
    story.append(_p(
        "The remaining validation is manual field work. Select 3-5 defects, measure them physically, compare manual dimensions against system dimensions, then validate quantities and BOQ cost with an engineer/contractor/quantity surveyor.",
        styles["body"],
    ))
    data = [
        [_p("Defect", styles["head"]), _p("Manual Measurement", styles["head"]), _p("System Comparison", styles["head"])],
        [_p("Crack", styles["cell"]), _p("Tape for length; crack gauge/ruler for widths at 3 points; average and max width.", styles["cell"]), _p("Compare length, average width, max width and severity band.", styles["cell"])],
        [_p("Spalling", styles["cell"]), _p("Tape/ruler for length and width; depth probe/steel scale for depth.", styles["cell"]), _p("Compare area = L x W and volume = area x depth.", styles["cell"])],
        [_p("Exposed Rebar", styles["cell"]), _p("Measure patch length/width, exposed bar length and count of bars.", styles["cell"]), _p("Compare cover restoration area and repair method.", styles["cell"])],
        [_p("Mold/Dampness", styles["cell"]), _p("Measure affected wall height and width.", styles["cell"]), _p("Compare affected area and treatment quantity.", styles["cell"])],
    ]
    story.append(_table(data, [28 * mm, 75 * mm, 75 * mm]))
    story.append(_p(
        "Difference percentage formula: |System value - Manual value| / Manual value x 100. Industry cost validation should compare system BOQ against a contractor/engineer estimate or previous rework bill.",
        styles["body"],
    ))


def _conclusion(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(_p("Conclusion", styles["h1"]))
    story.append(_p(
        "The system demonstrates an end-to-end workflow from defect image evidence to severity, quantity, BOQ, cost and retrieval-based remedy generation. The current report shows AI-assisted reasoning output for all detections and includes material, labour and equipment breakdowns. Final billing-level use requires physical site measurement and industry validation of rates and quantities.",
        styles["body"],
    ))


def build_report(input_dir: Path, output_pdf: Path) -> None:
    rows = _load_rows(input_dir)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
    )
    story: list[Any] = []
    story.append(_p("Construction Defect Detection using YOLO + RAG + BOQ", styles["title"]))
    story.append(_p("Final Professor Demonstration Report", styles["subtitle"]))
    _overview(story, styles)
    _flowchart(story, styles)
    _rag_documentation(story, styles)
    story.append(PageBreak())
    _model_eval(story, styles)
    _model_comparison(story, styles)
    _dataset_eval_notes(story, styles)
    story.append(PageBreak())
    _summary_results(story, rows, styles)
    _image_evidence(story, rows, input_dir, styles)
    _detail_sections(story, rows, styles)
    _manual_validation(story, styles)
    _conclusion(story, styles)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final professor demonstration report PDF")
    parser.add_argument("--input-dir", default="outputs/testimages-end-to-end")
    parser.add_argument("--output", default="outputs/final-professor-report.pdf")
    args = parser.parse_args()
    build_report(ROOT / args.input_dir, ROOT / args.output)
    print(f"PDF_CREATED: {ROOT / args.output}")


if __name__ == "__main__":
    main()