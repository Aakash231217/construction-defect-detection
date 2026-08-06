from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "construction-defect-journal-paper.pdf"
RESULTS_CSV = ROOT / "outputs" / "detection_severity_report.csv"
ANNOTATED_DIR = ROOT / "outputs" / "annotated"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F6FDB")
LIGHT_BLUE = colors.HexColor("#EAF2FF")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5B6675")
GRID = colors.HexColor("#C8D1DC")
PALE = colors.HexColor("#F5F7FA")
GREEN = colors.HexColor("#2F9E63")
ORANGE = colors.HexColor("#E08A1E")
RED = colors.HexColor("#D83A52")


def _escape(value: Any) -> str:
    return (
        "" if value is None else str(value)
    ).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=6,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.6,
            alignment=TA_JUSTIFY,
            leftIndent=8 * mm,
            rightIndent=8 * mm,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Subsubsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            textColor=INK,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11.5,
            leftIndent=12,
            firstLineIndent=-7,
            textColor=INK,
            spaceAfter=3,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=8.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=NAVY,
            backColor=colors.HexColor("#F4F7FB"),
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.3,
            leading=9.5,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=INK,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.9,
            leading=8.7,
            textColor=INK,
        ),
        "head": ParagraphStyle(
            "Head",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.9,
            leading=8.7,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            leftIndent=12,
            firstLineIndent=-12,
            textColor=INK,
            spaceAfter=3,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escape(text), style)


def _rich(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph("&#8226; " + _escape(text), styles["bullet"])


def _section(number: str, title: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return _p(f"{number}. {title}", styles["h1"])


def _subsection(number: str, title: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return _p(f"{number} {title}", styles["h2"])


def _table(
    data: list[list[Any]],
    widths: list[float],
    *,
    header: bool = True,
    font_size: float = 7,
) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, splitByRow=True)
    commands: list[tuple[Any, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    table.setStyle(TableStyle(commands))
    return table


def _box(drawing: Drawing, x: float, y: float, w: float, h: float, label: str, fill: Any = LIGHT_BLUE) -> None:
    drawing.add(Rect(x, y, w, h, fillColor=fill, strokeColor=BLUE, strokeWidth=1.1, rx=3, ry=3))
    lines = label.split("\n")
    start_y = y + h / 2 + (len(lines) - 1) * 4
    for index, line in enumerate(lines):
        drawing.add(
            String(
                x + w / 2,
                start_y - index * 9,
                line,
                fontName="Helvetica-Bold",
                fontSize=6.8,
                textAnchor="middle",
                fillColor=NAVY,
            )
        )


def _connector(drawing: Drawing, x1: float, y1: float, x2: float, y2: float) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=1.2))


def architecture_diagram() -> Drawing:
    drawing = Drawing(520, 205)
    top = [
        (5, 145, 78, 38, "Site image\ninput"),
        (105, 145, 88, 38, "Primary visual\ndetector"),
        (215, 145, 88, 38, "Secondary AI\nvision engine"),
        (325, 145, 88, 38, "Class alias +\nIoU fusion"),
        (435, 145, 78, 38, "Fused defect\nset"),
    ]
    for item in top:
        _box(drawing, *item)
    for left, right in zip(top, top[1:]):
        _connector(drawing, left[0] + left[2], left[1] + left[3] / 2, right[0], right[1] + right[3] / 2)

    lower = [
        (35, 55, 88, 38, "Structural\nelement"),
        (145, 55, 88, 38, "Severity +\nmeasurement"),
        (255, 55, 88, 38, "Quantity +\nBOQ"),
        (365, 55, 88, 38, "RAG remedy +\nmethod"),
    ]
    for item in lower:
        _box(drawing, *item, fill=colors.HexColor("#F5F9F6"))
    for left, right in zip(lower, lower[1:]):
        _connector(drawing, left[0] + left[2], left[1] + left[3] / 2, right[0], right[1] + right[3] / 2)
    _connector(drawing, 474, 145, 474, 115)
    _connector(drawing, 474, 115, 79, 115)
    _connector(drawing, 79, 115, 79, 93)
    drawing.add(String(260, 15, "Annotated evidence | severity | quantities | cost | time | grounded remedy", fontName="Helvetica-Oblique", fontSize=7.2, textAnchor="middle", fillColor=MUTED))
    return drawing


def fusion_diagram() -> Drawing:
    drawing = Drawing(500, 165)
    _box(drawing, 5, 95, 100, 42, "Primary\npredictions")
    _box(drawing, 5, 25, 100, 42, "Secondary\npredictions")
    _box(drawing, 155, 60, 105, 42, "Canonicalize\nclass aliases", fill=colors.HexColor("#FFF7E8"))
    _box(drawing, 310, 60, 105, 42, "Same class +\nIoU >= 0.50?", fill=colors.HexColor("#FFF7E8"))
    _box(drawing, 445, 60, 50, 42, "Merge", fill=colors.HexColor("#F1F8F3"))
    _connector(drawing, 105, 116, 155, 81)
    _connector(drawing, 105, 46, 155, 81)
    _connector(drawing, 260, 81, 310, 81)
    _connector(drawing, 415, 81, 445, 81)
    drawing.add(String(363, 35, "Duplicate: retain primary", fontName="Helvetica", fontSize=6.8, textAnchor="middle", fillColor=MUTED))
    drawing.add(String(363, 22, "Different class/location: retain both", fontName="Helvetica", fontSize=6.8, textAnchor="middle", fillColor=MUTED))
    return drawing


def rag_diagram() -> Drawing:
    drawing = Drawing(500, 190)
    _box(drawing, 5, 120, 92, 42, "Defect + severity\n+ measurement")
    _box(drawing, 122, 120, 92, 42, "Engineering\nknowledge base")
    _box(drawing, 239, 120, 92, 42, "Repair norms +\nunit rates")
    _box(drawing, 356, 120, 92, 42, "Lexical scoring\nTop-k = 4")
    _box(drawing, 204, 45, 105, 42, "Grounded prompt\n+ computed BOQ", fill=colors.HexColor("#FFF7E8"))
    _box(drawing, 355, 45, 105, 42, "AI answer or\ndeterministic fallback", fill=colors.HexColor("#F1F8F3"))
    for left, right in [(97, 122), (214, 239), (331, 356)]:
        _connector(drawing, left, 141, right, 141)
    _connector(drawing, 402, 120, 402, 102)
    _connector(drawing, 402, 102, 256, 102)
    _connector(drawing, 256, 102, 256, 87)
    _connector(drawing, 309, 66, 355, 66)
    drawing.add(String(250, 12, "Generation is constrained by retrieved context and auditable BOQ arithmetic", fontName="Helvetica-Oblique", fontSize=7.2, textAnchor="middle", fillColor=MUTED))
    return drawing


def _load_results() -> list[dict[str, str]]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_results(rows: list[dict[str, str]]) -> dict[str, Any]:
    source_counts = Counter(row.get("detection_source", "Unknown") for row in rows)
    severity_counts = Counter(row.get("severity", "Unknown") for row in rows)
    element_counts = Counter(row.get("structural_element", "Unknown") for row in rows)
    defect_counts = Counter(row.get("defect", "Unknown") for row in rows)
    cost_by_defect: dict[str, float] = defaultdict(float)
    total_cost = 0.0
    for row in rows:
        cost = float(row.get("est_cost_inr") or 0.0)
        total_cost += cost
        cost_by_defect[row.get("defect", "Unknown")] += cost
    return {
        "findings": len(rows),
        "images": len({row.get("image", "") for row in rows}),
        "source_counts": source_counts,
        "severity_counts": severity_counts,
        "element_counts": element_counts,
        "defect_counts": defect_counts,
        "cost_by_defect": dict(cost_by_defect),
        "total_cost": total_cost,
    }


def bar_chart(data: dict[str, float], title: str, *, width: float = 500, height: float = 180) -> Drawing:
    drawing = Drawing(width, height)
    drawing.add(String(8, height - 14, title, fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
    if not data:
        drawing.add(String(width / 2, height / 2, "No result data available", fontName="Helvetica", fontSize=8, textAnchor="middle", fillColor=MUTED))
        return drawing
    labels = list(data)
    values = [float(data[label]) for label in labels]
    maximum = max(values) or 1.0
    chart_left = 38
    chart_bottom = 42
    chart_width = width - 55
    chart_height = height - 72
    slot = chart_width / max(len(labels), 1)
    bar_width = min(44, slot * 0.58)
    palette = [BLUE, ORANGE, GREEN, RED, NAVY, colors.HexColor("#7A54D0")]
    drawing.add(Line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height, strokeColor=GRID, strokeWidth=0.8))
    drawing.add(Line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom, strokeColor=GRID, strokeWidth=0.8))
    for index, (label, value) in enumerate(zip(labels, values)):
        x = chart_left + index * slot + (slot - bar_width) / 2
        bar_height = chart_height * value / maximum
        drawing.add(Rect(x, chart_bottom, bar_width, bar_height, fillColor=palette[index % len(palette)], strokeColor=colors.white, strokeWidth=0.4))
        drawing.add(String(x + bar_width / 2, chart_bottom + bar_height + 4, f"{value:,.0f}", fontName="Helvetica-Bold", fontSize=6.5, textAnchor="middle", fillColor=INK))
        short_label = label.replace("exposed_reinforcement", "exposed rebar")
        if len(short_label) > 14:
            short_label = short_label[:13] + "."
        drawing.add(String(x + bar_width / 2, chart_bottom - 12, short_label, fontName="Helvetica", fontSize=6.2, textAnchor="middle", fillColor=MUTED))
    return drawing


def _scaled_image(path: Path, max_width: float, max_height: float) -> Flowable:
    if not path.exists():
        return Paragraph(f"Image unavailable: {_escape(path.name)}", _styles()["small"])
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def _footer(canvas: Any, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(16 * mm, 13 * mm, page_width - 16 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 8.5 * mm, "Journal manuscript: image-to-remedy construction defect decision support")
    canvas.drawRightString(page_width - 16 * mm, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _paragraphs(story: list[Any], paragraphs: Iterable[str], styles: dict[str, ParagraphStyle]) -> None:
    for text in paragraphs:
        story.append(_p(text, styles["body"]))


def build_paper(output_path: Path) -> None:
    styles = _styles()
    rows = _load_results()
    aggregate = _aggregate_results(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title="Integrated Construction Defect Detection, Severity Assessment, BOQ Costing and Retrieval-Augmented Remediation",
        author="Anonymous manuscript for academic review",
        subject="Journal-style research manuscript generated from implemented project evidence",
    )
    story: list[Any] = []

    story.append(Spacer(1, 12 * mm))
    story.append(_p("Integrated Visual Detection, Engineering Severity Assessment, Quantity-Based BOQ Costing, and Retrieval-Augmented Remediation for Concrete Surface Defects", styles["title"]))
    story.append(_p("Anonymous manuscript for academic review", styles["author"]))
    story.append(_p("Research prototype in construction informatics and civil-engineering decision support", styles["author"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_rich("<b>Abstract</b>", styles["abstract"]))
    abstract = (
        "Visual defect detection systems often terminate at a class label and bounding box, leaving engineers to independently interpret severity, quantify repair work, prepare rate analysis, and select a remedial method. This paper presents an implemented image-to-remedy decision-support prototype that integrates a hosted visual detector, a supplementary multimodal vision engine, class canonicalization and intersection-over-union fusion, structural-element classification, standards-informed severity rules, image-derived quantity estimation, norms-based bill-of-quantities computation, working-duration estimation, and retrieval-augmented remedy generation. The implemented detector vocabulary covers cracking, spalling, honeycombing, exposed reinforcement, mold or dampness, and efflorescence. Crack grading uses scaled width when a reference is supplied; otherwise all classes use explicitly marked area-ratio heuristics. BOQ amounts are not retrieved as final costs: item quantities are computed from work quantity and consumption norms, multiplied by stored unit rates, and aggregated with overheads and tax. The retrieval layer uses lexical relevance, defect and severity boosts, a computed BOQ block, and a deterministic fallback to constrain generated remedies. Deterministic tests validate detector fusion, alias normalization, severity logic, BOQ arithmetic, and retrieval behavior. A checked-in twelve-image demonstration contains seventeen findings and a planning-cost total of INR 44,416; these outputs demonstrate system integration rather than detector accuracy or contractual cost validity. Repository-reported model metrics are presented separately from reproducible evidence. The prototype shows how visual inspection can be connected to auditable engineering actions, while also identifying the need for labeled test-set evaluation, segmentation-based measurement, site calibration, current-rate validation, and independent professional review before field deployment."
    )
    story.append(_p(abstract, styles["abstract"]))
    story.append(_rich("<b>Keywords:</b> concrete defects; YOLO; computer vision; severity assessment; quantity estimation; bill of quantities; rate analysis; retrieval-augmented generation; repair planning; construction informatics", styles["abstract"]))
    story.append(Spacer(1, 4 * mm))
    highlights = [
        "A complete image-to-remedy pipeline links visual findings to severity, quantity, BOQ, cost, time, and method statements.",
        "Primary and supplementary detections are fused after alias normalization, retaining different-class overlaps and removing same-class duplicates at IoU >= 0.50.",
        "Displayed costs are auditable sums of material, labour, equipment, overhead, and tax rather than unconstrained language-model estimates.",
        "Retrieval-grounded remedy generation remains operational through a deterministic fallback when the external reasoning engine is unavailable.",
    ]
    story.append(_rich("<b>Research highlights</b>", styles["body"]))
    for item in highlights:
        story.append(_bullet(item, styles))
    story.append(PageBreak())

    story.append(_section("1", "Introduction", styles))
    _paragraphs(
        story,
        [
            "Concrete infrastructure inspection is still dominated by visual surveys, manual measurements, engineering judgment, and separately prepared repair estimates. Although object detectors can rapidly localize visible deterioration, a bounding box alone does not answer the questions that control intervention: How severe is the condition? What quantity should be repaired? Which materials, labour, and equipment are required? What is the preliminary cost and duration? Which method statement is defensible against recognized guidance?",
            "This separation creates a decision gap between automated perception and engineering action. A detector may achieve useful precision while remaining operationally incomplete because its output is not connected to measurement assumptions, standards, quantities, rates, or traceable remedy evidence. Conversely, unconstrained generative systems may produce fluent repair advice without preserving arithmetic consistency or source provenance. The present work addresses this gap through an integrated and auditable prototype.",
            "The system accepts an image, obtains primary and supplementary visual detections, normalizes class aliases, fuses detections, estimates a structural-element label, grades severity, estimates repair quantity, retrieves consumption norms and unit rates, builds a BOQ, estimates working duration, retrieves relevant engineering guidance, and renders a grounded remedy plan. Every output remains preliminary when physical scale, depth, cover, and field observations are unavailable.",
        ],
        styles,
    )
    story.append(_subsection("1.1", "Research questions", styles))
    for item in [
        "RQ1: Can heterogeneous detector outputs be fused without losing a defect class missed by the primary model?",
        "RQ2: Can image detections be transformed into explicit severity and quantity estimates with transparent assumptions?",
        "RQ3: Can repair cost be generated through auditable quantity x norm x rate arithmetic rather than free-form estimation?",
        "RQ4: Can retrieved engineering context and computed BOQ data constrain remedy generation while preserving an offline fallback?",
    ]:
        story.append(_bullet(item, styles))
    story.append(_subsection("1.2", "Contributions", styles))
    for item in [
        "A modular detector-fusion pipeline covering six visible defect classes and seven structural-element labels.",
        "Standards-informed severity rules with scale-aware crack grading and clearly labeled area-ratio fallbacks.",
        "Defect-specific quantity estimation and norms-based BOQ generation with material, labour, equipment, overhead, and tax breakup.",
        "A lexical retrieval and controlled-generation layer that embeds computed BOQ evidence and degrades safely to a deterministic answer.",
        "A reproducible evidence hierarchy separating deterministic tests, demo outputs, repository-reported model metrics, and future field validation.",
    ]:
        story.append(_bullet(item, styles))

    story.append(_section("2", "Background and Related Engineering Context", styles))
    _paragraphs(
        story,
        [
            "One-stage object detectors are attractive for construction inspection because they combine localization and classification in a single inference pass. The project supports hosted detection and local Ultralytics inference, while a supplementary vision engine covers classes outside the primary training distribution. The architecture is therefore hybrid: learned perception proposes visible defects, and deterministic engineering modules control severity, quantity, and arithmetic.",
            "Crack-width interpretation is informed by ACI 224R-01 and IS 456:2000. Spalling and exposed reinforcement are associated with ICRI 310.1, ACI 562, and Concrete Society TR54 repair intent. Honeycombing is interpreted using IS 456 workmanship provisions and ACI 309 consolidation guidance. These documents do not directly validate bounding-box measurements; they motivate the decision rules and required field checks.",
            "Retrieval-augmented generation is used as a presentation and synthesis layer rather than as the source of quantities or prices. Engineering chunks and norms are stored locally, retrieved by a transparent lexical score, and combined with deterministic BOQ calculations. This division is deliberate: language generation may organize a method statement, but cost arithmetic remains inspectable Python logic.",
        ],
        styles,
    )
    related = [
        [_p("Area", styles["head"]), _p("Common limitation", styles["head"]), _p("Implemented response", styles["head"])],
        [_p("Defect detection", styles["cell"]), _p("Single model misses out-of-distribution classes.", styles["cell"]), _p("Supplementary vision detector and same-class IoU fusion.", styles["cell"])],
        [_p("Severity", styles["cell"]), _p("Image class does not imply intervention priority.", styles["cell"]), _p("Rule-based severity using width, extent, depth, and cover-loss logic.", styles["cell"])],
        [_p("Costing", styles["cell"]), _p("Flat cost bands hide quantity and resource assumptions.", styles["cell"]), _p("Work quantity x resource norm x unit rate, followed by overhead and tax.", styles["cell"])],
        [_p("Generative remedy", styles["cell"]), _p("Unconstrained text can invent rates or standards.", styles["cell"]), _p("Retrieved context, embedded BOQ, explicit rules, deterministic fallback.", styles["cell"])],
    ]
    story.append(_table(related, [37 * mm, 66 * mm, 73 * mm]))
    story.append(_p("Table 1. Research gap and implemented design response.", styles["caption"]))

    story.append(PageBreak())
    story.append(_section("3", "System Architecture", styles))
    story.append(architecture_diagram())
    story.append(_p("Figure 1. End-to-end architecture from site image to grounded remedy and planning estimate.", styles["caption"]))
    _paragraphs(
        story,
        [
            "The implementation is organized around explicit ownership boundaries. Hosted inference is isolated in src/roboflow_model.py. Supplementary defect and structural-element vision prompts are isolated in src/vision_fallback.py. Fusion, manual review, grading coordination, and annotation are implemented in src/pipeline.py. Engineering decisions are separated across src/severity.py, src/cost_estimation.py, src/boq.py, and src/remedy_rag.py. The Streamlit application assembles these services for interactive use.",
            "The detector vocabulary comprises crack, spalling, honeycombing, exposed reinforcement, mold or dampness, and efflorescence. Canonical aliases map spall to spalling, honeycomb to honeycombing, exposed rebar to exposed reinforcement, and white bleeding to efflorescence. Structural-element labels comprise slab, wall, beam, column, staircase, footing, and other.",
        ],
        styles,
    )
    modules = [
        [_p("Module", styles["head"]), _p("Responsibility", styles["head"]), _p("Primary output", styles["head"])],
        [_p("roboflow_model.py", styles["cell"]), _p("Direct HTTPS hosted inference.", styles["cell"]), _p("Primary boxes, classes, confidence.", styles["cell"])],
        [_p("vision_fallback.py", styles["cell"]), _p("Supplementary defects and structural element.", styles["cell"]), _p("Secondary boxes and element label.", styles["cell"])],
        [_p("pipeline.py", styles["cell"]), _p("Class normalization, IoU fusion, grading, annotation.", styles["cell"]), _p("Fused and graded findings.", styles["cell"])],
        [_p("severity.py", styles["cell"]), _p("Engineering severity and remediation coordination.", styles["cell"]), _p("SeverityResult with cost and BOQ.", styles["cell"])],
        [_p("cost_estimation.py", styles["cell"]), _p("Repair quantity, preliminary rates, duration.", styles["cell"]), _p("Quantity and resource cost breakup.", styles["cell"])],
        [_p("boq.py", styles["cell"]), _p("Norm and rate retrieval; line-item arithmetic.", styles["cell"]), _p("Auditable BOQ and grand total.", styles["cell"])],
        [_p("remedy_rag.py", styles["cell"]), _p("Lexical retrieval and controlled remedy generation.", styles["cell"]), _p("Grounded method and source list.", styles["cell"])],
    ]
    story.append(_table(modules, [39 * mm, 82 * mm, 55 * mm]))
    story.append(_p("Table 2. Principal modules and outputs.", styles["caption"]))

    story.append(_subsection("3.1", "Detector fusion and provenance", styles))
    story.append(fusion_diagram())
    story.append(_p("Figure 2. Class-aware detector fusion. Different-class overlaps are retained because multiple conditions may coexist.", styles["caption"]))
    _paragraphs(
        story,
        [
            "Each primary prediction is retained. A secondary candidate is added unless an existing prediction has the same canonical class and an intersection-over-union of at least 0.50. Primary predictions therefore win same-class ties by insertion order. A crack and exposed reinforcement may occupy overlapping regions and are intentionally retained as separate findings. Confidence-weighted box fusion and cross-class non-maximum suppression are not implemented.",
            "The interactive application labels provenance generically as Primary detector or Secondary detector. API providers and credentials are not exposed in the frontend. If the secondary engine is unavailable, primary detections continue through the deterministic engineering pipeline.",
        ],
        styles,
    )
    story.append(_rich("IoU(A,B) = area(A intersection B) / area(A union B)", styles["equation"]))

    story.append(_section("4", "Engineering Assessment Methodology", styles))
    story.append(_subsection("4.1", "Image geometry and scale", styles))
    _paragraphs(
        story,
        [
            "Let wb and hb be the detection-box dimensions, and Wi and Hi be image dimensions. The affected-area ratio rA is used when no physical scale is available. When a reference marker is provided, the scale s converts pixel dimensions to millimetres. A bounding box is wider than an actual crack; therefore box-derived crack width is an upper bound and requires confirmation by crack gauge or segmentation.",
        ],
        styles,
    )
    story.append(_rich("rA = (wb x hb) / (Wi x Hi)     and     s = reference_mm / reference_px", styles["equation"]))
    story.append(_rich("Crack length L = max(wb, hb) x s;   bounding-box width W = min(wb, hb) x s", styles["equation"]))

    story.append(_subsection("4.2", "Severity rules", styles))
    severity_table = [
        [_p("Defect", styles["head"]), _p("Governing evidence", styles["head"]), _p("Implemented bands or escalation", styles["head"])],
        [_p("Crack", styles["cell"]), _p("Scaled width when available; otherwise area ratio.", styles["cell"]), _p("W <0.10 mm Minor; <0.30 Moderate; <0.70 Severe; otherwise Critical. Area fallback: 2%, 8%, 20%.", styles["cell"])],
        [_p("Spalling", styles["cell"]), _p("Area extent and optional depth.", styles["cell"]), _p("Area fallback: 1.5%, 6%, 15%. Depth >=25 mm escalates Severe; cover-depth reach escalates Critical.", styles["cell"])],
        [_p("Honeycombing", styles["cell"]), _p("Area extent and optional void depth.", styles["cell"]), _p("Area fallback: 1.5%, 6%, 15%. Depth >=25 mm escalates Severe.", styles["cell"])],
        [_p("Exposed reinforcement", styles["cell"]), _p("Loss of cover and affected extent.", styles["cell"]), _p("Minimum Severe; Critical at area ratio >=5% or deeper section-loss evidence.", styles["cell"])],
        [_p("Mold / dampness", styles["cell"]), _p("Affected surface extent.", styles["cell"]), _p("Area fallback: 2%, 8%, 20%; moisture source must be investigated.", styles["cell"])],
        [_p("Efflorescence", styles["cell"]), _p("Affected surface extent.", styles["cell"]), _p("Generic area heuristic in current implementation; dedicated BOQ norms are future work.", styles["cell"])],
    ]
    story.append(_table(severity_table, [35 * mm, 57 * mm, 84 * mm]))
    story.append(_p("Table 3. Implemented severity evidence and thresholds.", styles["caption"]))
    story.append(_p("Severity is a planning category rather than a structural safety certification. A large image-space box can produce a high area ratio even when the photograph is tightly cropped. Scaled field measurements and member context must supersede this fallback.", styles["body"]))

    story.append(_subsection("4.3", "Repair quantity estimation", styles))
    quantity_table = [
        [_p("Defect", styles["head"]), _p("Quantity", styles["head"]), _p("Implemented basis", styles["head"])],
        [_p("Crack", styles["cell"]), _p("Running metre", styles["cell"]), _p("Scaled visible length; severe factor 1.2 and critical factor 1.5. Unscaled fallback: 2 sqrt(rA) metres.", styles["cell"])],
        [_p("Spalling", styles["cell"]), _p("m2 or m3", styles["cell"]), _p("Scaled box area; deep severe repair may use area x depth. Unscaled fallback assumes 3 m2 photographed surface.", styles["cell"])],
        [_p("Honeycombing", styles["cell"]), _p("m2 or m3", styles["cell"]), _p("Affected surface area; optional volume where void depth is measured.", styles["cell"])],
        [_p("Exposed reinforcement", styles["cell"]), _p("m2", styles["cell"]), _p("Cover-restoration area; critical preliminary estimator applies an effective-area factor.", styles["cell"])],
        [_p("Mold / efflorescence", styles["cell"]), _p("m2", styles["cell"]), _p("Affected finish area; cause investigation remains outside image geometry.", styles["cell"])],
    ]
    story.append(_table(quantity_table, [37 * mm, 28 * mm, 111 * mm]))
    story.append(_p("Table 4. Repair quantity units and implemented assumptions.", styles["caption"]))
    story.append(_rich("Scaled area Q = (wb x s)(hb x s) / 1,000,000 m2;   unscaled fallback Q = 3 rA m2", styles["equation"]))

    story.append(_subsection("4.4", "Norms-based BOQ and final rate", styles))
    _paragraphs(
        story,
        [
            "For each repair record, local JSON data stores work unit, remedy, method steps, resource consumption norms, and unit-rate keys. The program retrieves the matching defect-severity record, derives each resource quantity, multiplies by its unit rate, and aggregates category totals. Cost is never accepted directly from the reasoning engine.",
        ],
        styles,
    )
    story.append(_rich("Resource quantity qi = work quantity Q x norm ni", styles["equation"]))
    story.append(_rich("Line amount Ci = qi x unit rate ri;   subtotal S = sum(Ci)", styles["equation"]))
    story.append(_rich("Overheads OH = 0.15 S;   GST = 0.18(S + OH);   grand total G = S + OH + GST = 1.357 S", styles["equation"]))
    story.append(_rich("Displayed final BOQ rate = G / Q", styles["equation"]))
    story.append(_p("The displayed cost is a preliminary planning total. Rates must be replaced with current local quotations or an approved schedule before billing. The BOQ currently uses surface-area norms for deep area repairs even when the preliminary estimator can express volume; this inconsistency is disclosed as a future refinement.", styles["body"]))

    story.append(_subsection("4.5", "Repair duration", styles))
    story.append(_rich("Duration = labour man-days / crew size + curing and mobilisation allowance", styles["equation"]))
    story.append(_p("The default crew size is two. Severity-dependent allowances are 0.5, 1, 2, and 3 days for Minor through Critical conditions. The implementation rounds to an integer working day and applies fallback bands when BOQ labour lines are unavailable. Sequencing, access, shutdown, weather, and inspection hold points are not modeled.", styles["body"]))

    story.append(PageBreak())
    story.append(_section("5", "Retrieval-Augmented Remedy Generation", styles))
    story.append(rag_diagram())
    story.append(_p("Figure 3. Retrieval-grounded remedy generation with deterministic fallback.", styles["caption"]))
    _paragraphs(
        story,
        [
            "The knowledge base contains defect- and severity-specific chunks for cracking, spalling, honeycombing, exposed reinforcement, and mold, together with general quantity-surveying and site-validation guidance. Retrieval is lexical and fully inspectable. It is not an embedding or vector-database implementation.",
            "The query combines defect class, severity, measurement basis, reason, initial remedial measure, repair time, quantity, quantity unit, and notes. Candidate chunks receive token-overlap relevance plus fixed boosts for exact defect, general guidance, and matching severity. The four highest-ranked chunks are inserted into the prompt together with the complete computed BOQ.",
        ],
        styles,
    )
    story.append(_rich("score = |Tq intersection Tc| / sqrt(|Tc|) + 3 Idefect + 1 Igeneral + 2 Iseverity", styles["equation"]))
    story.append(_subsection("5.1", "Generation constraints", styles))
    for item in [
        "Use only retrieved context and computed BOQ values.",
        "Show quantity x rate = amount for every cost line.",
        "Do not invent standards, norms, or rates outside the retrieved evidence.",
        "State that image quantities and rates are preliminary and require site confirmation.",
        "Return a deterministic grounded answer if the external reasoning call fails.",
    ]:
        story.append(_bullet(item, styles))
    story.append(_subsection("5.2", "Provenance and failure behavior", styles))
    story.append(_p("The remedy object retains retrieved chunks, source names, generated prompt, model-use state, and any generation error. The frontend uses provider-neutral wording. Credentials remain environment variables and are excluded from generated reports. The deterministic fallback preserves materials, labour, equipment, totals, method steps, limitations, and source citations even when no external model is available.", styles["body"]))

    story.append(_section("6", "Implementation and Deployment", styles))
    implementation = [
        [_p("Layer", styles["head"]), _p("Technology", styles["head"]), _p("Deployment role", styles["head"])],
        [_p("Interface", styles["cell"]), _p("Streamlit", styles["cell"]), _p("Upload, controls, annotations, inspection cards, tables, charts.", styles["cell"])],
        [_p("Primary vision", styles["cell"]), _p("Hosted detector or local Ultralytics weights", styles["cell"]), _p("Fast primary defect localization.", styles["cell"])],
        [_p("Supplementary vision", styles["cell"]), _p("External multimodal AI engine", styles["cell"]), _p("Out-of-distribution defect supplementation and element classification.", styles["cell"])],
        [_p("Engineering logic", styles["cell"]), _p("Python dataclasses and JSON knowledge", styles["cell"]), _p("Severity, quantity, rates, BOQ, duration, traceability.", styles["cell"])],
        [_p("Reporting", styles["cell"]), _p("ReportLab and HTML/CSV exporters", styles["cell"]), _p("Professor reports, journal manuscript, evidence archives.", styles["cell"])],
    ]
    story.append(_table(implementation, [38 * mm, 58 * mm, 80 * mm]))
    story.append(_p("Table 5. Implementation stack and deployment role.", styles["caption"]))
    _paragraphs(
        story,
        [
            "The hosted application avoids importing the heavier inference SDK and local GPU dependencies at startup. Primary hosted inference uses a direct HTTPS request. Local YOLO mode remains available where weights and Ultralytics are installed. The supplementary engine is optional: its failure does not prevent primary detections from reaching the deterministic engineering modules.",
            "Security depends on environment-managed credentials. Secret files are excluded from version control, provider names are not required in the user interface, and generated PDF artifacts contain no credential values. Exposed credentials must be revoked and rotated even when the local file is ignored by Git.",
        ],
        styles,
    )

    story.append(_section("7", "Experimental Design and Evidence Hierarchy", styles))
    evidence_table = [
        [_p("Evidence level", styles["head"]), _p("Status", styles["head"]), _p("Interpretation", styles["head"])],
        [_p("Deterministic unit and integration tests", styles["cell"]), _p("Executed", styles["cell"]), _p("Validates arithmetic, mappings, fusion, parser behavior, and RAG wiring.", styles["cell"])],
        [_p("Class-by-class image audit", styles["cell"]), _p("Executed on available representatives", styles["cell"]), _p("Demonstrates that each supported class can traverse at least one detector path; not an accuracy estimate.", styles["cell"])],
        [_p("Checked-in 12-image demonstration", styles["cell"]), _p("Available", styles["cell"]), _p("Demonstrates integrated outputs; combines primary, secondary, and manual evidence.", styles["cell"])],
        [_p("Repository-reported detector metrics", styles["cell"]), _p("Recorded from external evaluation screenshot", styles["cell"]), _p("Useful context, but not independently reproducible from local weights and labels.", styles["cell"])],
        [_p("Physical measurement and contractor validation", styles["cell"]), _p("Not yet conducted", styles["cell"]), _p("Required before claims of measurement, cost, or field accuracy.", styles["cell"])],
    ]
    story.append(_table(evidence_table, [42 * mm, 45 * mm, 89 * mm]))
    story.append(_p("Table 6. Evidence hierarchy used to avoid overstating prototype performance.", styles["caption"]))
    story.append(_p("The primary experimental principle is separation of concerns: deterministic tests support software and arithmetic claims; representative images support workflow demonstration; labeled test sets are required for detector metrics; and physical surveys are required for measurement and cost validation.", styles["body"]))

    story.append(_section("8", "Results", styles))
    story.append(_subsection("8.1", "Deterministic validation", styles))
    test_rows = [
        [_p("Test group", styles["head"]), _p("Verified behavior", styles["head"])],
        [_p("Severity and remediation", styles["cell"]), _p("Crack bands, exposed-rebar minimum severity, spalling depth path, cost breakup consistency.", styles["cell"])],
        [_p("BOQ arithmetic", styles["cell"]), _p("Resource quantity = work quantity x norm; amount = quantity x rate; category sums, overhead, tax, grand total.", styles["cell"])],
        [_p("RAG", styles["cell"]), _p("Retrieval ranking, prompt grounding, deterministic answer, optional client path.", styles["cell"])],
        [_p("Detector fusion", styles["cell"]), _p("Missed-class supplementation, same-class IoU deduplication, separate-location retention, alias normalization.", styles["cell"])],
        [_p("Workflow parsing", styles["cell"]), _p("Prediction extraction and base64 image output decoding.", styles["cell"])],
    ]
    story.append(_table(test_rows, [50 * mm, 126 * mm]))
    story.append(_p("Table 7. Deterministic behavior covered by executable tests.", styles["caption"]))
    story.append(_p("Sixteen deterministic test functions were executed successfully during the evidence review. The selected environment did not contain pytest, so executable test modules were run directly. An older archived report records four pytest cases passing in 21.07 seconds; it predates the expanded test suite.", styles["body"]))

    story.append(_subsection("8.2", "Integrated demonstration", styles))
    story.append(_p(f"The checked-in result table contains {aggregate['findings']} findings across {aggregate['images']} images. The planning-cost sum is INR {aggregate['total_cost']:,.0f}. This sum is an output of approximate image quantities and indicative rates, not a measured project bill.", styles["body"]))
    story.append(bar_chart(dict(aggregate["severity_counts"]), "Severity distribution in checked-in demonstration"))
    story.append(_p("Figure 4. Severity distribution. High counts reflect tightly cropped images and area-ratio fallback assumptions.", styles["caption"]))
    story.append(bar_chart(dict(aggregate["source_counts"]), "Finding provenance in checked-in demonstration"))
    story.append(_p("Figure 5. Finding provenance. Manual and supplementary findings are not detector ground truth.", styles["caption"]))
    story.append(bar_chart(dict(aggregate["cost_by_defect"]), "Aggregated preliminary cost by defect class"))
    story.append(_p("Figure 6. Aggregated planning cost by defect class; values must not be interpreted as contractual estimates.", styles["caption"]))

    story.append(_subsection("8.3", "Class-by-class representative audit", styles))
    audit = [
        [_p("Class", styles["head"]), _p("Representative result", styles["head"]), _p("Path", styles["head"])],
        [_p("Crack", styles["cell"]), _p("Detected at 0.55 confidence.", styles["cell"]), _p("Primary", styles["cell"])],
        [_p("Spalling", styles["cell"]), _p("Two findings at 0.62 and 0.45.", styles["cell"]), _p("Primary", styles["cell"])],
        [_p("Mold", styles["cell"]), _p("Detected at 0.81.", styles["cell"]), _p("Primary", styles["cell"])],
        [_p("Honeycombing", styles["cell"]), _p("Detected at 0.90 after targeted texture inspection prompt.", styles["cell"]), _p("Secondary", styles["cell"])],
        [_p("Exposed reinforcement", styles["cell"]), _p("Detected at 0.85 together with cracking.", styles["cell"]), _p("Secondary", styles["cell"])],
        [_p("Efflorescence", styles["cell"]), _p("Detected at 0.90 together with cracking.", styles["cell"]), _p("Secondary", styles["cell"])],
    ]
    story.append(_table(audit, [42 * mm, 91 * mm, 43 * mm]))
    story.append(_p("Table 8. Representative class audit. This is coverage evidence, not precision or recall.", styles["caption"]))

    story.append(_subsection("8.4", "Repository-reported detector metrics", styles))
    metrics = [
        [_p("Model", styles["head"]), _p("mAP@50", styles["head"]), _p("Precision", styles["head"]), _p("Recall", styles["head"]), _p("F1", styles["head"])],
        [_p("Honeycomb Concrete 1: YOLOv11 instance segmentation nano", styles["cell"]), _p("99.5%", styles["cell"]), _p("100.0%", styles["cell"]), _p("99.5%", styles["cell"]), _p("99.7%", styles["cell"])],
        [_p("Training dataset 3: hosted object detection", styles["cell"]), _p("77.9%", styles["cell"]), _p("83.8%", styles["cell"]), _p("73.3%", styles["cell"]), _p("78.2%", styles["cell"])],
        [_p("Training dataset 2: YOLOv11 object detection", styles["cell"]), _p("79.8%", styles["cell"]), _p("90.7%", styles["cell"]), _p("73.1%", styles["cell"]), _p("81.0%", styles["cell"])],
    ]
    story.append(_table(metrics, [78 * mm, 24 * mm, 25 * mm, 24 * mm, 22 * mm]))
    story.append(_p("Table 9. Metrics recorded in the repository from an external evaluation screenshot. They are not independently reproduced by this manuscript exporter.", styles["caption"]))

    story.append(_subsection("8.5", "Honeycombing case study", styles))
    case_image = ANNOTATED_DIR / "767_annotated.jpg"
    story.append(_scaled_image(case_image, 110 * mm, 100 * mm))
    story.append(_p("Figure 7. Honeycombing demonstration image with annotation. The image contains rough exposed aggregate adjacent to smoother concrete.", styles["caption"]))
    _paragraphs(
        story,
        [
            "The primary detector produced no finding on this image. The supplementary vision engine initially also returned an empty list. Raw-response inspection confirmed that the parser was not responsible. A concrete-pathology prompt requiring a two-pass scan of smooth-to-rough boundaries and partially occluded aggregate fields produced a honeycombing prediction at 0.90 confidence. The full Streamlit run then generated Critical severity from approximately 50% image extent, a preliminary total of INR 15,597, four working days, a grounded remedy, and populated charts.",
            "This case illustrates both the value and the risk of prompt-mediated visual supplementation. The result demonstrates recoverability of a missed class, but it is not a substitute for training-set expansion or labeled evaluation. Prompt tuning on a known image can overfit and must be evaluated on independent images.",
        ],
        styles,
    )

    story.append(PageBreak())
    story.append(_section("9", "Discussion", styles))
    story.append(_subsection("9.1", "Engineering value", styles))
    _paragraphs(
        story,
        [
            "The principal value of the prototype is not detector novelty; it is integration. The system preserves a chain from visible evidence to engineering interpretation and exposes assumptions that would otherwise remain implicit. A user can inspect the detection source, severity rationale, quantity basis, resource norms, rates, totals, duration, method statement, and retrieved sources in one interface.",
            "The dual-detector design improves class coverage while preserving primary predictions. Class-aware deduplication prevents the common spall versus spalling alias from creating duplicate findings. Different-class overlap is retained because cracking, spalling, and exposed reinforcement can coexist physically.",
            "The BOQ design is stronger than a flat rate band because it reveals why cost changes. Material consumption, labour man-days, and equipment days scale with work quantity. Overheads and tax are explicit. The final displayed rate and remedy grand total therefore reconcile by construction.",
        ],
        styles,
    )
    story.append(_subsection("9.2", "Interpretability and provenance", styles))
    story.append(_p("Interpretability is achieved through deterministic intermediate artifacts rather than through an explanation generated after the fact. Severity reasons state the governing measurement; quantities identify units and assumptions; BOQ lines preserve norm and unit rate; remedy output preserves retrieved sources. Detector provenance distinguishes primary, secondary, and manual findings in archived reports.", styles["body"]))
    story.append(_subsection("9.3", "Operational trade-offs", styles))
    tradeoffs = [
        [_p("Decision", styles["head"]), _p("Benefit", styles["head"]), _p("Cost or risk", styles["head"])],
        [_p("Always supplement hosted detection", styles["cell"]), _p("Finds classes missed even when primary returns another class.", styles["cell"]), _p("Additional latency, cost, and possible false positives.", styles["cell"])],
        [_p("Retain cross-class overlaps", styles["cell"]), _p("Represents coexisting deterioration.", styles["cell"]), _p("Can retain semantic disagreements requiring review.", styles["cell"])],
        [_p("Area-ratio fallback", styles["cell"]), _p("Produces a result without scale.", styles["cell"]), _p("Highly sensitive to framing; not a physical measurement.", styles["cell"])],
        [_p("Lexical RAG", styles["cell"]), _p("Transparent, dependency-light, deterministic scoring.", styles["cell"]), _p("Limited semantic recall compared with validated embeddings.", styles["cell"])],
        [_p("JSON rate database", styles["cell"]), _p("Auditable and easy to revise.", styles["cell"]), _p("Rates can become stale and are not location-aware.", styles["cell"])],
    ]
    story.append(_table(tradeoffs, [47 * mm, 64 * mm, 65 * mm]))
    story.append(_p("Table 10. Principal design trade-offs.", styles["caption"]))

    story.append(_section("10", "Threats to Validity and Limitations", styles))
    limitations = [
        "No labeled local test set, local weights, confusion matrix, or precision-recall curves are available to reproduce detector metrics.",
        "The twelve-image demonstration mixes primary, secondary, and manual findings and cannot estimate accuracy.",
        "Bounding boxes do not measure crack width; scaled box width is an upper bound. Segmentation or calibrated crack gauges are required.",
        "Unscaled area assumes a 3 m2 photographed surface and crack length assumes a 2 m image width; both are planning heuristics.",
        "A default 40 mm cover is used when actual cover is absent; cover depends on member, exposure, fire, and code provisions.",
        "The displayed BOQ uses surface-area norms for area repairs even when deep repair volume is estimated elsewhere.",
        "Efflorescence is detectable but lacks a dedicated BOQ and remedy-knowledge record; it currently follows generic handling.",
        "Structural-element classification has no labeled accuracy study and may return other or unknown.",
        "External vision and reasoning services introduce latency, cost, nondeterminism, availability, privacy, and model-change risks.",
        "Rate data is indicative and not automatically synchronized with an approved local schedule of rates.",
        "No physical measurement agreement, contractor-estimate comparison, or engineer inter-rater study has been completed.",
    ]
    for item in limitations:
        story.append(_bullet(item, styles))

    story.append(_section("11", "Proposed Field Validation Protocol", styles))
    _paragraphs(
        story,
        [
            "A defensible next experiment should separate detector accuracy, measurement accuracy, severity agreement, and cost agreement. At least 100 independently labeled images per major class should be reserved as a locked test set. Multiple defects in one image must be annotated separately. Evaluators should report per-class precision, recall, average precision, F1 score, confusion matrix, and confidence-threshold sensitivity for primary, secondary, and fused outputs.",
            "For physical validation, select representative cracks, spalls, honeycombed areas, exposed reinforcement, and moisture-related defects. Record crack length and width, area, depth, cover, bar exposure, and structural element using calibrated instruments. Compare image-derived and manual values using absolute error, percentage error, mean absolute percentage error where valid, and Bland-Altman analysis for repeated measurements.",
            "Severity agreement should be assessed independently by at least two civil or structural engineers. Cohen's kappa or weighted kappa can quantify ordinal agreement between system and expert grades. BOQ quantities should be compared against a quantity surveyor take-off, and rates against a dated local schedule or contractor quotation.",
        ],
        styles,
    )
    validation = [
        [_p("Study", styles["head"]), _p("Reference", styles["head"]), _p("Suggested metric", styles["head"])],
        [_p("Detection", styles["cell"]), _p("Independent labeled test set", styles["cell"]), _p("Per-class AP, precision, recall, F1; PR curves.", styles["cell"])],
        [_p("Dimensions", styles["cell"]), _p("Crack gauge, ruler, depth probe, calibrated marker", styles["cell"]), _p("MAE, percentage error, agreement limits.", styles["cell"])],
        [_p("Severity", styles["cell"]), _p("Two or more engineers", styles["cell"]), _p("Weighted kappa; disagreement review.", styles["cell"])],
        [_p("BOQ", styles["cell"]), _p("Quantity surveyor take-off", styles["cell"]), _p("Quantity variance by line item.", styles["cell"])],
        [_p("Cost", styles["cell"]), _p("Approved schedule / contractor quote", styles["cell"]), _p("Absolute and percentage difference.", styles["cell"])],
    ]
    story.append(_table(validation, [38 * mm, 75 * mm, 63 * mm]))
    story.append(_p("Table 11. Proposed validation studies and metrics.", styles["caption"]))

    story.append(_section("12", "Safety, Ethics, and Deployment Governance", styles))
    _paragraphs(
        story,
        [
            "The system is a preliminary decision-support tool and must not authorize occupancy, load changes, shoring removal, or permanent repair. Severe and Critical findings require engineer review. A false negative may leave deterioration untreated, while a false positive may trigger unnecessary cost; both consequences require human oversight.",
            "Site images may contain people, locations, asset identifiers, or confidential infrastructure details. Deployments should define image retention, access control, encryption, provider data-use terms, and deletion policies. API keys must be stored in deployment secrets, excluded from repositories and generated artifacts, and rotated immediately after exposure.",
            "Model and knowledge-base versions should be recorded with each report. Changes to prompts, rates, norms, thresholds, or external model versions can alter outputs. A production system therefore requires configuration versioning, audit logs, approval workflows, and periodic rate review.",
        ],
        styles,
    )

    story.append(_section("13", "Conclusion", styles))
    story.append(_p("This work demonstrates a practical image-to-remedy architecture for visible concrete defects. The implemented contribution is an auditable connection between heterogeneous visual detections and engineering outputs: severity, quantity, BOQ, rate analysis, duration, and retrieval-grounded remedy planning. The dual-detector path expands class coverage; canonicalization and IoU fusion preserve distinct conditions while removing duplicate aliases; deterministic arithmetic prevents generated text from controlling cost. The resulting prototype is suitable for academic demonstration and preliminary planning, but not for structural certification or contractual billing. Its next research stage is an independently labeled detector evaluation, calibrated field measurement study, engineer agreement analysis, and local-rate validation. With these controls, the architecture provides a credible foundation for transparent construction-maintenance decision support.", styles["body"]))

    story.append(PageBreak())
    story.append(_section("14", "References", styles))
    references = [
        "[1] American Concrete Institute, ACI 224R-01, Control of Cracking in Concrete Structures, Farmington Hills, Michigan.",
        "[2] Bureau of Indian Standards, IS 456:2000, Plain and Reinforced Concrete - Code of Practice, New Delhi.",
        "[3] American Concrete Institute, ACI 309R, Guide for Consolidation of Concrete, Farmington Hills, Michigan.",
        "[4] American Concrete Institute, ACI 562, Code Requirements for Assessment, Repair, and Rehabilitation of Existing Concrete Structures, Farmington Hills, Michigan.",
        "[5] International Concrete Repair Institute, Guideline No. 310.1R, Guide for Surface Preparation for the Repair of Deteriorated Concrete Resulting from Reinforcing Steel Corrosion.",
        "[6] The Concrete Society, Technical Report 54, Diagnosis of Deterioration in Concrete Structures - Identification of Defects, Evaluation and Condition Assessment.",
        "[7] Central Public Works Department, Analysis of Rates / Works Manual, Volume 4, Government of India; used as a style reference for repair norms and rate analysis.",
        "[8] Sika and equivalent manufacturer technical datasheets for crack sealing, injection systems, repair mortar, micro-concrete, grouts, bonding agents, and reinforcement passivation; exact products and current data must be confirmed before field use.",
        "[9] Ultralytics, YOLO documentation and software framework for object detection and instance segmentation.",
        "[10] Roboflow, hosted computer-vision dataset, training, evaluation, and inference platform documentation.",
        "[11] Streamlit, open-source Python framework for interactive data applications.",
        "[12] Python Software Foundation, Python language and standard library documentation.",
    ]
    for reference in references:
        story.append(_p(reference, styles["reference"]))

    story.append(_section("Appendix A", "Implementation Traceability Matrix", styles))
    traceability = [
        [_p("Paper claim", styles["head"]), _p("Implementation evidence", styles["head"]), _p("Validation evidence", styles["head"])],
        [_p("Hosted primary inference", styles["cell"]), _p("src/roboflow_model.py: run_model", styles["cell"]), _p("Representative crack/spall/mold audit", styles["cell"])],
        [_p("Secondary six-class vision", styles["cell"]), _p("src/vision_fallback.py: ALLOWED_CLASSES, detect_with_gpt", styles["cell"]), _p("Honeycomb, exposed steel, efflorescence audit", styles["cell"])],
        [_p("Alias normalization and fusion", styles["cell"]), _p("src/pipeline.py: normalise_detection_class, merge_predictions", styles["cell"]), _p("tests/test_detection_merge.py", styles["cell"])],
        [_p("Severity rules", styles["cell"]), _p("src/severity.py: estimate_severity and grading helpers", styles["cell"]), _p("tests/test_severity_remediation.py", styles["cell"])],
        [_p("Quantity and preliminary rates", styles["cell"]), _p("src/cost_estimation.py", styles["cell"]), _p("Cost consistency assertions", styles["cell"])],
        [_p("Norms-based BOQ", styles["cell"]), _p("src/boq.py; data/repair_norms.json", styles["cell"]), _p("Line quantity/rate and total assertions", styles["cell"])],
        [_p("RAG remedy", styles["cell"]), _p("src/remedy_rag.py; data/remedy_knowledge.json", styles["cell"]), _p("tests/test_remedy_rag.py", styles["cell"])],
        [_p("Interactive evidence", styles["cell"]), _p("app.py", styles["cell"]), _p("Browser-verified inspection cards and charts", styles["cell"])],
    ]
    story.append(_table(traceability, [55 * mm, 67 * mm, 54 * mm]))
    story.append(_p("Table A1. Traceability from manuscript claims to implementation and validation evidence.", styles["caption"]))

    story.append(_section("Appendix B", "Assumptions Requiring Field Confirmation", styles))
    assumptions = [
        "Reference scale is accurate, coplanar with the defect, and free from perspective distortion.",
        "Detection boxes tightly represent the damaged region; segmentation would be preferable for irregular defects.",
        "Unscaled photographs cover approximately 3 m2, and unscaled crack images span approximately 2 m.",
        "Default reinforcement cover of 40 mm is appropriate only as a placeholder, not a universal code value.",
        "Repair depth, substrate condition, reinforcement section loss, access, and temporary works are unknown unless explicitly measured.",
        "Consumption norms and unit rates are indicative and require product, project, location, date, and specification confirmation.",
        "Overheads of 15% and GST of 18% are configurable planning assumptions, not universally applicable commercial terms.",
        "Repair tasks are represented as sequential for total-duration summaries unless parallel execution is separately planned.",
    ]
    for item in assumptions:
        story.append(_bullet(item, styles))
    story.append(Spacer(1, 8))
    story.append(_p(f"Manuscript generated from repository evidence on {date.today().isoformat()}. No credentials are included.", styles["small"]))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an in-depth journal-style PDF paper")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PDF path")
    args = parser.parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    build_paper(output_path)
    print(f"JOURNAL_PAPER_CREATED: {output_path}")


if __name__ == "__main__":
    main()