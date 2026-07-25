"""Generate the complete end-to-end project report as a multi-page PDF.

Covers the entire system: detection, severity grading, structural-element
classification, RAG remedy generation, BOQ/cost estimation, implementation,
web application, deployment, results, validation methodology and limitations.

Run:
    python scripts/generate_project_report.py --out outputs/Project_Report.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
INK = colors.HexColor("#1b2433")
MUTED = colors.HexColor("#4b5563")
ACCENT = colors.HexColor("#3b74d4")
LIGHT = colors.HexColor("#eef4ff")
BORDER = colors.HexColor("#c9d3e3")
GREY_BG = colors.HexColor("#f4f6fa")

SEV_BG = {
    "Critical": colors.HexColor("#ffe0e6"),
    "Severe": colors.HexColor("#ffe7d6"),
    "Moderate": colors.HexColor("#fff3e0"),
    "Minor": colors.HexColor("#e7f7ee"),
}


def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=26, leading=31, textColor=INK, spaceAfter=6)
    s["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=13,
                                   leading=18, textColor=MUTED, alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                             fontSize=17, leading=21, textColor=ACCENT,
                             spaceBefore=2, spaceAfter=10)
    s["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=INK,
                             spaceBefore=12, spaceAfter=5)
    s["h3"] = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                             fontSize=10.8, leading=14, textColor=MUTED,
                             spaceBefore=9, spaceAfter=3)
    s["body"] = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.6, leading=14.2,
                               textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=13,
                                 bulletIndent=4, spaceAfter=3)
    s["code"] = ParagraphStyle("code", parent=ss["BodyText"], fontName="Courier",
                               fontSize=8.3, leading=11.4, textColor=INK,
                               backColor=GREY_BG, borderPadding=6, spaceAfter=7)
    s["caption"] = ParagraphStyle("caption", parent=ss["Normal"], fontSize=8.4,
                                  leading=11, textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=3, spaceAfter=10)
    s["cell"] = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.1, leading=10.6,
                               textColor=INK)
    s["cellb"] = ParagraphStyle("cellb", parent=s["cell"], fontName="Helvetica-Bold",
                                textColor=colors.white)
    s["toc1"] = ParagraphStyle("toc1", parent=ss["Normal"], fontSize=10.2, leading=17,
                               textColor=INK)
    s["toc2"] = ParagraphStyle("toc2", parent=ss["Normal"], fontSize=9.3, leading=14,
                               leftIndent=16, textColor=MUTED)
    return s


# ---------------------------------------------------------------------------
def make_doc(path, styles):
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    class Doc(BaseDocTemplate):
        def afterFlowable(self, flowable):
            from reportlab.platypus import Paragraph
            if isinstance(flowable, Paragraph):
                st = flowable.style.name
                if st == "h1":
                    self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
                elif st == "h2":
                    self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.15 * cm,
                          "Construction Defect Detection, Severity Assessment & "
                          "Retrieval-Augmented Remedy Generation")
        canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc = Doc(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
              topMargin=1.9 * cm, bottomMargin=2.1 * cm,
              title="Construction Defect Detection - Project Report",
              author="Project Report")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])
    return doc


# ---------------------------------------------------------------------------
# Flowable helpers
# ---------------------------------------------------------------------------
def T(styles):
    """Return builder helpers bound to the stylesheet."""
    from reportlab.platypus import Image, PageBreak, Paragraph, Spacer, Table, TableStyle

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def h1(text):
        return Paragraph(text, styles["h1"])

    def h2(text):
        return Paragraph(text, styles["h2"])

    def h3(text):
        return Paragraph(text, styles["h3"])

    def bullets(items):
        return [Paragraph(f"• &nbsp;{i}", styles["bullet"]) for i in items]

    def code(text):
        safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br/>").replace(" ", "&nbsp;"))
        return Paragraph(safe, styles["code"])

    def table(data, widths, header=True, align=None, font_size=8.1, zebra=True):
        rows = []
        for r_i, row in enumerate(data):
            cells = []
            for c in row:
                if hasattr(c, "wrapOn"):
                    cells.append(c)
                else:
                    st = "cellb" if (header and r_i == 0) else "cell"
                    cells.append(Paragraph(str(c), styles[st]))
            rows.append(cells)
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]
        if zebra:
            for i in range(1 if header else 0, len(rows)):
                if i % 2 == (0 if header else 1):
                    cmds.append(("BACKGROUND", (0, i), (-1, i), GREY_BG))
        if align:
            for col, a in align.items():
                cmds.append(("ALIGN", (col, 0), (col, -1), a))
        t.setStyle(TableStyle(cmds))
        return t

    def image(path, width_cm, caption=None):
        from PIL import Image as PILImage
        path = Path(path)
        if not path.exists():
            return []
        with PILImage.open(path) as im:
            w, h = im.size
        width = width_cm * cm
        height = width * h / w
        max_h = 15.5 * cm
        if height > max_h:
            height = max_h
            width = height * w / h
        out = [Image(str(path), width=width, height=height, hAlign="CENTER")]
        if caption:
            out.append(Paragraph(caption, styles["caption"]))
        return out

    def note(text, bg=LIGHT, border=ACCENT):
        pr = Paragraph(text, styles["body"])
        t = Table([[pr]], colWidths=[16.4 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 0.9, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    return dict(p=p, h1=h1, h2=h2, h3=h3, bullets=bullets, code=code, table=table,
                image=image, note=note, PageBreak=PageBreak, Spacer=Spacer)


# ---------------------------------------------------------------------------
def render_architecture_pngs(out_dir: Path) -> list[Path]:
    """Render the architecture diagram pages to PNG so they can be embedded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        sys.argv = ["x"]
        import scripts.generate_rag_architecture_pdf as arch

        class Shim:
            def __init__(self):
                self.n = 0
                self.files: list[Path] = []

            def savefig(self, fig, **kw):
                self.n += 1
                f = out_dir / f"arch_{self.n}.png"
                fig.savefig(f, dpi=150, facecolor="white", bbox_inches="tight")
                self.files.append(f)

        shim = Shim()
        arch.page_system(shim)
        arch.page_rag_core(shim)
        arch.page_mechanics(shim)
        paths = shim.files
    except Exception as err:  # diagrams are optional
        print(f"  ! architecture diagrams skipped: {err}")
    return paths


def render_result_charts(out_dir: Path, results: list[dict]) -> dict:
    """Render cost/time/severity charts to PNG for embedding in the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as err:
        print(f"  ! result charts skipped: {err}")
        return {}

    sev_hex = {"Critical": "#d83a52", "Severe": "#e2691a",
               "Moderate": "#e08a1e", "Minor": "#2f9e63"}
    idx = list(range(1, len(results) + 1))
    costs = [float(r.get("est_cost_inr") or 0) for r in results]
    days = [float(r.get("repair_time_days") or 0) for r in results]
    bar_colors = [sev_hex.get(r["severity"], "#7a8698") for r in results]
    paths: dict[str, Path] = {}

    def _finish(fig, name):
        fig.tight_layout()
        f = out_dir / name
        fig.savefig(f, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        paths[name] = f

    # 1. Estimated cost per detection
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.bar(idx, costs, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Detection #"); ax.set_ylabel("Estimated cost (INR)")
    ax.set_title("Estimated repair cost per detection", fontsize=11, fontweight="bold")
    ax.set_xticks(idx); ax.grid(axis="y", alpha=0.3)
    _finish(fig, "chart_cost.png")

    # 2. Repair time per detection
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.bar(idx, days, color="#e08a1e", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Detection #"); ax.set_ylabel("Repair time (working days)")
    ax.set_title("Estimated repair time per detection", fontsize=11, fontweight="bold")
    ax.set_xticks(idx); ax.grid(axis="y", alpha=0.3)
    _finish(fig, "chart_time.png")

    # 3. Severity distribution
    order = ["Minor", "Moderate", "Severe", "Critical"]
    counts = [sum(1 for r in results if r["severity"] == s) for s in order]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(order, counts, color=[sev_hex[s] for s in order], edgecolor="white")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of detections")
    ax.set_title("Severity distribution", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, "chart_severity.png")

    # 4. Cumulative cost (Pareto) curve
    ranked = sorted(results, key=lambda r: float(r.get("est_cost_inr") or 0), reverse=True)
    cum, run = [], 0.0
    for r in ranked:
        run += float(r.get("est_cost_inr") or 0)
        cum.append(run)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    xs = list(range(1, len(cum) + 1))
    ax.fill_between(xs, cum, color="#3b74d4", alpha=0.18)
    ax.plot(xs, cum, color="#3b74d4", marker="o", markersize=3, linewidth=1.6)
    ax.set_xlabel("Detections (ranked by cost)"); ax.set_ylabel("Cumulative cost (INR)")
    ax.set_title("Cumulative cost curve (Pareto)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    _finish(fig, "chart_pareto.png")

    return paths


def load_results() -> list[dict]:
    csv_path = ROOT / "outputs" / "detection_severity_report.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ===========================================================================
# Report content
# ===========================================================================
def build_story(styles, arch_pngs, chart_pngs, results, norms, knowledge):
    from reportlab.platypus import PageBreak, Spacer
    from reportlab.platypus.tableofcontents import TableOfContents

    f = T(styles)
    p, h1, h2, h3 = f["p"], f["h1"], f["h2"], f["h3"]
    bullets, code, table, image, note = (f["bullets"], f["code"], f["table"],
                                         f["image"], f["note"])
    S = story = []
    W = 16.4 * cm

    # ---------------- Cover ----------------
    S += [Spacer(1, 3.2 * cm)]
    S += [p("Construction Defect Detection,<br/>Severity Assessment &amp;<br/>"
            "Retrieval-Augmented Remedy Generation", "title")]
    S += [Spacer(1, 0.5 * cm)]
    S += [p("An end-to-end computer-vision and knowledge-grounded system for "
            "inspecting reinforced-concrete structures", "subtitle")]
    S += [Spacer(1, 2.2 * cm)]
    S += [table([
        ["Project", "Construction Defect Detection using AI"],
        ["Modules", "Detection · Severity Grading · Structural Element · RAG Remedy · BOQ"],
        ["Detection engine", "Roboflow hosted model + AI Engine vision fallback"],
        ["Knowledge base", "11 engineering guidance chunks + 9 norms records + 19 unit rates"],
        ["Standards basis", "ACI 224R-01, ACI 562, ACI 309, IS 456:2000, ICRI 310.1, TR54, CPWD"],
        ["Web application", "Streamlit (cloud-deployed, public link)"],
        ["Document type", "Complete technical project report"],
    ], [4.2 * cm, 12.2 * cm], header=False)]
    S += [PageBreak()]

    # ---------------- Abstract ----------------
    S += [h1("Abstract")]
    S += [p(
        "This project presents a complete, deployable system that inspects photographs of "
        "reinforced-concrete surfaces and produces an engineering-grade condition assessment. "
        "From a single image the system (i) detects visible defects such as cracks, spalling, "
        "honeycombing, exposed reinforcement, mould/dampness and efflorescence; (ii) grades the "
        "severity of every detected defect against recognised civil-engineering standards rather "
        "than an arbitrary threshold; (iii) identifies the structural element on which the defect "
        "occurs (slab, wall, beam, column, staircase or footing); (iv) retrieves relevant "
        "engineering guidance and repair norms from a curated knowledge base; and (v) generates a "
        "grounded repair remedy together with a Bill of Quantities in which every cost line is "
        "computed as quantity multiplied by rate.")]
    S += [p(
        "The detection layer combines a trained Roboflow object-detection model with a secondary "
        "AI Engine vision detector that covers defect classes outside the trained model's "
        "vocabulary, plus an expert-review override for cases both detectors miss. Every finding "
        "records its provenance, so an assessor can always see whether a detection came from the "
        "trained model, the AI Engine or a human reviewer. The remedy layer is a "
        "Retrieval-Augmented Generation (RAG) pipeline: guidance text and numeric norms are "
        "retrieved from two separate knowledge bases, and the cost is always arithmetically "
        "derived, never retrieved as free text. The system is delivered as a Streamlit web "
        "application deployed to the cloud with a public link, and as a reproducible batch "
        "reporting pipeline that produces annotated evidence PDFs.")]
    S += [p(
        "On a validation set of 12 site photographs the system produced 17 graded detections "
        "across four structural element types, with a defect-class distribution spanning cracks, "
        "spalls, honeycombing, exposed reinforcement, mould and efflorescence. The report "
        "documents the architecture, algorithms, engineering basis, implementation, deployment, "
        "results, validation methodology and limitations in full.")]

    S += [h2("Document structure")]
    S += bullets([
        "<b>Chapters 1–2</b> establish the problem and the engineering standards the system is built on.",
        "<b>Chapters 3–8</b> describe the architecture and each functional module in depth.",
        "<b>Chapters 9–11</b> cover implementation, the web application and cloud deployment.",
        "<b>Chapters 12–13</b> present results and the field-validation methodology.",
        "<b>Chapters 14–15</b> discuss limitations, future work and conclusions.",
        "<b>Appendices</b> list the knowledge base, norms, data schema and file inventory.",
    ])
    S += [PageBreak()]

    # ---------------- TOC ----------------
    S += [h1("Table of Contents")]
    toc = TableOfContents()
    toc.levelStyles = [styles["toc1"], styles["toc2"]]
    S += [toc, PageBreak()]

    # ================= 1. Introduction =================
    S += [h1("1. Introduction")]
    S += [h2("1.1 Background")]
    S += [p(
        "Reinforced concrete is the dominant structural material in modern construction, and its "
        "service life is governed largely by the condition of the cover concrete and the "
        "reinforcement it protects. Visual inspection remains the first and most widely used "
        "method of condition assessment: an engineer walks the structure, records defects, judges "
        "their seriousness and recommends remedial action. This process is reliable when performed "
        "by an experienced engineer, but it is slow, subjective, difficult to standardise across "
        "inspectors, and hard to scale across large asset portfolios.")]
    S += [p(
        "Two inspectors looking at the same crack frequently disagree on whether it is 'minor' or "
        "'moderate', because the judgement blends crack width, exposure condition, structural "
        "role and experience. Furthermore, translating an observation into a costed repair "
        "requires quantity surveying knowledge — measuring the defect, selecting a repair method, "
        "applying consumption norms and current rates — which is typically a separate exercise "
        "performed days later by a different person.")]
    S += [h2("1.2 Problem statement")]
    S += [p(
        "There is no single, reproducible workflow that takes a site photograph and returns a "
        "standards-referenced severity grade, an identification of the affected structural "
        "element, a defensible repair method and a transparent cost estimate. Existing computer- "
        "vision work largely stops at detection: it draws a bounding box and names the defect, but "
        "does not answer the questions an engineer actually needs answered — <i>how bad is it, "
        "what is it on, what do I do about it, and what will it cost?</i>")]
    S += [h2("1.3 Objectives")]
    S += bullets([
        "Detect common concrete surface defects from a single photograph, with a fallback path so "
        "defect classes outside the trained model are still reported rather than silently missed.",
        "Grade the severity of each defect using recognised standards (ACI, IS, ICRI) instead of an "
        "arbitrary area cut-off, and express the grade both as a label and as a numeric score.",
        "Quantify <i>how much</i> severity is present, not merely its label, via an affected-extent "
        "measure and an optional real-world millimetre scale.",
        "Identify the structural element (slab, wall, beam, column, staircase, footing) the defect "
        "belongs to, because the same defect has different consequences on different members.",
        "Generate a repair remedy that is grounded in retrieved engineering guidance rather than "
        "produced from unconstrained model memory.",
        "Produce a Bill of Quantities in which every cost is computed as quantity × rate from "
        "retrieved consumption norms and unit rates.",
        "Deliver the system as a usable web application with a shareable link, plus a reproducible "
        "batch report for documentary evidence.",
    ])
    S += [h2("1.4 Scope and limitations of scope")]
    S += [p(
        "The system performs <b>visual, image-based preliminary assessment</b>. It does not "
        "perform structural analysis, load rating, or non-destructive testing. It does not replace "
        "a structural engineer: every output explicitly states that quantities and severities are "
        "preliminary and must be confirmed on site. Defects that are not visible in the photograph "
        "(internal delamination, chloride ingress, carbonation depth, corrosion of embedded steel "
        "beneath sound cover) are outside its reach by construction.")]
    S += [h2("1.5 Contributions")]
    S += bullets([
        "A <b>multi-source detection layer</b> with explicit provenance tagging (trained model, AI "
        "Engine, expert review) so no finding is presented with a false origin.",
        "A <b>standards-based severity engine</b> that grades by crack width when a scale reference "
        "is supplied and by calibrated per-defect area bands otherwise.",
        "A <b>dual-knowledge-base RAG design</b> that separates narrative engineering guidance from "
        "numeric norms, guaranteeing that cost is computed and never hallucinated.",
        "An <b>automatic structural-element classifier</b> integrated into the assessment record.",
        "A <b>deployed application</b> and a reproducible evidence-report pipeline.",
    ])
    S += [PageBreak()]

    # ================= 2. Standards =================
    S += [h1("2. Engineering Standards and Basis")]
    S += [p(
        "A central design decision in this project is that severity is graded against published "
        "engineering criteria rather than an invented scale. This section records the standards "
        "used and how each maps into the software.")]
    S += [h2("2.1 Standards referenced")]
    S += [table([
        ["Standard", "Scope", "Use in this system"],
        ["ACI 224R-01", "Control of cracking in concrete structures; Table 4.1 tolerable crack widths by exposure",
         "Primary basis for crack severity when a millimetre scale is available"],
        ["IS 456:2000, Cl. 35.3.2", "Serviceability: 0.3 mm general surface crack-width limit",
         "Defines the Moderate/Severe crack boundary"],
        ["IS 456:2000, Cl. 12", "Concreting, compaction and workmanship; honeycombing",
         "Basis for honeycombing grading and remedy"],
        ["ACI 309", "Consolidation of concrete", "Cause analysis and remedy for honeycombing"],
        ["ICRI 310.1", "Surface preparation for repair of deteriorated concrete",
         "Basis for spalling and exposed-reinforcement grading and repair method"],
        ["ACI 562", "Code requirements for evaluation, repair and rehabilitation",
         "Escalation rules for severe/critical conditions requiring engineer review"],
        ["Concrete Society TR54", "Diagnosis of deterioration in concrete structures",
         "Supporting basis for spalling/delamination assessment"],
        ["CPWD Manual Vol. 4 (style)", "Consumption norms and rate analysis practice",
         "Basis for the BOQ norms database and rate structure"],
    ], [3.4 * cm, 6.0 * cm, 7.0 * cm])]

    S += [h2("2.2 Why width governs crack severity")]
    S += [p(
        "In practice a crack is judged primarily by its <b>width</b>, because width controls the "
        "ingress of moisture, chlorides and carbon dioxide to the reinforcement. ACI 224R-01 "
        "tabulates tolerable widths by exposure, and IS 456:2000 sets a general 0.3 mm surface "
        "limit. The system therefore uses width as the governing parameter whenever a scale "
        "reference makes a millimetre measurement possible, and falls back to surface extent only "
        "when it cannot measure width — clearly flagging that fallback as approximate.")]
    S += [h2("2.3 Why exposed reinforcement is never 'minor'")]
    S += [p(
        "Exposed reinforcement means the cover concrete has been lost. Per the intent of ICRI "
        "310.1 and ACI 562, the protective alkaline environment is gone and corrosion is active or "
        "imminent regardless of how small the exposed patch appears. The system therefore grades "
        "exposed reinforcement as <b>Severe as a minimum</b>, escalating to Critical when the "
        "exposure is extensive. This is an engineering rule encoded in software, not a statistical "
        "outcome, and it is one of the clearest examples of domain knowledge overriding raw "
        "image geometry.")]
    S += [note(
        "<b>Design principle.</b> Wherever image evidence and engineering doctrine disagree, "
        "doctrine wins. A 0.25% exposed-rebar patch is still graded Severe, because the standard "
        "says loss of cover is a serious durability defect irrespective of extent.")]
    S += [PageBreak()]

    # ================= 3. Architecture =================
    S += [h1("3. System Architecture")]
    S += [h2("3.1 Pipeline overview")]
    S += [p(
        "The system is a five-stage pipeline. Stages 1–3 convert a photograph into a structured, "
        "quantified query; stage 4 grounds that query in retrieved knowledge; stage 5 renders the "
        "result to the user or to a report.")]
    S += [table([
        ["Stage", "Module", "Input", "Output"],
        ["1. Image input", "Streamlit UI / batch script", "Photograph (+ optional scale reference)",
         "Image bytes, mm-per-pixel scale"],
        ["2. Detection", "src/roboflow_model.py, src/vision_fallback.py",
         "Image", "Defect class, bounding box, confidence, source"],
        ["3. Severity &amp; measurement", "src/severity.py, src/cost_estimation.py",
         "Box + image size + scale", "Severity level &amp; score, extent, quantity, cost breakup"],
        ["4. Knowledge grounding", "src/remedy_rag.py, src/boq.py",
         "Structured defect record", "Retrieved guidance, BOQ, grounded remedy"],
        ["5. Presentation", "app.py, scripts/generate_*_report.py",
         "All of the above", "Web UI, annotated image, CSV, PDF"],
    ], [2.6 * cm, 4.3 * cm, 4.6 * cm, 4.9 * cm])]

    S += [h2("3.2 Architecture diagrams")]
    if arch_pngs:
        S += image(arch_pngs[0], 16.2,
                   "Figure 3.1 — End-to-end system pipeline, with the RAG remedy layer expanded.")
        S += [PageBreak()]
        S += image(arch_pngs[1], 16.2,
                   "Figure 3.2 — Internal architecture of the remedy layer: query, dual knowledge "
                   "base, retrieval, augmentation and generation.")
        S += [PageBreak()]
        S += image(arch_pngs[2], 16.2,
                   "Figure 3.3 — Retrieval scoring mechanics and the guarantees that keep every "
                   "reported cost traceable.")
    else:
        S += [p("<i>Architecture diagrams unavailable in this build.</i>")]
    S += [PageBreak()]

    S += [h2("3.3 Design principles")]
    S += bullets([
        "<b>Provenance over convenience.</b> Every detection carries the source that produced it; "
        "a finding is never attributed to a component that did not generate it.",
        "<b>Computed, not recalled.</b> Numeric outputs (quantities, costs) are always arithmetic "
        "results of retrieved norms and rates.",
        "<b>Graceful degradation.</b> Missing an AI Engine key disables the fallback detector, the "
        "element classifier and enhanced remedy text, but detection and severity still function.",
        "<b>Single source of truth.</b> The shared pipeline module is used by both the web "
        "application and the batch report, so the live demo and the PDF cannot diverge.",
        "<b>Explicit uncertainty.</b> Every image-derived measurement is labelled preliminary and "
        "carries its measurement basis.",
    ])
    S += [PageBreak()]

    # ================= 4. Detection =================
    S += [h1("4. Defect Detection Module")]
    S += [h2("4.1 Primary detector — Roboflow hosted model")]
    S += [p(
        "The primary detector is a trained object-detection model hosted on Roboflow's serverless "
        "inference endpoint. The application posts a base64-encoded image over plain HTTPS and "
        "receives JSON containing the image dimensions and a list of predictions, each with a "
        "centre point, width, height, confidence and class label.")]
    S += [code(
        "POST {api_url}/{model_id}?api_key=...\n"
        "Content-Type: application/x-www-form-urlencoded\n"
        "body: base64(image bytes)\n\n"
        "-> { \"image\": {\"width\": W, \"height\": H},\n"
        "     \"predictions\": [ {x, y, width, height, confidence, class}, ... ] }")]
    S += [p(
        "Predictions below the configured confidence threshold (default 0.25) are discarded, and "
        "every surviving prediction is tagged with <font face='Courier'>source = \"Roboflow\"</font>. "
        "The bounding box is converted from Roboflow's centre-based representation to corner "
        "coordinates for drawing and severity computation.")]
    S += [h3("Deliberate removal of the vendor SDK")]
    S += [p(
        "The project originally used the official inference SDK. That package imports OpenCV at "
        "module load, which requires a binary wheel matching the exact Python version and system "
        "graphics libraries. On the hosted platform this produced a hard <font face='Courier'>"
        "ImportError: cv2</font> failure. Because only a single HTTP POST is required, the SDK was "
        "replaced with a direct <font face='Courier'>requests</font> call. This removed two heavy "
        "dependencies, eliminated an entire class of deployment failure and made the runtime "
        "portable across Python versions.")]

    S += [h2("4.2 Secondary detector — AI Engine vision fallback")]
    S += [p(
        "The trained model's vocabulary is limited to the classes present in its training set. "
        "During evaluation it returned <b>zero predictions</b> on several photographs that "
        "unmistakably contained defects — notably exposed reinforcement and efflorescence. Rather "
        "than report those images as defect-free, the system invokes a secondary AI Engine vision "
        "detector whenever the primary detector returns nothing.")]
    S += [p(
        "The AI Engine is prompted as a structural inspection specialist and constrained to a "
        "fixed vocabulary — crack, spalling, honeycombing, exposed_reinforcement, mold, "
        "efflorescence — returning strict JSON with a normalised bounding box and confidence for "
        "each finding. Normalised coordinates are converted to pixels using the true image size, "
        "producing predictions in exactly the same shape as the primary detector so the rest of "
        "the pipeline is unchanged. Every such finding is tagged <font face='Courier'>source = "
        "\"AI vision detector\"</font>.")]
    S += [note(
        "<b>Honesty guarantee.</b> Findings produced by the AI Engine are never presented as "
        "outputs of the trained model. The distinction is preserved in the user interface, the CSV "
        "export and the PDF report, so the trained model's measured performance is never inflated "
        "by results it did not produce.")]

    S += [h2("4.3 Expert review override")]
    S += [p(
        "Some conditions defeat both detectors. In evaluation, one photograph of honeycombed "
        "concrete around a service penetration was reported as defect-free by both, because the "
        "scene visually resembles loose rubble. Since the domain expert can identify it "
        "confidently, the system supports an expert annotation file "
        "(<font face='Courier'>data/manual_detections.json</font>) keyed by image name, holding a "
        "defect class and a normalised box. These merge into the pipeline tagged <font "
        "face='Courier'>source = \"Manual review\"</font> — the standard practice of recording "
        "expert-confirmed findings in an inspection report.")]
    S += [code(
        '{\n'
        '  "767.jpg": [\n'
        '    { "defect_class": "honeycombing", "confidence": 1.0,\n'
        '      "box": {"x_min":0.30,"y_min":0.0,"x_max":1.0,"y_max":0.80} }\n'
        '  ]\n'
        '}')]

    S += [h2("4.4 Detection source hierarchy")]
    S += [table([
        ["Priority", "Source", "Trigger", "Tag shown to user"],
        ["1", "Roboflow trained model", "Always attempted first", "via Roboflow"],
        ["2", "AI Engine vision detector", "Only when the primary returns zero detections",
         "via AI vision detector"],
        ["3", "Expert review", "Always merged if an entry exists for the image",
         "via Manual review"],
    ], [1.8 * cm, 4.4 * cm, 6.2 * cm, 4.0 * cm])]
    S += [p(
        "The fallback is deliberately <i>conditional</i> rather than always-on: when the trained "
        "model succeeds, its output is authoritative and the AI Engine is not consulted. This "
        "keeps the system anchored to the trained detector and limits cost and latency.")]
    S += [PageBreak()]

    # ================= 5. Severity =================
    S += [h1("5. Severity Assessment Engine")]
    S += [h2("5.1 Severity scale")]
    S += [p(
        "Severity is represented as an ordered enumeration so that competing criteria can be "
        "combined by taking the worst. Each level maps to a numeric score and a recommended "
        "management action.")]
    S += [table([
        ["Level", "Score", "Interpretation", "Recommended action"],
        ["Negligible", "0", "No meaningful defect", "Record only; no remedial action"],
        ["Minor", "1", "Cosmetic / durability watch item", "Routine monitoring; cosmetic repair when convenient"],
        ["Moderate", "2", "Repair should be planned", "Investigate cause; seal against moisture ingress"],
        ["Severe", "3", "Structural durability compromised", "Engage a structural engineer; limit loads; schedule remediation"],
        ["Critical", "4", "Immediate concern", "Immediate structural assessment; consider shoring or closure"],
    ], [2.4 * cm, 1.3 * cm, 5.0 * cm, 7.7 * cm])]
    S += [note(
        "<b>Note on terminology.</b> The implemented scale is Negligible / Minor / Moderate / "
        "Severe / Critical. Where a reviewer prefers the four-term scheme "
        "<i>Minor / Moderate / Major / Severe</i>, the mapping is direct: Minor→Minor, "
        "Moderate→Moderate, Severe→Major, Critical→Severe. Only the labels change; the underlying "
        "thresholds and engineering basis are unaffected.")]

    S += [h2("5.2 Crack grading by width (scale available)")]
    S += [p(
        "When a scale reference is supplied, the bounding box is converted to millimetres. The "
        "longer side is treated as crack length and the shorter side as an upper bound on width. "
        "Width is then graded against the ACI/IS bands:")]
    S += [table([
        ["Crack width", "Severity", "Engineering rationale"],
        ["&lt; 0.10 mm", "Minor", "Hairline; durability not materially affected"],
        ["0.10 – 0.30 mm", "Moderate", "Within the IS 456 general 0.3 mm limit; monitor"],
        ["0.30 – 0.70 mm", "Severe", "Exceeds the code limit; corrosion risk"],
        ["&gt; 0.70 mm", "Critical", "Structural concern; engineer review required"],
    ], [3.6 * cm, 2.6 * cm, 10.2 * cm])]
    S += [p(
        "The system is explicit that a rectangle is not a crack: the box width over-estimates the "
        "true crack width, so the millimetre path is conservative and the output states that the "
        "value must be confirmed with a crack gauge. Exact width would require pixel-level "
        "segmentation rather than a bounding box.")]

    S += [h2("5.3 Grading by surface extent (no scale)")]
    S += [p(
        "Without a scale reference the system computes the <b>area ratio</b> — the fraction of the "
        "image occupied by the defect box — and grades it against per-defect bands. Defects that "
        "imply material or section loss use tighter bands than surface-only defects.")]
    S += [table([
        ["Defect", "Minor below", "Moderate below", "Severe below", "Critical at/above"],
        ["Crack", "2%", "8%", "20%", "20%"],
        ["Spalling", "1.5%", "6%", "15%", "15%"],
        ["Honeycombing", "1.5%", "6%", "15%", "15%"],
        ["Exposed reinforcement", "1%", "5%", "12%", "12%"],
        ["Mould / dampness", "2%", "8%", "20%", "20%"],
        ["Default (unknown class)", "2%", "8%", "20%", "20%"],
    ], [4.6 * cm, 2.7 * cm, 3.0 * cm, 2.9 * cm, 3.2 * cm])]
    S += [p(
        "The affected extent is reported to the user as a percentage alongside the severity label "
        "and score, answering not only <i>how serious</i> the defect is but <i>how much</i> of the "
        "captured surface it covers.")]

    S += [h2("5.4 Defect-specific rules")]
    S += [h3("Exposed reinforcement")]
    S += [p("Graded <b>Severe</b> as a floor; escalated to <b>Critical</b> when the exposed area "
            "reaches 5% of the image. Loss of cover implies active corrosion regardless of extent.")]
    S += [h3("Spalling")]
    S += [p("Graded by surface extent, then escalated by depth when a depth measurement is "
            "supplied: a spall reaching the nominal cover (default 40 mm per IS 456) means the "
            "steel is effectively exposed and the defect becomes Critical; a spall of 25 mm or "
            "more is at least Severe. Depth governs over area when both are available.")]
    S += [h3("Honeycombing")]
    S += [p("Graded by surface extent and escalated to at least Severe when void depth reaches "
            "25 mm, since deep honeycombing reduces cover, strength and durability.")]
    S += [h3("Mould / dampness")]
    S += [p("Graded by affected area; treated as a moisture-ingress indicator whose remedy must "
            "begin by identifying and stopping the water source.")]

    S += [h2("5.5 Scale reference and the mm-per-pixel conversion")]
    S += [p(
        "A known object of known size in the photograph anchors pixel measurements to the real "
        "world. If a 100 mm marker spans 250 pixels, the scale is 0.4 mm per pixel, and every "
        "pixel dimension can then be converted to millimetres. This single input upgrades the "
        "assessment from relative (percentage of frame) to absolute (millimetres and square "
        "metres) and is the mechanism by which camera distance is neutralised — a point developed "
        "further in Chapter 13.")]
    S += [code("mm_per_pixel = reference_size_mm / reference_size_px\n"
               "crack_length_mm = max(box_w, box_h) x mm_per_pixel\n"
               "crack_width_mm  = min(box_w, box_h) x mm_per_pixel   # upper bound\n"
               "area_sq_m       = (box_w x mm_per_pixel) x (box_h x mm_per_pixel) / 1e6")]
    S += [PageBreak()]

    # ================= 6. Structural element =================
    S += [h1("6. Structural Element Classification")]
    S += [h2("6.1 Purpose")]
    S += [p(
        "The same defect carries very different consequences depending on the member it affects. "
        "A 0.4 mm crack in a non-structural partition wall is a maintenance item; the same crack "
        "in a column is a structural concern. Inspection records therefore always name the "
        "element. The system classifies the primary structural element visible in each photograph "
        "using the AI Engine, and records it alongside every detection.")]
    S += [h2("6.2 Element vocabulary")]
    S += [table([
        ["Element", "Definition used by the classifier"],
        ["slab", "Horizontal floor, roof or deck surface (soffit or top)"],
        ["wall", "Large vertical planar surface, retaining or shear wall"],
        ["beam", "Horizontal linear member spanning between supports; deeper than wide"],
        ["column", "Vertical linear supporting member (pier, pillar)"],
        ["staircase", "Steps, flights or stair waist slab"],
        ["footing", "Foundation, pile cap or plinth at or near ground level"],
        ["other", "Pipe, kerb, pavement, rubble or anything not a clear member above"],
    ], [3.0 * cm, 13.4 * cm])]
    S += [h2("6.3 Method")]
    S += [p(
        "The image is submitted to the AI Engine with a constrained instruction to return strict "
        "JSON containing one element label, a confidence and a short justification. The response "
        "is validated against the permitted vocabulary; unrecognised labels collapse to "
        "<i>other</i>, and any failure returns <i>unknown</i> with zero confidence so the pipeline "
        "never breaks. Classification is performed once per image, cached, and displayed as a "
        "prominent banner in the application, a column in the summary table and a field in the CSV "
        "export.")]
    S += [note(
        "The element label is explicitly marked <b>AI-classified</b> everywhere it appears, so a "
        "reviewer knows it is an automated visual judgement from a single photograph rather than a "
        "surveyed fact from a drawing.")]
    S += [PageBreak()]

    # ================= 7. RAG =================
    S += [h1("7. Retrieval-Augmented Remedy Generation (RAG)")]
    S += [h2("7.1 Motivation")]
    S += [p(
        "A language model asked directly for a concrete repair method will produce fluent text, "
        "but it may invent standards, quote norms that do not exist and state costs with "
        "unjustified confidence. For an engineering deliverable this is unacceptable. The remedy "
        "layer is therefore built as a Retrieval-Augmented Generation pipeline: the answer is "
        "constructed only from material retrieved out of curated knowledge bases plus the measured "
        "defect record.")]
    S += [h2("7.2 Dual knowledge base")]
    S += [p("The system deliberately separates <b>narrative guidance</b> from <b>numeric norms</b>, "
            "because they have different trust requirements and different failure modes.")]
    S += [table([
        ["", "KB-1: Engineering guidance", "KB-2: Norms and rates"],
        ["File", "data/remedy_knowledge.json", "data/repair_norms.json"],
        ["Size", f"{len(knowledge or [])} chunks",
         f"{len((norms or {}).get('repair_norms', []))} norm records, "
         f"{sum(len(v) for k, v in (norms or {}).get('unit_rates', {}).items() if isinstance(v, list))} unit rates"],
        ["Content", "How to repair: method narrative, cause, measurement basis, standard citation",
         "Consumption norms per unit of work; unit rates in INR for material, labour, equipment"],
        ["Used for", "The prose sections of the remedy and the cited sources",
         "The Bill of Quantities and every rupee figure"],
        ["Trust model", "Quoted as retrieved context", "Never quoted as cost; only multiplied"],
    ], [2.5 * cm, 7.1 * cm, 6.8 * cm])]

    S += [h2("7.3 Retrieval and relevance scoring")]
    S += [p(
        "Retrieval is a deliberately transparent lexical method rather than an opaque embedding "
        "search, so that any reviewer can reproduce why a particular chunk was selected. The "
        "query is assembled from the defect class, severity, measurement basis, severity reason, "
        "initial remedial measure and quantity notes. Query and chunk are tokenised (lower-cased, "
        "split on alphanumerics, tokens of two characters or fewer discarded) and scored:")]
    S += [code(
        "score = |query_tokens INTERSECT chunk_tokens| / sqrt(|chunk_tokens|)\n"
        "score += 3.0   if chunk.defect == query.defect      (exact defect match)\n"
        "score += 1.0   if chunk.defect == \"general\"         (always-relevant guidance)\n"
        "score += 2.0   if query.severity in chunk.severity  (severity match)\n"
        "keep chunks with score > 0, sort descending, take top_k = 4")]
    S += [p(
        "The square-root normalisation prevents long chunks from dominating merely because they "
        "contain more words. The additive boosts encode domain priority: a chunk written for the "
        "exact defect outranks a lexically similar chunk about a different defect, and "
        "severity-specific guidance outranks generic guidance.")]

    S += [h2("7.4 Augmentation — prompt construction")]
    S += [p("The retrieved material and the measured record are assembled into a single grounded "
            "prompt containing: the detected defect and severity; the measurement basis and "
            "severity reason; the computed quantity and its basis; the full norms-based BOQ "
            "rendered as text; the retrieved guidance chunks with their source citations; and an "
            "explicit set of rules. The answer is required to follow nine fixed headings:")]
    S += bullets([
        "1. Recommended remedy &nbsp;&nbsp; 2. Quantity basis &nbsp;&nbsp; 3. Materials with quantities",
        "4. Labour with man-days and rates &nbsp;&nbsp; 5. Equipment with usage and rates",
        "6. Cost estimate table (quantity × rate = amount) &nbsp;&nbsp; 7. Execution steps / method statement",
        "8. Site verification and limitations &nbsp;&nbsp; 9. Sources used",
    ])
    S += [h3("Encoded rules")]
    S += bullets([
        "Do not invent standards, norms or rates outside the retrieved context and BOQ.",
        "Every cost line must show the formula quantity × rate = amount.",
        "Clearly state when a quantity or cost is preliminary.",
        "State that final quantities and rates must be confirmed on site.",
    ])

    S += [h2("7.5 Generation — two paths, one contract")]
    S += [p(
        "The same grounded prompt drives two interchangeable generators. The <b>deterministic "
        "generator</b> composes the nine-section answer directly from the retrieved chunks and the "
        "BOQ using template logic — it requires no external service, always produces the same "
        "output for the same input, and is therefore testable and reproducible. The <b>AI Engine "
        "generator</b> sends the identical prompt to a language model at low temperature for more "
        "fluent prose. If the AI Engine is unavailable or errors, the system automatically returns "
        "the deterministic answer instead of failing.")]
    S += [note(
        "Because both paths consume the same retrieved context and the same computed BOQ, the "
        "<b>engineering content is identical</b>; only the fluency of the prose differs. The "
        "deterministic path is the default, which means the system is fully functional with no "
        "external language service at all.")]
    S += [h2("7.6 Output record")]
    S += [p("Each remedy returns the answer text, the exact prompt used (for audit), the retrieved "
            "chunks with their relevance scores, the de-duplicated list of cited sources, and "
            "flags indicating whether the AI Engine was used. Returning the prompt makes the whole "
            "generation step auditable after the fact.")]
    S += [PageBreak()]

    # ================= 8. BOQ =================
    S += [h1("8. Bill of Quantities and Cost Estimation")]
    S += [h2("8.1 The central rule: cost is computed, never retrieved")]
    S += [p(
        "The knowledge base stores only <b>norms</b> (how much material, labour or equipment is "
        "consumed per unit of work) and <b>unit rates</b> (INR per unit). No rupee total is ever "
        "stored or retrieved as text. Every figure in the BOQ is derived arithmetically, which "
        "makes it impossible for a fabricated price to enter the estimate.")]
    S += [code(
        "item quantity = work quantity x norm\n"
        "line amount   = item quantity x unit rate\n"
        "subtotal      = SUM(material) + SUM(labour) + SUM(equipment)\n"
        "overheads     = subtotal x 15%          (overheads & contingencies)\n"
        "GST           = (subtotal + overheads) x 18%\n"
        "GRAND TOTAL   = subtotal + overheads + GST")]

    S += [h2("8.2 Quantity estimation")]
    S += [p("The work quantity depends on the defect type and on severity, because severity "
            "changes the repair method:")]
    S += [table([
        ["Defect", "Unit", "Basis", "Severity effect"],
        ["Crack", "running metre", "Crack length",
         "Effective-length factor 1.0 / 1.0 / 1.2 / 1.5 for Minor / Moderate / Severe / Critical, "
         "because stitching and strengthening extend beyond the visible crack"],
        ["Spalling", "sq m, or cum", "Affected surface area; volume = area × depth",
         "Severe/Critical with a known depth are measured as volume"],
        ["Honeycombing", "sq m, or cum", "Affected area; volume when depth known",
         "As above"],
        ["Exposed reinforcement", "sq m", "Cover restoration area",
         "Effective-area factor 1.5 for Critical (full bar length must be exposed)"],
        ["Mould / dampness", "sq m", "Affected surface area", "Rate band changes with severity"],
    ], [3.3 * cm, 2.3 * cm, 4.2 * cm, 6.6 * cm])]

    S += [h2("8.3 Rate analysis")]
    S += [p("Each repair item carries a rate broken into material, labour and equipment "
            "components; the composite rate is their sum. Indicative planning rates (INR per unit) "
            "used by the cost module are:")]
    S += [table([
        ["Repair / severity", "Material", "Labour", "Equipment", "Composite"],
        ["Crack – Minor (sealant, per rmt)", "80", "50", "20", "150"],
        ["Crack – Moderate (epoxy injection)", "250", "150", "100", "500"],
        ["Crack – Severe (injection + stitching)", "500", "300", "200", "1,000"],
        ["Crack – Critical (strengthening)", "800", "500", "400", "1,700"],
        ["Spalling – Minor (per sq m)", "400", "250", "150", "800"],
        ["Spalling – Moderate", "700", "400", "250", "1,350"],
        ["Spalling – Severe (micro-concrete)", "1,200", "600", "500", "2,300"],
        ["Spalling – Critical (jacketing)", "2,000", "800", "700", "3,500"],
        ["Honeycombing – Minor (per sq m)", "350", "200", "100", "650"],
        ["Honeycombing – Moderate (grouting)", "600", "350", "250", "1,200"],
        ["Honeycombing – Severe", "1,000", "500", "450", "1,950"],
        ["Honeycombing – Critical", "1,800", "700", "600", "3,100"],
        ["Exposed rebar – Severe (per sq m)", "900", "500", "300", "1,700"],
        ["Exposed rebar – Critical", "1,500", "700", "500", "2,700"],
    ], [6.4 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm],
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"})]
    S += [p("<i>These are indicative planning rates in the CPWD style and must be replaced with "
            "current local market rates before any billing or contractual use.</i>")]

    S += [h2("8.4 Norms database")]
    S += [p("The norms database holds, for each defect and severity combination, the remedy name, "
            "the work unit, a method statement, the source, and lists of material, labour and "
            "equipment items with their consumption norms. Example — moderate crack, epoxy "
            "injection, per running metre:")]
    S += [code(
        "remedy      : Epoxy injection (low viscosity)\n"
        "work_unit   : running metre\n"
        "materials   : epoxy_injection_low_visc  0.35 litre / rmt\n"
        "              epoxy_sealant_paste       0.15 kg    / rmt\n"
        "              sealant_consumables       0.50 lumpsum / rmt\n"
        "labour      : skilled_labour            0.60 man-day / rmt\n"
        "              helper                    0.60 man-day / rmt\n"
        "equipment   : drilling_machine          0.20 day    / rmt\n"
        "method      : Mark and clean crack -> fix ports -> seal surface ->\n"
        "              inject epoxy under pressure -> cure -> remove ports")]
    S += [p("Multiplying each norm by the measured work quantity gives the item quantity, and "
            "multiplying that by the retrieved unit rate gives the line amount. Because the norm, "
            "the quantity, the rate and the product are all printed, every line of the BOQ can be "
            "checked by hand.")]
    S += [PageBreak()]

    # ================= 9. Implementation =================
    S += [h1("9. Implementation")]
    S += [h2("9.1 Technology stack")]
    S += [table([
        ["Layer", "Technology", "Role"],
        ["Detection (primary)", "Roboflow hosted inference (HTTPS + requests)", "Trained defect detector"],
        ["Detection (fallback), element, remedy prose", "AI Engine vision + language service",
         "Out-of-vocabulary detection, element classification, fluent remedy"],
        ["Severity, cost, BOQ, RAG", "Pure Python (no ML runtime)", "Deterministic engineering logic"],
        ["Web application", "Streamlit", "Interactive UI, caching, downloads"],
        ["Imaging", "Pillow (PIL)", "Decoding, annotation drawing"],
        ["Data handling", "pandas", "Tables and CSV export"],
        ["Reporting", "matplotlib, ReportLab", "Architecture diagrams and PDF reports"],
        ["Config / secrets", "python-dotenv + platform secrets", "Key management"],
    ], [4.6 * cm, 5.6 * cm, 6.2 * cm])]

    S += [h2("9.2 Module inventory")]
    S += [table([
        ["Module", "Lines", "Responsibility"],
        ["src/pipeline.py", "235", "Shared engine: detect → fallback → expert merge → grade → element"],
        ["src/severity.py", "506", "Standards-based severity grading and remediation guidance"],
        ["src/cost_estimation.py", "615", "Quantity estimation and rate analysis (quantity × rate)"],
        ["src/remedy_rag.py", "398", "Knowledge retrieval, prompt building, grounded generation"],
        ["src/boq.py", "259", "Norms retrieval and Bill of Quantities computation"],
        ["src/vision_fallback.py", "235", "AI Engine vision detector and structural-element classifier"],
        ["src/roboflow_model.py", "86", "Direct Roboflow HTTPS inference (no SDK, no OpenCV)"],
        ["src/roboflow_workflow.py", "203", "Legacy workflow runner (retained for reference)"],
        ["src/detect.py / src/train.py", "135 / 37", "Local YOLO batch inference and training utilities"],
        ["app.py", "~270", "Streamlit web application"],
        ["scripts/generate_detection_severity_report.py", "574", "Batch evidence PDF + CSV"],
        ["scripts/generate_rag_architecture_pdf.py", "452", "Architecture diagram generator"],
        ["tests/", "439", "Unit tests for severity, remedy RAG and workflow parsing"],
    ], [6.4 * cm, 1.8 * cm, 8.2 * cm])]

    S += [h2("9.3 Repository layout")]
    S += [code(
        "construction-defect-detection/\n"
        "  app.py                     Streamlit application\n"
        "  src/                       Core engine modules\n"
        "  data/                      Knowledge base + norms + expert annotations\n"
        "    remedy_knowledge.json      11 engineering guidance chunks\n"
        "    repair_norms.json          9 norm records + 19 unit rates\n"
        "    manual_detections.json     expert-confirmed annotations\n"
        "  scripts/                   Report and diagram generators\n"
        "  tests/                     Unit tests\n"
        "  outputs/                   Generated PDFs, CSVs, annotated images\n"
        "  requirements.txt           Slim runtime deps for deployment\n"
        "  requirements-dev.txt       Full local/dev deps\n"
        "  .streamlit/config.toml     Theme and server config\n"
        "  DEPLOY.md                  Deployment guide")]

    S += [h2("9.4 Data contract between stages")]
    S += [p("Every detection, regardless of which detector produced it, is normalised to a single "
            "dictionary shape before grading. This uniform contract is what allows three different "
            "detection sources to feed one severity engine without special cases.")]
    S += [code(
        "{ 'defect': str, 'confidence': float, 'source': str,\n"
        "  'box': (x1, y1, x2, y2),\n"
        "  'severity': str, 'score': int, 'area_pct': float,\n"
        "  'measured': str, 'reason': str, 'action': str, 'standard': str,\n"
        "  'remedial_measure': str, 'repair_time_estimate': str,\n"
        "  'cost_breakup': dict, 'boq_breakup': dict }")]
    S += [PageBreak()]

    # ================= 10. Web app =================
    S += [h1("10. Web Application")]
    S += [h2("10.1 User flow")]
    S += [p("The user uploads a photograph; the application runs the full pipeline and presents "
            "the assessment on a single screen. Results are cached against the image content and "
            "the analysis settings, so adjusting an unrelated control does not trigger new "
            "inference calls — this keeps the interface responsive and avoids redundant cost.")]
    S += [h2("10.2 Interface elements")]
    S += [table([
        ["Element", "Purpose"],
        ["Metric row", "Detection count, worst severity chip, structural element, image size"],
        ["Annotated image", "Severity-coloured boxes with numbered badges keyed to the detail panel"],
        ["Structural element banner", "Prominent element label with AI-classified confidence"],
        ["Per-detection card", "Defect name, severity chip, score /4, affected extent %, confidence, "
                               "source tag, four-segment severity meter, reason, recommended action, "
                               "measurement basis and standard"],
        ["Remedy expander", "Full grounded repair plan with cited sources"],
        ["Summary table", "All detections with element, source, severity, score, extent, action"],
        ["Downloads", "Annotated image (JPEG) and results (CSV)"],
    ], [4.6 * cm, 11.8 * cm])]
    S += [h2("10.3 Controls")]
    S += bullets([
        "<b>Detection confidence</b> — threshold for accepting primary-model predictions.",
        "<b>AI vision fallback</b> — enable the secondary detector for images the primary leaves empty.",
        "<b>Classify structural element</b> — enable element identification.",
        "<b>Scale reference</b> — supply a known real size and its pixel size to switch severity to "
        "true millimetre grading.",
        "<b>Repair remedy</b> — generate the grounded remedy; optionally use the AI Engine for "
        "richer prose instead of the deterministic text.",
    ])
    S += [h2("10.4 Colour language")]
    S += [table([
        ["Severity", "Colour", "Severity", "Colour"],
        ["Minor", "Green", "Severe", "Orange"],
        ["Moderate", "Amber", "Critical", "Red"],
    ], [3.4 * cm, 4.8 * cm, 3.4 * cm, 4.8 * cm])]
    S += [p("The same colours are used for bounding boxes, chips and the severity meter, so the "
            "image and the text panel read as one system.")]
    S += [h2("10.5 Graceful degradation")]
    S += [p("If the AI Engine key is absent, the fallback detector, element classification and "
            "enhanced remedy prose are disabled and clearly marked as such, while detection, "
            "severity grading, quantity estimation, BOQ and the deterministic remedy continue to "
            "operate. The application never presents a blank failure.")]
    S += [PageBreak()]

    # ================= 11. Deployment =================
    S += [h1("11. Deployment")]
    S += [h2("11.1 Hosting decision")]
    S += [p("The application is deployed to Streamlit Community Cloud, which is purpose-built for "
            "Streamlit applications, free, and connected directly to the source repository so "
            "every push redeploys automatically. An alternative container-based host was evaluated "
            "but its free tier no longer offers a suitable CPU instance for this class of "
            "application. Because all heavy inference is delegated to hosted APIs, the application "
            "itself needs no GPU and very little memory.")]
    S += [h2("11.2 Runtime dependencies")]
    S += [p("The deployment dependency set was deliberately minimised. Training and report-"
            "generation dependencies were moved to a separate development requirements file.")]
    S += [table([
        ["Runtime (deployed)", "Development only"],
        ["streamlit, pillow, pandas, numpy,<br/>requests, python-dotenv, AI Engine client",
         "ultralytics (local YOLO), opencv-python,<br/>inference-sdk, matplotlib, pyyaml"],
    ], [8.2 * cm, 8.2 * cm])]
    S += [h2("11.3 Configuration and secrets")]
    S += [p("No credential is stored in the repository. The environment file is git-ignored and "
            "keys are supplied through the hosting platform's secret manager, injected as "
            "environment variables that the code reads at runtime.")]
    S += [code(
        'ROBOFLOW_API_KEY   = "..."\n'
        'ROBOFLOW_MODEL_ID  = "training-dataset-1gvqr/2"\n'
        'ROBOFLOW_API_URL   = "https://serverless.roboflow.com"\n'
        'AI_ENGINE_API_KEY  = "..."   # AI Engine credential')]

    S += [h2("11.4 Deployment issues encountered and resolved")]
    S += [table([
        ["Issue", "Root cause", "Resolution"],
        ["ImportError: cv2 on startup",
         "The vendor inference SDK imports OpenCV at module load; no matching wheel for the "
         "platform's Python version",
         "Replaced the SDK with a direct HTTPS request; removed OpenCV and the SDK from runtime "
         "dependencies entirely"],
        ["Workflow endpoint returned a service misconfiguration",
         "The hosted workflow chained a classification block pointing at an unrelated sample model",
         "Switched the application to the direct detection model endpoint, which is stable"],
        ["Stale application code after deployment",
         "Commits existed locally but had not been pushed to the connected repository",
         "Pushed the branch; platform redeployed automatically"],
        ["Secrets rejected as invalid",
         "Environment-file syntax pasted into a TOML secrets field",
         "Converted every entry to quoted TOML key–value form"],
        ["Heavy build, slow cold start",
         "Deep-learning packages present in runtime requirements though unused by the hosted path",
         "Split requirements into slim runtime and full development sets"],
    ], [3.9 * cm, 5.9 * cm, 6.6 * cm])]
    S += [h2("11.5 Operational note")]
    S += [note(
        "The deployed application calls metered external services using the owner's credentials. "
        "A publicly reachable link therefore consumes the owner's quota. For assessment the link "
        "should be shared narrowly and paused afterwards, or protected by an access control and a "
        "spending limit.")]
    S += [PageBreak()]

    # ================= 12. Results =================
    S += [h1("12. Results and Evaluation")]
    S += [h2("12.1 Evaluation set")]
    S += [p("The system was evaluated on 12 photographs covering cracks, spalling, exposed "
            "reinforcement, mould and efflorescence across different structural elements. Every "
            "image was processed through the complete pipeline.")]
    if results:
        n_img = len({r["image"] for r in results})
        sev_count = {}
        src_count = {}
        el_count = {}
        def_count = {}
        for r in results:
            sev_count[r["severity"]] = sev_count.get(r["severity"], 0) + 1
            src_count[r["detection_source"]] = src_count.get(r["detection_source"], 0) + 1
            el_count[r["structural_element"]] = el_count.get(r["structural_element"], 0) + 1
            def_count[r["defect"]] = def_count.get(r["defect"], 0) + 1
        S += [table([
            ["Images processed", str(n_img), "Total detections", str(len(results))],
            ["Critical", str(sev_count.get("Critical", 0)), "Severe", str(sev_count.get("Severe", 0))],
            ["Moderate", str(sev_count.get("Moderate", 0)), "Minor", str(sev_count.get("Minor", 0))],
        ], [4.4 * cm, 3.6 * cm, 4.4 * cm, 4.0 * cm], header=False)]

        S += [h2("12.2 Distribution by detection source")]
        S += [table([["Source", "Detections", "Share"]] +
                    [[k, str(v), f"{v / len(results) * 100:.0f}%"]
                     for k, v in sorted(src_count.items(), key=lambda x: -x[1])],
                    [7.0 * cm, 4.6 * cm, 4.8 * cm], align={1: "RIGHT", 2: "RIGHT"})]
        S += [p("Roughly half of all findings came from the trained model and half from the AI "
                "Engine fallback. This is the single most important empirical result in the "
                "project: without the fallback, five of twelve photographs — including every "
                "exposed-reinforcement and efflorescence image — would have been reported as "
                "defect-free.")]

        S += [h2("12.3 Distribution by structural element and defect")]
        S += [table([["Structural element", "Detections", "Defect class", "Detections"]] +
                    [[k, str(v),
                      list(sorted(def_count.items(), key=lambda x: -x[1]))[i][0]
                      if i < len(def_count) else "",
                      str(list(sorted(def_count.items(), key=lambda x: -x[1]))[i][1])
                      if i < len(def_count) else ""]
                     for i, (k, v) in enumerate(sorted(el_count.items(), key=lambda x: -x[1]))],
                    [4.4 * cm, 3.4 * cm, 4.6 * cm, 4.0 * cm],
                    align={1: "RIGHT", 3: "RIGHT"})]

        # ---- Cost & time analysis ----
        S += [PageBreak()]
        S += [h2("12.4 Cost and time analysis")]
        total_cost = sum(float(r.get("est_cost_inr") or 0) for r in results)
        total_days = sum(float(r.get("repair_time_days") or 0) for r in results)
        longest = max((float(r.get("repair_time_days") or 0) for r in results), default=0)
        S += [p("For every detection the system estimates a repair cost (quantity × rate) and a "
                "repair duration in working days (labour man-days ÷ crew size, plus a curing and "
                "mobilisation allowance that grows with severity). The portfolio totals are:")]
        S += [table([
            ["Total estimated cost", f"INR {total_cost:,.0f}",
             "Total repair time (sequential)", f"{total_days:.0f} working days"],
            ["Detections costed", str(len(results)),
             "Longest single repair (critical path)", f"{longest:.0f} days"],
        ], [4.6 * cm, 3.6 * cm, 5.4 * cm, 2.8 * cm], header=False)]
        if chart_pngs.get("chart_cost.png"):
            S += image(chart_pngs["chart_cost.png"], 15.5,
                       "Figure 12.1 — Estimated repair cost per detection, coloured by severity.")
        if chart_pngs.get("chart_time.png"):
            S += image(chart_pngs["chart_time.png"], 15.5,
                       "Figure 12.2 — Estimated repair time (working days) per detection.")
        S += [PageBreak()]
        from reportlab.platypus import Table as _Tbl, TableStyle as _TS
        imgs = []
        if chart_pngs.get("chart_severity.png"):
            imgs += image(chart_pngs["chart_severity.png"], 7.6)
        if chart_pngs.get("chart_pareto.png"):
            imgs += image(chart_pngs["chart_pareto.png"], 7.6)
        if imgs:
            row = [i for i in imgs if hasattr(i, "wrapOn")][:2]
            if len(row) == 2:
                t = _Tbl([row], colWidths=[8.2 * cm, 8.2 * cm])
                t.setStyle(_TS([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                S += [t]
            else:
                S += imgs
            S += [p("Figure 12.3 — Severity distribution (left) and the cumulative cost "
                    "(Pareto) curve (right): a small number of high-severity defects account "
                    "for most of the estimated cost.", "caption")]

        S += [PageBreak()]
        S += [h2("12.5 Complete detection register")]
        S += [p("Every detection produced by the system, with its structural element, severity, "
                "affected extent, estimated cost, repair time and provenance.")]
        rows = [["#", "Image", "Element", "Defect", "Sev.", "Extent",
                 "Cost (INR)", "Days", "Source"]]
        for i, r in enumerate(results, 1):
            img_name = r["image"]
            if len(img_name) > 20:
                img_name = img_name[:18] + "…"
            rows.append([
                str(i), img_name, r["structural_element"].title(),
                r["defect"].replace("_", " "), r["severity"],
                f"{float(r['affected_extent_pct']):.1f}%",
                f"{float(r.get('est_cost_inr') or 0):,.0f}",
                str(r.get("repair_time_days", "")),
                r["detection_source"].replace("AI vision detector", "AI Engine"),
            ])
        S += [table(rows, [0.8 * cm, 3.0 * cm, 1.7 * cm, 2.3 * cm, 1.6 * cm,
                           1.4 * cm, 1.9 * cm, 1.0 * cm, 2.0 * cm], font_size=7.3,
                    align={5: "RIGHT", 6: "RIGHT", 7: "CENTER"})]
        S += [p("<i>Costs use indicative planning rates and image-based quantities; both are "
                "preliminary and must be confirmed with site measurement and current local "
                "rates before billing.</i>")]

    S += [PageBreak()]
    S += [h2("12.6 Worked examples")]
    annot = ROOT / "outputs" / "annotated"
    examples = [
        ("00096_annotated.jpg",
         "Figure 12.1 — Crack on a slab. Graded Critical (4/4) with an affected extent of 74.7% "
         "of the frame; detected by the trained model at 55% confidence."),
        ("76_annotated.jpg",
         "Figure 12.2 — Two separate spalls on a wall, both graded Critical, detected by the "
         "trained model. Numbered badges key each box to its detail card."),
        ("exposed_reinforcement_7-25_annotated.jpg",
         "Figure 12.3 — Column with three distinct findings from the AI Engine fallback: exposed "
         "reinforcement (Severe), spalling (Moderate) and a crack (Minor). The trained model "
         "returned nothing for this image."),
        ("white_bleeding_13-98_annotated.jpg",
         "Figure 12.4 — Efflorescence and a crack on a wall, both Moderate. Efflorescence is "
         "outside the trained model's vocabulary and was recovered by the fallback."),
        ("767_annotated.jpg",
         "Figure 12.5 — Honeycombing on a footing, recorded through expert review after both "
         "automated detectors returned no finding; graded Critical at 56% extent."),
    ]
    for fname, cap in examples:
        fp = annot / fname
        if fp.exists():
            S += image(fp, 11.0, cap)

    S += [h2("12.7 Observations")]
    S += bullets([
        "<b>Complementary detectors.</b> The trained model was strong on cracks, spalls and mould; "
        "the AI Engine recovered exposed reinforcement and efflorescence entirely.",
        "<b>Severity skew.</b> Ten of seventeen findings graded Critical, largely because the "
        "evaluation photographs are close-ups in which the defect fills much of the frame. This is "
        "an artefact of framing and is precisely what a scale reference corrects.",
        "<b>Extent versus box quality.</b> AI Engine boxes are looser than trained-model boxes, so "
        "their extent percentages are less reliable — visible in the 0.25% exposed-reinforcement "
        "figure. Severity remained correct because the engineering floor rule governs.",
        "<b>Element classification proved stable</b>, correctly distinguishing a cylindrical pier "
        "as a column and ground-level rubble-faced concrete as a footing.",
    ])
    S += [PageBreak()]

    # ================= 13. Validation =================
    S += [h1("13. Field Validation Methodology")]
    S += [p("Results so far establish that the system produces plausible, standards-referenced "
            "assessments. Establishing that its <i>measurements</i> are accurate requires "
            "comparison against physical reality. This chapter defines that protocol.")]
    S += [h2("13.1 Why camera geometry matters")]
    S += [p("Without a scale reference the system reports extent as a fraction of the image. That "
            "fraction depends on how far the camera was from the surface: photographing the same "
            "300 mm crack from 0.5 m and from 3 m produces very different extents and therefore "
            "potentially different severity grades. Camera distance is thus a confounding variable "
            "that must be either recorded or eliminated.")]
    S += [note("<b>Elimination is preferable to recording.</b> Placing a scale reference of known "
               "size in the frame makes the measurement independent of camera distance, because "
               "the mm-per-pixel factor is derived from the object itself.")]
    S += [h2("13.2 Validation protocol")]
    S += bullets([
        "<b>Step 1 — Capture.</b> Photograph a real defect with a scale reference (steel rule, "
        "scale card or coin of known dimension) lying in the same plane as the defect.",
        "<b>Step 2 — Record geometry.</b> Note the camera-to-surface distance, the approximate "
        "viewing angle (aim for perpendicular), and the lighting condition.",
        "<b>Step 3 — Measure physically.</b> Measure the true defect: crack width with a crack "
        "gauge or feeler gauge, crack length with a tape, spalled or honeycombed area with a tape, "
        "and depth with a depth probe where relevant.",
        "<b>Step 4 — Run the system.</b> Analyse the photograph twice: once without a scale "
        "reference and once with the scale entered, recording both outputs.",
        "<b>Step 5 — Compare.</b> Compute the error between the system's dimension and the "
        "measured dimension, and check whether the severity grade agrees with engineering "
        "judgement.",
    ])
    S += [h2("13.3 Recording template")]
    S += [table([
        ["Field", "Example", "Field", "Example"],
        ["Photo ID", "IMG_014", "Camera distance", "1.2 m"],
        ["Element", "Column", "Viewing angle", "~90° (perpendicular)"],
        ["Defect type", "Crack", "Scale reference used", "150 mm steel rule"],
        ["Actual width (gauge)", "0.35 mm", "System width (scaled)", "0.52 mm"],
        ["Actual length (tape)", "820 mm", "System length", "790 mm"],
        ["Width error", "+48.6%", "Length error", "−3.7%"],
        ["Severity — engineer", "Severe", "Severity — system", "Severe"],
        ["Extent (no scale)", "18.4%", "Agreement", "Grade matches"],
    ], [3.8 * cm, 4.0 * cm, 4.4 * cm, 4.2 * cm])]
    S += [h2("13.4 Expected outcomes and interpretation")]
    S += bullets([
        "<b>Length should validate well.</b> Crack length is the long axis of the bounding box and "
        "should track the taped length closely once scaled.",
        "<b>Width will over-estimate.</b> The box's short side is an upper bound on crack width, "
        "so a positive bias is expected and should be reported honestly rather than tuned away.",
        "<b>Grade agreement is the key metric.</b> For engineering purposes, whether the system "
        "assigns the same severity band as the engineer matters more than millimetre precision.",
        "<b>Without a scale, expect distance sensitivity.</b> Repeating the same defect at two "
        "distances demonstrates the magnitude of the effect and justifies the scale-reference "
        "workflow.",
    ])
    S += [h2("13.5 Recommended sample")]
    S += [p("A minimum of eight to ten new photographs spanning at least three defect types and "
            "three structural elements, each with a physical measurement and a recorded camera "
            "distance, is sufficient to characterise accuracy and to state a defensible confidence "
            "range in the conclusions.")]
    S += [PageBreak()]

    # ================= 14. Limitations =================
    S += [h1("14. Limitations and Future Work")]
    S += [h2("14.1 Known limitations")]
    S += [table([
        ["Limitation", "Consequence", "Mitigation in place"],
        ["Bounding box is not a defect outline",
         "Crack width is over-estimated; extent includes background",
         "Width declared an upper bound; user told to confirm with a gauge"],
        ["Extent depends on camera distance without a scale",
         "Severity can shift with framing",
         "Optional scale reference converts to true millimetres"],
        ["Trained model has a limited class vocabulary",
         "Some defects missed entirely",
         "AI Engine fallback plus expert-review override"],
        ["AI Engine boxes are loosely localised",
         "Extent percentages from fallback findings are less reliable",
         "Engineering floor rules govern severity; provenance is disclosed"],
        ["Element classified from one photograph",
         "A partial view can be ambiguous",
         "Labelled AI-classified with a confidence value"],
        ["Rates are indicative, not current market rates",
         "Costs are planning-level only",
         "Stated on every output; norms and rates are editable data files"],
        ["Depth is not observable from a photograph",
         "Volume-based quantities need manual depth input",
         "Depth accepted as an optional parameter that escalates severity"],
        ["No detection of sub-surface deterioration",
         "Delamination and corrosion under sound cover are invisible",
         "Scope explicitly limited to visual assessment"],
    ], [4.6 * cm, 5.4 * cm, 6.4 * cm])]

    S += [h2("14.2 Future work")]
    S += bullets([
        "<b>Segmentation instead of boxes.</b> Pixel-level masks would give true crack width and "
        "genuine defect area, removing the largest source of measurement error.",
        "<b>Expand the trained model.</b> Retraining with exposed reinforcement, efflorescence and "
        "honeycombing classes would reduce dependence on the fallback detector.",
        "<b>Automatic scale recovery.</b> Detecting a standard reference object, or using device "
        "depth sensors, would remove the manual scale-entry step.",
        "<b>Semantic retrieval.</b> Replacing lexical scoring with embeddings, while retaining the "
        "defect and severity boosts, could improve recall on unusually worded queries.",
        "<b>Multi-image and temporal assessment.</b> Comparing inspections over time would allow "
        "crack growth rates and deterioration trends to be reported.",
        "<b>Live rate integration.</b> Sourcing unit rates from a maintained schedule would make "
        "the BOQ contractually usable.",
        "<b>Formal accuracy study.</b> Executing the Chapter 13 protocol at scale would yield "
        "published precision, recall and grade-agreement statistics.",
    ])
    S += [PageBreak()]

    # ================= 15. Conclusion =================
    S += [h1("15. Conclusion")]
    S += [p("This project delivers a complete, working system that converts a photograph of a "
            "concrete surface into an engineering assessment: what the defect is, how serious it "
            "is and by how much, what structural element it affects, how it should be repaired, "
            "and what that repair is likely to cost.")]
    S += [p("Three design decisions distinguish it from a conventional detection demonstration. "
            "First, <b>severity is graded against published standards</b> — ACI 224R-01, IS "
            "456:2000, ICRI 310.1 and ACI 562 — with domain rules that override raw image geometry "
            "where engineering doctrine demands it. Second, <b>the remedy is retrieval-augmented "
            "and the cost is computed</b>: guidance is quoted from a curated knowledge base with "
            "source citations, and every rupee is the product of a retrieved norm and a retrieved "
            "rate, making fabricated pricing structurally impossible. Third, <b>provenance is "
            "preserved end to end</b>: findings from the trained model, the AI Engine fallback and "
            "expert review are always distinguishable, so the trained detector's performance is "
            "never overstated.")]
    S += [p("Evaluation on twelve photographs produced seventeen graded findings across four "
            "structural element types. The result that most shaped the final architecture was that "
            "the trained model returned nothing on five images that clearly contained defects; the "
            "fallback layer recovered them, and the provenance system reported honestly where each "
            "finding came from. The system is deployed as a public web application and is "
            "reproducible as a batch evidence report.")]
    S += [p("The principal remaining work is quantitative field validation: comparing system "
            "measurements against physical measurements under recorded camera geometry, as set out "
            "in Chapter 13. Completing that study would convert the present qualitative "
            "demonstration into a characterised measurement instrument with a stated accuracy.")]
    S += [PageBreak()]

    # ================= References =================
    S += [h1("References")]
    refs = [
        "ACI 224R-01 — <i>Control of Cracking in Concrete Structures</i>. American Concrete Institute.",
        "ACI 309R — <i>Guide for Consolidation of Concrete</i>. American Concrete Institute.",
        "ACI 562 — <i>Code Requirements for Evaluation, Repair, and Rehabilitation of Concrete "
        "Buildings</i>. American Concrete Institute.",
        "IS 456:2000 — <i>Plain and Reinforced Concrete — Code of Practice</i>, Cl. 12 "
        "(workmanship) and Cl. 35.3.2 (crack width limits). Bureau of Indian Standards.",
        "ICRI Guideline 310.1 — <i>Guide for Surface Preparation for the Repair of Deteriorated "
        "Concrete</i>. International Concrete Repair Institute.",
        "Concrete Society Technical Report TR54 — <i>Diagnosis of Deterioration in Concrete "
        "Structures</i>.",
        "CPWD Analysis of Rates / Works Manual Vol. 4 — consumption norms and rate-analysis "
        "practice. Central Public Works Department, Government of India.",
        "Manufacturer technical datasheets for epoxy injection, crack sealing and polymer-modified "
        "repair mortar systems (consumption figures).",
        "Roboflow — hosted object-detection training and serverless inference platform.",
        "Streamlit — open-source application framework for data and machine-learning interfaces.",
    ]
    for i, r in enumerate(refs, 1):
        S += [p(f"[{i}] &nbsp;{r}")]
    S += [PageBreak()]

    # ================= Appendices =================
    S += [h1("Appendix A — Engineering Knowledge Base")]
    S += [p("The retrieval corpus. Each chunk carries a defect key, the severity levels it applies "
            "to, a standard citation and the guidance text used to ground the remedy.")]
    if knowledge:
        rows = [["ID", "Defect", "Severity levels", "Cited source"]]
        for k in knowledge:
            rows.append([k.get("id", ""), k.get("defect", ""),
                         ", ".join(k.get("severity", [])), k.get("source", "")])
        S += [table(rows, [3.6 * cm, 3.0 * cm, 4.0 * cm, 5.8 * cm], font_size=7.6)]
        S += [h2("Sample chunk text")]
        for k in knowledge[:3]:
            S += [h3(f"{k.get('id','')} — {k.get('source','')}")]
            S += [p(f"<i>{k.get('content','')}</i>")]
    S += [PageBreak()]

    S += [h1("Appendix B — Norms and Unit Rates")]
    if norms:
        S += [h2("B.1 Repair norm records")]
        rows = [["ID", "Defect", "Severity", "Remedy", "Work unit"]]
        for r in norms.get("repair_norms", []):
            rows.append([r.get("id", ""), r.get("defect", ""),
                         ", ".join(r.get("severity", [])), r.get("remedy", ""),
                         r.get("work_unit", "")])
        S += [table(rows, [3.7 * cm, 2.5 * cm, 2.8 * cm, 5.0 * cm, 2.4 * cm], font_size=7.4)]
        S += [h2("B.2 Unit rates")]
        for cat in ("materials", "labour", "equipment"):
            items = norms.get("unit_rates", {}).get(cat, [])
            if not isinstance(items, list) or not items:
                continue
            S += [h3(cat.title())]
            rows = [["Item", "Description", "Unit", "Rate (INR)"]]
            for it in items:
                rows.append([it.get("item", ""), it.get("description", ""),
                             it.get("unit", ""), f"{it.get('rate', 0):,.0f}"])
            S += [table(rows, [4.4 * cm, 6.6 * cm, 2.6 * cm, 2.8 * cm], font_size=7.4,
                        align={3: "RIGHT"})]
    S += [PageBreak()]

    S += [h1("Appendix C — Output Data Schema")]
    S += [p("Schema of the CSV produced for every batch run and downloadable from the web "
            "application.")]
    S += [table([
        ["Column", "Meaning"],
        ["image", "Source photograph file name"],
        ["structural_element", "AI-classified element (slab, wall, beam, column, staircase, footing, other)"],
        ["defect", "Detected defect class"],
        ["detection_source", "Roboflow, AI Engine vision detector, or Manual review"],
        ["confidence", "Detector confidence, 0–1"],
        ["severity", "Minor, Moderate, Severe or Critical"],
        ["severity_score", "Numeric grade out of 4"],
        ["affected_extent_pct", "Defect box area as a percentage of image area"],
        ["measurement_basis", "How the grade was measured (area ratio, or scaled millimetres)"],
        ["standard", "Governing standard applied to this grade"],
        ["reason", "Engineering justification for the grade"],
        ["recommended_action", "Management action for the severity level"],
    ], [4.6 * cm, 11.8 * cm])]

    S += [h2("Appendix D — Deliverables")]
    S += [table([
        ["Deliverable", "Description"],
        ["Web application", "Deployed Streamlit app with a shareable link"],
        ["Detection &amp; severity report", "Per-image evidence PDF with annotated photographs, "
                                            "severity cards and a summary register"],
        ["Architecture document", "Diagrammatic PDF of the pipeline and RAG internals"],
        ["Results CSV", "One row per detection with the full schema above"],
        ["Annotated images", "Severity-coloured boxes with numbered badges"],
        ["This report", "Complete technical documentation of the project"],
        ["Source repository", "Application, engine modules, knowledge base, tests and generators"],
    ], [4.6 * cm, 11.8 * cm])]

    return story


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/Project_Report.pdf")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Rendering architecture diagrams ...")
    arch = render_architecture_pngs(ROOT / "outputs" / "_report_assets")
    print(f"  {len(arch)} diagram(s)")

    results = load_results()
    norms = load_json("data/repair_norms.json")
    knowledge = load_json("data/remedy_knowledge.json")
    print(f"Loaded {len(results)} detections, "
          f"{len((norms or {}).get('repair_norms', []))} norms, "
          f"{len(knowledge or [])} knowledge chunks")

    print("Rendering result charts ...")
    charts = render_result_charts(ROOT / "outputs" / "_report_assets", results)
    print(f"  {len(charts)} chart(s)")

    styles = build_styles()
    story = build_story(styles, arch, charts, results, norms, knowledge)

    doc = make_doc(out, styles)
    print("Building PDF (two passes for the table of contents) ...")
    doc.multiBuild(story)
    print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
