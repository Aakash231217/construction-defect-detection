"""Generate a professional, diagrammatic architecture PDF for the RAG remedy module.

This produces a multi-page vector PDF with boxes, layers and directional flow
arrows describing the Retrieval-Augmented Generation pipeline used to turn a
detected construction defect into a grounded repair remedy + Bill of Quantities.

Run:
    python scripts/generate_rag_architecture_pdf.py --out outputs/rag_architecture.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#1b2433"
MUTED = "#5b6675"
PAGE_BG = "#f4f6fa"

C_INPUT = ("#e8f1ff", "#3b74d4")     # blue   - input / UI
C_DETECT = ("#e7f7ee", "#2f9e63")    # green  - detection / severity
C_KB = ("#fff3e0", "#e08a1e")        # amber  - knowledge bases (data)
C_RETRIEVE = ("#f0eaff", "#7a54d0")  # purple - retrieval
C_GEN = ("#ffeaf0", "#d8447a")       # pink   - generation
C_OUTPUT = ("#e3f6f8", "#1f97a8")    # teal   - output
C_NEUTRAL = ("#eef1f6", "#7a8698")   # grey


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _new_page(pdf: PdfPages, title: str, subtitle: str) -> plt.Axes:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    # header band
    ax.add_patch(FancyBboxPatch((0, 63.2), 100, 6.8, boxstyle="square,pad=0",
                                facecolor=INK, edgecolor="none", zorder=1))
    ax.text(3, 66.9, title, fontsize=20, fontweight="bold", color="white",
            va="center", zorder=2)
    ax.text(3, 64.4, subtitle, fontsize=10.5, color="#b9c4d6", va="center", zorder=2)
    ax.text(97, 65.6, "Construction Defect Detection  •  RAG Remedy Engine",
            fontsize=8.5, color="#8492a8", va="center", ha="right", zorder=2)
    return ax, fig


def _footer(ax: plt.Axes, page_no: int, note: str) -> None:
    ax.plot([3, 97], [3.4, 3.4], color="#d5dbe6", lw=0.8, zorder=1)
    ax.text(3, 2.1, note, fontsize=7.8, color=MUTED, va="center")
    ax.text(97, 2.1, f"Page {page_no}", fontsize=8, color=MUTED,
            va="center", ha="right")


def box(ax, x, y, w, h, title, lines=None, colors=C_NEUTRAL, title_size=10.5,
        body_size=8.3, rounded=0.4, lw=1.6, mono=False, align="center"):
    """Draw a rounded titled box. (x, y) is the lower-left corner."""
    face, edge = colors
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
        mutation_aspect=0.9))
    cx = x + w / 2
    if lines:
        ax.text(cx, y + h - 1.15, title, fontsize=title_size, fontweight="bold",
                color=INK, ha="center", va="top", zorder=4)
        body = "\n".join(lines)
        family = "monospace" if mono else None
        if align == "left":
            ax.text(x + 1.2, y + h - 2.9, body, fontsize=body_size, color=MUTED,
                    ha="left", va="top", zorder=4, linespacing=1.5, family=family)
        else:
            ax.text(cx, y + h - 2.9, body, fontsize=body_size, color=MUTED,
                    ha="center", va="top", zorder=4, linespacing=1.5, family=family)
    else:
        ax.text(cx, y + h / 2, title, fontsize=title_size, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4)
    return (x, y, w, h)


def tag(ax, x, y, text, colors, size=8):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), 0.1, 0.1, boxstyle="round,pad=0.55",
                                facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3))
    ax.text(x, y, text, fontsize=size, color=INK, ha="center", va="center",
            fontweight="bold", zorder=4)


def arrow(ax, p1, p2, color=INK, lw=2.0, style="-|>", label=None,
          label_offset=(0, 1.1), ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=16, color=color,
        linewidth=lw, zorder=5, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=7.6, color=color, ha="center",
                va="center", style="italic", zorder=6)


def right(b):  # mid-right edge
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


# ===========================================================================
# PAGE 1 — End-to-end system pipeline
# ===========================================================================
def page_system(pdf: PdfPages) -> None:
    ax, fig = _new_page(
        pdf,
        "End-to-End System Pipeline",
        "From an uploaded site photo to a grounded, cost-backed repair remedy")

    # Row 1 : linear pipeline (top)
    b_up = box(ax, 3, 47, 16.5, 11,
               "1  Image Input",
               ["Streamlit UI (app.py)", "Site photo + optional",
                "100 mm scale card", "-> mm-per-pixel"],
               C_INPUT, align="left")

    b_det = box(ax, 24, 47, 16.5, 11,
                "2  Defect Detection",
                ["Roboflow workflow", "(src/detect.py)", "-> defect class",
                 "-> bounding box"],
                C_DETECT, align="left")

    b_sev = box(ax, 45, 47, 16.5, 11,
                "3  Severity + Measure",
                ["src/severity.py", "box -> real size", "Minor..Critical",
                 "-> quantity (rm / m2 / m3)"],
                C_DETECT, align="left")

    b_rag = box(ax, 66, 47, 16.5, 11,
                "4  RAG Remedy Layer",
                ["src/remedy_rag.py", "+ src/boq.py", "retrieve -> ground",
                 "-> generate"],
                C_RETRIEVE, align="left")

    b_out = box(ax, 84.5, 47, 12.5, 11,
                "5  Output",
                ["Remedy plan", "BOQ + cost", "Sources", "PDF / HTML"],
                C_OUTPUT, align="left")

    arrow(ax, right(b_up), left(b_det))
    arrow(ax, right(b_det), left(b_sev))
    arrow(ax, right(b_sev), left(b_rag))
    arrow(ax, right(b_rag), left(b_out))

    ax.text(50, 60.5, "The first three stages produce the STRUCTURED QUERY. Stage 4 (this document's focus) "
            "retrieves engineering knowledge and generates the answer.",
            fontsize=9, color=MUTED, ha="center", style="italic")

    # Zoom bracket into the RAG layer
    ax.text(74.2, 45.3, "▼ expanded below", fontsize=8, color=C_RETRIEVE[1],
            ha="center", fontweight="bold")

    # Row 2 : expanded RAG layer
    ax.add_patch(FancyBboxPatch((3, 8), 94, 33, boxstyle="round,pad=0.2,rounding_size=0.6",
                 facecolor="#faf8ff", edgecolor=C_RETRIEVE[1], linewidth=1.4,
                 linestyle=(0, (6, 4)), zorder=2))
    ax.text(5, 39, "STAGE 4  —  RAG REMEDY LAYER (retrieval-augmented generation)",
            fontsize=11, fontweight="bold", color=C_RETRIEVE[1], va="center")

    b_q = box(ax, 5.5, 20, 15, 12,
              "Structured Query",
              ["RemedyQuery:", "- defect_class", "- severity_level",
               "- measurement", "- quantity / cost"],
              C_INPUT, align="left", body_size=8)

    # two knowledge bases
    b_kb1 = box(ax, 26, 28.5, 20, 8.2,
                "KB-1  Guidance text",
                ["data/remedy_knowledge.json", "11 chunks: ACI / IS / ICRI",
                 "how-to-repair narrative"],
                C_KB, align="left", body_size=7.8)
    b_kb2 = box(ax, 26, 18.5, 20, 8.2,
                "KB-2  Norms + rates",
                ["data/repair_norms.json", "material / labour / equipment",
                 "consumption norms + INR rates"],
                C_KB, align="left", body_size=7.8)

    b_r1 = box(ax, 51, 28.5, 19.5, 8.2,
               "Retriever  (text)",
               ["retrieve_context()", "token overlap + defect/",
                "severity boost -> top-k"],
               C_RETRIEVE, align="left", body_size=7.8)
    b_r2 = box(ax, 51, 18.5, 19.5, 8.2,
               "Retriever  (BOQ)",
               ["retrieve_norms() + compute_boq()", "qty = work x norm",
                "amount = qty x rate"],
               C_RETRIEVE, align="left", body_size=7.8)

    b_gen = box(ax, 75, 22.5, 19.5, 12,
                "Grounded Generation",
                ["build_llm_prompt()", "-> AI Engine  OR", "-> deterministic",
                 "   grounded answer", "= RagRemedy"],
                C_GEN, align="left", body_size=7.8)

    arrow(ax, right(b_q), (26, 32.6), rad=0.05)
    arrow(ax, right(b_q), (26, 22.6), rad=-0.05)
    arrow(ax, right(b_kb1), left(b_r1))
    arrow(ax, right(b_kb2), left(b_r2))
    arrow(ax, right(b_r1), (75, 30), rad=0.05)
    arrow(ax, right(b_r2), (75, 26), rad=-0.05)
    arrow(ax, right(b_gen), (96.3, 28.5), style="-|>")
    ax.text(90.5, 20.5, "RagRemedy\n-> UI / report", fontsize=7.6, color=C_GEN[1],
            ha="center", va="center", fontweight="bold")

    _footer(ax, 1, "Detection & severity produce the query; the RAG layer grounds and generates the remedy.")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ===========================================================================
# PAGE 2 — RAG internal architecture (the core)
# ===========================================================================
def page_rag_core(pdf: PdfPages) -> None:
    ax, fig = _new_page(
        pdf,
        "RAG Module — Internal Architecture",
        "src/remedy_rag.py  +  src/boq.py   |   dual knowledge base, retrieve → augment → generate")

    # Column band labels
    bands = [
        (3, 18, "QUERY", C_INPUT),
        (23, 22, "KNOWLEDGE BASE", C_KB),
        (47, 20, "RETRIEVAL", C_RETRIEVE),
        (69, 15, "AUGMENT", C_NEUTRAL),
        (85.5, 11.5, "GENERATE", C_GEN),
    ]
    for x, w, name, col in bands:
        ax.add_patch(FancyBboxPatch((x, 6.5), w, 52.5,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     facecolor=col[0], edgecolor="none", alpha=0.30, zorder=1))
        ax.text(x + w / 2, 57.6, name, fontsize=9, fontweight="bold",
                color=col[1], ha="center", va="center", zorder=2)

    # --- QUERY ---
    b_q = box(ax, 4, 33, 16, 17,
              "RemedyQuery",
              ["defect_class", "severity_level", "measured", "reason",
               "remedial_measure", "cost_breakup", "boq_breakup"],
              C_INPUT, align="left", body_size=8.4, mono=True)
    ax.text(12, 31, "built from stages 2–3", fontsize=7.5, color=MUTED,
            ha="center", style="italic")

    # --- KNOWLEDGE BASE ---
    b_kb1 = box(ax, 24, 38.5, 20, 15,
                "KB-1 : Guidance",
                ["remedy_knowledge.json", "", "KnowledgeChunk:",
                 " id / defect / severity", " source (code) / content",
                 "", "11 chunks, standards-cited"],
                C_KB, align="left", body_size=7.8)
    b_kb2 = box(ax, 24, 20, 20, 15,
                "KB-2 : Norms + Rates",
                ["repair_norms.json", "", "repair_norms[]  (per defect):",
                 " material / labour /", " equipment consumption",
                 "unit_rates[]  (INR / unit)"],
                C_KB, align="left", body_size=7.8)

    # --- RETRIEVAL ---
    b_r1 = box(ax, 48, 38.5, 19, 15,
               "retrieve_context()",
               ["1. tokenise query", "2. token-overlap score:",
                "   overlap / sqrt(|chunk|)", "3. +3.0 defect match",
                "   +2.0 severity match", "   +1.0 general", "4. rank -> top_k=4"],
               C_RETRIEVE, align="left", body_size=7.7, mono=False)
    b_r2 = box(ax, 48, 20, 19, 15,
               "retrieve_norms()",
               ["+ compute_boq()", "match defect + severity",
                "qty = work_qty x norm", "amount = qty x rate",
                "sum material/labour/eqp", "+15% O&P, +18% GST",
                "-> BoqEstimate"],
               C_RETRIEVE, align="left", body_size=7.6)

    # --- AUGMENT ---
    b_aug = box(ax, 69.5, 26, 14.5, 21,
                "build_llm_prompt()",
                ["Assemble grounded", "context:", "", "• defect + severity",
                 "• measurement + qty", "• retrieved chunks", "• BOQ table",
                 "• strict rules:", "  no invented norms,", "  qty x rate = amount"],
                C_NEUTRAL, align="left", body_size=7.6)

    # --- GENERATE (single box, two selectable paths) ---
    b_gen = box(ax, 85.8, 26, 11.2, 21,
                "Generation",
                ["IF AI key set:", "  AI Engine", "  generation,", "  temp 0.2",
                 "", "ELSE (default):", "  generate_grounded", "  _answer()",
                 "  deterministic,", "  no key needed", "", "-> RagRemedy"],
                C_GEN, align="left", body_size=7.1)

    # --- OUTPUT card (bottom, spanning) ---
    b_ans = box(ax, 24, 7.5, 73, 9.6,
                "RagRemedy  (returned to UI / report)",
                ["answer  (9-section remedy)        •        prompt  (auditable trail)"
                 "        •        retrieved_context  (chunks + scores)",
                 "sources  (standards cited)        •        used_llm / model / llm_error"],
                C_OUTPUT, align="center", body_size=8.2, title_size=11)

    # arrows
    arrow(ax, right(b_q), left(b_kb1), rad=0.12)
    arrow(ax, right(b_q), left(b_kb2), rad=-0.12)
    arrow(ax, right(b_kb1), left(b_r1))
    arrow(ax, right(b_kb2), left(b_r2))
    arrow(ax, right(b_r1), (69.5, 40), rad=0.05)
    arrow(ax, right(b_r2), (69.5, 33), rad=-0.05)
    arrow(ax, right(b_aug), left(b_gen))
    arrow(ax, bottom(b_gen), (91.4, 17.1), color=C_GEN[1])

    _footer(ax, 2, "Dual retrieval (narrative + numeric norms) → single grounded prompt → LLM or deterministic answer.")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ===========================================================================
# PAGE 3 — Retrieval scoring & grounding guarantees
# ===========================================================================
def page_mechanics(pdf: PdfPages) -> None:
    ax, fig = _new_page(
        pdf,
        "Retrieval Scoring & Grounding Guarantees",
        "How relevance is ranked and why every number in the answer is traceable")

    # Left: scoring pipeline
    ax.text(4, 57.5, "A.  Relevance scoring  (retrieve_context)", fontsize=12,
            fontweight="bold", color=C_RETRIEVE[1])

    steps = [
        ("Query text", ["defect + severity + measurement + quantity notes"], C_INPUT),
        ("Tokenise", ["lowercase, split on [a-z0-9_]; drop tokens <= 2 chars"], C_NEUTRAL),
        ("Score each chunk", ["base = overlap / sqrt(|chunk tokens|)",
                              "+3.0 defect match, +2.0 severity match,",
                              "+1.0 'general' chunk"], C_RETRIEVE),
        ("Rank & cut", ["sort by score desc; keep top_k = 4; drop score = 0"], C_RETRIEVE),
        ("Top-k chunks", ["most relevant standards -> prompt + shown as sources"], C_OUTPUT),
    ]
    y = 50.5
    prev = None
    for name, lines, col in steps:
        h = 6.7 if len(lines) >= 3 else 5.0
        b = box(ax, 5, y - h, 40, h, name, lines, col, align="left",
                title_size=9.5, body_size=7.6)
        if prev is not None:
            arrow(ax, bottom(prev), top(b), lw=1.8)
        prev = b
        y -= 8.2

    # Right top: worked score example
    ax.text(52, 57.5, "B.  Why cost is never hallucinated", fontsize=12,
            fontweight="bold", color=C_KB[1])

    box(ax, 52, 39, 45, 16,
        "Cost is COMPUTED, not retrieved",
        ["The knowledge base stores only NORMS and RATES.",
         "The rupee figure is always derived:",
         "",
         "   item quantity  =  work quantity  x  norm",
         "   line amount    =  item quantity  x  unit rate",
         "   grand total    =  Σ lines  +15% O&P  +18% GST",
         "",
         "=> no free-text price can enter the BOQ;",
         "   every number traces to a norm x a rate."],
        C_KB, align="left", body_size=8.2, title_size=10.5)

    # Right bottom: grounding guardrails
    box(ax, 52, 20, 45, 17,
        "Grounding guardrails",
        ["• Answer built ONLY from retrieved chunks + query.",
         "• Prompt forbids inventing standards, norms or rates.",
         "• Every cost line must show  qty x rate = amount.",
         "• Sources (ACI / IS / ICRI / CPWD) listed with answer.",
         "• Deterministic fallback runs with no LLM key,",
         "  so output is reproducible and testable.",
         "• Prompt is returned too -> fully auditable trail.",
         "• Image-based quantities flagged 'preliminary,",
         "  confirm on site' in every remedy."],
        C_GEN, align="left", body_size=8.2, title_size=10.5)

    # bottom strip: knowledge base coverage
    ax.add_patch(FancyBboxPatch((5, 4.6), 92, 7.0,
                 boxstyle="round,pad=0.15,rounding_size=0.4",
                 facecolor="#eef4ff", edgecolor=C_INPUT[1], linewidth=1.3, zorder=2))
    ax.text(6.8, 10.0, "Knowledge base coverage", fontsize=9.5,
            fontweight="bold", color=C_INPUT[1], va="center")
    ax.text(6.8, 7.4,
            "Defects: crack · spalling · honeycombing · exposed reinforcement · mold / dampness        "
            "Severities: Minor · Moderate · Severe · Critical",
            fontsize=8.0, color=MUTED, va="center")
    ax.text(6.8, 5.6,
            "Standards cited: ACI 224R-01 · ACI 562 · IS 456:2000 · ICRI 310.1 · Concrete Society TR54 · CPWD-style norms",
            fontsize=8.0, color=MUTED, va="center")

    _footer(ax, 3, "Token-overlap ranking + defect/severity boosts select context; arithmetic (qty × rate) guarantees traceable cost.")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/rag_architecture.pdf",
                        help="Output PDF path (relative to project root)")
    args = parser.parse_args()

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        page_system(pdf)
        page_rag_core(pdf)
        page_mechanics(pdf)
        info = pdf.infodict()
        info["Title"] = "RAG Remedy Engine — Architecture"
        info["Subject"] = "Retrieval-Augmented Remedy Generation for Construction Defects"

    print(f"Wrote architecture PDF: {out_path}")


if __name__ == "__main__":
    main()
