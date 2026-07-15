"""Construction Defect Detection — Streamlit app.

End-to-end demo that matches the PDF report pipeline:
  detect (Roboflow model + AI vision fallback + expert overrides)
  -> grade severity  -> classify structural element  -> RAG repair remedy.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.pipeline import analyze, annotate, SEV_COLORS, SEV_ORDER, source_color
from src.remedy_rag import RemedyQuery, generate_rag_remedy
from src.severity import mm_per_pixel_from_reference

st.set_page_config(page_title="Construction Defect Detection",
                   page_icon="🏗️", layout="wide")

SEV_SHORT = {"Minor": "Minor", "Moderate": "Mod.", "Severe": "Severe", "Critical": "Crit."}


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def chip(text: str, face: str, edge: str, size: int = 12) -> str:
    return (f'<span style="background:{face};color:{edge};border:1px solid {edge};'
            f'border-radius:12px;padding:2px 12px;font-weight:700;font-size:{size}px;'
            f'white-space:nowrap">{text}</span>')


def sev_chip(level: str, size: int = 12) -> str:
    face, edge = SEV_COLORS.get(level, SEV_COLORS["Critical"])
    return chip(level.upper(), face, edge, size)


def sev_meter(level: str) -> str:
    cells = []
    for name in ["Minor", "Moderate", "Severe", "Critical"]:
        face, edge = SEV_COLORS[name]
        active = name == level
        bg = edge if active else "#eef1f6"
        col = "#ffffff" if active else "#9aa4b2"
        bd = edge if active else "#cdd4de"
        cells.append(
            f'<div style="flex:1;text-align:center;background:{bg};color:{col};'
            f'border:1px solid {bd};border-radius:5px;padding:3px 0;font-size:11px;'
            f'font-weight:{700 if active else 400}">{SEV_SHORT[name]}</div>')
    return f'<div style="display:flex;gap:4px;margin:6px 0">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Cached compute (so re-runs from widget clicks don't re-hit the APIs)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis(image_bytes: bytes, name: str, conf: float, use_fallback: bool,
                 do_element: bool, mm_per_pixel: float | None, gpt_model: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / name
        in_path.write_bytes(image_bytes)
        result = analyze(in_path, conf=conf, use_fallback=use_fallback,
                         do_element=do_element, mm_per_pixel=mm_per_pixel,
                         gpt_model=gpt_model)
        out_path = Path(td) / "annotated.jpg"
        annotate(in_path, result["graded"], out_path)
        result["annotated"] = out_path.read_bytes()
    return result


@st.cache_data(show_spinner=False)
def run_remedy(defect: str, severity: str, measured: str, reason: str,
               remedial_measure: str, repair_time: str, cost_breakup: dict,
               boq_breakup: dict, use_openai: bool, model: str) -> dict:
    remedy = generate_rag_remedy(
        RemedyQuery(defect_class=defect, severity_level=severity, measured=measured,
                    reason=reason, remedial_measure=remedial_measure,
                    repair_time_estimate=repair_time, cost_breakup=cost_breakup,
                    boq_breakup=boq_breakup),
        use_openai=use_openai, openai_model=model)
    return {"answer": remedy.answer, "sources": list(remedy.sources),
            "used_llm": remedy.used_llm, "model": remedy.model,
            "llm_error": remedy.llm_error}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏗️ Construction Defect Detection")
st.caption("Detect concrete defects, grade severity, identify the structural "
           "element, and generate a grounded repair remedy — from a single photo.")

has_roboflow = bool(os.getenv("ROBOFLOW_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))
if not has_roboflow:
    st.error("ROBOFLOW_API_KEY is not set. Add it in the Space secrets (or .env) "
             "to run detection.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
use_fallback = st.sidebar.checkbox(
    "AI vision fallback", value=has_openai, disabled=not has_openai,
    help="When the primary model finds nothing, a secondary AI vision detector "
         "looks for defects it does not cover (e.g. exposed reinforcement).")
do_element = st.sidebar.checkbox(
    "Classify structural element", value=has_openai, disabled=not has_openai,
    help="Identify slab / wall / beam / column / staircase from the photo.")
gpt_model = "gpt-4o"

st.sidebar.markdown("---")
st.sidebar.subheader("Scale reference (optional)")
st.sidebar.caption("Grade cracks by real width (ACI 224R / IS 456). Leave off for "
                   "area-based grading.")
use_scale = st.sidebar.checkbox("Use a scale reference", value=False)
mm_per_pixel: float | None = None
if use_scale:
    ref_mm = st.sidebar.number_input("Reference real size (mm)", 0.0, value=100.0, step=10.0)
    ref_px = st.sidebar.number_input("Reference size in image (px)", 0.0, value=250.0, step=10.0)
    if ref_mm > 0 and ref_px > 0:
        mm_per_pixel = mm_per_pixel_from_reference(ref_mm, ref_px)
        st.sidebar.caption(f"Scale: {mm_per_pixel:.4f} mm/px")

st.sidebar.markdown("---")
st.sidebar.subheader("Repair remedy (RAG)")
gen_remedy = st.sidebar.checkbox("Generate repair remedy", value=True)
use_openai_rag = st.sidebar.checkbox("Use OpenAI for remedy text", value=False,
                                     disabled=not has_openai)
openai_model = st.sidebar.text_input("OpenAI model", "gpt-4o-mini",
                                     disabled=not use_openai_rag)

if not has_openai:
    st.sidebar.info("OPENAI_API_KEY not set — AI fallback, element classification "
                    "and OpenAI remedy text are disabled. Detection + severity still work.")

with st.sidebar.expander("How it works"):
    st.markdown(
        "1. **Detection** — Roboflow model; an AI vision detector fills gaps it "
        "doesn't cover, and expert-reviewed annotations can be added.\n"
        "2. **Severity** — graded per ACI / IS / ICRI standards (area-based unless "
        "a scale is given).\n"
        "3. **Structural element** — AI-classified (slab/wall/beam/column/staircase).\n"
        "4. **Remedy** — retrieval-augmented plan grounded in engineering norms + BOQ.")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a concrete surface image",
                            type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("⬆️ Upload an image to run the analysis.")
    st.stop()

image_bytes = uploaded.getvalue()
with st.spinner("Analysing image (detection · severity · structural element)…"):
    result = run_analysis(image_bytes, uploaded.name, confidence, use_fallback,
                          do_element, mm_per_pixel, gpt_model)

graded = result["graded"]
element = (result.get("element") or {}).get("element", "unknown")
elem_conf = (result.get("element") or {}).get("confidence", 0.0)
worst = min((g["severity"] for g in graded),
            key=lambda s: SEV_ORDER.index(s) if s in SEV_ORDER else 99) if graded else "—"

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Detections", len(graded))
m2.markdown("**Worst severity**")
m2.markdown(sev_chip(worst) if graded else "—", unsafe_allow_html=True)
m3.metric("Structural element", element.title(),
          help=f"AI-classified · {elem_conf*100:.0f}% confidence" if elem_conf else None)
m4.metric("Image size", f"{result['width']}×{result['height']}")

st.markdown("---")

left, right = st.columns([1.15, 1])

with left:
    st.image(result["annotated"], caption="Annotated detection (boxes coloured by "
             "severity; numbers key to the panel)", use_container_width=True)
    st.download_button("⬇️ Download annotated image", result["annotated"],
                       file_name=f"{Path(uploaded.name).stem}_annotated.jpg",
                       mime="image/jpeg")

with right:
    # structural element banner
    st.markdown(
        f'<div style="border:1.5px solid #3b74d4;background:#eef4ff;border-radius:8px;'
        f'padding:10px 14px;margin-bottom:10px">'
        f'<span style="color:#3b74d4;font-weight:700;font-size:12px">STRUCTURAL ELEMENT</span>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<span style="font-size:22px;font-weight:800;color:#1b2433">{element.title()}</span>'
        f'<span style="color:#5b6675;font-size:12px;font-style:italic">'
        f'{"AI-classified · " + format(elem_conf*100, ".0f") + "%" if elem_conf else "AI-classified"}'
        f'</span></div></div>', unsafe_allow_html=True)

    if not graded:
        st.warning("No defects detected at the selected confidence threshold.")
        if result.get("roboflow_empty") and not use_fallback and has_openai:
            st.caption("Tip: enable the AI vision fallback in the sidebar to catch "
                       "defects the primary model does not cover.")
    for i, g in enumerate(graded, start=1):
        face, edge = SEV_COLORS.get(g["severity"], SEV_COLORS["Critical"])
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:6px"><span style="font-size:16px;font-weight:700;color:#1b2433">'
            f'{i}. {g["defect"].replace("_", " ").title()}</span>{sev_chip(g["severity"])}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:12px;color:#1b2433;margin-top:2px">'
            f'Severity <b>{g["score"]}/4</b> &nbsp;·&nbsp; extent <b>{g["area_pct"]:.1f}%</b>'
            f' &nbsp;·&nbsp; conf {g["confidence"]*100:.0f}% &nbsp;·&nbsp; '
            f'<span style="color:{source_color(g["source"])};font-style:italic;font-weight:600">'
            f'via {g["source"]}</span></div>', unsafe_allow_html=True)
        st.markdown(sev_meter(g["severity"]), unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#5b6675"><b>Reason:</b> '
                    f'{g["reason"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{edge}"><b>Recommended action:</b> '
                    f'{g["action"]}</div>', unsafe_allow_html=True)
        st.caption(f"Basis: {g['measured']} · Standard: {g['standard']}")

        if gen_remedy:
            with st.expander(f"Repair remedy plan — {g['defect'].replace('_',' ').title()}"):
                with st.spinner("Generating grounded remedy…"):
                    rem = run_remedy(g["defect"], g["severity"], g["measured"],
                                     g["reason"], g["remedial_measure"],
                                     g["repair_time_estimate"], g["cost_breakup"],
                                     g["boq_breakup"], use_openai_rag, openai_model)
                if rem["used_llm"]:
                    st.success(f"Generated with OpenAI model: {rem['model']}")
                elif rem["llm_error"]:
                    st.warning(f"OpenAI unavailable; showing grounded fallback: {rem['llm_error']}")
                st.markdown(rem["answer"])
                if rem["sources"]:
                    st.caption("Sources: " + "; ".join(rem["sources"]))
        st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #e6e9ef'>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Table + CSV
# ---------------------------------------------------------------------------
if graded:
    st.subheader("Detection summary")
    rows = [{
        "#": i, "Structural element": element.title(),
        "Defect": g["defect"].replace("_", " ").title(),
        "Detection source": g["source"], "Confidence": round(g["confidence"], 3),
        "Severity": g["severity"], "Severity score": f"{g['score']}/4",
        "Affected extent %": round(g["area_pct"], 2), "Basis": g["measured"],
        "Standard": g["standard"], "Recommended action": g["action"],
    } for i, g in enumerate(graded, start=1)]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(uploaded.name).stem}_detections.csv",
                       mime="text/csv")

st.caption("Detection: Roboflow model + AI vision detector fallback. Structural "
           "element and (optional) remedy text are AI-generated. Image-based "
           "measurements are preliminary and must be confirmed on site.")
