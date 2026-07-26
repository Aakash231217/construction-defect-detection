from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from PIL import Image

from src.cost_estimation import estimate_repair_days
from src.pipeline import annotate, classify_element
from src.roboflow_model import RoboflowModelError, run_model
from src.remedy_rag import RemedyQuery, generate_rag_remedy
from src.severity import estimate_severity, mm_per_pixel_from_reference


SEVERITY_COLORS = {
    "Negligible": ("#eef1f6", "#7a8698"),
    "Minor": ("#e7f7ee", "#2f9e63"),
    "Moderate": ("#fff3e0", "#e08a1e"),
    "Severe": ("#ffe7d6", "#e2691a"),
    "Critical": ("#ffe0e6", "#d83a52"),
}
SEVERITY_SHORT = {"Minor": "Minor", "Moderate": "Mod.", "Severe": "Severe", "Critical": "Crit."}


def _severity_chip(level: str) -> str:
    face, edge = SEVERITY_COLORS.get(level, SEVERITY_COLORS["Critical"])
    return (
        f'<span style="background:{face};color:{edge};border:1px solid {edge};'
        f'border-radius:12px;padding:2px 12px;font-weight:700;font-size:12px;'
        f'white-space:nowrap">{level.upper()}</span>'
    )


def _severity_meter(level: str) -> str:
    cells = []
    for name in ("Minor", "Moderate", "Severe", "Critical"):
        _, edge = SEVERITY_COLORS[name]
        active = name == level
        cells.append(
            f'<div style="flex:1;text-align:center;background:{edge if active else "#eef1f6"};'
            f'color:{"#ffffff" if active else "#9aa4b2"};'
            f'border:1px solid {edge if active else "#cdd4de"};border-radius:5px;'
            f'padding:5px 0;font-size:12px;font-weight:{700 if active else 400}">'
            f'{SEVERITY_SHORT[name]}</div>'
        )
    return f'<div style="display:flex;gap:6px;margin:8px 0">{"".join(cells)}</div>'


def _render_detection_details(rows: list[dict], rag_reports: list[tuple], element_result: dict) -> None:
    element = str(element_result.get("element", "unknown") or "unknown").replace("_", " ").title()
    element_confidence = float(element_result.get("confidence", 0.0) or 0.0)
    confidence_label = (
        f"AI-classified · {element_confidence * 100:.0f}%"
        if element_confidence
        else "AI classification unavailable"
    )
    st.markdown(
        f'<div style="border:1.5px solid #3b74d4;background:#eef4ff;border-radius:8px;'
        f'padding:14px 16px;margin-bottom:14px">'
        f'<span style="color:#3b74d4;font-weight:700;font-size:12px">STRUCTURAL ELEMENT</span>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<span style="font-size:24px;font-weight:800;color:#1b2433">{element}</span>'
        f'<span style="color:#5b6675;font-size:12px;font-style:italic">{confidence_label}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    for index, (row, rag_entry) in enumerate(zip(rows, rag_reports), start=1):
        rag_report = rag_entry[2] if isinstance(rag_entry, tuple) else rag_entry
        severity = str(row["Severity"])
        _, edge = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["Critical"])
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">'
            f'<span style="font-size:18px;font-weight:700;color:#1b2433">{index}. {row["Defect"]}</span>'
            f'{_severity_chip(severity)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:13px;color:#1b2433;margin-top:4px">'
            f'Severity <b>{row["Severity Score"]}/4</b> &nbsp;·&nbsp; '
            f'extent <b>{float(row["Affected Area %"]):.1f}%</b> &nbsp;·&nbsp; '
            f'conf {float(row["Confidence"]) * 100:.0f}% &nbsp;·&nbsp; '
            f'<span style="color:#3b74d4;font-style:italic;font-weight:600">'
            f'via {row["Detection Source"]}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_severity_meter(severity), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:13px;color:#5b6675"><b>Reason:</b> {row["Remark"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:13px;color:{edge};margin-top:4px"><b>Recommended action:</b> '
            f'{row["Recommended Action"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:13px;color:#1b2433;margin-top:8px">'
            f'<b>Est. repair cost:</b> INR {float(row["Total Repair Cost (INR)"]):,.0f} &nbsp;·&nbsp; '
            f'<b>Repair time:</b> ~{row["Repair Time (days)"]} working day(s) '
            f'<span style="color:#5b6675">(band: {row["Repair Time Estimate"]})</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f'Basis: {row["Basis"]} · Standard: {row["Standard"]}')
        with st.expander(f'Repair remedy plan — {row["Defect"]}'):
            if rag_report.used_llm:
                st.success("Generated with the AI engine.")
            elif rag_report.llm_error:
                st.warning("AI engine unavailable; showing the grounded remedy instead.")
            st.markdown(rag_report.answer)
            if rag_report.sources:
                st.caption("Sources: " + "; ".join(rag_report.sources))
        st.markdown("<hr style='margin:10px 0;border:none;border-top:1px solid #e6e9ef'>", unsafe_allow_html=True)


def _render_cost_time_charts(rows: list[dict]) -> None:
    if not rows:
        return

    st.subheader("Cost & Time Analysis")
    total_cost = sum(float(row["Total Repair Cost (INR)"]) for row in rows)
    total_days = sum(int(row["Repair Time (days)"]) for row in rows)
    longest = max(int(row["Repair Time (days)"]) for row in rows)
    cost_metric, time_metric, longest_metric = st.columns(3)
    cost_metric.metric("Total estimated cost", f"INR {total_cost:,.0f}")
    time_metric.metric(
        "Total repair time",
        f"{total_days} working days",
        help="Sum of all repairs (sequential upper bound).",
    )
    longest_metric.metric(
        "Longest single repair",
        f"{longest} days",
        help="Critical-path duration if repairs run in parallel.",
    )

    labels = [f"{index}. {row['Defect']}" for index, row in enumerate(rows, start=1)]
    cost_column, time_column = st.columns(2)
    with cost_column:
        st.markdown("**Cost breakdown per defect (INR)**")
        cost_data = pd.DataFrame(
            {
                "Material": [float(row["Material Cost (INR)"]) for row in rows],
                "Labour": [float(row["Labour Cost (INR)"]) for row in rows],
                "Equipment": [float(row["Equipment Cost (INR)"]) for row in rows],
            },
            index=labels,
        )
        st.bar_chart(cost_data, height=260)
    with time_column:
        st.markdown("**Repair time per defect (working days)**")
        time_data = pd.DataFrame(
            {"Repair days": [int(row["Repair Time (days)"]) for row in rows]},
            index=labels,
        )
        st.bar_chart(time_data, height=260, color="#e08a1e")

    severity_column, pareto_column = st.columns(2)
    with severity_column:
        st.markdown("**Severity distribution**")
        severity_order = ["Minor", "Moderate", "Severe", "Critical"]
        counts = {
            severity: sum(1 for row in rows if row["Severity"] == severity)
            for severity in severity_order
        }
        severity_data = pd.DataFrame(
            {"Detections": list(counts.values())},
            index=list(counts.keys()),
        )
        st.bar_chart(severity_data, height=260, color="#d83a52")
    with pareto_column:
        st.markdown("**Cumulative cost curve (Pareto)**")
        ranked = sorted(rows, key=lambda row: float(row["Total Repair Cost (INR)"]), reverse=True)
        cumulative_costs: list[float] = []
        running_cost = 0.0
        for row in ranked:
            running_cost += float(row["Total Repair Cost (INR)"])
            cumulative_costs.append(round(running_cost, 2))
        pareto_data = pd.DataFrame(
            {"Cumulative cost (INR)": cumulative_costs},
            index=[f"{index}. {row['Defect']}" for index, row in enumerate(ranked, start=1)],
        )
        st.area_chart(pareto_data, height=260, color="#3b74d4")

    st.caption(
        "Cost = norms-based BOQ total including overheads and GST. Repair time = "
        "labour man-days / crew + curing allowance. Confirm quantities and local rates on site."
    )


def _render_analysis_summary(
    rows: list[dict],
    element_result: dict,
    image_width: float,
    image_height: float,
) -> None:
    severity_rank = {"Negligible": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    worst_severity = max(
        (str(row["Severity"]) for row in rows),
        key=lambda level: severity_rank.get(level, -1),
        default="None",
    )
    element = str(element_result.get("element", "unknown") or "unknown").replace("_", " ").title()
    confidence = float(element_result.get("confidence", 0.0) or 0.0)

    detection_metric, severity_metric, element_metric, size_metric = st.columns(4)
    detection_metric.metric("Detections", len(rows))
    severity_metric.metric("Worst severity", worst_severity)
    element_metric.metric(
        "Structural element",
        element,
        help=f"AI-classified with {confidence * 100:.0f}% confidence" if confidence else "AI classification unavailable",
    )
    size_metric.metric("Image size", f"{int(image_width)} x {int(image_height)} px")
    st.divider()


st.set_page_config(page_title="Construction Defect Detection", layout="wide")
st.title("Construction Defect Detection using YOLO")

mode = st.sidebar.radio("Inference Mode", ["Roboflow Hosted Model", "Local YOLO Weights"])
confidence = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Scale reference (optional)")
st.sidebar.caption(
    "Provide a known object in the photo to grade cracks by real width "
    "(ACI 224R / IS 456). Leave blank to grade by surface area only."
)
use_scale = st.sidebar.checkbox("Use a scale reference", value=False)
mm_per_pixel: float | None = None
if use_scale:
    ref_mm = st.sidebar.number_input("Reference real size (mm)", min_value=0.0, value=100.0, step=10.0)
    ref_px = st.sidebar.number_input("Reference size in image (px)", min_value=0.0, value=250.0, step=10.0)
    if ref_mm > 0 and ref_px > 0:
        mm_per_pixel = mm_per_pixel_from_reference(ref_mm, ref_px)
        st.sidebar.caption(f"Scale: {mm_per_pixel:.4f} mm/px")

st.sidebar.markdown("---")
st.sidebar.subheader("RAG remedy generation")
use_openai_rag = st.sidebar.checkbox("Use OpenAI for remedy text", value=False)
openai_model = st.sidebar.text_input("OpenAI model", "gpt-4o-mini", disabled=not use_openai_rag)
st.sidebar.caption(
    "OpenAI uses the retrieved engineering context and detected defect data. "
    "Set OPENAI_API_KEY in your environment or .env file."
)
classify_structure = st.sidebar.checkbox(
    "Classify structural element",
    value=True,
    help="Identify the primary slab, wall, beam, column, staircase, footing, or other element.",
)

uploaded_file = st.file_uploader("Upload a concrete surface image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to run defect detection.")
    st.stop()

with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temporary_file:
    temporary_file.write(uploaded_file.getbuffer())
    temporary_path = temporary_file.name

image = Image.open(temporary_path).convert("RGB")
image_width, image_height = image.size

if mode == "Roboflow Hosted Model":
    st.image(image, caption="Input Image", use_container_width=True)
    if st.button("Run Roboflow Detection", type="primary"):
        try:
            inference = run_model(temporary_path)
            predictions = [
                item
                for item in inference.get("predictions", [])
                if float(item.get("confidence", 0.0)) >= confidence
            ]
            inference_image = inference.get("image", {})
            source_width = float(inference_image.get("width", image_width))
            source_height = float(inference_image.get("height", image_height))
            st.success("Roboflow hosted detection completed.")

            if isinstance(predictions, list) and predictions:
                rows = []
                rag_reports = []
                graded_detections = []
                for item in predictions:
                    defect_class = str(item.get("class", ""))
                    severity = estimate_severity(
                        defect_class=defect_class or "defect",
                        box_width=float(item.get("width", 0.0)),
                        box_height=float(item.get("height", 0.0)),
                        image_width=source_width,
                        image_height=source_height,
                        mm_per_pixel=mm_per_pixel,
                    )
                    box_width = float(item.get("width", 0.0))
                    box_height = float(item.get("height", 0.0))
                    center_x = float(item.get("x", 0.0))
                    center_y = float(item.get("y", 0.0))
                    graded_detections.append(
                        {
                            "box": (
                                center_x - box_width / 2,
                                center_y - box_height / 2,
                                center_x + box_width / 2,
                                center_y + box_height / 2,
                            ),
                            "severity": severity.level,
                        }
                    )
                    rag_remedy = generate_rag_remedy(
                        RemedyQuery(
                            defect_class=defect_class or "defect",
                            severity_level=severity.level,
                            measured=severity.measured,
                            reason=severity.reason,
                            remedial_measure=severity.remedial_measure,
                            repair_time_estimate=severity.repair_time_estimate,
                            cost_breakup=severity.cost_breakup,
                            boq_breakup=severity.boq_breakup,
                        ),
                        use_openai=use_openai_rag,
                        openai_model=openai_model,
                    )
                    rag_reports.append((defect_class.replace("_", " ").title(), severity.level, rag_remedy))
                    rows.append(
                        {
                            "Defect": defect_class.replace("_", " ").title(),
                            "Confidence": round(float(item.get("confidence", 0.0)), 3),
                            "Severity": severity.level,
                            "Severity Score": severity.score,
                            "Affected Area %": round(severity.area_ratio * 100, 2),
                            "Detection Source": "Roboflow",
                            "Basis": severity.measured,
                            "Standard": severity.standard,
                            "Remark": severity.reason,
                            "Recommended Action": severity.recommended_action,
                            "Remedial Measure": severity.remedial_measure,
                            "Repair Quantity": severity.cost_breakup.get("quantity", ""),
                            "Material Rate (INR)": severity.cost_breakup.get("material_rate", ""),
                            "Labour Rate (INR)": severity.cost_breakup.get("labour_rate", ""),
                            "Equipment Rate (INR)": severity.cost_breakup.get("equipment_rate", ""),
                            "Final BOQ Rate incl. OH/GST (INR)": round(severity.cost_breakup.get("composite_rate", 0), 2),
                            "Material Cost (INR)": round(severity.cost_breakup.get("material_cost", 0), 2),
                            "Labour Cost (INR)": round(severity.cost_breakup.get("labour_cost", 0), 2),
                            "Equipment Cost (INR)": round(severity.cost_breakup.get("equipment_cost", 0), 2),
                            "Total Repair Cost (INR)": round(severity.cost_breakup.get("total_cost", 0), 2),
                            "Repair Time (days)": estimate_repair_days(severity.boq_breakup, severity.level),
                            "Repair Time Estimate": severity.repair_time_estimate,
                            "RAG Sources": "; ".join(rag_remedy.sources),
                        }
                    )
                annotated_path = annotate(
                    temporary_path,
                    graded_detections,
                    Path("outputs/workflow/hosted_detection.jpg"),
                )
                element_result = (
                    classify_element(temporary_path, model="gpt-4o-mini")
                    if classify_structure
                    else {"element": "not classified", "confidence": 0.0}
                )
                _render_analysis_summary(rows, element_result, source_width, source_height)
                output_column, table_column = st.columns([1.1, 1])
                with output_column:
                    st.image(
                        str(annotated_path),
                        caption="Detection Output (severity-colored boxes)",
                        use_container_width=True,
                    )
                with table_column:
                    _render_detection_details(rows, rag_reports, element_result)
                st.subheader("Detection Summary")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                _render_cost_time_charts(rows)
            else:
                st.warning("No defects returned by the hosted model.")
        except (RoboflowModelError, FileNotFoundError) as error:
            st.error(str(error))
    st.stop()

model_path = st.sidebar.text_input("Model weights", "models/best.pt")
if not Path(model_path).exists():
    st.error("Model weights were not found. Train the model first or place best.pt in the models folder.")
    st.stop()

from ultralytics import YOLO

model = YOLO(model_path)
results = model.predict(source=temporary_path, conf=confidence)
result = results[0]
annotated_image = result.plot()[:, :, ::-1]

rows: list[dict] = []
rag_reports = []
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    class_id = int(box.cls[0].item())
    defect_class = result.names[class_id]
    severity = estimate_severity(
        defect_class=defect_class,
        box_width=x2 - x1,
        box_height=y2 - y1,
        image_width=image_width,
        image_height=image_height,
        mm_per_pixel=mm_per_pixel,
    )
    rag_remedy = generate_rag_remedy(
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
        use_openai=use_openai_rag,
        openai_model=openai_model,
    )
    rag_reports.append((defect_class.replace("_", " ").title(), severity.level, rag_remedy))
    rows.append(
        {
            "Defect": defect_class.replace("_", " ").title(),
            "Confidence": round(float(box.conf[0].item()), 3),
            "Severity": severity.level,
            "Severity Score": severity.score,
            "Affected Area %": round(severity.area_ratio * 100, 2),
            "Detection Source": "Local YOLO",
            "Basis": severity.measured,
            "Standard": severity.standard,
            "Remark": severity.reason,
            "Recommended Action": severity.recommended_action,
            "Remedial Measure": severity.remedial_measure,
            "Repair Quantity": severity.cost_breakup.get("quantity", ""),
            "Material Rate (INR)": severity.cost_breakup.get("material_rate", ""),
            "Labour Rate (INR)": severity.cost_breakup.get("labour_rate", ""),
            "Equipment Rate (INR)": severity.cost_breakup.get("equipment_rate", ""),
            "Final BOQ Rate incl. OH/GST (INR)": round(severity.cost_breakup.get("composite_rate", 0), 2),
            "Material Cost (INR)": round(severity.cost_breakup.get("material_cost", 0), 2),
            "Labour Cost (INR)": round(severity.cost_breakup.get("labour_cost", 0), 2),
            "Equipment Cost (INR)": round(severity.cost_breakup.get("equipment_cost", 0), 2),
            "Total Repair Cost (INR)": round(severity.cost_breakup.get("total_cost", 0), 2),
            "Repair Time (days)": estimate_repair_days(severity.boq_breakup, severity.level),
            "Repair Time Estimate": severity.repair_time_estimate,
            "RAG Sources": "; ".join(rag_remedy.sources),
        }
    )

element_result = (
    classify_element(temporary_path, model="gpt-4o-mini")
    if classify_structure
    else {"element": "not classified", "confidence": 0.0}
)
_render_analysis_summary(rows, element_result, image_width, image_height)

left, right = st.columns([1.2, 1])
with left:
    st.image(annotated_image, caption="Detection Output", use_container_width=True)
with right:
    if rows:
        _render_detection_details(rows, rag_reports, element_result)
    else:
        st.warning("No defect detected at the selected confidence threshold.")

if rows:
    st.subheader("Detection Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
_render_cost_time_charts(rows)
