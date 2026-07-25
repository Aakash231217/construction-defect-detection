from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline import annotate
from src.roboflow_model import RoboflowModelError, run_model
from src.remedy_rag import RemedyQuery, generate_rag_remedy
from src.severity import estimate_severity, mm_per_pixel_from_reference


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
                            "Affected Area %": round(severity.area_ratio * 100, 2),
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
                            "Repair Time Estimate": severity.repair_time_estimate,
                            "RAG Sources": "; ".join(rag_remedy.sources),
                        }
                    )
                annotated_path = annotate(
                    temporary_path,
                    graded_detections,
                    Path("outputs/workflow/hosted_detection.jpg"),
                )
                output_column, table_column = st.columns([1.1, 1])
                with output_column:
                    st.image(
                        str(annotated_path),
                        caption="Detection Output (severity-colored boxes)",
                        use_container_width=True,
                    )
                with table_column:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.subheader("RAG Remedy Generation")
                for index, (defect_name, severity_level, rag_remedy) in enumerate(rag_reports, start=1):
                    with st.expander(f"{index}. {defect_name} - {severity_level} remedy plan"):
                        if rag_remedy.used_llm:
                            st.success(f"Generated with OpenAI model: {rag_remedy.model}")
                        elif rag_remedy.llm_error:
                            st.warning(f"OpenAI generation unavailable, showing grounded fallback: {rag_remedy.llm_error}")
                        st.markdown(rag_remedy.answer)
                        st.caption("LLM prompt is generated from retrieved context and can be sent to Mistral 7B or another model.")
                        st.code(rag_remedy.prompt, language="text")
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
            "Affected Area %": round(severity.area_ratio * 100, 2),
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
            "Repair Time Estimate": severity.repair_time_estimate,
            "RAG Sources": "; ".join(rag_remedy.sources),
        }
    )

left, right = st.columns([1.2, 1])
with left:
    st.image(annotated_image, caption="Detection Output", use_container_width=True)
with right:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("RAG Remedy Generation")
        for index, (defect_name, severity_level, rag_remedy) in enumerate(rag_reports, start=1):
            with st.expander(f"{index}. {defect_name} - {severity_level} remedy plan"):
                if rag_remedy.used_llm:
                    st.success(f"Generated with OpenAI model: {rag_remedy.model}")
                elif rag_remedy.llm_error:
                    st.warning(f"OpenAI generation unavailable, showing grounded fallback: {rag_remedy.llm_error}")
                st.markdown(rag_remedy.answer)
                st.caption("LLM prompt is generated from retrieved context and can be sent to Mistral 7B or another model.")
                st.code(rag_remedy.prompt, language="text")
    else:
        st.warning("No defect detected at the selected confidence threshold.")
