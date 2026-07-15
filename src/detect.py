from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from remedy_rag import RemedyQuery, generate_rag_remedy
from severity import estimate_severity, mm_per_pixel_from_reference


def run_detection(
    model_path: str,
    image_path: str,
    output_dir: str,
    confidence: float,
    mm_per_pixel: float | None = None,
    use_openai_rag: bool = False,
    openai_model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=confidence, save=True, project=str(output_path), name="predictions")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_height, image_width = image.shape[:2]

    rows: list[dict] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            defect_class = names[class_id]
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
            rows.append(
                {
                    "class": defect_class,
                    "confidence": round(confidence_score, 3),
                    "severity": severity.level,
                    "area_ratio": round(severity.area_ratio, 4),
                    "basis": severity.measured,
                    "standard": severity.standard,
                    "reason": severity.reason,
                    "recommended_action": severity.recommended_action,
                    "remedial_measure": severity.remedial_measure,
                    "repair_quantity": severity.cost_breakup.get("quantity", ""),
                    "material_rate": severity.cost_breakup.get("material_rate", ""),
                    "labour_rate": severity.cost_breakup.get("labour_rate", ""),
                    "equipment_rate": severity.cost_breakup.get("equipment_rate", ""),
                    "composite_rate": severity.cost_breakup.get("composite_rate", ""),
                    "material_cost": round(severity.cost_breakup.get("material_cost", 0), 2),
                    "labour_cost": round(severity.cost_breakup.get("labour_cost", 0), 2),
                    "equipment_cost": round(severity.cost_breakup.get("equipment_cost", 0), 2),
                    "total_repair_cost": round(severity.cost_breakup.get("total_cost", 0), 2),
                    "repair_cost_estimate": severity.repair_cost_estimate,
                    "repair_time_estimate": severity.repair_time_estimate,
                    "rag_remedy_plan": rag_remedy.answer,
                    "rag_sources": "; ".join(rag_remedy.sources),
                    "rag_used_openai": rag_remedy.used_llm,
                    "rag_model": rag_remedy.model,
                    "rag_error": rag_remedy.llm_error,
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_path / "detections.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect construction defects using trained YOLO weights")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--ref-mm", type=float, default=None, help="Known reference size in mm (for crack-width grading)")
    parser.add_argument("--ref-px", type=float, default=None, help="That reference's size in pixels")
    parser.add_argument("--use-openai-rag", action="store_true", help="Use OpenAI to generate RAG remedy text")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for RAG remedy text")
    args = parser.parse_args()

    mm_per_pixel = None
    if args.ref_mm and args.ref_px:
        mm_per_pixel = mm_per_pixel_from_reference(args.ref_mm, args.ref_px)

    table = run_detection(
        args.model,
        args.image,
        args.output,
        args.conf,
        mm_per_pixel,
        use_openai_rag=args.use_openai_rag,
        openai_model=args.openai_model,
    )
    if table.empty:
        print("No defects detected.")
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
