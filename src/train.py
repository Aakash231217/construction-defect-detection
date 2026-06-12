from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO for construction defect detection")
    parser.add_argument("--data", default="config/dataset.yaml", help="YOLO dataset YAML path")
    parser.add_argument("--config", default="config/model.yaml", help="Training config YAML path")
    parser.add_argument("--model", default=None, help="Base checkpoint, for example yolo11n.pt or yolov8n.pt")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    base_model = args.model or config.get("base_model", "yolo11n.pt")

    model = YOLO(base_model)
    model.train(
        data=args.data,
        imgsz=int(config.get("image_size", 640)),
        epochs=int(config.get("epochs", 50)),
        batch=int(config.get("batch_size", 8)),
        project=config.get("project", "runs/train"),
        name=config.get("experiment_name", "construction_defect_yolo"),
    )


if __name__ == "__main__":
    main()
