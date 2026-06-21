"""Run the Roboflow "Detect and Classify 2" workflow.

Grounded in the workflow's real definition (workspace: aakashs-workspace-zqqzu,
workflow_id: detect-and-classify-2):

  inputs : image (InferenceImage)
  outputs: output_image, predictions, dynamic_crop,
           detection_predictions, classification_predictions

The workflow runs an object-detection model (``training-dataset-1gvqr/2``),
dynamically crops each detection, classifies the crops, replaces the detection
classes with the classification result, and renders bounding-box + label
visualizations.

`output_image` and `dynamic_crop` come back as base64-encoded images, so we
decode them to disk instead of holding them in memory or logging them.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError

# Real output names declared by the workflow (source of truth).
WORKFLOW_OUTPUT_KEYS = (
    "output_image",
    "predictions",
    "dynamic_crop",
    "detection_predictions",
    "classification_predictions",
)

DEFAULT_WORKSPACE = "aakashs-workspace-zqqzu"
DEFAULT_WORKFLOW_ID = "detect-and-classify-2"
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.5


class RoboflowWorkflowError(RuntimeError):
    """Raised when the Roboflow workflow request fails."""


@dataclass
class WorkflowResult:
    """Parsed, payload-light view of a single workflow result entry."""

    predictions: list[dict[str, Any]] = field(default_factory=list)
    saved_images: dict[str, Path] = field(default_factory=dict)
    raw_keys: list[str] = field(default_factory=list)
    image_width: float = 0.0
    image_height: float = 0.0


def _looks_like_base64_image(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 256:
        return False
    head = value.split(",", 1)[-1][:64]
    try:
        base64.b64decode(head, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def _save_base64_image(value: str, destination: Path) -> Path:
    payload = value.split(",", 1)[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(payload))
    return destination


def _trim_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields we use; drop heavy segmentation polygons."""
    keep = ("x", "y", "width", "height", "confidence", "class", "class_id", "detection_id")
    return {key: prediction[key] for key in keep if key in prediction}


def parse_workflow_result(entry: dict[str, Any], output_dir: Path) -> WorkflowResult:
    """Parse a single workflow result entry defensively from its real keys."""
    result = WorkflowResult(raw_keys=list(entry.keys()))

    predictions_block = entry.get("predictions")
    if isinstance(predictions_block, dict):
        image_block = predictions_block.get("image")
        if isinstance(image_block, dict):
            result.image_width = float(image_block.get("width", 0.0))
            result.image_height = float(image_block.get("height", 0.0))
        predictions_block = predictions_block.get("predictions")
    if isinstance(predictions_block, list):
        result.predictions = [
            _trim_prediction(item) for item in predictions_block if isinstance(item, dict)
        ]

    for key, value in entry.items():
        if _looks_like_base64_image(value):
            destination = output_dir / f"{key}.jpg"
            result.saved_images[key] = _save_base64_image(value, destination)

    return result


def run_workflow(
    image_path: str,
    *,
    use_cache: bool = True,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> list[dict[str, Any]]:
    """Run the Detect and Classify workflow on one image.

    Returns the raw Roboflow response (a list with one entry per input image).
    Use `parse_workflow_result` to get a payload-light, typed view.
    """
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE", DEFAULT_WORKSPACE)
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID", DEFAULT_WORKFLOW_ID)
    api_url = os.getenv("ROBOFLOW_API_URL", DEFAULT_API_URL)

    if not api_key:
        raise RoboflowWorkflowError(
            "ROBOFLOW_API_KEY is missing. Create a .env file from .env.example first "
            "(find your key at app.roboflow.com/settings/api)."
        )

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": str(image_file)},
                use_cache=use_cache,
            )
        except HTTPCallErrorError as error:
            last_error = error
            if "Service misconfiguration" in str(error):
                raise RoboflowWorkflowError(
                    "Roboflow returned 'Service misconfiguration'. The workflow's classification "
                    "block points to a sample model ('car-colors-1smyc/5'). Open the workflow in "
                    "Roboflow and either remove the classification step or point it at a real "
                    "construction-defect classifier. The object-detection model "
                    "('training-dataset-1gvqr/2') works on its own."
                ) from error
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise RoboflowWorkflowError(f"Roboflow workflow request failed: {error}") from error
        except Exception as error:  # transient network/timeout errors
            last_error = error
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise RoboflowWorkflowError(f"Roboflow workflow request failed: {error}") from error

    raise RoboflowWorkflowError(f"Roboflow workflow request failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Detect and Classify Roboflow workflow on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--no-cache", action="store_true", help="Disable Roboflow workflow cache")
    parser.add_argument("--out", default="outputs/workflow", help="Directory to write decoded images to")
    args = parser.parse_args()

    try:
        raw = run_workflow(image_path=args.image, use_cache=not args.no_cache)
    except (RoboflowWorkflowError, FileNotFoundError) as error:
        print(error)
        return

    if not raw:
        print("Workflow returned an empty response.")
        return

    parsed = parse_workflow_result(raw[0], Path(args.out))
    print("Output keys:", parsed.raw_keys)
    print(f"Detections: {len(parsed.predictions)}")
    pprint(parsed.predictions)
    for key, path in parsed.saved_images.items():
        print(f"Saved image output '{key}' -> {path}")


if __name__ == "__main__":
    main()