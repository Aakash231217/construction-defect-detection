"""Direct Roboflow hosted-inference call (no inference-sdk / OpenCV dependency).

The official ``inference-sdk`` pulls in OpenCV (``cv2``), which needs a matching
wheel for the running Python version and system GL libraries — fragile on hosted
Streamlit. We only need a single HTTP POST, so we call the Roboflow hosted
inference endpoint directly with ``requests`` and return the same JSON shape
(``{"image": {...}, "predictions": [...]}``) the rest of the pipeline expects.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from pprint import pprint

import requests
from dotenv import load_dotenv


class RoboflowModelError(RuntimeError):
    pass


def run_model(image_path: str) -> dict:
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "training-dataset-1gvqr/2")
    api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").rstrip("/")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Add it to your .env file first.")

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    try:
        response = requests.post(
            f"{api_url}/{model_id}",
            params={"api_key": api_key},
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
    except requests.RequestException as error:
        raise RoboflowModelError(f"Roboflow model request failed: {error}") from error

    if response.status_code == 404:
        raise RoboflowModelError(
            f"Roboflow could not find model '{model_id}'. Check that ROBOFLOW_MODEL_ID is the "
            "actual deployed project/version slug, e.g. 'project-slug/1', and that the API key "
            "belongs to the same workspace."
        )
    if response.status_code == 401:
        raise RoboflowModelError("Roboflow rejected the API key (401). Check ROBOFLOW_API_KEY.")
    if response.status_code != 200:
        raise RoboflowModelError(
            f"Roboflow model request failed [{response.status_code}]: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as error:
        raise RoboflowModelError(
            f"Roboflow returned a non-JSON response: {response.text[:200]}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Roboflow model inference on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    try:
        result = run_model(args.image)
        pprint(result)
    except (RoboflowModelError, FileNotFoundError) as error:
        print(error)


if __name__ == "__main__":
    main()
