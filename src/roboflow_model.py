from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from inference_sdk.http.errors import HTTPCallErrorError
from inference_sdk import InferenceHTTPClient


class RoboflowModelError(RuntimeError):
    pass


def run_model(image_path: str) -> object:
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "training-dataset-1gqvr/2")
    api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Add it to your .env file first.")

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    try:
        return client.infer(str(image_file), model_id=model_id)
    except HTTPCallErrorError as error:
        message = str(error)
        if "404" in message or "resource not found" in message.lower():
            raise RoboflowModelError(
                f"Roboflow could not find model '{model_id}'. Check that ROBOFLOW_MODEL_ID is the actual deployed project/version slug, "
                "for example 'project-slug/1', and that the API key belongs to the same workspace."
            ) from error
        raise RoboflowModelError(f"Roboflow model request failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Roboflow model inference on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    try:
        result = run_model(args.image)
        pprint(result)
    except RoboflowModelError as error:
        print(error)


if __name__ == "__main__":
    main()