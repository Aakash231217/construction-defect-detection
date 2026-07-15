"""GPT-4o vision fallback detector.

The primary detector is the Roboflow model. Its training set only covers a few
classes (crack, spalling, mould), so it returns nothing on defects such as
exposed reinforcement or efflorescence / white bleeding. For those images we run
a second-pass detection with a GPT vision model, which identifies the defect and
an approximate bounding box. Results are returned in the SAME shape as the
Roboflow predictions (centre-based x, y, width, height in pixels) so the rest of
the pipeline (severity grading, annotation, reporting) is unchanged.

Every detection produced here is clearly tagged ``source = "GPT-4o vision"`` so
the report never presents an LLM guess as if it were the trained detector.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

# Defect vocabulary the severity engine understands (plus efflorescence).
ALLOWED_CLASSES = (
    "crack",
    "spalling",
    "honeycombing",
    "exposed_reinforcement",
    "mold",
    "efflorescence",
)

_PROMPT = f"""You are a senior civil / structural inspection engineer acting as a
construction-defect detector for reinforced-concrete surfaces.

Look at the image and report ONLY clearly visible concrete defects. Use exactly
these class names: {", ".join(ALLOWED_CLASSES)}.
Guidance:
- exposed_reinforcement = visible / corroded steel bars, loss of concrete cover.
- efflorescence = white salt deposits / white bleeding / leaching stains.
- mold = dark damp / fungal patches, water staining.
- spalling = broken-away or delaminated concrete, missing chunks.
- crack = linear fracture.
- honeycombing = rough, porous concrete with exposed coarse aggregate and voids
  between the stones because the cement paste did not fill around them (poor
  compaction / segregation). Looks like clustered gravel/stones with gaps, no
  smooth paste surface. Common on formwork faces, footings and around openings.

Return STRICT JSON only, no prose:
{{"detections": [
   {{"defect_class": "<one of the allowed classes>",
     "confidence": <0.0-1.0>,
     "box": {{"x_min": <0-1>, "y_min": <0-1>, "x_max": <0-1>, "y_max": <0-1>}}}}
]}}
Rules:
- box coordinates are normalised (0..1) relative to width/height, x_max>x_min, y_max>y_min.
- Draw the box tightly around the defect region actually visible.
- If a defect type spans the whole surface (e.g. widespread efflorescence), the box may be large.
- If there is genuinely no visible defect, return {{"detections": []}}.
- Do not invent defects that are not clearly visible.
"""


STRUCTURAL_ELEMENTS = (
    "slab", "wall", "beam", "column", "staircase", "footing", "other",
)

_ELEMENT_PROMPT = f"""You are a structural engineer reviewing a site photo of a
reinforced-concrete surface. Identify the PRIMARY structural element visible.
Choose exactly one label from: {", ".join(STRUCTURAL_ELEMENTS)}.
Guidance:
- slab = horizontal floor/roof/deck soffit or top surface.
- wall = large vertical planar surface / retaining wall / shear wall.
- beam = horizontal linear member spanning between supports (deeper than wide).
- column = vertical linear supporting member (pier / pillar).
- staircase = steps / flights / stair waist slab.
- footing = foundation / pile cap / plinth at/near ground.
- other = pipe, kerb, pavement, rubble or anything not a clear member above.
Return STRICT JSON only:
{{"element": "<one label>", "confidence": <0.0-1.0>, "reason": "<short phrase>"}}
"""


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _make_client(api_key: str | None):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    from openai import OpenAI

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=resolved_key)


def classify_structural_element(
    image_path: str | Path,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Classify the primary structural element (slab/wall/beam/column/staircase...).

    Returns ``{"element", "confidence", "reason"}``. Falls back to
    ``{"element": "unknown", ...}`` on any failure.
    """
    image_path = Path(image_path)
    if client is None:
        client = _make_client(api_key)
    data_url = f"data:image/jpeg;base64,{_encode_image(image_path)}"
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You classify concrete structural "
                 "elements precisely and return strict JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": _ELEMENT_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return {"element": "unknown", "confidence": 0.0, "reason": ""}

    element = str(parsed.get("element", "unknown")).strip().lower()
    if element not in STRUCTURAL_ELEMENTS:
        element = "other" if element else "unknown"
    return {
        "element": element,
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
        "reason": str(parsed.get("reason", "")),
    }


def detect_with_gpt(
    image_path: str | Path,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    client: Any | None = None,
    max_detections: int = 6,
) -> tuple[list[dict[str, Any]], int, int]:
    """Detect defects with a GPT vision model.

    Returns ``(predictions, image_width, image_height)`` where each prediction is
    ``{"x", "y", "width", "height", "confidence", "class", "source"}`` with pixel
    coordinates (centre-based), matching the Roboflow prediction shape.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as im:
        width, height = im.size

    if client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        from openai import OpenAI

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=resolved_key)

    data_url = f"data:image/jpeg;base64,{_encode_image(image_path)}"
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a precise construction-defect vision detector. "
                "You never hallucinate defects and you return strict JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return [], width, height

    preds: list[dict[str, Any]] = []
    for item in parsed.get("detections", [])[:max_detections]:
        defect = str(item.get("defect_class", "")).strip().lower().replace(" ", "_")
        if defect not in ALLOWED_CLASSES:
            continue
        box = item.get("box", {}) or {}
        try:
            x_min = max(0.0, min(1.0, float(box["x_min"])))
            y_min = max(0.0, min(1.0, float(box["y_min"])))
            x_max = max(0.0, min(1.0, float(box["x_max"])))
            y_max = max(0.0, min(1.0, float(box["y_max"])))
        except (KeyError, TypeError, ValueError):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        bw = (x_max - x_min) * width
        bh = (y_max - y_min) * height
        cx = (x_min + x_max) / 2 * width
        cy = (y_min + y_max) / 2 * height
        preds.append({
            "x": cx, "y": cy, "width": bw, "height": bh,
            "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
            "class": defect,
            "source": "AI vision detector",
        })
    return preds, width, height
