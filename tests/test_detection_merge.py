from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import merge_predictions, normalise_detection_class  # noqa: E402


def _prediction(defect_class: str, x: float = 100, y: float = 100) -> dict:
    return {
        "class": defect_class,
        "x": x,
        "y": y,
        "width": 80,
        "height": 80,
        "confidence": 0.8,
    }


def test_secondary_adds_honeycombing_missed_by_primary() -> None:
    merged = merge_predictions([_prediction("crack")], [_prediction("honeycombing")])
    assert {item["class"] for item in merged} == {"crack", "honeycombing"}


def test_same_class_overlapping_detection_is_deduplicated() -> None:
    merged = merge_predictions([_prediction("honeycombing")], [_prediction("honeycombing")])
    assert len(merged) == 1


def test_non_overlapping_same_class_detection_is_retained() -> None:
    merged = merge_predictions(
        [_prediction("honeycombing", x=100)],
        [_prediction("honeycombing", x=300)],
    )
    assert len(merged) == 2


def test_aliases_are_canonicalized_before_deduplication() -> None:
    merged = merge_predictions([_prediction("spall")], [_prediction("spalling")])
    assert len(merged) == 1
    assert merged[0]["class"] == "spalling"


def test_supported_aliases_map_to_engineering_classes() -> None:
    assert normalise_detection_class("honeycomb") == "honeycombing"
    assert normalise_detection_class("exposed rebar") == "exposed_reinforcement"
    assert normalise_detection_class("white bleeding") == "efflorescence"


if __name__ == "__main__":
    test_secondary_adds_honeycombing_missed_by_primary()
    test_same_class_overlapping_detection_is_deduplicated()
    test_non_overlapping_same_class_detection_is_retained()
    test_aliases_are_canonicalized_before_deduplication()
    test_supported_aliases_map_to_engineering_classes()
    print("All primary/secondary detection merge tests passed.")