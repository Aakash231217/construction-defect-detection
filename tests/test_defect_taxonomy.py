"""Coverage for the defect vocabulary added from the three Roboflow projects.

The datasets contribute nine class names between them. Every one of those must
survive the whole chain -- alias -> severity -> quantity -> rate -> BOQ -- because
a class that reaches the report without a norms record silently falls back to a
generic rate and produces a number nobody can defend.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defect_taxonomy import (  # noqa: E402
    CANONICAL_CLASSES,
    normalise_defect,
    preferred_label,
    same_family,
)
from src.pipeline import merge_predictions  # noqa: E402
from src.remedy_rag import RemedyQuery, retrieve_context  # noqa: E402
from src.severity import estimate_severity  # noqa: E402
from src.vision_fallback import ALLOWED_CLASSES  # noqa: E402

# The class names as they appear in the three Roboflow projects.
DATASET_CLASSES = (
    "exposed_reinforcement",
    "red_bleeding",
    "white_bleeding",
    "crack",
    "mold",
    "peeling_paint",
    "spall",
    "stairstep_crack",
    "water_seepage",
)


def _prediction(defect_class: str, x: float = 400, y: float = 300,
                confidence: float = 0.8) -> dict:
    return {
        "class": defect_class,
        "x": x,
        "y": y,
        "width": 200,
        "height": 150,
        "confidence": confidence,
    }


def test_dataset_labels_map_onto_canonical_classes() -> None:
    """The two non-standard "bleeding" labels get the engineering names."""
    assert normalise_defect("white_bleeding") == "efflorescence"
    assert normalise_defect("red_bleeding") == "rust_staining"
    assert normalise_defect("spall") == "spalling"
    for label in DATASET_CLASSES:
        assert normalise_defect(label) in CANONICAL_CLASSES, label


def test_every_dataset_class_reaches_a_norms_backed_boq() -> None:
    """No dataset class may fall through to the unrecognised-defect allowance."""
    for label in DATASET_CLASSES:
        result = estimate_severity(label, 300, 200, 1000, 800)
        assert result.boq_breakup["norms_found"], f"{label} has no norms record"
        assert result.boq_breakup["grand_total"] > 0, label
        assert "BOQ total" in result.repair_cost_estimate, label


def test_cracks_are_measured_by_length_and_areas_by_area() -> None:
    for label in ("crack", "stairstep_crack"):
        result = estimate_severity(label, 300, 200, 1000, 800)
        assert result.cost_breakup["quantity_unit"] == "running metre", label
    for label in ("spalling", "efflorescence", "peeling_paint", "water_seepage"):
        result = estimate_severity(label, 300, 200, 1000, 800)
        assert result.cost_breakup["quantity_unit"] == "sq m", label


def test_non_structural_defects_are_capped_below_critical() -> None:
    """A stain covering the whole frame is still not a structural emergency."""
    for label, cap in (("efflorescence", "Moderate"), ("peeling_paint", "Moderate"),
                       ("rust_staining", "Moderate"), ("water_seepage", "Severe")):
        result = estimate_severity(label, 950, 760, 1000, 800)  # ~90% of frame
        assert result.level == cap, f"{label} graded {result.level}, expected {cap}"
    # Structural defects keep their full range.
    assert estimate_severity("spalling", 950, 760, 1000, 800).level == "Critical"


def test_stepped_crack_carries_a_minimum_grade() -> None:
    """Stepped masonry cracking means settlement however short the crack is."""
    result = estimate_severity("stairstep_crack", 20, 15, 1000, 800)
    assert result.level == "Moderate"


def test_finish_defects_cost_far_less_than_concrete_repair() -> None:
    """Guards the regression where paint was quoted at spalling rates."""
    paint = estimate_severity("peeling_paint", 300, 200, 1000, 800)
    spalling = estimate_severity("spalling", 300, 200, 1000, 800)
    assert paint.cost_breakup["total_cost"] * 5 < spalling.cost_breakup["total_cost"]


def test_unrecognised_class_is_not_quoted_as_a_boq() -> None:
    result = estimate_severity("some_unknown_defect", 300, 200, 1000, 800)
    assert not result.boq_breakup["norms_found"]
    assert "BOQ total" not in result.repair_cost_estimate
    assert "engineer estimate required" in result.repair_cost_estimate


def test_family_members_are_recognised() -> None:
    assert same_family("crack", "stairstep_crack")
    assert same_family("mold", "water_seepage")
    assert same_family("red_bleeding", "exposed_reinforcement")
    assert not same_family("crack", "spalling")
    assert preferred_label("crack", "stairstep_crack") == "stairstep_crack"
    assert preferred_label("mold", "water_seepage") == "water_seepage"


def test_one_defect_labelled_two_ways_is_costed_once() -> None:
    """A wall crack seen as `crack` and `stairstep_crack` is a single item."""
    merged = merge_predictions(
        [_prediction("crack", confidence=0.6)],
        [_prediction("stairstep_crack", x=405, y=305, confidence=0.9)],
    )
    assert len(merged) == 1
    assert merged[0]["class"] == "stairstep_crack"  # specific diagnosis wins
    assert merged[0]["confidence"] == 0.9
    assert merged[0]["merged_from"] == ["crack"]


def test_damp_patch_seen_as_mold_and_seepage_is_costed_once() -> None:
    merged = merge_predictions(
        [_prediction("mold", confidence=0.7)],
        [_prediction("water_seepage", x=402, y=298, confidence=0.8)],
    )
    assert len(merged) == 1
    assert merged[0]["class"] == "water_seepage"


def test_different_families_overlapping_are_both_kept() -> None:
    """Spalling inside a cracked region is two real defects, not one."""
    merged = merge_predictions([_prediction("crack")], [_prediction("spalling")])
    assert len(merged) == 2


def test_distant_same_class_detections_are_both_kept() -> None:
    merged = merge_predictions(
        [_prediction("crack", x=200, y=200)],
        [_prediction("crack", x=800, y=600)],
    )
    assert len(merged) == 2


def test_vision_vocabulary_matches_the_taxonomy() -> None:
    """A class GPT can return but the tables do not know gets no rate analysis."""
    assert set(ALLOWED_CLASSES) == CANONICAL_CLASSES


def test_every_canonical_class_has_rag_knowledge() -> None:
    for defect in sorted(CANONICAL_CLASSES):
        hits = retrieve_context(
            RemedyQuery(defect_class=defect, severity_level="Moderate"), top_k=3
        )
        assert any(hit.chunk.defect == defect for hit in hits), defect


if __name__ == "__main__":
    test_dataset_labels_map_onto_canonical_classes()
    test_every_dataset_class_reaches_a_norms_backed_boq()
    test_cracks_are_measured_by_length_and_areas_by_area()
    test_non_structural_defects_are_capped_below_critical()
    test_stepped_crack_carries_a_minimum_grade()
    test_finish_defects_cost_far_less_than_concrete_repair()
    test_unrecognised_class_is_not_quoted_as_a_boq()
    test_family_members_are_recognised()
    test_one_defect_labelled_two_ways_is_costed_once()
    test_damp_patch_seen_as_mold_and_seepage_is_costed_once()
    test_different_families_overlapping_are_both_kept()
    test_distant_same_class_detections_are_both_kept()
    test_vision_vocabulary_matches_the_taxonomy()
    test_every_canonical_class_has_rag_knowledge()
    print("All defect taxonomy, severity cap, cost and dedup tests passed.")
