"""Canonical defect vocabulary shared by every stage of the pipeline.

Detection, severity grading, cost estimation, BOQ generation and the RAG remedy
lookup all key their tables on the defect class name. Before this module each
stage carried its own alias map, so adding a class meant editing five different
normalisers and silently getting wrong rates wherever one was missed.

Everything now normalises through :func:`normalise_defect`.

Dataset naming note
-------------------
The Roboflow projects label efflorescence as ``white_bleeding`` and corrosion
staining as ``red_bleeding``. Neither matches the concrete-technology meaning of
*bleeding* (water rising to the surface of fresh concrete), so both are renamed
to the standard terms here and the dataset names survive only as aliases.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical classes
# ---------------------------------------------------------------------------
# Structural defects imply loss of section or load-path damage.
STRUCTURAL_CLASSES = frozenset({
    "crack",
    "stairstep_crack",
    "spalling",
    "honeycombing",
    "exposed_reinforcement",
})

# Durability / finish defects. These matter for service life and are worth
# reporting, but they are not loss of section, so the severity engine caps them
# below the top condition states.
NON_STRUCTURAL_CLASSES = frozenset({
    "efflorescence",
    "water_seepage",
    "mold",
    "peeling_paint",
    "rust_staining",
})

CANONICAL_CLASSES = STRUCTURAL_CLASSES | NON_STRUCTURAL_CLASSES

# ---------------------------------------------------------------------------
# Defect families used to suppress double-counting
# ---------------------------------------------------------------------------
# A single wall crack can be returned as ``crack`` by one detector and
# ``stairstep_crack`` by another. Both are real labels for the same physical
# defect, so overlapping boxes within a family must be merged rather than
# costed twice. The first entry in each tuple is the label kept on a merge: the
# more specific diagnosis wins, because it carries the better remedy.
DEFECT_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("stairstep_crack", "crack"),
    ("water_seepage", "mold", "efflorescence"),
    ("exposed_reinforcement", "rust_staining"),
)

_FAMILY_OF: dict[str, int] = {
    member: index
    for index, family in enumerate(DEFECT_FAMILIES)
    for member in family
}
_PRIORITY_IN_FAMILY: dict[str, int] = {
    member: rank
    for family in DEFECT_FAMILIES
    for rank, member in enumerate(family)
}

# ---------------------------------------------------------------------------
# Aliases: dataset / detector labels -> canonical class
# ---------------------------------------------------------------------------
DEFECT_ALIASES: dict[str, str] = {
    # spalling
    "spall": "spalling",
    "spalled_concrete": "spalling",
    "concrete_spalling": "spalling",
    # honeycombing
    "honeycomb": "honeycombing",
    "honey_combing": "honeycombing",
    # exposed reinforcement
    "exposed_rebar": "exposed_reinforcement",
    "rebar": "exposed_reinforcement",
    "reinforcement_exposed": "exposed_reinforcement",
    "exposed_steel": "exposed_reinforcement",
    # mould / damp
    "mould": "mold",
    "fungus": "mold",
    "dampness": "mold",
    "damp_patch": "mold",
    "moisture": "mold",
    # efflorescence (dataset: white_bleeding)
    "white_bleeding": "efflorescence",
    "leaching": "efflorescence",
    "lime_leaching": "efflorescence",
    "salt_deposit": "efflorescence",
    # rust staining (dataset: red_bleeding)
    "red_bleeding": "rust_staining",
    "rust_stain": "rust_staining",
    "corrosion_stain": "rust_staining",
    "rust": "rust_staining",
    # water seepage
    "seepage": "water_seepage",
    "water_leakage": "water_seepage",
    "leakage": "water_seepage",
    "water_infiltration": "water_seepage",
    # peeling paint
    "paint_peeling": "peeling_paint",
    "flaking_paint": "peeling_paint",
    "blistering_paint": "peeling_paint",
    "paint_delamination": "peeling_paint",
    # stepped masonry cracking
    "stair_step_crack": "stairstep_crack",
    "staircase_crack": "stairstep_crack",
    "step_crack": "stairstep_crack",
    "stepped_crack": "stairstep_crack",
    "diagonal_crack": "stairstep_crack",
}


def normalise_defect(value: object) -> str:
    """Map any detector/dataset label onto the canonical class name.

    Unknown labels are returned in normalised form (lowercase, underscores)
    rather than forced into a default, so downstream code can detect that it has
    no rate table for them instead of quoting the wrong repair.
    """
    key = str(value or "defect").strip().lower().replace(" ", "_").replace("-", "_")
    return DEFECT_ALIASES.get(key, key)


def is_known(defect_class: str) -> bool:
    """True when the class has severity, cost and BOQ tables behind it."""
    return normalise_defect(defect_class) in CANONICAL_CLASSES


def same_family(left: str, right: str) -> bool:
    """True when two classes describe the same physical defect.

    Used to suppress double-counting when different detectors label one defect
    differently (e.g. ``crack`` and ``stairstep_crack`` on the same wall crack).
    """
    left_key, right_key = normalise_defect(left), normalise_defect(right)
    if left_key == right_key:
        return True
    left_family = _FAMILY_OF.get(left_key)
    return left_family is not None and left_family == _FAMILY_OF.get(right_key)


def preferred_label(left: str, right: str) -> str:
    """Pick which label to keep when two family members are merged.

    The more specific diagnosis wins because it carries the better remedy:
    ``stairstep_crack`` over ``crack``, ``water_seepage`` over ``mold``.
    """
    left_key, right_key = normalise_defect(left), normalise_defect(right)
    left_rank = _PRIORITY_IN_FAMILY.get(left_key)
    right_rank = _PRIORITY_IN_FAMILY.get(right_key)
    if left_rank is None:
        return right_key if right_rank is not None else left_key
    if right_rank is None:
        return left_key
    return left_key if left_rank <= right_rank else right_key
