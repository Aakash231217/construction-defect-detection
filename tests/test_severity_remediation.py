from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.severity import estimate_severity  # noqa: E402


def test_crack_result_includes_remediation_cost_and_time() -> None:
    result = estimate_severity(
        defect_class="crack",
        box_width=20,
        box_height=4,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.05,
    )

    assert result.level == "Moderate"
    assert "epoxy" in result.remedial_measure.lower() or "seal" in result.remedial_measure.lower()
    assert "INR" in result.repair_cost_estimate
    assert result.repair_time_estimate


def test_exposed_rebar_uses_exposed_reinforcement_repair_guidance() -> None:
    result = estimate_severity(
        defect_class="exposed_rebar",
        box_width=150,
        box_height=80,
        image_width=1000,
        image_height=800,
    )

    assert result.level == "Severe"
    assert "steel" in result.remedial_measure.lower()
    assert "sq m" in result.repair_cost_estimate


def test_crack_cost_estimate_has_quantity_rate_and_cost_breakup() -> None:
    """Cost = Quantity x Rate, with material/labour/equipment breakup."""
    result = estimate_severity(
        defect_class="crack",
        box_width=200,
        box_height=5,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.05,
    )

    # Severity should be Moderate (width = 5px * 0.05 = 0.25 mm)
    assert result.level == "Moderate"

    cost = result.cost_estimate
    assert cost is not None, "Cost estimate must be computed"

    # Quantity must be in running metres (crack length)
    assert "running metre" in cost.quantity.unit.value
    assert cost.quantity.value > 0, "Crack length quantity must be positive"

    # Rate breakup must have material, labour, equipment
    assert cost.rate.material_rate > 0
    assert cost.rate.labour_rate > 0
    assert cost.rate.equipment_rate > 0

    # Cost = Quantity x Composite rate
    expected_total = cost.quantity.value * cost.rate.composite_rate
    assert abs(cost.total_cost - expected_total) < 0.01

    # Individual cost breakups must sum to total
    breakup_sum = cost.material_cost + cost.labour_cost + cost.equipment_cost
    assert abs(breakup_sum - cost.total_cost) < 0.01

    # cost_breakup dict must be populated
    assert result.cost_breakup["quantity_value"] > 0
    assert result.cost_breakup["total_cost"] > 0


def test_spalling_cost_estimate_uses_area_quantity() -> None:
    """Spalling quantity should be in sq m (or cum if depth known)."""
    result = estimate_severity(
        defect_class="spalling",
        box_width=300,
        box_height=200,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.05,
    )

    cost = result.cost_estimate
    assert cost is not None
    assert cost.quantity.value > 0
    # Without depth, unit should be sq m
    assert "sq m" in cost.quantity.unit.value

    # Cost breakup must be valid
    assert cost.total_cost > 0
    assert cost.material_cost > 0
    assert cost.labour_cost > 0
    assert cost.equipment_cost > 0


def test_spalling_with_depth_uses_volume_quantity() -> None:
    """When depth is provided for severe spalling, quantity should be in cum."""
    result = estimate_severity(
        defect_class="spalling",
        box_width=300,
        box_height=200,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.05,
        depth_mm=50.0,
    )

    cost = result.cost_estimate
    assert cost is not None
    # With depth and severe grade, unit should be cum (volume)
    assert "cum" in cost.quantity.unit.value or "sq m" in cost.quantity.unit.value
    assert cost.total_cost > 0


def test_boq_lines_use_norms_and_qty_x_rate() -> None:
    """BOQ items must come from RAG norms, and every amount = quantity x rate."""
    result = estimate_severity(
        defect_class="crack",
        box_width=200,
        box_height=5,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.05,
    )

    boq = result.boq
    assert boq is not None and boq.norms_found, "Norms record must be retrieved"
    assert boq.work_unit == "running metre"
    assert boq.lines, "BOQ must contain line items"

    categories = {line.category for line in boq.lines}
    assert {"material", "labour", "equipment"} <= categories

    for line in boq.lines:
        # item quantity = work quantity x norm
        assert abs(line.quantity - boq.work_quantity * line.norm) < 1e-6
        # amount = quantity x rate (cost never retrieved directly)
        assert abs(line.amount - line.quantity * line.rate) < 1e-6

    material_sum = sum(l.amount for l in boq.lines if l.category == "material")
    labour_sum = sum(l.amount for l in boq.lines if l.category == "labour")
    equipment_sum = sum(l.amount for l in boq.lines if l.category == "equipment")
    assert abs(boq.material_total - material_sum) < 1e-6
    assert abs(boq.labour_total - labour_sum) < 1e-6
    assert abs(boq.equipment_total - equipment_sum) < 1e-6

    subtotal = material_sum + labour_sum + equipment_sum
    assert abs(boq.subtotal - subtotal) < 1e-6
    assert abs(boq.overheads - subtotal * 0.15) < 1e-6
    assert abs(boq.gst - (subtotal + boq.overheads) * 0.18) < 1e-6
    assert abs(boq.grand_total - (subtotal + boq.overheads + boq.gst)) < 1e-6

    # User-facing estimate must match the RAG remedy's BOQ grand total.
    assert abs(result.cost_breakup["total_cost"] - boq.grand_total) < 0.01
    assert abs(
        result.cost_breakup["composite_rate"]
        - boq.grand_total / boq.work_quantity
    ) < 0.01

    # Moderate crack must include epoxy injection material and labour man-days
    descriptions = " ".join(line.description.lower() for line in boq.lines)
    assert "epoxy" in descriptions
    assert any(line.quantity_unit == "man-day" for line in boq.lines)


def test_boq_spalling_uses_sq_m_norms_with_mortar_kg() -> None:
    """Spalling BOQ should retrieve mortar consumption in kg per sq m."""
    result = estimate_severity(
        defect_class="spalling",
        box_width=300,
        box_height=200,
        image_width=1000,
        image_height=800,
        mm_per_pixel=0.5,
    )

    boq = result.boq
    assert boq is not None and boq.norms_found
    assert boq.work_unit == "sq m"
    material_units = {l.quantity_unit for l in boq.lines if l.category == "material"}
    assert "kg" in material_units, "Mortar/micro-concrete must be quantified in kg"


if __name__ == "__main__":
    test_crack_result_includes_remediation_cost_and_time()
    test_exposed_rebar_uses_exposed_reinforcement_repair_guidance()
    test_crack_cost_estimate_has_quantity_rate_and_cost_breakup()
    test_spalling_cost_estimate_uses_area_quantity()
    test_spalling_with_depth_uses_volume_quantity()
    test_boq_lines_use_norms_and_qty_x_rate()
    test_boq_spalling_uses_sq_m_norms_with_mortar_kg()
    print("All severity remediation, cost estimation and BOQ tests passed.")