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


if __name__ == "__main__":
    test_crack_result_includes_remediation_cost_and_time()
    test_exposed_rebar_uses_exposed_reinforcement_repair_guidance()
    print("All severity remediation tests passed.")