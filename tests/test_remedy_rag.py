from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.remedy_rag import RemedyQuery, generate_rag_remedy, retrieve_context  # noqa: E402


class _FakeMessage:
    content = "OpenAI grounded remedy generated from retrieved context."


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_rag_retrieves_defect_and_severity_specific_context() -> None:
    query = RemedyQuery(
        defect_class="crack",
        severity_level="Moderate",
        measured="width 0.25 mm (scaled)",
        remedial_measure="Route-and-seal or epoxy inject the crack; control moisture ingress.",
    )

    retrieved = retrieve_context(query, top_k=3)

    assert retrieved
    assert retrieved[0].chunk.defect == "crack"
    assert "Moderate" in retrieved[0].chunk.severity
    assert "epoxy" in retrieved[0].chunk.content.lower() or "route-and-seal" in retrieved[0].chunk.content.lower()


def test_rag_answer_includes_boq_quantities_rates_and_sources() -> None:
    query = RemedyQuery(
        defect_class="spalling",
        severity_level="Severe",
        measured="depth 50 mm + area-ratio",
        reason="Deep spall (~50 mm); significant section loss.",
        remedial_measure="Deep patch repair with steel treatment/replacement and micro-concrete/shotcrete.",
        repair_time_estimate="3-7 days",
        cost_breakup={
            "quantity": "0.03 cum",
            "quantity_description": "Area 0.50 sq m x depth 50 mm = 0.025 cum",
        },
        boq_breakup={
            "norms_found": True,
            "remedy": "Break-out and micro-concrete section reinstatement with steel treatment",
            "work_quantity": 0.5,
            "work_unit": "sq m",
            "source": "ICRI 310.1 / ACI 562 / CPWD micro-concrete norms",
            "method_steps": "Break out unsound concrete -> clean/passivate steel -> micro-concrete.",
            "lines": [
                {
                    "category": "material",
                    "item": "micro_concrete",
                    "description": "Free-flow micro-concrete",
                    "norm": 90.0,
                    "norm_unit": "kg per sq m",
                    "quantity": 45.0,
                    "quantity_unit": "kg",
                    "rate": 55.0,
                    "amount": 2475.0,
                },
                {
                    "category": "labour",
                    "item": "skilled_labour",
                    "description": "Skilled labour (mason/technician)",
                    "norm": 1.0,
                    "norm_unit": "man-day per sq m",
                    "quantity": 0.5,
                    "quantity_unit": "man-day",
                    "rate": 850.0,
                    "amount": 425.0,
                },
            ],
            "material_total": 2475.0,
            "labour_total": 425.0,
            "equipment_total": 0.0,
            "subtotal": 2900.0,
            "overheads": 435.0,
            "gst": 600.3,
            "grand_total": 3935.3,
        },
    )

    rag = generate_rag_remedy(query)

    # Answer must show norms-based BOQ with quantity x rate = amount lines
    assert "Materials required with quantities (from RAG norms)" in rag.answer
    assert "Free-flow micro-concrete" in rag.answer
    assert "45.0 kg x INR 55" in rag.answer
    assert "man-day" in rag.answer
    assert "GRAND TOTAL" in rag.answer
    assert "0.03 cum" in rag.answer
    assert "ICRI" in " ".join(rag.sources)
    # Prompt must carry the BOQ and instruct qty x rate formula
    assert "Norms-based BOQ" in rag.prompt
    assert "quantity x rate = amount" in rag.prompt
    assert "Retrieved engineering context" in rag.prompt
    assert "Severe" in rag.prompt


def test_openai_rag_uses_grounded_prompt_with_fake_client() -> None:
    client = _FakeOpenAIClient()
    query = RemedyQuery(
        defect_class="crack",
        severity_level="Moderate",
        measured="width 0.25 mm (scaled)",
        remedial_measure="Route-and-seal or epoxy inject the crack; control moisture ingress.",
        repair_time_estimate="1-2 days",
        cost_breakup={"quantity": "0.01 running metre (rmt)", "total_cost": 5.0},
    )

    rag = generate_rag_remedy(
        query,
        use_openai=True,
        openai_model="gpt-4o-mini",
        openai_client=client,
    )

    assert rag.used_llm is True
    assert rag.model == "gpt-4o-mini"
    assert rag.answer == "OpenAI grounded remedy generated from retrieved context."
    request = client.chat.completions.last_request
    assert request["model"] == "gpt-4o-mini"
    assert "Retrieved engineering context" in request["messages"][1]["content"]
    assert "Do not invent standards" in request["messages"][0]["content"]


if __name__ == "__main__":
    test_rag_retrieves_defect_and_severity_specific_context()
    test_rag_answer_includes_boq_quantities_rates_and_sources()
    test_openai_rag_uses_grounded_prompt_with_fake_client()
    print("All RAG remedy tests passed.")