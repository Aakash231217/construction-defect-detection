"""Retrieval-augmented remedy generation for detected defects.

The RAG flow is intentionally lightweight and dependency-free:

1. Retrieve small engineering guidance chunks from ``data/remedy_knowledge.json``.
2. Combine retrieved context with the detected defect, severity, quantity and cost data.
3. Generate a grounded remedy plan and an LLM prompt that can be sent to Mistral 7B,
   Ollama, OpenAI, or any other chat model later.

This gives the project a defensible RAG layer without requiring a live LLM key for
testing or demonstration. If an external LLM is added later, the prompt returned by
``generate_rag_remedy`` should be used as the model input, and the retrieved source
chunks should be displayed with the answer.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "remedy_knowledge.json"


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    defect: str
    severity: tuple[str, ...]
    source: str
    content: str


@dataclass(frozen=True)
class RemedyQuery:
    defect_class: str
    severity_level: str
    measured: str = ""
    reason: str = ""
    remedial_measure: str = ""
    repair_time_estimate: str = ""
    cost_breakup: dict[str, Any] | None = None
    boq_breakup: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class RagRemedy:
    answer: str
    prompt: str
    retrieved_context: tuple[RetrievedChunk, ...]
    sources: tuple[str, ...]
    used_llm: bool = False
    model: str = ""
    llm_error: str = ""


def _normalise_defect(defect_class: str) -> str:
    key = defect_class.strip().lower().replace(" ", "_").replace("-", "_")
    if key in {"exposed_rebar", "rebar", "reinforcement_exposed"}:
        return "exposed_reinforcement"
    if key in {"spall", "spalled_concrete"}:
        return "spalling"
    if key in {"mould", "dampness", "damp_patch", "moisture"}:
        return "mold"
    return key


def _severity_key(severity_level: str) -> str:
    return severity_level.strip().lower()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}


def load_knowledge_base(path: Path = KNOWLEDGE_PATH) -> list[KnowledgeChunk]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[KnowledgeChunk] = []
    for item in raw_items:
        chunks.append(
            KnowledgeChunk(
                id=str(item["id"]),
                defect=_normalise_defect(str(item["defect"])),
                severity=tuple(str(value) for value in item["severity"]),
                source=str(item["source"]),
                content=str(item["content"]),
            )
        )
    return chunks


def _query_text(query: RemedyQuery) -> str:
    cost = query.cost_breakup or {}
    return " ".join(
        [
            query.defect_class,
            query.severity_level,
            query.measured,
            query.reason,
            query.remedial_measure,
            str(cost.get("quantity", "")),
            str(cost.get("quantity_description", "")),
            str(cost.get("quantity_unit", "")),
            str(cost.get("notes", "")),
        ]
    )


def retrieve_context(
    query: RemedyQuery,
    *,
    top_k: int = 4,
    knowledge: Iterable[KnowledgeChunk] | None = None,
) -> tuple[RetrievedChunk, ...]:
    """Retrieve the most relevant chunks for a defect and severity."""
    chunks = list(knowledge if knowledge is not None else load_knowledge_base())
    defect = _normalise_defect(query.defect_class)
    severity = _severity_key(query.severity_level)
    query_tokens = _tokens(_query_text(query))

    ranked: list[RetrievedChunk] = []
    for chunk in chunks:
        chunk_tokens = _tokens(" ".join([chunk.defect, chunk.source, chunk.content, *chunk.severity]))
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / math.sqrt(max(len(chunk_tokens), 1))

        if chunk.defect == defect:
            score += 3.0
        elif chunk.defect == "general":
            score += 1.0

        if any(_severity_key(item) == severity for item in chunk.severity):
            score += 2.0

        if score > 0:
            ranked.append(RetrievedChunk(chunk=chunk, score=score))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return tuple(ranked[:top_k])


def _boq_prompt_block(boq: dict[str, Any] | None) -> str:
    """Render the norms-based BOQ as text for the LLM prompt."""
    if not boq or not boq.get("norms_found", False):
        return "No norms-based BOQ available; ask for engineer estimate."

    lines = []
    for line in boq.get("lines", []):
        lines.append(
            f"  - [{line['category']}] {line['description']}: "
            f"norm {line['norm']} {line['norm_unit']} -> "
            f"quantity {line['quantity']} {line['quantity_unit']} x "
            f"rate INR {line['rate']}/{line['quantity_unit']} = INR {line['amount']}"
        )
    return (
        f"Remedy (from norms database): {boq.get('remedy', '')}\n"
        f"Work quantity: {boq.get('work_quantity', '')} {boq.get('work_unit', '')}\n"
        f"Norms/rates source: {boq.get('source', '')}\n"
        f"Method statement: {boq.get('method_steps', '')}\n"
        f"BOQ line items (item quantity = work quantity x norm; amount = quantity x rate):\n"
        + "\n".join(lines)
        + f"\n  Material total: INR {boq.get('material_total', 0)}"
        + f"\n  Labour total: INR {boq.get('labour_total', 0)}"
        + f"\n  Equipment total: INR {boq.get('equipment_total', 0)}"
        + f"\n  Subtotal: INR {boq.get('subtotal', 0)}"
        + f"\n  Overheads & contingencies (15%): INR {boq.get('overheads', 0)}"
        + f"\n  GST (18%): INR {boq.get('gst', 0)}"
        + f"\n  GRAND TOTAL: INR {boq.get('grand_total', 0)}"
    )


def build_llm_prompt(query: RemedyQuery, retrieved: Iterable[RetrievedChunk]) -> str:
    """Build a grounded prompt suitable for Mistral 7B or another LLM."""
    cost = query.cost_breakup or {}
    context_text = "\n\n".join(
        f"[{item.chunk.id}] Source: {item.chunk.source}\n{item.chunk.content}"
        for item in retrieved
    )

    return f"""You are a civil-engineering repair assistant. Generate a practical remedy plan only from the retrieved context and the detected defect data.

Detected defect data:
- Defect: {query.defect_class}
- Severity: {query.severity_level}
- Measurement basis: {query.measured or 'Not provided'}
- Severity reason: {query.reason or 'Not provided'}
- Initial remedial measure: {query.remedial_measure or 'Not provided'}
- Repair time estimate: {query.repair_time_estimate or 'Not provided'}
- Quantity: {cost.get('quantity', 'Not calculated')}
- Quantity basis: {cost.get('quantity_description', 'Not calculated')}

Norms-based BOQ (retrieved from norms database; cost computed as quantity x rate, never retrieved directly):
{_boq_prompt_block(query.boq_breakup)}

Retrieved engineering context:
{context_text}

Write the answer with these headings:
1. Recommended remedy
2. Quantity basis
3. Materials required with quantities (from BOQ norms)
4. Labour required with man-days and rates (from BOQ norms)
5. Equipment required with usage and rates (from BOQ norms)
6. Cost estimate table (item quantity x rate = amount, with totals)
7. Execution steps / method statement
8. Site verification and limitations
9. Sources used

Rules:
- Do not invent standards, norms or rates outside the retrieved context and BOQ.
- Every cost line must show the formula: quantity x rate = amount.
- Clearly state if the quantity or cost is preliminary.
- Mention that final quantities and rates must be confirmed on site.
""".strip()


def _format_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"INR {value:,.2f}"
    return "Not calculated"


def _boq_answer_block(boq: dict[str, Any] | None) -> str:
    """Render BOQ sections for the deterministic grounded answer."""
    if not boq or not boq.get("norms_found", False):
        return (
            "Materials, labour and equipment BOQ\n"
            "- No norms record found for this defect/severity; a structural engineer "
            "must prepare the estimate."
        )

    material_lines, labour_lines, equipment_lines = [], [], []
    for line in boq.get("lines", []):
        text = (
            f"- {line['description']}: norm {line['norm']} {line['norm_unit']} -> "
            f"{line['quantity']} {line['quantity_unit']} x INR {line['rate']:.0f} "
            f"= INR {line['amount']:.2f}"
        )
        if line["category"] == "material":
            material_lines.append(text)
        elif line["category"] == "labour":
            labour_lines.append(text)
        else:
            equipment_lines.append(text)

    return f"""Materials required with quantities (from RAG norms)
{chr(10).join(material_lines)}

Labour required (from RAG norms)
{chr(10).join(labour_lines)}

Equipment required (from RAG norms)
{chr(10).join(equipment_lines)}

Cost estimate (BOQ: every amount = quantity x rate)
- Material total: INR {boq.get('material_total', 0):,.2f}
- Labour total: INR {boq.get('labour_total', 0):,.2f}
- Equipment total: INR {boq.get('equipment_total', 0):,.2f}
- Subtotal: INR {boq.get('subtotal', 0):,.2f}
- Overheads & contingencies (15%): INR {boq.get('overheads', 0):,.2f}
- GST (18%): INR {boq.get('gst', 0):,.2f}
- GRAND TOTAL: INR {boq.get('grand_total', 0):,.2f}

Method statement (from norms database)
- {boq.get('method_steps', 'Confirm method with engineer.')}
- Norms/rates source: {boq.get('source', '')}"""


def generate_grounded_answer(query: RemedyQuery, retrieved: Iterable[RetrievedChunk]) -> str:
    """Generate a deterministic, grounded remedy answer from retrieved chunks."""
    retrieved_items = tuple(retrieved)
    cost = query.cost_breakup or {}
    sources = tuple(dict.fromkeys(item.chunk.source for item in retrieved_items))
    context_points = "\n".join(f"- {item.chunk.content}" for item in retrieved_items)
    source_points = "\n".join(f"- {source}" for source in sources)

    return f"""Recommended remedy
- Defect: {query.defect_class.replace('_', ' ').title()}
- Severity: {query.severity_level}
- Recommended action: {query.remedial_measure or 'Confirm repair method after site inspection.'}

Quantity basis
- Measurement basis: {query.measured or 'Not provided'}
- Repair quantity: {cost.get('quantity', 'Not calculated')}
- Quantity explanation: {cost.get('quantity_description', 'Quantity must be confirmed on site.')}

{_boq_answer_block(query.boq_breakup)}

Execution steps
{context_points}

Site verification and limitations
- Image-based quantities are preliminary and should be checked with scale reference, crack gauge, depth measurement and site inspection.
- Final rates should be replaced with current local material, labour and equipment rates before billing.
- Severe or critical defects require structural engineer review before permanent repair.
- Repair time estimate: {query.repair_time_estimate or 'Not provided'}

Sources used
{source_points}
""".strip()


def generate_openai_answer(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    client: Any | None = None,
) -> str:
    """Generate an answer with OpenAI using the already-grounded RAG prompt.

    The OpenAI import is kept inside this function so the rest of the project can
    still run without the package installed. Tests can pass a fake ``client`` to
    validate the wiring without making a network call.
    """
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

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious civil-engineering repair assistant. "
                    "Use only the retrieved context and detected defect data. "
                    "Do not invent standards, quantities, rates, or site facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response")
    return str(content).strip()


def generate_rag_remedy(
    query: RemedyQuery,
    *,
    top_k: int = 4,
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
    openai_client: Any | None = None,
) -> RagRemedy:
    """Retrieve guidance and produce a grounded remedy answer plus LLM prompt."""
    retrieved = retrieve_context(query, top_k=top_k)
    prompt = build_llm_prompt(query, retrieved)
    sources = tuple(dict.fromkeys(item.chunk.source for item in retrieved))
    fallback_answer = generate_grounded_answer(query, retrieved)

    if use_openai:
        try:
            answer = generate_openai_answer(prompt, model=openai_model, client=openai_client)
            return RagRemedy(
                answer=answer,
                prompt=prompt,
                retrieved_context=retrieved,
                sources=sources,
                used_llm=True,
                model=openai_model,
            )
        except Exception as error:
            return RagRemedy(
                answer=fallback_answer,
                prompt=prompt,
                retrieved_context=retrieved,
                sources=sources,
                used_llm=False,
                model=openai_model,
                llm_error=str(error),
            )

    return RagRemedy(answer=fallback_answer, prompt=prompt, retrieved_context=retrieved, sources=sources)