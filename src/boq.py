"""BOQ (Bill of Quantities) generation from RAG-retrieved norms and rates.

This module implements the professor-required flow:

    1. Work quantity comes from detection/measurement (running metre, sq m, cum).
    2. NORMS (consumption per unit of work) are retrieved from the RAG norms
       database ``data/repair_norms.json``:
         - material norms  (e.g. epoxy 0.35 litre per running metre)
         - labour norms    (e.g. skilled 0.60 man-day per running metre)
         - equipment norms (e.g. injection pump 0.20 day per running metre)
    3. UNIT RATES are also retrieved from the same database
       (e.g. skilled labour INR 850 per man-day, epoxy INR 1100 per litre).
    4. Cost is NEVER retrieved directly.  Every line item is computed:

           item quantity = work quantity x norm
           item cost     = item quantity x unit rate

       and totals are summed to build the BOQ.

The database values follow CPWD Manual Vol.4-style norms and manufacturer
datasheet consumption figures; they must be updated with current local rates
before contractual use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NORMS_PATH = Path(__file__).resolve().parents[1] / "data" / "repair_norms.json"


@dataclass(frozen=True)
class BoqLine:
    """One BOQ line item: quantity derived from norm, cost = quantity x rate."""

    category: str          # material | labour | equipment
    item: str              # e.g. epoxy_injection_low_visc
    description: str       # human-readable name
    norm: float            # consumption per unit of work
    norm_unit: str         # e.g. "litre per running metre"
    quantity: float        # work_quantity x norm
    quantity_unit: str     # litre / kg / man-day / day
    rate: float            # INR per quantity_unit (from RAG rate table)
    amount: float          # quantity x rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "item": self.item,
            "description": self.description,
            "norm": self.norm,
            "norm_unit": self.norm_unit,
            "quantity": round(self.quantity, 3),
            "quantity_unit": self.quantity_unit,
            "rate": self.rate,
            "amount": round(self.amount, 2),
        }


@dataclass(frozen=True)
class BoqEstimate:
    """Full BOQ for one repair item."""

    remedy: str
    work_unit: str
    work_quantity: float
    source: str
    method_steps: str
    lines: tuple[BoqLine, ...]
    material_total: float
    labour_total: float
    equipment_total: float
    subtotal: float
    overheads: float          # overheads & contingencies (approx 15%)
    gst: float                # GST @ 18%
    grand_total: float
    norms_found: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "remedy": self.remedy,
            "work_unit": self.work_unit,
            "work_quantity": round(self.work_quantity, 3),
            "source": self.source,
            "method_steps": self.method_steps,
            "lines": [line.to_dict() for line in self.lines],
            "material_total": round(self.material_total, 2),
            "labour_total": round(self.labour_total, 2),
            "equipment_total": round(self.equipment_total, 2),
            "subtotal": round(self.subtotal, 2),
            "overheads": round(self.overheads, 2),
            "gst": round(self.gst, 2),
            "grand_total": round(self.grand_total, 2),
            "norms_found": self.norms_found,
        }

    def formatted_table(self) -> str:
        """Plain-text BOQ table for reports/LLM prompts."""
        header = (
            f"Remedy: {self.remedy}\n"
            f"Work quantity: {self.work_quantity:.2f} {self.work_unit}\n"
            f"Norms/rates source: {self.source}\n\n"
            f"{'Category':<10} {'Item':<42} {'Qty':>9} {'Unit':<9} {'Rate':>9} {'Amount':>10}\n"
            + "-" * 95
        )
        rows = [
            f"{line.category:<10} {line.description:<42} {line.quantity:>9.2f} "
            f"{line.quantity_unit:<9} {line.rate:>9.0f} {line.amount:>10.2f}"
            for line in self.lines
        ]
        totals = (
            "-" * 95 + "\n"
            f"{'Material total':<73}{self.material_total:>20.2f}\n"
            f"{'Labour total':<73}{self.labour_total:>20.2f}\n"
            f"{'Equipment total':<73}{self.equipment_total:>20.2f}\n"
            f"{'Subtotal':<73}{self.subtotal:>20.2f}\n"
            f"{'Overheads & contingencies (15%)':<73}{self.overheads:>20.2f}\n"
            f"{'GST (18%)':<73}{self.gst:>20.2f}\n"
            f"{'GRAND TOTAL (INR)':<73}{self.grand_total:>20.2f}"
        )
        return "\n".join([header, *rows, totals])


def _normalise_defect(defect_class: str) -> str:
    key = defect_class.strip().lower().replace(" ", "_").replace("-", "_")
    if key in {"exposed_rebar", "rebar", "reinforcement_exposed"}:
        return "exposed_reinforcement"
    if key in {"spall", "spalled_concrete"}:
        return "spalling"
    if key in {"mould", "dampness", "damp_patch", "moisture"}:
        return "mold"
    return key


def load_norms_database(path: Path = NORMS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate_lookup(database: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten the unit_rates section into {item: {rate, unit, description}}."""
    lookup: dict[str, dict[str, Any]] = {}
    for category in ("labour", "materials", "equipment"):
        for entry in database.get("unit_rates", {}).get(category, []):
            lookup[entry["item"]] = entry
    return lookup


def retrieve_norms(
    defect_class: str,
    severity_level: str,
    database: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Retrieve the norms record for a defect + severity from the RAG database."""
    db = database if database is not None else load_norms_database()
    defect = _normalise_defect(defect_class)
    severity = severity_level.strip().title()
    for record in db.get("repair_norms", []):
        if _normalise_defect(record["defect"]) == defect and severity in record["severity"]:
            return record
    return None


def _quantity_unit_from_norm(norm_unit: str) -> str:
    """'litre per running metre' -> 'litre'; 'man-day per sq m' -> 'man-day'."""
    return norm_unit.split(" per ")[0].strip()


def compute_boq(
    defect_class: str,
    severity_level: str,
    work_quantity: float,
    *,
    database: dict[str, Any] | None = None,
    overhead_fraction: float = 0.15,
    gst_fraction: float = 0.18,
) -> BoqEstimate:
    """Build the BOQ: retrieve norms + rates from RAG, compute cost = qty x rate.

    Parameters
    ----------
    defect_class
        Detected defect (crack, spalling, honeycombing, exposed_rebar...).
    severity_level
        Severity label (Minor, Moderate, Severe, Critical).
    work_quantity
        Measured repair quantity in the record's work unit
        (running metre for cracks, sq m for area repairs).
    """
    db = database if database is not None else load_norms_database()
    record = retrieve_norms(defect_class, severity_level, db)
    rates = _rate_lookup(db)

    if record is None:
        return BoqEstimate(
            remedy="No norms record found; engineer estimate required",
            work_unit="",
            work_quantity=work_quantity,
            source="",
            method_steps="",
            lines=(),
            material_total=0.0,
            labour_total=0.0,
            equipment_total=0.0,
            subtotal=0.0,
            overheads=0.0,
            gst=0.0,
            grand_total=0.0,
            norms_found=False,
        )

    lines: list[BoqLine] = []
    category_totals = {"material": 0.0, "labour": 0.0, "equipment": 0.0}

    for category, key in (("material", "materials"), ("labour", "labour"), ("equipment", "equipment")):
        for norm_entry in record.get(key, []):
            item = norm_entry["item"]
            rate_entry = rates.get(item, {})
            rate = float(rate_entry.get("rate", 0.0))
            description = str(rate_entry.get("description", item))
            quantity = work_quantity * float(norm_entry["norm"])
            amount = quantity * rate
            category_totals[category] += amount
            lines.append(
                BoqLine(
                    category=category,
                    item=item,
                    description=description,
                    norm=float(norm_entry["norm"]),
                    norm_unit=str(norm_entry["norm_unit"]),
                    quantity=quantity,
                    quantity_unit=_quantity_unit_from_norm(str(norm_entry["norm_unit"])),
                    rate=rate,
                    amount=amount,
                )
            )

    subtotal = sum(category_totals.values())
    overheads = subtotal * overhead_fraction
    gst = (subtotal + overheads) * gst_fraction
    grand_total = subtotal + overheads + gst

    return BoqEstimate(
        remedy=str(record["remedy"]),
        work_unit=str(record["work_unit"]),
        work_quantity=work_quantity,
        source=str(record["source"]),
        method_steps=str(record.get("method_steps", "")),
        lines=tuple(lines),
        material_total=category_totals["material"],
        labour_total=category_totals["labour"],
        equipment_total=category_totals["equipment"],
        subtotal=subtotal,
        overheads=overheads,
        gst=gst,
        grand_total=grand_total,
    )
