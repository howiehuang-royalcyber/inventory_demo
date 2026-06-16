"""ABOM Interpreter — turns an overloaded ABOM into a structured
Configuration Model.

Two-pass design:
  Pass 1: structural extraction (required vs. option groups, via Find# collisions)
          — handled deterministically. This is mechanical and does not need an LLM.
  Pass 2: constraint extraction (free-text Notes -> formal constraints)
          — uses Claude with structured tool-call output.

If no ANTHROPIC_API_KEY is set, Pass 2 falls back to a heuristic regex-based
extractor so the demo still runs end-to-end. The seam where the real LLM
plugs in is clearly marked.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Optional

import pandas as pd

from .models import (
    ChoiceRef,
    ConfigurationModel,
    Constraint,
    OptionGroup,
    Provenance,
    RequiredGroup,
)


# ---------------------------------------------------------------------------
# Pass 1 — deterministic structural extraction
# ---------------------------------------------------------------------------

def extract_structure(abom_df: pd.DataFrame, vehicle_id: str) -> tuple[list[RequiredGroup], list[OptionGroup]]:
    """Group rows by Find# and classify each group as required or option."""
    df = abom_df.copy()
    df["_row"] = df.index + 2  # +2 to mirror Excel/CSV 1-based + header row

    required: list[RequiredGroup] = []
    options: list[OptionGroup] = []

    for find_num, group in df.groupby("find_num"):
        rows = group["_row"].tolist()
        parts = group["component"].tolist()
        qty = int(group["qty_per"].iloc[0])

        if len(parts) == 1:
            required.append(
                RequiredGroup(
                    group_id=_derive_group_id(parts[0], find_num),
                    parts=parts,
                    qty_per_vehicle=qty,
                    find_num=int(find_num),
                    provenance=Provenance(abom_rows=rows, rule="single_row_at_find"),
                    confidence=1.0,
                    sme_status="pending",
                )
            )
        else:
            options.append(
                OptionGroup(
                    group_id=_derive_option_group_id(parts, find_num),
                    find_num=int(find_num),
                    qty_per_vehicle=qty,
                    select="exactly_one",
                    choices=parts,
                    provenance=Provenance(abom_rows=rows, rule="find#_collision"),
                    confidence=0.97,
                    sme_status="pending",
                )
            )

    return required, options


def _derive_group_id(part: str, find_num: int) -> str:
    # crude but human-friendly: take the leading token
    return part.split("-")[0] if "-" in part else f"FIND_{find_num}"


def _derive_option_group_id(parts: list[str], find_num: int) -> str:
    # find common prefix; fall back to a synthetic label
    tokens = [p.split("-")[0] for p in parts]
    if len(set(tokens)) == 1:
        return tokens[0]
    # find longest common alpha prefix
    common = os.path.commonprefix(parts).rstrip("-_")
    if common:
        return common.split("-")[0]
    return f"OPT_{find_num}"


# ---------------------------------------------------------------------------
# Pass 2 — constraint extraction from free-text notes
# ---------------------------------------------------------------------------

CONSTRAINT_TOOL = {
    "name": "emit_constraints",
    "description": (
        "Emit any compatibility constraints expressed in the provided ABOM row's "
        "notes field. Return an empty list if the notes do not express a constraint."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["implies", "excludes", "requires_one_of", "forbidden_with"],
                            "description": "Type of constraint expressed in the notes",
                        },
                        "if_group": {"type": "string", "description": "The group ID of the subject part (the part this row is for)"},
                        "if_choice": {"type": "string", "description": "The component / part number this row defines"},
                        "then_group": {"type": "string", "description": "For 'implies': the dependent group ID"},
                        "then_choice": {"type": "string", "description": "For 'implies': the dependent part number"},
                        "excluded_group": {"type": "string", "description": "For 'excludes' / 'forbidden_with': the incompatible group"},
                        "excluded_choice": {"type": "string", "description": "For 'excludes' / 'forbidden_with': the incompatible part"},
                        "source_phrase": {"type": "string", "description": "The exact substring of the notes that expresses this constraint"},
                        "confidence": {"type": "number", "description": "0.0 to 1.0"},
                    },
                    "required": ["type", "if_group", "if_choice", "source_phrase", "confidence"],
                },
            }
        },
        "required": ["constraints"],
    },
}


def extract_constraints(
    abom_df: pd.DataFrame,
    option_groups: list[OptionGroup],
    required_groups: list[RequiredGroup],
) -> list[Constraint]:
    """Walk rows with non-empty notes and ask Claude to formalize any constraints."""
    df = abom_df.copy()
    df["_row"] = df.index + 2

    # Build a part -> group_id index so the LLM can be steered to use canonical group ids.
    part_to_group = {}
    for og in option_groups:
        for c in og.choices:
            part_to_group[c] = og.group_id
    for rg in required_groups:
        for p in rg.parts:
            part_to_group[p] = rg.group_id

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_real_llm = bool(api_key)

    out: list[Constraint] = []
    for _, row in df.iterrows():
        notes = (row.get("notes") or "").strip()
        if not notes:
            continue
        if use_real_llm:
            cons = _extract_with_claude(row, notes, part_to_group)
        else:
            cons = _extract_with_heuristic(row, notes, part_to_group)
        out.extend(cons)

    return out


def _extract_with_claude(row, notes: str, part_to_group: dict[str, str]) -> list[Constraint]:
    """Real LLM path — uses Anthropic API with forced tool call."""
    try:
        import anthropic
    except ImportError:
        return _extract_with_heuristic(row, notes, part_to_group)

    client = anthropic.Anthropic()
    subject_part = row["component"]
    subject_group = part_to_group.get(subject_part, "UNKNOWN")

    known_groups_str = "\n".join(
        f"  - {gid}: contains parts {sorted({p for p, g in part_to_group.items() if g == gid})}"
        for gid in sorted(set(part_to_group.values()))
    )

    system = (
        "You are an engineering BOM analyst. Read the Notes field on a single ABOM row "
        "and extract any compatibility constraints it expresses. Use the canonical group "
        "IDs provided. If the notes do not express a constraint between parts, return an "
        "empty list. Be conservative: only extract constraints that are explicit."
    )
    user = (
        f"Subject part: {subject_part} (group: {subject_group})\n"
        f"Notes: {notes!r}\n\n"
        f"Known groups and their parts:\n{known_groups_str}\n\n"
        "Extract any constraints expressed in the notes."
    )

    try:
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=1024,
            system=system,
            tools=[CONSTRAINT_TOOL],
            tool_choice={"type": "tool", "name": "emit_constraints"},
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        print(f"[interpreter] LLM call failed for row {row['_row']}: {e}; falling back to heuristic")
        return _extract_with_heuristic(row, notes, part_to_group)

    tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
    if not tool_use:
        return []
    payload = tool_use.input

    return _coerce_constraints(payload.get("constraints", []), row, subject_group, subject_part)


def _extract_with_heuristic(row, notes: str, part_to_group: dict[str, str]) -> list[Constraint]:
    """Deterministic fallback: regex over the notes column.

    Recognises three phrasings used in the demo data:
      - "req <PART>"  / "Pairs with <PART>" / "requires <PART>"   -> implies
      - "Not compatible with <PART>" / "not valid with <PART>"    -> forbidden_with
      - "clearance issue w/ <PART>"                               -> forbidden_with
    """
    subject_part = row["component"]
    subject_group = part_to_group.get(subject_part, "UNKNOWN")
    out: list[Constraint] = []

    # implies
    for m in re.finditer(r"(?:req(?:uires)?|pairs with)\s+([A-Z][A-Z0-9\-]+)", notes, re.IGNORECASE):
        target = m.group(1).upper()
        tg = part_to_group.get(target)
        if not tg:
            continue
        out.append(
            Constraint(
                type="implies",
                **{"if": ChoiceRef(group=subject_group, choice=subject_part)},
                then=ChoiceRef(group=tg, choice=target),
                provenance=Provenance(abom_rows=[int(row["_row"])], source_text=notes),
                confidence=0.85,
                sme_status="pending",
            )
        )

    # forbidden_with
    for m in re.finditer(
        r"(?:not\s+(?:valid|compatible)\s+w(?:ith|/)|clearance issue\s+w(?:ith|/))\s+([A-Z][A-Z0-9\-]+)",
        notes,
        re.IGNORECASE,
    ):
        target = m.group(1).upper()
        tg = part_to_group.get(target)
        if not tg:
            continue
        out.append(
            Constraint(
                type="forbidden_with",
                **{"if": ChoiceRef(group=subject_group, choice=subject_part)},
                excluded=ChoiceRef(group=tg, choice=target),
                provenance=Provenance(abom_rows=[int(row["_row"])], source_text=notes),
                confidence=0.80,
                sme_status="pending",
            )
        )

    return out


def _coerce_constraints(items: list[dict], row, subject_group: str, subject_part: str) -> list[Constraint]:
    out: list[Constraint] = []
    for c in items:
        kind = c.get("type")
        if not kind:
            continue
        if_ref = ChoiceRef(group=c.get("if_group", subject_group), choice=c.get("if_choice", subject_part))
        prov = Provenance(abom_rows=[int(row["_row"])], source_text=c.get("source_phrase"))
        conf = float(c.get("confidence", 0.7))
        if kind == "implies":
            tg = c.get("then_group")
            tc = c.get("then_choice")
            if not (tg and tc):
                continue
            out.append(
                Constraint(
                    type="implies",
                    **{"if": if_ref},
                    then=ChoiceRef(group=tg, choice=tc),
                    provenance=prov,
                    confidence=conf,
                )
            )
        elif kind in ("excludes", "forbidden_with"):
            eg = c.get("excluded_group")
            ec = c.get("excluded_choice")
            if not (eg and ec):
                continue
            out.append(
                Constraint(
                    type=kind,
                    **{"if": if_ref},
                    excluded=ChoiceRef(group=eg, choice=ec),
                    provenance=prov,
                    confidence=conf,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def interpret_abom(abom_df: pd.DataFrame, vehicle_id: str = "CREW-CRU") -> ConfigurationModel:
    """End-to-end ABOM -> Configuration Model."""
    required, options = extract_structure(abom_df, vehicle_id)
    constraints = extract_constraints(abom_df, options, required)

    return ConfigurationModel(
        vehicle_id=vehicle_id,
        abom_version="dummy-v1",
        required_groups=required,
        option_groups=options,
        constraints=constraints,
        interpreter_meta={
            "pass1": "deterministic structural extraction by Find#",
            "pass2": "Claude (anthropic) with tool-call output" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic regex fallback (no ANTHROPIC_API_KEY)",
        },
    )


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/crew_cru_abom.csv")
    cm = interpret_abom(df)
    print(json.dumps(cm.model_dump(by_alias=True), indent=2, default=str))
