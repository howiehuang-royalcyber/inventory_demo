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

try:  # Literal is used in annotations for the real-ABOM helpers below.
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore


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


# ---------------------------------------------------------------------------
# Real-ABOM path — operates on the cleaned DataFrame from src/abom_loader.py
# ---------------------------------------------------------------------------

# Rules-text phrases that do NOT express an inter-part constraint and should
# be skipped before paying for an LLM round-trip.
_NON_CONSTRAINT_RULES = {
    "default required",
    "default required.",
    "default required for usa & canada.",
    "default required in usa",
    "at least one must be chosen for every vehicle",
    "only one can be chosen",
    "dr = default required",
    "standard offering",
    "see column k",
    "tied to feature option",
}


def _variant_code(row, variant: str) -> str:
    return (row.get("agm_code") if variant == "AGM" else row.get("li_code")) or ""


def _group_id_from_section(section: str | None, fallback: str) -> str:
    if not section:
        return fallback
    # Normalize: take first 3 words, uppercase, alnum + underscore.
    toks = re.findall(r"[A-Za-z0-9]+", section)[:4]
    if not toks:
        return fallback
    return "_".join(t.upper() for t in toks)


def _split_into_groups(rows: list[dict], variant: str) -> list[list[dict]]:
    """Group consecutive applicable rows by their ``section`` field.

    Within the real ABOM the section heading is the natural option-group
    boundary (e.g. "STEERING WHEEL", "MIRROR COLOR", "CHARGER SYSTEM").
    Rows without a section fall into their own singleton group, keyed by
    description prefix.
    """
    by_section: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get("section") or f"_NOSEC_{r['part_number']}"
        by_section[key].append(r)
    return list(by_section.values())


def _build_pass1_real(
    df: pd.DataFrame, variant: str
) -> tuple[list[RequiredGroup], list[OptionGroup], dict[str, str]]:
    """Pass 1 (heuristic) for the real ABOM.

    Returns (required_groups, option_groups, part_to_group_id).
    """
    # Filter rows applicable to this variant: code in {X, DR, B}.
    applicable_codes = {"X", "DR", "B"}
    rows: list[dict] = []
    for _, r in df.iterrows():
        code = _variant_code(r, variant)
        if code in applicable_codes:
            rows.append(r.to_dict())

    groups = _split_into_groups(rows, variant)

    required: list[RequiredGroup] = []
    options: list[OptionGroup] = []
    part_to_group: dict[str, str] = {}

    # Counter to ensure group_id uniqueness when sections repeat / collide.
    seen_ids: dict[str, int] = defaultdict(int)

    def _unique(gid: str) -> str:
        seen_ids[gid] += 1
        return gid if seen_ids[gid] == 1 else f"{gid}_{seen_ids[gid]}"

    for grp in groups:
        parts = [g["part_number"] for g in grp]
        excel_rows = [int(g["excel_row"]) for g in grp]
        codes = [_variant_code(g, variant) for g in grp]
        section = grp[0].get("section")
        base_gid = _group_id_from_section(section, fallback=f"GRP_{excel_rows[0]}")

        has_x = any(c == "X" for c in codes)
        has_dr = any(c == "DR" for c in codes)

        if len(parts) >= 2 and has_x:
            # Multi-choice option group. DR (if present) is the default.
            gid = _unique(base_gid)
            options.append(
                OptionGroup(
                    group_id=gid,
                    find_num=excel_rows[0],  # excel row used as a "find" handle
                    qty_per_vehicle=1,
                    select="exactly_one",
                    choices=parts,
                    provenance=Provenance(
                        abom_rows=excel_rows,
                        rule="section_group+variant_code",
                        source_text=section,
                    ),
                    confidence=0.9 if has_dr else 0.75,
                    sme_status="pending",
                )
            )
            for p in parts:
                part_to_group[p] = gid
        else:
            # Required group (singleton or all-DR section).
            for g, p, er, code in zip(grp, parts, excel_rows, codes):
                gid = _unique(_group_id_from_section(section, fallback=p.split(",")[0]))
                required.append(
                    RequiredGroup(
                        group_id=gid,
                        parts=[p],
                        qty_per_vehicle=1,
                        find_num=er,
                        provenance=Provenance(
                            abom_rows=[er],
                            rule="single_row_default_required" if code == "DR" else "single_row",
                            source_text=section,
                        ),
                        confidence=1.0 if code == "DR" else 0.9,
                        sme_status="pending",
                    )
                )
                part_to_group[p] = gid

    return required, options, part_to_group


def _extract_constraints_real(
    df: pd.DataFrame,
    variant: str,
    option_groups: list[OptionGroup],
    required_groups: list[RequiredGroup],
    part_to_group: dict[str, str],
    use_llm: bool,
) -> list[Constraint]:
    """Pass 2 for the real ABOM. Walks rules_text and extracts constraints."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_real_llm = bool(api_key) and use_llm

    constraints: list[Constraint] = []
    for _, r in df.iterrows():
        if _variant_code(r, variant) not in {"X", "DR", "B"}:
            continue
        rules = r.get("rules_text")
        if not rules:
            continue
        norm = str(rules).strip().lower().rstrip(".")
        if norm in _NON_CONSTRAINT_RULES:
            continue
        # Build a shim row dict so the existing extractors keep working.
        shim = {
            "component": r["part_number"],
            "_row": int(r["excel_row"]),
            "notes": rules,
        }
        # Convert to a pandas Series so .get works the same way.
        shim_row = pd.Series(shim)
        if use_real_llm:
            cons = _extract_with_claude(shim_row, str(rules), part_to_group)
        else:
            cons = _extract_real_heuristic(shim_row, str(rules), part_to_group)
        constraints.extend(cons)
    return constraints


# Pattern: a 8+ digit Club Car part number embedded in free text.
_PART_NUM_RE = re.compile(r"\b(\d{8,})\b")


def _extract_real_heuristic(row, rules: str, part_to_group: dict[str, str]) -> list[Constraint]:
    """Heuristic constraint extractor tuned for the real ABOM's phrasings.

    Recognises:
      - "Required when <PART> is selected" / "Must be selected when <PART>" -> implies
      - "Only <PART> can be used for this color" -> implies
      - "<X> and <Y> require each other" -> implies (both directions)
      - "not compatible with <PART>" / "not valid with <PART>" -> forbidden_with
    """
    subject_part = row["component"]
    subject_group = part_to_group.get(subject_part, "UNKNOWN")
    out: list[Constraint] = []

    def _ref(part: str) -> ChoiceRef | None:
        g = part_to_group.get(part)
        if not g:
            return None
        return ChoiceRef(group=g, choice=part)

    lower = rules.lower()
    parts_in_text = _PART_NUM_RE.findall(rules)

    # implies: "required when X is selected" / "must be selected when X"
    for trigger in (
        r"required when\s+(\d{8,})",
        r"must be selected when\s+(\d{8,})",
        r"only\s+(?:table\s+)?(\d{8,})\s+can be used",
        r"required when\s+(\d{8,})\s+is selected",
    ):
        for m in re.finditer(trigger, lower):
            target = m.group(1)
            target_ref = _ref(target)
            if not target_ref:
                continue
            out.append(
                Constraint(
                    type="implies",
                    **{"if": target_ref},  # if target chosen, then this part required
                    then=ChoiceRef(group=subject_group, choice=subject_part),
                    provenance=Provenance(
                        abom_rows=[int(row["_row"])],
                        source_text=rules,
                        rule="real_abom_heuristic",
                    ),
                    confidence=0.75,
                    sme_status="pending",
                )
            )

    # "X and Y require each other" — bidirectional implies.
    m = re.search(r"\((\d{8,})\)\s+and\s+\w+\s*\((\d{8,})\)\s+require each other", lower)
    if m:
        a, b = m.group(1), m.group(2)
        ra, rb = _ref(a), _ref(b)
        if ra and rb:
            out.append(
                Constraint(
                    type="implies",
                    **{"if": ra},
                    then=rb,
                    provenance=Provenance(abom_rows=[int(row["_row"])], source_text=rules, rule="bidir_pair"),
                    confidence=0.8,
                )
            )
            out.append(
                Constraint(
                    type="implies",
                    **{"if": rb},
                    then=ra,
                    provenance=Provenance(abom_rows=[int(row["_row"])], source_text=rules, rule="bidir_pair"),
                    confidence=0.8,
                )
            )

    # forbidden_with
    for m in re.finditer(
        r"not\s+(?:compatible|valid)\s+w(?:ith|/)\s+(\d{8,})", lower
    ):
        target_ref = _ref(m.group(1))
        if not target_ref:
            continue
        out.append(
            Constraint(
                type="forbidden_with",
                **{"if": ChoiceRef(group=subject_group, choice=subject_part)},
                excluded=target_ref,
                provenance=Provenance(
                    abom_rows=[int(row["_row"])], source_text=rules, rule="real_abom_heuristic"
                ),
                confidence=0.75,
            )
        )

    # "Mirror to match Exterior Body color" — cross-group exclusivity.
    # Emit as a soft hint (a placeholder constraint with low confidence)
    # so an SME can finalize the pairing in the review stage.
    if "mirror to match exterior body" in lower:
        out.append(
            Constraint(
                type="requires_one_of",
                **{"if": ChoiceRef(group=subject_group, choice=subject_part)},
                one_of=[],
                provenance=Provenance(
                    abom_rows=[int(row["_row"])],
                    source_text=rules,
                    rule="mirror_matches_body_color_hint",
                ),
                confidence=0.4,
                sme_status="pending",
            )
        )

    return out


def interpret_real_abom(
    df: pd.DataFrame,
    variant: str,
    use_llm: bool = True,
) -> ConfigurationModel:
    """Build a Configuration Model from the cleaned real-ABOM DataFrame.

    Args:
        df: Output of :func:`src.abom_loader.load_real_abom`.
        variant: ``"AGM"`` or ``"LI"``.
        use_llm: If False, force the heuristic Pass-2 fallback even when
            ``ANTHROPIC_API_KEY`` is set. Used for the demo script.
    """
    if variant not in ("AGM", "LI"):
        raise ValueError(f"variant must be 'AGM' or 'LI', got {variant!r}")

    required, options, part_to_group = _build_pass1_real(df, variant)
    constraints = _extract_constraints_real(
        df, variant, options, required, part_to_group, use_llm=use_llm
    )

    vehicle_id = "47773464001-CRU-LOUNGE-AGM" if variant == "AGM" else "47787623001-CRU-LOUNGE-LI"

    return ConfigurationModel(
        vehicle_id=vehicle_id,
        abom_version="real-abom-v1",
        required_groups=required,
        option_groups=options,
        constraints=constraints,
        interpreter_meta={
            "variant": variant,
            "pass1": "section + variant-code heuristic",
            "pass2": (
                "Claude (anthropic) with tool-call output"
                if (os.environ.get("ANTHROPIC_API_KEY") and use_llm)
                else "heuristic regex fallback"
            ),
            "source": "real ABOM (Excel)",
        },
    )


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/crew_cru_abom.csv")
    cm = interpret_abom(df)
    print(json.dumps(cm.model_dump(by_alias=True), indent=2, default=str))
