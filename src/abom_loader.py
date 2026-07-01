"""Real ABOM loader.

Reads the customer's Excel ABOM (sheet ``Model Description``) and returns a
clean, normalized DataFrame the interpreter can chew on.

The raw sheet has:
  - a header at row index 6 (so spreadsheet row 8 is the column header row)
  - section divider rows (blank part_number, description = section heading)
  - parent VEH SKU rows for the AGM / LI variants
  - per-component rows with variant-availability codes (X / DR / D / B) and a
    free-text Rules column

This loader returns one row per real component with columns:
  cn, part_number, description, marketing_desc, agm_code, li_code,
  rules_text, section, excel_row.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


VALID_CODES = {"X", "DR", "D", "B"}

# Description-cell prefixes that mark parent VEH SKUs (skip these).
_PARENT_VEH_PREFIXES = ("VEH,",)


def _norm_code(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().upper()
    if s in VALID_CODES:
        return s
    # treat "-", blank, " ", etc. as empty
    return ""


def _norm_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _looks_like_section_header(description: Optional[str]) -> bool:
    """Section divider rows have a description but no part number.

    These also tend to be short ALL-CAPS-ish labels or end with no comma.
    The caller has already filtered on missing part_number, so any row that
    *only* has description (and an optional marketing_desc explainer) gets
    treated as a section divider.
    """
    if not description:
        return False
    # Most section dividers are short-ish and don't start with a part-like
    # token. Treat anything reaching this function (no part_number) as a
    # section header — that matches how the sheet is structured.
    return True


def load_real_abom(path: str) -> pd.DataFrame:
    """Load and clean the real Club Car ABOM Excel.

    Returns a DataFrame with one row per real component (section dividers,
    blank rows, and parent VEH SKUs removed).
    """
    raw = pd.read_excel(path, sheet_name="Model Description", header=6)

    # The first row of the dataframe is actually the explicit header labels
    # ("CN#", "Part Number", "Description", ...). Skip it.
    # Map known columns by position — the sheet uses several "Unnamed: N"
    # columns, so positional mapping is the safest.
    cols = list(raw.columns)
    # Expected positional layout (A..K):
    # 0 CN#  1 PartNumber  2 Description  3 MarketingDesc  4 unused
    # 5 AGM code  6 LI code  7 Rules  8 BC  9 DC  10 K
    df = raw.iloc[1:].copy()  # drop the embedded header echo row
    df = df.rename(
        columns={
            cols[0]: "cn",
            cols[1]: "part_number",
            cols[2]: "description",
            cols[3]: "marketing_desc",
            cols[5]: "agm_code",
            cols[6]: "li_code",
            cols[7]: "rules_text",
        }
    )

    # excel_row: header is at spreadsheet row 7 (1-indexed), so first data
    # row (raw index 0) is excel row 8. After dropping the echo row (raw
    # index 0), the first kept row (raw index 1) is excel row 9.
    df["excel_row"] = df.index.to_series().apply(lambda i: int(i) + 8)

    # Walk rows in order to track the current section header.
    out_rows: list[dict] = []
    current_section: Optional[str] = None

    for _, r in df.iterrows():
        part = _norm_str(r.get("part_number"))
        desc = _norm_str(r.get("description"))
        mkt = _norm_str(r.get("marketing_desc"))
        cn = _norm_str(r.get("cn"))
        rules = _norm_str(r.get("rules_text"))
        agm = _norm_code(r.get("agm_code"))
        li = _norm_code(r.get("li_code"))
        excel_row = int(r["excel_row"])

        # Blank row — skip but don't reset the section.
        if not part and not desc:
            continue

        # Section divider: no part_number but has a description.
        if not part and desc:
            if _looks_like_section_header(desc):
                current_section = desc
            continue

        # Parent VEH SKU rows — skip.
        if desc and any(desc.startswith(p) for p in _PARENT_VEH_PREFIXES):
            continue

        # Real component row.
        out_rows.append(
            {
                "cn": cn,
                "part_number": part,
                "description": desc or "",
                "marketing_desc": mkt,
                "agm_code": agm,
                "li_code": li,
                "rules_text": rules,
                "section": current_section,
                "excel_row": excel_row,
            }
        )

    clean = pd.DataFrame(out_rows)
    return clean


# Curated post-process splits for known interpreter mis-groupings.
# These are the kind of fix an SME would make in Stage 3 on the real product;
# we apply them up-front so the demo lands without burdening the reviewer.
# Format: { existing_group_id: { new_group_id: [parts], "_required": [parts] } }
CURATED_GROUP_FIXES = {
    "ELECTRICAL_SYSTEM": {
        # Interpreter conflated battery installs (8/16 kWh) with the LI
        # electrical install. Split into a real battery choice; promote the
        # LI electrical install to required.
        "LI_BATTERY_INSTALL": ["47787614001", "47787614002"],
        "_required": ["47787188001"],
    },
}


def _apply_curated_splits(cm):
    """Split mis-grouped option groups per CURATED_GROUP_FIXES."""
    from .models import ConfigurationModel, OptionGroup, RequiredGroup, Provenance

    new_option_groups = []
    new_required = list(cm.required_groups)

    for g in cm.option_groups:
        if g.group_id not in CURATED_GROUP_FIXES:
            new_option_groups.append(g)
            continue
        spec = CURATED_GROUP_FIXES[g.group_id]
        prov_note = f"Curated split of {g.group_id} (SME-grade fix)"
        # Build new option groups from spec entries that aren't "_required"
        for new_gid, parts in spec.items():
            if new_gid == "_required":
                continue
            present_choices = [p for p in parts if p in g.choices]
            if len(present_choices) < 2:
                continue
            new_option_groups.append(OptionGroup(
                group_id=new_gid,
                find_num=g.find_num,
                qty_per_vehicle=g.qty_per_vehicle,
                select="exactly_one",
                choices=present_choices,
                provenance=Provenance(
                    abom_rows=g.provenance.abom_rows,
                    source_text=prov_note,
                    rule="curated_split",
                ),
                confidence=g.confidence,
                sme_status="edited",
            ))
        # Promote _required parts
        for p in spec.get("_required", []):
            if p in g.choices:
                new_required.append(RequiredGroup(
                    group_id=f"REQ_{p}",
                    parts=[p],
                    qty_per_vehicle=g.qty_per_vehicle,
                    find_num=g.find_num,
                    provenance=Provenance(
                        abom_rows=g.provenance.abom_rows,
                        source_text=prov_note,
                        rule="curated_split",
                    ),
                    confidence=g.confidence,
                    sme_status="edited",
                ))

    return ConfigurationModel(
        vehicle_id=cm.vehicle_id,
        abom_version=cm.abom_version,
        required_groups=new_required,
        option_groups=new_option_groups,
        constraints=cm.constraints,
        interpreter_meta={**cm.interpreter_meta, "curated_splits_applied": True},
    )


def simplify_for_demo(cm, max_choices: int = 6):
    """Collapse oversized option groups into RequiredGroup defaults.

    The real ABOM has two mega-groups (PROGRAMMING_FEATURES with 13 choices,
    FEATURE_OPTIONS_REQUIRED with 18) that aren't meaningful for the
    buildable-units story and would blow up enumeration. For each such group
    we pick a representative choice (prefer one that is the target of a
    constraint, else the first) and demote it to a required group of qty 1.

    Returns a NEW ConfigurationModel — input cm is not mutated.
    """
    from .models import ConfigurationModel, OptionGroup, RequiredGroup, Provenance

    keep_groups: list[OptionGroup] = []
    new_required: list[RequiredGroup] = []
    demoted_choices: dict[str, str] = {}  # group_id -> chosen part

    constraint_targets = set()
    for c in cm.constraints:
        if c.then:
            constraint_targets.add((c.then.group, c.then.choice))
        if c.excluded:
            constraint_targets.add((c.excluded.group, c.excluded.choice))
        if c.one_of:
            for ch in c.one_of:
                constraint_targets.add((ch.group, ch.choice))

    for g in cm.option_groups:
        if len(g.choices) <= max_choices:
            keep_groups.append(g)
            continue
        # Pick representative: a constraint-referenced choice if any, else first
        rep = next((c for c in g.choices if (g.group_id, c) in constraint_targets), g.choices[0])
        demoted_choices[g.group_id] = rep
        new_required.append(RequiredGroup(
            group_id=f"{g.group_id}__DEFAULT",
            parts=[rep],
            qty_per_vehicle=g.qty_per_vehicle,
            find_num=g.find_num,
            provenance=Provenance(
                abom_rows=g.provenance.abom_rows,
                source_text=f"Demoted to default for demo (group had {len(g.choices)} choices)",
                rule="simplify_for_demo",
            ),
            confidence=g.confidence,
            sme_status=g.sme_status,
        ))

    # Drop constraints that referenced demoted groups (they'd be unenforceable)
    new_constraints = []
    demoted_group_ids = {g.group_id for g in cm.option_groups if g.group_id not in {kg.group_id for kg in keep_groups}}
    for c in cm.constraints:
        refs = {c.if_.group}
        if c.then:
            refs.add(c.then.group)
        if c.excluded:
            refs.add(c.excluded.group)
        if c.one_of:
            refs.update(ch.group for ch in c.one_of)
        if refs & demoted_group_ids:
            continue
        new_constraints.append(c)

    return ConfigurationModel(
        vehicle_id=cm.vehicle_id,
        abom_version=cm.abom_version,
        required_groups=cm.required_groups + new_required,
        option_groups=keep_groups,
        constraints=new_constraints,
        interpreter_meta={**cm.interpreter_meta, "simplified": True,
                          "demoted_groups": list(demoted_group_ids)},
    )


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/Copy of 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx"
    df = load_real_abom(p)
    print(f"Loaded {len(df)} component rows")
    print(df.head(10).to_string())
