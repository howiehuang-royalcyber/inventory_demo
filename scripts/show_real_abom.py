"""Demo script: load the real ABOM, run the interpreter on both variants,
print a quick summary.

Runs without an ANTHROPIC_API_KEY — uses the heuristic Pass-2 fallback.
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.abom_loader import load_real_abom  # noqa: E402
from src.interpreter import interpret_real_abom  # noqa: E402


ABOM_PATH = os.path.join(
    ROOT, "data", "Copy of 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx"
)


def _applicable(df, col):
    return df[df[col].isin(["X", "DR", "B"])]


def main() -> None:
    df = load_real_abom(ABOM_PATH)
    print(f"Loaded {len(df)} component rows after cleaning")

    agm_rows = _applicable(df, "agm_code")
    li_rows = _applicable(df, "li_code")
    both = df[
        df["agm_code"].isin(["X", "DR", "B"]) & df["li_code"].isin(["X", "DR", "B"])
    ]
    print(f"  applicable to AGM: {len(agm_rows)}")
    print(f"  applicable to LI:  {len(li_rows)}")
    print(f"  applicable to both: {len(both)}")
    print()

    for variant in ("AGM", "LI"):
        print(f"=== Variant: {variant} ===")
        cm = interpret_real_abom(df, variant=variant, use_llm=False)
        print(f"  option groups:   {len(cm.option_groups)}")
        print(f"  required groups: {len(cm.required_groups)}")
        print(f"  constraints:     {len(cm.constraints)}")

        print("  top 5 option groups:")
        for og in cm.option_groups[:5]:
            print(
                f"    - {og.group_id}: {len(og.choices)} choices "
                f"(rows {og.provenance.abom_rows[:3]}..., e.g. {og.choices[:3]})"
            )

        print("  top 5 required groups:")
        for rg in cm.required_groups[:5]:
            print(
                f"    - {rg.group_id}: {rg.parts} (excel row {rg.provenance.abom_rows})"
            )

        print("  top 5 constraints:")
        for c in cm.constraints[:5]:
            src = (c.provenance.source_text or "")[:60]
            tail = ""
            if c.then is not None:
                tail = f" -> {c.then.group}:{c.then.choice}"
            elif c.excluded is not None:
                tail = f" XX {c.excluded.group}:{c.excluded.choice}"
            print(
                f"    - {c.type} {c.if_.group}:{c.if_.choice}{tail} "
                f"(row {c.provenance.abom_rows}, conf={c.confidence:.2f}, src={src!r})"
            )
        print()


if __name__ == "__main__":
    main()
