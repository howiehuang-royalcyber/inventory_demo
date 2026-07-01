"""Generate synthetic inventory for the Club Car POC demo.

Reads the real ABOM, pulls real unit costs from the customer's cost sheet,
and assigns on-hand quantities engineered to produce a believable demo story:

  - Default stock: ~target_vehicles * 1.1 of each part (loose slack).
  - PLANTED TIGHT: 16 kWh lithium battery (47787614002) — binds at higher LI mix.
  - PLANTED TIGHT: one mirror color (47765017001 black) — sets up cross-group
    exclusivity with body color.
  - PLANTED EXCESS + AGED: AGM-only batteries (47782768001/002) — slow movers
    as the business shifts to lithium.
  - Cost fallback: parts not in customer cost file get a description-based
    heuristic price (so the dollar math is non-zero everywhere).

Outputs data/inventory_synthetic.csv with columns:
  part_number, description, on_hand, unit_cost, aging_days, warehouse
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.abom_loader import load_real_abom


# --- Demo tuning knobs ----------------------------------------------------

TARGET_VEHICLES = 80          # baseline "this month" target across both SKUs
TARGET_LI_SHARE = 0.65        # matches the customer's 2025-26 trend
DEFAULT_SLACK = 1.10          # most parts have 10% slack above target

# Planted constraints — these drive the demo storyline.
PLANTED: dict[str, dict] = {
    "47787614002": {  # LI 16 kWh battery install — the binding hero
        "on_hand": 30,
        "warehouse": "P",
        "note": "16 kWh batteries — bottleneck of the LI shift",
    },
    "47787614001": {  # LI 8 kWh battery install — also tight (drives 16kWh use)
        "on_hand": 25,
        "warehouse": "P",
        "note": "8 kWh batteries — short stock forces 16 kWh demand",
    },
    "47765017001": {  # Mirror sapphire — cross-group exclusivity demo target
        "qty_factor": 0.45,
        "warehouse": "A",
        "note": "Sapphire mirrors — short; demonstrates body-color exclusivity",
    },
    "47782768001": {  # AGM battery (default) — slow mover
        "qty_factor": 2.50,
        "aging_days": 220,
        "warehouse": "A",
        "note": "AGM battery — slow mover, aging stock",
    },
    "47782768002": {  # AGM battery (alt)
        "qty_factor": 2.20,
        "aging_days": 210,
        "warehouse": "A",
        "note": "AGM battery alt — aging",
    },
}


def _cost_heuristic(part: str, desc: str) -> float:
    """Fallback cost when a part isn't in the customer cost sheet."""
    d = (desc or "").upper()
    if "BATTERY" in d and ("LI" in d or "LION" in d):
        return 1800.0 if "16" in d else 1200.0
    if "BATTERY" in d:
        return 950.0
    if "MOTOR" in d or "MCU" in d:
        return 1450.0
    if "CHARGE" in d or "CHGR" in d:
        return 380.0
    if "SEAT" in d:
        return 220.0
    if "MIRROR" in d:
        return 65.0
    if "WINDSHIELD" in d:
        return 180.0
    if "CANOPY" in d or "ROOF" in d:
        return 310.0
    if "STEERING" in d:
        return 140.0
    if "WHEEL" in d:
        return 95.0
    if "DECAL" in d:
        return 12.0
    if "KIT" in d or "INSTL" in d:
        return 250.0
    if "SWP" in part or "SWF" in part:
        return 0.50  # software part number
    return 35.0


def _is_required_for(variant_code: str) -> bool:
    return variant_code == "DR"


def _is_applicable_for(variant_code: str) -> bool:
    return variant_code in ("DR", "X", "B")


def main() -> None:
    abom_path = ROOT / "data" / "Copy of 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx"
    df = load_real_abom(str(abom_path))

    # Real costs from customer file (560 parts)
    real_costs = pd.read_csv(ROOT / "data" / "real_costs.csv")
    cost_map = {str(r.part_number): float(r.Current or r.Standard) for r in real_costs.itertuples()}

    # Per-vehicle demand: parts marked DR for a variant are needed in every
    # vehicle of that variant. Parts marked X are needed in roughly 1/N of
    # vehicles where N = number of X-marked choices in that group (proxy for
    # "one option pick per vehicle"). For the demo this is approximated as
    # 0.5 * applicable demand for X-marked parts.
    li_target = int(round(TARGET_VEHICLES * TARGET_LI_SHARE))
    agm_target = TARGET_VEHICLES - li_target

    rows = []
    for r in df.itertuples():
        agm_app = _is_applicable_for(r.agm_code)
        li_app = _is_applicable_for(r.li_code)
        if not (agm_app or li_app):
            continue

        # Estimate demand. Many X-marked parts end up as singleton required
        # groups in the interpreter (no real choice alternatives), so treat
        # both DR and X applicability as full demand. Distribution across
        # actual option choices is then a planning concern, not a stock one.
        agm_demand = agm_target if agm_app else 0
        li_demand = li_target if li_app else 0
        base_demand = max(1, int(round(agm_demand + li_demand)))

        # Default on-hand: demand * slack
        on_hand = int(round(base_demand * DEFAULT_SLACK))
        aging = 30
        warehouse = "A"

        planted = PLANTED.get(r.part_number)
        if planted:
            if "on_hand" in planted:
                on_hand = int(planted["on_hand"])
            else:
                on_hand = int(round(base_demand * planted["qty_factor"]))
            aging = planted.get("aging_days", aging)
            warehouse = planted.get("warehouse", warehouse)

        unit_cost = cost_map.get(r.part_number) or _cost_heuristic(r.part_number, r.description)

        rows.append({
            "part_number": r.part_number,
            "description": r.description,
            "on_hand": on_hand,
            "unit_cost": round(float(unit_cost), 2),
            "aging_days": aging,
            "warehouse": warehouse,
            "agm_code": r.agm_code,
            "li_code": r.li_code,
        })

    out = pd.DataFrame(rows).sort_values("part_number")
    out_path = ROOT / "data" / "inventory_synthetic.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} parts")
    print(f"Targets: {TARGET_VEHICLES} vehicles total ({agm_target} AGM / {li_target} LI)")
    print()
    print("Planted constraints:")
    for p, meta in PLANTED.items():
        row = out[out["part_number"] == p]
        if row.empty:
            print(f"  ! {p}: NOT FOUND in ABOM — check part number")
            continue
        r = row.iloc[0]
        print(f"  {p}  {r.description[:38]:38s}  on_hand={r.on_hand:4d}  "
              f"cost=${r.unit_cost:7.2f}  ({meta['note']})")


if __name__ == "__main__":
    main()
