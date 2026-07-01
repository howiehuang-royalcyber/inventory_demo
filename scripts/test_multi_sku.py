"""Synthetic 2-SKU test fixture for optimize_multi_sku.

Run with the project venv:
    source .venv/bin/activate && python scripts/test_multi_sku.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ValidConfiguration
from src.optimizer import optimize_multi_sku


def make_configs() -> dict[str, list[ValidConfiguration]]:
    """Hand-rolled valid configs for two SKUs sharing FRAME-01 but differing on
    powertrain (MC-AGM vs MC-LI) and battery (BATTERY-AGM vs BATTERY-LI-8KWH /
    BATTERY-LI-16KWH)."""
    agm = [
        ValidConfiguration(
            config_id="AGM-A",
            vehicle_id="CRU-AGM",
            choices={"powertrain": "MC-AGM", "seat": "SEAT-BLK"},
            parts_list={"FRAME-01": 1, "MC-AGM": 1, "BATTERY-AGM": 1, "SEAT-BLK": 2},
        ),
        ValidConfiguration(
            config_id="AGM-B",
            vehicle_id="CRU-AGM",
            choices={"powertrain": "MC-AGM", "seat": "SEAT-BEI"},
            parts_list={"FRAME-01": 1, "MC-AGM": 1, "BATTERY-AGM": 1, "SEAT-BEI": 2},
        ),
    ]
    li = [
        ValidConfiguration(
            config_id="LI-8",
            vehicle_id="CRU-LI",
            choices={"powertrain": "MC-LI", "battery": "BATTERY-LI-8KWH", "seat": "SEAT-BLK"},
            parts_list={"FRAME-01": 1, "MC-LI": 1, "BATTERY-LI-8KWH": 1, "SEAT-BLK": 2},
        ),
        ValidConfiguration(
            config_id="LI-16",
            vehicle_id="CRU-LI",
            choices={"powertrain": "MC-LI", "battery": "BATTERY-LI-16KWH", "seat": "SEAT-BLK"},
            parts_list={"FRAME-01": 1, "MC-LI": 1, "BATTERY-LI-16KWH": 1, "SEAT-BLK": 2},
        ),
        ValidConfiguration(
            config_id="LI-16-BEI",
            vehicle_id="CRU-LI",
            choices={"powertrain": "MC-LI", "battery": "BATTERY-LI-16KWH", "seat": "SEAT-BEI"},
            parts_list={"FRAME-01": 1, "MC-LI": 1, "BATTERY-LI-16KWH": 1, "SEAT-BEI": 2},
        ),
    ]
    return {"AGM": agm, "LI": li}


def make_inventory() -> dict[str, int]:
    return {
        "FRAME-01": 100,         # tight overall constraint
        "MC-AGM": 60,
        "MC-LI": 80,
        "BATTERY-AGM": 60,
        "BATTERY-LI-8KWH": 20,
        "BATTERY-LI-16KWH": 50,
        "SEAT-BLK": 200,
        "SEAT-BEI": 60,
    }


def print_result(label: str, res: dict, mix: dict | None) -> None:
    print(f"\n=== {label} ===")
    print(f"mix target: {mix}")
    print(f"total: {res['total_vehicles']}  by_sku: {res['by_sku']}")
    total = res["total_vehicles"]
    if total:
        for sku, n in res["by_sku"].items():
            print(f"  {sku} share: {n/total:.2%}")
    bnd = [(b.part_number, b.on_hand, b.consumed) for b in res["binding_constraints"]]
    print(f"binding: {bnd}")
    if res["unlock_suggestions"]:
        u = res["unlock_suggestions"][0]
        print(f"top unlock: +{u.additional_qty_needed} {u.part_number} -> +{u.additional_vehicles_unlocked} vehicles")


def main() -> None:
    configs = make_configs()
    inv = make_inventory()
    costs = {p: 100.0 for p in inv}

    # 1. No mix constraint — pure max
    r0 = optimize_multi_sku(configs, inv, costs=costs)
    print_result("no mix (max total)", r0, None)

    # 2. 65% Lithium
    mix1 = {"AGM": 0.35, "LI": 0.65}
    r1 = optimize_multi_sku(configs, inv, costs=costs, mix=mix1)
    print_result("65% LI / 35% AGM", r1, mix1)

    # 3. 90% Lithium
    mix2 = {"AGM": 0.1, "LI": 0.9}
    r2 = optimize_multi_sku(configs, inv, costs=costs, mix=mix2)
    print_result("90% LI / 10% AGM", r2, mix2)

    # 4. All-lithium
    mix3 = {"AGM": 0.0, "LI": 1.0}
    r3 = optimize_multi_sku(configs, inv, costs=costs, mix=mix3)
    print_result("100% LI (drop AGM)", r3, mix3)

    # 5. 16kWh battery shipment late
    overrides = {"BATTERY-LI-16KWH": 0}
    r4 = optimize_multi_sku(configs, inv, costs=costs, mix=mix1, inventory_overrides=overrides)
    print_result("65% LI, 16kWh late", r4, mix1)

    # --- assertions ---
    tol = 0.05
    for label, r, mix in [
        ("65% LI", r1, mix1),
        ("90% LI", r2, mix2),
        ("100% LI", r3, mix3),
    ]:
        total = r["total_vehicles"]
        if total == 0:
            continue
        for sku, target in mix.items():
            actual = r["by_sku"][sku] / total
            lower = max(0.0, target - tol)
            assert actual + 1e-9 >= lower, (
                f"{label}: {sku} share {actual:.3f} below lower bound {lower:.3f}"
            )
    # All-lithium must have AGM == 0
    assert r3["by_sku"]["AGM"] == 0, "expected AGM=0 when share=0"
    # All-lithium total should be <= 90%/65% LI total ? Not necessarily; just check feasibility
    assert r0["total_vehicles"] >= r1["total_vehicles"], (
        "unconstrained total should be >= any mix-constrained total"
    )
    print("\nAll mix-constraint assertions passed.")


if __name__ == "__main__":
    main()
