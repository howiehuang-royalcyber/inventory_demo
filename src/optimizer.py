"""Buildable-units optimizer.

Mixed-integer program:
  Variables: x_i = number of vehicles built using configuration i (integer >= 0)
  Constraints: for every part p,  sum_i qty(p, i) * x_i  <=  on_hand[p]
  Objective: maximise sum_i x_i

After solving:
  - Binding constraints: parts where consumed == on_hand
  - Unlock analysis: marginal vehicles unlocked per +1 unit of each binding part
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from ortools.linear_solver import pywraplp

from .models import (
    BindingConstraint,
    BuildPlanLine,
    BuildPlanResult,
    UnlockSuggestion,
    ValidConfiguration,
)


def optimize(
    configs: list[ValidConfiguration],
    inventory: dict[str, int],
    costs: Optional[dict[str, float]] = None,
    aging_days: Optional[dict[str, int]] = None,
    exclude_configs: Optional[set[str]] = None,
    inventory_overrides: Optional[dict[str, int]] = None,
) -> BuildPlanResult:
    if not configs:
        return BuildPlanResult(
            total_vehicles=0, plan=[], parts_consumed={},
            binding_constraints=[], unlock_suggestions=[],
        )

    if inventory_overrides:
        inventory = {**inventory, **inventory_overrides}
    if exclude_configs:
        configs = [c for c in configs if c.config_id not in exclude_configs]

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available")

    x = {c.config_id: solver.IntVar(0, solver.infinity(), f"x_{c.config_id}") for c in configs}

    # Build part constraints
    all_parts: set[str] = set()
    for c in configs:
        all_parts.update(c.parts_list.keys())

    part_constraints = {}
    for p in all_parts:
        on_hand = max(0, int(inventory.get(p, 0)))
        ct = solver.Constraint(0, on_hand)
        for c in configs:
            qty = c.parts_list.get(p, 0)
            if qty:
                ct.SetCoefficient(x[c.config_id], qty)
        part_constraints[p] = (ct, on_hand)

    # Objective: maximise vehicles; lightly prefer aging stock if provided
    obj = solver.Objective()
    for c in configs:
        weight = 1.0
        if aging_days:
            # tiny tiebreaker per part age: total age of parts in this config
            age = sum(aging_days.get(p, 0) * q for p, q in c.parts_list.items())
            weight = 1.0 + 1e-6 * age
        obj.SetCoefficient(x[c.config_id], weight)
    obj.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return BuildPlanResult(
            total_vehicles=0, plan=[], parts_consumed={},
            binding_constraints=[], unlock_suggestions=[],
        )

    # Extract plan
    plan: list[BuildPlanLine] = []
    consumed: dict[str, int] = {}
    total = 0
    for c in configs:
        q = int(round(x[c.config_id].solution_value()))
        if q <= 0:
            continue
        total += q
        plan.append(BuildPlanLine(config_id=c.config_id, quantity=q, choices=c.choices))
        for p, qty_per in c.parts_list.items():
            consumed[p] = consumed.get(p, 0) + q * qty_per

    # Binding constraints
    binding: list[BindingConstraint] = []
    for p, (_, on_hand) in part_constraints.items():
        used = consumed.get(p, 0)
        if on_hand > 0 and used == on_hand:
            binding.append(BindingConstraint(
                part_number=p, on_hand=on_hand, consumed=used, slack=0,
            ))

    # Unlock analysis: for each binding part, re-solve with +N units and see marginal vehicles.
    unlock: list[UnlockSuggestion] = []
    for bc in binding:
        # Try a small bump (e.g., +5) per binding part, take the average marginal.
        bump = 5
        bumped_inventory = {**inventory, bc.part_number: inventory.get(bc.part_number, 0) + bump}
        bumped_result = _solve_simple(configs, bumped_inventory, costs, aging_days)
        delta_vehicles = bumped_result.total_vehicles - total
        if delta_vehicles <= 0:
            continue
        # vehicles per unit
        per_unit = delta_vehicles / bump
        unit_cost = costs.get(bc.part_number) if costs else None
        vpd = (per_unit / unit_cost) if (unit_cost and unit_cost > 0) else None
        unlock.append(UnlockSuggestion(
            part_number=bc.part_number,
            additional_qty_needed=bump,
            additional_vehicles_unlocked=delta_vehicles,
            estimated_cost=(unit_cost * bump) if unit_cost else None,
            vehicles_per_dollar=vpd,
        ))

    # Sort unlock by vehicles unlocked desc
    unlock.sort(key=lambda u: u.additional_vehicles_unlocked, reverse=True)

    return BuildPlanResult(
        total_vehicles=total, plan=sorted(plan, key=lambda p: -p.quantity),
        parts_consumed=consumed, binding_constraints=binding,
        unlock_suggestions=unlock,
    )


def optimize_multi_sku(
    configs_by_sku: dict[str, list[ValidConfiguration]],
    inventory: dict[str, int],
    costs: Optional[dict[str, float]] = None,
    aging_days: Optional[dict[str, int]] = None,
    mix: Optional[dict[str, float]] = None,
    exclude_configs: Optional[set[str]] = None,
    inventory_overrides: Optional[dict[str, int]] = None,
    mix_tolerance: float = 0.05,
) -> dict:
    """Multi-SKU MIP. Variables x_{sku}_{config_id}, shared part pool.

    If `mix` is provided, adds linear inequalities so each SKU's share of total
    builds is within `mix_tolerance` of the target (one-sided lower bounds, which
    together with the totals implicitly enforce upper bounds since shares sum to 1).
    """
    skus = list(configs_by_sku.keys())
    if not skus or all(not v for v in configs_by_sku.values()):
        return {
            "total_vehicles": 0,
            "by_sku": {s: 0 for s in skus},
            "plan_by_sku": {s: [] for s in skus},
            "parts_consumed": {},
            "binding_constraints": [],
            "unlock_suggestions": [],
        }

    if inventory_overrides:
        inventory = {**inventory, **inventory_overrides}
    if exclude_configs:
        configs_by_sku = {
            s: [c for c in cs if c.config_id not in exclude_configs]
            for s, cs in configs_by_sku.items()
        }

    result = _solve_multi(
        configs_by_sku, inventory, costs, aging_days, mix, mix_tolerance
    )
    if result is None:
        return {
            "total_vehicles": 0,
            "by_sku": {s: 0 for s in skus},
            "plan_by_sku": {s: [] for s in skus},
            "parts_consumed": {},
            "binding_constraints": [],
            "unlock_suggestions": [],
        }

    total, by_sku, plan_by_sku, consumed, part_on_hand = result

    # Binding constraints
    binding: list[BindingConstraint] = []
    for p, on_hand in part_on_hand.items():
        used = consumed.get(p, 0)
        if on_hand > 0 and used == on_hand:
            binding.append(BindingConstraint(
                part_number=p, on_hand=on_hand, consumed=used, slack=0,
            ))

    # Unlock analysis
    unlock: list[UnlockSuggestion] = []
    for bc in binding:
        bump = 5
        bumped_inv = {**inventory, bc.part_number: inventory.get(bc.part_number, 0) + bump}
        bumped = _solve_multi(configs_by_sku, bumped_inv, costs, aging_days, mix, mix_tolerance)
        if bumped is None:
            continue
        delta = bumped[0] - total
        if delta <= 0:
            continue
        per_unit = delta / bump
        unit_cost = costs.get(bc.part_number) if costs else None
        vpd = (per_unit / unit_cost) if (unit_cost and unit_cost > 0) else None
        unlock.append(UnlockSuggestion(
            part_number=bc.part_number,
            additional_qty_needed=bump,
            additional_vehicles_unlocked=delta,
            estimated_cost=(unit_cost * bump) if unit_cost else None,
            vehicles_per_dollar=vpd,
        ))
    unlock.sort(key=lambda u: u.additional_vehicles_unlocked, reverse=True)

    return {
        "total_vehicles": total,
        "by_sku": by_sku,
        "plan_by_sku": plan_by_sku,
        "parts_consumed": consumed,
        "binding_constraints": binding,
        "unlock_suggestions": unlock,
    }


def _solve_multi(
    configs_by_sku: dict[str, list[ValidConfiguration]],
    inventory: dict[str, int],
    costs: Optional[dict[str, float]],
    aging_days: Optional[dict[str, int]],
    mix: Optional[dict[str, float]],
    mix_tolerance: float,
):
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available")

    # Variables per (sku, config)
    x: dict[tuple[str, str], object] = {}
    for sku, cs in configs_by_sku.items():
        for c in cs:
            x[(sku, c.config_id)] = solver.IntVar(0, solver.infinity(), f"x_{sku}_{c.config_id}")

    # Part constraints (shared pool)
    all_parts: set[str] = set()
    for cs in configs_by_sku.values():
        for c in cs:
            all_parts.update(c.parts_list.keys())

    part_on_hand: dict[str, int] = {}
    for p in all_parts:
        on_hand = max(0, int(inventory.get(p, 0)))
        ct = solver.Constraint(0, on_hand)
        for sku, cs in configs_by_sku.items():
            for c in cs:
                q = c.parts_list.get(p, 0)
                if q:
                    ct.SetCoefficient(x[(sku, c.config_id)], q)
        part_on_hand[p] = on_hand

    # Per-SKU totals as auxiliary expressions; mix constraints
    # total_sku - share * total_all >= -tol * total_all
    # => total_sku - (share - tol) * total_all >= 0
    # We linearise by writing this directly over the x variables:
    #   sum_{c in sku} x - (share - tol) * sum_{all c} x >= 0
    if mix:
        # validate keys
        for s in mix:
            if s not in configs_by_sku:
                raise ValueError(f"mix key {s!r} not in configs_by_sku")
        for sku, share in mix.items():
            lower = max(0.0, share - mix_tolerance)
            if lower <= 0:
                continue
            ct = solver.Constraint(0, solver.infinity())
            for (s2, cid), var in x.items():
                if s2 == sku:
                    ct.SetCoefficient(var, 1.0 - lower)
                else:
                    ct.SetCoefficient(var, -lower)
        # If a SKU's target is exactly 0, force its variables to 0
        for sku, share in mix.items():
            if share <= 0:
                for (s2, cid), var in x.items():
                    if s2 == sku:
                        ct = solver.Constraint(0, 0)
                        ct.SetCoefficient(var, 1.0)

    # Objective: maximise total vehicles + tiny aging tiebreaker
    obj = solver.Objective()
    for (sku, cid), var in x.items():
        weight = 1.0
        if aging_days:
            cfg = next(c for c in configs_by_sku[sku] if c.config_id == cid)
            age = sum(aging_days.get(p, 0) * q for p, q in cfg.parts_list.items())
            weight = 1.0 + 1e-6 * age
        obj.SetCoefficient(var, weight)
    obj.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    by_sku: dict[str, int] = {s: 0 for s in configs_by_sku}
    plan_by_sku: dict[str, list[BuildPlanLine]] = {s: [] for s in configs_by_sku}
    consumed: dict[str, int] = {}
    total = 0
    for sku, cs in configs_by_sku.items():
        for c in cs:
            q = int(round(x[(sku, c.config_id)].solution_value()))
            if q <= 0:
                continue
            total += q
            by_sku[sku] += q
            plan_by_sku[sku].append(
                BuildPlanLine(config_id=c.config_id, quantity=q, choices=c.choices)
            )
            for p, qty_per in c.parts_list.items():
                consumed[p] = consumed.get(p, 0) + q * qty_per
    for sku in plan_by_sku:
        plan_by_sku[sku].sort(key=lambda p: -p.quantity)

    return total, by_sku, plan_by_sku, consumed, part_on_hand


def _solve_simple(configs, inventory, costs, aging_days) -> BuildPlanResult:
    """Inner solve without unlock recursion."""
    solver = pywraplp.Solver.CreateSolver("CBC")
    x = {c.config_id: solver.IntVar(0, solver.infinity(), f"x_{c.config_id}") for c in configs}
    all_parts = set().union(*[c.parts_list.keys() for c in configs])
    for p in all_parts:
        on_hand = max(0, int(inventory.get(p, 0)))
        ct = solver.Constraint(0, on_hand)
        for c in configs:
            q = c.parts_list.get(p, 0)
            if q:
                ct.SetCoefficient(x[c.config_id], q)
    obj = solver.Objective()
    for c in configs:
        obj.SetCoefficient(x[c.config_id], 1.0)
    obj.SetMaximization()
    solver.Solve()
    total = sum(int(round(v.solution_value())) for v in x.values())
    return BuildPlanResult(
        total_vehicles=total, plan=[], parts_consumed={},
        binding_constraints=[], unlock_suggestions=[],
    )


def inventory_from_df(df: pd.DataFrame) -> dict[str, int]:
    return {row["part_number"]: int(row["on_hand"]) for _, row in df.iterrows()}


def costs_from_df(df: pd.DataFrame) -> dict[str, float]:
    return {row["part_number"]: float(row.get("unit_cost", 0.0)) for _, row in df.iterrows()}


def aging_from_df(df: pd.DataFrame) -> dict[str, int]:
    return {row["part_number"]: int(row.get("aging_days", 0)) for _, row in df.iterrows()}


if __name__ == "__main__":
    from .interpreter import interpret_abom
    from .enumerator import enumerate_valid_configurations

    abom = pd.read_csv("data/crew_cru_abom.csv")
    inv = pd.read_csv("data/inventory.csv")
    cm = interpret_abom(abom)
    configs = enumerate_valid_configurations(cm)
    result = optimize(
        configs,
        inventory_from_df(inv),
        costs=costs_from_df(inv),
        aging_days=aging_from_df(inv),
    )
    print(f"buildable: {result.total_vehicles}")
    print(f"binding: {[(b.part_number, b.on_hand) for b in result.binding_constraints]}")
    print(f"unlock top 3: {[(u.part_number, u.additional_vehicles_unlocked) for u in result.unlock_suggestions[:3]]}")
