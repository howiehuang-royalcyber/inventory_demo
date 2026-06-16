"""Permutation enumerator.

Given an approved Configuration Model, enumerate every valid configuration
(one concrete buildable parts list per configuration) using OR-Tools CP-SAT.
"""
from __future__ import annotations

from itertools import product

from ortools.sat.python import cp_model

from .models import ConfigurationModel, ValidConfiguration


def enumerate_valid_configurations(cm: ConfigurationModel, max_solutions: int = 10000) -> list[ValidConfiguration]:
    """Return all valid (group_choice) assignments that satisfy the constraints."""
    if not cm.option_groups:
        # degenerate: a single required-only configuration
        return [_materialize(cm, {})]

    groups = cm.option_groups
    name_to_group = {g.group_id: g for g in groups}
    choice_index: dict[str, dict[str, int]] = {
        g.group_id: {c: i for i, c in enumerate(g.choices)} for g in groups
    }

    model = cp_model.CpModel()
    vars_ = {g.group_id: model.NewIntVar(0, len(g.choices) - 1, g.group_id) for g in groups}

    # Encode constraints
    for con in cm.constraints:
        if_g = con.if_.group
        if_c = con.if_.choice
        if if_g not in vars_ or if_c not in choice_index.get(if_g, {}):
            # constraint refers to a required group or unknown — skip
            continue
        if_idx = choice_index[if_g][if_c]

        if con.type == "implies" and con.then is not None:
            tg, tc = con.then.group, con.then.choice
            if tg not in vars_ or tc not in choice_index.get(tg, {}):
                continue
            then_idx = choice_index[tg][tc]
            # (vars_[if_g] == if_idx) -> (vars_[tg] == then_idx)
            b = model.NewBoolVar(f"impl_{if_g}_{if_c}_{tg}_{tc}")
            model.Add(vars_[if_g] == if_idx).OnlyEnforceIf(b)
            model.Add(vars_[if_g] != if_idx).OnlyEnforceIf(b.Not())
            model.Add(vars_[tg] == then_idx).OnlyEnforceIf(b)

        elif con.type in ("excludes", "forbidden_with") and con.excluded is not None:
            eg, ec = con.excluded.group, con.excluded.choice
            if eg not in vars_ or ec not in choice_index.get(eg, {}):
                continue
            ex_idx = choice_index[eg][ec]
            b = model.NewBoolVar(f"excl_{if_g}_{if_c}_{eg}_{ec}")
            model.Add(vars_[if_g] == if_idx).OnlyEnforceIf(b)
            model.Add(vars_[if_g] != if_idx).OnlyEnforceIf(b.Not())
            model.Add(vars_[eg] != ex_idx).OnlyEnforceIf(b)

    solver = cp_model.CpSolver()
    collector = _SolutionCollector(vars_, max_solutions)
    solver.parameters.enumerate_all_solutions = True
    solver.Solve(model, collector)

    valid: list[ValidConfiguration] = []
    for i, sol in enumerate(collector.solutions):
        choices = {gid: name_to_group[gid].choices[idx] for gid, idx in sol.items()}
        valid.append(_materialize(cm, choices, config_idx=i))
    return valid


def _materialize(cm: ConfigurationModel, choices: dict[str, str], config_idx: int = 0) -> ValidConfiguration:
    parts: dict[str, int] = {}
    for rg in cm.required_groups:
        for p in rg.parts:
            parts[p] = parts.get(p, 0) + rg.qty_per_vehicle
    for og in cm.option_groups:
        chosen = choices.get(og.group_id)
        if not chosen:
            continue
        parts[chosen] = parts.get(chosen, 0) + og.qty_per_vehicle
    return ValidConfiguration(
        config_id=f"CFG-{config_idx:04d}",
        vehicle_id=cm.vehicle_id,
        choices=choices,
        parts_list=parts,
    )


class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, vars_: dict, limit: int):
        super().__init__()
        self._vars = vars_
        self._limit = limit
        self.solutions: list[dict] = []

    def on_solution_callback(self) -> None:
        if len(self.solutions) >= self._limit:
            self.StopSearch()
            return
        self.solutions.append({gid: int(self.Value(v)) for gid, v in self._vars.items()})


if __name__ == "__main__":
    import pandas as pd
    from .interpreter import interpret_abom

    df = pd.read_csv("data/crew_cru_abom.csv")
    cm = interpret_abom(df)
    configs = enumerate_valid_configurations(cm)
    print(f"valid configurations: {len(configs)}")
    for c in configs[:3]:
        print(c.model_dump())
