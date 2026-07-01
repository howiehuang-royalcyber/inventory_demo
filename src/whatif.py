"""Natural-language what-if layer.

Closed-world intent surface (5 supported intents). Claude translates the
NL question into one of these intent objects. The Python service then applies
the intent to the optimizer inputs and re-solves. A second LLM call (optional)
narrates the diff.

If ANTHROPIC_API_KEY is unset, a simple keyword-based parser fills in.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .models import BuildPlanResult, ConfigurationModel, ValidConfiguration
from .optimizer import optimize, optimize_multi_sku


# Real ABOM part IDs (from 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx).
BATTERY_CONCEPT_PARTS = {
    "16kwh": ["47787614002"],   # INSTL, BATTERY, 72V SYS, LI 16KW
    "8kwh": ["47787614001"],    # INSTL, BATTERY, 72V SYS, LI 8KW
    "agm": ["47782768001", "47782768002"],
    "lithium": ["47787614001", "47787614002"],
}


INTENT_TOOL = {
    "name": "resolve_intent",
    "description": (
        "Translate the user's natural-language what-if question into a structured intent. "
        "Choose exactly one intent. If the question doesn't fit any supported intent, "
        "use 'unsupported' with a brief reason."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "exclude_choice",
                    "force_choice",
                    "assume_inventory",
                    "unlock_top_n",
                    "compare_scenarios",
                    "set_mix",
                    "unlock_for_mix",
                    "delay_part",
                    "unsupported",
                ],
            },
            "group_id": {"type": "string"},
            "choice": {"type": "string"},
            "part_number": {"type": "string"},
            "quantity": {"type": "integer"},
            "n": {"type": "integer"},
            "reason": {"type": "string"},
            "agm_share": {"type": "number"},
            "li_share": {"type": "number"},
            "battery_concept": {"type": "string"},
            "weeks_late": {"type": "number"},
        },
        "required": ["intent"],
    },
}


@dataclass
class Intent:
    kind: str
    payload: dict


def resolve_intent_with_claude(question: str, cm: ConfigurationModel, inventory: dict[str, int]) -> Intent:
    """Use Claude to translate NL -> intent. Fall back to heuristic if no key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _heuristic_intent(question, cm, inventory)

    try:
        import anthropic
    except ImportError:
        return _heuristic_intent(question, cm, inventory)

    group_summary = "\n".join(
        f"  - {g.group_id}: choices = {g.choices}" for g in cm.option_groups
    )
    parts_summary = ", ".join(sorted(inventory.keys()))

    system = (
        "You translate inventory what-if questions into a fixed, machine-readable intent. "
        "Use only the listed groups and parts. Choose exactly one intent."
    )
    user = (
        f"Question: {question}\n\n"
        f"Option groups:\n{group_summary}\n\n"
        f"Known part numbers: {parts_summary}\n"
    )

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=512,
            system=system,
            tools=[INTENT_TOOL],
            tool_choice={"type": "tool", "name": "resolve_intent"},
            messages=[{"role": "user", "content": user}],
        )
        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if tool_use:
            return Intent(kind=tool_use.input.get("intent", "unsupported"), payload=tool_use.input)
    except Exception as e:
        print(f"[whatif] LLM intent resolution failed: {e}")

    return _heuristic_intent(question, cm, inventory)


def _heuristic_intent(question: str, cm: ConfigurationModel, inventory: dict[str, int]) -> Intent:
    q = question.lower()

    # set_mix: "drop AGM and go all-lithium" / "shift mix to 80% lithium"
    if re.search(r"\b(drop|skip|no)\b.*\bagm\b", q) or re.search(r"\ball[- ]?lithium\b", q) or re.search(r"\ball[- ]?li\b", q):
        return Intent("set_mix", {"agm_share": 0.0, "li_share": 1.0})
    m = re.search(r"(\d{1,3})\s*%\s*(?:lithium|li)\b", q)
    if m:
        li = int(m.group(1)) / 100.0
        return Intent("set_mix", {"agm_share": max(0.0, 1.0 - li), "li_share": li})
    m = re.search(r"(\d{1,3})\s*%\s*agm\b", q)
    if m:
        agm = int(m.group(1)) / 100.0
        return Intent("set_mix", {"agm_share": agm, "li_share": max(0.0, 1.0 - agm)})

    # unlock_for_mix: "which part should we expedite to unlock the most lithium builds"
    if ("expedite" in q or "unlock" in q) and ("lithium" in q or " li " in f" {q} "):
        return Intent("unlock_for_mix", {"li_share": 0.9, "agm_share": 0.1, "n": 1})

    # delay_part: "16kWh battery shipment is two weeks late"
    if re.search(r"\b16\s*kwh\b", q) and ("late" in q or "delay" in q or "miss" in q):
        return Intent("delay_part", {"battery_concept": "16kwh", "weeks_late": 2})
    if re.search(r"\b8\s*kwh\b", q) and ("late" in q or "delay" in q or "miss" in q):
        return Intent("delay_part", {"battery_concept": "8kwh", "weeks_late": 2})

    # exclude_choice: "skip deluxe canopy" / "without red seats"
    for g in cm.option_groups:
        for c in g.choices:
            cl = c.lower()
            tokens = re.split(r"[-_]", cl)
            if any(re.search(rf"(?:skip|exclude|without|no)\b.*{re.escape(t.lower())}", q) for t in tokens if len(t) > 2):
                return Intent("exclude_choice", {"group_id": g.group_id, "choice": c})

    # force_choice: "use steel wheels only"
    for g in cm.option_groups:
        for c in g.choices:
            cl = c.lower()
            tokens = re.split(r"[-_]", cl)
            if any(re.search(rf"(?:only|must|force)\b.*{re.escape(t.lower())}", q) for t in tokens if len(t) > 2):
                return Intent("force_choice", {"group_id": g.group_id, "choice": c})

    # assume_inventory: "what if we had 20 more batteries"
    m = re.search(r"(?:had|add(?:ed)?|more)\s+(\d+)\s+(?:more\s+)?([A-Z0-9\-]+)", question)
    if m:
        return Intent("assume_inventory", {"part_number": m.group(2).upper(), "quantity": int(m.group(1))})

    # unlock top N
    m = re.search(r"unlock\s+the\s+most|expedite|top\s+(\d+)", q)
    if m:
        n = int(m.group(1) or 3) if m.lastindex else 3
        return Intent("unlock_top_n", {"n": n})

    return Intent("unsupported", {"reason": "Could not match any supported intent."})


def apply_intent(
    intent: Intent,
    configs: list[ValidConfiguration],
    inventory: dict[str, int],
    costs: Optional[dict[str, float]] = None,
    aging: Optional[dict[str, int]] = None,
) -> tuple[BuildPlanResult, str]:
    """Apply intent to optimizer inputs, re-solve, return result + short narration."""
    if intent.kind == "exclude_choice":
        g = intent.payload["group_id"]
        c = intent.payload["choice"]
        excl = {cfg.config_id for cfg in configs if cfg.choices.get(g) == c}
        result = optimize(configs, inventory, costs, aging, exclude_configs=excl)
        return result, f"Excluded configurations using {c} ({g})."

    if intent.kind == "force_choice":
        g = intent.payload["group_id"]
        c = intent.payload["choice"]
        excl = {cfg.config_id for cfg in configs if cfg.choices.get(g) != c}
        result = optimize(configs, inventory, costs, aging, exclude_configs=excl)
        return result, f"Forced all configurations to use {c} ({g})."

    if intent.kind == "assume_inventory":
        part = intent.payload["part_number"]
        qty = int(intent.payload["quantity"])
        overrides = {part: inventory.get(part, 0) + qty}
        result = optimize(configs, inventory, costs, aging, inventory_overrides=overrides)
        return result, f"Assumed +{qty} units of {part}."

    if intent.kind == "unlock_top_n":
        n = int(intent.payload.get("n") or 3)
        result = optimize(configs, inventory, costs, aging)
        top = result.unlock_suggestions[:n]
        narration = "Top unlock suggestions:\n" + "\n".join(
            f"  • +{u.additional_qty_needed} {u.part_number} -> +{u.additional_vehicles_unlocked} vehicles"
            + (f" (~${u.estimated_cost:,.0f})" if u.estimated_cost else "")
            for u in top
        )
        return result, narration

    # unsupported
    result = optimize(configs, inventory, costs, aging)
    return result, f"Unsupported intent: {intent.payload.get('reason', '')}"


def _resolve_concept_part(concept: str, inventory: dict[str, int]) -> Optional[str]:
    """Find a real part number in inventory matching the given concept."""
    candidates = BATTERY_CONCEPT_PARTS.get(concept.lower(), [])
    for cand in candidates:
        if cand in inventory:
            return cand
    # Fallback: substring search
    needle = concept.lower().replace(" ", "")
    for p in inventory:
        if needle in p.lower().replace("-", "").replace("_", ""):
            return p
    return None


def apply_intent_multi_sku(
    intent: Intent,
    configs_by_sku: dict[str, list[ValidConfiguration]],
    inventory: dict[str, int],
    costs: Optional[dict[str, float]] = None,
    aging: Optional[dict[str, int]] = None,
    baseline_mix: Optional[dict[str, float]] = None,
) -> tuple[dict, str]:
    """Apply intent to multi-SKU optimizer inputs, re-solve, return result + narration.

    `baseline_mix` is what the UI is currently displaying — used when the intent
    doesn't override the mix.
    """
    if intent.kind == "set_mix":
        agm = float(intent.payload.get("agm_share", 0.0))
        li = float(intent.payload.get("li_share", 1.0))
        mix = {"AGM": agm, "LI": li}
        result = optimize_multi_sku(configs_by_sku, inventory, costs, aging, mix=mix)
        return result, f"Set mix to {int(agm*100)}% AGM / {int(li*100)}% Lithium."

    if intent.kind == "unlock_for_mix":
        li = float(intent.payload.get("li_share", 0.9))
        agm = float(intent.payload.get("agm_share", 1.0 - li))
        mix = {"AGM": agm, "LI": li}
        result = optimize_multi_sku(configs_by_sku, inventory, costs, aging, mix=mix)
        top = (result.get("unlock_suggestions") or [])[:1]
        if top:
            u = top[0]
            cost = f" (~${u.estimated_cost:,.0f})" if u.estimated_cost else ""
            narr = (
                f"At a {int(li*100)}% Lithium mix, expedite +{u.additional_qty_needed} "
                f"{u.part_number} to unlock +{u.additional_vehicles_unlocked} vehicles{cost}."
            )
        else:
            narr = "No single-part expedite would unlock more vehicles at that mix."
        return result, narr

    if intent.kind == "delay_part":
        concept = intent.payload.get("battery_concept", "16kwh")
        weeks = intent.payload.get("weeks_late", 2)
        part = _resolve_concept_part(concept, inventory)
        if not part:
            result = optimize_multi_sku(
                configs_by_sku, inventory, costs, aging, mix=baseline_mix
            )
            return result, (
                f"Couldn't find a part for concept '{concept}' in inventory. "
                f"Showing baseline mix."
            )
        overrides = {part: 0}
        result = optimize_multi_sku(
            configs_by_sku, inventory, costs, aging,
            mix=baseline_mix, inventory_overrides=overrides,
        )
        return result, (
            f"Assumed {part} ({concept}) is {weeks} weeks late — set on-hand to 0."
        )

    # Fallback: re-solve with baseline mix
    result = optimize_multi_sku(
        configs_by_sku, inventory, costs, aging, mix=baseline_mix
    )
    return result, f"Unsupported multi-SKU intent: {intent.kind}"
