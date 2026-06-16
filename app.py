"""Club Car Inventory Intelligence — Streamlit demo.

Run:
  streamlit run app.py

Set ANTHROPIC_API_KEY in env to enable the real LLM interpreter and NL layer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.enumerator import enumerate_valid_configurations
from src.interpreter import interpret_abom
from src.models import ConfigurationModel
from src.optimizer import aging_from_df, costs_from_df, inventory_from_df, optimize
from src.whatif import apply_intent, resolve_intent_with_claude


DATA = Path("data")

st.set_page_config(
    page_title="Club Car Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- helpers ----------

def _llm_status() -> str:
    return "✅ Claude (real LLM)" if os.environ.get("ANTHROPIC_API_KEY") else "⚠️ Heuristic fallback (no ANTHROPIC_API_KEY set)"


def _load_abom() -> pd.DataFrame:
    return pd.read_csv(DATA / "crew_cru_abom.csv")


def _load_inventory() -> pd.DataFrame:
    return pd.read_csv(DATA / "inventory.csv")


# ---------- sidebar ----------

st.sidebar.title("Club Car Inventory Intelligence")
st.sidebar.caption("Crew CRU — AI-driven buildable units POC")
st.sidebar.markdown(f"**LLM:** {_llm_status()}")

page = st.sidebar.radio(
    "Stage",
    [
        "1. Ingest (Bronze/Silver)",
        "2. AI Interpret ABOM",
        "3. SME Review",
        "4. Enumerate Configurations",
        "5. Optimize Build Plan",
        "6. NL What-If",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline state**\n\n"
    f"- ABOM rows: {len(_load_abom())}\n"
    f"- Inventory parts: {len(_load_inventory())}\n"
    f"- Configuration Model: {'✅' if 'cm' in st.session_state else '—'}\n"
    f"- Valid configs: {len(st.session_state.get('configs', []))}\n"
    f"- Build plan: {'✅' if 'plan' in st.session_state else '—'}"
)


# =====================================================================
# Stage 1 — Ingest
# =====================================================================
if page.startswith("1"):
    st.title("Stage 1 — Ingest")
    st.write(
        "Excel/CSV files land here. In production, they arrive in OCI Object Storage and "
        "are loaded into ADW Bronze tables. For the POC we read them directly."
    )
    abom = _load_abom()
    inv = _load_inventory()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Crew CRU ABOM (overloaded)")
        st.caption("Notice the multiple rows at the same Find# — these are the option groups the AI must detect.")
        st.dataframe(abom, height=520, use_container_width=True)
    with c2:
        st.subheader("Inventory on hand")
        st.dataframe(inv, height=520, use_container_width=True)


# =====================================================================
# Stage 2 — AI Interpret
# =====================================================================
elif page.startswith("2"):
    st.title("Stage 2 — AI Interpret ABOM")
    st.write(
        "The interpreter reads the overloaded ABOM and produces a structured Configuration "
        "Model. Pass 1 detects required vs. option groups deterministically by Find# "
        "collision. Pass 2 reads the Notes column with Claude and formalises any "
        "compatibility constraints. Every extracted item carries provenance and confidence."
    )

    if st.button("Run interpreter", type="primary"):
        with st.spinner("Interpreting ABOM…"):
            abom = _load_abom()
            cm = interpret_abom(abom, vehicle_id="CREW-CRU")
            st.session_state["cm"] = cm
            st.session_state["cm_original"] = cm.model_copy(deep=True)
        st.success(f"Extracted {len(cm.required_groups)} required groups, "
                   f"{len(cm.option_groups)} option groups, {len(cm.constraints)} constraints.")

    if "cm" not in st.session_state:
        st.info("Click **Run interpreter** to extract the Configuration Model.")
    else:
        cm: ConfigurationModel = st.session_state["cm"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Required groups ({len(cm.required_groups)})")
            st.dataframe(
                pd.DataFrame([
                    {"group": g.group_id, "find#": g.find_num, "parts": ", ".join(g.parts),
                     "qty/veh": g.qty_per_vehicle, "confidence": g.confidence}
                    for g in cm.required_groups
                ]),
                use_container_width=True,
            )
            st.subheader(f"Option groups ({len(cm.option_groups)})")
            st.dataframe(
                pd.DataFrame([
                    {"group": g.group_id, "find#": g.find_num, "choices": ", ".join(g.choices),
                     "select": g.select, "qty/veh": g.qty_per_vehicle, "confidence": g.confidence}
                    for g in cm.option_groups
                ]),
                use_container_width=True,
            )
        with c2:
            st.subheader(f"Constraints ({len(cm.constraints)})")
            rows = []
            for c in cm.constraints:
                if c.type == "implies":
                    rule = f"if {c.if_.choice} then {c.then.choice}"
                elif c.type in ("excludes", "forbidden_with"):
                    rule = f"{c.if_.choice} ⊗ {c.excluded.choice}"
                else:
                    rule = c.type
                rows.append({
                    "type": c.type,
                    "rule": rule,
                    "source_text": c.provenance.source_text,
                    "row#": c.provenance.abom_rows,
                    "confidence": c.confidence,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("Configuration Model (JSON)")
            with st.expander("Show JSON"):
                st.code(json.dumps(cm.model_dump(by_alias=True), indent=2, default=str), language="json")

        st.caption(f"Interpreter meta: {cm.interpreter_meta}")


# =====================================================================
# Stage 3 — SME Review
# =====================================================================
elif page.startswith("3"):
    st.title("Stage 3 — SME Review")
    st.write(
        "An SME approves, edits, or rejects each AI-extracted item. Approved items flow "
        "downstream. This is what makes the AI output honest — every decision is reviewable."
    )

    if "cm" not in st.session_state:
        st.warning("Run Stage 2 first.")
        st.stop()

    cm: ConfigurationModel = st.session_state["cm"]

    st.subheader("Option groups")
    for i, g in enumerate(cm.option_groups):
        with st.expander(f"{g.group_id}  (find# {g.find_num}, conf {g.confidence:.2f})  •  status: {g.sme_status}"):
            st.write(f"Choices: `{g.choices}`  •  qty/veh: {g.qty_per_vehicle}  •  select: {g.select}")
            st.write(f"Source rows: {g.provenance.abom_rows}")
            cols = st.columns(3)
            if cols[0].button(f"Approve", key=f"og_app_{i}"):
                g.sme_status = "approved"
            if cols[1].button(f"Reject", key=f"og_rej_{i}"):
                g.sme_status = "rejected"
            if cols[2].button(f"Reset", key=f"og_res_{i}"):
                g.sme_status = "pending"

    st.subheader("Constraints")
    if not cm.constraints:
        st.info("No constraints extracted.")
    for i, c in enumerate(cm.constraints):
        if c.type == "implies":
            rule = f"if **{c.if_.choice}** then **{c.then.choice}**"
        elif c.type in ("excludes", "forbidden_with"):
            rule = f"**{c.if_.choice}** ⊗ **{c.excluded.choice}**"
        else:
            rule = c.type
        with st.expander(f"{c.type}  •  conf {c.confidence:.2f}  •  status: {c.sme_status}"):
            st.markdown(rule)
            st.write(f"Source phrase: _{c.provenance.source_text}_")
            st.write(f"Source row: {c.provenance.abom_rows}")
            cols = st.columns(3)
            if cols[0].button(f"Approve", key=f"con_app_{i}"):
                c.sme_status = "approved"
            if cols[1].button(f"Reject", key=f"con_rej_{i}"):
                c.sme_status = "rejected"
            if cols[2].button(f"Reset", key=f"con_res_{i}"):
                c.sme_status = "pending"

    # Filter rejected constraints out of the working model
    if st.button("Lock approved model", type="primary"):
        cm.constraints = [c for c in cm.constraints if c.sme_status != "rejected"]
        cm.option_groups = [g for g in cm.option_groups if g.sme_status != "rejected"]
        st.session_state["cm_approved"] = cm
        st.success("Approved model locked. Proceed to Stage 4.")


# =====================================================================
# Stage 4 — Enumerate
# =====================================================================
elif page.startswith("4"):
    st.title("Stage 4 — Enumerate Valid Configurations")
    st.write(
        "Using the approved Configuration Model, OR-Tools (CP-SAT) enumerates every valid "
        "configuration. Each one is a clean, fully-specified parts list."
    )

    cm = st.session_state.get("cm_approved") or st.session_state.get("cm")
    if cm is None:
        st.warning("Run Stage 2 (and ideally Stage 3) first.")
        st.stop()

    if st.button("Enumerate", type="primary"):
        with st.spinner("Solving…"):
            configs = enumerate_valid_configurations(cm)
            st.session_state["configs"] = configs

    if "configs" in st.session_state:
        configs = st.session_state["configs"]
        # raw combinatorial size for comparison
        raw = 1
        for g in cm.option_groups:
            raw *= max(1, len(g.choices))
        st.metric("Valid configurations", len(configs), delta=f"vs {raw} raw combinations (constraints removed {raw - len(configs)})")

        df = pd.DataFrame([{"config_id": c.config_id, **c.choices} for c in configs])
        st.dataframe(df, use_container_width=True, height=520)


# =====================================================================
# Stage 5 — Optimize
# =====================================================================
elif page.startswith("5"):
    st.title("Stage 5 — Optimize Build Plan")
    st.write(
        "Mixed-integer programming over the valid configurations and on-hand inventory. "
        "Output: optimal build plan, binding parts, and unlock analysis."
    )

    if "configs" not in st.session_state:
        st.warning("Run Stage 4 first.")
        st.stop()

    if st.button("Optimize", type="primary"):
        inv_df = _load_inventory()
        with st.spinner("Solving MIP…"):
            plan = optimize(
                st.session_state["configs"],
                inventory_from_df(inv_df),
                costs=costs_from_df(inv_df),
                aging_days=aging_from_df(inv_df),
            )
            st.session_state["plan"] = plan
            st.session_state["inventory"] = inventory_from_df(inv_df)
            st.session_state["costs"] = costs_from_df(inv_df)
            st.session_state["aging"] = aging_from_df(inv_df)

    if "plan" in st.session_state:
        plan = st.session_state["plan"]
        st.metric("Total buildable Crew CRU vehicles", plan.total_vehicles)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Build plan")
            df = pd.DataFrame([
                {"config_id": p.config_id, "build_qty": p.quantity, **p.choices}
                for p in plan.plan
            ])
            st.dataframe(df, use_container_width=True, height=320)

            st.subheader("Parts consumed")
            cons = pd.DataFrame(
                [{"part": p, "consumed": q} for p, q in sorted(plan.parts_consumed.items(), key=lambda x: -x[1])]
            )
            st.dataframe(cons, use_container_width=True, height=240)

        with c2:
            st.subheader("Binding constraints — what's stopping more builds")
            bnd = pd.DataFrame([b.model_dump() for b in plan.binding_constraints])
            st.dataframe(bnd, use_container_width=True)

            st.subheader("Unlock analysis")
            unl = pd.DataFrame([
                {
                    "part": u.part_number,
                    "buy_qty": u.additional_qty_needed,
                    "extra_vehicles": u.additional_vehicles_unlocked,
                    "est_cost_$": (f"${u.estimated_cost:,.0f}" if u.estimated_cost else "n/a"),
                    "vehicles_per_$": (f"{u.vehicles_per_dollar:.5f}" if u.vehicles_per_dollar else "n/a"),
                }
                for u in plan.unlock_suggestions
            ])
            st.dataframe(unl, use_container_width=True)


# =====================================================================
# Stage 6 — NL What-If
# =====================================================================
elif page.startswith("6"):
    st.title("Stage 6 — Natural-Language What-If")
    st.write(
        "Ask a what-if question in plain English. The system maps it to one of five supported "
        "intents, re-runs the optimizer, and explains the diff."
    )

    if "configs" not in st.session_state or "plan" not in st.session_state:
        st.warning("Run Stages 4 and 5 first.")
        st.stop()

    st.markdown(
        "**Try one of these:**\n\n"
        "- _What if we skip deluxe canopies this month?_\n"
        "- _What happens if we have 20 more BATTERY-48V?_\n"
        "- _Which one part should we expedite to unlock the most vehicles?_\n"
        "- _Use only steel wheels._"
    )

    q = st.text_input("Your question", value="")
    if st.button("Ask", type="primary") and q:
        cm = st.session_state.get("cm_approved") or st.session_state["cm"]
        with st.spinner("Resolving intent…"):
            intent = resolve_intent_with_claude(q, cm, st.session_state["inventory"])
        st.markdown(f"**Resolved intent:** `{intent.kind}`")
        st.json(intent.payload)

        if intent.kind == "unsupported":
            st.warning(intent.payload.get("reason", "Unsupported"))
        else:
            with st.spinner("Re-solving…"):
                result, narration = apply_intent(
                    intent,
                    st.session_state["configs"],
                    st.session_state["inventory"],
                    st.session_state.get("costs"),
                    st.session_state.get("aging"),
                )

            baseline = st.session_state["plan"].total_vehicles
            new_total = result.total_vehicles
            delta = new_total - baseline
            cols = st.columns(3)
            cols[0].metric("Baseline buildable", baseline)
            cols[1].metric("Scenario buildable", new_total, delta=delta)
            cols[2].metric("Δ vehicles", delta)

            st.markdown(f"**Narration:** {narration}")

            if result.plan:
                st.subheader("Scenario build plan")
                df = pd.DataFrame([
                    {"config_id": p.config_id, "build_qty": p.quantity, **p.choices}
                    for p in result.plan
                ])
                st.dataframe(df, use_container_width=True, height=320)

            if result.binding_constraints:
                st.subheader("Scenario binding constraints")
                st.dataframe(
                    pd.DataFrame([b.model_dump() for b in result.binding_constraints]),
                    use_container_width=True,
                )
