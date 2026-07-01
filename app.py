"""Club Car Inventory Intelligence — Streamlit demo.

Run:
  streamlit run app.py

Set ANTHROPIC_API_KEY in env to enable the real LLM interpreter and NL layer.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Load .env for local development. On Streamlit Cloud, secrets are injected as
# env vars automatically, so this is a no-op there.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except (FileNotFoundError, KeyError):
    pass

from src.abom_loader import _apply_curated_splits, load_real_abom, simplify_for_demo
from src.enumerator import enumerate_valid_configurations
from src.interpreter import interpret_real_abom
from src.models import ConfigurationModel
from src.optimizer import optimize_multi_sku
from src.whatif import apply_intent_multi_sku, resolve_intent_with_claude


DATA = Path("data")
ABOM_PATH = DATA / "Copy of 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx"
INVENTORY_PATH = DATA / "inventory_synthetic.csv"

SKU_LABELS = {
    "AGM": "47773464001 — CRU Lounge, Electric AGM",
    "LI": "47787623001 — CRU Lounge, Lithium",
}

st.set_page_config(
    page_title="Club Car Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- cached data loaders ----------

@st.cache_data(show_spinner=False)
def _load_abom_df() -> pd.DataFrame:
    return load_real_abom(str(ABOM_PATH))


@st.cache_data(show_spinner=False)
def _load_inventory_df() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_PATH)


def _inventory_dict() -> dict[str, int]:
    df = _load_inventory_df()
    return {str(r.part_number): int(r.on_hand) for r in df.itertuples()}


def _costs_dict() -> dict[str, float]:
    df = _load_inventory_df()
    return {str(r.part_number): float(r.unit_cost) for r in df.itertuples()}


def _aging_dict() -> dict[str, int]:
    df = _load_inventory_df()
    return {str(r.part_number): int(r.aging_days) for r in df.itertuples()}


def _description_for(part: str) -> str:
    df = _load_inventory_df()
    hit = df[df["part_number"].astype(str) == str(part)]
    return str(hit.iloc[0]["description"]) if len(hit) else ""


# ---------- helpers ----------

def _llm_status() -> str:
    return "Claude (real LLM)" if os.environ.get("ANTHROPIC_API_KEY") else "Heuristic fallback (no ANTHROPIC_API_KEY set)"


def _build_plan_workbook(multi: dict, cm_by_sku: dict, inventory: dict, mix: dict, scenario_label: str = "Baseline") -> bytes:
    """Render a multi-SKU build-plan result to a multi-sheet xlsx workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary = pd.DataFrame([
            {"Metric": "Scenario", "Value": scenario_label},
            {"Metric": "Exported at", "Value": datetime.now().isoformat(timespec="seconds")},
            {"Metric": "AGM mix target", "Value": f"{mix.get('AGM', 0):.0%}"},
            {"Metric": "Lithium mix target", "Value": f"{mix.get('LI', 0):.0%}"},
            {"Metric": "Total buildable vehicles", "Value": multi["total_vehicles"]},
            {"Metric": "AGM (47773464001)", "Value": multi["by_sku"].get("AGM", 0)},
            {"Metric": "Lithium (47787623001)", "Value": multi["by_sku"].get("LI", 0)},
            {"Metric": "Binding parts", "Value": len(multi["binding_constraints"])},
        ])
        summary.to_excel(xw, sheet_name="Summary", index=False)

        rows = []
        for sku, lines in multi["plan_by_sku"].items():
            for p in lines:
                rows.append({"sku": sku, "config_id": p.config_id, "build_qty": p.quantity, **p.choices})
        pd.DataFrame(rows or [{}]).to_excel(xw, sheet_name="Build Plan", index=False)

        parts_rows = []
        for part, used in sorted(multi["parts_consumed"].items(), key=lambda x: -x[1]):
            on_hand = inventory.get(part, 0)
            parts_rows.append({
                "part_number": part, "description": _description_for(part),
                "consumed": used, "on_hand": on_hand,
                "remaining": max(0, on_hand - used),
                "is_binding": used == on_hand and on_hand > 0,
            })
        pd.DataFrame(parts_rows).to_excel(xw, sheet_name="Parts Consumed", index=False)

        bnd = pd.DataFrame([b.model_dump() for b in multi["binding_constraints"]]) if multi["binding_constraints"] else pd.DataFrame()
        bnd.to_excel(xw, sheet_name="Binding Constraints", index=False)

        unl = pd.DataFrame([{
            "part_number": u.part_number, "description": _description_for(u.part_number),
            "buy_qty": u.additional_qty_needed,
            "extra_vehicles_unlocked": u.additional_vehicles_unlocked,
            "estimated_cost_usd": u.estimated_cost,
            "vehicles_per_dollar": u.vehicles_per_dollar,
        } for u in multi["unlock_suggestions"]]) if multi["unlock_suggestions"] else pd.DataFrame()
        unl.to_excel(xw, sheet_name="Unlock Analysis", index=False)

    return buf.getvalue()


def _xlsx_filename(prefix: str = "cru_build_plan") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"


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
abom_df_meta = _load_abom_df()
inv_df_meta = _load_inventory_df()
st.sidebar.markdown(
    "**Pipeline state**\n\n"
    f"- ABOM parts: {len(abom_df_meta)}\n"
    f"- Inventory parts: {len(inv_df_meta)}\n"
    f"- Interpreter: {'✅' if 'cm_by_sku' in st.session_state else '—'}\n"
    f"- Configs by SKU: {'✅' if 'configs_by_sku' in st.session_state else '—'}\n"
    f"- Build plan: {'✅' if 'multi_plan' in st.session_state else '—'}"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**POC note:** ABOM is the real customer file "
    "(47773463001). Inventory and on-hand quantities are synthetic — "
    "Club Car maintains real stock in Infor XA (warehouses A & P), which "
    "we would extract for production."
)


# =====================================================================
# Stage 1 — Ingest
# =====================================================================
if page.startswith("1"):
    st.title("Stage 1 — Ingest")
    st.write(
        "Excel/CSV files land here. In production they arrive in OCI Object Storage and "
        "are loaded into ADW Bronze tables. For the POC we read them directly from the "
        "customer's real ABOM file and a synthetic inventory snapshot."
    )

    abom_df = _load_abom_df()
    inv_df = _load_inventory_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("ABOM parts (cleaned)", len(abom_df))
    c2.metric("Applicable to AGM", int((abom_df["agm_code"].isin(["DR", "X", "B"])).sum()))
    c3.metric("Applicable to Lithium", int((abom_df["li_code"].isin(["DR", "X", "B"])).sum()))

    st.subheader("ABOM — `Copy of 47773463001 ABOM, CRU, LOUNGE, ELEC.xlsx`")
    st.caption(
        "Real customer ABOM. Notice the AGM and LI columns: `DR` = default required, "
        "`X` = optional choice, `B` = rule-based. The free-text **Rules** column is what the "
        "AI interpreter reads in Stage 2."
    )
    st.dataframe(
        abom_df[["excel_row", "part_number", "description", "agm_code", "li_code", "rules_text", "section"]],
        height=420, width='stretch',
    )

    st.subheader("Inventory on hand (synthetic for POC)")
    st.caption("Warehouses A & P. Real data lives in Infor XA; this snapshot is engineered to surface a believable LI-shift bottleneck.")
    st.dataframe(inv_df, height=380, width='stretch')


# =====================================================================
# Stage 2 — AI Interpret
# =====================================================================
elif page.startswith("2"):
    st.title("Stage 2 — AI Interpret ABOM")
    st.write(
        "The interpreter reads the overloaded ABOM and produces a structured Configuration Model "
        "**per SKU** (AGM and Lithium share one ABOM but use different rows). Pass 1 groups parts "
        "into required vs. option groups. Pass 2 reads the free-text Rules column with Claude to "
        "extract compatibility constraints. Every extracted item carries provenance (the actual "
        "spreadsheet row) and a confidence score."
    )

    if st.button("Run interpreter for both SKUs", type="primary"):
        with st.spinner("Interpreting ABOM for AGM and Lithium…"):
            df = _load_abom_df()
            use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
            cm_by_sku: dict[str, ConfigurationModel] = {}
            for sku in ("AGM", "LI"):
                cm = interpret_real_abom(df, sku, use_llm=use_llm)
                cm = _apply_curated_splits(cm)  # SME-grade fix for known mis-groupings
                cm_by_sku[sku] = cm
            st.session_state["cm_by_sku"] = cm_by_sku
        agm_cm = cm_by_sku["AGM"]
        li_cm = cm_by_sku["LI"]
        st.success(
            f"AGM: {len(agm_cm.option_groups)} option groups, "
            f"{len(agm_cm.required_groups)} required, {len(agm_cm.constraints)} constraints.   "
            f"·   Lithium: {len(li_cm.option_groups)} option groups, "
            f"{len(li_cm.required_groups)} required, {len(li_cm.constraints)} constraints."
        )

    if "cm_by_sku" not in st.session_state:
        st.info("Click **Run interpreter for both SKUs** to extract the Configuration Models.")
        st.stop()

    cm_by_sku: dict[str, ConfigurationModel] = st.session_state["cm_by_sku"]

    tabs = st.tabs([SKU_LABELS["AGM"], SKU_LABELS["LI"]])
    for tab, sku in zip(tabs, ("AGM", "LI")):
        cm = cm_by_sku[sku]
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Option groups ({len(cm.option_groups)})")
                st.dataframe(pd.DataFrame([
                    {"group": g.group_id, "choices": ", ".join(g.choices),
                     "n_choices": len(g.choices), "select": g.select,
                     "qty/veh": g.qty_per_vehicle, "confidence": g.confidence}
                    for g in cm.option_groups
                ]), width='stretch', height=320)

                st.subheader(f"Required groups ({len(cm.required_groups)})")
                st.dataframe(pd.DataFrame([
                    {"group": g.group_id, "parts": ", ".join(g.parts),
                     "qty/veh": g.qty_per_vehicle, "excel_row": g.provenance.abom_rows}
                    for g in cm.required_groups
                ]), width='stretch', height=240)

            with c2:
                st.subheader(f"Constraints ({len(cm.constraints)})")
                con_rows = []
                for c in cm.constraints:
                    if c.type == "implies" and c.then:
                        rule = f"{c.if_.choice} → {c.then.choice}"
                    elif c.type in ("excludes", "forbidden_with") and c.excluded:
                        rule = f"{c.if_.choice} ⊗ {c.excluded.choice}"
                    elif c.type == "requires_one_of":
                        rule = f"{c.if_.choice} requires one of …"
                    else:
                        rule = c.type
                    con_rows.append({
                        "type": c.type, "rule": rule,
                        "source_phrase": (c.provenance.source_text or "")[:80],
                        "excel_row": c.provenance.abom_rows,
                        "confidence": c.confidence,
                    })
                st.dataframe(pd.DataFrame(con_rows), width='stretch', height=420)

                st.subheader("Interpreter meta")
                st.json(cm.interpreter_meta)


# =====================================================================
# Stage 3 — SME Review
# =====================================================================
elif page.startswith("3"):
    st.title("Stage 3 — SME Review")
    st.write(
        "An SME approves, edits, or rejects each AI-extracted item. Approved items flow downstream. "
        "This is what makes the AI output honest — every decision is reviewable and the provenance "
        "links back to the exact ABOM row."
    )

    if "cm_by_sku" not in st.session_state:
        st.warning("Run Stage 2 first.")
        st.stop()

    cm_by_sku: dict[str, ConfigurationModel] = st.session_state["cm_by_sku"]
    sku = st.selectbox(
        "SKU to review", options=["AGM", "LI"],
        format_func=lambda s: SKU_LABELS[s],
    )
    cm = cm_by_sku[sku]

    st.info(
        "Below: each AI extraction with its provenance and a confidence score. "
        "Low-confidence items (e.g. the mirror↔body color rule) are exactly where SME review earns its keep."
    )

    st.subheader("Option groups")
    for i, g in enumerate(cm.option_groups):
        with st.expander(f"{g.group_id}  ·  {len(g.choices)} choices  ·  conf {g.confidence:.2f}  ·  status: {g.sme_status}"):
            st.write(f"Choices: `{g.choices}`")
            st.write(f"qty/vehicle: {g.qty_per_vehicle}  ·  select: {g.select}")
            st.write(f"Source ABOM rows: {g.provenance.abom_rows}")
            if g.provenance.source_text:
                st.caption(f"_{g.provenance.source_text}_")
            cols = st.columns(3)
            if cols[0].button("Approve", key=f"og_app_{sku}_{i}"):
                g.sme_status = "approved"
            if cols[1].button("Reject", key=f"og_rej_{sku}_{i}"):
                g.sme_status = "rejected"
            if cols[2].button("Reset", key=f"og_res_{sku}_{i}"):
                g.sme_status = "pending"

    st.subheader("Constraints")
    if not cm.constraints:
        st.info("No constraints extracted.")
    for i, c in enumerate(cm.constraints):
        if c.type == "implies" and c.then:
            rule = f"if **{c.if_.choice}** then **{c.then.choice}**"
        elif c.type in ("excludes", "forbidden_with") and c.excluded:
            rule = f"**{c.if_.choice}** ⊗ **{c.excluded.choice}**"
        elif c.type == "requires_one_of":
            rule = f"**{c.if_.choice}** requires one of (SME to specify)"
        else:
            rule = c.type
        flag = "⚠️ low conf" if c.confidence < 0.5 else ""
        with st.expander(f"{c.type} · conf {c.confidence:.2f} {flag} · row {c.provenance.abom_rows} · {c.sme_status}"):
            st.markdown(rule)
            if c.provenance.source_text:
                st.caption(f'Source: _"{c.provenance.source_text}"_')
            cols = st.columns(3)
            if cols[0].button("Approve", key=f"con_app_{sku}_{i}"):
                c.sme_status = "approved"
            if cols[1].button("Reject", key=f"con_rej_{sku}_{i}"):
                c.sme_status = "rejected"
            if cols[2].button("Reset", key=f"con_res_{sku}_{i}"):
                c.sme_status = "pending"

    if st.button("Lock approved model", type="primary"):
        cm.constraints = [c for c in cm.constraints if c.sme_status != "rejected"]
        cm.option_groups = [g for g in cm.option_groups if g.sme_status != "rejected"]
        st.session_state.setdefault("cm_approved_by_sku", {})[sku] = cm
        st.success(f"Approved model for {SKU_LABELS[sku]} locked. Proceed to Stage 4.")


# =====================================================================
# Stage 4 — Enumerate
# =====================================================================
elif page.startswith("4"):
    st.title("Stage 4 — Enumerate Valid Configurations")
    st.write(
        "Using each approved Configuration Model, OR-Tools (CP-SAT) enumerates valid configurations "
        "per SKU. The raw cartesian space across the real ABOM is ~33 billion combinations; large "
        "non-meaningful option groups (programming features, feature-options) are auto-collapsed to "
        "their defaults so we focus on the variants that actually drive supply (battery, mirror, "
        "windshield, roof, etc.)."
    )

    if "cm_by_sku" not in st.session_state:
        st.warning("Run Stage 2 first.")
        st.stop()

    # Prefer SME-approved model where available; fall back to raw interpreter
    # output for any SKU the user didn't explicitly lock in Stage 3.
    approved = st.session_state.get("cm_approved_by_sku", {})
    cm_by_sku: dict[str, ConfigurationModel] = {
        sku: approved.get(sku, cm) for sku, cm in st.session_state["cm_by_sku"].items()
    }

    if st.button("Enumerate both SKUs", type="primary"):
        with st.spinner("Solving CP-SAT…"):
            configs_by_sku: dict[str, list] = {}
            simplified_by_sku: dict[str, ConfigurationModel] = {}
            for sku, cm in cm_by_sku.items():
                cm_s = simplify_for_demo(cm, max_choices=10)
                configs_by_sku[sku] = enumerate_valid_configurations(cm_s, max_solutions=5000)
                simplified_by_sku[sku] = cm_s
            st.session_state["configs_by_sku"] = configs_by_sku
            st.session_state["simplified_by_sku"] = simplified_by_sku

    if "configs_by_sku" not in st.session_state:
        st.info("Click **Enumerate both SKUs** to materialize configurations.")
        st.stop()

    configs_by_sku: dict[str, list] = st.session_state["configs_by_sku"]
    simplified_by_sku: dict[str, ConfigurationModel] = st.session_state["simplified_by_sku"]

    cols = st.columns(2)
    for col, sku in zip(cols, ("AGM", "LI")):
        cs = configs_by_sku[sku]
        cm_s = simplified_by_sku[sku]
        raw = 1
        for g in cm_s.option_groups:
            raw *= max(1, len(g.choices))
        with col:
            st.subheader(SKU_LABELS[sku])
            c1, c2 = st.columns(2)
            c1.metric("Valid configurations", len(cs))
            c2.metric("vs raw cartesian", f"{raw:,}")
            df_cfg = pd.DataFrame([{"config_id": c.config_id, **c.choices} for c in cs[:500]])
            st.dataframe(df_cfg, width='stretch', height=420)
            if len(cs) > 500:
                st.caption(f"Showing first 500 of {len(cs)} (capped for display).")


# =====================================================================
# Stage 5 — Optimize
# =====================================================================
elif page.startswith("5"):
    st.title("Stage 5 — Optimize Build Plan")
    st.write(
        "Mixed-integer optimization across both SKUs sharing one inventory pool. Customer business "
        "context: the **mix is shifting fast to Lithium** (2025 shipments: 208 AGM / 142 LI; 2026 YTD: "
        "12 AGM / 124 LI). The slider lets you re-plan against a target Lithium share. "
        "Output: optimal build plan, binding parts, and ranked unlock suggestions in vehicles and dollars."
    )

    if "configs_by_sku" not in st.session_state:
        st.warning("Run Stage 4 first.")
        st.stop()

    configs_by_sku = st.session_state["configs_by_sku"]
    inv = _inventory_dict()
    costs = _costs_dict()
    aging = _aging_dict()

    li_pct = st.slider(
        "Lithium mix target (%) — AGM gets the remainder",
        min_value=0, max_value=100, value=65, step=5,
        help="Default 65% reflects the 2025–26 actuals leaning lithium.",
    )
    mix = {"AGM": 1.0 - li_pct / 100.0, "LI": li_pct / 100.0}

    @st.cache_data(show_spinner=False)
    def _solve(li_pct_key: int):
        return optimize_multi_sku(configs_by_sku, inv, costs=costs, aging_days=aging, mix=mix)

    with st.spinner("Solving MIP…"):
        multi = _solve(li_pct)

    st.session_state["multi_plan"] = multi
    st.session_state["multi_mix"] = mix
    st.session_state["multi_inventory"] = inv
    st.session_state["multi_costs"] = costs
    st.session_state["multi_aging"] = aging

    total = multi["total_vehicles"]
    by_sku = multi["by_sku"]
    binding = multi["binding_constraints"]
    if binding:
        top = binding[0]
        desc = _description_for(top.part_number) or top.part_number
        binding_phrase = f"**{desc.strip()}** (`{top.part_number}`)"
    else:
        binding_phrase = "no binding part"

    st.markdown(
        f"### At a **{li_pct}% Lithium** mix, you can build **{total}** vehicles "
        f"before hitting {binding_phrase}."
    )

    cols = st.columns(3)
    cols[0].metric("Total buildable", total)
    cols[1].metric(f"AGM (47773464001)", by_sku.get("AGM", 0))
    cols[2].metric(f"Lithium (47787623001)", by_sku.get("LI", 0))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Build plan by SKU")
        rows = []
        for sku, lines in multi["plan_by_sku"].items():
            for p in lines:
                rows.append({"sku": sku, "config_id": p.config_id, "build_qty": p.quantity, **p.choices})
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', height=360)
        else:
            st.info("No buildable plan at this mix.")

        st.subheader("Parts consumed (top 30)")
        cons = pd.DataFrame([
            {"part": p, "description": _description_for(p), "consumed": q, "on_hand": inv.get(p, 0)}
            for p, q in sorted(multi["parts_consumed"].items(), key=lambda x: -x[1])
        ][:30])
        st.dataframe(cons, width='stretch', height=320)

    with c2:
        st.subheader("Binding constraints — what's stopping more builds")
        if binding:
            bnd_df = pd.DataFrame([
                {**b.model_dump(), "description": _description_for(b.part_number)}
                for b in binding
            ])
            st.dataframe(bnd_df, width='stretch', height=300)
        else:
            st.info("No binding constraint.")

        st.subheader("Unlock analysis — best next $ to spend")
        unl_rows = [
            {
                "part": u.part_number,
                "description": _description_for(u.part_number),
                "buy_qty": u.additional_qty_needed,
                "extra_vehicles": u.additional_vehicles_unlocked,
                "est_cost_$": (f"${u.estimated_cost:,.0f}" if u.estimated_cost else "n/a"),
                "vehicles_per_$": (f"{u.vehicles_per_dollar:.6f}" if u.vehicles_per_dollar else "n/a"),
            }
            for u in multi["unlock_suggestions"]
        ]
        if unl_rows:
            st.dataframe(pd.DataFrame(unl_rows), width='stretch', height=380)
        else:
            st.info("No unlock suggestions (no binding parts to bump).")

    # Excel export
    xlsx_bytes = _build_plan_workbook(multi, st.session_state.get("cm_by_sku", {}), inv, mix, scenario_label=f"{li_pct}% Lithium mix")
    st.download_button(
        label="⬇️ Export build plan to Excel",
        data=xlsx_bytes,
        file_name=_xlsx_filename(),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =====================================================================
# Stage 6 — NL What-If
# =====================================================================
elif page.startswith("6"):
    st.title("Stage 6 — Natural-Language What-If")
    st.write(
        "Ask a what-if question in plain English. Claude maps it to a structured intent, the optimizer "
        "re-solves, and the diff is narrated."
    )

    if "configs_by_sku" not in st.session_state or "multi_plan" not in st.session_state:
        st.warning("Run Stages 4 and 5 first.")
        st.stop()

    st.markdown(
        "**Try one of these:**\n\n"
        "- _What if we drop AGM and go all-lithium this month?_\n"
        "- _What if we shift mix to 80% lithium?_\n"
        "- _Which one part should we expedite to unlock the most lithium builds?_\n"
        "- _What if our 16kWh battery shipment is two weeks late?_"
    )

    q = st.text_input("Your question", value="")
    if st.button("Ask", type="primary") and q:
        cm_for_intent = next(iter(st.session_state["cm_by_sku"].values()))
        inv = st.session_state["multi_inventory"]
        with st.spinner("Resolving intent…"):
            intent = resolve_intent_with_claude(q, cm_for_intent, inv)
        st.markdown(f"**Resolved intent:** `{intent.kind}`")
        st.json(intent.payload)

        if intent.kind == "unsupported":
            st.warning(intent.payload.get("reason", "Unsupported"))
            st.stop()

        with st.spinner("Re-solving multi-SKU…"):
            multi_result, narration = apply_intent_multi_sku(
                intent,
                st.session_state["configs_by_sku"],
                st.session_state["multi_inventory"],
                st.session_state.get("multi_costs"),
                st.session_state.get("multi_aging"),
                baseline_mix=st.session_state.get("multi_mix"),
            )

        baseline = st.session_state["multi_plan"]["total_vehicles"]
        new_total = multi_result["total_vehicles"]
        delta = new_total - baseline
        cols = st.columns(4)
        cols[0].metric("Baseline buildable", baseline)
        cols[1].metric("Scenario total", new_total, delta=delta)
        cols[2].metric("AGM", multi_result["by_sku"].get("AGM", 0))
        cols[3].metric("Lithium", multi_result["by_sku"].get("LI", 0))
        st.markdown(f"**Narration:** {narration}")

        rows = []
        for sku, lines in multi_result["plan_by_sku"].items():
            for p in lines:
                rows.append({"sku": sku, "config_id": p.config_id, "build_qty": p.quantity, **p.choices})
        if rows:
            st.subheader("Scenario build plan")
            st.dataframe(pd.DataFrame(rows), width='stretch', height=320)

        if multi_result["binding_constraints"]:
            st.subheader("Scenario binding constraints")
            st.dataframe(
                pd.DataFrame([
                    {**b.model_dump(), "description": _description_for(b.part_number)}
                    for b in multi_result["binding_constraints"]
                ]),
                width='stretch',
            )

        if multi_result["unlock_suggestions"]:
            st.subheader("Scenario unlock analysis")
            st.dataframe(
                pd.DataFrame([
                    {
                        "part": u.part_number,
                        "description": _description_for(u.part_number),
                        "buy_qty": u.additional_qty_needed,
                        "extra_vehicles": u.additional_vehicles_unlocked,
                        "est_cost_$": (f"${u.estimated_cost:,.0f}" if u.estimated_cost else "n/a"),
                    }
                    for u in multi_result["unlock_suggestions"]
                ]),
                width='stretch',
            )
