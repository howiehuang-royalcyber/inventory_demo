# Club Car Inventory Intelligence — POC Demo

AI-driven buildable-units optimization for the Crew CRU. Demonstrates the four
capabilities the client asked for:

1. **AI ABOM interpretation** — read an overloaded ABOM, identify required vs.
   optional components, and extract compatibility constraints from free-text
   notes — automatically.
2. **Permutation generation** — enumerate every valid configuration under the
   extracted constraints.
3. **Optimization** — pick the mix that maximises buildable vehicles against
   on-hand inventory; surface binding constraints and unlock analysis.
4. **Natural-language what-if** — re-plan dynamically from plain-English
   questions.

## Quick start

```bash
cd clubcar-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable the real LLM (Claude). Without this the demo uses a
# deterministic heuristic fallback for the constraint extractor and the NL
# intent resolver.
export ANTHROPIC_API_KEY=sk-ant-...
# Optional: pick a model (default claude-sonnet-4-6)
# export ANTHROPIC_MODEL=claude-opus-4-7

streamlit run app.py
```

Open http://localhost:8501.

## Repo layout

```
clubcar-demo/
├── app.py                      # Streamlit UI — 6 stage pages
├── data/
│   ├── crew_cru_abom.csv       # 25-row overloaded Crew CRU ABOM
│   └── inventory.csv           # on-hand inventory + cost + aging
├── src/
│   ├── models.py               # Configuration Model (pydantic) — the contract
│   ├── interpreter.py          # Two-pass ABOM interpreter (Pass 1 deterministic, Pass 2 Claude)
│   ├── enumerator.py           # OR-Tools CP-SAT permutation generator
│   ├── optimizer.py            # OR-Tools CBC MIP build-plan solver + unlock analysis
│   └── whatif.py               # NL → intent → re-solve → narrate
└── requirements.txt
```

## How the demo maps to the design document

| Design doc stage | Demo location |
|---|---|
| Stage 1 — Ingestion to Bronze | Sidebar `Stage 1 — Ingest` |
| Stage 2 — Silver normalisation | Implicit (CSV is already clean) |
| Stage 3 — ABOM Interpreter (AI) | `Stage 2 — AI Interpret ABOM` + `src/interpreter.py` |
| Stage 4 — SME Review | `Stage 3 — SME Review` |
| Stage 5 — Permutation generation | `Stage 4 — Enumerate Configurations` |
| Stage 6 — Optimisation | `Stage 5 — Optimize Build Plan` |
| Stage 7 — NL What-If (AI) | `Stage 6 — NL What-If` |
| Stage 8 — Validation & demo | The whole app is the demo |

## Demo walkthrough

1. **Stage 1 — Ingest.** Show the ABOM. Point out the three seat rows at Find#
   210 with qty 1 — that's the "overloaded" pattern. SQL alone would treat all
   three as required and over-consume seats.
2. **Stage 2 — AI Interpret.** Click *Run interpreter*. Show the extracted
   option groups (seat, wheel, canopy, dash, body, windshield), the constraints
   (canopy-deluxe → dash-deluxe; folding windshield ⊗ deluxe canopy; green body
   ⊗ red seat), and the provenance pointing back to specific ABOM rows.
3. **Stage 3 — SME Review.** Approve / reject items. Lock the model.
4. **Stage 4 — Enumerate.** Show how 216 raw combinations become a smaller set
   of valid configurations after constraints.
5. **Stage 5 — Optimize.** The MIP returns the buildable-units number, the
   binding parts (likely `DASH-STD` and `DASH-DLX`), and ranked unlock
   suggestions in vehicles and dollars.
6. **Stage 6 — NL What-If.** Try:
   - *What if we skip deluxe canopies this month?*
   - *What happens if we have 20 more BATTERY-48V?*
   - *Which one part should we expedite to unlock the most vehicles?*

## Constraints encoded in the dummy data

The ABOM `notes` column contains three rules a planner would normally read by
hand:

- `CANOPY-DLX` — *"Deluxe canopy - req DASH-DLX, clearance issue w/ folding
  windshield"* → `implies` (canopy-deluxe → dash-deluxe) **and**
  `forbidden_with` (canopy-deluxe ⊗ folding windshield).
- `BODY-GRN` — *"Mkt promo green. Not compatible w/ SEAT-RED-01."` →
  `forbidden_with` (green body ⊗ red seat).
- `WS-FOLDING` — *"Folding windshield - not valid with deluxe canopy."` →
  `forbidden_with` (folding ⊗ canopy-deluxe).

These are the constraints the AI interpreter must extract from unstructured
text. They are also exactly the kind of rule a planner has to catch by eye
today.

## What this demo deliberately is **not**

- It is not an enterprise pipeline. ADW, OCI Object Storage, IAM, networking,
  and CI/CD are documented in the design doc but not modelled here.
- It is not multi-vehicle or multi-period. Crew CRU only.
- It is not an open-ended NL interface. Five supported intents, by design.
- It is not a writeback to ERP. Outputs are tables and narrations.

These are the same boundaries the design document calls out for the POC scope.
