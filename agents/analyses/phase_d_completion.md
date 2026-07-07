# Phase D — Within-Run Dynamics / Early-Warning Layer: Completion Note

Implementation record for the spec `agents/analyses/phase_d_early_warning.md`.
The narrative deliverable is `results/phase_d_report/<date>/reports/README.md`; this
note is the code/decision trail.

## §0 decisions (user-confirmed 2026-07-07 — all defaults)

| Parameter | Value |
|---|---|
| Horizon H | 60 operating days (30/90 secondary) |
| Landmark cadence | 30 operating days |
| Capacity | top-5% / top-10% |
| Guard gap g | 7 op-days (sensitivity {3,7,14}); `stop_date` = pull |
| Eligibility | ≥30 valid telemetry days before first landmark |
| Scope | all-cause primary; hydraulic & electro-thermal secondary |

## What was built

**Reusable logic (`backend/analysis/`)**
- `data/landmark_features.py` — operating-day mapping (qliq>0), leakage-proof
  trailing-window feature builder (14 & 30 op-day windows), per-(H×g) guard-gapped
  targets, warehouse frame builder, latest-frame loader. Pure feature/target
  functions are unit-testable without the DB.
- `models/survival/landmark_cox.py` — install-cohort & landmark-calendar splits;
  episode-split builder + `CoxTimeVaryingFitter` fit (D2); landmark super-model Cox
  fit/predict (D3); age+stratum conditional-KM baseline (the null to beat).
- `models/survival/alarm_metrics.py` — precision/recall@capacity, lead-time,
  false-alarms/pump-year, dynamic AUC, well-cluster bootstrap of a metric delta.

**Scripts (`scripts/run/phase_d_*.py`)** — thin: `landmarks` (D0), `precursors` (D1),
`tv_cox` (D2), `landmark_models` (D3/D4), `watchlist` (D5), `report` (D6).

**Tests (34, all pass)** — `test_landmark_features.py` (truncation-invariance rule 1,
guard-gap rule 2, per-feature correctness, frame builder), `test_landmark_cox.py`
(split/episode/baseline), `test_alarm_metrics.py` (the verdict metrics).
`cd backend && python -m pytest tests/ -q` = 168 passed; the 4 pre-existing
`analysis.plotting`/seaborn collection errors are untouched.

## Headline result — an honest null (the C5 lesson, one layer up)

- Landmark frame: 13,328 landmarks over 1,610 eligible runs; primary H=60/g=7 →
  1,756 fail-in-window positives, 10,058 negatives.
- **D4 verdict (install-cohort split, test = installs after 2023-12-31):** dynamic
  (age+stratum+dynamics) AUC **0.470** vs baseline (age+stratum) **0.502**, delta
  **−0.032 [−0.078, +0.014]** — CI includes 0 at every H×g. Within-run dynamics do
  **not** beat age+stratum out of sample. At top-10% capacity the dynamic model
  catches 9% of failures vs 11% for the baseline.
- D2 shows real *in-sample* signal (restarts, load variability, declining qliq slope
  move the hazard; restarts lean electro-thermal, diffuse) — but per the
  prediction-first discipline this earns no conclusion without an out-of-sample delta,
  and there is none. Explanation ≠ prediction, exactly as for Phase C's θ.
- D5 therefore took the **honest-null branch**: no watchlist shipped; instead a
  telemetry-improvement recommendation (`phase_d_watchlist/.../reports/`).

## Notable data-quality finding

The qliq operating-day clock and `ttf_true_best_days` disagree by >10% on **701/2,046**
runs (~a third). The operating-time definition needs reconciling before any within-run
feature can be trusted — a ceiling on the whole layer, carried forward as an open question.

## Deviations / caveats (stated in outputs)

- `CoxTimeVaryingFitter` has no cluster-robust option in this lifelines version, so D2
  SEs are model-based (anti-conservative) — read as screening signals, not confirmatory;
  the verdict lives in D4. D2 episode clustering is run-level (`row_id`) because two runs
  of one well can overlap in operating age (forbidden in a counting-process model).
- `days_since_restart` is only 13–19% covered (most windows have no restart) → excluded
  from the models, kept in the D1 atlas; it is the most under-recorded precursor.
