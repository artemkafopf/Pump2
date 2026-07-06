# `vt_60hz` — 60 Hz safety scenario analysis

## Task / Research question

Assess whether stable operation at 60 Hz is safe for Vt field wells.
Specifically: does the "true 60 Hz" cohort (mean freq >58 Hz) have materially
shorter run life than the normal-frequency cohort after accounting for confounders?
The decision context is a field-wide frequency policy.

## Status

`[x] complete`

Last run: 2026-06-23  
Entry point: `scripts/analyze_vt_60hz_prompt.py`, `scripts/analyze_vt_60hz_scenario.py`  
Results: `analysis_outputs/vt_60hz_final_presentation_2026_06_23/`  
Docs: `docs/vt_60hz_master_ru.md`, `docs/vt_60hz_slides_ru.pptx`

## Data

| Source | Access | Notes |
|--------|--------|-------|
| V03 failures xlsx | `resolve_v03_all_path()` | Full run history incl. Vt |
| Telemetry SQLite | `resolve_telemetry_db_path()` | Daily freq per well; used to build freq group |
| Warehouse mart | `mart__vt_freq55` | Pre-aggregated freq features, TTF |
| Tech-regime SQLite | `resolve_techregime_db_path()` | Supplementary operating params |

## Methods & models

- **Kaplan-Meier by frequency group**: Low/Normal/High/True60; log-rank tests; RMST at 365 d
- **Weibull AFT**: β and η per group; confidence intervals via bootstrap
- **Latent Weibull mixture (2-component)**: separates infant-mortality and wear-out sub-populations
- **Cox PH with covariates**: HR for True60 adjusting for field, contractor, pump type
- **CatBoost classifier** (optional scenario): feature importance for early failure prediction
- **Scenario decomposition**: sensitivity to freq threshold (55 / 58 Hz) and infant cutoff

## Key findings

- "True 60 Hz" cohort (mean >58 Hz, n_fail ≥ 50) does **not** show significantly shorter median
  survival vs. normal 50–55 Hz group after infant mortality exclusion
- Infant mortality (first 90 days) drives the apparent early-failure signal in High-freq group
- Weibull β ≈ 1.0–1.1 for both High and Normal groups → near-random failure (not accelerated wear)
- Stable 60 Hz operation assessed as **not harmful** based on available data
- ПЭД+КЛ joint failure is the highest-risk cause category at Vt regardless of frequency

## Known limitations / open questions

- True-60 Hz cohort size limited (n_fail ~50–60); power low for detecting moderate HR
- Telemetry coverage drops before 2022 — earlier cohorts excluded
- No direct temperature or vibration data; ageing proxies are indirect

## Outputs

```
analysis_outputs/
  vt_60hz_prompt_analysis_2026_06_22/   figures/ tables/
  vt_60hz_final_presentation_2026_06_23/ figures/ tables/
  vt_60hz_codex_rework_2026_06_22/
docs/
  vt_60hz_master_ru.md
  vt_60hz_slides_ru.pptx
  vt_60hz_safety_assessment.md
  vt_60hz_vt_decision_summary.md
```

## How to re-run

```bash
python scripts/analyze_vt_60hz_prompt.py
python scripts/analyze_vt_60hz_scenario.py
```

## Related analyses

- [[vt_failure]] — parent analysis; this is a focused follow-up on the 60 Hz sub-question
- [[vt_freq_latent]] — Bayesian re-analysis of the same frequency stratification
- [[vt_freq55_exposure]] — complementary exposure-time view
