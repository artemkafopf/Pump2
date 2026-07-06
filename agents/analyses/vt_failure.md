# `vt_failure` — VT field ESP failure survival analysis

## Task / Research question

Identify whether operating ESP units in the Vt field at high frequency (>55 Hz) leads
to shorter run life compared to normal frequency.  Characterise failure modes by cause
(КЛ, ПЭД, shaft fracture, etc.) and test confounders (chemistry, field, contractor).
Inform the decision on whether to restrict frequency at Vt.

## Status

`[x] complete`

Last run: 2026-06-18  
Entry point: `python backend/analysis/workflows/vt_failure/run.py all`  
Results: `results/vt_failure/YYYY-MM-DD/` (new); `analysis_outputs/vt_failure_analysis/` (legacy)

## Data

| Source | Access | Notes |
| --- | --- | --- |
| V03 failures xlsx | `resolve_v03_failures_path()` | 2 634 runs; failure flag + cause classification |
| Warehouse mart | `mart__vt_freq55` SQLite table | mean freq, pct days >55 Hz, TTF, features |
| Telemetry SQLite | `resolve_telemetry_db_path()` | daily freq/load for feature construction |
| Lab chemistry SQLite | `resolve_lab_db_path()` | Cl⁻, Ca²⁺, SO₄²⁻ proxy for scale |

## Methods & models

- **Phase 0 — Data audit**: coverage, censoring rate, date consistency
- **Phase 1 — Descriptive baseline**: run counts, failure rates, field breakdown
- **Phase 2 — Infant mortality sweep**: Kaplan-Meier with 45/60/90/120-day exclusion thresholds
- **Phase 3 — Kaplan-Meier by frequency group**: Low/Normal/High/Very-High curves; log-rank tests
- **Phase 4 — Weibull shape analysis**: β per field×group; infant vs. wear-out classification
- **Phase 5 — Competing risks (CIF)**: cumulative incidence per failure cause; Fine-Gray approach
- **Phase 6 — Duty & chemistry proxies**: correlation of daily salt/gypsum load with TTF
- **Phase 7 — Confounding test**: propensity-matched comparison; does freq group predict field?
- **Phase 8 — Cox proportional hazards**: HR for frequency group adjusting for field, contractor, chemistry
- **Phase 9 — Vt deep-dive**: Vt-only cohort; frequency definition sensitivity (mean vs. pct)
- **Phase 10 — CatBoost infant classifier**: predict early failure (<90 d) from mature features
- **Phase 11 — Sensitivity checks**: alternate freq thresholds, alternate infant cutoffs
- **Phase TRF — TRF capacity hypothesis**: test whether over-capacity pumps (kpod<0.7) explain failures
- **Phase FREQDEF — Frequency definition comparison**: mean_freq vs. pct_above_55Hz definitions

## Key findings

- КЛ (cable) and ПЭД (motor) show β<1 — infant / random failure; not wear-out
- High-frequency group (>55 Hz mean) does not show significantly shorter survival at Vt after
  adjusting for field and contractor (Cox HR not significant)
- TRF capacity hypothesis **rejected**: kpod<0.7 does not independently predict failure
- Infant mortality (first 90 days) accounts for ~30% of failures; dominant at Vt
- Chemistry proxies (gypsum, chloride) correlate weakly with TTF in univariate analysis

## Known limitations / open questions

- Frequency data coverage sparse for early years; some runs excluded
- CIF model relies on cause classification from V03 — subject to reclassification errors
- No direct pressure measurements for many runs (telemetry only)

## Outputs

```text
results/vt_failure/YYYY-MM-DD/
  tables/   step0_audit/, phase8_cox/, executive_summary_numbers.csv
  figures/  phase1_descriptive/ … phase11_sensitivity/
  models/   phase10_catboost/
  manifest.json

analysis_outputs/vt_failure_analysis/   ← legacy reference runs
```

## How to re-run

```bash
python analysis/vt_failure/run.py all
# Specific phases:
python analysis/vt_failure/run.py 3,4,8
```

## Related analyses

- [[vt_60hz]] — follow-up focusing specifically on true 60 Hz (>58 Hz) operation
- [[vt_freq_latent]] — Bayesian latent Weibull re-analysis of the same Vt cohort
- [[vt_freq55_exposure]] — time-in-exposure view of the same frequency question
