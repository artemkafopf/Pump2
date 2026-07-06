# `vt_freq55_exposure` — Vt frequency-exposure analysis

## Task / Research question

Characterise the *exposure* dimension of the frequency question: instead of asking
"does high-frequency mean shorter TTF?", ask "how much time did failed wells spend
above 55 Hz, and does exposure fraction predict failure timing?"
Builds scatter plots (TTF vs. share-of-time >55 Hz), histograms, and KM curves
stratified by exposure bin across combinations of infant-exclusion and mount-year cutoffs.

## Status

`[x] complete`

Last run: 2026-06-XX  
Entry point: `scripts/analyze_vt_freq55_exposure.py`  
Results: `analysis_outputs/vt_freq55_exposure/`

## Data

| Source | Access | Notes |
|--------|--------|-------|
| V03 failures xlsx | `resolve_v03_all_path()` | Vt field; mount year 2023–2026 |
| Telemetry (daily merged) | `scripts.data_utils.load_daily_merged` | Daily freq per well to compute exposure fraction |

Key derived variable: `freq_above_55hz_pct` = fraction of run days with mean freq >55 Hz.

## Methods & models

- **Scatter plot**: TTF vs. exposure fraction; coloured by mount year; failure events marked
- **KM by exposure bin**: Low (<10%), Medium (10–50%), High (>50%) exposure;
  log-rank test between bins
- **Combination sweep**: all combinations of:
  - Infant exclusion: 30, 60, 90 days
  - Mount-year cutoff: >2023, >2024, >2025
- **Failure histogram by bin**: raw count and rate by exposure group

## Key findings

- No monotonic relationship between exposure fraction and TTF at Vt
- KM curves by exposure bin overlap substantially; log-rank p-value not significant
- High-exposure (>50% time above 55 Hz) group is small (n<30) — low statistical power
- Pattern consistent with [[vt_failure]] and [[vt_60hz]]: frequency is not the primary driver

## Known limitations / open questions

- Exposure fraction computed from merged telemetry; sparse coverage for some wells
- Mount-year cutoff restricts sample size significantly (>2025 leaves n<50)
- Does not adjust for confounders (field, contractor)

## Outputs

```
analysis_outputs/vt_freq55_exposure/
  plots/          ← scatter + KM PNG per combination
  combinations/   ← per-combination CSVs
```

## How to re-run

```bash
python scripts/analyze_vt_freq55_exposure.py
```

## Related analyses

- [[vt_failure]] — classical survival analysis on the same Vt cohort
- [[vt_60hz]] — focuses on true-60Hz threshold rather than exposure fraction
- [[vt_freq_latent]] — Bayesian deconvolution of the same frequency question
