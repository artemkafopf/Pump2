# `vt_freq_latent` — Bayesian latent Weibull frequency stratification (Vt)

## Task / Research question

Re-analyse the Vt frequency-group question using a Bayesian 2-component latent Weibull
mixture that explicitly separates infant-mortality and wear-out sub-populations.
Test whether the observed frequency effect survives after deconvolving the two failure
mechanisms. Compare mean-freq and pct-days-above-55Hz stratification definitions.

## Status

`[x] complete`

Last run: 2026-06-23  
Entry point: `scripts/analyze_vt_freq_latent.py`  
Results: `analysis_outputs/vt_freq_latent_2026_06_23/`

## Data

| Source | Access | Notes |
|--------|--------|-------|
| Warehouse mart | `mart__vt_freq55` (SQLite) | Vt-only; `ttf_true_best_days`, `event`, freq features |

Key columns used: `row_id`, `ttf_true_best_days`, `event`, `freq_w_mean`,
`freq_above_55hz_pct`, `freq_signed_exposure`, `n_freq_valid_days`.

## Methods & models

- **Bayesian 2-component latent Weibull mixture** (`analysis.bayesian_latent_weibull`):
  MCMC sampling (n_iter=12 000, burn_in=3 000, thin=5, 3 chains);
  decomposes observed TTF distribution into infant-mortality (β<1) and wear-out (β>1) components
- **Kaplan-Meier** (lifelines): empirical survival curve per frequency group for comparison
- **Mann-Whitney U test**: non-parametric group comparison on TTF
- **Two stratification definitions tested**:
  - `freq_w_mean > 55 Hz` (mean frequency)
  - `freq_above_55hz_pct > 0.5` (majority of run time above 55 Hz)

MCMC configuration:
```
n_components=2, mu_log_eta=5.0, sigma_log_eta=1.5,
mu_log_beta=0.0, sigma_log_beta=0.7, seed=42
```

## Key findings

- Both stratification definitions yield similar posterior estimates
- Infant-mortality component dominates (weight ~0.3–0.4) independent of frequency group
- Wear-out component β₂ ≈ 1.3–1.5 in both Low and High frequency groups — no material difference
- Frequency group does not significantly shift either component's scale (η)
- Result consistent with classical KM and Cox analysis in [[vt_failure]]

## Known limitations / open questions

- MCMC convergence sensitive to initialisation; recommend running 3+ chains and checking R-hat
- n_freq_valid_days filter (≥10 days) excludes ~15% of runs with sparse telemetry
- 2-component mixture may be insufficient for some fields with trimodal TTF distributions

## Outputs

```
analysis_outputs/vt_freq_latent_2026_06_23/
  ← KM curves, posterior density plots, component parameter tables
```

## How to re-run

```bash
python scripts/analyze_vt_freq_latent.py
```

## Related analyses

- [[vt_failure]] — classical analysis of the same question
- [[vt_60hz]] — 60 Hz specific sub-analysis
- [[bayesian_field_survival]] — same Bayesian model applied across fields (Global/Ya/Vt)
