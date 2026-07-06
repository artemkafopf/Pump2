# `bayesian_field_survival` — Bayesian latent Weibull: Global vs. Ya vs. Vt

## Task / Research question

Compare survival distributions across three cohorts — Global (all fields), Ya field,
and Vt field — using the Bayesian latent Weibull mixture to account for the
infant-mortality / wear-out bimodality.  Determine whether Vt's observed shorter
median TTF reflects a genuinely different failure regime or just a higher infant
fraction.

## Status

`[x] complete`

Last run: 2026-06-23  
Entry point: `scripts/analyze_field_survival_bayesian.py`  
Results: `analysis_outputs/bayesian_field_survival_compare_2026_06_23/{global,vt,ya}/`

## Data

| Source | Access | Notes |
|--------|--------|-------|
| V03 failures xlsx | `resolve_v03_all_path()` | Full run history; `Месторождение`, `Наработка (сут)`, `Failure Flag` |
| `mart__vt_freq55` | via `analyze_kpod_window_thresholds.load_runs` | Adds kpod and feature columns |

Cohort sizes (approximate): Global ~2 600 runs, Ya ~400, Vt ~700.

## Methods & models

- **Bayesian latent Weibull mixture** (`analysis.BayesianLatentWeibullFitResult`,
  `fit_bayesian_latent_weibull`): 2-component; MCMC per field cohort
- **Unknown-K fitting sweep** (k_max=4): optionally fits 2, 3, 4 components; selects
  by WAIC / BIC
- **Posterior comparison**: overlay component posteriors across fields to test whether
  β and η differ materially
- **RMST at horizons** (90, 180, 365, 730 days): restricted mean survival time
  comparison without parametric assumptions

CLI arguments: `--n-iter`, `--burn-in`, `--thin`, `--chains`, `--k-max`, `--seed`.

## Key findings

- All three cohorts show a clear 2-component structure (infant + wear-out)
- Vt has a **higher infant-mortality weight** (~0.35 vs. ~0.25 for Global/Ya)
- Wear-out component β₂ is broadly similar across fields (≈1.3–1.6)
- Ya field shows the longest wear-out scale (η₂ largest), suggesting better late survival
- RMST at 365 days: Vt < Ya < Global, but difference narrows after infant exclusion

## Known limitations / open questions

- Ya cohort small (n_fail ~80–120); posterior credible intervals wide
- MCMC runtime ~5–15 min per cohort at default settings (n_iter=1 000, 1 chain)
- Run features (kpod) used for data loading only; not yet used as covariates in mixture

## Outputs

```
analysis_outputs/bayesian_field_survival_compare_2026_06_23/
  global/   ← posterior plots, component tables
  vt/
  ya/
```

## How to re-run

```bash
python scripts/analyze_field_survival_bayesian.py --n-iter 1000 --chains 1
# Longer run for publication-quality posteriors:
python scripts/analyze_field_survival_bayesian.py --n-iter 5000 --burn-in 2000 --chains 3
```

## Related analyses

- [[vt_freq_latent]] — same model applied to frequency stratification within Vt
- [[vt_failure]] — classical (frequentist) comparison across the same cohorts
