# `mature_ttf` — Mature-phase TTF stress-based Weibull model

## Task / Research question

Build a Weibull stress-life model that predicts mature-phase TTF (run life excluding
infant mortality) as a function of operating stress terms (under-loading, over-GLF,
H₂S exposure, salt load, load variability).  Identifies which stress terms
significantly accelerate failure in the wear-out regime.

## Status

`[x] complete`

Last run: 2026-06-XX  
Entry point: `scripts/run_mature_ttf_workflow.py`  
Results: `analysis_outputs/mature_ttf_phase1_full/`

## Data

| Source | Access | Notes |
|--------|--------|-------|
| V03 failures xlsx | via `scripts.build_run_features.load_runs` | Base run records |
| Run features | `scripts.build_run_features.build_run_features` | Built from telemetry + techregime |
| Feature matrix | `scripts.analyze_failure_horizon.prepare_feature_matrix` | Mature-window aggregates |

Key feature groups (mature-window `_m_` prefix = no future leakage):
- kpod_freq_w_mean, frac_kpod_below_0p7_w — pump operating point
- frac_pzab_below_1_w — under-pressure proxy
- frac_glf_above_thr_w — gas interference proxy
- h2s_proxy_mg_l — acid environment
- salt_proxy_w_mean, gypsum_proxy_m_mean — scaling potential
- load_w_std — load variability / cycling stress

## Methods & models

- **Transform selection** (`analysis.transform_selection.rank_transform_candidates`):
  ranks monotone stress transforms (negative_excess, positive_excess,
  relative_abs_deviation) by Spearman correlation with log-TTF
- **Weibull stress-life model** (`analysis.weibull_model.fit_weibull_stress_model`):
  parametric AFT; stress terms enter as multiplicative scale modifiers;
  gradient-based MLE with L-BFGS-B
- **CatBoost regressor** (optional): non-parametric baseline for feature importance;
  used to validate stress-term ranking from transform selection
- **Train/test split**: deterministic hash split on `row_id` (test_fraction=0.2)
  via `analysis.modeling_config.stable_hash_test_mask`

Stress priority list: kpod_freq_w_mean (weight 0.10), frac_kpod_below_0p7_w (0.10),
frac_pzab_below_1_w (0.10), frac_glf_above_thr_w (0.10), h2s_proxy_mg_l (100.0),
salt_proxy_w_mean (0.01), load_w_std (2.0).

## Key findings

- H₂S proxy is the strongest stress accelerator when present (but sparse coverage)
- Under-loading (kpod<0.7) consistently ranks high across fields
- GLF overshoot adds moderate acceleration; salt/gypsum proxies show weaker signal
- CatBoost and Weibull stress rankings broadly agree on top 3 features

## Known limitations / open questions

- `_m_` features use fixed mature window (DEFAULT_MATURE_WINDOW_DAYS=90); sensitivity
  to window choice not yet tested
- assert_no_explanatory_features enforces no `_w_` leakage; `_m_` features still
  contain some information from the full run if the mature window is short
- Model not yet deployed as a scoring pipeline

## Outputs

```
analysis_outputs/mature_ttf_phase1_full/
  ← feature importance, Weibull parameter tables, stress-term rankings
```

## How to re-run

```bash
python scripts/run_mature_ttf_workflow.py
```

## Related analyses

- [[uvch]] — applies a similar feature space but uses CatBoost AFT as the primary model
- [[vt_failure]] — Phase 10 (CatBoost infant classifier) uses overlapping features
