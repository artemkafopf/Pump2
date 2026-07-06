# `uvch` — UVCH wells CatBoost AFT survival model

## Task / Research question

Build a predictive survival model for UVCH (ultra-high-viscosity chemical) wells
using CatBoost in Accelerated Failure Time (AFT) mode.  Goal: rank wells by predicted
remaining life and identify which operational features (frequency, GLF, kpod, chemistry
proxies) most strongly accelerate failure.

## Status

`[x] complete`

Last run: 2026-06-XX  
Entry point: `scripts/analyze_uvch_catboost_aft.py`  
Sensitivity: `scripts/analyze_uvch_catboost_aft_sensitivity.py`  
Results: `docs/uvch_claude/` (legacy; new runs → `results/uvch_catboost_aft/`)

## Data

| Source | Access | Notes |
|--------|--------|-------|
| Warehouse DB | `WAREHOUSE_DIR / "pump2.db"` | `mart__vt_freq55`-compatible mart for Vt, Za, Az, Ic |
| Features | Pre-built in mart | freq_w_mean, freq_above_55hz_pct, glf_m_mean, load_m_mean, kpod_m_mean, h2s_proxy, gypsum_proxy, salt_proxy, nominal_flow, motor_power, pbubble |
| Target | `ttf_true_best_days` + `event` flag | Right-censored; uses mature-phase TTF |

Training fields: Vt, Za, Az, Ic (TRAIN_FIELDS).  
Test fraction: 20% (deterministic hash split on `row_id`).

## Methods & models

- **CatBoost AFT** (`loss_function='SurvivalAft'`): gradient-boosted trees with AFT loss;
  handles censoring natively
- **Baseline Weibull AFT**: β=1.316, η=240.3 (global); β=1.295, η=232.5 (Vt sub-cohort)
- **Concordance index (C-index)**: primary metric; evaluated on held-out 20% test set
- **Kaplan-Meier** (lifelines): used to visualise predicted risk deciles
- **Transform selection** (`analysis.transform_selection`): ranked stress-term transforms
  before feature inclusion
- **Sensitivity analysis** (separate script): sweep over hyperparameters and feature subsets

## Key findings

- CatBoost AFT C-index ≈ 0.62–0.65 on test set (vs. ~0.55 for baseline Weibull)
- Top predictors: `kpod_m_mean`, `freq_above_55hz_pct`, `h2s_proxy_mg_l`, `run_number`
- Wells with kpod<0.7 and freq>55 Hz simultaneously show the highest predicted hazard
- Model exports: `.cbm` (CatBoost binary), `catboost_aft_params.json`, calibration plot

## Known limitations / open questions

- Training data dominated by Vt field; model may not generalise to other fields
- Chemistry proxies (gypsum, salt) are indirect — no direct lab measurements for all wells
- AFT distributional assumption not fully verified; Weibull AFT is assumed

## Outputs

```
docs/uvch_claude/
  catboost_aft_uvch.csv         ← predictions
  catboost_aft_model.cbm        ← saved model
  catboost_aft_params.json
  catboost_aft_report.md
  catboost_aft_calibration.png
docs/uvch_codex/                ← Codex-generated alternative run
  figures/ tables/
```

## How to re-run

```bash
python scripts/analyze_uvch_catboost_aft.py
python scripts/analyze_uvch_catboost_aft_sensitivity.py
```

## Related analyses

- [[vt_failure]] — provides the survival framework; UVCH extends it with a predictive model
- [[mature_ttf]] — alternative stress-based Weibull approach on the same feature space
