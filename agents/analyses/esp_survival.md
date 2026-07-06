---
name: esp-survival
description: Stratified latent-mixture Weibull survival model for ESP failures across multiple oilfields, contractors, and H2S exposure levels. In-progress.
metadata:
  type: project
---

# ESP Failure Survival Analysis

## Task / Research question

Build a stratified K=2 latent Weibull mixture survival model for ESP failures
(1,573 runs, 77 columns) that:
1. Separates early-failure (C1: infant mortality) from wear-out (C2: aging) modes per
   stratum — because single-Weibull fits yield β ≈ 0.8–1.0, producing implausible RUL.
2. Quantifies the effect of operational and chemical parameters via Extended Cox regression.
3. Produces individual RUL estimates conditioned on field, H2S class, contractor,
   and operating covariates.

## Status

- **Status**: in-progress
- **Last run**: 2026-06-30
- **Entry point**: `python scripts/run/esp_survival.py [phases]` or
  `python backend/analysis/workflows/esp_survival/run.py [phases]`
- **Phases**: 0=audit, 1=single-Weibull, 2=K2-mixture, 3=diagnostics,
  4=contractor, 5=Extended-Cox, 6=RUL

## Data

| Source | Access | Key info |
|---|---|---|
| `Отказы свод с анализом_БДА_В03_failures.xlsx` | `resolve_v03_failures_path()` | 1,573 runs; 77 columns; sheet "Свод" |
| Failure Flag column | `Failure Flag`: 1=failure, -1=censored | 1,537 failures, 36 censored |

**Key columns**: `Месторождение` (field), `Кислый/Некислый` (H2S class),
`Наработка (сут)` (TTE), `Принадлежность` (contractor), `Признак отказа` (early/mature).

## Stratification

All 91 sour wells are in Vt field (91/274 Vt runs = 33%). H2S is a mandatory
stratum — not a Cox covariate — because it changes β, η, and w simultaneously.

| Stratum | n_failures | fit_mode |
|---|---|---|
| Ya_nonsour | 733 | independent_K2 |
| Vt_sour | 91 | independent_K2 (pilot) |
| Vt_nonsour | 178 | independent_K2 |
| Az_nonsour | 145 | independent_K2 |
| Za_nonsour | 145 | independent_K2 |
| Ic_nonsour | 150 | independent_K2 |
| Mc_nonsour | 41 | independent_K2 |
| Da_nonsour | 32 | two_stage_K2 |

## Methods and models

- **Phase 0**: Stratum viability audit; per-stratum × covariate coverage;
  pad-level imputation feasibility (CV of chemistry per pad).
- **Phase 1**: Kaplan-Meier + single Weibull MLE per stratum; flag β < 0.9.
- **Phase 2**: K=2 latent mixture via **EM on raw data** (`weibull_em.py`).
  Constraints: β₂ > max(1, β₁) (hard; prevents label swap), η₂ ≥ 2·η₁ (hard),
  w₁ < 0.75 (soft; flagged as DEGENERATE). 12 Halton starts per stratum.
  Two-pass: (1) independent fits for all ≥40-failure strata; (2) two-stage fits
  for 20–39-failure strata with β₁, β₂ fixed to median of non-degenerate independent fits.
- **Phase 3**: Bootstrap CI (50 resamples, raw-data resampling + EM); RMSE vs KM;
  component assignment r_j1; cross-validation against `Признак отказа` and `Причина отказа УЭЦН`.
- **Phase 4**: KM per contractor within Vt sour; multi-group log-rank; stratified Cox;
  Schoenfeld PH test; w₁ per contractor (EM with fixed shapes from stratum model).
- **Phase 5**: Spearman |ρ| > 0.65 screening; univariate stratified Cox per Tier 1–3
  covariate; multivariate Extended Cox (X + X·log(t) for PH violators);
  cluster-robust SE on well_key; VIF check.
- **Phase 6**: Conditional survival RUL = ∫ S(t|t₀) dt; mixture baseline + Cox
  adjustment; profiles at t₀ = 0–900 days; success-criterion checks.

## Key findings (EM-based run 2026-07-01)

### Phase 0 — Data audit
- 1,477 usable runs (1,573 raw − 96 zero-TTE); 1,466 failures, 11 censored (0.7%)
- All 89 sour failures in Vt field; confirmed 47% explicitly attributed to H2S cause
- Contractor ordering in Vt sour (median TTE): Шлюмберже 72d > Борец 48d > Новые технологии 34d
- 28% of pads have only one well (no within-pad averaging — field-level fallback)

### Phase 1 — Single Weibull
- 6/9 strata flagged β < 0.9 (mixture suspects); Vt sour β=0.988 ≈ 1.0 (near-exponential)
- Confirms single-Weibull inadequacy for all major non-sour fields

### Phase 2 — K=2 mixture via EM (PRIMARY RESULT)
- **β₂ > 1.0 in all 9 strata** ✓ — wear-out mode recovered everywhere
- **Vt sour pilot** (key deliverable):
  - C1 (H2S attack): β₁=1.347, η₁=14d, w₁=0.258
  - C2 (wear-out in H2S): β₂=1.392, η₂=114d — η₂/η₁=7.9 ✓
  - RMSE=0.016 (vs single-Weibull RMSE=0.032 — 2.0× better)
- **Az_nonsour** most physically interpretable: β₁=0.992, η₁=67d, β₂=1.863, η₂=480d, w₁=0.528
- **Ic_nonsour** cleanest fit: β₁=1.209, β₂=1.413, w₁=0.254, η₂/η₁=14.0
- **Ya_nonsour**: β₁≈β₂≈1.09 (quasi-exponential modes, not shape-separated); η₂/η₁=13.1 provides scale separation
- **Vt_nonsour**: DEGENERATE (w₁=0.944, β₂=9.416 near boundary) — genuine data ambiguity
- **Za_nonsour**: DEGENERATE weight (w₁=0.799) but good shapes β₁=0.931, β₂=2.494
- Global shapes (two-stage priors): β₁=1.145, β₂=1.403 (median of non-degenerate independent fits)

### Phase 3 — Diagnostics (EM bootstrap)
- Mixture better than single Weibull (ΔRMSE > 0) in 8/9 strata; Mc_nonsour borderline (−0.003)
- Bootstrap CIs consistent with EM point estimates for all 6 independent strata ✓
- Largest ΔRMSE improvements: Ic_nonsour +0.030, Az_nonsour +0.019, Vt_sour +0.016

### Phase 4 — Contractor (EM with fixed stratum shapes)
- **All three focus strata → promote_to_stratum** (w₁ varies >10pp across contractors)
  - Vt_sour: w₁_range=0.231, logrank_p=0.169 (PH holds)
  - Vt_nonsour: w₁_range=0.287, logrank_p=0.031
  - Ya_nonsour: w₁_range=0.405, logrank_p<0.001 ← largest contractor effect
- Ya_nonsour contractor effect much larger with EM (was 0.065 with WLS β₁≤1 constraint)

### Phase 5 — Extended Cox
- High-correlation pairs: log_cl ↔ log_ca (ρ=0.752); log_so4 ↔ log_ca (ρ=−0.723); neither univariate significant so both excluded from multivariate
- 10 univariate significant at p<0.10: motor_load, water_cut, delta_bep, glr, p_bot, p_bubble, log_kvch, log_h2s_conc, ph, n_stages_ratio
- Multivariate concordance = **0.877** (strong discrimination)
- PH violations: log_kvch, motor_load, motor_load_x_logt, n_stages_ratio, p_bubble, ph
- VIF: motor_load + motor_load_x_logt high (expected by construction in Extended Cox)

### Phase 6 — RUL
- **Vt sour** t₀=60d: RUL=79.6d < Vt nonsour 218.5d ✓; decreasing with age (86→80→73→62d) ✓
- Vt_nonsour and Ya_nonsour: RUL increases with age (degenerate/quasi-exponential fits)
- Vt_nonsour RUL inflated by degenerate fit (w₁=0.944 infant mortality component)

## Known limitations

- Pad-level imputation for water chemistry is the dominant uncertainty source
  (≥50% missing for Cl⁻, pH, SO₄²⁻). Imputed values treated as fixed — not yet
  propagated through bootstrap for RUL CI.
- Extended Cox (Phase 5) is exploratory — covariate selection is data-driven and
  requires out-of-sample validation.
- Contractor effect in Vt sour is based on small n per contractor (23–34 runs).

## Outputs

```
results/esp_survival_phase0_audit/<date>/tables/
  stratum_summary.csv
  covariate_coverage.csv
  pad_feasibility.csv
  contractor_field_audit.csv

results/esp_survival_phase1_weibull/<date>/figures/
  phase1_km_weibull_grid.png
  phase1_single_weibull.csv

results/esp_survival_phase2_mixture/<date>/figures/
  phase2_mixture_grid.png
  phase2_mixture_params.csv

results/esp_survival_phase3_diagnostics/<date>/tables/
  phase3_diagnostics.csv
  component_assignment_<stratum>.csv
  phase3_vt_sour_component_assignment.png
  phase3_w1_comparison.png

results/esp_survival_phase4_contractor/<date>/tables/
  phase4_contractor_km.png
  phase4_contractor_analysis.csv

results/esp_survival_phase5_cox/<date>/tables/
  phase5a_spearman_corr.csv
  phase5b_univariate_cox.csv
  phase5c_multivariate_cox.csv
  phase5c_schoenfeld_test.csv
  phase5d_vif.csv

results/esp_survival_phase6_rul/<date>/figures/
  phase6_rul_profiles.png
  phase6_vt_sour_contractor_rul.png
  phase6_rul_snapshot.csv
```

## How to re-run

```bash
# Full pipeline
python scripts/run/esp_survival.py all

# Specific phases only
python scripts/run/esp_survival.py 0,1,2,3

# Core survival analysis (no Cox, no RUL)
python scripts/run/esp_survival.py 0,1,2,3,4
```

## Related analyses

- [[vt_failure]] — 13-phase Vt-only analysis (predecessor)
- [[vt_60hz]] — frequency exposure survival analysis
- [[bayesian_field_survival]] — hierarchical Bayesian cross-field comparison
