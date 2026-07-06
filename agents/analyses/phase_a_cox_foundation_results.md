# Phase A — Cox Foundation: Implementation Results (2026-07-06)

Implementation of `agents/analyses/phase_a_cox_foundation.md`. All new reusable logic
lives in `backend/analysis/`; scripts in `scripts/run/` are thin CLI wrappers. Every
output goes through `results_dir()`; `ttf_mix` is the duration column throughout.

## What was built

| Task | New / changed code | Output slug |
|---|---|---|
| **T1** chemistry refit | `data/chemistry_run_features.py` (`tte_col`, `window`, GLF fix), `scripts/run/phase_a_chem_ttfmix.py` | `phase_a_chem_ttfmix` |
| **T2** B50 bootstrap CIs | `models/survival/bootstrap_ci.py`, `scripts/run/phase_a_b50_ci.py` | `phase_a_b50_ci` |
| **T3** ΔAIC K=1 vs K=2 | `workflows/esp_survival/phase2_mixture.py` (`_model_to_row`) | `esp_survival_ttf_mix_phase2` |
| **T4** mixed-clock sensitivity | `scripts/run/phase_a_ttf_sensitivity.py` | `phase_a_ttf_sensitivity` |
| **T5** label hygiene | `data/label_hygiene.py`, `scripts/run/phase_a_label_hygiene.py` | `phase_a_label_hygiene` |
| **T6** extended Cox fix | `scripts/run/phase_a_extended_cox_fix.py` (+ README annotation) | `phase_a_extended_cox_fix` |
| **T7** covariate registry | `data/pump_type_parser.py`, `data/run_covariates.py`, `scripts/run/build_run_covariates.py` | `phase_a_covariate_registry` |

Tests: `backend/tests/test_pump_type_parser.py` (17), `test_run_covariates.py` (8, DB-guarded),
`test_label_hygiene.py` (9). All pass. (6 unrelated pre-existing test modules fail to import
from the in-progress `analysis.plotting` refactor — not touched here.)

## Key findings

**T1 — chemistry on ttf_mix.** Refit on the operating-time clock with the fixed GLF
aggregation. The **КВЧ missing-indicator is highly significant (β≈0.27, p≈0.0003)** — КВЧ
missingness is informative (wells get sampled *because* solids are suspected → MNAR), so the
imputed coefficient is biased; the complete-case β (0.119) is larger in magnitude than the
imputed β (0.053), as mean-imputation attenuation predicts. `chem_final_coeffs.csv` is stamped
`clock=ttf_mix, window=early`. Surviving covariates: `log_h2s_proxy` (HR≈1.13) and `log_glf`
(HR≈0.85, still "protective" — flagged suspect).
*Known limitation:* the `early` window recomputes ion chemistry only; GLF/H₂S are t0/whole-run,
so `window_compare` is flat for the two survivors. Early-window GLF is available in the T7
registry (`glf_mean_opdays`) for a follow-up refit.

**T3 — ΔAIC.** Mixture (K=2) wins by AIC for the large strata (Ya +13.1, Az +9.7, Ic +9.0,
Vt_sour +8.0) but the single Weibull is preferred for the small ones (Mc −3.2, Da −0.9) — a
principled verdict RMSE-to-KM could not give. Columns `nll_k1/nll_k2/aic_k1/aic_k2/delta_aic/
mixture_wins_aic` are now in every `phase2_mixture_params.csv`.

**T4 — mixed-clock sensitivity.** Excluding the 510 `ttf_true_source='missing'` runs moves B50
**>15% in 4 strata** (Az +22.7%, Vt_nonsour +19.8%, Vt_sour +19.2%, Ic +15.5%). The mixed clock
is not innocuous — flag these strata in the ESP survival index. `pct_mixed_clock` is now in the
stratum inventory.

**T5 — label hygiene.** `'НКТ '` (trailing space) now merges into `'НКТ'` (24→21 node
categories). Two field names are unmapped and routed to `needs_review.csv` (**not guessed**):
`ЯНГКМ`, `Гораздинское`. Applying the 10 mg/l H₂S threshold to all fields, **226 non-Vt runs
would flip to sour** (Za 112, Az 66, Ya 39, Ic 9) — reported only; stratum definitions unchanged.

**T6 — extended Cox inference.** The old γ p-values/R² are marked descriptive-only in the
vt_failure README.
*Corrected 2026-07-06:* the first implementation (30 d grid, covariate at episode **end**)
reported γ_cox = −0.851 — **retracted**: failing subjects carried `log(own failure time)` while
at-risk subjects carried `log(grid edge)`, a mechanical negative bias concentrated at early
times. The corrected fit (episode-split at **every distinct failure time**, so the covariate is
the shared risk-set time) gives **γ_cox = +0.196** (SE 0.103, p = 0.058), well-bootstrap 95% CI
**[+0.000, +0.415]** — implied HR(t) ≈ 2.2 at 30 d → ≈ 3.6 at 365 d, a mild, borderline
increase. Bootstrapping the *old* curve-regression γ honestly gives **+0.079 [−0.120, +0.266]**
— the interval **includes 0 and excludes the mixture-based +0.343**, so that estimate was a
curve-preprocessing artifact. Net: H₂S time-trend is weakly positive at best; the M3 headline
HR≈2.3 is unaffected. Lesson recorded: two estimators of one effect disagreeing **in sign**
means an implementation bug until proven otherwise.

**T7 — covariate registry.** `build_run_covariates()` produces one row per run (2,634) with the
full Block 1–4 registry, `*_missing`/`*_imputed_src` provenance, log transforms, and `stratum_key`.
pump_type parser hits **98.4%** (target ≥90%). The registry reports **measured** coverage: КВЧ is
only **~14% measured** at run level (683 non-null samples / 384 wells) — far below the ~41% quoted
in `chemistry_cox.md`. Outputs: `registry.csv`, `registry_by_stratum.csv` (558 low-coverage cells
flagged), `spearman_matrix.csv`, `pump_type_parse_report.csv` + `pump_type_unparsed.csv`, coverage
heatmap + distribution grid.

## Open questions for the user (surfaced, not resolved)

See `results/phase_a_covariate_registry/2026-07-06/reports/OPEN_QUESTIONS.md`:

1. **`vg_m` semantics/units** — labelled "setting depth?" (0–2,740 m); true meaning unconfirmed.
2. **`curvature_flag` units** — 0.31–0.51 cluster; degrees per 10 m? Direction assumed, not usable
   quantitatively until confirmed.
3. **gabarit → OD (mm) mapping** — confirmed only for 5/5А/6; the rest are conventional ГОСТ sizes
   pending confirmation (`GABARIT_OD_MM`, `REDA_SERIES_OD_INCH`).
4. **Tubing (НКТ) diameter** — not in the warehouse; raise as a data request to the data owner.
5. **Non-Vt sour count (226)** — review before extending sour strata beyond Vt.
6. **КВЧ measured coverage (~14%)** — lower than assumed; КВЧ coefficients are imputation-driven.

## ⚠ Repo finding — `backend/analysis/data/` is gitignored

`.gitignore:11` is `data/` (unanchored), which matches **every** directory named `data`,
including `backend/analysis/data/`. As a result the whole source package there — the
**pre-existing** `chemistry_run_features.py` as well as the new `pump_type_parser.py`,
`run_covariates.py`, `label_hygiene.py` — is invisible to git (`git ls-files
backend/analysis/data/` is empty). The modules exist and run on disk, but they will not be
committed until this is fixed. Recommended one-line fix (surfaced, not applied — it is a
team policy file): anchor the rule to the repo root, `data/` → `/data/`, so the top-level
data warehouse stays ignored while source packages are tracked.

## Not done / deferred (as scoped in the spec)

- T5 pipeline wiring: the hygiene helpers are pure functions; wiring `trim_failed_node` /
  `normalize_field` into `scripts/pipeline.py` at ingest is a separate reviewable step.
- No Block 2/3 model fitting (Phase C), no VBA changes, no stratum-definition changes — per spec.
