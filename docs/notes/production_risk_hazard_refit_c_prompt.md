# Prompt: Workstream C — Refit Hazard Layer On Accepted Baseline And Validate OOS

You are working in `D:\GitHub\Pump2` on the production-risk workflow.

## Context

The accepted production baseline is now:

`results/esp_survival_vba_models/2026-07-15-mc2023plus/`

Important files:

- `esp_models.csv` — accepted survival baseline;
- `esp_time_map.csv` — placement map + gated idle-hazard candidate;
- `esp_observed_failures.csv` — A-population observed numerator used by D/E validation;
- `esp_run_covariates.csv` — existing run-level covariates;
- `esp_cox_coeffs.csv` — OLD hazard layer, **not refit** on the accepted baseline.

The current default config already points to this bundle:

`backend/analysis/workflows/production_risk/config.py`

Manual calibration factors have been retired:

- `_CALIBRATION_FACTORS = {}`
- `_REPORTING_FIELD_CALIBRATION_FACTORS = {}`

D/E acceptance of the baseline over `2024-01..2026-06`:

| field | observed | predicted | fact/model |
|---|---:|---:|---:|
| Мирнинский УН | 41 | 41.27 | 0.99 |
| Ярактинский УН | 212 | 223.65 | 0.95 |
| Верхнетирский УН | 223 | 218.52 | 1.02 |
| ГЛОБАЛЬНО | 680 | 707.00 | 0.96 |

The baseline passes without hazard. Your task is **not** to change that baseline unless
a bug is found. Your task is to refit and honestly validate the hazard layer on top of
this accepted baseline.

Read first:

1. `docs/notes/production_risk_current_model_report_ru.md`
2. `docs/notes/production_risk_current_model_changes_machine.json`
3. `docs/notes/production_risk_ql_hazard_verification_plan.md`
4. `docs/notes/production_risk_joint_refit_plan.md`
5. `backend/analysis/workflows/production_risk/failure_rate.py`
6. `backend/analysis/workflows/production_risk/survival.py`
7. `backend/analysis/workflows/production_risk/crosswalk.py`
8. existing hazard/QL scripts under `scripts/run/` and `backend/analysis/workflows/esp_survival/`

## Objective

Refit Workstream C hazard layer against the accepted baseline
`2026-07-15-mc2023plus`, then validate whether the hazard layer improves OOS performance.

The expected result may be null. Do not tune to win. If hazard does not improve OOS,
keep it as stress/sensitivity only.

## Review guardrails (must satisfy — added 2026-07-15 after model review)

These four are the difference between a trustworthy OOS verdict and a silently invalid
one. They are binding requirements, not suggestions.

1. **Fit link must equal the serve mechanic.** The production replay does not apply a
   generic monthly Cox. Static run-level covariates enter as a proportional-hazard
   multiplier via `HazardLayer.apply_theta` in `survival.py`:
   `eta -> eta / theta^(1/beta)` (i.e. `theta` multiplies the Weibull hazard). Ql enters
   as a **daily** conditional-hazard exponent inside `ql_hazard_theta`/`project_well`:
   `q_eff = 1 - (1 - q)^theta` with `theta = exp(z * (beta + gamma*log(max(age,1))))`.
   Whatever you fit must be **served through these exact mechanics** or the serve code
   must be changed to match the fit. A monthly Poisson/cloglog fit is acceptable **only**
   if its multiplier is then applied through `apply_theta` (static) and the daily
   `1-(1-q)^theta` (Ql). Prefer C3 option 1 precisely because it maps onto `apply_theta`.
   Emit a numeric round-trip proof in `C_train_serve_skew_check.csv`.

2. **Ql age-interaction is a serve property, not decoration.** Ql is served through
   `coef = beta + gamma*log(age)`. You cannot fit without `gamma` and serve with it, or
   vice-versa. If the refit drops the age interaction, `ql_hazard_theta` must drop it too;
   if it keeps it, the fitted `gamma` must be the served `gamma`. Assert this coupling
   explicitly in the skew check.

3. **Mc OOS is underpowered — do not let it drive promotion.** Mc has ~41 observed
   failures over the full 2024-2026 window, so the 2025-01..2026-06 scoring slice holds
   only ~a dozen Mc events. Judge hazard **improvement** on Ya/Vt/global where events are
   plentiful. Treat Mc fact/model purely as a "must not break" guardrail. Do not report a
   Mc-driven OOS lift or degradation as decisive.

4. **Numerator consistency.** The hazard training numerator and the OOS scoring numerator
   must both use the same A-population event attribution as `esp_observed_failures.csv`
   (680 global), **not** raw Свод counts (915 global). If the baseline offset and the OOS
   numerator disagree, the layer will show a spurious lift. Verify the fit numerator sums
   to the D/E A-population totals before fitting.

Additional notes:

- **Do not regenerate `esp_run_covariates.csv`.** Run-level covariate *values* are
  historical and independent of the Mc baseline swap; only the *coefficients* need
  refitting. The file being older than `esp_models.csv` is expected, not a bug.
- **Audit vintage covariates for double-counting.** The accepted baseline already
  vintage-selects Mc (`install_2023plus`). Keeping `install_2023plus`/`install_pre2020`
  as hazard covariates risks re-absorbing vintage for Mc specifically. Report the vintage
  coefficient with and without Mc in the fit population.

## Current hazard status

The old hazard layer is still present but stale:

- `esp_cox_coeffs.csv` was fit before the accepted Mc 2023+ baseline;
- Ql dynamic hazard uses:
  - source: `chem_ql_transform_screen/2026-07-14`;
  - covariate: `log_mean_qliq_m3d_field_ref_clip5`;
  - beta: `-0.587058`;
  - gamma: `0.093473`;
  - cap: `log(5)`;
- current ship reason is `physical_sensitivity_not_oos_validated`.

These parameters must be treated as prior/stale reference, not as accepted final values.

## Required Work

### C0. Freeze inputs and make a reproducible runner

Create a runner, suggested:

`scripts/run/production_risk_hazard_refit_c.py`

It must:

- default to bundle date `2026-07-15-mc2023plus`;
- not overwrite the accepted production bundle;
- write outputs to `results/production_risk_hazard_refit_c/<date>/`;
- write a manifest with input paths, code version, and config;
- emit all tables needed for review.

Do not rebuild the EXE in this workstream.

### C1. Build hazard training table on the accepted baseline

Use the accepted survival model to compute baseline expected hazard/exposure for the
same A-population used in D/E.

The training table must include:

- run/well identity;
- model field and reporting field;
- stratum resolved by `StrataModel`;
- month;
- operating age at month start;
- operating exposure in month after accepted time-map/idle policy;
- observed failure flag/count using the accepted A-population numerator;
- baseline predicted failure probability or expected failure count;
- covariates used by hazard layer.

Use the accepted time-map/idle policy consistently with production replay.

### C2. Covariates

Keep and audit the existing covariate groups:

- GLF;
- load;
- Kpod:
  - `frac_kpod_below_0p7`;
  - `kpod_freq_mean`;
- frequency;
- curvature;
- well history:
  - `log_run_seq`;
  - `log_days_since_prev_failure`;
- vintage:
  - `install_pre2020`;
  - `install_2023plus`.

Refit Ql as a time-varying/monthly covariate.

Candidate Ql transform:

`z = clip(log((1 + Ql_m3d) / (1 + Ql_ref_field_m3d)), +/- log(5))`

where Ql must be computed the same way in fit and serve:

- monthly liquid volume divided by operating days where appropriate;
- field reference from training only;
- same clipping;
- same fallback for missing/zero op-days.

Also test whether an age interaction remains justified:

`z * log(max(age, 1))`

Do not carry old `beta=-0.587058`, `gamma=0.093473` forward without refitting.

### C3. Fit design

Use a transparent hazard overlay design. Acceptable options:

1. stratified Cox / Poisson-style complementary-log-log model with baseline offset;
2. lifelines time-varying Cox if the local code already supports it robustly;
3. a simpler grouped monthly GLM if it is easier to validate and explain.

Whichever is chosen, document the exact likelihood/target:

- what is the event unit;
- what is the exposure offset;
- how baseline survival probability enters;
- how repeated months per run are handled;
- how censoring and failure month are handled.

The fitted layer must produce a multiplier `theta` or equivalent that can be applied in
the production replay without train/serve skew.

### C4. OOS gate

Use an honest time split:

- fit: observations up to `2024-12`;
- score: `2025-01..2026-06`.

Compare:

- accepted baseline only;
- accepted baseline + refit hazard layer;
- optionally old hazard layer as a reference line.

Metrics required:

- per-field and global fact/model ratio;
- monthly rate MAE;
- log loss or Poisson deviance if available;
- discrimination metric if meaningful, e.g. AUC/C-index on monthly failure risk;
- calibration by predicted-risk quantile;
- Ql-specific empirical sanity check by age bands:
  - `<180`;
  - `180..533`;
  - `>533`;
  and Ql tercile.

Acceptance rule:

Hazard may be promoted beyond stress/sensitivity only if it improves OOS monthly-rate
MAE or likelihood/deviance without breaking Mc/Ya/Vt/global fact/model. Otherwise, keep
hazard as stress-only and say so plainly.

### C5. Train/serve skew check

Add an explicit check that the fitted Ql transform and production computation match:

- same field reference values;
- same cap;
- same missing-value behavior;
- same age interaction;
- same sign convention;
- same units.

Emit:

- `C_train_serve_skew_check.csv`;
- `C_covariate_availability.csv`;
- `C_ql_reference_values.csv`.

### C6. Output artifacts

Write at minimum:

- `C_hazard_training_table.csv` or parquet if large;
- `C_refit_coefficients.csv`;
- `C_old_vs_new_coefficients.csv`;
- `C_oos_fact_model.csv`;
- `C_oos_monthly_metrics.csv`;
- `C_oos_calibration_by_quantile.csv`;
- `C_ql_age_tercile_sanity.csv`;
- `C_covariate_availability.csv`;
- `C_train_serve_skew_check.csv`;
- `C_manifest.json`;
- `C_findings.md`.

If the refit passes OOS:

- create a candidate bundle under the results folder only, not production root;
- include `esp_cox_coeffs.csv` candidate and Ql config candidate;
- do not change `config.py` or rebuild the EXE until explicitly approved.

If it does not pass:

- keep current production bundle unchanged;
- document that hazard remains stress/sensitivity only;
- recommend removing or relabeling stale hazard if it is misleading.

## Implementation constraints

- Do not re-enable manual calibration factors.
- Do not change accepted survival baseline unless there is a discovered bug.
- Do not overwrite `results/esp_survival_vba_models/2026-07-15-mc2023plus/`.
- Do not rebuild the EXE in this task.
- Keep changes scoped and testable.
- Add focused tests for any reusable hazard/QL transform functions.
- Verify with `python -m py_compile` and relevant pytest tests.

## Final response expected

Report:

1. where the outputs are;
2. whether hazard passed OOS;
3. what changed in coefficients, especially Ql;
4. whether hazard should remain stress-only or be promoted;
5. tests/commands run.
