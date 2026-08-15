# Production Risk Ql Field-Ref Review Handoff

Date: 2026-07-14

This note summarizes the local changes introduced while adding the Ql hazard layer,
updating the Excel/VBA/EXE delivery, and investigating the March-June 2026 failure
rate behavior. It is intended as input for a separate review agent.

## Current Deliverable

Active package:

- `dist/Pump2ProductionRisk_QlFieldRef/Pump2ProductionRisk_QlFieldRef.exe`
- `dist/Pump2ProductionRisk_QlFieldRef/ProductionRiskLauncher.xlsm`
- `dist/Pump2ProductionRisk_QlFieldRef/data/inputs/Отказы свод с анализом.xlsx`
- `dist/Pump2ProductionRisk_QlFieldRef/results/Риск_добычи_УЭЦН_2026_2027.xlsx`
- `dist/Pump2ProductionRisk_QlFieldRef/results/Прогноз_ремонтов.xlsx`
- `dist/Pump2ProductionRisk_QlFieldRef/results/Прогноз_ремонтов_hazard.xlsx`
- detailed tables under
  `dist/Pump2ProductionRisk_QlFieldRef/results/production_risk_forecast/2026-07-14/tables/`

Latest packaged run used the self-contained `Свод` workbook from the package input
folder and showed factual data through `2026-06`.

## Model Choice Shipped

The shipped Ql hazard is the field-reference capped-log model:

```text
covariate = log_mean_qliq_m3d_field_ref_clip5
z = clip(log1p(Ql_m3d) - field_ref_log, +/- log(5))
eta(age) = z * (beta + gamma * log(max(age, 1)))
theta = exp(eta)
```

Constants are in `backend/analysis/workflows/production_risk/config.py`:

- `QL_HAZARD_ENABLED = True`
- `QL_HAZARD_BETA = -0.587058`
- `QL_HAZARD_GAMMA = 0.093473`
- field refs: `Az`, `Da`, `Ic`, `Mc`, `Vt`, `Ya`, `Za`

The global-ref Ql variant was evaluated but not shipped. The current package is
back on Ql field-ref clip5.

## Main Production-Risk Code Changes

### Path and packaged input source

- `backend/analysis/paths.py`
  - `resolve_prediction_workbook_path()` now prefers
    `data/inputs/Отказы свод с анализом.xlsx` after an explicit environment
    variable.
  - This makes the production-risk `Свод` source stable and explicit.

- `scripts/build_production_risk_exe.ps1`
  - Requires `data/inputs/Отказы свод с анализом.xlsx`.
  - Copies that workbook into the distribution at
    `dist/<Name>/data/inputs/Отказы свод с анализом.xlsx`.
  - Adds preflight checks that fixed calibration hooks exist and Ql hazard is
    enabled.
  - Passes the packaged workbook path to the generated launcher.

- `scripts/deploy/build_production_risk_launcher.py`
  - Adds the Ql hazard note to the generated Excel/VBA launcher metadata.

- `vba/mdlProductionRiskLauncher.bas`
  - Updates the visible calibration/hazard note.

### Ql hazard mechanics

- `backend/analysis/workflows/production_risk/survival.py`
  - Adds `ql_hazard_theta()`.
  - Adds `current_pump_p_fail_monthly()` for horizon probability with monthly
    op-days/calendar-days and optional Ql hazard.
  - Extends `project_well()` with optional `log_ql_monthly` and `model_field`.
  - Applies Ql hazard at daily failure-probability level:
    `q_eff = 1 - (1 - q_base) ** theta`.
  - Extends `WellState` with optional `water_rate_m3d`.

- `backend/analysis/workflows/production_risk/layers.py`
  - Computes monthly `planned_ql_m3d = liquid_volume / op_days`.
  - In stress/hazard scenario only, passes `log1p(Ql)` into projection.
  - Uses monthly Ql hazard for `p_fail_90d`.
  - Adds output/audit columns:
    `planned_ql_m3d`, `planned_qw_m3d`, `theta_static`, `theta_ql`,
    `theta_qw`.
  - Keeps `theta_qw = 1.0`; Qw is not shipped as a hazard.

- `backend/analysis/workflows/production_risk/failure_rate.py`
  - The historical stress failure-rate line now also includes the hazard layer.
  - `_append_interval_predictions()` applies Ql hazard to historical interval
    replay when `use_ql_hazard=True`.
  - `_hist_predicted_failures_by_field()` now accepts `hazard`, `hazard_mode`,
    and `bundle_date`.
  - Historical hazard uses static Cox covariates plus Ql field-ref clip5 for the
    stress scenario.
  - Coverage metadata exposes `historical_hazard_layer`.

### Validation chart / forecast / calibration semantics

- `backend/analysis/workflows/production_risk/failure_rate.py`
  - The validation chart is documented and computed as conditional monthly
    active-fleet risk on both sides of `forecast_start`.
  - `_SURVIVAL_WEIGHT_FIELDS` is now empty. The earlier Mc survival-weight path
    is no longer active in the shipped line.
  - Fixed transparent calibration factors remain hard-coded and are not recomputed.
  - Factual failure-rate display now includes `2026-05` and `2026-06`.

- `backend/analysis/workflows/production_risk/export_excel.py`
  - Method sheet now explicitly separates:
    - validation chart: conditional monthly active-fleet risk,
    - forecast/prognosis: renewal simulation with downtime, oil loss, repair load,
    - calibration: fixed transparent factors.
  - Adds Russian labels for Ql fields and Ql theta.

- `backend/analysis/workflows/production_risk/run.py`
  - Shares the `HazardLayer` with both base and stress failure-rate computations.
  - Adds an optional CatBoost comparison path, off by default.

- `scripts/run/production_risk.py`
  - Adds `--catboost-compare`, off by default and intended for Python/CLI only.

### Plan parsing / water-rate support

- `backend/analysis/workflows/production_risk/crosswalk.py`
  - `PlanData` now includes oil m3, produced-water volume/rate, and registry
    water-rate frames.
  - Produced water is parsed from produced-water rows only; PPD/injection water
    is deliberately excluded.
  - If direct produced-water volume is missing, produced water is derived as
    `liquid_volume - oil_volume_m3` where possible.
  - Adds producer metadata for planned produced water and positive-water months.

### Tests and dependencies

- `backend/tests/test_production_risk.py`
  - Adds a test that Ql hazard changes the projected failure path.
  - Extends plan-load test coverage for oil m3 and produced-water parsing, including
    exclusion of PPD water.

- `backend/requirements.txt`
  - Adds `lifelines`, `scikit-learn`, and `statsmodels` for the analysis/model
    comparison work.

## Supporting / Experimental Analysis Changes

These are present in the working tree but are not the main shipped EXE path:

- `backend/analysis/features/restart_events.py`
  - Adds optional truncation to the first N operating days for early-window restart
    features.

- `backend/analysis/data/equipment_big.py`
  - Adds contractor extraction from equipment source fields.

- `backend/analysis/bayesian_latent_weibull.py`
  - Backward-compat shim now re-exports private Weibull helper functions used by
    downstream scripts/tests.

- `backend/analysis/plotting.py`
  - Backward-compat shim now re-exports private plotting helper frames.

- `scripts/build_run_features.py`
  - Import-order cleanup only.

Untracked analysis scripts and notes exist for CatBoost/regression and chemistry
screens. Reviewers should separate those from the production-risk deliverable unless
the review scope explicitly includes exploratory analysis.

## Generated Chemistry / Hazard Evidence

Relevant result directories:

- `results/chem_ion_rate_proxies`
- `results/chem_cumulative_ion_proxies`
- `results/chem_ql_field_ref`
- `results/chem_ql_transform_screen/2026-07-14`

The final selected model was Ql field-ref clip5, not Qw and not ion-specific
proxies. Qw is carried as an audit/output field only in the shipped projection.

## Failure-Rate Behavior Notes

### Mirninsky jump around forecast start

Earlier investigation found that the validation chart and prognosis are different
objects:

- validation chart: conditional active-fleet monthly risk, no renewal chain,
- forecast/prognosis: renewal simulation with downtime and replacement.

The workbook text was updated to make this distinction visible.

### March-June 2026 decline

The latest diagnostic focused on why predicted rates decrease from `2026-03` to
`2026-06`.

From `F_failure_rate_monthly.csv`:

```text
ГЛОБАЛЬНО:
2026-03 fleet=489 predicted=35.575841 rate=0.072752
2026-06 fleet=577 predicted=34.185656 rate=0.059247

Верхнетирский УН:
2026-03 fleet=128 predicted=15.173697 rate=0.118545
2026-06 fleet=150 predicted=11.057983 rate=0.073720
```

Conclusion:

- The global decline is mostly driven by Верхнетирский.
- In Верхнетирский, sour contribution collapses while non-sour contribution is
  roughly flat/slightly up:
  - `Vt_sour_*`: about `6.84 -> 1.95` expected failures,
  - `Vt_nonsour_*`: about `8.34 -> 9.10` expected failures.
- 2026 starts increasingly skew non-sour:
  - March: `15 nonsour`, `6 sour`,
  - April: `23 nonsour`, `3 sour`,
  - May: `18 nonsour`, `1 sour`,
  - June: `1 nonsour`, `1 sour`.
- There is also a June denominator effect: 14 Верхнетирский wells are active in
  the production-plan fleet denominator but have no overlapping Big/Свод history
  pump interval, so they add fleet size without adding historical predicted
  failures.
- The same shape appears in `F_failure_rate_hazard_monthly.csv`, so this decline
  is not a Ql hazard artifact.

## Review Focus

Recommended high-value review points:

1. Confirm Ql hazard sign/age interaction is intentional:
   `beta < 0`, `gamma > 0`, field-ref centered, clipped at log ratio 5.
2. Confirm using liquid volume divided by op-days gives m3/day consistently for
   both history and forecast.
3. Confirm historical stress line should include both static Cox hazard and Ql
   monthly hazard, while base remains no hazard.
4. Check whether the June denominator-only active wells in Верхнетирский should
   remain in the validation denominator when they lack Big/Свод intervals.
5. Check that packaged-source precedence and launcher pathing behave correctly
   both from repo and from frozen EXE distribution.
6. Verify that fixed calibration factors are acceptable and sufficiently visible
   in the workbook/launcher.
7. Decide whether untracked CatBoost/regression/chemistry scripts are in review
   scope or should be ignored for the deliverable review.

## Suggested Verification Commands

```powershell
python -m pytest backend/tests/test_production_risk.py

python scripts/run/production_risk.py --full-tables

powershell -ExecutionPolicy Bypass -File scripts/build_production_risk_exe.ps1 `
  -Name Pump2ProductionRisk_QlFieldRef
```

The latest package had already been run successfully and regenerated factual data
through `2026-06`.
