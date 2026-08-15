# Workstream D — joint validation findings

Date: 2026-07-15

Runner: `scripts/run/production_risk_joint_validation.py`
Outputs: `results/production_risk_joint_validation/2026-07-15/`

This is a validation gate only: no bundle shipped, no EXE rebuilt, and manual calibration
factors are disabled in-process for the scenario ladder.

## D0 time-map holdout

The true pre-fit holdout slightly confirms the B map as a placement model, not a level
model:

| split | model | monthly op-day MAE | median total ННО abs % |
|---|---|---:|---:|
| holdout | scalar_field_trainfit | 8.86 | 19.2% |
| holdout | age_season_map_trainfit_unscaled | 7.91 | 22.3% |
| holdout | age_season_map_scaled_to_run_nno | 14.16 | 0.0% |

Holdout placement improvement is **10.75%**, just above the 10% D0 threshold. Total
`ННО` reconstruction remains worse unscaled, so the D interpretation stays: use the map
only as within-run monthly placement weights, with source `ННО` as the clock.

## D0 idle re-attribution

Last-producing-month re-attribution moves 37 global failures out of recorded idle months.
Focus-field result:

| field | recorded idle frac | re-attributed idle frac | gate |
|---|---:|---:|---|
| Мирнинский УН | 1.199 | 0.901 | pass_real_idle |
| Ярактинский УН | 0.672 | 0.328 | sensitivity/misdated |
| Верхнетирский УН | 0.571 | 0.414 | pass_real_idle |
| GLOBAL | 0.688 | 0.412 | pass_real_idle |

Mc does not collapse to zero after re-attribution; however, adding real-idle exposure
still does not solve Mc in D1.

## D1 scenario ladder, manual factors = 1.0

Fact/model over 2024-01..2026-06:

| scenario | Mc | Ya | Vt | Global |
|---|---:|---:|---:|---:|
| A_only | 1.544 | 1.056 | 1.230 | 1.132 |
| A_plus_time_placement | 1.381 | 0.948 | 1.119 | 1.016 |
| A_plus_idle_reattributed | 1.381 | 0.948 | 1.119 | 1.016 |
| A_plus_idle_hazard | 1.282 | 0.948 | 1.020 | 0.975 |

Best current line is `A_plus_idle_hazard`: Ya, Vt, and Global pass the 0.90-1.10 gate,
but **Мирнинский remains high at 1.28**. D therefore **does not pass** and no presentation
bundle should be shipped yet.

Interpretation:

- Workstream A fixed the main over-prediction; after A+B, the model now tends to
  **under-predict Mc failures**, not over-predict them.
- Time placement helps global/Ya/Vt enough to matter operationally, but it is not a
  standalone level fix.
- Idle hazard helps Vt and Global and nudges Mc, but not enough.
- The remaining blocker is Mc baseline/vintage/exposure, not manual calibration removal
  or Ql/CatBoost.

## Required next step

Resolved by Workstream E:

1. Tested Mc 2023+ / 2024+ variants inside the same D ladder.
2. Accepted `install_2023plus` as the Mc production baseline.
3. Reran D with `results/production_risk_mc_variant_validation/2026-07-15/variant_models/esp_models_mc_install_2023plus.csv`.
4. Promoted the passing candidate bundle to
   `results/esp_survival_vba_models/2026-07-15-mc2023plus/`.

Accepted D rerun:

| scenario | Mc | Ya | Vt | Global |
|---|---:|---:|---:|---:|
| A_plus_idle_hazard, Mc 2023+ | 0.994 | 0.948 | 1.020 | 0.962 |

The normal production path was then verified with manual factors retired and bundle-level
A-population observed counts:

`python scripts/run/production_risk.py --no-excel --full-tables`

Production output `results/production_risk_forecast/2026-07-15/tables/F_failure_rate_monthly.csv`
matches the accepted D line over 2024-01..2026-06.

Files:

- `tables/D_time_map_holdout.csv`
- `tables/D_idle_reattribution.csv`
- `tables/D_scenario_ladder_fact_model.csv`
- `tables/D_scenario_ladder_monthly.csv`
- `tables/D_scenario_ladder_focus.csv`
