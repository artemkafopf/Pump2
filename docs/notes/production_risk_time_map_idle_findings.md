# Workstream B — time map + idle-month decision (findings)

Date: 2026-07-15

Builder: `scripts/run/refit_time_map_idle_months.py` (end-to-end from repo root).
Outputs: `results/esp_time_map_refit/2026-07-15/`.

This is the Workstream-B candidate paired with the frozen Workstream-A population
(`results/esp_survival_big_censored_refit/2026-07-15/fit_population.csv`). Bundle not
shipped, EXE not rebuilt, manual calibration factors untouched.

## What was built

One row per run-month from the A population, enriched with:

- calendar overlap days;
- daily tech-regime operating flag from `proc__daily_operating`;
- `qliq > 0` audit flag from `proc__daily_merged`;
- model field/stratum, calendar month, and operating-age band;
- plan active-producing status for 2024-01..2026-06.

The bundle-facing artifact is `esp_time_map.csv`:

- `row_type=uptime`: empirical uptime by model field, operating-age band, and
  calendar month, with global fallback rows;
- `row_type=idle_hazard`: fitted `lambda_idle / lambda_producing` by reporting field,
  clipped to `[0, 1]`.

Runtime wiring is optional: old bundles have no `esp_time_map.csv`, so the production-risk
replay keeps the legacy `uptime_factor`/linear interpolation branch. If a bundle contains
`esp_time_map.csv`, `StrataModel` loads it and:

- historical replay allocates known run `ННО` across months using the time-map profile
  instead of flat linear placement;
- intervals without known `ННО` use mapped uptime instead of scalar `uptime_factor`;
- plan-idle months receive reduced hazard exposure when the field idle fraction is > 0;
- forward projection adds reduced idle exposure in zero-runtime plan months, with audit
  columns `time_map_idle_op_days_added` and `time_map_idle_source`.

## Time-map diagnostics

`time_map_reconstruction_metrics.csv`:

| split | model | monthly op-day MAE | median total ННО abs % |
|---|---|---:|---:|
| holdout | scalar_field | 8.79 | 18.8% |
| holdout | age_season_map_unscaled | **7.82** | 22.2% |
| holdout | age_season_map_scaled_to_run_nno | 14.00 | **0.0%** |

Interpretation: the age/season map improves monthly placement vs the scalar, but raw
daily tech-regime totals do not reconstruct source `ННО` below the 10% target. The
deployed historical replay therefore treats source `ННО` as the clock ground truth and
uses the map for **placement weights**, scaling monthly weights back to known run `ННО`.
That satisfies total-life consistency by construction for runs with `ННО`; unscaled map
rows remain the fallback for intervals without total op-days.

This is the key B1 reconciliation: `ННО`/A `tte` remains the model clock; daily
tech-regime is used for within-run calendar placement, not for replacing run totals.

## Idle-month decision

`idle_hazard_decision.csv` shows idle hazard is not close to zero. Focus fields:

| field | idle failures | producing failures | idle/prod fraction | decision |
|---|---:|---:|---:|---|
| Мирнинский УН | 10 | 31 | 1.000 | fit idle hazard |
| Ярактинский УН | 25 | 187 | 0.672 | fit idle hazard |
| Верхнетирский УН | 28 | 195 | 0.571 | fit idle hazard |
| GLOBAL | 102 | 578 | 0.688 | fit idle hazard |

Decision: **do not attribute idle failures wholesale to the last producing month**.
Use a fitted idle-hazard fraction in replay/projection. Мирнинский is capped at 1.0
because its empirical idle hazard is slightly above producing-month hazard on the A
population.

## Files

- `esp_time_map.csv` — bundle-facing B artifact.
- `tables/time_map_run_month_dataset.csv` — B1 run-month dataset.
- `tables/uptime_map_rows.csv` — uptime map rows only.
- `tables/idle_hazard_decision.csv` — idle vs producing empirical hazards.
- `tables/time_map_reconstruction_metrics.csv` — scalar vs map diagnostics.

## Caveat for D

The strict B4 target is only met in the deployed `scaled_to_run_nno` historical mode.
The unscaled daily-tech-regime map improves monthly placement but misses source `ННО`
by ~22% on holdout. D should validate fact/model with the scaled historical replay and
watch whether added idle exposure over-corrects Mc.
