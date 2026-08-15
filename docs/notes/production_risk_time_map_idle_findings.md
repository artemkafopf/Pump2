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

## Time-map diagnostics — mostly a null lever

`time_map_reconstruction_metrics.csv`:

| split | model | monthly op-day MAE | median total ННО abs % |
|---|---|---:|---:|
| holdout | scalar_field | 8.79 | 18.8% |
| holdout | age_season_map_unscaled | **7.82** | 22.2% |
| holdout | age_season_map_scaled_to_run_nno | 14.00 | **0.0%** |

Interpretation: this is mostly a **negative Workstream-B result**. The age/season map
improves monthly placement only modestly vs the scalar (`8.79 → 7.82` MAE, about 11%),
while raw daily-tech-regime totals reconstruct source `ННО` worse (`18.8% → 22.2%`).
The global mapped uptime (`~0.757`) is very close to the legacy scalar (`~0.745`), so the
old constant `uptime_factor` was not a material over-prediction lever.

The deployed historical replay therefore treats source `ННО` as the clock ground truth
and uses the map for **placement weights**, scaling monthly weights back to known run
`ННО`. That gives 0% total-life error by construction for runs with `ННО`; it is not
evidence that the map predicts total operating life. Unscaled map rows remain the fallback
for intervals without total op-days.

This is the key B1 reconciliation: `ННО`/A `tte` remains the model clock; daily
tech-regime is used for within-run calendar placement, not for replacing run totals.

Methodology caveat: the reported "holdout" was split after fitting the map, so it is not
a true out-of-sample holdout. D must rerun this with a run/well split applied before map
fitting if the map is considered for promotion.

## Idle-month decision — provisional, Mc not cleared

`idle_hazard_decision.csv` shows idle hazard is not close to zero. Focus fields:

| field | idle failures | producing failures | idle/prod fraction | decision |
|---|---:|---:|---:|---|
| Мирнинский УН | 10 | 31 | 1.000 | fit idle hazard |
| Ярактинский УН | 25 | 187 | 0.672 | fit idle hazard |
| Верхнетирский УН | 28 | 195 | 0.571 | fit idle hazard |
| GLOBAL | 102 | 578 | 0.688 | fit idle hazard |

Provisional decision: idle failures are not zero, but this table does **not** fully
disambiguate real idle hazard from failure-date/pull-date misdating. A pump that fails
mid-month can appear as non-producing in the plan for that same month, mechanically
inflating the idle bucket. Мирнинский is the risky case: its raw idle/producing ratio is
above 1 and is capped to 1.0, exactly the pattern expected from misdated pull-months.

D must not enable the fitted idle fraction for Мирнинский until a last-producing-month
re-attribution check is run. Historically, known-`ННО` intervals only redistribute fixed
op-days across months, so idle fraction changes monthly shape more than total level; the
larger level effect is forward projection in planned-idle months, where Mc idle fraction
= 1.0 could increase predictions and worsen over-prediction.

## Files

- `esp_time_map.csv` — bundle-facing B artifact.
- `tables/time_map_run_month_dataset.csv` — B1 run-month dataset.
- `tables/uptime_map_rows.csv` — uptime map rows only.
- `tables/idle_hazard_decision.csv` — idle vs producing empirical hazards.
- `tables/time_map_reconstruction_metrics.csv` — scalar vs map diagnostics.

## Caveat for D

Treat the time map as placement-only and near-null; keep A `ННО`/`tte` as the model
clock. Do not promote the Mc idle fraction without the misdating disambiguation. Ya/Vt
idle fractions can be tested as sensitivity, but D acceptance should compare:

1. A baseline with legacy scalar exposure;
2. A + placement-only time map (known `ННО` scaled);
3. A + placement map + idle fractions, with Mc separately gated by re-attribution.
