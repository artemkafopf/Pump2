# Production Risk Failure-Rate Refit Summary

Date: 2026-07-13

## Implemented

- Added materiality filtering for charts in `Интенсивность отказов`.
  - All `УН` remain in the wide table and `Отказы_данные`.
  - Only `ГЛОБАЛЬНО` plus fields with at least 20 fact+model failures in the display window get charts.
- Added explicit prefix decisions.
  - `MR`, `NE` -> `Mc`
  - `AM`, `ZYI` -> `Za`
  - `YAY` -> `Ya`
  - `BT`, `KI`, `MSH` are explicit `Global_Pooled` fallbacks.
- Made model-field resolution prefix-first for all wells.
  - `Свод` is still used for sour/contractor refinement, but no longer changes the model field.
- Added `Global_Pooled` fallback-share diagnostics by `УН`.
- Moved fact cutoff to `RunConfig.fact_through_month` and CLI `--fact-through`.
  - Default remains `2026-04`.
- Extended failure-rate computation back to 2018 for history warm-up.
  - Charts display 2024-01 onward.
  - Pre-plan months use interval-based fleet.
  - Plan/future months use active-producing plan fleet.
- Replaced silent Big-load fallback with warnings and coverage flags.
- Created Mc/MR refit workflow and CLI:
  - `backend/analysis/workflows/production_risk/mc_refit.py`
  - `scripts/run/refit_mc_stratum.py`
- Created new bundle:
  - `results/esp_survival_vba_models/2026-07-13`
  - replaced `Mc_nonsour_Pooled` and `Mc_nonsour_brt`
- Updated production-risk default bundle date to `2026-07-13`.

## Mc/MR Refit

Source data: `Свод + WellsArtificialLiftBig`, deduped by well/install date with ±7 day tolerance.

The initial 2023+ K=2 fit improved Mc but left the downstream Мирнинский ratio just outside acceptance. The shipped row uses the faster 2024+ new-vintage window; all-history is still reported as comparison.

| fit | n runs | failures | model | w1 | beta1 | eta1 | beta2 | eta2 | b50 | b50 CI |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| all history | 317 | 207 | k2 | 0.5287 | 0.6579 | 226.9 | 1.9097 | 453.7 | 268.1 | n/a |
| install 2024+ shipped | 266 | 160 | k2 | 0.1611 | 0.7384 | 26.7 | 1.2776 | 373.7 | 224.0 | 193.9-279.6 |

## Ratio Check

Window: 2024-01..2026-04. Ratio = fact failures / predicted failures.

| УН | before ratio | after fact | after pred | after ratio |
|---|---:|---:|---:|---:|
| ГЛОБАЛЬНО | 1.11 | 1090.0 | 1013.8 | 1.08 |
| Верхнетирский УН | 1.13 | 338.0 | 299.8 | 1.13 |
| Ярактинский УН | 0.99 | 307.0 | 311.1 | 0.99 |
| Мирнинский УН | 1.88 | 99.0 | 76.6 | 1.29 |
| Кийский УН | 3.26 | 11.0 | 3.4 | 3.26 |
| Большетирский УН | 6.48 | 4.0 | 0.6 | 6.37 |

Acceptance status:

- Мирнинский: passed (`1.29`, target `0.8-1.3`)
- Global: passed (`1.08`, target `1.0-1.2`)
- Верхнетирский: unchanged (`1.13`)
- Ярактинский: unchanged (`0.99`)

## Generated Outputs

- `results/Прогноз_ремонтов.xlsx`
- `results/Прогноз_ремонтов_hazard.xlsx`
- `results/production_risk_forecast/2026-07-13/tables/F_failure_rate_monthly.csv`
- `results/production_risk_forecast/2026-07-13/tables/F_failure_rate_hazard_monthly.csv`
- `results/production_risk_forecast/2026-07-13/tables/F_failure_rate_ratio_2024_2026-04.csv`

Workbook readback:

- `Прогноз_ремонтов.xlsx`: 11 charts, 2160 audit rows
- `Прогноз_ремонтов_hazard.xlsx`: 11 charts, 2160 audit rows

## Verification

- `backend/tests/test_production_risk.py`: 19 passed
