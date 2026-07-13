# Production Risk Failure-Rate Handoff Prompt

You are working in `D:\GitHub\Pump2` on the production-risk workflow for ESP pump failure and repair forecasting.

## User Goal

Ship the Excel/VBA/EXE production-risk tool after fixing the repair forecast and failure-rate reporting. The current deliverables must include:

- `Риск_добычи_УЭЦН_2026_2027.xlsx`
- `Прогноз_ремонтов.xlsx`
- `Прогноз_ремонтов_hazard.xlsx`
- `ProductionRiskLauncher.xlsm`
- rebuilt `Pump2ProductionRisk.exe`

The user specifically wanted the same hazard-aware failure-rate reporting added to `Прогноз_ремонтов_hazard.xlsx`.

## Implemented Changes

1. Repair forecast daily status is now failures-only.
   - In `repair_compat.py`, status `0` marks only the predicted failure event day.
   - Status `1` marks every other day.
   - Downtime / repair-in-progress blocks are no longer painted into the daily grid.
   - Existing workbook shape is preserved for VBA/client compatibility.

2. Failure-rate reporting was added through `failure_rate.py`.
   - Adds native Excel sheets:
     - `Интенсивность отказов`
     - `Отказы_данные`
   - Charts compare fact versus model prediction by `УН` and globally.
   - Uses openpyxl native charts, not matplotlib, so Cyrillic labels survive packaging.

3. Fleet denominator was corrected.
   - Fleet size is counted from the production-plan master `Сводные данные`.
   - A well counts in a month only when:
     - `Отработанное время > 0`
     - and oil or liquid production is positive.
   - This avoids inflating the denominator with runtime rows for non-producing wells.
   - This was important because the company does not have roughly 939 operating oil wells at the latest actual months; the corrected active producing fleet is around 500 in late history.

4. Reporting field changed from `Месторождение` to `УН`.
   - `crosswalk.py` now reads the plan/license-area column into `producer_meta.license_area`.
   - Failure-rate grouping uses `УН`.
   - Survival strata still come from well-code prefix:
     - `Ya`, `Vt`, `Za`, `Ic`, `Az`, `Mc`, `Da`
   - A single `УН` can span multiple survival strata.

5. Historical model line now uses observed installation intervals.
   - Historical prediction is no longer a simulated renewal chain.
   - It replays observed pump intervals from installation dates.
   - `WellsArtificialLiftBig.xlsx` backfills missing starts/boundaries from `Свод`.
   - `Свод` remains a fallback and supplies sour/contractor strata refinements where possible.

6. Observed failures are supplemented and deduplicated.
   - Actual failures come from `Свод` plus `WellsArtificialLiftBig.xlsx`.
   - Deduplication is by well/date.
   - Failures are restricted to wells present in the master fleet, keeping numerator and denominator populations aligned.

7. Incomplete actual months are blanked.
   - Fact line is blanked after the last complete actual month.
   - Current cutoff is `2026-04`.
   - Months such as `2026-05` and later show prognosis only.

8. Forecast failure rates are written to both repair workbooks.
   - `Прогноз_ремонтов.xlsx` gets the primary/base failure-rate layer.
   - `Прогноз_ремонтов_hazard.xlsx` gets the stress/hazard-aware failure-rate layer.
   - Full-table export also writes:
     - `F_failure_rate_monthly.csv`
     - `F_failure_rate_hazard_monthly.csv`

9. `WellsArtificialLiftBig.xlsx` was wired through the CLI, config, manifest, launcher, VBA, and PyInstaller build.
   - New CLI flag: `--equipment-big`
   - New config field: `RunConfig.equipment_big_path`
   - Launcher cell `B9` stores the Big workbook path.
   - VBA passes `--equipment-big`.
   - PyInstaller hidden imports include:
     - `analysis.data.equipment_big`
     - `analysis.data.pump_type_parser`
     - `scripts.data_utils`

10. Launcher date handling was fixed.
    - Launcher stores `B6` and `B7` as ISO text:
      - `2026-07-01`
      - `2027-12-01`
    - VBA `DateArg` accepts ISO text or normal Excel dates and passes `yyyy-mm-dd`.
    - This prevents locale/date coercion bugs in headless Excel and packaged deployment.

## Key Findings

- The initial future failure-rate spike was not caused by a different Weibull model alone.
- The major denominator issue was fleet calculation: counting `Отработанное время` alone over-counted wells at risk.
- Using active producing wells makes the rate scale much more plausible.
- The strong decline and later jump in the historical prediction was largely due to missing observed installation/replacement starts, especially visible in the `Vt` effect.
- Adding `WellsArtificialLiftBig.xlsx` stabilizes historical pump interval reconstruction and improves alignment with actual failures.
- Replacement downtime semantics matter:
  - The repair forecast grid is failures-only.
  - Failure-rate computation should account for the scenario projection and actual observed intervals, but the status grid no longer paints downtime blocks.
- The hazard/stress workbook can show lower failure counts than base in some future months because longer P75-style downtime leaves less active ageing mass in the projection.

## Latest Verification

The rebuilt frozen EXE was run with:

```powershell
dist\Pump2ProductionRisk\Pump2ProductionRisk.exe `
  --pp-master "D:\Projects\Pumps\data\pp\ТМ-06_2026_2027_Р50_Базовый_Мастер файл.xlsx" `
  --gtm-schedule "D:\Projects\Pumps\data\pp\М08 2025 (СД 2026 - 2027)\ДФ_04.xlsx" `
  --prediction-workbook "D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_simple_prediction_2026-07-09_samples_ru_90d_compact.xlsm" `
  --techregime-workbook "D:\Projects\Pumps\data\tr\ТР_НЕФТЬ 30.06.2026.xlsm" `
  --equipment-big "D:\Projects\Pumps\data\big\WellsArtificialLiftBig.xlsx" `
  --forecast-start 2026-07-01 `
  --horizon-end 2027-12-01
```

Packaged outputs were generated in `dist\Pump2ProductionRisk\results`.

Readback examples for `ГЛОБАЛЬНО`:

- Base workbook:
  - `2026-04`: fleet `508`, fact `38`, model `38.4510`, fact rate `0.074803`, model rate `0.075691`
  - `2026-07`: fleet `611`, model `54.6221`, model rate `0.089398`
  - `2027-08`: fleet `796`, model `68.3266`, model rate `0.085837`
- Hazard workbook:
  - `2026-04`: fleet `508`, fact `38`, model `38.4510`, fact rate `0.074803`, model rate `0.075691`
  - `2026-07`: fleet `611`, model `53.2777`, model rate `0.087198`
  - `2027-08`: fleet `796`, model `64.8030`, model rate `0.081411`

VBA launcher smoke test:

- `STATUS=Completed`
- command included `--equipment-big`
- forecast dates passed as `2026-07-01` and `2027-12-01`
- launcher was reset after smoke test:
  - `B10=Ready`
  - `B12` empty

Full test suite:

```text
300 passed
```

Deployment package created:

```text
D:\GitHub\Pump2\dist\Pump2ProductionRisk_deployment_2026-07-13_154440.zip
```

## Files To Review First

- `backend/analysis/workflows/production_risk/failure_rate.py`
- `backend/analysis/workflows/production_risk/run.py`
- `backend/analysis/workflows/production_risk/repair_compat.py`
- `backend/analysis/workflows/production_risk/crosswalk.py`
- `scripts/run/production_risk.py`
- `scripts/deploy/build_production_risk_launcher.py`
- `scripts/build_production_risk_exe.ps1`
- `vba/mdlProductionRiskLauncher.bas`
- `backend/tests/test_production_risk.py`

## Deployment Reminder

The binary package is generated under `dist\` and is not committed. Rebuild it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_production_risk_exe.ps1
```

Then rerun the EXE and, if needed, zip `dist\Pump2ProductionRisk\*`.
