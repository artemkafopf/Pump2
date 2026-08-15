# Production Risk Kpod / Reporting / New-Launch Uncertainty Handoff Prompt

Paste this into a fresh Codex session in `D:\GitHub\Pump2`. The tree already contains the implementation described below; your job is to verify it honestly, fix any regressions, and produce a clear verdict.

## Context

The production-risk workflow has just been extended after the Kpod U-shape hazard work. The recent changes are intentionally stress/sensitivity-oriented, not promoted into the base P50 forecast.

Key principle: keep the baseline forecast honest and avoid overclaiming. Kpod and new-launch uncertainty are stress-side dials unless proper OOS/causal validation says otherwise.

## Implemented Changes To Verify

1. Kpod monthly reporting:
   - Full historical + forecast monthly Kpod table exists.
   - Output CSV:
     `results/production_risk_forecast/2026-07-15/tables/A_kpod_by_field_monthly.csv`
   - Standalone workbook:
     `results/Kpod_by_field_monthly.xlsx`
   - Main prognosis workbook:
     `results/Риск_добычи_УЭЦН_2026_2027.xlsx`
   - Table includes average Kpod, Ql, Qo, Qnom, and well count.
   - Charts include Kpod, average liquid rate, and average oil rate.

2. Reporting field correction:
   - `Большетирское НМ` must display as `Bt - Большетирское НМ`, not `Vt - Большетирское НМ`.
   - `Bt` is a reporting field code here; do not blindly map it to a trained survival stratum unless the registry has a defensible `Bt_*` model.
   - Verify `Vt - Большетирское НМ` is absent from Kpod outputs.

3. Field-specific Kpod hazard:
   - Kpod hazard now supports field-specific parameters.
   - Config CSV:
     `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_kpod_hazard.csv`
   - Supported rows include:
     - `field.Bt.k_hi`
     - `field.Bt.beta_over`
     - `field.Vt.beta_over`
     - `field.Ya.k_lo`
   - Global rows still act as fallback.
   - Current field rows match the old global behavior unless tuned.

4. New-launch uncertainty hazard:
   - Added as a stress-only hazard layer for future/new pump launches.
   - Config CSV:
     `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_uncertainty_hazard.csv`
   - Formula:
     `theta_uncertainty = min(1 + u0 * exp(-age / tau_days), cap)`
   - Applies to:
     - `new_plan_only`
     - `new_df_esp`
     - `new_tr_esp`
   - Current values are expert-prior / conservative sensitivity values, not estimated coefficients:
     - global `u0=0.25`, `tau_days=60`, `cap=1.30`
     - `Bt`: `u0=0.30`, `tau_days=60`, `cap=1.35`
   - Must not affect exact/known-age pump states.
   - Output column `theta_uncertainty` must appear in by-well outputs and Excel.

5. Frozen EXE / launcher:
   - Standard dist folder should include both:
     - `dist/Pump2ProductionRisk/Pump2ProductionRisk.exe`
     - `dist/Pump2ProductionRisk/ProductionRiskLauncher.xlsm`
   - Launcher should point to the standard full source workbook, not compact samples:
     - `B3 = D:\GitHub\Pump2\data\inputs\Отказы свод с анализом.xlsx`
     - `B8 = D:\GitHub\Pump2\dist\Pump2ProductionRisk\Pump2ProductionRisk.exe`
     - `B9 = D:\Projects\Pumps\data\big\WellsArtificialLiftBig.xlsx`

6. Frozen-package warning fix:
   - `WellsArtificialLiftBig load failed; falling back to Свод-only history: No module named 'scripts'` should be gone.
   - The root cause was `analysis.data.equipment_big` importing `scripts.data_utils.normalize_well_key`; this was replaced by a local normalizer.
   - `scripts.data_utils` should not be a hidden import in:
     - `Pump2ProductionRisk.spec`
     - `scripts/build_production_risk_exe.ps1`
   - The openpyxl warning about data-validation extension is harmless and may remain.

## Files Expected To Be Touched / Relevant

- `backend/analysis/workflows/production_risk/config.py`
- `backend/analysis/workflows/production_risk/survival.py`
- `backend/analysis/workflows/production_risk/layers.py`
- `backend/analysis/workflows/production_risk/failure_rate.py`
- `backend/analysis/workflows/production_risk/export_excel.py`
- `backend/analysis/data/equipment_big.py`
- `backend/tests/test_production_risk.py`
- `backend/tests/test_hazard_refit_c.py`
- `Pump2ProductionRisk.spec`
- `scripts/build_production_risk_exe.ps1`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_kpod_hazard.csv`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_uncertainty_hazard.csv`

## Required Verification Commands

Run at minimum:

```powershell
$env:PYTHONPATH='D:\GitHub\Pump2\backend;D:\GitHub\Pump2'
pytest backend\tests\test_equipment_big.py backend\tests\test_production_risk.py backend\tests\test_hazard_refit_c.py -q
```

Expected recent result was `63 passed` when including `test_equipment_big.py`, and `40 passed` for production-risk + hazard-refit only.

Regenerate outputs:

```powershell
$env:PYTHONPATH='D:\GitHub\Pump2\backend;D:\GitHub\Pump2'
python scripts\run\production_risk.py --full-tables
```

Rebuild frozen package:

```powershell
pyinstaller Pump2ProductionRisk.spec --clean --noconfirm
python scripts\deploy\build_production_risk_launcher.py `
  --output dist\Pump2ProductionRisk\ProductionRiskLauncher.xlsm `
  --exe dist\Pump2ProductionRisk\Pump2ProductionRisk.exe `
  --prediction-workbook data\inputs\Отказы свод с анализом.xlsx
```

Optional packaged smoke run:

```powershell
.\dist\Pump2ProductionRisk\Pump2ProductionRisk.exe `
  --no-excel `
  --forecast-start 2026-07-01 `
  --horizon-end 2026-07-01 `
  --prediction-workbook data\inputs\Отказы свод с анализом.xlsx `
  --equipment-big D:\Projects\Pumps\data\big\WellsArtificialLiftBig.xlsx
```

For the smoke run, no `No module named 'scripts'` Big-loader warning should appear.

## Specific Data Checks

Check Kpod reporting labels:

```powershell
Import-Csv results\production_risk_forecast\2026-07-15\tables\A_kpod_by_field_monthly.csv |
  Group-Object field_label |
  Select-Object Name,Count |
  Sort-Object Name |
  Format-Table -AutoSize
```

Expected:
`Bt - Большетирское НМ` exists with monthly rows.
`Vt - Большетирское НМ` does not exist.

Check uncertainty stress behavior:

```powershell
@'
import pandas as pd
p='results/production_risk_forecast/2026-07-15/tables/A_production_at_risk_by_well.csv'
df=pd.read_csv(p)
stress=df[df['scenario'].eq('stress_p75')]
print(stress['theta_uncertainty'].describe().to_string())
print(stress.groupby('state_label')['theta_uncertainty'].agg(['min','max','mean','count']).to_string())
print('new max', stress.loc[stress['state_label'].str.startswith('new'), 'theta_uncertainty'].max())
print('exact max', stress.loc[~stress['state_label'].str.startswith('new'), 'theta_uncertainty'].max())
'@ | python -
```

Expected:
- new-launch rows have `theta_uncertainty > 1`
- exact/mature states have `theta_uncertainty == 1`
- recent check showed new max `1.3`, exact max `1.0`

Check launcher:

```powershell
@'
from openpyxl import load_workbook
p=r'D:\GitHub\Pump2\dist\Pump2ProductionRisk\ProductionRiskLauncher.xlsm'
wb=load_workbook(p, keep_vba=True, data_only=False)
ws=wb['Запуск']
for cell in ['B3','B8','B9']:
    print(cell, ws[cell].value)
wb.close()
'@ | python -
```

## Honest Validation Still Required

The uncertainty hazard is not estimated yet. It is currently a reasonable conservative sensitivity value.

Before promoting it beyond stress/sensitivity:

1. Build a historical pseudo-forecast dataset:
   - classify launches that would have been `new_plan_only`, `new_df_esp`, or `new_tr_esp` at forecast time;
   - compare them to known-age launches and exact active pumps.
2. Fit an age-decaying excess hazard:
   - estimate `u0`, `tau_days`, and caps by field if sample size permits.
3. Validate OOS:
   - monthly failure counts
   - by-field failure counts
   - calibration of 90-day risk
   - production loss / repair load impact
4. Confirm it does not just encode missing-data bias or workover scheduling artifacts.
5. Keep base P50 unchanged unless OOS improves without breaking Mc/Ya/Vt/global calibration.

## Final Verdict To Produce

End with:

- what passed;
- what failed or was fixed;
- whether the frozen EXE and launcher were rebuilt;
- exact paths to updated outputs;
- a clear statement that Kpod and uncertainty hazards remain stress sensitivity dials unless validated.
