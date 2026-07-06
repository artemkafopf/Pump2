# ESP VBA Prediction Modules -- Setup Guide

## Files

| File | Module name | Purpose |
|---|---|---|
| `mdlMath.bas` | `mdlMath` | WeibullSF, Gamma, Simpson integration, Bisection |
| `mdlLatentWeibull.bas` | `mdlLatentWeibull` | K=2 mixture SF, RUL, quantiles, hazard, component |
| `mdlModelRegistry.bas` | `mdlModelRegistry` | Parameter cache, stratum lookup, model sheet builder |
| `mdlPublicFunctions.bas` | `mdlPublicFunctions` | Worksheet UDFs (ESP_RUL, ESP_B50, ESP_Predict, ...) |
| `mdlBatchProcess.bas` | `mdlBatchProcess` | RunPredictions macro, ImportModelCSV, ExportSurvivalCurve |
| `ThisWorkbook_setup.bas` | ThisWorkbook | Auto-load registry on Workbook_Open |

## First-time setup

1. **Open the target workbook** as `.xlsm` (save a copy if it was `.xlsx`)

2. **Open VBA editor**: Alt+F11

3. **Import all standard modules** (File -> Import File):
   - `mdlMath.bas`
   - `mdlLatentWeibull.bas`
   - `mdlModelRegistry.bas`
   - `mdlPublicFunctions.bas`
   - `mdlBatchProcess.bas`

4. **Paste `ThisWorkbook_setup.bas` content** into the `ThisWorkbook` code module
   (double-click `ThisWorkbook` in the Project pane, paste the `Workbook_Open` sub)

5. **Create the model parameter sheet**: In the Immediate window (Ctrl+G) type:
   ```
   CreateModelSheet
   ```
   This creates the `ESP_Models` sheet pre-populated with all fitted parameters.

6. **Run batch predictions**: In the Immediate window:
   ```
   RunPredictions
   ```
   Or assign it to a button. Fills columns BY-CH in the data sheet.

## Worksheet UDF reference

All functions are callable in cells once the modules are imported.

| Formula | Description |
|---|---|
| `=ESP_RUL(I2, A2, BW2, D2)` | Expected remaining life for a running pump |
| `=ESP_B50(A2, BW2, D2)` | Model median TTF for a new pump |
| `=ESP_B10(A2, BW2, D2)` | 10th percentile TTF (early-risk bound) |
| `=ESP_B90(A2, BW2, D2)` | 90th percentile TTF (maintenance horizon) |
| `=ESP_TTF_Mean(A2, BW2, D2)` | Mean TTF (exact closed form) |
| `=ESP_SF(I2, A2, BW2, D2)` | Survival probability at current age |
| `=ESP_Hazard(I2, A2, BW2, D2)` | Hazard rate at current age (per day) |
| `=ESP_Component(I2, A2, BW2, D2)` | "C1 (early failure)" or "C2 (wear-out)" |
| `=ESP_Stratum(A2, BW2, D2)` | Stratum key used (for audit) |
| `=ESP_IsDegenerate(A2, BW2, D2)` | TRUE if the model is flagged degenerate |
| `=ESP_Predict(I2, BX2, A2, BW2, D2)` | Auto: RUL if running, B50 if failed, "" if suspended |

Column references (data sheet):
- **A** = Field
- **D** = Contractor (raw Russian name -- mapped internally to brt/slb/oth)
- **I** = Age in service (days)
- **BW** = H2S class (raw Russian cell value -- mapped internally to sour/nonsour)
- **BX** = Failure Flag (0=running, 1=failed, -1=suspended)

## Contractor group codes

The VBA maps raw contractor names from column D to short ASCII codes at runtime.
Source files contain no Cyrillic; matching uses `Chr()` code-point sequences.

| Code | Contractor |
|---|---|
| `brt` | Borec |
| `slb` | Schlumberger |
| `oth` | All others (nt, novomet, ink, ...) |
| `Pooled` | Field-level pooled (no contractor split) |
| `unk` | Blank / None (falls back to Pooled) |

## Updating model parameters

After running the Python pipeline to refit models:

```bash
cd D:\GitHub\Pump2
python scripts/run/export_model_csv_for_vba.py
```

Then in Excel (Immediate window or button):
```
ImportModelCSV "D:\GitHub\Pump2\results\esp_survival_vba_models\YYYY-MM-DD\esp_models.csv"
```

Or run `ImportModelCSV` with no argument to open a file picker.

## Lookup priority

For each row the VBA tries these stratum keys in order:

1. `{field}_{h2s}_{contractor_code}` -- e.g. `Ya_nonsour_brt`
2. `{field}_{h2s}_Pooled` -- e.g. `Ya_nonsour_Pooled`
3. `Global_Pooled` -- domain-knowledge fallback (beta=1.1, eta=41/500d)

Fields where contractor split is not modelled (Ic, Az, Mc, Da) only have `_Pooled`
rows, so the lookup naturally falls to priority 2 for them.

Rows with blank/None contractor fall directly to priority 2 (all 2,040 running
pumps with missing contractor data are handled this way).

## Output columns (written by RunPredictions)

| Column | Header | Content |
|---|---|---|
| BY | `ESP_Stratum` | Stratum key resolved |
| BZ | `ESP_B10` | 10th pct TTF (days) |
| CA | `ESP_B50` | Median TTF (days) |
| CB | `ESP_B90` | 90th pct TTF (days) |
| CC | `ESP_RUL` | RUL for running pumps |
| CD | `ESP_TTF_Pred` | Model B50 for failed pumps |
| CE | `ESP_SF_Now` | S(t) at current age |
| CF | `ESP_Component` | C1 or C2 at current age |
| CG | `ESP_Hazard` | Hazard rate at current age |
| CH | `ESP_Degenerate` | TRUE if model flagged degenerate |

## Validation spot-checks

Run these in a scratch cell to confirm the modules are working:

```
=ESP_RUL(30, "Vt", "sour", "slb")      -> ~100d  (slb  w1=0.135, eta1=9d)
=ESP_RUL(60, "Vt", "sour", "brt")      ->  ~78d
=ESP_B50("Ya", "nonsour", "brt")        -> ~403d
=ESP_B50("Ya", "nonsour", "oth")        -> ~174d  (Novomet -- high w1=0.618)
=ESP_B10("Vt", "sour", "slb")          ->   ~5d  (acute H2S mode)
=ESP_Stratum("Ic", "nonsour", "brt")   -> "Ic_nonsour_Pooled"
=ESP_Stratum("Bt", "nonsour", "brt")   -> "Global_Pooled"
```

Note: the UDFs accept either the raw Russian cell value (e.g. from `=D2`) **or**
the Latin code directly. Both work because `ContractorGroup()` passes unknown
strings straight to `"oth"`, while codes like `"brt"` and `"slb"` are not
Cyrillic and will also fall to `"oth"` -- so pass raw cell references for correct
contractor matching; use the Latin codes only in direct spot-checks if you
temporarily hard-code a Latin string.
