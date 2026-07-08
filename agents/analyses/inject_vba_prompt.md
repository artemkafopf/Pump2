# Prompt for Codex — inject the ESP VBA modules into the workbook automatically

## Goal

Write a Windows Python script that programmatically loads the ESP prediction VBA
modules from `D:\GitHub\Pump2\vba\` into the target Excel workbook, replacing any old
copies, injects the `ThisWorkbook` auto-load code, imports the model bundle, runs the
validation macro, and saves — so a human never has to remove/re-import modules by
hand. It must be idempotent (safe to run repeatedly).

## Target files

- Workbook (input file): `D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsm`
  (must be `.xlsm`; if it is `.xlsx`, save a `.xlsm` copy first).
- VBA source folder: `D:\GitHub\Pump2\vba\`
- Model+hazard bundle folder (latest dated): pick the newest
  `D:\GitHub\Pump2\results\esp_survival_vba_models\<YYYY-MM-DD>\` that contains
  `esp_models.csv` (currently `2026-07-08`).

## Modules

Import these as standard modules (`.bas`), removing any existing component of the same
name first: `mdlMath`, `mdlLatentWeibull`, `mdlModelRegistry`, `mdlModeMix`,
`mdlPublicFunctions`, `mdlCoxHR`, `mdlHazardLayer`, `mdlBatchProcess`,
`mdlValidation`, `mdlModelSeed`.

`ThisWorkbook_setup.bas` is NOT importable as a module — it is document-module code.
Inject its body into the `ThisWorkbook` component's code module instead (see below).

## Approach (use pywin32 / win32com — the VBProject object model)

```python
import win32com.client as win32   # pip install pywin32
```

1. **Preconditions & guardrails.**
   - Require Excel installed. Fail with a clear message if `win32com` is missing.
   - The script needs "Trust access to the VBA project object model" enabled
     (Excel > File > Options > Trust Center > Trust Center Settings > Macro Settings).
     Detect the `AccessVBOM` registry value under
     `HKCU\Software\Microsoft\Office\<ver>\Excel\Security` and, if 0, print exact
     instructions to enable it (do not silently flip it unless a `--enable-vbom` flag
     is passed; if passed, set it to 1 for the detected Office version).
   - **Back up** the workbook to a timestamped copy before touching it.
   - Refuse to run if the workbook is open read-only.

2. **Open Excel hidden, macros enabled.**
   ```python
   app = win32.DispatchEx("Excel.Application")
   app.Visible = False
   app.DisplayAlerts = False
   app.AutomationSecurity = 3          # msoAutomationSecurityForceDisable? -> use 1 (Low) so macros run
   wb = app.Workbooks.Open(path, ReadOnly=False)
   ```
   (Use `app.AutomationSecurity = 1` = msoAutomationSecurityLow so `Application.Run`
   can execute the imported macros.)

3. **Remove old ESP components** (idempotent). Iterate `wb.VBProject.VBComponents`;
   for any whose `.Name` is in the module list above (Type `vbext_ct_StdModule` = 1),
   call `wb.VBProject.VBComponents.Remove(comp)`. Do this in a collected pass (don't
   mutate while iterating). Do NOT remove document modules or `ThisWorkbook`.

4. **Import the `.bas` modules** via
   `wb.VBProject.VBComponents.Import(r"D:\GitHub\Pump2\vba\<name>.bas")` for each module
   in the list (skip `ThisWorkbook_setup.bas`).

5. **Inject the `ThisWorkbook` code.** Read `ThisWorkbook_setup.bas`, strip the
   `Attribute VB_Name = "ThisWorkbook"` line and any leading `Attribute`/`Option`
   lines, and replace the ThisWorkbook module body:
   ```python
   cm = wb.VBProject.VBComponents("ThisWorkbook").CodeModule
   cm.DeleteLines(1, cm.CountOfLines)
   cm.AddFromString(workbook_open_sub_text)   # the Private Sub Workbook_Open ... End Sub
   ```
   (Only the `Workbook_Open` sub — the three `Call Load...` lines.)

6. **Import the bundle + validate.** Run the macros via `Application.Run`:
   ```python
   app.Run("ImportBundle", r"D:\GitHub\Pump2\results\esp_survival_vba_models\2026-07-08")
   app.Run("ValidateRegistry")
   # optionally: app.Run("RunPredictions")
   ```
   `ImportBundle`/`ValidateRegistry` write to the `ESP_Status` / `ESP_Validation`
   sheets (never MsgBox from those paths), so automation will not block. If the
   workbook is protected/read-only the macros' own `PrepWorkbook` guard handles it.

7. **Read back the result.** After `ValidateRegistry`, read the `ESP_Validation`
   sheet's used range and print the pass/fail table to stdout (and detect the SUMMARY
   row = "ALL GREEN"/"…FAILURE(S)"). Read `ESP_Status` last few rows too.

8. **Save & close.** `wb.Save()`, `wb.Close(SaveChanges=True)`, `app.Quit()`, and
   release COM objects (`del`, `pythoncom.CoUninitialize()` if used). Wrap everything
   in try/finally so Excel never leaks a hidden process; on error, print the traceback
   and still Quit.

## CLI

```
python scripts/deploy/inject_vba.py [--workbook PATH] [--vba-dir PATH]
                                    [--bundle PATH] [--run-predictions]
                                    [--enable-vbom] [--no-validate]
```
Defaults to the paths above; `--bundle` defaults to the newest dated folder with
`esp_models.csv`. Exit non-zero if `ValidateRegistry` is not all-green.

## Constraints

- Windows + desktop Excel only (COM). State this in the module docstring and fail
  fast elsewhere.
- Do not edit the `.bas` files — they are the source of truth (generated seed
  included). Only import them.
- Keep it a thin deploy script under `scripts/deploy/`; no changes to the analysis
  code. Use `analysis.paths`-style discovery only if convenient, else hardcode the
  documented paths with CLI overrides.
- Idempotent, backup-first, never leave a stray Excel process.

## Done when

Running the script on a clean workbook removes old ESP modules, imports the current
`vba/*.bas`, wires `ThisWorkbook`, imports the `2026-07-08` bundle, runs
`ValidateRegistry`, prints an all-green `ESP_Validation` table, and saves the `.xlsm`
— repeatable without manual VBA-editor steps.
