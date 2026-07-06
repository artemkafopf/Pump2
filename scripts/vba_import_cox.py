"""Import updated VBA modules into the ESP prediction workbook and validate Cox layer.

Steps:
    1. Open the workbook (or connect to already-open instance).
    2. Remove stale copies of the four affected modules.
    3. Import fresh .bas files from vba/.
    4. Run CreateCoxSheet() to populate the ESP_CoxCoeffs sheet.
    5. Validate: ESP_Theta(-0.342, 110.3, 2.249) must equal 1.000.

Usage:
    python scripts/vba_import_cox.py

Requirements:
    pip install pywin32
    Excel must be allowed to run macros (security level Medium or lower).
    Trust access to the VBA project object model must be enabled:
        File → Options → Trust Center → Trust Center Settings →
        Macro Settings → check "Trust access to the VBA project object model".
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output so box-drawing characters and Cyrillic print correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATH = Path(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx")
VBA_DIR = REPO_ROOT / "vba"

# Modules to replace (name in VBA project → .bas file)
MODULES_TO_IMPORT: list[tuple[str, Path]] = [
    ("mdlCoxHR",       VBA_DIR / "mdlCoxHR.bas"),
    ("mdlBatchProcess", VBA_DIR / "mdlBatchProcess.bas"),
    ("mdlModelRegistry", VBA_DIR / "mdlModelRegistry.bas"),
]

# Modules that must NOT be touched (math primitives and public UDFs)
PROTECTED_MODULES = {"mdlMath", "mdlLatentWeibull", "mdlPublicFunctions"}


def _connect_excel():
    import win32com.client as win32
    try:
        xl = win32.GetActiveObject("Excel.Application")
        print("Connected to running Excel instance.")
    except Exception:
        xl = win32.Dispatch("Excel.Application")
        xl.Visible = True
        print("Started new Excel instance.")
    return xl


def _open_or_get_workbook(xl, path: Path):
    wb_name = path.name
    for wb in xl.Workbooks:
        if wb.Name.lower() == wb_name.lower():
            print(f"Workbook already open: {wb.Name}")
            return wb
    print(f"Opening: {path}")
    wb = xl.Workbooks.Open(str(path))
    time.sleep(2)
    return wb


def _remove_module(vba_project, module_name: str) -> bool:
    for i in range(1, vba_project.VBComponents.Count + 1):
        comp = vba_project.VBComponents.Item(i)
        if comp.Name == module_name:
            vba_project.VBComponents.Remove(comp)
            print(f"  Removed existing module: {module_name}")
            return True
    return False


def _import_module(vba_project, bas_path: Path) -> None:
    vba_project.VBComponents.Import(str(bas_path))
    print(f"  Imported: {bas_path.name}")


def main() -> None:
    # Sanity checks
    for _, bas in MODULES_TO_IMPORT:
        if not bas.exists():
            print(f"ERROR: Missing .bas file: {bas}")
            sys.exit(1)

    import win32com.client as win32

    xl = _connect_excel()
    wb = _open_or_get_workbook(xl, WORKBOOK_PATH)

    vba_project = wb.VBProject

    print("\n── Replacing VBA modules ─────────────────────────────────────")
    for mod_name, bas_path in MODULES_TO_IMPORT:
        if mod_name in PROTECTED_MODULES:
            print(f"  SKIP (protected): {mod_name}")
            continue
        _remove_module(vba_project, mod_name)
        _import_module(vba_project, bas_path)

    print("\n── Running CreateCoxSheet() ──────────────────────────────────")
    wb_name = wb.Name
    for macro_name in (
        f"'{wb_name}'!mdlModelRegistry.CreateCoxSheet",
        f"'{wb_name}'!CreateCoxSheet",
        "mdlModelRegistry.CreateCoxSheet",
        "CreateCoxSheet",
    ):
        try:
            xl.Application.Run(macro_name)
            print(f"  CreateCoxSheet() executed via: {macro_name}")
            break
        except Exception:
            continue
    else:
        print("  WARNING: Could not run CreateCoxSheet() via COM automation.")
        print("  Run manually: Alt+F8 → CreateCoxSheet → Run")

    print("\n── Validation ───────────────────────────────────────────────")
    # Put the validation formula in a temporary cell on Sheet1 or the active sheet
    try:
        ws_temp = wb.Worksheets(1)
        temp_cell = ws_temp.Cells(1, 200)   # far-right, out of the way
        temp_cell.Formula = "=ESP_Theta(-0.342, 110.3, 2.249)"
        xl.Application.Calculate()
        time.sleep(0.5)
        val = float(temp_cell.Value)
        temp_cell.Clear()
        if abs(val - 1.0) < 0.001:
            print(f"  PASS: ESP_Theta(-0.342, 110.3, 2.249) = {val:.6f}  (expected 1.000)")
        else:
            print(f"  FAIL: ESP_Theta(-0.342, 110.3, 2.249) = {val:.6f}  (expected 1.000)")
    except Exception as exc:
        print(f"  Validation cell test failed: {exc}")
        print("  Verify manually: in any cell type =ESP_Theta(-0.342, 110.3, 2.249)")
        print("  Expected result: 1.000")

    print("\n── Saving workbook ──────────────────────────────────────────")
    try:
        wb.Save()
        print(f"  Saved: {WORKBOOK_PATH.name}")
    except Exception as exc:
        print(f"  Save failed (workbook may be read-only or format issue): {exc}")
        print("  Save manually with Ctrl+S.")

    print("\nDone.")


if __name__ == "__main__":
    main()
