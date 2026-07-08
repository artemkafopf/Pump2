"""Deploy ESP VBA modules into the target Excel workbook via COM automation.

Windows + desktop Excel only. This script:
  - backs up the workbook,
  - replaces the ESP VBA modules from ``vba/*.bas``,
  - injects ``ThisWorkbook`` startup code from ``ThisWorkbook_setup.bas``,
  - imports the newest ESP VBA bundle,
  - runs ``ValidateRegistry`` (and optionally ``RunPredictions``),
  - prints the ``ESP_Validation`` / ``ESP_Status`` sheet contents,
  - saves and closes the workbook cleanly.

The script is intentionally thin: the ``.bas`` files remain the single source of truth.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VBA_DIR = REPO_ROOT / "vba"
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "results" / "esp_survival_vba_models"
DEFAULT_WORKBOOK = Path(
    r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsm"
)
MODULE_NAMES = [
    "mdlMath",
    "mdlLatentWeibull",
    "mdlModelRegistry",
    "mdlModeMix",
    "mdlPublicFunctions",
    "mdlCoxHR",
    "mdlHazardLayer",
    "mdlBatchProcess",
    "mdlValidation",
    "mdlModelSeed",
]
THISWORKBOOK_BAS = "ThisWorkbook_setup.bas"
AUTOMATION_SECURITY_LOW = 1
VBEXT_CT_STDMODULE = 1
FILE_FORMAT_XLSM = 52


@dataclass
class ValidationSummary:
    rows: list[tuple[str, ...]]
    all_green: bool
    summary_detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject the Pump2 ESP VBA modules into the target workbook, "
            "import the newest bundle, validate, and save."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="Target workbook path (.xlsm preferred; .xlsx will be converted to .xlsm).",
    )
    parser.add_argument(
        "--vba-dir",
        type=Path,
        default=DEFAULT_VBA_DIR,
        help="Folder containing the VBA .bas files.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Specific bundle folder. Defaults to the newest dated folder with esp_models.csv.",
    )
    parser.add_argument(
        "--run-predictions",
        action="store_true",
        help="Run RunPredictions after validation.",
    )
    parser.add_argument(
        "--enable-vbom",
        action="store_true",
        help="Enable 'Trust access to the VBA project object model' for the detected Office version.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ValidateRegistry and the validation summary check.",
    )
    return parser.parse_args()


def fail_fast_environment() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This script only works on Windows with desktop Excel installed.")


def import_excel_modules():
    try:
        import pythoncom  # type: ignore
        import pywintypes  # type: ignore
        import win32com.client as win32  # type: ignore
        import winreg
    except ImportError as exc:
        raise RuntimeError(
            "pywin32 is required. Install it with: pip install pywin32"
        ) from exc
    return pythoncom, pywintypes, win32, winreg


def newest_bundle(bundle_root: Path) -> Path:
    if not bundle_root.exists():
        raise FileNotFoundError(f"Bundle root not found: {bundle_root}")
    candidates = []
    for child in bundle_root.iterdir():
        if child.is_dir() and (child / "esp_models.csv").exists():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No dated bundle folders with esp_models.csv found under {bundle_root}"
        )
    return max(candidates, key=lambda p: p.name)


def ensure_vba_sources(vba_dir: Path) -> None:
    missing = [name for name in MODULE_NAMES if not (vba_dir / f"{name}.bas").exists()]
    setup_path = vba_dir / THISWORKBOOK_BAS
    if not setup_path.exists():
        missing.append(THISWORKBOOK_BAS)
    if missing:
        raise FileNotFoundError(f"Missing VBA source files in {vba_dir}: {', '.join(missing)}")


def ensure_bundle_files(bundle_dir: Path) -> None:
    required = ["esp_models.csv", "esp_mode_mix.csv", "mode_group_map.csv"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing bundle files in {bundle_dir}: {', '.join(missing)}")


def create_excel_app(win32: Any):
    app = win32.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = AUTOMATION_SECURITY_LOW
    return app


def access_vbom_value(winreg: Any, office_version: str) -> int:
    key_path = rf"Software\Microsoft\Office\{office_version}\Excel\Security"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "AccessVBOM")
            return int(value)
    except FileNotFoundError:
        return 0


def set_access_vbom(winreg: Any, office_version: str, enabled: bool) -> None:
    key_path = rf"Software\Microsoft\Office\{office_version}\Excel\Security"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        winreg.SetValueEx(key, "AccessVBOM", 0, winreg.REG_DWORD, 1 if enabled else 0)


def vbom_instructions(office_version: str) -> str:
    return (
        f"'Trust access to the VBA project object model' is disabled for Office {office_version}.\n"
        "Enable it in Excel:\n"
        "  File > Options > Trust Center > Trust Center Settings > Macro Settings > "
        "Trust access to the VBA project object model\n"
        "Or rerun this script with --enable-vbom to set AccessVBOM=1 automatically."
    )


def ensure_vbom_access(app: Any, winreg: Any, enable_vbom: bool) -> tuple[Any, str]:
    office_version = str(app.Version)
    if access_vbom_value(winreg, office_version) == 1:
        return app, office_version
    if not enable_vbom:
        raise RuntimeError(vbom_instructions(office_version))
    set_access_vbom(winreg, office_version, enabled=True)
    app.Quit()
    app = None
    print(f"Enabled AccessVBOM for Office {office_version}. Relaunching Excel...")
    return None, office_version


def resolve_target_workbook(workbook_path: Path) -> tuple[Path, Path | None]:
    workbook_path = workbook_path.expanduser()
    suffix = workbook_path.suffix.lower()
    if suffix not in {".xlsm", ".xlsx"}:
        raise ValueError(f"Workbook must be .xlsm or .xlsx, got: {workbook_path}")

    if suffix == ".xlsx":
        if not workbook_path.exists():
            raise FileNotFoundError(f"Workbook not found: {workbook_path}")
        return workbook_path.with_suffix(".xlsm"), workbook_path

    if workbook_path.exists():
        return workbook_path, None

    xlsx_peer = workbook_path.with_suffix(".xlsx")
    if xlsx_peer.exists():
        return workbook_path, xlsx_peer
    raise FileNotFoundError(f"Workbook not found: {workbook_path}")


def backup_workbook(workbook_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = workbook_path.with_name(f"{workbook_path.stem}.backup_{stamp}{workbook_path.suffix}")
    shutil.copy2(workbook_path, backup)
    return backup


def ensure_xlsm_copy(app: Any, target_xlsm: Path, source_xlsx: Path | None) -> None:
    if source_xlsx is None:
        return
    if target_xlsm.exists():
        print(f"Using existing macro-enabled workbook: {target_xlsm}")
        return
    print(f"Creating macro-enabled copy: {target_xlsm}")
    wb = app.Workbooks.Open(str(source_xlsx), ReadOnly=False)
    try:
        wb.SaveAs(str(target_xlsm), FileFormat=FILE_FORMAT_XLSM)
    finally:
        wb.Close(SaveChanges=False)


def open_workbook(app: Any, workbook_path: Path):
    wb = app.Workbooks.Open(
        str(workbook_path),
        ReadOnly=False,
        IgnoreReadOnlyRecommended=True,
        Notify=False,
    )
    if wb.ReadOnly:
        wb.Close(SaveChanges=False)
        raise RuntimeError(f"Workbook opened read-only and cannot be modified: {workbook_path}")
    return wb


def remove_old_modules(vbproject: Any) -> list[str]:
    to_remove = []
    for index in range(1, vbproject.VBComponents.Count + 1):
        comp = vbproject.VBComponents.Item(index)
        if comp.Name in MODULE_NAMES and comp.Type == VBEXT_CT_STDMODULE:
            to_remove.append(comp)
    removed = [comp.Name for comp in to_remove]
    for comp in to_remove:
        vbproject.VBComponents.Remove(comp)
    return removed


def import_modules(vbproject: Any, vba_dir: Path) -> list[str]:
    imported: list[str] = []
    for name in MODULE_NAMES:
        vbproject.VBComponents.Import(str(vba_dir / f"{name}.bas"))
        imported.append(name)
    return imported


def extract_workbook_open(setup_path: Path) -> str:
    text = setup_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cleaned: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Private Sub Workbook_Open()"):
            in_block = True
        if in_block:
            cleaned.append(line)
            if stripped == "End Sub":
                break
    if not cleaned:
        raise RuntimeError(f"Could not find Workbook_Open in {setup_path}")
    return "\n".join(cleaned) + "\n"


def inject_thisworkbook(vbproject: Any, setup_path: Path) -> None:
    workbook_open = extract_workbook_open(setup_path)
    code_module = vbproject.VBComponents("ThisWorkbook").CodeModule
    line_count = int(code_module.CountOfLines)
    if line_count > 0:
        code_module.DeleteLines(1, line_count)
    code_module.AddFromString(workbook_open)


def run_macro(app: Any, workbook_name: str, macro_name: str, *args: Any) -> Any:
    qualified = f"'{workbook_name}'!{macro_name}"
    return app.Run(qualified, *args)


def normalize_rows(values: Any) -> list[tuple[str, ...]]:
    if values is None:
        return []
    if not isinstance(values, tuple):
        return [(str(values),)]
    if values and not isinstance(values[0], tuple):
        return [tuple("" if cell is None else str(cell) for cell in values)]
    rows = []
    for row in values:
        rows.append(tuple("" if cell is None else str(cell) for cell in row))
    return rows


def read_sheet_rows(wb: Any, sheet_name: str) -> list[tuple[str, ...]]:
    ws = wb.Worksheets(sheet_name)
    return normalize_rows(ws.UsedRange.Value)


def print_table(title: str, rows: Iterable[tuple[str, ...]]) -> None:
    rows = list(rows)
    print(f"\n{title}")
    if not rows:
        print("  <empty>")
        return
    width_count = max(len(row) for row in rows)
    widths = [0] * width_count
    for row in rows:
        for idx in range(width_count):
            cell = row[idx] if idx < len(row) else ""
            widths[idx] = max(widths[idx], len(cell))
    for row in rows:
        padded = []
        for idx in range(width_count):
            cell = row[idx] if idx < len(row) else ""
            padded.append(cell.ljust(widths[idx]))
        print("  " + " | ".join(padded).rstrip())


def summarize_validation(rows: list[tuple[str, ...]]) -> ValidationSummary:
    summary_detail = ""
    all_green = False
    for row in rows:
        if len(row) >= 3 and row[0].strip().upper() == "SUMMARY":
            summary_detail = row[1].strip()
            all_green = row[2].strip().upper() == "PASS" and summary_detail.upper() == "ALL GREEN"
            break
    return ValidationSummary(rows=rows, all_green=all_green, summary_detail=summary_detail)


def tail_rows(rows: list[tuple[str, ...]], count: int = 8) -> list[tuple[str, ...]]:
    return rows[-count:] if len(rows) > count else rows


def deploy(args: argparse.Namespace) -> int:
    fail_fast_environment()
    pythoncom, pywintypes, win32, winreg = import_excel_modules()

    workbook_target, workbook_source = resolve_target_workbook(args.workbook)
    vba_dir = args.vba_dir.expanduser().resolve()
    bundle_dir = (args.bundle.expanduser().resolve() if args.bundle else newest_bundle(DEFAULT_BUNDLE_ROOT))

    ensure_vba_sources(vba_dir)
    ensure_bundle_files(bundle_dir)

    pythoncom.CoInitialize()
    app = None
    wb = None
    saved = False
    validation_summary: ValidationSummary | None = None

    try:
        app = create_excel_app(win32)
        app, office_version = ensure_vbom_access(app, winreg, args.enable_vbom)
        if app is None:
            app = create_excel_app(win32)
        print(f"Excel version detected: {office_version}")
        app.EnableEvents = False

        ensure_xlsm_copy(app, workbook_target, workbook_source)
        backup = backup_workbook(workbook_target)
        print(f"Backup created: {backup}")

        wb = open_workbook(app, workbook_target)
        vbproject = wb.VBProject

        removed = remove_old_modules(vbproject)
        if removed:
            print("Removed old modules: " + ", ".join(removed))
        imported = import_modules(vbproject, vba_dir)
        print("Imported modules: " + ", ".join(imported))
        inject_thisworkbook(vbproject, vba_dir / THISWORKBOOK_BAS)
        print("Injected ThisWorkbook Workbook_Open code.")
        app.EnableEvents = True

        run_macro(app, wb.Name, "ImportBundle", str(bundle_dir))
        print(f"ImportBundle completed from: {bundle_dir}")

        if not args.no_validate:
            run_macro(app, wb.Name, "ValidateRegistry")
            validation_rows = read_sheet_rows(wb, "ESP_Validation")
            validation_summary = summarize_validation(validation_rows)
            print_table("ESP_Validation", validation_rows)
            status_rows = tail_rows(read_sheet_rows(wb, "ESP_Status"))
            print_table("ESP_Status (tail)", status_rows)
        if args.run_predictions:
            run_macro(app, wb.Name, "RunPredictions")
            print("RunPredictions completed.")

        wb.Save()
        saved = True
        print(f"Saved workbook: {workbook_target}")

        if validation_summary is not None and not validation_summary.all_green:
            print(
                "ValidateRegistry did not report ALL GREEN. "
                f"Summary detail: {validation_summary.summary_detail or '<missing>'}"
            )
            return 1
        return 0

    except pywintypes.com_error as exc:  # type: ignore[attr-defined]
        print("COM automation error:")
        print(exc)
        traceback.print_exc()
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        return 2
    finally:
        if wb is not None:
            try:
                wb.Close(SaveChanges=bool(saved))
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


def main() -> int:
    args = parse_args()
    return deploy(args)


if __name__ == "__main__":
    raise SystemExit(main())
