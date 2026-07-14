"""Build the macro-enabled Excel launcher for the packaged production-risk workflow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for item in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from analysis.paths import (  # noqa: E402
    resolve_equipment_big_path,
    resolve_gtm_schedule_path,
    resolve_pp_master_path,
    resolve_prediction_workbook_path,
    resolve_techregime_workbook_path,
)
from scripts.deploy.inject_vba import (  # noqa: E402
    create_excel_app,
    ensure_vbom_access,
    fail_fast_environment,
    import_excel_modules,
)


DEFAULT_OUTPUT = REPO_ROOT / "dist" / "Pump2ProductionRisk" / "ProductionRiskLauncher.xlsm"
DEFAULT_MODULE = REPO_ROOT / "vba" / "mdlProductionRiskLauncher.bas"
MODEL_CALIBRATION_NOTE = (
    "ВНИМАНИЕ: прогноз отказов содержит явно отмеченные ручные поправки "
    "(Mc survival-weight, Ya/Vt и УН-калибровка; глобально = сумма УН)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Pump2 Excel launcher.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--pp-master", type=Path, default=None)
    parser.add_argument("--gtm-schedule", type=Path, default=None)
    parser.add_argument("--prediction-workbook", type=Path, default=None)
    parser.add_argument("--techregime-workbook", type=Path, default=None)
    parser.add_argument("--equipment-big", type=Path, default=None)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--enable-vbom", action="store_true")
    return parser.parse_args()


def set_value(ws, address: str, value, *, bold: bool = False) -> None:
    cell = ws.Range(address)
    if address in {"B6", "B7"}:
        cell.NumberFormat = "@"
    cell.Value = value
    cell.Font.Bold = bold


def build_sheet(ws, values: dict[str, object]) -> None:
    ws.Name = "Запуск"
    ws.Cells.Clear()
    ws.Range("A1:B22").Font.Name = "Calibri"
    ws.Range("A1:B22").Font.Size = 11

    labels = {
        "A1": "Файл 1: план добычи",
        "A2": "Файл 2: график ГТМ / ДФ",
        "A3": "Файл 3: прогноз / лист Свод",
        "A4": "Файл 4: текущий техрежим",
        "A5": "Простой ремонта, сут",
        "A6": "Начало прогноза",
        "A7": "Последний месяц",
        "A8": "Исполняемый файл",
        "A9": "Файл 5: WellsArtificialLiftBig",
        "A10": "Статус",
        "A11": "Время обновления",
        "A12": "Последняя команда",
        "A13": "Папка результатов",
        "A14": "Поправки модели",
    }
    for address, label in labels.items():
        set_value(ws, address, label, bold=True)
    for address, value in values.items():
        set_value(ws, address, value)

    ws.Range("B5").ClearContents()
    ws.Range("B6:B7").NumberFormat = "@"
    ws.Range("B11").NumberFormat = "dd.mm.yyyy hh:mm:ss"
    ws.Range("B10").Value = "Ready"
    ws.Range("B14").Value = MODEL_CALIBRATION_NOTE
    ws.Range("B12:B14").WrapText = True
    ws.Range("A1:A14").Interior.Color = 0xE2F0D9
    ws.Range("B1:B9").Interior.Color = 0xFFF2CC
    ws.Range("B14").Interior.Color = 0xFCE4D6
    ws.Range("A1:A14").Borders.LineStyle = 1
    ws.Range("B1:B14").Borders.LineStyle = 1
    ws.Columns("A").ColumnWidth = 29
    ws.Columns("B").ColumnWidth = 95
    ws.Rows("1:14").RowHeight = 23
    ws.Rows("12:14").RowHeight = 46

    button = ws.Shapes.AddShape(1, ws.Range("A17").Left, ws.Range("A17").Top, ws.Range("A17:B18").Width, 42)
    button.Name = "btnRunProductionRisk"
    button.TextFrame.Characters().Text = "Run forecast"
    button.OnAction = "RunProductionRisk"
    button.Fill.ForeColor.RGB = 0x4F81BD
    button.Line.ForeColor.RGB = 0x385D8A
    button.TextFrame.Characters().Font.Color = 0xFFFFFF
    button.TextFrame.Characters().Font.Bold = True
    button.TextFrame.HorizontalAlignment = -4108
    ws.Range("A1").Select()


def main() -> int:
    args = parse_args()
    fail_fast_environment()
    pythoncom, _, win32, winreg = import_excel_modules()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exe_path = (args.exe or (output.parent / "Pump2ProductionRisk.exe")).expanduser().resolve()
    module_path = args.module.expanduser().resolve()
    if not exe_path.exists():
        raise FileNotFoundError(exe_path)
    if not module_path.exists():
        raise FileNotFoundError(module_path)

    values = {
        "B1": str((args.pp_master or resolve_pp_master_path()).resolve()),
        "B2": str((args.gtm_schedule or resolve_gtm_schedule_path()).resolve()),
        "B3": str((args.prediction_workbook or resolve_prediction_workbook_path()).resolve()),
        "B4": str((args.techregime_workbook or resolve_techregime_workbook_path()).resolve()),
        "B6": "2026-07-01",
        "B7": "2027-12-01",
        "B8": str(exe_path),
        "B9": str((args.equipment_big or resolve_equipment_big_path()).resolve()),
    }

    pythoncom.CoInitialize()
    app = None
    workbook = None
    try:
        app = create_excel_app(win32)
        app, _ = ensure_vbom_access(app, winreg, args.enable_vbom)
        if app is None:
            app = create_excel_app(win32)
        app.EnableEvents = False

        workbook = app.Workbooks.Add()
        while workbook.Worksheets.Count > 1:
            workbook.Worksheets(workbook.Worksheets.Count).Delete()
        build_sheet(workbook.Worksheets(1), values)

        workbook.VBProject.VBComponents.Import(str(module_path))

        if output.exists():
            output.unlink()
        workbook.SaveAs(str(output), FileFormat=52)
        print(f"Built Excel launcher: {output}")
        return 0
    finally:
        if workbook is not None:
            try:
                workbook.Close(SaveChanges=False)
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    raise SystemExit(main())
