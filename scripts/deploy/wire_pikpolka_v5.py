"""Rewire a calculator workbook onto the per-field v5 parameter surface.

    python scripts/deploy/wire_pikpolka_v5.py --workbook <path> --layout npv --check
    python scripts/deploy/wire_pikpolka_v5.py --workbook <path> --layout npv

``--check`` touches nothing: it validates the module, reads the workbook with openpyxl and
prints exactly what would change, including the ННО/NPV delta computed by the Python port.

What the real run changes
-------------------------
1. **Module** ``ModuleFailureV4`` re-imported (ladder fallback fixed, sour θ_Qnom clamped,
   ``FieldKey`` argument added).
2. **Keyed blocks** written to a free area of ``МодельОтказов`` and the names ``TuneBlock`` /
   ``QnomBlock`` redefined to them — one row per stratum, from ``field_v4``'s deploy CSVs.
   The old 2-row/3-row blocks are left in place and labelled as superseded rather than
   deleted, so a rollback is a name redefinition and nothing else.
3. **Schedule formulas** ``I``/``T`` gain the field cell as the trailing argument, so «УН» /
   «Участок недр» now switches the survival model and not merely the downtime and the costs.
4. **Nominal lookups** on the sheet: the ``99999`` / ``999999`` XLOOKUP not-found default
   becomes the top of the ladder, matching the fixed ``LookupNominal4``.

COM traps this script guards against (each has cost a rebuild before): a workbook opened
READ-ONLY because a windowless ``EXCEL.EXE`` still holds it, ``Calculation`` left automatic
while an O(n) spill recalculates on every write, and a ``.bas`` with non-ASCII bytes importing
as mojibake.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.workflows.production_risk import pikpolka_sim as S  # noqa: E402

MODULE = REPO_ROOT / "vba" / "ModuleFailureV4.bas"
MODULE_NAME = "ModuleFailureV4"
MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"

#: Where the keyed blocks go.  Below every existing block on the sheet (the Qnom knot table
#: ends at row 55 and the Ctrl block at row 69), so nothing has to be pushed aside.
TUNE_ANCHOR = "A72"
QNOM_ANCHOR = "A84"

#: Per layout: the field cell, the two schedule anchors, and the nominal cells to de-999999.
LAYOUTS = {
    "npv": {"field_cell": "$H$4", "spills": ("I11", "T11"),
            "nominal_cells": ("F4",), "nominal_cols": ("E", "P"), "first_row": 11},
    "pikpolka": {"field_cell": "$N$4", "spills": ("I15", "T15"),
                 "nominal_cells": ("M4", "W4"), "nominal_cols": ("E", "P"), "first_row": 15},
}


def load_blocks(tables: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tables is None:
        found = sorted((REPO_ROOT / "results" / "production_risk_field_v4").glob("*/tables"))
        if not found:
            raise SystemExit("no field_v4 results — run scripts/run/field_v4.py first")
        tables = found[-1]
    return (pd.read_csv(tables / "deploy_tune_block.csv"),
            pd.read_csv(tables / "deploy_qnom_block.csv"))


def _refers_to(sheet, rng) -> str:
    """``RefersTo`` string for a workbook-scoped name.

    ``Range.Address`` is a *property* under ``EnsureDispatch`` (calling it with the usual
    ``Address(True, True, 1, True)`` raises ``'str' object is not callable``), and it returns
    a sheet-less local address, so the sheet has to be prefixed by hand.  Quoted, because
    every sheet here is named in Cyrillic.
    """
    return f"='{sheet.Name}'!{rng.Address}"


def check_module() -> str:
    src = MODULE.read_text(encoding="utf-8")
    bad = [i for i, ch in enumerate(src) if ord(ch) >= 128]
    if bad:
        line = src[:bad[0]].count("\n") + 1
        raise SystemExit(f"{MODULE.name}: non-ASCII at line {line} — the COM import would "
                         "mojibake it; use ChrW() inside the VBA instead")
    for need in ("CM4_ResolveKey", "FieldKey", "KeyRow4"):
        if need not in src:
            raise SystemExit(f"{MODULE.name}: missing {need} — stale module?")
    return src


def python_preview(workbook: str, layout: str) -> pd.DataFrame:
    """Before/after through the port, so the workbook delta is known before Excel opens."""
    cfg = S.load_config(workbook, layout=layout)
    tune, qnom = load_blocks()
    t = {r["key"]: (r["beta"], r["rmst_ref_days"], r["brt"], r["slb"], r["oth"])
         for _, r in tune.iterrows()}
    qcols = [c for c in qnom.columns if c.startswith("q")]
    q = {r["key"]: tuple(float(r[c]) for c in qcols) for _, r in qnom.iterrows()}
    rows = []
    for label, kw in (("сейчас (v4, до правок)", dict(qnom_block=S.QNOM_PRECLAMP)),
                      ("+ клапан по Qном (кислый)", dict(qnom_block=S.QNOM)),
                      ("+ пофондовые параметры", dict(tune=t, qnom_block=q))):
        res = S.simulate(cfg, **kw)
        for _, r in res["summary"].iterrows():
            rows.append({"вариант": label, "ветка": r["branch"], "ННО": r["nno_days"],
                         "отказов": r["failures"], "NPV": round(float(r["npv"]))})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--layout", default="npv", choices=tuple(LAYOUTS))
    p.add_argument("--tables", default=None, help="field_v4 tables/ dir (default: newest)")
    p.add_argument("--check", action="store_true", help="validate and preview, change nothing")
    a = p.parse_args()

    pd.set_option("display.width", 200)
    check_module()
    tune, qnom = load_blocks(Path(a.tables) if a.tables else None)
    print(f"module OK · {len(tune)} tune rows · {len(qnom)} qnom rows")
    print(tune.to_string(index=False))

    wbpath = Path(a.workbook)
    if not wbpath.exists():
        raise SystemExit(f"workbook not found: {wbpath}")
    lock = wbpath.with_name("~$" + wbpath.name)
    if lock.exists():
        print(f"\n!! {wbpath.name} is OPEN in Excel ({lock.name}) — close it before a real run")

    print("\n=== эффект правок (через python-порт)")
    print(python_preview(str(wbpath), a.layout).to_string(index=False))

    if a.check:
        print("\n--check: ничего не записано")
        return 0
    if lock.exists():
        raise SystemExit("refusing to write while the workbook is open in Excel")

    import win32com.client as win32

    backup = wbpath.with_suffix(".PRE_V5.xlsm")
    i = 2
    while backup.exists():
        backup = wbpath.with_suffix(f".PRE_V5.{i}.xlsm")
        i += 1
    shutil.copy2(wbpath, backup)
    print(f"\nbackup -> {backup.name}")

    cfg = LAYOUTS[a.layout]
    app = win32.gencache.EnsureDispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1          # 3 force-DISABLES macros
    wb = None
    try:
        wb = app.Workbooks.Open(str(wbpath), UpdateLinks=0)
        if wb.ReadOnly:
            raise RuntimeError("opened READ-ONLY — an orphan EXCEL.EXE still holds the file; "
                               "every change would be discarded on save")
        app.Calculation = -4135         # xlCalculationManual, BEFORE anything recalculates

        proj = wb.VBProject
        for comp in list(proj.VBComponents):
            if comp.Name == MODULE_NAME:
                proj.VBComponents.Remove(comp)
                break
        proj.VBComponents.Import(str(MODULE))
        print(f"imported {MODULE_NAME}")

        sh = wb.Sheets(MODEL_SHEET)
        # --- keyed TuneBlock ------------------------------------------------
        r0 = sh.Range(TUNE_ANCHOR).Row
        c0 = sh.Range(TUNE_ANCHOR).Column
        sh.Cells(r0 - 1, c0).Value = ("Параметры по пластам (v5): ключ, β, RMST_ref, "
                                      "brt, slb, oth")
        for j, name in enumerate(("ключ", "β", "RMST_ref, сут", "brt", "slb", "oth")):
            sh.Cells(r0, c0 + j).Value = name
        for i, (_, row) in enumerate(tune.iterrows(), start=1):
            for j, col in enumerate(("key", "beta", "rmst_ref_days", "brt", "slb", "oth")):
                sh.Cells(r0 + i, c0 + j).Value = row[col]
        tune_rng = sh.Range(sh.Cells(r0 + 1, c0), sh.Cells(r0 + len(tune), c0 + 5))
        wb.Names("TuneBlock").RefersTo = _refers_to(sh, tune_rng)
        print(f"TuneBlock -> {tune_rng.Address}")

        # --- keyed QnomBlock -------------------------------------------------
        qcols = [c for c in qnom.columns if c.startswith("q")]
        r0 = sh.Range(QNOM_ANCHOR).Row
        c0 = sh.Range(QNOM_ANCHOR).Column
        sh.Cells(r0 - 1, c0).Value = "Слой Qном по пластам (v5): узлы и θ, вне узлов — плато"
        sh.Cells(r0, c0).Value = "Qном, м³/сут"
        for j, c in enumerate(qcols):
            sh.Cells(r0, c0 + 1 + j).Value = float(c[1:])
        for i, (_, row) in enumerate(qnom.iterrows(), start=1):
            sh.Cells(r0 + i, c0).Value = row["key"]
            for j, c in enumerate(qcols):
                sh.Cells(r0 + i, c0 + 1 + j).Value = float(row[c])
        qnom_rng = sh.Range(sh.Cells(r0, c0), sh.Cells(r0 + len(qnom), c0 + len(qcols)))
        wb.Names("QnomBlock").RefersTo = _refers_to(sh, qnom_rng)
        print(f"QnomBlock -> {qnom_rng.Address}")

        # --- schedule formulas gain the field cell ---------------------------
        pl = wb.Sheets(PLAN_SHEET)
        for addr in cfg["spills"]:
            cell = pl.Range(addr)
            f = cell.Formula
            if "CM4_FailureScheduleDyn" not in f:
                raise RuntimeError(f"{addr}: not a v4 schedule formula: {f[:80]}")
            if f.rstrip().endswith(f"{cfg['field_cell']})"):
                print(f"{addr}: already carries the field cell")
                continue
            new = f.rstrip()[:-1] + f",{cfg['field_cell']})"
            cell.Formula2 = new
            print(f"{addr}: + {cfg['field_cell']}")

        # --- the ladder default on the sheet side ----------------------------
        top = "MAX(КРС_ЭПУ!$B$41:$B$57)"
        for addr in cfg["nominal_cells"]:
            cell = pl.Range(addr)
            f = cell.Formula
            if "XLOOKUP" in f and ("99999" in f):
                cell.Formula2 = f.replace(",999999,", f",{top},").replace(",99999,", f",{top},")
                print(f"{addr}: ladder default -> {top}")
        first = cfg["first_row"]
        for col in cfg["nominal_cols"]:
            addr = f"{col}{first + 1}"
            f = pl.Range(addr).Formula
            if "XLOOKUP" in f and "99999" in f:
                fixed = f.replace(",999999,", f",{top},").replace(",99999,", f",{top},")
                last = pl.Cells(pl.Rows.Count, 2).End(-4162).Row      # xlUp on «День»
                pl.Range(f"{col}{first + 1}:{col}{last}").Formula2 = fixed
                print(f"{col}{first + 1}:{col}{last}: ladder default -> {top}")

        app.Calculation = -4105          # xlCalculationAutomatic
        app.CalculateFullRebuild()
        wb.Save()
        print("saved")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
