"""Make the netback lookup treat `Mc` and `Mr` as one field (Мирнинский).

    python scripts/deploy/alias_mc_to_mr_netback.py --workbook <path> --layout npv|pikpolka

`Экономика` carries the field only as **Mr** («Мирнинский участок», 19 661 ₽/т); the survival
model's stratum is **Mc**, which already covers both pads (the population build folds the `MR`
prefix into `Mc`, 131 + 63 runs).  So selecting `Mc` sent an approximate-match ``VLOOKUP`` into
an unsorted list with no matching key and it returned a *plausible* wrong netback — Δ NPV came
out at −2 147 млн with nothing on the sheet saying anything was wrong.

The fix normalises the code inside the lookup rather than duplicating the operator's data::

    VLOOKUP(<field>, …)  ->  VLOOKUP(IF(<field>="Mc","Mr",<field>), …)

One value, one row, no second copy of 19 661 to drift out of step.

⚠ Not touched here: `КРС_ЭПУ` carries **both** `Mc` (12 сут) and `Mr` (11 сут) for «смена УЭЦН».
If the two codes are one field those two numbers disagree with each other, and picking one is
the operator's call, not this script's.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PLAN_SHEET = "ПрогнозРемонтов"
#: layout -> (field cell, first grid row, the two revenue columns)
LAYOUTS = {"npv": ("H$4", 11, ("L", "W")),
           "pikpolka": ("N$4", 15, ("L", "W"))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--layout", required=True, choices=tuple(LAYOUTS))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()

    wbpath = Path(a.workbook)
    if not wbpath.exists():
        raise SystemExit(f"workbook not found: {wbpath}")
    if wbpath.with_name("~$" + wbpath.name).exists():
        raise SystemExit(f"{wbpath.name} is open in Excel — close it first")

    field, first, cols = LAYOUTS[a.layout]
    alias = f'IF({field}="Mc","Mr",{field})'
    pat = re.compile(r"VLOOKUP\(" + re.escape(field) + r",")
    print(f"{wbpath.name}: VLOOKUP({field}, …)  ->  VLOOKUP({alias}, …)")
    if a.check:
        return 0

    import win32com.client as win32

    backup = wbpath.with_suffix(".PRE_MCALIAS.xlsm")
    i = 2
    while backup.exists():
        backup = wbpath.with_suffix(f".PRE_MCALIAS.{i}.xlsm")
        i += 1
    shutil.copy2(wbpath, backup)
    print(f"backup -> {backup.name}")

    app = win32.gencache.EnsureDispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    wb = None
    try:
        wb = app.Workbooks.Open(str(wbpath), UpdateLinks=0)
        if wb.ReadOnly:
            raise RuntimeError("opened READ-ONLY — an orphan EXCEL.EXE still holds the file")
        app.Calculation = -4135
        pl = wb.Sheets(PLAN_SHEET)
        last = pl.Cells(pl.Rows.Count, 2).End(-4162).Row          # xlUp over «День»
        keep = pl.Range(field.replace("$", "")).Value

        for col in cols:
            src = pl.Range(f"{col}{first}").Formula
            if "VLOOKUP" not in src:
                raise RuntimeError(f"{col}{first}: not a netback formula: {src[:80]}")
            if alias in src:
                print(f"  {col}: already aliased")
                continue
            new = pat.sub(f"VLOOKUP({alias},", src)
            if new == src:
                raise RuntimeError(f"{col}{first}: pattern did not match: {src[:120]}")
            pl.Range(f"{col}{first}:{col}{last}").Formula = new
            print(f"  {col}{first}:{col}{last} rewritten")

        app.Calculation = -4105
        app.CalculateFullRebuild()
        npv = ("X4", "Y4") if a.layout == "pikpolka" else ("P4", "Q4")
        for code in ("Mr", "Mc"):
            pl.Range(field.replace("$", "")).Value = code
            app.CalculateFullRebuild()
            print(f"  {code}: NPV {pl.Range(npv[0]).Value/1e6:>10.2f} / "
                  f"{pl.Range(npv[1]).Value/1e6:<10.2f} млн")
        pl.Range(field.replace("$", "")).Value = keep
        app.CalculateFullRebuild()
        wb.Save()
        print(f"saved (поле возвращено на {keep})")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
