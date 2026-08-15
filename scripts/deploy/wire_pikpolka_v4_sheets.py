"""Convert the ПикПолка workbook fully onto unified v4 — data, charts and the schedule.

    python scripts/deploy/wire_pikpolka_v4_sheets.py

Run AFTER ``wire_pikpolka_v4.py`` (module) and ``wire_pikpolka_v4_blocks.py`` (parameter
blocks).  This is the third and last step: it repoints everything the sheets actually compute.

What it changes
---------------
``МодельОтказов``
  * the Kpod / frequency / rate display blocks move from ``CM_LifeDays`` (v1) to
    ``CM4_LifeDays``, and their grids are widened to the ranges the new forms are defined on
    (Kpod 0.15–1.8, frequency 35–70);
  * **the rate block changes variable**: v1 charted Ql, v4 is a nameplate model, so the axis
    becomes Qном on the fitted knots.  Leaving a Ql axis fed by a Qном model would be a
    mislabelled chart, not a cosmetic difference;
  * the multiplier divisor becomes ``CM4_LifeAtRef`` rather than ``CM4_RmstRef``.  With
    θ_freq(50) = 0.9 the bare baseline is no longer the life at the reference operating point,
    so dividing by it would put every curve ~5 % off and hide the shift in plain sight.

``ПрогнозРемонтов``
  * ``FailureScheduleDyn`` → ``CM4_FailureScheduleDyn`` for both scenario columns, passing the
    control and nameplate blocks.  Arguments are otherwise unchanged: the v1 function already
    derived the standard nominal internally, which is exactly what θ_Qном needs.

The v1 module is left in place, untouched, so the previous behaviour stays recoverable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from wire_pikpolka_v4 import (  # noqa: E402
    FREQ, KPOD, QNOM_KNOTS, WORKBOOK, life_days, theta_freq, theta_kpod, theta_qnom)

MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"

#: Grids the new forms are actually defined on (the old ones stopped short at both ends),
#: sized to the free rows each block has.  The sheet is NOT a blank canvas: the Kpod block is
#: boxed in by "Проверка применимости" at row 22 and the rate block by the Qном knot table at
#: row 52, so the grids are chosen to fit rather than the neighbours pushed out of the way.
KPOD_GRID = [0.15, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95, 1.0, 1.2, 1.5, 1.8]          # rows 9–19
FREQ_GRID = [35.0, 38.0, 40.0, 43.0, 46.0, 50.0, 53.0, 56.0, 60.0, 63.0, 66.0, 70.0]  # D9–D20
QNOM_GRID = list(QNOM_KNOTS)                                                    # rows 38–45

KPOD_FIRST, FREQ_FIRST, QNOM_FIRST = 9, 9, 38
#: Cleared before writing.  Exact, not generous — a 40-row sweep here silently ate the
#: applicability check and the block headers on the first attempt.
CLEAR = ("A9:B19", "D9:E20", "A38:B49")
REF = "CM4_LifeAtRef($B$2,TuneBlock,CtrlBlock,QnomBlock)"


def _life(cells: str) -> str:
    return (f"=CM4_LifeDays($B$2,\"Борец\",{cells},TuneBlock,CtrlBlock,QnomBlock)/{REF}")


def main() -> int:
    import win32com.client as win32

    app = win32.gencache.EnsureDispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    wb = None
    try:
        wb = app.Workbooks.Open(str(WORKBOOK), UpdateLinks=0)
        if wb.ReadOnly:
            raise RuntimeError("workbook opened READ-ONLY — close every Excel instance first")
        sh = wb.Sheets(MODEL_SHEET)

        # ---- clear the old blocks so a shorter new grid cannot leave orphans ----
        for addr in CLEAR:
            rng = sh.Range(addr)
            if rng.MergeCells:            # ClearContents raises on a partial merge anyway
                raise RuntimeError(f"{addr} overlaps a merged cell — refusing to clear blindly")
            rng.ClearContents()

        # ---- Kpod block ------------------------------------------------------
        sh.Range("A8").Value = "Кпод"
        sh.Range("B8").Value = "Множитель ресурса RMST (Кпод), отн. опорной точки"
        for i, k in enumerate(KPOD_GRID):
            r = KPOD_FIRST + i
            sh.Cells(r, 1).Value = k
            sh.Cells(r, 2).Formula = _life(f"250,50,A{r}")
        kpod_last = KPOD_FIRST + len(KPOD_GRID) - 1

        # ---- frequency block -------------------------------------------------
        sh.Range("D8").Value = "Частота, Гц"
        sh.Range("E8").Value = "Множитель ресурса RMST (частота), отн. опорной точки"
        for i, f in enumerate(FREQ_GRID):
            r = FREQ_FIRST + i
            sh.Cells(r, 4).Value = f
            sh.Cells(r, 5).Formula = _life(f"250,D{r},0.825")
        freq_last = FREQ_FIRST + len(FREQ_GRID) - 1

        # ---- rate block: Ql -> Qnom -----------------------------------------
        sh.Range("A36").Value = "Множитель ресурса по НОМИНАЛЬНОЙ подаче (unified v4)"
        sh.Range("A37").Value = "Qном, м³/сут"
        sh.Range("B37").Value = "Множитель ресурса RMST (Qном), отн. опорной точки"
        for i, q in enumerate(QNOM_GRID):
            r = QNOM_FIRST + i
            sh.Cells(r, 1).Value = q
            sh.Cells(r, 2).Formula = _life(f"A{r},50,0.825")
        qnom_last = QNOM_FIRST + len(QNOM_GRID) - 1

        sh.Range("B6").Formula = f"={REF}"
        print(f"{MODEL_SHEET}: Kpod A{KPOD_FIRST}:B{kpod_last}, "
              f"freq D{FREQ_FIRST}:E{freq_last}, Qnom A{QNOM_FIRST}:B{qnom_last}")

        # ---- charts ----------------------------------------------------------
        specs = [
            ("Множитель ресурса RMST vs Кпод",
             f"Кпод (Qном 250, 50 Гц)", KPOD_FIRST, kpod_last, "A", "B"),
            ("Множитель ресурса RMST vs частота",
             f"Частота (Qном 250, Кпод 0.825)", FREQ_FIRST, freq_last, "D", "E"),
            ("Множитель ресурса RMST vs Qном",
             f"Qном (50 Гц, Кпод 0.825)", QNOM_FIRST, qnom_last, "A", "B"),
        ]
        for co, (title, sname, r0, r1, xc, yc) in zip(sh.ChartObjects(), specs):
            ch = co.Chart
            while ch.SeriesCollection().Count > 1:
                ch.SeriesCollection(ch.SeriesCollection().Count).Delete()
            se = ch.SeriesCollection(1)
            se.Formula = (f'=SERIES("{sname}",'
                          f"'{MODEL_SHEET}'!${xc}${r0}:${xc}${r1},"
                          f"'{MODEL_SHEET}'!${yc}${r0}:${yc}${r1},1)")
            ch.HasTitle = True
            ch.ChartTitle.Text = title
            print(f"  chart '{co.Name}' -> {title}  [{xc}{r0}:{yc}{r1}]")

        # ---- ПрогнозРемонтов: schedule on the v4 layers ----------------------
        pl = wb.Sheets(PLAN_SHEET)
        swapped = []
        for addr in ("I15", "T15"):
            old = pl.Range(addr).Formula2
            if "FailureScheduleDyn" not in old:
                print(f"  {PLAN_SHEET}!{addr}: no schedule formula, skipped")
                continue
            new = old.replace("FailureScheduleDyn", "CM4_FailureScheduleDyn")
            if new.rstrip().endswith("TuneBlock)"):
                new = new.rstrip()[:-len("TuneBlock)")] + "TuneBlock,CtrlBlock,QnomBlock)"
            # .Formula2, NOT .Formula.  The schedule returns an array that must SPILL down the
            # column; assigning through .Formula applies legacy implicit-intersection
            # semantics, which silently prefixes "@" and collapses 1094 days to one cell.
            pl.Range(addr).Formula2 = new
            swapped.append(addr)
            back = pl.Range(addr).Formula2
            if "@" in back:
                raise RuntimeError(f"{addr} came back with implicit intersection: {back[:80]}")
            print(f"  {PLAN_SHEET}!{addr} -> {new[:120]}")
        app.CalculateFullRebuild()

        # ---- verify the sheet against Python ---------------------------------
        bad = 0
        for i, k in enumerate(KPOD_GRID):
            got = float(sh.Cells(KPOD_FIRST + i, 2).Value)
            want = life_days(False, "brt", 250.0, 50.0, k) / life_days(False, "brt", 250.0, 50.0, 0.825)
            bad += abs(got - want) > 1e-6
        for i, f in enumerate(FREQ_GRID):
            got = float(sh.Cells(FREQ_FIRST + i, 5).Value)
            want = life_days(False, "brt", 250.0, f, 0.825) / life_days(False, "brt", 250.0, 50.0, 0.825)
            bad += abs(got - want) > 1e-6
        for i, q in enumerate(QNOM_GRID):
            got = float(sh.Cells(QNOM_FIRST + i, 2).Value)
            want = life_days(False, "brt", q, 50.0, 0.825) / life_days(False, "brt", 250.0, 50.0, 0.825)
            bad += abs(got - want) > 1e-6
        n = len(KPOD_GRID) + len(FREQ_GRID) + len(QNOM_GRID)
        print(f"\nМодельОтказов display blocks: {n - bad}/{n} agree with Python")

        # The schedule is a spilling dynamic array, not a legacy CSE array — check that it
        # actually spilled down the column rather than collapsing to one cell (which is what
        # a #VALUE! from the UDF looks like from the outside).
        for addr in swapped:
            cell = pl.Range(addr)
            try:
                spill = cell.SpillingToRange
                n_spill = spill.Cells.Count
                spill_addr = spill.Address
            except Exception:
                n_spill, spill_addr = 1, addr
            vals = [r[0] for r in pl.Range(f"{addr[0]}15:{addr[0]}1108").Value
                    if r[0] is not None]
            fails = sum(1 for v in vals if v == 0)
            ok = n_spill > 1000 and vals and set(vals) <= {0, 1, 0.0, 1.0}
            print(f"{PLAN_SHEET}!{addr}: spill {spill_addr} ({n_spill} cells), "
                  f"{len(vals)} values, {fails} failures  {'OK' if ok else '<< PROBLEM'}")
            if not ok:
                bad += 1
                print(f"    first values: {vals[:6]}")

        if bad:
            print("\n!! NOT SAVED — the sheet disagrees with Python")
            return 1
        wb.Save()
        print(f"\nsaved {WORKBOOK.name}")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
