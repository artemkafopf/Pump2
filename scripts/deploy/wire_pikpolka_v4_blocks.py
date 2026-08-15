"""Rewrite the ``МодельОтказов`` parameter blocks to the unified v4 / operator-set values.

    python scripts/deploy/wire_pikpolka_v4_blocks.py

``ModuleFailureV4`` carries all of these as compiled defaults, so the workbook computes
correctly without the sheet.  The blocks exist so the numbers are *visible and editable* — and
so the sheet stops displaying the superseded ones, which is worse than displaying nothing.

The control block grows from 6 cells to 10 (the frequency form changed from m60/s/shift to
three anchors plus an optional fourth).  ``ReadCtrl4`` ignores any block shorter than 10 cells
outright rather than half-applying it, so until the named range is widened here the sheet's
controls are inert — safe, but dead.  Widening it is the point of this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))

from wire_pikpolka_v4 import (  # noqa: E402
    BASE, CONTRACTOR, FREQ, KPOD, QNOM_KNOTS, QNOM_NS, QNOM_SR, WORKBOOK,
    life_days, rmst730, theta_freq, theta_kpod)

SHEET = "МодельОтказов"
TUNE = "B32:F33"
QNOM = "B53:I55"
CTRL_FIRST_ROW, CTRL_ROWS = 58, 10
CTRL_ADDR = f"B{CTRL_FIRST_ROW}:B{CTRL_FIRST_ROW + CTRL_ROWS - 1}"

CTRL = [
    ("θ частоты при 40 Гц", FREQ[0]),
    ("θ частоты при 50 Гц  (НЕ 1 — сдвигает уровень слоя)", FREQ[1]),
    ("θ частоты при 60 Гц", FREQ[2]),
    ("θ частоты при 70 Гц  (4-я опора; 0 = чистая парабола)", FREQ[3]),
    ("θ_Кпод при 0.2", KPOD[0]),
    ("θ_Кпод при 1.2", KPOD[1]),
    ("Кпод — начало плато (θ = 1)", KPOD[2]),
    ("Кпод — конец плато", KPOD[3]),
    ("Кпод — показатель левого плеча (1 линейно, 2 парабола)", KPOD[4]),
    ("Кпод — показатель правого плеча", KPOD[5]),
]


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
        sh = wb.Sheets(SHEET)

        # --- TuneBlock: beta, RMST_ref(days), brt, slb, oth -------------------
        sh.Range("A30").Value = ("Настраиваемые параметры модели — unified v4 "
                                 "(Qном + подрядчик подобраны, частота и Кпод заданы)")
        for i, sour in enumerate((False, True)):
            b0, e0 = BASE[sour]
            row = 32 + i
            sh.Cells(row, 2).Value = round(b0, 7)
            sh.Cells(row, 3).Value = round(rmst730(e0, b0), 4)
            sh.Cells(row, 4).Value = CONTRACTOR["brt"]
            sh.Cells(row, 5).Value = CONTRACTOR["slb"]
            sh.Cells(row, 6).Value = CONTRACTOR["oth"]
        print(f"TuneBlock {TUNE} <- unified v4 baselines + overlap-window contractor levels")

        # --- QnomBlock: knots / nonsour / sour --------------------------------
        sh.Range("A52").Value = "Слой Qном (unified v4): узлы и θ, вне узлов — плато"
        sh.Range("A53").Value = "Qном, м³/сут"
        sh.Range("A54").Value = "θ некислый"
        sh.Range("A55").Value = "θ кислый"
        for j, (k, ns, sr) in enumerate(zip(QNOM_KNOTS, QNOM_NS, QNOM_SR)):
            sh.Cells(53, 2 + j).Value = k
            sh.Cells(54, 2 + j).Value = ns
            sh.Cells(55, 2 + j).Value = sr
        print(f"QnomBlock {QNOM} <- knots {QNOM_KNOTS[0]:g}..{QNOM_KNOTS[-1]:g}")

        # --- CtrlBlock: 6 cells -> 10 ----------------------------------------
        sh.Range("A57").Value = "Управляемые параметры слоёв частоты и Кпод (задаются оператором)"
        for i, (label, value) in enumerate(CTRL):
            sh.Cells(CTRL_FIRST_ROW + i, 1).Value = label
            sh.Cells(CTRL_FIRST_ROW + i, 2).Value = value
        # the old layout had a clamp-bounds label just past the block; the clamp is fixed
        # at 35..70 in code now, so leaving the label would advertise a control that is gone
        for r in range(CTRL_FIRST_ROW + CTRL_ROWS, CTRL_FIRST_ROW + CTRL_ROWS + 3):
            if sh.Cells(r, 1).Value:
                sh.Cells(r, 1).ClearContents()
                sh.Cells(r, 2).ClearContents()
        sh.Cells(CTRL_FIRST_ROW + CTRL_ROWS + 1, 1).Value = (
            "Обрезка частоты жёстко 35–70 Гц, Кпод 0.15–1.8 — заданы в коде "
            "(полином неограничен, обрезка несущая)")
        # Both row and column must be absolute.  A half-anchored string like "$B58:$B67" is a
        # RELATIVE row reference, which Excel re-anchors against the active cell — the name
        # silently landed on B75:B84 and the module then read blank cells.
        last = CTRL_FIRST_ROW + CTRL_ROWS - 1
        wb.Names("CtrlBlock").RefersTo = f"='{SHEET}'!$B${CTRL_FIRST_ROW}:$B${last}"
        print(f"CtrlBlock  {CTRL_ADDR} <- 10 cells (was 6)")

        # --- verify the module reads the sheet and agrees with Python ---------
        tb = wb.Names("TuneBlock").RefersToRange
        cb = wb.Names("CtrlBlock").RefersToRange
        qb = wb.Names("QnomBlock").RefersToRange
        # .Address is a PROPERTY in this gencache binding, not a callable method
        print(f"\nCtrlBlock now refers to {cb.Address} ({cb.Cells.Count} cells)")

        bad, worst = 0, 0.0
        for sour in (0, 1):
            for c in ("brt", "slb", "oth"):
                for q in (60.0, 250.0, 640.0, 1600.0):
                    for f in (35.0, 40.0, 50.0, 60.0, 70.0):
                        for k in (0.2, 0.8, 1.2, 1.8):
                            got = float(app.Run("ModuleFailureV4.CM4_LifeDays",
                                                float(sour), c, q, f, k, tb, cb, qb))
                            want = life_days(bool(sour), c, q, f, k)
                            worst = max(worst, abs(got - want))
                            bad += abs(got - want) > 1e-4
        n = 2 * 3 * 4 * 5 * 4
        print(f"CM4_LifeDays THROUGH THE SHEET BLOCKS: {n - bad}/{n} agree, "
              f"worst |Δ| = {worst:.2e} d")
        if bad:
            print("!! NOT SAVED — the sheet blocks disagree with Python")
            return 1

        for f_ in (35.0, 40.0, 50.0, 60.0, 70.0):
            g = float(app.Run("ModuleFailureV4.CM4_ThetaFreq", f_, cb))
            assert abs(g - theta_freq(f_, *FREQ)) < 1e-9, f"θ_freq({f_}) {g}"
        for k_ in (0.2, 0.7, 0.95, 1.2, 1.8):
            g = float(app.Run("ModuleFailureV4.CM4_ThetaKpod", k_, cb))
            assert abs(g - theta_kpod(k_, *KPOD)) < 1e-9, f"θ_Kpod({k_}) {g}"
        print("layer UDFs driven from CtrlBlock: exact")

        wb.Save()
        print(f"\nsaved {WORKBOOK.name}")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
