"""Put **every** model parameter under `МодельОтказов!A30`, and delete the stale duplicates.

    python scripts/deploy/consolidate_model_sheet.py --workbook <path> [--check]

After the v5 rewire the sheet had the parameters in two places: the header at A30 announced
«Настраиваемые параметры модели», but the block under it was still the **dead** 2-row Vt-only
TuneBlock, while the live keyed blocks sat 40 rows further down past the charts.  The old
Qnom block was worse than dead — it still carried the pre-clamp sour row (θ(1600) = 1.2034),
so anyone reading the parameter section saw numbers the model had stopped using.

This rewrites the section as three numbered blocks in one contiguous run::

    A30  header
    A32  1) Базовый уровень и подрядчик — по пластам      -> TuneBlock
    A44  2) Слой Qном — по пластам                        -> QnomBlock
    A56  3) Частота и Кпод — заданы оператором            -> CtrlBlock
    A68  границы и допущения

Mechanics: rows are **inserted** rather than the display tables overwritten, so the three
charts keep their series (they follow the shift), and the orphaned blocks below are deleted
only *after* the three names have been repointed.  ННО and NPV are compared before and after —
a pure re-layout must not move a single number.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"

#: Rows inserted ahead of the first display table, sized to the section written below.
INSERT_AT, INSERT_N = 36, 36

#: (row, text) written into column A of the consolidated section.  Data rows come from the
#: existing keyed blocks, so this script never re-types a fitted number.
HEADER = ("Настраиваемые параметры модели — unified v4 / v5: "
          "Qном и подрядчик ПОДОБРАНЫ по каждому пласту, частота и Кпод ЗАДАНЫ оператором")
T_TITLE = "1) Базовый уровень и подрядчик — по пластам  (именованный блок TuneBlock)"
Q_TITLE = ("2) Слой Qном — по пластам  (QnomBlock): узлы и θ, "
           "вне крайних узлов θ держится плато")
C_TITLE = ("3) Частота и Кпод — ЗАДАЮТСЯ оператором, не подбирались  (CtrlBlock)")
NOTE_1 = ("Обрезка: Qном 60–1600, частота 35–70 Гц, Кпод 0.15–1.8 — заданы в коде "
          "(полином частоты неограничен, обрезка несущая)")
NOTE_2 = ("Допущение: слой Кпод — операторская «ванна»; подогнанный слой на Vt монотонно РАСТЁТ, "
          "т.е. недогруженный насос он считает защищённым, а «ванна» — наказывает")

TUNE_HEAD = ("ключ (поле)", "β (форма)", "RMST_ref, сут", "brt", "slb", "oth")
#: column -> reading hint.  B = β, C = RMST_ref (bigger is LONGER life), D:F = contractor
#: multipliers on the hazard (bigger is worse).  Keyed by the actual column number.
HINTS = {2: "Больше — хуже", 3: "Больше — лучше", 4: "Больше — хуже"}

ROWS = {"t_title": 32, "t_hint": 32, "t_head": 33, "t_first": 34,
        "q_title": 44, "q_head": 45, "q_first": 46,
        "c_title": 56, "c_first": 57,
        "note1": 68, "note2": 69}


def _read_existing(sh, first_row: int, n_rows: int, n_cols: int) -> list[list]:
    return [[sh.Cells(first_row + r, 1 + c).Value for c in range(n_cols)] for r in range(n_rows)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()

    wbpath = Path(a.workbook)
    if not wbpath.exists():
        raise SystemExit(f"workbook not found: {wbpath}")
    lock = wbpath.with_name("~$" + wbpath.name)
    if lock.exists():
        raise SystemExit(f"{wbpath.name} is open in Excel ({lock.name}) — close it first")
    if a.check:
        print(f"{wbpath.name}: would insert {INSERT_N} rows at {INSERT_AT}, rewrite A30:I69, "
              f"repoint TuneBlock/QnomBlock/CtrlBlock and delete the orphaned blocks below")
        return 0

    import win32com.client as win32

    backup = wbpath.with_suffix(".PRE_CONSOLIDATE.xlsm")
    i = 2
    while backup.exists():
        backup = wbpath.with_suffix(f".PRE_CONSOLIDATE.{i}.xlsm")
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
        app.Calculation = -4135                      # manual, before anything recalculates
        sh, pl = wb.Sheets(MODEL_SHEET), wb.Sheets(PLAN_SHEET)

        before = [c.Value for c in (pl.Range("R4"), pl.Range("Z4"))] \
            if pl.Range("R3").Value else [pl.Range("N4").Value, pl.Range("O4").Value]

        # ---- read what we are about to move ---------------------------------
        tune_rng = wb.Names("TuneBlock").RefersToRange
        qnom_rng = wb.Names("QnomBlock").RefersToRange
        ctrl_rng = wb.Names("CtrlBlock").RefersToRange
        tune = _read_existing(sh, tune_rng.Row, tune_rng.Rows.Count, 6)
        qnom_first = qnom_rng.Row
        qnom_knots = [sh.Cells(qnom_first, 1 + c).Value for c in range(9)]
        qnom = _read_existing(sh, qnom_first + 1, qnom_rng.Rows.Count - 1, 9)
        ctrl = [(sh.Cells(ctrl_rng.Row + r, 1).Value, sh.Cells(ctrl_rng.Row + r, 2).Value)
                for r in range(ctrl_rng.Rows.Count)]
        if len(tune) < 2 or len(qnom) < 2 or len(ctrl) != 10:
            raise RuntimeError(f"unexpected block sizes: tune {len(tune)}, qnom {len(qnom)}, "
                               f"ctrl {len(ctrl)} — refusing to rewrite blindly")
        print(f"read: {len(tune)} tune rows, {len(qnom)} qnom rows, {len(ctrl)} ctrl cells")

        # ---- make room; the charts follow their series through an insert ----
        sh.Rows(f"{INSERT_AT}:{INSERT_AT + INSERT_N - 1}").Insert()
        print(f"inserted {INSERT_N} rows at {INSERT_AT}")

        # ---- clear the dead legacy block that sat directly under the header --
        sh.Range("A31:I35").ClearContents()

        # ---- 1) TuneBlock ---------------------------------------------------
        sh.Range("A30").Value = HEADER
        for c in (2, 3, 4):
            sh.Cells(30, c).Value = None                 # hints move onto the block title row
        sh.Cells(ROWS["t_title"], 1).Value = T_TITLE
        for c, txt in HINTS.items():
            sh.Cells(ROWS["t_hint"], c).Value = txt
        for c, txt in enumerate(TUNE_HEAD):
            sh.Cells(ROWS["t_head"], 1 + c).Value = txt
        for r, row in enumerate(tune):
            for c, v in enumerate(row):
                sh.Cells(ROWS["t_first"] + r, 1 + c).Value = v
        t_last = ROWS["t_first"] + len(tune) - 1

        # ---- 2) QnomBlock ---------------------------------------------------
        sh.Cells(ROWS["q_title"], 1).Value = Q_TITLE
        for c, v in enumerate(qnom_knots):
            sh.Cells(ROWS["q_head"], 1 + c).Value = v
        for r, row in enumerate(qnom):
            for c, v in enumerate(row):
                sh.Cells(ROWS["q_first"] + r, 1 + c).Value = v
        q_last = ROWS["q_first"] + len(qnom) - 1

        # ---- 3) CtrlBlock ---------------------------------------------------
        sh.Cells(ROWS["c_title"], 1).Value = C_TITLE
        for r, (label, value) in enumerate(ctrl):
            sh.Cells(ROWS["c_first"] + r, 1).Value = label
            sh.Cells(ROWS["c_first"] + r, 2).Value = value
        c_last = ROWS["c_first"] + len(ctrl) - 1

        sh.Cells(ROWS["note1"], 1).Value = NOTE_1
        sh.Cells(ROWS["note2"], 1).Value = NOTE_2

        # ---- repoint the names BEFORE deleting anything ---------------------
        def refers(r1, c1, r2, c2):
            return f"='{sh.Name}'!{sh.Range(sh.Cells(r1, c1), sh.Cells(r2, c2)).Address}"

        wb.Names("TuneBlock").RefersTo = refers(ROWS["t_first"], 1, t_last, 6)
        wb.Names("QnomBlock").RefersTo = refers(ROWS["q_head"], 1, q_last, 9)
        wb.Names("CtrlBlock").RefersTo = refers(ROWS["c_first"], 2, c_last, 2)
        print(f"TuneBlock A{ROWS['t_first']}:F{t_last} · "
              f"QnomBlock A{ROWS['q_head']}:I{q_last} · B{ROWS['c_first']}:B{c_last}")

        # ---- drop the orphans (everything below the surviving chart table) --
        first_orphan = INSERT_AT + INSERT_N + 16          # = old row 52, the legacy Qnom title
        last_used = sh.UsedRange.Row + sh.UsedRange.Rows.Count - 1
        if last_used >= first_orphan:
            sh.Rows(f"{first_orphan}:{last_used}").Delete()
            print(f"deleted orphaned rows {first_orphan}:{last_used}")

        app.Calculation = -4105
        app.CalculateFullRebuild()
        after = [c.Value for c in (pl.Range("R4"), pl.Range("Z4"))] \
            if pl.Range("R3").Value else [pl.Range("N4").Value, pl.Range("O4").Value]
        print(f"ННО до {before} -> после {after}")
        if [round(float(x or 0)) for x in before] != [round(float(x or 0)) for x in after]:
            raise RuntimeError("a re-layout changed the answer — NOT saving")
        wb.Save()
        print("saved")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
