"""Turn «3) Частота и Кпод» into a three-scenario block with a selector.

    python scripts/deploy/wire_freq_kpod_scenarios.py --workbook <path> [--check]

Frequency and Kpod are the two layers the model does **not** measure — fitted free, frequency
is a null in every field and Kpod's deployed bathtub disagrees in sign with its own fit at low
loading.  A single column of numbers hides that; three columns and a selector make the
assumption an input.

    B  Базовый          censoring-aware reading (geomean of the Ya tent and reconciled curves;
                        Kpod = the fitted monotone layer, where low Кпод PROTECTS)
    C  Стресс           the operator prior kept as the pessimistic arm (Kpod = the bathtub)
    D  Пользовательский  = B^(1−w)·C^w for the six multipliers, B+w(C−B) for the four shape
                        parameters, w in one cell (0 → Базовый, 1 → Стресс)
    E  В расчёте        =CHOOSE(ПрогнозРемонтов!$A$4, B, C, D)  ← what CtrlBlock points at

Why the blend is geometric on the anchors: θ's compose multiplicatively and the layer is
piecewise-linear in **log** θ, so averaging θ linearly bends the curve into a shape neither
scenario supports.  Plateau edges and shape exponents are not multipliers and average linearly.

θ(50 Гц) is **1.00 in both scenarios by design**.  It is a level, not a severity: the shipped
0.9 re-levels the whole layer, which is why `ННО_ref` read 198.5 д while `RMST_ref` said 183.8.
Stress belongs in the arms.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"
SELECTOR = "A4"          # on ПрогнозРемонтов, at the head of the input row
SELECTOR_LABEL = "A3"

#: Labels for the ten CtrlBlock rows; the **numbers** come from the backend so the sheet, the
#: Python twin and the TTF viewer cannot drift apart.
LABELS = (
    "θ частоты при 40 Гц",
    "θ частоты при 50 Гц  (уровень слоя — в обоих сценариях 1)",
    "θ частоты при 60 Гц",
    "θ частоты при 70 Гц  (4-я опора; данных выше ~62 Гц нет)",
    "θ_Кпод при 0.2",
    "θ_Кпод при 1.2",
    "Кпод — начало плато (θ = 1)",
    "Кпод — конец плато",
    "Кпод — показатель левого плеча (1 линейно, 2 парабола)",
    "Кпод — показатель правого плеча",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from analysis.workflows.production_risk import pikpolka_sim as S  # noqa: E402

#: (label, best guess, stress) in sheet order.
SCENARIOS = [(lab, S.CTRL_BASE[k], S.CTRL_STRESS[k])
             for lab, k in zip(LABELS, S.CTRL_ORDER)]
#: rows 1–6 are multipliers (geometric blend); 7–10 are shape parameters (linear blend).
N_MULT = len(S.CTRL_MULTIPLIERS)

TITLE = ("3) Частота и Кпод — ЗАДАЮТСЯ оператором, не подбирались  (CtrlBlock). "
         "Сценарий выбирается в ПрогнозРемонтов!A4")
CAPTIONS = ("B — Базовый", "C — Стресс", "D — Пользовательский", "E — В РАСЧЁТЕ")
W_LABEL = "вес стресса w в пользовательском сценарии (0 = базовый, 1 = стресс)"
W_DEFAULT = 0.5
SEL_LABEL = "Сценарий θ частоты/Кпод: 1 = Базовый, 2 = Стресс, 3 = Пользовательский"
SEL_DEFAULT = 1

NOTE_SCEN = ("Базовый = censoring-aware чтение Ya (тент × reconciled) и ПОДОГНАННЫЙ слой Кпод "
             "(низкий Кпод защищает). Стресс = операторская опора и «ванна». "
             "Полоса — это расхождение двух гипотез, а не доверительный интервал: "
             "при свободной подгонке частота — НОЛЬ во всех фондах (CV: Vt −0.6, Az −0.3, "
             "Ic −0.8, Ya +1.1, Za +1.9), т.е. статистически нижняя граница — θ ≡ 1.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--selector", type=int, default=SEL_DEFAULT, choices=(1, 2, 3))
    p.add_argument("--weight", type=float, default=W_DEFAULT)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()

    wbpath = Path(a.workbook)
    if not wbpath.exists():
        raise SystemExit(f"workbook not found: {wbpath}")
    lock = wbpath.with_name("~$" + wbpath.name)
    if lock.exists():
        raise SystemExit(f"{wbpath.name} is open in Excel ({lock.name}) — close it first")
    if a.check:
        for label, b, c in SCENARIOS:
            print(f"  {label[:52]:54s} B={b:<6} C={c}")
        return 0

    import win32com.client as win32

    backup = wbpath.with_suffix(".PRE_SCENARIOS.xlsm")
    i = 2
    while backup.exists():
        backup = wbpath.with_suffix(f".PRE_SCENARIOS.{i}.xlsm")
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
        sh, pl = wb.Sheets(MODEL_SHEET), wb.Sheets(PLAN_SHEET)

        ctrl = wb.Names("CtrlBlock").RefersToRange
        if ctrl.Rows.Count != len(SCENARIOS):
            raise RuntimeError(f"CtrlBlock is {ctrl.Rows.Count} rows, expected {len(SCENARIOS)}")
        # ⚠ capture the row NUMBER before inserting: a live COM Range shifts with the sheet,
        # so reading ctrl.Row afterwards returns the moved position and the block lands two
        # rows too low, on top of the old values.
        ctrl_row = int(ctrl.Row)
        title_row = ctrl_row - 1                      # «3) …» sits immediately above
        # two spare rows: column captions live on the title row, w gets its own
        sh.Rows(f"{ctrl_row}:{ctrl_row + 1}").Insert()
        first = ctrl_row + 2
        w_row = title_row + 1
        print(f"title {title_row}, weight {w_row}, block {first}:{first + 9}")

        sh.Cells(title_row, 1).Value = TITLE
        for j, cap in enumerate(CAPTIONS):
            sh.Cells(title_row, 2 + j).Value = cap
        sh.Cells(w_row, 1).Value = W_LABEL
        sh.Cells(w_row, 4).Value = float(a.weight)

        for r, (label, b, c) in enumerate(SCENARIOS):
            row = first + r
            sh.Cells(row, 1).Value = label
            sh.Cells(row, 2).Value = b
            sh.Cells(row, 3).Value = c
            if r < N_MULT:                            # θ multipliers → geometric blend
                sh.Cells(row, 4).Formula = f"=B{row}^(1-$D${w_row})*C{row}^$D${w_row}"
            else:                                     # shape parameters → linear blend
                sh.Cells(row, 4).Formula = f"=B{row}+$D${w_row}*(C{row}-B{row})"
            sh.Cells(row, 5).Formula = (
                f"=CHOOSE({PLAN_SHEET}!${SELECTOR[0]}${SELECTOR[1:]},B{row},C{row},D{row})")
        last = first + len(SCENARIOS) - 1

        wb.Names("CtrlBlock").RefersTo = (
            f"='{sh.Name}'!{sh.Range(sh.Cells(first, 5), sh.Cells(last, 5)).Address}")
        print(f"CtrlBlock -> E{first}:E{last}")

        note_row = last + 2
        while sh.Cells(note_row, 1).Value:            # keep the existing clamp/assumption notes
            note_row += 1
        sh.Cells(note_row, 1).Value = NOTE_SCEN

        # ---- selector on the input row --------------------------------------
        pl.Range(SELECTOR_LABEL).Value = SEL_LABEL
        cell = pl.Range(SELECTOR)
        cell.Value = int(a.selector)
        try:
            cell.Validation.Delete()
            cell.Validation.Add(Type=3, AlertStyle=1, Operator=1, Formula1="1,2,3")
            cell.Validation.InputTitle = "Сценарий"
            cell.Validation.InputMessage = SEL_LABEL
        except Exception as exc:                      # validation is a nicety, not the wiring
            print(f"  (validation skipped: {exc})")

        app.Calculation = -4105
        app.CalculateFullRebuild()
        nno = ("R4", "Z4") if pl.Range("R3").Value == "ННО 6 мес Пик" else ("N4", "O4")
        for s in (1, 2, 3):
            pl.Range(SELECTOR).Value = s
            app.CalculateFullRebuild()
            print(f"  сценарий {s}: ННО {pl.Range(nno[0]).Value:.0f} / "
                  f"{pl.Range(nno[1]).Value:.0f}")
        pl.Range(SELECTOR).Value = int(a.selector)
        app.CalculateFullRebuild()
        wb.Save()
        print(f"saved (селектор = {a.selector}, w = {a.weight})")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


if __name__ == "__main__":
    raise SystemExit(main())
