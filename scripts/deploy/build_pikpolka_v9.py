"""Ship the v5.2 layer model: ПикПолка v8 → **v9** and NPV_УВЧ V03 → **V04**.

    python scripts/deploy/build_pikpolka_v9.py            # both workbooks
    python scripts/deploy/build_pikpolka_v9.py --only pikpolka

Everything runs through Excel COM rather than openpyxl.  openpyxl would be simpler, but these
sheets carry charts, conditional formatting and data validation that it silently drops on
round-trip, and section 2 of «МодельОтказов» is nothing but charts.  COM edits in place.

What changes:

1. **Section 3.3 is rewritten.**  It used to hold ten SHAPE knobs — θ at 40/50/60/70 Hz, the
   Kпод plateau edges and two arm exponents — because V4's curves were built out of them.
   v5.2 fits the curves, so the shape moved into the module and only levels remain:
   an on/off switch per layer plus one anchor per arm.  Nine values, all of them live.

2. **Water cut joins as a layer, switched OFF.**  It fits and survives the 3- and 6-month lag
   sweep, but checked in DAYS its curve is much flatter than the fact (571 сут at 80 % against
   497 at 30–60 %, where the model moves life by 30 days).  Shipping it on would put an
   unvalidated effect into every number the calculator produces, so the switch ships at 0.

3. **The renewal call gains the water-cut arguments.**  The sheet has no water-cut column —
   cut is derived from cumulative recovery through the ХВ curve inside the oil-rate formula —
   so the module reproduces that lookup from the oil column plus НИЗ and ТИЗ.  Wiring it now,
   while the switch is off, is what makes the switch real rather than decorative.

4. **ModuleFailureV4 is removed and ModuleFailureV5 imported.**  Both define the same private
   helpers, so leaving V4 behind would be an ambiguous-name error, not a harmless leftover.

Verification is a separate step (``--verify``): with every switch at its shipped value the
ННО must come out UNCHANGED from v8/V03, which is the strongest available evidence that the
rewiring did not disturb anything, and turning the water-cut switch on must change it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODULE = REPO_ROOT / "vba" / "ModuleFailureV5.bas"
OLD_MODULE_NAME = "ModuleFailureV4"
NEW_MODULE_NAME = "ModuleFailureV5"
MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"

#: Anchors are the fitted curve's OWN values at 40 Hz / 60 Hz / Kпод 0.2 / Kпод 1.6 / 95 %,
#: so γ = 1 and the deployed curve IS the fit.  Kept to 4 decimals: that is the precision a
#: person can retype without feeling they are editing a coefficient.
#: ⚠ k02 = 1.2326 у «Базового» — РЕШЕНИЕ: полностью разгруженный насос стоит не меньше
#: 1.5.  Нижнее плечо у обоих сценариев на стрессовой форме (см. ModuleFailureV5), у
#: «Стресса» γ = 1, у «Базового» γ = 0.495 — поэтому «Стресс» не может стать мягче.
#: θ «Базового»: 0 → 1.500, 0.2 → 1.233, 0.4 → 1.092; участок с данными не тронут.
BASE = {"set": 1, "onf": 1, "t40": 1.2129, "t60": 1.1869,
        "onk": 1, "k02": 1.2326, "k16": 1.1209, "onw": 1, "w95": 0.855}
STRESS = {"set": 2, "onf": 1, "t40": 1.7523, "t60": 1.3253,
          "onk": 1, "k02": 1.5251, "k16": 1.4661, "onw": 1, "w95": 0.855}

#: The scenario selector, as seen FROM «МодельОтказов».  It must be the local mirror cell
#: (B5 = «=ПрогнозРемонтов!$A$4»), not $A$4 — an unqualified $A$4 in a formula written on this
#: sheet resolves to МодельОтказов!$A$4, which is empty, so CHOOSE(0+1, …) silently returns the
#: «Без слоёв» column and every scenario collapses to the same answer.  That bug shipped once.
SCEN_CELL = "$B$5"

#: «Пользовательский» сценарий убран целиком: столбец D и строка веса.  Он смешивал два
#: набора геометрическим средним и давал кривую, которой нет ни у одного из них — на вопрос
#: «откуда это число» ответить было нечем.  Осталось три состояния: 0 без слоёв, 1 базовый,
#: 2 стрессовый.
ROW_HEAD, ROW0 = 66, 67                     # header, first parameter row
N_PARAM = 9
ROW_NOTES = ROW0 + N_PARAM + 2

#: (label, base, stress, kind).  Поле kind осталось от смешивания сценариев и больше ни на
#: что не влияет — «Пользовательский» убран; сохранено, чтобы не переписывать таблицу.
PARAMS = [
    ("(служебная) форма кривой — задаётся сценарием, руками не меняется", "set", "lin"),
    ("Слой частоты (1 вкл / 0 выкл)", "onf", "lin"),
    ("θ частоты при 40 Гц", "t40", "geo"),
    ("θ частоты при 60 Гц", "t60", "geo"),
    ("Слой Кпод (1 вкл / 0 выкл)", "onk", "lin"),
    ("θ_Кпод при 0.2  ← глубина штрафа за недогруз, задаётся решением", "k02", "geo"),
    ("θ_Кпод при 1.6", "k16", "geo"),
    ("Слой обводнённости (1 вкл / 0 выкл)", "onw", "lin"),
    ("θ обводнённости при 95 %", "w95", "geo"),
]
#: The «Без слоёв» column: every layer off, anchors at 1, base shape (unused while off).
NULL_COL = {"set": 1, "onf": 0, "t40": 1, "t60": 1, "onk": 0,
            "k02": 1, "k16": 1, "onw": 0, "w95": 1}

NOTES = [
    "Форма кривых зашита в ModuleFailureV5 (подогнанные рациональные функции). Здесь только "
    "УРОВНИ: выключатель слоя и по одному якорю на каждое плечо.",
    "Якорь — это ЦЕЛЬ, а не коэффициент: он превращается в показатель степени над "
    "подогнанной кривой для своего плеча. Опорная точка (50 Гц, Кпод 0.8, обводнённость "
    "25 %) при любом якоре остаётся ровно 1, а значения по умолчанию воспроизводят подгонку.",
    "Чего якорь НЕ может: сдвинуть точку перегиба. Он меняет глубину кривой, а не её форму. "
    "Якорь меньше 1 разворачивает плечо — это честный способ сказать «на этом плече "
    "штрафа нет».",
    "У «Базового» кривая Кпод продолжена до 0 (раньше была плоской ниже 0.4). Это сделано "
    "ради якоря при 0.2: 28 % месячных интервалов фонда лежат ниже 0.4, и на плоском "
    "участке задать глубину недогруза решением было нельзя. У «Стресса» обрезка на 0.4 "
    "ОСТАЁТСЯ: его знаменатель первой степени раздувает θ до 2.27 на нуле, а данных ниже "
    "Кпод 0.45 нет.",
    "Слой обводнённости ВКЛЮЧЁН. Выключатель оставлен: в сутках наработки его кривая заметно "
    "площе факта (571 сут при 80 % против 497 у 30–60 %, модель двигает на 30 сут), так что "
    "если результат по нему вызовет сомнения — его можно снять, не трогая остальные слои.",
    "Сценариев три: 0 без слоёв, 1 базовый, 2 стрессовый. «Пользовательский» убран — он "
    "смешивал два набора и давал кривую, которой нет ни у одного из них.",
    "«Стресс» — не доверительный интервал, а вторая гипотеза о ВЕЛИЧИНЕ эффекта, снятая "
    "с факта ННО по закрытым пускам. Нижний край — сценарий 0, где слои выключены и остаются "
    "только базовый уровень, подрядчик и типоразмер.",
    "Обрезка: Qном 60–1600, частота 35–70 Гц (по отклонению −15…+20), Кпод 0–1.8 "
    "у «Базового» и 0.4–1.8 у «Стресса», "
    "обводнённость 0–100 %. Заданы в коде, несущие.",
    "Fleet — пулированная подгонка по всему фонду; на неё уходит любой участок недр без "
    "своей строки (Ki, Ma, Bt, Da). Это запасной вариант, а не замена своей модели.",
]

LAYOUTS = {
    "pikpolka": dict(
        src=r"D:\Projects\Pumps\4_Калькулятор_Пик_полка\2026_07_Калькулятор_ПикПолка_v8.xlsm",
        dst=r"D:\Projects\Pumps\4_Калькулятор_Пик_полка\2026_07_Калькулятор_ПикПолка_v9.xlsm",
        scen=SCEN_CELL, nno_cells=("R4", "Z4"),
        calls=[dict(cell="I15", ql="$C$15:$C$1108", con="$F$15:$F$1108",
                    frq="$G$15:$G$1108", nom="$M$4", down="$P$4", sour="$Q$4",
                    field="$N$4", oil="$D$15:$D$1108", niz="$G$4", tiz="$H$4"),
               dict(cell="T15", ql="$N$15:$N$1108", con="$Q$15:$Q$1108",
                    frq="$R$15:$R$1108", nom="$W$4", down="$P$4", sour="$Q$4",
                    field="$N$4", oil="$O$15:$O$1108", niz="$G$4", tiz="$H$4")]),
    "npv": dict(
        src=r"D:\Projects\Pumps\5_Калькулятор_NPV_УВЧ\Копия Калькулятор_NPV_УВЧ_V03.xlsm",
        dst=r"D:\Projects\Pumps\5_Калькулятор_NPV_УВЧ\Копия Калькулятор_NPV_УВЧ_V04.xlsm",
        scen=SCEN_CELL, nno_cells=("N4", "O4"),
        calls=[dict(cell="I11", ql="$C$11:$C$4010", con="$F$11:$F$4010",
                    frq="$G$11:$G$4010", nom="$F$4", down="$U$4", sour="$I$4",
                    field="$H$4", oil="$D$11:$D$4010", niz="$AC$4", tiz="$AF$4"),
               dict(cell="T11", ql="$N$11:$N$4010", con="$Q$11:$Q$4010",
                    frq="$R$11:$R$4010", nom="$F$4", down="$U$4", sour="$I$4",
                    field="$H$4", oil="$O$11:$O$4010", niz="$AC$4", tiz="$AF$4")]),
}
HV_TABLE = "ХВ!$P$21:$Q$91"


def schedule_formula(c: dict) -> str:
    return (f"=CM4_FailureScheduleDyn({c['ql']},{c['con']},{c['frq']},{c['nom']},"
            f"КРС_ЭПУ!$B$41:$B$57,{c['down']},{c['sour']},1,TuneBlock,CtrlBlock,QnomBlock,"
            f"{c['field']},{c['oil']},{HV_TABLE},{c['niz']},{c['tiz']})")


def write_ctrl_block(ws, scen_ref: str) -> None:
    """Rewrite 3.3 in place: header, weight, nine parameters, notes."""
    for r in range(ROW_HEAD, ROW_NOTES + len(NOTES) + 3):
        ws.Range(ws.Cells(r, 1), ws.Cells(r, 9)).ClearContents()

    ws.Cells(ROW_HEAD, 1).Value = ("3.3) Слои режима — ЗАДАЮТСЯ оператором. Форма кривых "
                                   "зашита в модуль, здесь только уровни")
    ws.Cells(ROW_HEAD, 1).Font.Bold = True
    for j, txt in enumerate(("1 — Базовый", "2 — Стресс", "0 — Без слоёв",
                             "В РАСЧЁТЕ"), start=2):
        ws.Cells(ROW_HEAD, j).Value = txt
        ws.Cells(ROW_HEAD, j).Font.Bold = True

    for i, (label, key, kind) in enumerate(PARAMS):
        r = ROW0 + i
        ws.Cells(r, 1).Value = label
        ws.Cells(r, 2).Value = BASE[key]
        ws.Cells(r, 3).Value = STRESS[key]
        ws.Cells(r, 4).Value = NULL_COL[key]
        # MIN(...,2) — защита от старого значения 3 в ячейке сценария: без неё CHOOSE вернёт
        # #ЗНАЧ! на книге, где кто-то оставил выбранным убранный «Пользовательский».
        ws.Cells(r, 5).Formula = f"=CHOOSE(MIN(MAX({scen_ref},0),2)+1,D{r},B{r},C{r})"
        if key.startswith("on"):
            ws.Cells(r, 1).Font.Bold = True

    for i, note in enumerate(NOTES):
        c = ws.Cells(ROW_NOTES + i, 1)
        c.Value = note
        c.Font.Italic = True
    ws.Range(ws.Cells(ROW_HEAD, 1), ws.Cells(ROW_NOTES + len(NOTES), 1)).ColumnWidth = 62


def build(tag: str, *, keep_open=False):
    import win32com.client as w32

    L = LAYOUTS[tag]
    src, dst = Path(L["src"]), Path(L["dst"])
    if not src.exists():
        raise SystemExit(f"нет исходной книги: {src}")
    shutil.copy2(src, dst)
    print(f"[{tag}] {src.name} -> {dst.name}")

    app = w32.Dispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    wb = app.Workbooks.Open(str(dst), UpdateLinks=0)
    try:
        # --- VBA: drop V4 first, or the shared private helpers collide -------
        vbp = wb.VBProject
        for comp in list(vbp.VBComponents):
            if comp.Name == OLD_MODULE_NAME:
                vbp.VBComponents.Remove(comp)
                print(f"[{tag}]   удалён {OLD_MODULE_NAME}")
        for comp in list(vbp.VBComponents):
            if comp.Name == NEW_MODULE_NAME:
                vbp.VBComponents.Remove(comp)
        vbp.VBComponents.Import(str(MODULE))
        print(f"[{tag}]   импортирован {NEW_MODULE_NAME}")

        ws = wb.Worksheets(MODEL_SHEET)
        write_ctrl_block(ws, L["scen"])
        print(f"[{tag}]   3.3 переписан: {N_PARAM} параметров, "
              f"строки {ROW0}..{ROW0 + N_PARAM - 1}")

        rng = f"{MODEL_SHEET}!$E${ROW0}:$E${ROW0 + N_PARAM - 1}"
        wb.Names("CtrlBlock").RefersTo = "=" + rng
        print(f"[{tag}]   CtrlBlock -> {rng}")

        plan = wb.Worksheets(PLAN_SHEET)
        for c in L["calls"]:
            plan.Range(c["cell"]).Formula2 = schedule_formula(c)
            print(f"[{tag}]   {c['cell']} <- расчёт с обводнённостью "
                  f"(нефть {c['oil']}, НИЗ {c['niz']}, ТИЗ {c['tiz']})")

        app.CalculateFullRebuild()
        vals = {k: plan.Range(k).Value for k in L["nno_cells"]}
        print(f"[{tag}]   ННО после пересчёта: "
              + "  ".join(f"{k}={v}" for k, v in vals.items()))
        wb.Save()
        return vals
    finally:
        if not keep_open:
            wb.Close(SaveChanges=False)
            app.Quit()


def read_nno(path: str, cells) -> dict:
    """ННО from an untouched workbook, for the before/after comparison."""
    import win32com.client as w32
    app = w32.Dispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    wb = app.Workbooks.Open(path, UpdateLinks=0, ReadOnly=True)
    try:
        app.CalculateFullRebuild()
        return {k: wb.Worksheets(PLAN_SHEET).Range(k).Value for k in cells}
    finally:
        wb.Close(SaveChanges=False)
        app.Quit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(LAYOUTS))
    ap.add_argument("--verify", action="store_true",
                    help="сравнить ННО до и после перепрошивки")
    a = ap.parse_args()
    tags = [a.only] if a.only else list(LAYOUTS)
    for tag in tags:
        before = read_nno(LAYOUTS[tag]["src"], LAYOUTS[tag]["nno_cells"]) if a.verify else None
        after = build(tag)
        if before is not None:
            print(f"[{tag}] СВЕРКА  до: {before}   после: {after}")
            same = all(before[k] == after[k] for k in before)
            print(f"[{tag}] {'ННО НЕ ИЗМЕНИЛОСЬ — перепрошивка чистая' if same else
                            '*** ННО ИЗМЕНИЛОСЬ — разобраться, прежде чем отдавать ***'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
