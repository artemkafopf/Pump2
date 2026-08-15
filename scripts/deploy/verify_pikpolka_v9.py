"""Acceptance check for ПикПолка v9 / NPV V04 against the versions they replace.

    python scripts/deploy/verify_pikpolka_v9.py

Four questions, in the order they can actually falsify the build:

1. **Сценарий 0 must be unchanged.**  With every layer switched off only the baseline,
   the contractor and the nameplate remain — none of which v5.2 touched.  If this moves, the
   rewiring disturbed something it had no business touching.  This is the real invariant; an
   earlier version of this script asserted that ALL scenarios were unchanged, which was simply
   the wrong criterion: the whole point of v9 is that the layer curves change.

2. **The other scenarios must MOVE, and differ from each other.**  A build where every
   scenario returns the same number is the failure mode that already happened once — an
   unqualified ``$A$4`` in the CHOOSE resolved to an empty cell on «МодельОтказов», so every
   scenario silently read the «Без слоёв» column.  Identical answers are the symptom.

3. **«Стресс» must not give a longer life than «Базовый».**  It is an upper bound on the
   effect, so it can only shorten.

4. **The water-cut switch must be live.**  It ships ON; flipping it to 0 has to change the
   answer.  If it does not, the layer is wired to nothing and the switch is decorative.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))

from build_pikpolka_v9 import LAYOUTS, PLAN_SHEET, ROW0  # noqa: E402

WCUT_SWITCH_ROW = ROW0 + 7          # «Слой обводнённости (1 вкл / 0 выкл)»
#: «Пользовательский» убран из книги — проверять больше нечего, осталось три состояния.
SCEN = {0: "Без слоёв", 1: "Базовый", 2: "Стресс"}


def main() -> int:
    import win32com.client as w32

    app = w32.Dispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    bad = 0
    try:
        for tag, L in LAYOUTS.items():
            cells = L["nno_cells"]
            got = {}
            for ver, path in (("старая", L["src"]), ("новая", L["dst"])):
                wb = app.Workbooks.Open(path, UpdateLinks=0, ReadOnly=True)
                pl = wb.Worksheets(PLAN_SHEET)
                keep = pl.Range("A4").Value
                for scen in SCEN:
                    pl.Range("A4").Value = scen
                    app.CalculateFullRebuild()
                    got[(ver, scen)] = tuple(pl.Range(c).Value for c in cells)
                if ver == "новая":
                    ws = wb.Worksheets("МодельОтказов")
                    pl.Range("A4").Value = 1
                    for col in (2, 3, 4):          # Базовый / Стресс / Без слоёв
                        ws.Cells(WCUT_SWITCH_ROW, col).Value = 0
                    app.CalculateFullRebuild()
                    got[("новая+обв", 1)] = tuple(pl.Range(c).Value for c in cells)
                pl.Range("A4").Value = keep
                wb.Close(SaveChanges=False)

            print(f"\n=== {tag}   ячейки ННО {cells}")
            print(f"{'сценарий':>20} {'старая':>18} {'новая':>18}")
            for scen, nm in SCEN.items():
                a, b = got[("старая", scen)], got[("новая", scen)]
                print(f"{scen} {nm:>18} {str(a):>18} {str(b):>18}")

            inv = got[("старая", 0)] == got[("новая", 0)]
            print(f"\n  1. сценарий 0 не изменился ......... {'ДА' if inv else 'НЕТ ***'}")
            bad += 0 if inv else 1

            distinct = len({got[("новая", s)] for s in SCEN})
            ok2 = distinct == len(SCEN)
            print(f"  2. сценарии различаются ({distinct} из 3) ... "
                  f"{'ДА' if ok2 else 'НЕТ *** селектор не работает'}")
            bad += 0 if ok2 else 1

            base, stress = got[("новая", 1)], got[("новая", 2)]
            ok3 = all(s <= b for s, b in zip(stress, base))
            print(f"  3. «Стресс» не длиннее «Базового» .. {'ДА' if ok3 else 'НЕТ ***'}"
                  f"   ({base} -> {stress})")
            bad += 0 if ok3 else 1

            off = got[("новая+обв", 1)]
            ok4 = off != base
            print(f"  4. выключатель обводнённости живой . {'ДА' if ok4 else 'НЕТ *** ни на что не влияет'}"
                  f"   (вкл {base} -> выкл {off})")
            bad += 0 if ok4 else 1
    finally:
        app.Quit()

    print(f"\n{'ПРИЁМКА ПРОЙДЕНА' if bad == 0 else f'ПРОВАЛЕНО ПРОВЕРОК: {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
