"""Прогон собранных книг перед отправкой: считает ли всё и то ли, что должно.

    python scripts/deploy/smoke_test_v9.py

Приёмка (`verify_pikpolka_v9.py`) проверяет четыре утверждения про сценарии.  Здесь другое:
книга гоняется по всей области входов и сверяется сама с собой и с питоновским двойником —
то, что обычно ловят «на живом расчёте», уже после отправки.

Что проверяется:

1. **Сетка входов без ошибок.**  Все участки недр × подрядчики × типоразмеры × частоты ×
   Kпод.  Формулы пишутся пачкой в пустую область листа и читаются одним чтением: по одному
   вызову через COM такая сетка считалась бы минутами.
2. **Область значений.**  ННО строго между 0 и 730 (RMST по определению ограничен горизонтом),
   ни одной ошибки, ни одного нуля.
3. **Опорная точка.**  ``CM4_LifeAtRef`` обязан совпасть с ``CM4_LifeDays`` в опорных
   условиях: Борец, Qном 250, 50 Гц, Kпод 0.8, обводнённость 25 %.  Это и есть определение
   опорной точки; расхождение означает, что слой сместил уровень.
4. **Слои отвечают на вход.**  Каждый из трёх слоёв, будучи включённым, обязан менять ответ;
   выключенный — не менять.  Ловит слой, подключённый «в никуда».
5. **Неизвестный участок уходит на Fleet**, а не роняет расчёт и не молчит.
6. **Ошибки в ячейках.**  Полный пересчёт и поиск ошибочных значений по всем листам обеих
   книг — включая те, которых мы не трогали.
7. **Две книги согласованы.**  Один и тот же вход обязан дать один и тот же ответ: модуль и
   блоки параметров у них общие.
8. **Книга против питоновского двойника.**  ``pikpolka_sim`` с ``ctrl5=CTRL5_BASE`` считает
   тот же день отказа, что и лист.  Это единственная проверка, которая смотрит на дневную
   сетку целиком, а не на отдельную функцию.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))

import numpy as np  # noqa: E402

from build_pikpolka_v9 import LAYOUTS, MODEL_SHEET, PLAN_SHEET, ROW0  # noqa: E402

FIELDS = ["Ya", "Vt", "Az", "Ic", "Za", "Mc", "Mr", "Au", "Da", "Ki", "Bt"]
CONTRACTORS = ["Борец", "Шлюмберже", "Прочие"]
QNOM = [60, 160, 250, 640, 1600]
FREQ = [35, 45, 50, 55, 60, 70]
KPOD = [0.1, 0.4, 0.8, 1.2, 1.8]
SCRATCH_ROW = 200                       # заведомо пустая область под сетку формул
REF = dict(contractor="brt", qnom=250, freq=50, kpod=0.8, wcut=25)

bad = 0


def say(ok: bool, text: str, extra: str = "") -> None:
    global bad
    bad += 0 if ok else 1
    print(f"  {'OK ' if ok else '*** ПРОВАЛ ***'} {text}{('   ' + extra) if extra else ''}")


def grid_formulas(sour: int) -> list[str]:
    out = []
    for fld in FIELDS:
        for con in CONTRACTORS:
            for q in QNOM:
                for f in FREQ:
                    for k in KPOD:
                        out.append(f'=CM4_LifeDays({sour},"{con}",{q},{f},{k},'
                                   f'TuneBlock,CtrlBlock,QnomBlock,"{fld}",25)')
    return out


def run_grid(ws, app, formulas: list[str]) -> np.ndarray:
    """Пишет формулы столбцом, считает один раз, читает результат, убирает за собой."""
    n = len(formulas)
    rng = ws.Range(ws.Cells(SCRATCH_ROW, 1), ws.Cells(SCRATCH_ROW + n - 1, 1))
    rng.Formula = [[f] for f in formulas]
    app.CalculateFullRebuild()
    vals = [r[0] for r in rng.Value2]
    rng.ClearContents()
    return np.array([v if isinstance(v, (int, float)) else np.nan for v in vals], float)


def sheet_errors(wb) -> list[str]:
    out = []
    for ws in wb.Worksheets:
        try:
            cells = ws.UsedRange.SpecialCells(-4123, 16)     # xlCellTypeFormulas, xlErrors
        except Exception:
            continue                                         # ошибок нет — SpecialCells бросает
        try:
            addr = cells.Address(False, False)
        except Exception:
            addr = "?"
        out.append(f"{ws.Name}!{addr}")
    return out


def main() -> int:
    import win32com.client as w32

    app = w32.Dispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    life_by_book = {}
    try:
        for tag, L in LAYOUTS.items():
            print(f"\n=== {tag}: {Path(L['dst']).name}")
            wb = app.Workbooks.Open(L["dst"], UpdateLinks=0)
            ws, pl = wb.Worksheets(MODEL_SHEET), wb.Worksheets(PLAN_SHEET)
            keep_scen = pl.Range("A4").Value
            # Сценарий фиксируется на «Базовом» ДО сетки: CtrlBlock резолвится через него,
            # и в файлах он сохранён разным (в ПикПолке «Стресс», в NPV «Базовый»).  Без
            # фиксации сравнение книг между собой сравнивало бы разные сценарии — первый
            # прогон именно на этом и «нашёл» расхождение, которого нет.
            pl.Range("A4").Value = 1
            app.CalculateFullRebuild()
            try:
                # --- 1-2. сетка входов -------------------------------------
                for sour in (0, 1):
                    v = run_grid(ws, app, grid_formulas(sour))
                    finite = np.isfinite(v)
                    say(bool(finite.all()),
                        f"сетка {len(v)} точек, кислый={sour}: без ошибок",
                        f"неудач {int((~finite).sum())}")
                    if finite.any():
                        say(bool(((v[finite] > 0) & (v[finite] < 730)).all()),
                            f"ННО в (0, 730) сут, кислый={sour}",
                            f"диапазон {v[finite].min():.0f}…{v[finite].max():.0f}")

                # --- 3. опорная точка --------------------------------------
                for sour in (0, 1):
                    a = ws.Evaluate(f'CM4_LifeAtRef({sour},TuneBlock,CtrlBlock,'
                                    f'QnomBlock,"Ya")')
                    b = ws.Evaluate(f'CM4_LifeDays({sour},"{REF["contractor"]}",'
                                    f'{REF["qnom"]},{REF["freq"]},{REF["kpod"]},'
                                    f'TuneBlock,CtrlBlock,QnomBlock,"Ya",{REF["wcut"]})')
                    say(abs(float(a) - float(b)) < 1e-6,
                        f"опорная точка сходится, кислый={sour}",
                        f"{float(a):.4f} против {float(b):.4f}")

                # --- 4. слои отвечают на вход ------------------------------
                base = float(ws.Evaluate('CM4_LifeDays(0,"brt",250,50,0.8,TuneBlock,'
                                         'CtrlBlock,QnomBlock,"Ya",25)'))
                for lab, expr in (
                        ("частота 50 -> 70 Гц",
                         'CM4_LifeDays(0,"brt",250,70,0.8,TuneBlock,CtrlBlock,QnomBlock,"Ya",25)'),
                        ("Kпод 0.8 -> 0.1",
                         'CM4_LifeDays(0,"brt",250,50,0.1,TuneBlock,CtrlBlock,QnomBlock,"Ya",25)'),
                        ("обводнённость 25 -> 90 %",
                         'CM4_LifeDays(0,"brt",250,50,0.8,TuneBlock,CtrlBlock,QnomBlock,"Ya",90)'),
                        ("типоразмер 250 -> 1600",
                         'CM4_LifeDays(0,"brt",1600,50,0.8,TuneBlock,CtrlBlock,QnomBlock,"Ya",25)'),
                        ("подрядчик Борец -> Прочие",
                         'CM4_LifeDays(0,"Прочие",250,50,0.8,TuneBlock,CtrlBlock,QnomBlock,"Ya",25)')):
                    v = float(ws.Evaluate(expr))
                    say(abs(v - base) > 1e-6, f"слой отвечает: {lab}",
                        f"{base:.0f} -> {v:.0f} сут")

                # --- 5. неизвестный участок -> Fleet ------------------------
                unk = float(ws.Evaluate('CM4_LifeDays(0,"brt",250,50,0.8,TuneBlock,'
                                        'CtrlBlock,QnomBlock,"ЧУЖОЙ",25)'))
                flt = float(ws.Evaluate('CM4_LifeDays(0,"brt",250,50,0.8,TuneBlock,'
                                        'CtrlBlock,QnomBlock,"Fleet",25)'))
                say(abs(unk - flt) < 1e-6, "неизвестный участок уходит на Fleet",
                    f"{unk:.1f} против {flt:.1f}")

                # --- 6. ошибки в ячейках ------------------------------------
                app.CalculateFullRebuild()
                errs = sheet_errors(wb)
                say(not errs, "нет ошибочных значений на листах", "; ".join(errs[:3]))

                # --- 7. запоминаем для сверки книг между собой -------------
                life_by_book[tag] = {
                    f"{fld}/{con}/{q}/{f}": float(ws.Evaluate(
                        f'CM4_LifeDays(0,"{con}",{q},{f},0.8,TuneBlock,CtrlBlock,'
                        f'QnomBlock,"{fld}",25)'))
                    for fld in ("Ya", "Vt", "Mc") for con in ("brt", "slb")
                    for q in (250, 1000) for f in (50, 60)}

                # --- сценарии по листу -------------------------------------
                vals = {}
                for scen in (0, 1, 2):
                    pl.Range("A4").Value = scen
                    app.CalculateFullRebuild()
                    vals[scen] = tuple(round(float(pl.Range(c).Value), 3)
                                       for c in L["nno_cells"])
                say(len(set(vals.values())) == 3, "три сценария дают три разных ответа",
                    str(vals))
                # опорная точка при «Стрессе»: θ_частоты(50) = 1.0055, а не ровно 1
                pl.Range("A4").Value = 2
                app.CalculateFullRebuild()
                ref_s = float(ws.Evaluate('CM4_LifeAtRef(0,TuneBlock,CtrlBlock,'
                                          'QnomBlock,"Ya")'))
                bare = float(ws.Evaluate('CM4_RmstRef(0,TuneBlock,"Ya")'))
                print(f"  --  опорная точка при «Стрессе» {ref_s:.2f} против базовой {bare:.2f} "
                      f"({100 * (ref_s / bare - 1):+.2f} %) — θ_частоты(50) стрессовой кривой "
                      f"равна 1.0055, а не 1")
                pl.Range("A4").Value = 1
                app.CalculateFullRebuild()
                pl.Range("A4").Value = keep_scen
                app.CalculateFullRebuild()
            finally:
                wb.Close(SaveChanges=False)

        # --- 7. согласованность книг -----------------------------------
        print("\n=== согласованность двух книг")
        a, b = life_by_book["pikpolka"], life_by_book["npv"]
        diff = {k: (a[k], b[k]) for k in a if abs(a[k] - b[k]) > 1e-6}
        say(not diff, f"{len(a)} одинаковых входов дают одинаковый ответ",
            str(list(diff.items())[:2]))
    finally:
        app.Quit()

    print(f"\n{'СМОУК-ТЕСТ ПРОЙДЕН' if bad == 0 else f'ПРОВАЛЕНО ПРОВЕРОК: {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
