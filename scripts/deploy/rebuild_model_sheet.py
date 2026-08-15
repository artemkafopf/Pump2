"""Rebuild `МодельОтказов` as three blocks and drop every superseded calculation.

    python scripts/deploy/rebuild_model_sheet.py --workbook <path> --layout npv|pikpolka

The sheet had accreted three generations of the model.  What it looked like before:
inputs that half the sheet ignored, a «Проверка применимости» block still checking **v1**
bounds (Ql 100–823, Кпод 0.2–1.5) and calling ``ClampNote_v1``, three stray
``FailureFlag_v1`` cells in column N returning ``#VALUE!``, curve tables hardcoded to
«Борец» **and to Vt** — on a workbook whose model had just become per-field — and the
parameters split between the section header and a run of rows 40 below it.

After:

    1) ВВОД          the operating point the curves are drawn at; field, кислый and the θ
                     scenario are read from ПрогнозРемонтов so the sheet cannot disagree
                     with the forecast, and the resolved model key is shown
    2) КРИВЫЕ        Кпод / частота / Qном tables + their charts, all driven by block 1
                     (so they follow the selected field instead of always showing Vt)
    3) ПАРАМЕТРЫ     TuneBlock, QnomBlock, the four-scenario CtrlBlock, and the notes

Removed for good: the v1 applicability block, ``ClampNote_v1``, the ``FailureFlag_v1``
cells, and — in a second pass, after the sheet is saved — the whole ``ModuleFailureV1``
component.  The script refuses to remove it if any other VBA component still calls into it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

MODULE = REPO_ROOT / "vba" / "ModuleFailureV4.bas"
MODULE_NAME = "ModuleFailureV4"
#: The whole superseded line: the v1 model and the harness that exists only to test it.
LEGACY_MODULES = ("ModuleFailureV1", "TestFailureFlagV1")
MODEL_SHEET = "МодельОтказов"
PLAN_SHEET = "ПрогнозРемонтов"

LAYOUTS = {"npv": {"field": "$H$4", "sour": "$I$4", "scen": "$A$4"},
           "pikpolka": {"field": "$N$4", "sour": "$Q$4", "scen": "$A$4"}}

KPOD_GRID = [0.15, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95, 1.0, 1.2, 1.5, 1.8]
FREQ_GRID = [35, 38, 40, 43, 46, 50, 53, 56, 60, 63, 66, 70]
QNOM_GRID = [60, 100, 160, 250, 400, 640, 1000, 1600]

#: The frequency/Kpod scenario columns — one definition, imported so the two deploy scripts
#: cannot drift apart.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))
from wire_freq_kpod_scenarios import N_MULT, SCENARIOS  # noqa: E402

#: Scenario 0 — the layers switched OFF, θ_freq ≡ 1 and θ_Kpod ≡ 1, aligned with SCENARIOS.
#: Not a cosmetic "all ones": with both Kpod anchors at 1 the arm coefficient is exactly 0, and
#: the plateau is widened to the whole clamp so the arm cannot fire at all.  This is the
#: statistically defensible floor — fitted free, frequency is a null in every field, and it is
#: what the band should be read against.
NULL_VALUES = [1.0, 1.0, 1.0, 1.0,          # θ частоты 40 / 50 / 60 / 70
               1.0, 1.0,                     # θ_Кпод 0.2 / 1.2
               0.15, 1.8,                    # плато на всю область обрезки
               1.0, 1.0]                     # показатели плеч

#: Selector value -> column offset used by CHOOSE(selector + 1, …).
SCEN_LABEL = ("Сценарий θ частоты/Кпод: 0 = Без слоёв (θ ≡ 1), 1 = Базовый, "
              "2 = Стресс, 3 = Пользовательский")

NOTES = [
    "Обрезка: Qном 60–1600, частота 35–70 Гц, Кпод 0.15–1.8 — заданы в коде "
    "(полином частоты неограничен, обрезка несущая).",
    "Слой Кпод: «Базовый» — ПОДОГНАННЫЙ слой (низкий Кпод защищает, +8.4 CV), "
    "«Стресс» — операторская «ванна» (низкий Кпод наказывает). Это расхождение в ЗНАКЕ.",
    "Сценарий 0 — слои частоты и Кпод ВЫКЛЮЧЕНЫ (θ ≡ 1): остаются только базовый уровень, "
    "подрядчик и Qном, т.е. ровно то, что подобрано по данным. Это честный нижний край.",
    "Полоса сценариев — расхождение двух гипотез, а не доверительный интервал: при свободной "
    "подгонке частота — НОЛЬ во всех фондах (CV: Vt −0.6, Az −0.3, Ic −0.8, Ya +1.1, Za +1.9), "
    "т.е. статистически нижняя граница — θ ≡ 1.",
    "Fleet — пулированная подгонка по всему фонду (2308 прогонов, 1204 отказа); на неё уходит "
    "любой участок недр, у которого нет своей строки (Ki, Ma, Bt, Da). Уровни подрядчика "
    "взяты ВНУТРИ участков (фикс-эффекты), а вот слой Qном подогнан на пуле БЕЗ поправки на "
    "участок — различия базовых уровней между участками, скоррелированные с типоразмером, "
    "частично попадают в наклон. Это запасной вариант, а не замена своей модели.",
]


def load_blocks(tables: Path | None = None):
    if tables is None:
        found = sorted((REPO_ROOT / "results" / "production_risk_field_v4").glob("*/tables"))
        if not found:
            raise SystemExit("no field_v4 results — run scripts/run/field_v4.py first")
        tables = found[-1]
    return (pd.read_csv(tables / "deploy_tune_block.csv"),
            pd.read_csv(tables / "deploy_qnom_block.csv"), tables)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--layout", required=True, choices=tuple(LAYOUTS))
    p.add_argument("--tables", default=None)
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--keep-legacy-module", action="store_true")
    a = p.parse_args()

    wbpath = Path(a.workbook)
    lock = wbpath.with_name("~$" + wbpath.name)
    if not wbpath.exists():
        raise SystemExit(f"workbook not found: {wbpath}")
    if lock.exists():
        raise SystemExit(f"{wbpath.name} is open in Excel — close it first")
    tune, qnom, src = load_blocks(Path(a.tables) if a.tables else None)
    print(f"parameters from {src}: {len(tune)} tune rows, {len(qnom)} qnom rows")

    import win32com.client as win32

    backup = wbpath.with_suffix(".PRE_REBUILD.xlsm")
    i = 2
    while backup.exists():
        backup = wbpath.with_suffix(f".PRE_REBUILD.{i}.xlsm")
        i += 1
    shutil.copy2(wbpath, backup)
    print(f"backup -> {backup.name}")

    L = LAYOUTS[a.layout]
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
        nno_cells = ("R4", "Z4") if a.layout == "pikpolka" else ("N4", "O4")
        before = [pl.Range(c).Value for c in nno_cells]

        # ---- wipe: charts first, then every cell -----------------------------
        for co in list(sh.ChartObjects()):
            co.Delete()
        sh.Cells.ClearContents()
        print("sheet cleared")

        REF = ("CM4_LifeAtRef($B$3,TuneBlock,CtrlBlock,QnomBlock,$B$2)")

        # ---- 1) ВВОД ---------------------------------------------------------
        sh.Range("A1").Value = "1) ВВОД — точка, в которой строятся кривые блока 2"
        rows1 = [
            ("Участок недр (УН)", f"={PLAN_SHEET}!{L['field']}"),
            ("Кислый фонд (1 / 0)", f"={PLAN_SHEET}!{L['sour']}"),
            ("Ключ модели (определяется автоматически)",
             "=CM4_ResolveKey($B$2,$B$3,TuneBlock,QnomBlock)"),
            ("Сценарий θ (0 Без слоёв / 1 Базовый / 2 Стресс / 3 Пользовательский)",
             f"={PLAN_SHEET}!{L['scen']}"),
            ("Подрядчик", "Борец"),
            ("Qном, м³/сут", 250),
            ("Частота, Гц", 50),
            ("Кпод", 0.825),
        ]
        for r, (label, val) in enumerate(rows1, start=2):
            sh.Cells(r, 1).Value = label
            if isinstance(val, str) and val.startswith("="):
                sh.Cells(r, 2).Formula = val
            else:
                sh.Cells(r, 2).Value = val
        sh.Range("A10").Value = "ННО в этой точке, сут"
        sh.Range("B10").Formula = (
            "=CM4_LifeDays($B$3,$B$6,$B$7,$B$8,$B$9,TuneBlock,CtrlBlock,QnomBlock,$B$2)")
        sh.Range("A11").Value = "ННО в опорной точке (Борец, Qном 250, 50 Гц, Кпод в плато), сут"
        sh.Range("B11").Formula = "=" + REF
        sh.Range("A12").Value = ("Поля УН / кислый / сценарий берутся из ПрогнозРемонтов — "
                                 "лист не может разойтись с прогнозом. Подрядчик, Qном, "
                                 "частота и Кпод здесь свои: это точка построения кривых.")

        # ---- 2) КРИВЫЕ -------------------------------------------------------
        sh.Range("A14").Value = ("2) КРИВЫЕ — множитель ресурса RMST(0,730) относительно "
                                 "опорной точки; изменяется одна координата, остальные "
                                 "держатся на опорных")
        first = 16
        specs = [
            (1, "Кпод", KPOD_GRID,
             lambda cell: f"=CM4_LifeDays($B$3,$B$6,$B$7,$B$8,{cell},"
                          f"TuneBlock,CtrlBlock,QnomBlock,$B$2)/{REF}"),
            (4, "Частота, Гц", FREQ_GRID,
             lambda cell: f"=CM4_LifeDays($B$3,$B$6,$B$7,{cell},$B$9,"
                          f"TuneBlock,CtrlBlock,QnomBlock,$B$2)/{REF}"),
            (7, "Qном, м³/сут", QNOM_GRID,
             lambda cell: f"=CM4_LifeDays($B$3,$B$6,{cell},$B$8,$B$9,"
                          f"TuneBlock,CtrlBlock,QnomBlock,$B$2)/{REF}"),
        ]
        chart_ranges = []
        for col, title, grid, f in specs:
            sh.Cells(first - 1, col).Value = title
            sh.Cells(first - 1, col + 1).Value = "Множитель ресурса"
            for j, x in enumerate(grid):
                r = first + j
                sh.Cells(r, col).Value = x
                # ⚠ Range.Address is a PROPERTY under EnsureDispatch — Address(False, False)
                # raises "'str' object is not callable".
                sh.Cells(r, col + 1).Formula = f(sh.Cells(r, col).Address)
            chart_ranges.append((title, col, first, first + len(grid) - 1))

        for k, (title, col, r1, r2) in enumerate(chart_ranges):
            co = sh.ChartObjects().Add(sh.Range("K16").Left + k * 300,
                                       sh.Range("K16").Top, 290, 200)
            ch = co.Chart
            ch.ChartType = 75                                   # xlXYScatterLinesNoMarkers
            # Build the series explicitly rather than via SetSourceData: on a two-column
            # range Excel is free to read it as two value series, which silently plots the
            # x-axis as a curve.
            while ch.SeriesCollection().Count > 0:
                ch.SeriesCollection(1).Delete()
            s = ch.SeriesCollection().NewSeries()
            s.XValues = sh.Range(sh.Cells(r1, col), sh.Cells(r2, col))
            s.Values = sh.Range(sh.Cells(r1, col + 1), sh.Cells(r2, col + 1))
            ch.HasTitle = True
            ch.ChartTitle.Text = f"Множитель ресурса vs {title}"
            ch.HasLegend = False
        print(f"3 charts rebuilt over rows {first}:{first + len(FREQ_GRID) - 1}")

        # ---- 3) ПАРАМЕТРЫ ----------------------------------------------------
        base = first + len(FREQ_GRID) + 2
        sh.Cells(base, 1).Value = ("3) ПАРАМЕТРЫ МОДЕЛИ — Qном и подрядчик ПОДОБРАНЫ "
                                   "по каждому участку недр, частота и Кпод ЗАДАНЫ оператором")
        t_title = base + 2
        sh.Cells(t_title, 1).Value = "3.1) Базовый уровень и подрядчик  (TuneBlock)"
        for c, txt in ((2, "Больше — хуже"), (3, "Больше — лучше"), (4, "Больше — хуже")):
            sh.Cells(t_title, c).Value = txt
        for c, txt in enumerate(("ключ (участок недр)", "β (форма)", "RMST_ref, сут",
                                 "brt", "slb", "oth")):
            sh.Cells(t_title + 1, 1 + c).Value = txt
        t_first = t_title + 2
        for r, (_, row) in enumerate(tune.iterrows()):
            for c, key in enumerate(("key", "beta", "rmst_ref_days", "brt", "slb", "oth")):
                sh.Cells(t_first + r, 1 + c).Value = row[key]
        t_last = t_first + len(tune) - 1

        q_title = t_last + 2
        sh.Cells(q_title, 1).Value = ("3.2) Слой Qном  (QnomBlock): узлы и θ, "
                                      "вне крайних узлов θ держится плато")
        qcols = [c for c in qnom.columns if c.startswith("q")]
        q_head = q_title + 1
        sh.Cells(q_head, 1).Value = "Qном, м³/сут"
        for c, name in enumerate(qcols):
            sh.Cells(q_head, 2 + c).Value = float(name[1:])
        q_first = q_head + 1
        for r, (_, row) in enumerate(qnom.iterrows()):
            sh.Cells(q_first + r, 1).Value = row["key"]
            for c, name in enumerate(qcols):
                sh.Cells(q_first + r, 2 + c).Value = float(row[name])
        q_last = q_first + len(qnom) - 1

        c_title = q_last + 2
        sh.Cells(c_title, 1).Value = ("3.3) Частота и Кпод — ЗАДАЮТСЯ оператором, "
                                      "не подбирались  (CtrlBlock)")
        for j, cap in enumerate(("1 — Базовый", "2 — Стресс", "3 — Пользовательский",
                                 "0 — Без слоёв (θ ≡ 1)", "В РАСЧЁТЕ")):
            sh.Cells(c_title, 2 + j).Value = cap
        w_row = c_title + 1
        sh.Cells(w_row, 1).Value = ("вес стресса w в пользовательском сценарии "
                                    "(0 = базовый, 1 = стресс)")
        sh.Cells(w_row, 4).Value = float(a.weight)
        c_first = w_row + 1
        for r, (label, b, c) in enumerate(SCENARIOS):
            row = c_first + r
            sh.Cells(row, 1).Value = label
            sh.Cells(row, 2).Value = b
            sh.Cells(row, 3).Value = c
            sh.Cells(row, 4).Formula = (f"=B{row}^(1-$D${w_row})*C{row}^$D${w_row}"
                                        if r < N_MULT
                                        else f"=B{row}+$D${w_row}*(C{row}-B{row})")
            sh.Cells(row, 5).Value = NULL_VALUES[r]
            # selector 0..3 -> CHOOSE is 1-based, and column E (the null case) comes first
            sh.Cells(row, 6).Formula = f"=CHOOSE($B$5+1,E{row},B{row},C{row},D{row})"
        c_last = c_first + len(SCENARIOS) - 1

        for j, note in enumerate(NOTES):
            sh.Cells(c_last + 2 + j, 1).Value = note

        def refers(r1, c1, r2, c2):
            return f"='{sh.Name}'!{sh.Range(sh.Cells(r1, c1), sh.Cells(r2, c2)).Address}"

        wb.Names("TuneBlock").RefersTo = refers(t_first, 1, t_last, 6)
        wb.Names("QnomBlock").RefersTo = refers(q_head, 1, q_last, 9)
        wb.Names("CtrlBlock").RefersTo = refers(c_first, 6, c_last, 6)
        print(f"TuneBlock A{t_first}:F{t_last} · QnomBlock A{q_head}:I{q_last} · "
              f"CtrlBlock F{c_first}:F{c_last}")

        # selector on the input row, with the 0 case in the list
        sel = L["scen"].replace("$", "")
        pl.Range("A3").Value = SCEN_LABEL
        cur = pl.Range(sel).Value
        pl.Range(sel).Value = 1 if cur is None else int(cur)
        try:
            v = pl.Range(sel).Validation
            v.Delete()
            v.Add(Type=3, AlertStyle=1, Operator=1, Formula1="0,1,2,3")
            v.InputTitle = "Сценарий"
            v.InputMessage = SCEN_LABEL
        except Exception as exc:
            print(f"  (validation skipped: {exc})")

        sh.Columns("A").ColumnWidth = 52
        for col in "BCDEFGHI":
            sh.Columns(col).ColumnWidth = 12

        app.Calculation = -4105
        app.CalculateFullRebuild()
        after = [pl.Range(c).Value for c in nno_cells]
        print(f"ННО до {before} -> после {after}")
        if [round(float(x or 0)) for x in before] != [round(float(x or 0)) for x in after]:
            # The re-layout itself must not move a number, but this pass also refreshes the
            # parameter blocks, so a change can be legitimate — report it, do not hide it.
            print("   ^ прогноз изменился: это правки блоков параметров, не перекладка листа")
        wb.Save()
        print("sheet saved")

        # ---- VBA pass: refresh v4, drop v1 (sheets are saved first, by design) --
        proj = wb.VBProject
        users = []
        for comp in proj.VBComponents:
            if comp.Name in LEGACY_MODULES or comp.Name == MODULE_NAME:
                continue
            try:
                cm = comp.CodeModule
                txt = cm.Lines(1, cm.CountOfLines) if cm.CountOfLines else ""
            except Exception:
                txt = "FailureFlag_v1"          # unreadable → assume in use, keep the module
            for token in ("FailureFlag_v1", "ClampNote_v1", "FailureSchedule", "CM_Life",
                          "CM_Theta", "CM_RmstRef", "CM_Contractor"):
                if token in txt:
                    users.append((comp.Name, token))
        for comp in list(proj.VBComponents):
            if comp.Name == MODULE_NAME:
                proj.VBComponents.Remove(comp)
        proj.VBComponents.Import(str(MODULE))
        print(f"re-imported {MODULE_NAME}")
        if users:
            print(f"!! legacy modules kept — still referenced by {users}")
        elif a.keep_legacy_module:
            print("   legacy modules kept (--keep-legacy-module)")
        else:
            # ⚠ read the name BEFORE Remove(): afterwards the component object is dead and
            # comp.Name raises "The handle is invalid".
            for comp in list(proj.VBComponents):
                name = comp.Name
                if name in LEGACY_MODULES:
                    proj.VBComponents.Remove(comp)
                    print(f"removed {name}")

        app.CalculateFullRebuild()
        final = [pl.Range(c).Value for c in nno_cells]
        print(f"ННО после чистки VBA: {final}")
        for s in (0, 1, 2, 3):
            pl.Range(L["scen"].replace("$", "")).Value = s
            app.CalculateFullRebuild()
            print(f"  сценарий {s}: ННО "
                  f"{pl.Range(nno_cells[0]).Value:.0f} / {pl.Range(nno_cells[1]).Value:.0f}")
        pl.Range(L["scen"].replace("$", "")).Value = 1
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
