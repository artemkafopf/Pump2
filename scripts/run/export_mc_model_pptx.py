"""PPTX-презентация по модели отказов УЭЦН Мирнинского УН (3 слайда, по-русски).

Слайд 1 — KM против подгонки Weibull + таблица ВСЕХ параметров смеси.
Слайд 2 — разрез по Ql + параметры Cox-слоя и откуда они взяты.
Слайд 3 — как строится график ремонтов: помесячная вероятность отказа (столбики),
          счёт числа отказавших скважин (перенос остатка) и порядок отказов (долг).

Фигуры 1–2 берутся готовыми из `export_mc_model_slides.py` (их надо прогнать раньше);
фигура 3 строится здесь, потому что она про процедуру раскладки, а не про подгонку.

Запуск:  python scripts/run/export_mc_model_pptx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0x21, 0x66, 0xAC)
RED = RGBColor(0xB2, 0x18, 0x2B)
HORIZON = 18


# ────────────────────────────── фигура слайда 3 ──────────────────────────────
def build_schedule_figure(out: Path) -> tuple[Path, dict]:
    """Столбики помесячной вероятности отказа + целочисленная раскладка с переносом."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    asof = pd.Timestamp(str(C.RunConfig().forecast_start))
    pop = P.build(str(C.RunConfig().forecast_start))
    model = M.fit_mc_model(pop)
    codes = set(pop[pop["field"] == "Mc"]["code"])
    live = M.live_fleet(pop, asof, field="Mc")
    ent = M.plan_entrants(set(live["code"]), codes, asof, HORIZON)
    fleet = M.active_fleet_by_month(codes=codes | set(ent["code"]))
    monthly, per_well = M.forecast(model, live, asof=asof, months=HORIZON,
                                   fleet_by_month=fleet, entrants=ent)
    by_code = M.allocate_events(monthly, per_well)

    alloc: dict[str, int] = {}
    for periods in by_code.values():
        for p in periods:
            alloc[str(p)] = alloc.get(str(p), 0) + 1
    months = list(monthly["месяц"])
    E = monthly["ожидаемые события"].to_numpy(float)
    N = np.array([alloc.get(m, 0) for m in months], dtype=float)
    pct = monthly["модель, % от парка"].to_numpy(float)

    # перенос остатка — воспроизводим ту же арифметику для иллюстрации
    carry, carries = 0.0, []
    for e in E:
        total = e + carry
        carry = total - int(np.floor(total))
        carries.append(carry)

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.55", "text.color": "0.15",
        "xtick.color": "0.35", "ytick.color": "0.35",
        "grid.color": "0.88", "grid.linewidth": 0.8,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})
    x = np.arange(len(months))

    ax1.grid(True, axis="y", zorder=0)
    ax1.bar(x, pct, color="#2166AC", width=0.68, zorder=3)
    ax1.set_ylabel("Вероятность отказа,\n% от парка в месяц")
    ax1.set_title("Шаг 1. Помесячная вероятность отказа: p(m) = ожидаемые отказы / парк в работе",
                  fontsize=12, loc="left")

    ax2.grid(True, axis="y", zorder=0)
    ax2.bar(x, E, color="#9EC4E4", width=0.68, zorder=3,
            label="ожидание E(m) — дробное")
    ax2.step(x, N, where="mid", color=RED_HEX, lw=2.2, zorder=5,
             label="в графике n(m) — целое (с переносом остатка)")
    ax2.plot(x, carries, color="#67001A", lw=1.4, ls=":", zorder=4,
             label="накопленный остаток (переносится вперёд)")
    ax2.set_ylabel("Отказов в месяц")
    ax2.set_xlabel("Месяц прогноза")
    ax2.set_title("Шаг 2. Сколько отказов ставим: n(m) = ⌊E(m) + остаток⌋, остаток копится по парку",
                  fontsize=12, loc="left")
    ax2.legend(frameon=False, fontsize=10, loc="upper left")

    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([m for m in months][::2], rotation=45, ha="right")
    fig.tight_layout()
    p = out / "slide4_schedule_bins.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    return p, {
        "sum_E": float(E.sum()), "sum_N": int(N.sum()),
        "n_live": int(len(live)), "n_ent": int(len(ent)),
        "months": HORIZON, "model": model,
        "fleet_first": int(monthly["в работе"].iloc[0]),
        "fleet_last": int(monthly["в работе"].iloc[-1]),
        "pct_min": float(np.nanmin(pct)), "pct_max": float(np.nanmax(pct)),
    }


RED_HEX = "#B2182B"


# ──────────────────────────────── помощники pptx ────────────────────────────────
def _title(slide, text: str, sub: str = "") -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.24), Inches(12.4), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = INK
    r.font.name = "Calibri"
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(13)
        r2.font.color.rgb = MUTED
        r2.font.name = "Calibri"


def _pic(slide, path: Path, x, y, w) -> None:
    slide.shapes.add_picture(str(path), x, y, width=w)


def _table(slide, rows: list[list[str]], x, y, w, h, col_w=None,
           head_fill=ACCENT, font=11.0):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h)
    tbl = shape.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for i, row in enumerate(rows):
        # Строка-подзаголовок группы: заполнена только первая ячейка. Помечаем заливкой,
        # иначе в плотной таблице разделы сливаются.
        group = i > 0 and len(row) > 1 and not str(row[1]).strip()
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            para = cell.text_frame.paragraphs[0]
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = head_fill
            elif group:
                cell.fill.fore_color.rgb = RGBColor(0xDD, 0xE7, 0xF2)
            else:
                cell.fill.fore_color.rgb = (RGBColor(0xF7, 0xF7, 0xF7) if i % 2
                                            else RGBColor(0xFF, 0xFF, 0xFF))
            if not para.runs:            # пустая ячейка — стилить нечего
                continue
            r = para.runs[0]
            r.font.size = Pt(font)
            r.font.name = "Calibri"
            r.font.bold = bool(i == 0 or group)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
    return tbl


def _bullets(slide, x, y, w, h, items: list[tuple[str, str]], size=12.0) -> None:
    """items = [(заголовок, текст)]; заголовок жирным, текст обычным."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for head, body in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(7)
        if head:
            r = p.add_run()
            r.text = head + " "
            r.font.bold = True
            r.font.size = Pt(size)
            r.font.color.rgb = INK
            r.font.name = "Calibri"
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(size)
        r2.font.color.rgb = MUTED
        r2.font.name = "Calibri"


# ──────────────────────────────────── слайды ────────────────────────────────────
def main() -> None:
    figs_dir = results_dir("production_risk_mc_model_slides") / "figures"
    f1 = figs_dir / "slide1_km_vs_weibull.png"
    f2 = figs_dir / "slide2_km_by_ql.png"
    for f in (f1, f2):
        if not f.exists():
            raise SystemExit("нет %s — сначала прогнать export_mc_model_slides.py" % f)

    f3, info = build_schedule_figure(figs_dir)
    m: M.SurvivalModel = info["model"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    blank = prs.slide_layouts[6]

    # ── Слайд 1 ─────────────────────────────────────────────────────────────
    s1 = prs.slides.add_slide(blank)
    _title(s1, "Модель наработки до отказа УЭЦН — Мирнинский УН",
           "Монтажи 2024+, некислые. Событие — ОТКАЗ; ГТМ и работающие насосы в цензуру. "
           "Смесь двух распределений Вейбулла (cal c0 k2)")
    _pic(s1, f1, Inches(0.4), Inches(1.55), Inches(7.75))
    _table(s1, [
        ["Параметр", "Значение", "Смысл"],
        ["w₁", "%.4f" % m.w1, "доля 1-й компоненты"],
        ["β₁", "%.3f" % m.beta1, "форма 1-й компоненты"],
        ["η₁, сут", "%.2f" % m.eta1, "масштаб 1-й компоненты"],
        ["β₂", "%.3f" % m.beta2, "форма 2-й компоненты"],
        ["η₂, сут", "%.1f" % m.eta2, "масштаб 2-й компоненты"],
        ["τ, сут", "%.0f" % m.tau, "горизонт RMST (95-й проц.)"],
        ["Пробегов", "%d" % m.n_runs, "в подгонке"],
        ["Отказов", "%d" % m.n_events, "событий"],
        ["Цензурировано", "%d" % m.n_censored, "ГТМ + в работе"],
        ["Отсечка c, сут", "%.0f" % m.cut_days, "ранние НЕ вырезаны"],
    ], Inches(8.45), Inches(1.62), Inches(4.5), Inches(3.9), col_w=[1.15, 0.95, 1.6], font=10.5)
    _bullets(s1, Inches(8.45), Inches(5.75), Inches(4.5), Inches(1.5), [
        ("S(t) =", "w₁·exp(−(t/η₁)^β₁) + (1−w₁)·exp(−(t/η₂)^β₂)"),
        ("Проверка:", "RMST(0) модели %.0f сут против %.0f у факта (KM) — расхождение 1.4%%; "
                      "модельная кривая не выходит за 95%% ДИ внутри τ."
                      % (m.rmst(0.0), 347)),
    ], size=11)

    # ── Слайд 2 ─────────────────────────────────────────────────────────────
    s2 = prs.slides.add_slide(blank)
    _title(s2, "Слой дебита жидкости Ql — параметры Cox и их происхождение",
           "Риск умножается на θ: S(t | Ql) = S₀(t)^θ. Слой СЦЕНАРНЫЙ — в базовый расчёт не включён")
    _pic(s2, f2, Inches(0.4), Inches(1.55), Inches(7.75))
    _table(s2, [
        ["Параметр", "Значение"],
        ["β (при log-Ql)", "ln(1.253) = 0.226"],
        ["θ(Ql)", "exp(β·[ln(1+Ql) − ref])"],
        ["ref (опора)", "медиана ln(1+Ql) по парку"],
        ["Цап на |ln(1+Ql) − ref|", "±ln(5)"],
        ["θ при Ql = 3×медианы", "1.28"],
        ["θ при Ql = 5×медианы", "1.44"],
        ["Применение", "q_eff = 1 − (1 − q)^θ"],
    ], Inches(8.45), Inches(1.62), Inches(4.5), Inches(2.7), col_w=[1.35, 1.35], font=10.5)
    _bullets(s2, Inches(8.45), Inches(4.55), Inches(4.5), Inches(2.7), [
        ("Откуда β.", "Landmark-схема: ковариата берётся из окна [0, 30) суток, риск-набор — "
                      "дожившие до 30 суток. Оценено глобально по 19 стратам, n = 1741, p < 1e-4."),
        ("Проверка на Vt_nonsour.", "Cox на 476 пробегах / 209 отказах даёт β = 0.296 "
                                    "(95% ДИ 0.116…0.476, p = 0.001) — интервал накрывает 0.226."),
        ("Чего β НЕ значит.", "Это межскважинная связь: годится для ранжирования, но частично "
                              "несёт неизмеренные свойства скважины и не является рычагом. "
                              "Внутрипробеговый causal-эффект отдельный и меньше (0.07)."),
        ("Статус.", "Гейт c4 слой не промотировал (выигрыш MAE −0.1% при пороге 5%), "
                    "поэтому в базовом прогнозе θ ≡ 1."),
    ], size=10.5)

    # ── Слайд 3 ─────────────────────────────────────────────────────────────
    s3 = prs.slides.add_slide(blank)
    _title(s3, "Как из модели строится график: число отказов и их порядок",
           "Модель даёт МЕСЯЧНУЮ вероятность, а не дату. Отказ ставится на 1-е число своего месяца")
    _pic(s3, f3, Inches(0.4), Inches(1.5), Inches(7.5))
    _bullets(s3, Inches(8.2), Inches(1.6), Inches(4.8), Inches(5.4), [
        ("1. Вероятность.", "Для каждого насоса за месяц: p = 1 − S(a+Δ)/S(a), где a — его "
                            "возраст. Складываем по парку → ожидание E(m) (дробное)."),
        ("2. Сколько отказов.", "n(m) = ⌊E(m) + остаток⌋. Дробный хвост НЕ выбрасывается, "
                                "а копится по всему парку и добавляет отказ, когда сумма "
                                "переходит 1. Итог за %d мес: ΣE = %.1f, целых событий в "
                                "графике — %d."
                                % (info["months"], info["sum_E"], info["sum_N"])),
        ("3. Кто отказывает.", "risk_30d — ожидаемое число отказов насоса за месяц. ДОЛГ — сумма "
                               "risk_30d, накопленная С ДАТЫ РАСЧЁТА (у всех стартует с нуля, "
                               "прошлая наработка в него не входит). Событие достаётся насосам с "
                               "наибольшим долгом, после чего его долг уменьшается на 1."),
        ("Почему долг, а не «самый рискованный».", "Строгая очередь по риску всегда отдаёт отказ "
                                                   "старому насосу (6.4%/мес против 4.7% у нового) "
                                                   "и морит новые скважины. Долг сохраняет и месячный "
                                                   "итог, и порядок по возрасту."),
        ("Откуда берётся порядок.", "Из ФОРМЫ ОПАСНОСТИ, а не из правила долга: долг у всех "
                                    "стартует с нуля, решают только приросты внутри горизонта. У Mc "
                                    "risk_30d растёт с возрастом (0.026 при 30 сут → 0.039 при 150), "
                                    "отсюда «старые первыми», Спирмен возраст↔дата = −0.99. На страте "
                                    "с beta<1 тот же код дал бы обратный порядок."),
        ("4. Обновление.", "Отказавший насос сразу меняется на новый с возраста 0 и снова копит "
                           "риск — поэтому одна скважина может отказать за горизонт дважды."),
        ("Парк.", "Активные добывающие скважины из плана ПП: %d → %d за горизонт "
                  "(%d живых насосов + %d вводимых)."
                  % (info["fleet_first"], info["fleet_last"], info["n_live"], info["n_ent"])),
    ], size=10.5)

    out = results_dir("production_risk_mc_model_slides")
    path = out / "Модель_отказов_Мирнинский.pptx"
    prs.save(str(path))
    print("записано:", path)
    print("слайдов: %d | фигура слайда 3: %s" % (len(prs.slides.__iter__.__self__._sldIdLst), f3.name))
    print("ΣE=%.1f → %d отказов | парк %d→%d | живых %d + ВНС %d"
          % (info["sum_E"], info["sum_N"], info["fleet_first"], info["fleet_last"],
             info["n_live"], info["n_ent"]))


if __name__ == "__main__":
    main()
