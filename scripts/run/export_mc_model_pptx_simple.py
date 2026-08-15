"""Простая одностраничная справка по модели отказов УЭЦН Мирнинского.

Один слайд: одна фигура + уравнение кривой дожития + таблица параметров. Ни разбора
происхождения β, ни описания сборки графика ремонтов — только модель и её параметры.

Фигура объединяет четыре кривые на одной выборке:
  * KM всей популяции и подгонка Вейбулла при θ = 1 (базовый случай);
  * KM подвыборки Qж > 2×медианы и модельная кривая при Qж = 3×медианы (θ ≈ 1.28).

Порог у факта (2×) и у модели (3×) РАЗНЫЕ намеренно: срез по дебиту маргинальный —
он вбирает и прочие отличия высокодебитных скважин, — поэтому эмпирическая группа
ложится примерно на модельную кривую более высокого дебита.

⚠ Qж считается по окну в 30 рабочих суток, поэтому его нет у пробегов, отказавших
раньше: подвыборка обусловлена дожитием и лишена младенческого слоя. На слайде это
отражено только числом покрытия в подписи фигуры.

Запуск:  python scripts/run/export_mc_model_pptx_simple.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import km_figure as KF
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import t0_covariates as T0

from export_mc_model_pptx import _pic, _table, _title, SLIDE_H, SLIDE_W
from export_mc_model_slides import C_FAIL, C_MODEL, QL, _style

KM_MULT = 1.0      # порог факта:   Qж > медианы
MODEL_MULT = 3.0   # модельная кривая: Qж = 3×медианы


def _mult_label(mult: float) -> str:
    """«медианы» вместо неуклюжего «1×медианы»."""
    return "медианы" if abs(mult - 1.0) < 1e-9 else "%g×медианы" % mult


def build_figure(mc: pd.DataFrame, model: M.SurvivalModel, out: Path) -> tuple[Path, dict]:
    plt = _style()
    cov, _ = T0.build(mc)
    have = cov[cov["ql"].notna() & (cov["ql"] > 0)]
    med = float(have["ql"].median())
    ref_log = float(np.median(np.log1p(have["ql"])))
    hi = have[have["ql"] > KM_MULT * med]

    beta = M.QL_BETA_LANDMARK
    th = float(np.exp(beta * (np.log1p(MODEL_MULT * med) - np.log1p(med))))

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.grid(True, axis="y", zorder=0)

    xb, yb, lb, ub = KF._km(mc["tte"].to_numpy(float),
                            (mc["event"] == 1).astype(int).to_numpy())
    xh, yh, lh, uh = KF._km(hi["tte"].to_numpy(float),
                            (hi["event"] == 1).astype(int).to_numpy())
    xmax = float(np.percentile(mc["tte"], 99))

    ax.fill_between(xb, lb, ub, step="post", color=C_MODEL, alpha=0.13, lw=0, zorder=2)
    ax.step(xb, yb, where="post", color=C_MODEL, lw=2.4, zorder=4,
            label="Факт (KM): весь парк — %d пробегов, %d отказов"
                  % (len(mc), int((mc["event"] == 1).sum())))
    ax.fill_between(xh, lh, uh, step="post", color=C_FAIL, alpha=0.13, lw=0, zorder=2)
    ax.step(xh, yh, where="post", color=C_FAIL, lw=2.4, zorder=4,
            label="Факт (KM): %s > %s — %d пробегов, %d отказов"
                  % (QL, _mult_label(KM_MULT), len(hi), int((hi["event"] == 1).sum())))

    grid = np.linspace(0, xmax, 700)
    s0 = model.survival(grid)
    ax.plot(grid, s0, color=C_MODEL, lw=2.2, ls="--", zorder=5,
            label="Модель Вейбулла, базовый случай (θ = 1.00)")
    ax.plot(grid, s0 ** th, color=C_FAIL, lw=2.2, ls="--", zorder=5,
            label="Модель при %s = %s (θ = %.2f)" % (QL, _mult_label(MODEL_MULT), th))

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Наработка, сут")
    ax.set_ylabel("S(t) — доля неотказавших насосов")
    ax.set_title("Мирнинский УН — наработка на отказ: факт против модели",
                 fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=10.5)
    fig.tight_layout()
    p = out / "simple_km_weibull_ql.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, {"med": med, "n_hi": len(hi), "ev_hi": int((hi["event"] == 1).sum()),
               "n_have": len(have), "theta": th}


def main() -> None:
    pop = P.build(str(C.RunConfig().forecast_start))
    model = M.fit_mc_model(pop)
    mc = P.select(pop, "Mc")
    mc = mc[mc["tte"] > 0].copy()

    out = results_dir("production_risk_mc_model_slides")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, info = build_figure(mc, model, figs)

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    s = prs.slides.add_slide(prs.slide_layouts[6])

    _title(s, "Модель наработки на отказ УЭЦН — Мирнинский УН",
           "Некислые. Событие — ОТКАЗ; ГТМ и работающие насосы цензурированы")
    _pic(s, fig, Inches(0.30), Inches(1.05), Inches(8.0))

    # уравнение — отдельным блоком, крупнее таблицы
    box = s.shapes.add_textbox(Inches(8.55), Inches(1.05), Inches(4.5), Inches(1.5))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (txt, sz, bold) in enumerate([
        ("Кривая дожития", 12, True),
        ("S(t) = w₁·exp(−(t/η₁)^β₁) + (1−w₁)·exp(−(t/η₂)^β₂)", 11.5, False),
        ("Поправка на дебит жидкости", 12, True),
        ("S(t | Qж) = S₀(t)^θ,   θ = exp(β·[ln(1+Qж) − ref])", 11.5, False),
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.name = "Calibri"

    _table(s, [
        ["Параметр", "Значение"],
        ["w₁ — доля 1-й компоненты", "%.4f" % model.w1],
        ["β₁ / η₁, сут — форма и масштаб 1-й", "%.3f / %.2f" % (model.beta1, model.eta1)],
        ["β₂ / η₂, сут — форма и масштаб 2-й", "%.3f / %.1f" % (model.beta2, model.eta2)],
        ["τ, сут — горизонт RMST (95-й проц.)", "%.0f" % model.tau],
        ["Пробегов / отказов / цензурировано", "%d / %d / %d"
         % (model.n_runs, model.n_events, model.n_censored)],
        ["RMST(0) / MRL(0), сут", "%.0f / %.0f" % (model.rmst(0.0), model.mrl(0.0))],
        ["β — коэффициент при log-Qж", "ln(1.253) = 0.226"],
        ["ref — опора", "медиана ln(1+Qж) по парку"],
        ["θ при Qж = 2× / 3× / 5× медианы", "1.17 / 1.28 / 1.44"],
    ], Inches(8.55), Inches(2.85), Inches(4.5), Inches(3.1),
        col_w=[2.55, 1.35], font=9.0)

    path = out / "Модель_отказов_Мирнинский_простая.pptx"
    prs.save(str(path))
    print("записано:", path)
    print("медиана Qж = %.0f | Qж > %s: %d пробегов, %d отказов | θ(%s) = %.2f"
          % (info["med"], _mult_label(KM_MULT), info["n_hi"], info["ev_hi"],
             _mult_label(MODEL_MULT), info["theta"]))


if __name__ == "__main__":
    main()
