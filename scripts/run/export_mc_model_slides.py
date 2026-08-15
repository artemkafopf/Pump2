"""Слайды по модели Мирнинского (cal c0 k2): KM vs Weibull и разрез по Ql.

Слайд 1 — KM «только отказы» (ННО, ГТМ и работающие в цензуру) + подгонка cal c0 k2
на ТОЙ ЖЕ выборке (`fit_mc_model` и `km_figure` используют одинаковый отбор:
`P.select(pop, "Mc")` = Mc + nonsour + монтажи 2024+, tte>0).

Слайд 2 — та же кривая, разрезанная по Ql (median-split), плюс модельные кривые
S(t|Ql) = S0(t)^θ, θ = exp(β·(log1p(Ql) − ref)) при Ql = медиана и Ql = 2×медиана.
β берётся из `esp_forecast.QL_BETA_LANDMARK`; рядом печатается Cox-β, оценённый на
самих данных Mc, — чтобы было видно, подтверждается ли перенесённый коэффициент.

Ql — из `t0_covariates.build`: среднее qliq за первые 30 РАБОЧИХ суток пробега.
⚠ Окно t0 есть не у всех пробегов (телеметрия начинается позже части монтажей);
покрытие печатается и выносится в подпись, а не замалчивается.

Запуск:  python scripts/run/export_mc_model_slides.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import km_figure as KF
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import t0_covariates as T0

ASOF = "2026-07-01"
# Палитра проверена scripts/validate_palette.js (light): ΔE protan 21.1, normal 28.7 —
# все шесть проверок PASS. Модель отличается ещё и штрихом (secondary encoding).
C_FAIL = "#B2182B"       # эмпирика / высокий Ql
C_MODEL = "#2166AC"      # модель / базовая кривая
C_FAIL_DARK = "#67001A"  # второй шаг красной секвенции (Ql = 3×медианы)
# Подпись дебита жидкости на графиках — принятая в отчётности кириллица, а не Ql.
QL = "Qж"
GRID = "0.88"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.55", "axes.labelcolor": "0.2",
        "text.color": "0.15", "xtick.color": "0.35", "ytick.color": "0.35",
        "grid.color": GRID, "grid.linewidth": 0.8,
    })
    return plt


def slide1(mc: pd.DataFrame, model: M.SurvivalModel, out: Path) -> dict:
    """KM (только отказы) + Weibull cal c0 k2 на той же выборке."""
    plt = _style()
    t = mc["tte"].to_numpy(float)
    e = (mc["event"] == 1).astype(int).to_numpy()
    x, y, lo, hi = KF._km(t, e)
    tau = model.tau
    grid = np.linspace(0, float(x.max()), 600)
    s_hat = model.survival(grid)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.grid(True, axis="y", zorder=0)
    ax.fill_between(x, lo, hi, step="post", color=C_FAIL, alpha=0.16, lw=0,
                    label="95% ДИ Каплана—Мейера", zorder=2)
    ax.step(x, y, where="post", color=C_FAIL, lw=2.4, zorder=4,
            label="Kaplan–Meier (факт, только отказы)")
    ax.plot(grid, s_hat, color=C_MODEL, lw=2.4, ls="--", zorder=5,
            label="Weibull cal c0 k2 (модель)")
    ax.axvline(tau, color="0.5", ls=":", lw=1.3, zorder=1)
    ax.text(tau + 6, 0.97, "τ = %.0f сут\n(95-й перцентиль)" % tau,
            color="0.4", fontsize=10, va="top")

    ax.set_xlim(0, float(x.max()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Наработка, сут")
    ax.set_ylabel("S(t) — доля неотказавших насосов")
    ax.set_title(
        "Мирнинский УН, монтажи 2024+ — наработка до ОТКАЗА\n"
        "Kaplan–Meier против подгонки cal c0 k2 (%d пробегов, %d отказов, %d цензурировано)"
        % (model.n_runs, model.n_events, model.n_censored),
        fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    fig.tight_layout()
    p = out / "slide1_km_vs_weibull.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    km_rmst = KF._rmst(x, y, tau)
    return {"path": p, "km_rmst": km_rmst, "model_rmst": model.rmst(0.0),
            "km_tail": float(y[-1]), "model_tail": float(model.survival(float(x.max())))}


def fit_mixture(sel: pd.DataFrame, cut_days: float = 0.0) -> M.SurvivalModel:
    """Та же смесь двух Вейбуллов, что и `fit_mc_model`, но для ЛЮБОГО среза.

    Нужна для Vt_nonsour: `fit_mc_model` жёстко берёт Mc. Ограничения
    beta2 > max(1, beta1) и eta2 >= 2*eta1 те же — иначе компоненты меняются местами.
    """
    from analysis.models.survival.weibull_em import fit_latent_weibull_em

    g = sel[sel["tte"] > 0].copy()
    g["event"] = (g["event"] == 1).astype(int)
    early = (g["event"] == 1) & (g["tte"] < cut_days)
    fitset = g[~early]
    r = fit_latent_weibull_em(fitset["tte"].to_numpy(float),
                              fitset["event"].to_numpy(int), num_starts=60)
    m = r.model
    return M.SurvivalModel(
        w1=float(m.weight_1),
        beta1=float(m.component_1.beta), eta1=float(m.component_1.eta),
        beta2=float(m.component_2.beta), eta2=float(m.component_2.eta),
        n_runs=int(len(fitset)), n_events=int(fitset["event"].sum()),
        n_censored=int((fitset["event"] == 0).sum()), n_cut=int(early.sum()),
        cut_days=float(cut_days),
        tau=float(np.percentile(fitset["tte"].to_numpy(float), 95)),
    )


def slide2(mc: pd.DataFrame, model: M.SurvivalModel, out: Path,
           fname: str = "slide2_km_by_ql.png",
           field_label: str = "Мирнинский УН 2024+") -> dict:
    """KM базовая + KM при Ql>=медианы, против модели S0^θ (θ=1 и θ при 2×медианы)."""
    plt = _style()
    cov_df, cov = T0.build(mc)
    have = cov_df[cov_df["ql"].notna() & (cov_df["ql"] > 0)].copy()
    med = float(have["ql"].median())
    ref_log = float(np.median(np.log1p(have["ql"])))
    lo_g = have[have["ql"] < med]
    hi_g = have[have["ql"] >= med]

    res = {"n_total": int(len(mc)), "n_with_ql": int(len(have)), "median_ql": med,
           "n_lo": int(len(lo_g)), "n_hi": int(len(hi_g)),
           "ev_lo": int((lo_g["event"] == 1).sum()), "ev_hi": int((hi_g["event"] == 1).sum())}

    # Cox на самих данных Mc — проверка перенесённого β, а не украшение.
    try:
        from lifelines import CoxPHFitter
        d = have[["tte"]].copy()
        d["event"] = (have["event"] == 1).astype(int)
        d["logql"] = np.log1p(have["ql"]) - ref_log
        cph = CoxPHFitter().fit(d, duration_col="tte", event_col="event")
        res["cox_beta"] = float(cph.params_["logql"])
        res["cox_p"] = float(cph.summary.loc["logql", "p"])
        res["cox_lo"] = float(cph.confidence_intervals_.loc["logql"].iloc[0])
        res["cox_hi"] = float(cph.confidence_intervals_.loc["logql"].iloc[1])
    except Exception as exc:                       # noqa: BLE001
        res["cox_error"] = str(exc)

    beta = M.QL_BETA_LANDMARK

    def theta(mult: float) -> float:
        return float(np.exp(beta * (np.log1p(mult * med) - np.log1p(med))))

    th_3x, th_5x = theta(3.0), theta(5.0)
    res["beta_landmark"] = float(beta)
    res["theta_3x"], res["theta_5x"] = th_3x, th_5x
    res["theta_2x"] = theta(2.0)          # справочно: средняя θ группы ≈ этой величине

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.grid(True, axis="y", zorder=0)
    xmax = float(np.percentile(have["tte"], 98))
    # Базовая кривая — ВСЯ популяция без слоя Ql: именно к ней относится модель при
    # θ=1, и именно она содержит младенческий слой, которого нет в Ql-подвыборке.
    base = (mc, C_MODEL, "вся популяция, без слоя %s" % QL)
    for g, color, lab in (base, (hi_g, C_FAIL, "%s ≥ медианы" % QL)):
        x, y, l, h = KF._km(g["tte"].to_numpy(float), (g["event"] == 1).astype(int).to_numpy())
        ax.fill_between(x, l, h, step="post", color=color, alpha=0.13, lw=0, zorder=2)
        ax.step(x, y, where="post", color=color, lw=2.4, zorder=4,
                label="KM: %s (n=%d, отказов %d)" % (lab, len(g), int((g["event"] == 1).sum())))
    grid = np.linspace(0, xmax, 600)
    s0 = model.survival(grid)
    # Модельные кривые — упорядоченный ряд по Ql, поэтому это СЕКВЕНЦИЯ, а не категории:
    # θ=1 синим (пара к базовой KM), 2× и 3× — два шага красного от светлого к тёмному
    # плюс разный штрих. Третий КАТЕГОРИАЛЬНЫЙ цвет тут не проходит валидатор
    # (оранжевый vs красный ΔE 11.8 при норме 15).
    ax.plot(grid, s0, color=C_MODEL, lw=2.0, ls="--", zorder=5,
            label="Модель при %s = медиана (θ=1.00)" % QL)
    ax.plot(grid, s0 ** th_3x, color=C_FAIL, lw=2.0, ls="--", zorder=5,
            label="Модель при %s = 3×медианы (θ=%.2f)" % (QL, th_3x))
    ax.plot(grid, s0 ** th_5x, color=C_FAIL_DARK, lw=2.0, ls="-.", zorder=5,
            label="Модель при %s = 5×медианы (θ=%.2f)" % (QL, th_5x))

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Наработка, сут")
    ax.set_ylabel("S(t) — доля неотказавших насосов")
    ax.set_title(
        "%s — высокодебитные скважины против базовой кривой\n"
        "KM без слоя и при %s ≥ медианы против модели S(t)^θ, θ=(%s/%s_ref)^ln(1.253); "
        "%s известен у %d из %d пробегов"
        % (field_label, QL, QL, QL, QL, len(have), len(mc)),
        fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=10.5)
    fig.tight_layout()
    p = out / fname
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # RMST — сравнимая цифра вместо разглядывания кривых.
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    gt = np.linspace(0, model.tau, 2000)
    s0t = model.survival(gt)

    def _rmst_km(g):
        x, y, _, _ = KF._km(g["tte"].to_numpy(float), (g["event"] == 1).astype(int).to_numpy())
        return KF._rmst(x, y, model.tau)

    res["rmst_base_km"] = _rmst_km(mc)
    res["rmst_hi_km"] = _rmst_km(hi_g)
    res["rmst_model_1"] = float(tz(s0t, gt))
    res["rmst_model_3x"] = float(tz(s0t ** th_3x, gt))
    res["rmst_model_5x"] = float(tz(s0t ** th_5x, gt))
    res["gap_fact"] = res["rmst_base_km"] - res["rmst_hi_km"]
    res["gap_model_3x"] = res["rmst_model_1"] - res["rmst_model_3x"]
    res["gap_model_5x"] = res["rmst_model_1"] - res["rmst_model_5x"]
    res["ev_early_all"] = int((mc[mc["event"] == 1]["tte"] <= 30).sum())
    res["ev_early_hi"] = int((hi_g[hi_g["event"] == 1]["tte"] <= 30).sum())
    res["tau"] = float(model.tau)
    res["path"] = p
    res["coverage"] = cov.as_frame().to_dict("records")[0]
    return res


def main() -> None:
    pop = P.build(ASOF)
    model = M.fit_mc_model(pop)
    mc = P.select(pop, "Mc")
    mc = mc[mc["tte"] > 0].copy()

    out = results_dir("production_risk_mc_model_slides")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    s1 = slide1(mc, model, figs)
    s2 = slide2(mc, model, figs)

    # Vt_nonsour — поле, НА КОТОРОМ и оценивался β слоя Ql (Ql +0.246 p=0.039,
    # PH со стратификацией по подрядчику). Правило «монтажи 2024+» — только для Mc,
    # поэтому здесь берётся вся история. Модель — та же смесь, подогнанная на Vt.
    vt = P.select(pop, "Vt", h2s="nonsour")
    vt = vt[vt["tte"] > 0].copy()
    vt_model = fit_mixture(vt)
    s3 = slide2(vt, vt_model, figs, fname="slide3_vt_km_by_ql.png",
                field_label="Верхнетирский, некислые (Vt_nonsour)")

    print("МОДЕЛЬ Mc cal c0 k2: w1=%.4f β1=%.3f η1=%.2f β2=%.3f η2=%.1f | τ=%.0f"
          % (model.w1, model.beta1, model.eta1, model.beta2, model.eta2, model.tau))
    print("Слайд 1: %s" % s1["path"].name)
    print("  RMST(0): KM %.0f сут vs модель %.0f сут" % (s1["km_rmst"], s1["model_rmst"]))
    print("  хвост S: KM %.3f vs модель %.3f" % (s1["km_tail"], s1["model_tail"]))
    for tag, s, mdl in (("Слайд 2 (Mc)", s2, model), ("Слайд 3 (Vt_nonsour)", s3, vt_model)):
        print()
        print("%s: %s" % (tag, s["path"].name))
        print("  модель: w1=%.4f β1=%.3f η1=%.2f β2=%.3f η2=%.1f | τ=%.0f | пробегов %d, отказов %d"
              % (mdl.w1, mdl.beta1, mdl.eta1, mdl.beta2, mdl.eta2, mdl.tau,
                 mdl.n_runs, mdl.n_events))
        print("  Ql: медиана %.0f м3/сут | покрытие %d/%d пробегов"
              % (s["median_ql"], s["n_with_ql"], s["n_total"]))
        print("  Ql>=медианы: n=%d, отказов %d | ранних (<=30 сут) отказов: всего %d, в группе %d"
              % (s["n_hi"], s["ev_hi"], s["ev_early_all"], s["ev_early_hi"]))
        if "cox_beta" in s:
            print("  Cox β (на этих данных) = %.3f (95%% ДИ %.3f..%.3f, p=%.4f) | перенесённый β=%.3f"
                  % (s["cox_beta"], s["cox_lo"], s["cox_hi"], s["cox_p"], s["beta_landmark"]))
        print("  θ: 3×медианы %.3f | 5×медианы %.3f (справочно 2× = %.3f)"
              % (s["theta_3x"], s["theta_5x"], s["theta_2x"]))
        print("  RMST до τ=%.0f: база KM %.0f | Ql>=мед KM %.0f || модель θ=1 %.0f | 3× %.0f | 5× %.0f"
              % (s["tau"], s["rmst_base_km"], s["rmst_hi_km"], s["rmst_model_1"],
                 s["rmst_model_3x"], s["rmst_model_5x"]))
        print("  РАЗРЫВ: факт %.0f сут vs модель 3× %.0f сут, 5× %.0f сут"
              % (s["gap_fact"], s["gap_model_3x"], s["gap_model_5x"]))
    print()
    print("figures:", figs)


if __name__ == "__main__":
    main()
