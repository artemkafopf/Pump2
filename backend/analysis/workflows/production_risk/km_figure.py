"""График Kaplan-Meier по месторождению/страте: МРП (все подъёмы) против ННО (только отказы).

Две кривые строятся на ОДНИХ И ТЕХ ЖЕ пробегах — различается только определение
события.  Это и есть картинка «два эстиманда», из-за смешения которых в проекте
уже ломались выводы:

* синяя «Все подъёмы (МРП)» — all-cause: ГТМ = событие; цензура = только работающие;
* красная «Только отказы (ННО)» — cause-specific: ГТМ И работающие цензурированы.

RMST = площадь под кривой до ``tau`` (p95 наблюдённых наработок).  Дальше tau кривая
не интегрируется: хвост KM держится на единицах насосов и площадь там выдумана.
Это честная замена ``Sum t / N``, которая завышена всегда, когда есть работающие насосы.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import esp_population as P

TAU_PCT = 95.0
COLOR_ALL = "#1f5f9e"      # МРП — все подъёмы
COLOR_FAIL = "#c1272d"     # ННО — только отказы


def _km(t: np.ndarray, e: np.ndarray):
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(t, e)
    sf = km.survival_function_
    ci = km.confidence_interval_
    return (sf.index.to_numpy(float), sf.iloc[:, 0].to_numpy(float),
            ci.iloc[:, 0].to_numpy(float), ci.iloc[:, 1].to_numpy(float))


def _rmst(x: np.ndarray, y: np.ndarray, tau: float) -> float:
    m = x <= tau
    if m.sum() < 2:
        return float("nan")
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y[m], x[m]))


def build_data(
    as_of: date | pd.Timestamp,
    field: str,
    h2s: str = "nonsour",
    contractor: str | list[str] | None = None,
) -> dict:
    """Собрать обе кривые на одном наборе пробегов."""
    def _sel(pop):
        g = P.select(pop, field, h2s=h2s, contractor=None)
        if contractor is not None:
            groups = [contractor] if isinstance(contractor, str) else list(contractor)
            g = g[g["contractor_group"].isin(groups)]
        return g[g["tte"] > 0]

    all_cause = _sel(P.build(as_of, gtm_is_failure=True))
    cause_spec = _sel(P.build(as_of, gtm_is_failure=False))
    if len(all_cause) != len(cause_spec):
        raise AssertionError("две сборки популяции разошлись по числу пробегов — кривые несравнимы")

    t = all_cause["tte"].to_numpy(float)
    tau = float(np.percentile(t, TAU_PCT))
    xa, ya, la, ua = _km(t, all_cause["event"].to_numpy(int))
    xf, yf, lf, uf = _km(cause_spec["tte"].to_numpy(float), cause_spec["event"].to_numpy(int))

    return {
        "n": len(t),
        "pulls": int(all_cause["event"].sum()),
        "failures": int(cause_spec["event"].sum()),
        "running": int(all_cause["end"].isna().sum()),
        "tau": tau,
        "all": (xa, ya, la, ua),
        "fail": (xf, yf, lf, uf),
        "rmst_all": _rmst(xa, ya, tau),
        "rmst_fail": _rmst(xf, yf, tau),
        "naive_mrp": t.sum() / max(int(all_cause["event"].sum()), 1),
        "naive_nno": t.sum() / max(int(cause_spec["event"].sum()), 1),
    }


def build_figure(data: dict, title: str, subtitle_extra: str = ""):
    """Отрисовать график в стиле эталонного графика Мирнинского."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 9))
    tau = data["tau"]
    xa, ya, la, ua = data["all"]
    xf, yf, lf, uf = data["fail"]

    # Заливка = ровно та площадь, которую называет подпись: только до tau.
    # За tau кривая держится на единицах насосов — закрашивать там нечестно.
    ma, mf = xa <= tau, xf <= tau
    ax.fill_between(xa[ma], 0, ya[ma], step="post", color=COLOR_ALL, alpha=0.07)
    ax.fill_between(xa, la, ua, step="post", color=COLOR_ALL, alpha=0.18, linewidth=0)
    ax.fill_between(xf, lf, uf, step="post", color=COLOR_FAIL, alpha=0.18, linewidth=0)
    ax.step(xa, ya, where="post", color=COLOR_ALL, lw=2.4, label="Все подъёмы (МРП)")
    ax.step(xf, yf, where="post", color=COLOR_FAIL, lw=2.4, label="Только отказы (ННО)")

    xmax = float(min(np.max(xa), tau * 1.6))
    ax.axvline(tau, color="0.45", ls="--", lw=1.4)
    ax.text(tau + xmax * 0.008, 0.985, f"τ = {tau:.0f} сут\n(95-й перцентиль)\nправее площадь не считаем",
            color="0.35", fontsize=10.5, va="top")

    ax.annotate(f"МРП = RMST = площадь под синей кривой = {data['rmst_all']:.0f} сут",
                xy=(tau * 0.20, float(np.interp(tau * 0.20, xa, ya))), xytext=(tau * 0.30, 0.80),
                color=COLOR_ALL, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=COLOR_ALL, lw=1.2))
    ax.annotate(f"ННО = RMST по отказам = {data['rmst_fail']:.0f} сут",
                xy=(tau * 0.62, float(np.interp(tau * 0.62, xf, yf))), xytext=(tau * 0.66, 0.56),
                color=COLOR_FAIL, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=COLOR_FAIL, lw=1.2))

    sub = (f"{data['n']} пробегов: {data['pulls']} подъёмов ({data['failures']} отказов), "
           f"{data['running']} в работе (цензурировано)")
    if subtitle_extra:
        sub += f"\n{subtitle_extra}"
    ax.set_title(f"{title}\n{sub}", fontsize=15)
    ax.set_xlabel("Наработка, сут", fontsize=12)
    ax.set_ylabel("S(t) — доля неподнятых насосов", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, xmax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)
    fig.tight_layout()
    return fig
