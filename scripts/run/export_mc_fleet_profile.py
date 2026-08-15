"""Профиль парка Мирнинского по месяцам: возраст насосов и дебит жидкости Qж.

Отдельные графики (в презентацию не входят) — чем парк был в каждый месяц и чем станет:
стареет он или молодеет и как менялась загрузка по жидкости.

ИСТОРИЯ (2024-01…2026-06)
  * ВОЗРАСТ — средний и медианный возраст насосов, работающих на начало месяца, по
    скважинам из АКТИВНОГО ДОБЫВАЮЩЕГО парка плана ПП: тот же знаменатель, что в
    интенсивности отказов, иначе цифры несопоставимы с остальной отчётностью.
  * Qж — из `proc__daily_merged` (дни с qliq > 0). Тот же источник, из которого слой Qж
    берёт ковариату.

ПРОГНОЗ (18 мес от даты расчёта)
  * ВОЗРАСТ — прогоняется ТА ЖЕ динамика, что в `esp_forecast.forecast`: масса возраста
    сдвигается на сутки, доля `q(a)` отказывает и возвращается в возраст 0 (обновление),
    ВНС входят в свой месяц с нуля. Считается по агрегированному распределению парка,
    поэтому доступны и среднее, и медиана.
  * Qж — из ПЛАНА ПП (`registry_liquid_rate`), телеметрии на будущее нет.

⚠ Телеметрия и план по Qж РАСХОДЯТСЯ (на 2025-06 среднее 208 против 157), поэтому они
показаны РАЗНЫМИ панелями и не сшиваются в одну линию: план рисуется на всём диапазоне,
чтобы расхождение было видно на общей истории, а не спрятано на стыке.

Запуск:  python scripts/run/export_mc_fleet_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import t0_covariates as T0

from export_mc_model_slides import C_FAIL, C_MODEL, QL, _style

FIRST, LAST = "2024-01", "2026-06"
HORIZON = 18


def _q_stats(vals: pd.Series) -> tuple[float, float, int]:
    v = vals[vals > 0]
    if not len(v):
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.median()), int(len(v))


def build_history(pop: pd.DataFrame, fleet: dict, plan_ql: pd.DataFrame) -> pd.DataFrame:
    mc = pop[pop["field"] == "Mc"].copy()
    codes = set(mc["code"].astype(str))

    dailies = T0.load_dailies(sorted({c.casefold() for c in codes}))
    dailies = dailies[dailies["qliq"].notna() & (dailies["qliq"] > 0)].copy()
    dailies["code"] = dailies["well_key"].map(crosswalk.norm_well)
    dailies["month"] = dailies["dt"].dt.to_period("M").astype(str)
    tel = dailies.groupby(["month", "code"])["qliq"].mean().reset_index()

    rows = []
    for per in pd.period_range(pd.Period(FIRST, "M"), pd.Period(LAST, "M"), freq="M"):
        mo = str(per)
        m0, m1 = per.start_time, per.end_time.normalize()
        active = fleet.get(mo, set())
        ages = []
        for r in mc.itertuples():
            if r.install > m1:
                continue
            end = r.end if pd.notna(r.end) else None
            if end is not None and end < m0:
                continue
            if str(r.code) in active:
                ages.append(max(0.0, float((m0 - r.install).days)))
        t_mean, t_med, t_n = _q_stats(tel[(tel["month"] == mo)
                                          & (tel["code"].isin(active))]["qliq"])
        p_mean, p_med, p_n = (_q_stats(plan_ql.loc[sorted(active & set(plan_ql.index)), mo])
                              if mo in plan_ql.columns else (np.nan, np.nan, 0))
        rows.append({
            "месяц": mo, "период": "история", "в работе": len(active),
            "возраст средний, сут": float(np.mean(ages)) if ages else np.nan,
            "возраст медиана, сут": float(np.median(ages)) if ages else np.nan,
            "Qж телеметрия среднее": t_mean, "Qж телеметрия медиана": t_med,
            "скважин с Qж (телеметрия)": t_n,
            "Qж план среднее": p_mean, "Qж план медиана": p_med,
        })
    return pd.DataFrame(rows)


def forecast_age(model, live: pd.DataFrame, entrants: pd.DataFrame,
                 asof: pd.Timestamp, months: int) -> pd.DataFrame:
    """Средний/медианный возраст парка вперёд — та же динамика, что в `forecast`."""
    sim = M._sim_frame(live, entrants)
    horizon = int(months * 31 + 5)
    max_age = int(sim["age"].max()) + horizon + 2
    q = model.daily_fail_prob(max_age)
    ages_axis = np.arange(len(q), dtype=float)
    per_list = M._months(asof, asof + pd.DateOffset(months=months - 1))
    day_month = {i: pd.Period(asof + pd.Timedelta(days=i), "M") for i in range(horizon)}

    agg: dict[pd.Period, np.ndarray] = {}
    for r in sim.itertuples():
        u = np.zeros(len(q), dtype=float)
        sd = int(r.start_day)
        if sd <= 0:
            u[int(r.age)] = 1.0
        for i in range(horizon):
            if sd > 0 and i == sd:
                u[0] = 1.0
            per = day_month[i]
            if i == 0 or day_month[i - 1] != per:          # снимок на 1-е число
                agg[per] = agg.get(per, np.zeros(len(q))) + u
            fails = u * q
            n_fail = float(fails.sum())
            surv = u - fails
            nxt = np.zeros_like(u)
            nxt[1:] = surv[:-1]
            nxt[0] = n_fail                                # обновление: новый с нуля
            u = nxt

    rows = []
    for per in per_list:
        w = agg.get(per)
        if w is None or w.sum() <= 0:
            rows.append({"месяц": str(per), "возраст средний, сут": np.nan,
                         "возраст медиана, сут": np.nan})
            continue
        tot = w.sum()
        mean = float((ages_axis * w).sum() / tot)
        med = float(ages_axis[np.searchsorted(np.cumsum(w), tot * 0.5)])
        rows.append({"месяц": str(per), "возраст средний, сут": mean,
                     "возраст медиана, сут": med})
    return pd.DataFrame(rows)


def build_forecast(pop: pd.DataFrame, model, asof: pd.Timestamp,
                   plan_ql: pd.DataFrame) -> pd.DataFrame:
    codes = set(pop[pop["field"] == "Mc"]["code"])
    live = M.live_fleet(pop, asof, field="Mc")
    ent = M.plan_entrants(set(live["code"]), codes, asof, HORIZON)
    fleet = M.active_fleet_by_month(codes=codes | set(ent["code"]))

    df = forecast_age(model, live, ent, asof, HORIZON)
    out = []
    for _, r in df.iterrows():
        mo = r["месяц"]
        active = fleet.get(mo, set())
        p_mean, p_med, _ = (_q_stats(plan_ql.loc[sorted(active & set(plan_ql.index)), mo])
                            if mo in plan_ql.columns else (np.nan, np.nan, 0))
        out.append({
            "месяц": mo, "период": "прогноз", "в работе": len(active),
            "возраст средний, сут": r["возраст средний, сут"],
            "возраст медиана, сут": r["возраст медиана, сут"],
            "Qж телеметрия среднее": np.nan, "Qж телеметрия медиана": np.nan,
            "скважин с Qж (телеметрия)": 0,
            "Qж план среднее": p_mean, "Qж план медиана": p_med,
        })
    return pd.DataFrame(out)


def build_figure(df: pd.DataFrame, split: int, out: Path) -> Path:
    plt = _style()
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    hist = x < split

    def _split_line(ax, col, color, label, ls="-"):
        """История сплошной, прогноз того же цвета — тем же стилем, но полупрозрачно."""
        y = df[col].to_numpy(float)
        ax.plot(x[hist], y[hist], color=color, lw=2.4, ls=ls, zorder=4, label=label)
        ax.plot(x[~hist], y[~hist], color=color, lw=2.4, ls=ls, zorder=4, alpha=0.45)

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", zorder=0)
        ax.axvline(split - 0.5, color="0.45", ls=":", lw=1.4, zorder=1)

    _split_line(ax1, "возраст средний, сут", C_MODEL, "среднее")
    _split_line(ax1, "возраст медиана, сут", C_MODEL, "медиана", ls="--")
    ax1.set_ylabel("Возраст насоса, сут")
    ax1.set_title("Возраст работающих насосов — Мирнинский УН "
                  "(слева факт, справа прогноз с обновлением)", fontsize=12, loc="left")
    ax1.legend(frameon=False, fontsize=10, loc="upper left")

    # Только план ПП: он один покрывает и историю, и прогноз. Телеметрия убрана с
    # графика по решению пользователя — её колонки остаются в CSV.
    _split_line(ax2, "Qж план среднее", C_FAIL, "среднее")
    _split_line(ax2, "Qж план медиана", C_FAIL, "медиана", ls="--")
    ax2.set_ylabel("%s, м³/сут" % QL)
    ax2.set_xlabel("Месяц")
    ax2.set_title("%s по плану ПП — история и прогноз" % QL, fontsize=12, loc="left")
    ax2.legend(frameon=False, fontsize=10, loc="upper left")

    ax2.set_xticks(x[::3])
    ax2.set_xticklabels(df["месяц"][::3], rotation=45, ha="right")
    fig.tight_layout()
    p = out / "fleet_age_and_ql.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    asof = pd.Timestamp(str(C.RunConfig().forecast_start))
    pop = P.build(str(C.RunConfig().forecast_start))
    model = M.fit_mc_model(pop)

    cfg = C.RunConfig()
    plan = crosswalk.load_plan(cfg.forecast_start, cfg.horizon_end,
                               master_path=cfg.pp_master_path)
    plan_ql = plan.registry_liquid_rate
    plan_ql.index = plan_ql.index.astype(str)

    codes = set(pop[pop["field"] == "Mc"]["code"])
    fleet_hist = M.active_fleet_by_month(codes=codes)

    hist = build_history(pop, fleet_hist, plan_ql)
    fut = build_forecast(pop, model, asof, plan_ql)
    df = pd.concat([hist, fut], ignore_index=True)

    out = results_dir("production_risk_mc_fleet_profile")
    figs, tabs = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    p = build_figure(df, len(hist), figs)
    csv = tabs / "fleet_age_and_ql.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    print("фигура:", p)
    print("таблица:", csv)
    print()
    show = df[["месяц", "период", "в работе", "возраст средний, сут",
               "возраст медиана, сут", "Qж телеметрия среднее", "Qж план среднее"]]
    print(show.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
