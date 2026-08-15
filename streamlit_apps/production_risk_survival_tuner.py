"""Production-risk survival tuner.

Interactive check on the shipped Weibull registry: edit a stratum's survival
parameters and immediately see

  1. the fitted S(t) against the Kaplan-Meier of the censoring-corrected run
     population it was fit on, and
  2. the monthly failure rate — replayed history AND forecast — against the
     observed fact for one reporting УН.

Nothing here writes to the bundle; use the "Экспорт" panel to copy a candidate
row out once a parameter set looks right.

Run:  streamlit run streamlit_apps/production_risk_survival_tuner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lifelines import KaplanMeierFitter

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk, failure_rate as FR, layers
from analysis.workflows.production_risk.survival import HazardLayer, StrataModel

st.set_page_config(page_title="Production risk — survival tuner", layout="wide")

REFIT_ROOT = REPO_ROOT / "results" / "esp_survival_big_censored_refit"
PARAM_KEYS = ("w1", "beta1", "eta1", "beta2", "eta2", "uptime_factor")

KM_COLOR = "#1A2744"
FIT_COLOR = "#FF8C00"
REF_COLORS = ["#1F5C99", "#2ca02c", "#9467bd", "#8c564b"]
OBS_COLOR = "#1A2744"
PRED_COLOR = "#FF8C00"


# ── survival math ─────────────────────────────────────────────────────────────
def survival_curve(t: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """S(t) for the shipped 2-component latent Weibull mixture."""
    return StrataModel.S(t, params)


def life_quantile(params: dict[str, float], q: float, t_max: float = 20000.0) -> float:
    """Numeric inverse of S: the age where S(t) == 1 - q."""
    grid = np.arange(0.0, t_max, 1.0)
    s = survival_curve(grid, params)
    target = 1.0 - q
    below = np.nonzero(s <= target)[0]
    return float(grid[below[0]]) if below.size else float("nan")


@st.cache_data(show_spinner=False)
def load_population(refit_date: str) -> pd.DataFrame:
    path = REFIT_ROOT / refit_date / "fit_population.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")
    df["tte"] = pd.to_numeric(df["tte"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["tte"])


@st.cache_data(show_spinner=False)
def load_registry(bundle_date: str) -> pd.DataFrame:
    return pd.read_csv(C.model_registry_path(bundle_date), encoding="utf-8-sig")


def registry_params(bundle_date: str, stratum: str) -> dict[str, float] | None:
    df = load_registry(bundle_date)
    hit = df[df["stratum"] == stratum]
    if hit.empty:
        return None
    row = hit.iloc[0]
    out = {}
    for key in PARAM_KEYS:
        value = row.get(key, np.nan)
        out[key] = float(value) if pd.notna(value) else (1.0 if key == "uptime_factor" else 0.0)
    return out


# ── heavy pipeline pieces (cached; the fleet load is ~30s once) ───────────────
@st.cache_resource(show_spinner="Загрузка плана ПП, Свод и ТР ...")
def load_context():
    cfg = C.RunConfig()
    plan = crosswalk.load_plan(cfg.forecast_start, cfg.horizon_end, cfg.pp_master_path)
    esp = crosswalk.load_esp_source(cfg.bundle_date, cfg.prediction_workbook_path)
    gtm = crosswalk.load_gtm(cfg.gtm_schedule_path)
    tr = crosswalk.load_current_techregime_status(cfg.techregime_workbook_path)
    downtime = crosswalk.derive_downtime_quantiles(esp)
    well_field = FR.build_well_field(plan)
    return cfg, plan, esp, gtm, tr, downtime, well_field


def build_model(bundle_date: str, stratum: str, params: dict[str, float]) -> StrataModel:
    model = StrataModel(bundle_date=bundle_date)
    row = dict(model.by_stratum.get(stratum, {}))
    row.update(params)
    model.by_stratum[stratum] = row
    return model


@st.cache_data(show_spinner="Пересчёт интенсивности отказов ...")
def failure_rate_for(un: str, bundle_date: str, stratum: str, params_key: tuple, hazard_mode: str) -> pd.DataFrame:
    """Recompute one УН's monthly fact-vs-model line with the edited parameters."""
    cfg, plan, esp, gtm, tr, (gd, fd), well_field = load_context()
    params = dict(zip(PARAM_KEYS, params_key))
    model = build_model(bundle_date, stratum, params)
    hazard = HazardLayer(bundle_date=bundle_date)

    states, _ = layers.build_well_states(plan, esp, gtm, model, cfg, current_status=tr)
    scoped = [s for s in states if well_field.get(s.code) == un]
    if not scoped:
        return pd.DataFrame()

    scenario_id = C.PRIMARY_SCENARIO_ID if hazard_mode == "baseline" else C.STRESS_SCENARIO_ID
    specs = tuple(s for s in cfg.scenarios if s.scenario_id == scenario_id)
    projection = layers.run_projection(
        scoped, specs, model, hazard, gd, fd, plan.fwd_months,
        cfg.changeout_p90, downtime_override_days=cfg.downtime_override_days,
    )
    result = FR.compute(
        plan, esp, projection, cfg, scenario_id=scenario_id,
        model=model, hazard=hazard, only_fields={un},
    )
    out = result.monthly[result.monthly["field"] == un].copy()
    out.attrs["forecast_first_month"] = result.forecast_first_month
    return out


# ── sidebar ───────────────────────────────────────────────────────────────────
st.title("Настройка параметров выживаемости УЭЦН")

refit_dates = sorted((p.name for p in REFIT_ROOT.iterdir() if (p / "fit_population.csv").exists()), reverse=True)
bundle_dates = sorted((p.name for p in C.BUNDLE_ROOT.iterdir() if (p / "esp_models.csv").exists()), reverse=True)

with st.sidebar:
    st.header("Данные")
    refit_date = st.selectbox("Популяция для KM", refit_dates, index=0)
    bundle_date = st.selectbox("Бандл (базовые параметры)", bundle_dates,
                               index=bundle_dates.index(C.BUNDLE_DATE) if C.BUNDLE_DATE in bundle_dates else 0)

    population = load_population(refit_date)
    registry = load_registry(bundle_date)

    strata = sorted(registry["stratum"].dropna().unique())
    default_stratum = "Mc_nonsour_Pooled" if "Mc_nonsour_Pooled" in strata else strata[0]
    stratum = st.selectbox("Страта", strata, index=strata.index(default_stratum))

    st.header("Параметры Вейбулла")
    base = registry_params(bundle_date, stratum) or dict.fromkeys(PARAM_KEYS, 0.0)
    if st.button("Сбросить к бандлу", use_container_width=True):
        for key in PARAM_KEYS:
            st.session_state[f"p_{key}"] = base[key]

    for key in PARAM_KEYS:
        st.session_state.setdefault(f"p_{key}", base[key])

    w1 = st.slider("w1 (вес компоненты 1)", 0.0, 1.0, key="p_w1", step=0.001, format="%.3f")
    c1, c2 = st.columns(2)
    with c1:
        beta1 = st.number_input("beta1", 0.05, 60.0, key="p_beta1", step=0.01, format="%.4f")
        beta2 = st.number_input("beta2", 0.05, 60.0, key="p_beta2", step=0.01, format="%.4f")
    with c2:
        eta1 = st.number_input("eta1, сут", 1.0, 20000.0, key="p_eta1", step=1.0, format="%.2f")
        eta2 = st.number_input("eta2, сут", 1.0, 20000.0, key="p_eta2", step=1.0, format="%.2f")
    uptime_factor = st.number_input("uptime_factor", 0.1, 1.0, key="p_uptime_factor", step=0.005, format="%.4f")

    st.caption("w1=0 сводит смесь к одной компоненте (k1). Часы модели — сутки наработки.")

params = {key: float(st.session_state[f"p_{key}"]) for key in PARAM_KEYS}
params_key = tuple(params[key] for key in PARAM_KEYS)
dirty = any(abs(params[k] - base[k]) > 1e-9 for k in PARAM_KEYS)

tab_km, tab_rate, tab_export = st.tabs(
    ["1 · Выживаемость (KM / Weibull)", "2 · Интенсивность отказов (факт vs модель)", "3 · Экспорт"]
)


# ── tab 1: KM vs Weibull ──────────────────────────────────────────────────────
with tab_km:
    parts = stratum.split("_")
    field, h2s = parts[0], parts[1]
    contractor = parts[2] if len(parts) > 2 else "Pooled"

    left, right = st.columns([1, 3])
    with left:
        window = st.radio("Окно установки", ["Все", "2023+", "2024+"], index=0, horizontal=False)
        by_contractor = st.checkbox(
            "Только подрядчик страты", value=False,
            disabled=contractor == "Pooled",
            help="Pooled-страты фитятся по всем подрядчикам сразу.",
        )

    pop = population[(population["field"] == field) & (population["h2s_class"] == h2s)].copy()
    if window == "2023+":
        pop = pop[pop["install_date"] >= pd.Timestamp("2023-01-01")]
    elif window == "2024+":
        pop = pop[pop["install_date"] >= pd.Timestamp("2024-01-01")]
    if by_contractor and contractor != "Pooled":
        pop = pop[pop["contractor_group"] == contractor]

    if pop.empty or pop["event"].sum() == 0:
        st.warning(f"Нет наблюдений с отказами для {stratum} в выбранном окне.")
    else:
        kmf = KaplanMeierFitter().fit(pop["tte"], pop["event"])
        km_median = float(kmf.median_survival_time_)
        ci = kmf.confidence_interval_
        b50 = life_quantile(params, 0.5)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Пробегов", f"{len(pop)}")
        m2.metric("Отказов / цензур", f"{int(pop['event'].sum())} / {int((pop['event'] == 0).sum())}")
        m3.metric("Медиана KM, сут", f"{km_median:.0f}")
        m4.metric("B50 модели, сут", f"{b50:.0f}", delta=f"{b50 - km_median:+.0f} к KM")
        m5.metric("B20 / B80, сут", f"{life_quantile(params, 0.2):.0f} / {life_quantile(params, 0.8):.0f}")

        t_max = float(np.nanpercentile(pop["tte"], 99)) * 1.3
        grid = np.arange(0.0, max(t_max, 400.0), 1.0)

        fig = go.Figure()
        km_t = kmf.survival_function_.index.to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=np.concatenate([km_t, km_t[::-1]]),
            y=np.concatenate([ci.iloc[:, 1].to_numpy(), ci.iloc[:, 0].to_numpy()[::-1]]),
            fill="toself", fillcolor="rgba(26,39,68,0.12)", line=dict(width=0),
            hoverinfo="skip", name="KM 95% ДИ",
        ))
        fig.add_trace(go.Scatter(
            x=km_t, y=kmf.survival_function_.iloc[:, 0].to_numpy(),
            mode="lines", line=dict(color=KM_COLOR, width=2.5, shape="hv"),
            name=f"Kaplan-Meier (n={len(pop)})",
        ))
        fig.add_trace(go.Scatter(
            x=grid, y=survival_curve(grid, params), mode="lines",
            line=dict(color=FIT_COLOR, width=3),
            name="Текущие параметры" + (" (изменены)" if dirty else ""),
        ))
        for idx, ref_bundle in enumerate(b for b in bundle_dates if b != bundle_date):
            ref = registry_params(ref_bundle, stratum)
            if ref is None:
                continue
            fig.add_trace(go.Scatter(
                x=grid, y=survival_curve(grid, ref), mode="lines",
                line=dict(color=REF_COLORS[idx % len(REF_COLORS)], width=1.5, dash="dot"),
                name=f"бандл {ref_bundle}", visible="legendonly",
            ))
        fig.add_hline(y=0.5, line=dict(color="#888", width=1, dash="dash"))
        fig.update_layout(
            height=520, hovermode="x unified",
            xaxis_title="Наработка, сут", yaxis_title="S(t)",
            yaxis=dict(range=[0, 1.02]), legend=dict(orientation="h", y=-0.18),
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        horizon = float(kmf.percentile(0.2)) if np.isfinite(kmf.percentile(0.2)) else t_max
        mask = grid <= horizon
        km_on_grid = np.interp(grid[mask], km_t, kmf.survival_function_.iloc[:, 0].to_numpy())
        max_ds = float(np.max(np.abs(survival_curve(grid[mask], params) - km_on_grid))) if mask.any() else float("nan")
        verdict = "принимается" if max_ds < 0.05 else "НЕ принимается"
        st.caption(
            f"max|ΔS| на надёжном горизонте (0–{horizon:.0f} сут): **{max_ds:.4f}** → {verdict} "
            f"(порог приёмки рефита 0.05)."
        )

        with st.expander("Причины подъёма (событие vs цензура)"):
            st.dataframe(
                pop.groupby(["pull_reason", "event"]).size().rename("n").reset_index()
                   .sort_values("n", ascending=False),
                use_container_width=True, hide_index=True,
            )


# ── tab 2: failure rate vs fact ───────────────────────────────────────────────
with tab_rate:
    cfg, plan, esp, gtm, tr, downtime, well_field = load_context()
    uns = sorted({v for v in well_field.values() if v and v != "Без УН"})
    default_un = "Мирнинский УН" if "Мирнинский УН" in uns else uns[0]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        un = st.selectbox("Участок недр", uns, index=uns.index(default_un))
    with c2:
        hazard_mode = st.radio("Сценарий", ["baseline", "stress"], index=0, horizontal=True)
    with c3:
        st.write("")
        run = st.button("Пересчитать", type="primary", use_container_width=True)

    n_wells = sum(1 for w, f in well_field.items() if f == un)
    st.caption(
        f"{un}: {n_wells} скважин в плане. Пересчёт ~10–20 с; результат кэшируется "
        f"по набору параметров."
    )

    if run or st.session_state.get("rate_done"):
        st.session_state["rate_done"] = True
        monthly = failure_rate_for(un, bundle_date, stratum, params_key, hazard_mode)
        if monthly.empty:
            st.warning(f"Для {un} не построено ни одного состояния скважины.")
        else:
            monthly = monthly.sort_values("month")
            forecast_first = monthly.attrs.get("forecast_first_month", cfg.forecast_start.strftime("%Y-%m"))
            view = monthly[monthly["month"] >= "2024-01"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=view["month"], y=view["observed_rate"], mode="lines+markers",
                line=dict(color=OBS_COLOR, width=2.5), marker=dict(size=6),
                name="Факт (набл. отказы / фонд)",
            ))
            fig.add_trace(go.Scatter(
                x=view["month"], y=view["predicted_rate"], mode="lines",
                line=dict(color=PRED_COLOR, width=2.5), name="Модель",
            ))
            fig.add_vline(x=forecast_first, line=dict(color="#888", width=1, dash="dash"))
            fig.add_annotation(x=forecast_first, y=1.02, yref="paper", showarrow=False,
                               text="прогноз →", font=dict(size=11, color="#888"))
            fig.update_layout(
                height=430, hovermode="x unified",
                xaxis_title="Месяц", yaxis_title="Отказов на скважину в месяц",
                legend=dict(orientation="h", y=-0.22), margin=dict(t=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            hist = view[view["month"] < forecast_first]
            years = hist.assign(year=hist["month"].str[:4]).groupby("year").agg(
                факт=("observed_failures", "sum"),
                модель=("predicted_failures", "sum"),
                фонд_ср=("fleet_size", "mean"),
            ).reset_index()
            years["модель/факт"] = (years["модель"] / years["факт"].replace(0, np.nan)).round(2)
            years["модель"] = years["модель"].round(1)
            years["фонд_ср"] = years["фонд_ср"].round(1)
            st.subheader("История: факт vs модель по годам")
            st.dataframe(years, use_container_width=True, hide_index=True)
            st.caption(
                "«модель/факт» — главный критерий калибровки: 1.0 = попадание, "
                ">1 = модель переоценивает число отказов, <1 = недооценивает."
            )

            with st.expander("Помесячная таблица"):
                st.dataframe(
                    view[["month", "fleet_size", "observed_failures", "predicted_failures",
                          "observed_rate", "predicted_rate"]].round(4),
                    use_container_width=True, hide_index=True,
                )
    else:
        st.info("Задайте параметры слева и нажмите «Пересчитать».")


# ── tab 3: export ─────────────────────────────────────────────────────────────
with tab_export:
    st.subheader("Строка-кандидат для esp_models.csv")
    st.caption(
        "Приложение НЕ пишет в бандл. Скопируйте строку и внесите её в "
        "candidate-бандл отдельным шагом, чтобы изменение прошло ревью."
    )
    row = {"stratum": stratum, **{k: round(params[k], 6) for k in PARAM_KEYS}}
    for q, name in ((0.2, "b20"), (0.5, "b50"), (0.8, "b80")):
        row[name] = round(life_quantile(params, q), 1)
    st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)
    st.code(pd.DataFrame([row]).to_csv(index=False), language="text")

    st.subheader("Текущий бандл")
    st.dataframe(load_registry(bundle_date), use_container_width=True, hide_index=True)
