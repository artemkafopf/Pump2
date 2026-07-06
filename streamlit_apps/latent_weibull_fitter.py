"""
Latent Weibull Mixture Fitter
Fits K-component Weibull mixture to the empirical KM curve by minimising
sum-of-squared deviations: Σ (S_mix(t_i) - KM(t_i))².
After fitting, all parameters are editable via sliders.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

REPO = Path(__file__).resolve().parents[1]
DB   = REPO / "data" / "warehouse" / "pump2.db"

st.set_page_config(
    page_title="Latent Weibull Fitter",
    page_icon=":material/auto_graph:",
    layout="wide",
)

COMP_COLORS = ["#1F5C99", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
MIX_COLOR   = "#FF8C00"
KM_COLOR    = "#1A2744"


# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB))
    df = pd.read_sql(
        "SELECT run_days, event, ttf_true_best_days, field FROM mart__vt_freq55",
        conn,
    )
    conn.close()
    df["ttf"] = df["ttf_true_best_days"].fillna(df["run_days"])
    return df


def kaplan_meier(t_arr: np.ndarray, e_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_arr, float)
    e = np.asarray(e_arr, float)
    ok = (t > 0) & np.isfinite(t)
    t, e = t[ok], e[ok]
    order = np.argsort(t)
    t, e = t[order], e[order]
    times, surv = [0.0], [1.0]
    S = 1.0
    for i in range(len(t)):
        n_risk = len(t) - i
        if e[i] == 1:
            S *= 1.0 - 1.0 / n_risk
        if i == len(t) - 1 or t[i] != t[i + 1]:
            times.append(float(t[i]))
            surv.append(S)
    return np.array(times), np.array(surv)


# ── Weibull helpers ───────────────────────────────────────────────────────────
def wb_sf(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    return np.exp(-((t / eta) ** beta))


def _decode(params: np.ndarray, k: int) -> list[dict]:
    """Decode raw optimiser params → list of {beta, eta, weight}."""
    betas, etas, raw_w = [], [], []
    idx = 0
    for j in range(k):
        betas.append(float(np.exp(params[idx])))
        etas.append(float(np.exp(params[idx + 1])))
        idx += 2
        if j < k - 1:
            raw_w.append(float(params[idx]))
            idx += 1
    raw_w_arr = np.array(raw_w)
    denom = 1.0 + float(np.sum(np.exp(np.clip(raw_w_arr, -20, 20))))
    weights = [float(np.exp(np.clip(r, -20, 20)) / denom) for r in raw_w_arr]
    weights.append(1.0 / denom)
    out = [{"beta": b, "eta": n, "weight": w} for b, n, w in zip(betas, etas, weights)]
    out.sort(key=lambda c: c["eta"])
    return out


def mixture_from_components(t: np.ndarray, components: list[dict]) -> np.ndarray:
    s = np.zeros_like(t, dtype=float)
    for comp in components:
        s += comp["weight"] * wb_sf(t, comp["beta"], comp["eta"])
    return s


def fit_km(km_t: np.ndarray, km_s: np.ndarray, k: int,
           n_restarts: int = 30) -> tuple[list[dict], float]:
    mask   = km_t > 0
    t_fit  = km_t[mask]
    s_fit  = km_s[mask]
    if len(t_fit) > 300:
        idx    = np.round(np.linspace(0, len(t_fit) - 1, 300)).astype(int)
        t_fit  = t_fit[idx]
        s_fit  = s_fit[idx]

    n_params = 3 * k - 1

    def objective(p):
        betas, etas, raw_w = [], [], []
        idx = 0
        for j in range(k):
            betas.append(np.exp(p[idx]));  etas.append(np.exp(p[idx + 1]));  idx += 2
            if j < k - 1:
                raw_w.append(p[idx]);  idx += 1
        raw_w_arr = np.array(raw_w)
        denom  = 1.0 + np.sum(np.exp(np.clip(raw_w_arr, -20, 20)))
        weights = list(np.exp(np.clip(raw_w_arr, -20, 20)) / denom) + [1.0 / denom]
        s_mix  = sum(w * wb_sf(t_fit, b, n) for w, b, n in zip(weights, betas, etas))
        return float(np.sum((s_mix - s_fit) ** 2))

    p50      = float(np.percentile(t_fit, 50))
    rng      = np.random.default_rng(0)
    best_val, best_comp = np.inf, None

    for trial in range(n_restarts):
        x0 = []
        eta_seeds = np.sort(rng.uniform(p50 * 0.2, p50 * 4.0, k))
        for j in range(k):
            beta0 = rng.uniform(0.35, 2.0)
            eta0  = float(eta_seeds[j]) * rng.uniform(0.7, 1.3)
            x0.extend([np.log(beta0), np.log(max(eta0, 1.0))])
            if j < k - 1:
                x0.append(rng.uniform(-1.5, 1.5))

        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8,
                                "maxiter": 150_000, "maxfev": 400_000})
        if res.fun < best_val:
            best_val  = float(res.fun)
            best_comp = _decode(res.x, k)

    return best_comp, best_val


def _ssr(km_t, km_s, components):
    mask  = km_t > 0
    s_mix = mixture_from_components(km_t[mask], components)
    return float(np.sum((km_s[mask] - s_mix) ** 2))


def _build_figure(km_t, km_s, components, max_t, field_label):
    t_fine = np.linspace(0.5, min(float(max_t), float(km_t.max())), 1000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=km_t, y=km_s, mode="lines", line_shape="hv",
        line=dict(color=KM_COLOR, width=2.5),
        name=f"KM эмпирическая",
    ))
    k = len(components)
    for j, comp in enumerate(components):
        fig.add_trace(go.Scatter(
            x=t_fine, y=wb_sf(t_fine, comp["beta"], comp["eta"]),
            mode="lines",
            line=dict(color=COMP_COLORS[j % len(COMP_COLORS)], width=1.5, dash="dot"),
            name=f"K{j+1}  w={comp['weight']:.2f}  β={comp['beta']:.3f}  η={comp['eta']:.0f}",
        ))
    mix_s = mixture_from_components(t_fine, components)
    ssr   = _ssr(km_t, km_s, components)
    fig.add_trace(go.Scatter(
        x=t_fine, y=mix_s, mode="lines",
        line=dict(color=MIX_COLOR, width=3),
        name=f"Смесь (K={k})  SSR={ssr:.4f}",
    ))
    fig.update_layout(
        title=f"KM vs Латентная смесь Вейбулла (K={k}) — {field_label}",
        xaxis_title="ННО (дней)", yaxis_title="Вероятность выживания S(t)",
        height=500, yaxis=dict(range=[0, 1.02]), xaxis=dict(range=[0, max_t]),
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


def _build_residual_figure(km_t, km_s, components):
    mask      = km_t > 0
    mix_at_km = mixture_from_components(km_t[mask], components)
    residuals = km_s[mask] - mix_at_km
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_t[mask], y=residuals, mode="lines",
                             line=dict(color="#555", width=1.2), name="KM − смесь"))
    fig.add_hline(y=0, line=dict(color="red", dash="dash", width=0.8))
    fig.update_layout(title="Невязка: KM(t) − S_mix(t)",
                      xaxis_title="Дней", yaxis_title="Остаток",
                      height=220, margin=dict(t=40, b=30))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("Латентная смесь Вейбулла — подбор по KM-кривой")
st.caption(
    "Минимизируется сумма квадратов отклонений от эмпирической KM-кривой: "
    "Σ (S_mix(tᵢ) − KM(tᵢ))².  После подбора параметры можно корректировать вручную."
)

df_all = load_data()
fields = sorted(df_all["field"].dropna().unique().tolist())

# ── sidebar: global controls ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Настройки")
    sel_field = st.selectbox("Месторождение / Фонд", ["Global (все)"] + fields)
    k = st.slider("Число компонент (K)", min_value=1, max_value=5, value=2, step=1)
    n_restarts = st.slider("Перезапусков оптимизатора", min_value=5, max_value=100, value=30, step=5)
    max_t_days = st.number_input("Горизонт графика (дней)", min_value=100, value=900, step=100)
    st.divider()
    st.markdown(f"**Всего прогонов:** {len(df_all):,}")

# ── filter data ───────────────────────────────────────────────────────────────
sub   = df_all if sel_field == "Global (все)" else df_all[df_all["field"] == sel_field]
t_arr = sub["ttf"].dropna().values
e_arr = sub.loc[sub["ttf"].notna(), "event"].values
km_t, km_s = kaplan_meier(t_arr, e_arr)

# ── session-state keys ────────────────────────────────────────────────────────
SS_COMP   = "fitted_components"
SS_FIELD  = "fitted_field"
SS_K      = "fitted_k"

for key in (SS_COMP, SS_FIELD, SS_K):
    if key not in st.session_state:
        st.session_state[key] = None

# ── metrics row ───────────────────────────────────────────────────────────────
m_cols = st.columns(4)
m_cols[0].metric("Прогонов", f"{len(t_arr):,}")
m_cols[1].metric("Отказов", f"{int(e_arr.sum()):,}")
if len(t_arr):
    km_med_idx = np.searchsorted(-km_s, -0.5)
    km_med = float(km_t[min(km_med_idx, len(km_t) - 1)])
    m_cols[2].metric("KM медиана (дн.)", f"{km_med:.0f}")
m_cols[3].metric("KM до 365 дн.", f"{float(km_s[np.searchsorted(km_t, 365.0) - 1]):.2f}" if km_t.max() >= 365 else "—")

# ── main figure placeholder ───────────────────────────────────────────────────
fig_placeholder = st.empty()

# Draw plain KM while no fit yet
def _plain_km_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_t, y=km_s, mode="lines", line_shape="hv",
                             line=dict(color=KM_COLOR, width=2.5),
                             name=f"KM эмпирическая  (n={len(t_arr):,})"))
    fig.update_layout(title=f"KM-кривая — {sel_field}",
                      xaxis_title="ННО (дней)", yaxis_title="Вероятность выживания S(t)",
                      height=480, yaxis=dict(range=[0, 1.02]),
                      xaxis=dict(range=[0, max_t_days]))
    return fig

if st.session_state[SS_COMP] is None or st.session_state[SS_FIELD] != sel_field or st.session_state[SS_K] != k:
    fig_placeholder.plotly_chart(_plain_km_fig(), use_container_width=True)

# ── Fit button ────────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 5])
with btn_col:
    do_fit = st.button("Подобрать модель", type="primary", use_container_width=True)

if do_fit:
    with st.spinner(f"Fitting K={k} ({n_restarts} restarts)…"):
        fitted, ssr0 = fit_km(km_t, km_s, k, n_restarts=n_restarts)
    st.session_state[SS_COMP]  = fitted
    st.session_state[SS_FIELD] = sel_field
    st.session_state[SS_K]     = k
    st.success(f"Подбор завершён  SSR={ssr0:.5f}")

# ── Editable parameter panel + figure ─────────────────────────────────────────
fitted_comp = st.session_state.get(SS_COMP)
if fitted_comp is not None and st.session_state[SS_FIELD] == sel_field and st.session_state[SS_K] == k:

    st.subheader("Параметры компонент (редактируемые)")

    n_comp = len(fitted_comp)
    # Build one column per component + one for weights
    param_cols = st.columns(n_comp)
    live_components: list[dict] = []
    raw_weights: list[float] = []

    for j, comp in enumerate(fitted_comp):
        with param_cols[j]:
            st.markdown(f"**Компонента {j+1}**")
            color_dot = f'<span style="color:{COMP_COLORS[j % len(COMP_COLORS)]};font-size:1.4em">●</span>'
            st.markdown(color_dot, unsafe_allow_html=True)

            beta_v = st.number_input(
                f"β{j+1} (форма)", min_value=0.05, max_value=8.0,
                value=round(comp["beta"], 3), step=0.01,
                key=f"beta_{j}_{sel_field}_{k}",
            )
            eta_v = st.number_input(
                f"η{j+1} (масштаб, дн.)", min_value=1.0, max_value=10_000.0,
                value=round(comp["eta"], 0), step=1.0,
                key=f"eta_{j}_{sel_field}_{k}",
            )
            w_v = st.number_input(
                f"w{j+1} (вес)", min_value=0.001, max_value=0.999,
                value=round(comp["weight"], 3), step=0.001,
                format="%.3f",
                key=f"w_{j}_{sel_field}_{k}",
            )
            raw_weights.append(w_v)
            live_components.append({"beta": float(beta_v), "eta": float(eta_v),
                                    "weight": float(w_v)})

    # Normalise weights so they sum to 1
    total_w = sum(c["weight"] for c in live_components) or 1.0
    for comp in live_components:
        comp["weight"] /= total_w

    if abs(total_w - 1.0) > 0.01:
        st.info(f"Веса нормированы (сумма была {total_w:.3f} → 1.000)")

    # Live figure
    live_fig = _build_figure(km_t, km_s, live_components, max_t_days, sel_field)
    fig_placeholder.plotly_chart(live_fig, use_container_width=True)

    # Residuals + summary
    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.plotly_chart(_build_residual_figure(km_t, km_s, live_components),
                        use_container_width=True)
    with right_col:
        ssr_live = _ssr(km_t, km_s, live_components)
        st.metric("SSR (текущие параметры)", f"{ssr_live:.5f}")
        rows = []
        for j, comp in enumerate(live_components):
            mean_j = comp["eta"] * float(gamma_fn(1 + 1 / comp["beta"]))
            med_j  = comp["eta"] * np.log(2) ** (1 / comp["beta"])
            rows.append({
                "K":         f"K{j+1}",
                "w":         f"{comp['weight']:.3f}",
                "β":         f"{comp['beta']:.3f}",
                "η":         f"{comp['eta']:.0f}",
                "Медиана":   f"{med_j:.0f} дн.",
                "Среднее":   f"{mean_j:.0f} дн.",
                "Режим":     "убыв. риск" if comp["beta"] < 1 else ("возраст. риск" if comp["beta"] > 1 else "пост."),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
