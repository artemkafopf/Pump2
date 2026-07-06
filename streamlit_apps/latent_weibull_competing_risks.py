from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from analysis import (
    CompetingRiskModel,
    ThreeComponentLatentWeibullModel,
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    build_survival_curve_weights,
    competing_curve_frame,
    competing_next_window_probabilities,
    competing_overall_survival,
    filter_km_frame_for_fit,
    fit_latent_weibull_to_km_frame,
    fit_three_component_latent_weibull_to_km_frame,
    fit_weibull_to_km_frame,
    kaplan_meier_frame,
    latent_curve_frame,
    latent_life_quantile,
    latent_next_window_failure_probability,
    latent_posterior_given_failure,
    latent_posterior_given_survival,
    latent_remaining_life_quantile,
    latent_survival,
    weibull_survival,
)
from scripts.analyze_kpod_window_thresholds import ALL_PATH, load_runs


st.set_page_config(
    page_title="Latent Weibull and Competing Risks",
    page_icon=":material/query_stats:",
    layout="wide",
)


MANUAL_SCOPE = "Manual / no field data"
WEIGHT_SCHEME_LABELS = {
    "risk_weighted": "Risk-weighted",
    "uniform": "Uniform",
    "sqrt_risk": "Sqrt-risk",
    "blended_tail": "Blended tail emphasis",
}
WEIGHT_SCHEME_CODES = {label: code for code, label in WEIGHT_SCHEME_LABELS.items()}
LatentModel = TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _normalize_weight_pair(weight_1: float, weight_2: float) -> tuple[float, float]:
    weights = np.asarray([float(weight_1), float(weight_2)], dtype=float)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1])


def _clip_weight_1(weight_1: float) -> float:
    return float(np.clip(float(weight_1), 1e-6, 1.0 - 1e-6))


def _normalize_three_weights(weight_1: float, weight_2: float) -> tuple[float, float, float]:
    w1 = float(np.clip(float(weight_1), 1e-6, 1.0 - 2e-6))
    max_w2 = max(1e-6, 1.0 - w1 - 1e-6)
    w2 = float(np.clip(float(weight_2), 1e-6, max_w2))
    w3 = 1.0 - w1 - w2
    if w3 <= 1e-6:
        w2 = max(1e-6, w2 - (1e-6 - w3) - 1e-9)
        w3 = 1.0 - w1 - w2
    return float(w1), float(w2), float(w3)


def _build_latent_model_from_inputs(
    mode_count: int,
    *,
    class_1_label: str,
    class_2_label: str,
    class_3_label: str,
    weight_1_input: float,
    weight_2_input: float,
    beta_1_input: float,
    eta_1_input: float,
    beta_2_input: float,
    eta_2_input: float,
    beta_3_input: float,
    eta_3_input: float,
) -> tuple[LatentModel, float, float, float | None]:
    if int(mode_count) == 2:
        effective_w1 = _clip_weight_1(weight_1_input)
        model = TwoComponentLatentWeibullModel(
            weight_1=effective_w1,
            component_1=WeibullParameters(beta=_clip_latent_beta(beta_1_input), eta=eta_1_input, label=class_1_label),
            component_2=WeibullParameters(beta=_clip_latent_beta(beta_2_input), eta=eta_2_input, label=class_2_label),
        )
        return model, float(model.weight_1), float(model.weight_2), None
    effective_w1, effective_w2, effective_w3 = _normalize_three_weights(weight_1_input, weight_2_input)
    model = ThreeComponentLatentWeibullModel(
        weight_1=effective_w1,
        weight_2=effective_w2,
        component_1=WeibullParameters(beta=_clip_latent_beta(beta_1_input), eta=eta_1_input, label=class_1_label),
        component_2=WeibullParameters(beta=_clip_latent_beta(beta_2_input), eta=eta_2_input, label=class_2_label),
        component_3=WeibullParameters(beta=_clip_latent_beta(beta_3_input), eta=eta_3_input, label=class_3_label),
    )
    return model, effective_w1, effective_w2, effective_w3


def _clip_weight_gamma(gamma: float) -> float:
    return float(np.clip(float(gamma), 0.0, 2.0))


def _clip_weight_tail_lambda(tail_lambda: float) -> float:
    return float(np.clip(float(tail_lambda), 0.0, 5.0))


def _clip_weight_tail_power(tail_power: float) -> float:
    return float(np.clip(float(tail_power), 0.0, 4.0))


def _clip_latent_beta(beta: float) -> float:
    return float(np.clip(float(beta), 0.05, 8.0))


def _latent_component_rows(model: LatentModel) -> list[tuple[WeibullParameters, float]]:
    if isinstance(model, ThreeComponentLatentWeibullModel):
        return [
            (model.component_1, float(model.weight_1)),
            (model.component_2, float(model.weight_2)),
            (model.component_3, float(model.weight_3)),
        ]
    return [
        (model.component_1, float(model.weight_1)),
        (model.component_2, float(model.weight_2)),
    ]


def _latent_mean_life(model: LatentModel) -> float:
    total = 0.0
    for component, weight in _latent_component_rows(model):
        total += float(weight) * component.eta * math.gamma(1.0 + 1.0 / component.beta)
    return float(total)


def _weibull_mean_life(params: WeibullParameters) -> float:
    return float(params.eta * math.gamma(1.0 + 1.0 / params.beta))


def _weibull_life_quantile(probability: float, params: WeibullParameters) -> float:
    p = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    return float(params.eta * np.power(-np.log(1.0 - p), 1.0 / params.beta))


def _weibull_curve_frame(params: WeibullParameters, max_time: float, num_points: int = 400) -> pd.DataFrame:
    time_grid = np.linspace(0.0, max(float(max_time), 1.0), int(num_points))
    survival = np.asarray(weibull_survival(time_grid, params), dtype=float)
    return pd.DataFrame(
        {
            "time": time_grid,
            "survival": survival,
            "failure": 1.0 - survival,
        }
    )


def _line_figure(frame: pd.DataFrame, series: list[tuple[str, str]], title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    for column, label in series:
        figure.add_trace(go.Scatter(x=frame["time"], y=frame[column], mode="lines", name=label))
    figure.update_layout(title=title, height=420, xaxis_title="Time", yaxis_title=y_title, legend_title="Series")
    return figure


def _latent_dashboard_figure(
    frame: pd.DataFrame,
    model: LatentModel,
    observed_km: pd.DataFrame | None = None,
    title_suffix: str = "",
) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, subplot_titles=("Survival", "Hazard", "Density", "Posterior Among Survivors"))
    figure.add_trace(go.Scatter(x=frame["time"], y=frame["survival"], mode="lines", name="Mixture survival", line={"width": 3}), row=1, col=1)
    for index, (component, _) in enumerate(_latent_component_rows(model), start=1):
        figure.add_trace(
            go.Scatter(x=frame["time"], y=frame[f"component_{index}_survival"], mode="lines", name=f"{component.label} survival"),
            row=1,
            col=1,
        )
    if observed_km is not None and not observed_km.empty:
        figure.add_trace(
            go.Scatter(
                x=observed_km["time"],
                y=observed_km["survival"],
                mode="lines+markers",
                name="Observed KM",
                line={"dash": "dash", "width": 2},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(go.Scatter(x=frame["time"], y=frame["hazard"], mode="lines", name="Mixture hazard"), row=1, col=2)
    figure.add_trace(go.Scatter(x=frame["time"], y=frame["density"], mode="lines", name="Mixture density"), row=2, col=1)
    for index, (component, _) in enumerate(_latent_component_rows(model), start=1):
        figure.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame[f"posterior_{index}_survival"],
                mode="lines",
                name=f"P({component.label} | survive to t)",
            ),
            row=2,
            col=2,
        )
    title = "Latent Weibull Mixture Overview"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    figure.update_layout(height=780, title=title)
    figure.update_xaxes(title_text="Time", row=1, col=1)
    figure.update_xaxes(title_text="Time", row=1, col=2)
    figure.update_xaxes(title_text="Time", row=2, col=1)
    figure.update_xaxes(title_text="Time", row=2, col=2)
    return figure


def _latent_fit_comparison_figure(
    current_curve: pd.DataFrame,
    km_frame: pd.DataFrame,
    classical_curve: pd.DataFrame | None = None,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=current_curve["time"],
            y=current_curve["survival"],
            mode="lines",
            name="Current mixture fit",
            line={"width": 3},
        )
    )
    if classical_curve is not None and not classical_curve.empty:
        figure.add_trace(
            go.Scatter(
                x=classical_curve["time"],
                y=classical_curve["survival"],
                mode="lines",
                name="Current classical Weibull",
                line={"width": 3, "dash": "dot"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=km_frame["time"],
            y=km_frame["survival"],
            mode="lines+markers",
            name="Observed KM",
            line={"dash": "dash"},
        )
    )
    figure.update_layout(
        title="Observed Field Survival vs Current Fits",
        height=420,
        xaxis_title="Time",
        yaxis_title="Survival",
    )
    return figure


def _competing_dashboard_figure(frame: pd.DataFrame, model: CompetingRiskModel) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Overall Survival", "Overall Hazard", "Cause-Specific CIF", "Cause-Specific Hazards"),
    )
    figure.add_trace(go.Scatter(x=frame["time"], y=frame["survival"], mode="lines", name="Overall survival"), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["time"], y=frame["overall_hazard"], mode="lines", name="Overall hazard"), row=1, col=2)
    for index, cause in enumerate(model.causes, start=1):
        figure.add_trace(
            go.Scatter(x=frame["time"], y=frame[f"cif_{index}"], mode="lines", name=f"{cause.label} CIF"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=frame["time"], y=frame[f"hazard_{index}"], mode="lines", name=f"{cause.label} hazard"),
            row=2,
            col=2,
        )
    figure.update_layout(height=780, title="Competing Risks Overview")
    return figure


@st.cache_data(show_spinner=False)
def load_field_runs() -> pd.DataFrame:
    runs = load_runs(ALL_PATH).copy()
    keep = runs[["row_id", "Месторождение", "Наработка (сут)", "Failure Flag"]].copy()
    keep = keep.rename(
        columns={
            "Месторождение": "field",
            "Наработка (сут)": "duration",
            "Failure Flag": "event",
        }
    )
    keep["field"] = keep["field"].astype("string")
    keep["duration"] = pd.to_numeric(keep["duration"], errors="coerce")
    keep["event"] = pd.to_numeric(keep["event"], errors="coerce")
    keep = keep.loc[keep["duration"].notna() & keep["event"].isin([0, 1])].copy()
    return keep.reset_index(drop=True)


def field_scope_options(runs: pd.DataFrame) -> list[str]:
    fields = sorted(runs["field"].dropna().astype(str).unique().tolist())
    return [MANUAL_SCOPE, "Global"] + fields


def km_for_scope(runs: pd.DataFrame, scope: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if scope == MANUAL_SCOPE:
        return None, None
    if scope == "Global":
        subset = runs.copy()
    else:
        subset = runs.loc[runs["field"].astype(str) == scope].copy()
    if subset.empty:
        return subset, None
    km = kaplan_meier_frame(
        subset["duration"].to_numpy(dtype=float),
        subset["event"].to_numpy(dtype=int),
    )
    return subset, km


def _fit_error_metrics(
    model: LatentModel,
    km_frame: pd.DataFrame | None,
    *,
    weight_scheme: str,
    weight_gamma: float,
    weight_tail_lambda: float,
    weight_tail_power: float,
    min_n_risk: int,
) -> dict[str, float] | None:
    if km_frame is None or km_frame.empty:
        return None
    fit_frame = filter_km_frame_for_fit(km_frame, min_n_risk=min_n_risk)
    fitted = np.asarray(latent_survival(fit_frame["time"].to_numpy(dtype=float), model), dtype=float)
    observed = fit_frame["survival"].to_numpy(dtype=float)
    n_risk = fit_frame["n_risk"].to_numpy(dtype=float) if "n_risk" in fit_frame.columns else None
    weights = build_survival_curve_weights(
        fit_frame["time"].to_numpy(dtype=float),
        n_risk=n_risk,
        weight_scheme=weight_scheme,
        gamma=weight_gamma,
        tail_lambda=weight_tail_lambda,
        tail_power=weight_tail_power,
    )
    residual = fitted - observed
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "weighted_rmse": float(np.sqrt(np.mean(weights * np.square(residual)))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "points_used": int(len(fit_frame)),
    }


def _weibull_fit_error_metrics(
    params: WeibullParameters,
    km_frame: pd.DataFrame | None,
    *,
    weight_scheme: str,
    weight_gamma: float,
    weight_tail_lambda: float,
    weight_tail_power: float,
    min_n_risk: int,
) -> dict[str, float] | None:
    if km_frame is None or km_frame.empty:
        return None
    fit_frame = filter_km_frame_for_fit(km_frame, min_n_risk=min_n_risk)
    fitted = np.asarray(weibull_survival(fit_frame["time"].to_numpy(dtype=float), params), dtype=float)
    observed = fit_frame["survival"].to_numpy(dtype=float)
    n_risk = fit_frame["n_risk"].to_numpy(dtype=float) if "n_risk" in fit_frame.columns else None
    weights = build_survival_curve_weights(
        fit_frame["time"].to_numpy(dtype=float),
        n_risk=n_risk,
        weight_scheme=weight_scheme,
        gamma=weight_gamma,
        tail_lambda=weight_tail_lambda,
        tail_power=weight_tail_power,
    )
    residual = fitted - observed
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "weighted_rmse": float(np.sqrt(np.mean(weights * np.square(residual)))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "points_used": int(len(fit_frame)),
    }


for key, value in {
    "latent_mode_count": 3,
    "latent_class_1_label": "Early-life",
    "latent_class_2_label": "Mid-life",
    "latent_class_3_label": "Lifecycle",
    "latent_weight_1_input": 0.35,
    "latent_weight_2_input": 0.30,
    "latent_beta_1_input": 0.85,
    "latent_eta_1_input": 120.0,
    "latent_beta_2_input": 1.20,
    "latent_eta_2_input": 260.0,
    "latent_beta_3_input": 1.90,
    "latent_eta_3_input": 420.0,
    "classical_beta_input": 1.35,
    "classical_eta_input": 300.0,
    "optimize_latent_weight_1": True,
    "optimize_latent_weight_2": True,
    "optimize_latent_beta_1": True,
    "optimize_latent_eta_1": True,
    "optimize_latent_beta_2": True,
    "optimize_latent_eta_2": True,
    "optimize_latent_beta_3": True,
    "optimize_latent_eta_3": True,
    "optimize_classical_beta": True,
    "optimize_classical_eta": True,
    "latent_fit_scope": MANUAL_SCOPE,
    "latent_fit_max_iter": 400,
    "latent_fit_num_starts": 8,
    "latent_fit_weight_scheme": WEIGHT_SCHEME_LABELS["blended_tail"],
    "latent_fit_weight_gamma": 0.5,
    "latent_fit_weight_tail_lambda": 1.0,
    "latent_fit_weight_tail_power": 1.0,
    "latent_fit_min_n_risk": 5,
}.items():
    st.session_state.setdefault(key, value)

pending_widget_updates = st.session_state.pop("latent_pending_widget_updates", None)
if pending_widget_updates:
    for key, value in pending_widget_updates.items():
        st.session_state[key] = value

w1_state, w2_state, _ = _normalize_three_weights(
    st.session_state["latent_weight_1_input"],
    st.session_state["latent_weight_2_input"],
)
st.session_state["latent_weight_1_input"] = w1_state
st.session_state["latent_weight_2_input"] = w2_state
st.session_state["latent_beta_1_input"] = _clip_latent_beta(st.session_state["latent_beta_1_input"])
st.session_state["latent_beta_2_input"] = _clip_latent_beta(st.session_state["latent_beta_2_input"])
st.session_state["latent_beta_3_input"] = _clip_latent_beta(st.session_state["latent_beta_3_input"])
st.session_state["latent_fit_weight_gamma"] = _clip_weight_gamma(st.session_state["latent_fit_weight_gamma"])
st.session_state["latent_fit_weight_tail_lambda"] = _clip_weight_tail_lambda(st.session_state["latent_fit_weight_tail_lambda"])
st.session_state["latent_fit_weight_tail_power"] = _clip_weight_tail_power(st.session_state["latent_fit_weight_tail_power"])


st.title("Latent Weibull and Competing Risks Playground")
st.caption(
    "This app supports fitting either a two-component or three-component latent Weibull mixture directly to observed field survival curves "
    "for `Global` or a selected field, and also fits a classical single Weibull for direct comparison before you keep varying all parameters manually."
)

runs = load_field_runs()
scope_options = field_scope_options(runs)

with st.sidebar:
    st.header("Global Controls")
    max_time = st.number_input("Plot horizon", min_value=30.0, value=720.0, step=30.0)
    num_points = st.slider("Curve resolution", min_value=200, max_value=1200, value=500, step=50)
    current_age = st.number_input("Current age", min_value=0.0, value=180.0, step=10.0)
    prediction_horizon = st.number_input("Future window", min_value=1.0, value=90.0, step=5.0)
    quantile_probability = st.slider("Remaining-life quantile p", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

    st.header("Field Survival Overlay")
    selected_scope = st.selectbox("Observed survival source", options=scope_options, key="latent_fit_scope")
    optimizer_max_iter = st.slider("Optimizer max iterations", min_value=50, max_value=1000, value=int(st.session_state["latent_fit_max_iter"]), step=50, key="latent_fit_max_iter")
    optimizer_num_starts = st.slider("Optimizer starting points", min_value=1, max_value=24, value=int(st.session_state["latent_fit_num_starts"]), step=1, key="latent_fit_num_starts")
    weight_scheme_label = st.selectbox("Curve weighting", options=list(WEIGHT_SCHEME_CODES.keys()), key="latent_fit_weight_scheme")
    weight_gamma = st.slider("Risk exponent gamma", min_value=0.0, max_value=2.0, value=float(st.session_state["latent_fit_weight_gamma"]), step=0.05, key="latent_fit_weight_gamma")
    weight_tail_lambda = st.slider("Tail boost lambda", min_value=0.0, max_value=5.0, value=float(st.session_state["latent_fit_weight_tail_lambda"]), step=0.1, key="latent_fit_weight_tail_lambda")
    weight_tail_power = st.slider("Tail power p", min_value=0.0, max_value=4.0, value=float(st.session_state["latent_fit_weight_tail_power"]), step=0.1, key="latent_fit_weight_tail_power")
    min_n_risk = st.number_input("Ignore tail below n_risk", min_value=0, max_value=100, value=int(st.session_state["latent_fit_min_n_risk"]), step=1, key="latent_fit_min_n_risk")

weight_scheme_code = WEIGHT_SCHEME_CODES[weight_scheme_label]

field_subset, observed_km = km_for_scope(runs, selected_scope)

tabs = st.tabs(["Latent Weibull", "Competing Risks", "Interpretation"])

with tabs[0]:
    latent_cols = st.columns(5)
    with latent_cols[0]:
        st.subheader("Mixture")
        latent_mode_count = st.radio("Latent modes", options=[2, 3], horizontal=True, key="latent_mode_count")
        class_1_label = st.text_input("Class 1 label", key="latent_class_1_label")
        class_2_label = st.text_input("Class 2 label", key="latent_class_2_label")
        class_3_label = st.text_input("Class 3 label", key="latent_class_3_label", disabled=int(latent_mode_count) == 2)
        weight_1_input = st.number_input("Weight 1", min_value=0.001, max_value=0.998, step=0.01, format="%.3f", key="latent_weight_1_input")
        weight_2_input = st.number_input("Weight 2", min_value=0.001, max_value=0.998, step=0.01, format="%.3f", key="latent_weight_2_input", disabled=int(latent_mode_count) == 2)
        optimize_weight_1 = st.checkbox("Optimize weight 1", key="optimize_latent_weight_1")
        optimize_weight_2 = st.checkbox("Optimize weight 2", key="optimize_latent_weight_2", disabled=int(latent_mode_count) == 2)
        if int(latent_mode_count) == 2:
            w1_preview = _clip_weight_1(weight_1_input)
            st.caption(f"Weight 2 = 1 - Weight 1 = {1.0 - w1_preview:.3f}")
        else:
            w1_preview, w2_preview, w3_preview = _normalize_three_weights(weight_1_input, weight_2_input)
            st.caption(f"Weight 3 = 1 - Weight 1 - Weight 2 = {w3_preview:.3f}")
    with latent_cols[1]:
        st.subheader("Class 1")
        beta_1_input = st.number_input("Beta 1", min_value=0.05, max_value=8.0, value=0.85, step=0.05, key="latent_beta_1_input")
        eta_1_input = st.number_input("Eta 1", min_value=1.0, value=120.0, step=10.0, key="latent_eta_1_input")
        optimize_beta_1 = st.checkbox("Optimize beta 1", key="optimize_latent_beta_1")
        optimize_eta_1 = st.checkbox("Optimize eta 1", key="optimize_latent_eta_1")
    with latent_cols[2]:
        st.subheader("Class 2")
        beta_2_input = st.number_input("Beta 2", min_value=0.05, max_value=8.0, value=1.20, step=0.05, key="latent_beta_2_input")
        eta_2_input = st.number_input("Eta 2", min_value=1.0, value=260.0, step=10.0, key="latent_eta_2_input")
        optimize_beta_2 = st.checkbox("Optimize beta 2", key="optimize_latent_beta_2")
        optimize_eta_2 = st.checkbox("Optimize eta 2", key="optimize_latent_eta_2")
    with latent_cols[3]:
        st.subheader("Class 3")
        beta_3_input = st.number_input("Beta 3", min_value=0.05, max_value=8.0, value=1.90, step=0.05, key="latent_beta_3_input", disabled=int(latent_mode_count) == 2)
        eta_3_input = st.number_input("Eta 3", min_value=1.0, value=420.0, step=10.0, key="latent_eta_3_input", disabled=int(latent_mode_count) == 2)
        optimize_beta_3 = st.checkbox("Optimize beta 3", key="optimize_latent_beta_3", disabled=int(latent_mode_count) == 2)
        optimize_eta_3 = st.checkbox("Optimize eta 3", key="optimize_latent_eta_3", disabled=int(latent_mode_count) == 2)
    with latent_cols[4]:
        st.subheader("Classical Weibull")
        classical_beta_input = st.number_input("Classical beta", min_value=0.05, value=1.35, step=0.05, key="classical_beta_input")
        classical_eta_input = st.number_input("Classical eta", min_value=1.0, value=300.0, step=10.0, key="classical_eta_input")
        optimize_classical_beta = st.checkbox("Optimize classical beta", key="optimize_classical_beta")
        optimize_classical_eta = st.checkbox("Optimize classical eta", key="optimize_classical_eta")

    if int(latent_mode_count) == 3:
        eta_inputs_are_ordered = bool(eta_1_input <= eta_2_input <= eta_3_input)
        if not eta_inputs_are_ordered:
            st.warning("For the 3-class latent fit we only enforce `eta1 <= eta2 <= eta3`. The optimizer will return classes re-ordered by eta.")

    latent_model, effective_w1, effective_w2, effective_w3 = _build_latent_model_from_inputs(
        int(latent_mode_count),
        class_1_label=class_1_label,
        class_2_label=class_2_label,
        class_3_label=class_3_label,
        weight_1_input=weight_1_input,
        weight_2_input=weight_2_input,
        beta_1_input=beta_1_input,
        eta_1_input=eta_1_input,
        beta_2_input=beta_2_input,
        eta_2_input=eta_2_input,
        beta_3_input=beta_3_input,
        eta_3_input=eta_3_input,
    )
    classical_params = WeibullParameters(beta=classical_beta_input, eta=classical_eta_input, label="Classical Weibull")

    fit_button_disabled = observed_km is None or observed_km.empty
    fit_button_label = "Calculate optimal fits to observed survival"
    if fit_button_disabled:
        fit_button_label = "Select Global or a field to enable fitting"
    optimize_any = any(
        [
            optimize_weight_1,
            optimize_weight_2 and int(latent_mode_count) == 3,
            optimize_beta_1,
            optimize_eta_1,
            optimize_beta_2,
            optimize_eta_2,
            optimize_beta_3 and int(latent_mode_count) == 3,
            optimize_eta_3 and int(latent_mode_count) == 3,
            optimize_classical_beta,
            optimize_classical_eta,
        ]
    )
    if not optimize_any:
        st.info("All parameters are fixed. Pressing fit will only score the current latent and classical curves against the observed survival.")
    fit_pressed = st.button(fit_button_label, type="primary", disabled=fit_button_disabled)

    if fit_pressed and observed_km is not None and not observed_km.empty:
        if int(latent_mode_count) == 2:
            assert isinstance(latent_model, TwoComponentLatentWeibullModel)
            fit_result = fit_latent_weibull_to_km_frame(
                observed_km,
                initial_model=latent_model,
                optimize_beta_1=optimize_beta_1,
                optimize_eta_1=optimize_eta_1,
                optimize_beta_2=optimize_beta_2,
                optimize_eta_2=optimize_eta_2,
                optimize_weight_1=optimize_weight_1,
                optimize_weight_2=False,
                weight_scheme=weight_scheme_code,
                weight_gamma=float(weight_gamma),
                weight_tail_lambda=float(weight_tail_lambda),
                weight_tail_power=float(weight_tail_power),
                min_n_risk=int(min_n_risk),
                num_starts=int(optimizer_num_starts),
                max_iter=int(optimizer_max_iter),
            )
        else:
            assert isinstance(latent_model, ThreeComponentLatentWeibullModel)
            fit_result = fit_three_component_latent_weibull_to_km_frame(
                observed_km,
                initial_model=latent_model,
                optimize_beta_1=optimize_beta_1,
                optimize_eta_1=optimize_eta_1,
                optimize_beta_2=optimize_beta_2,
                optimize_eta_2=optimize_eta_2,
                optimize_beta_3=optimize_beta_3,
                optimize_eta_3=optimize_eta_3,
                optimize_weight_1=optimize_weight_1,
                optimize_weight_2=optimize_weight_2,
                weight_scheme=weight_scheme_code,
                weight_gamma=float(weight_gamma),
                weight_tail_lambda=float(weight_tail_lambda),
                weight_tail_power=float(weight_tail_power),
                min_n_risk=int(min_n_risk),
                num_starts=int(optimizer_num_starts),
                max_iter=int(optimizer_max_iter),
            )
        classical_fit_result = fit_weibull_to_km_frame(
            observed_km,
            initial_params=classical_params,
            optimize_beta=optimize_classical_beta,
            optimize_eta=optimize_classical_eta,
            weight_scheme=weight_scheme_code,
            weight_gamma=float(weight_gamma),
            weight_tail_lambda=float(weight_tail_lambda),
            weight_tail_power=float(weight_tail_power),
            min_n_risk=int(min_n_risk),
            num_starts=int(optimizer_num_starts),
            max_iter=int(optimizer_max_iter),
        )
        fitted_w1 = float(np.clip(fit_result.model.weight_1, 0.001, 0.999))
        pending_updates = {
            "latent_beta_1_input": float(fit_result.model.component_1.beta),
            "latent_eta_1_input": float(fit_result.model.component_1.eta),
            "latent_beta_2_input": float(fit_result.model.component_2.beta),
            "latent_eta_2_input": float(fit_result.model.component_2.eta),
            "latent_weight_1_input": fitted_w1,
            "classical_beta_input": float(classical_fit_result.params.beta),
            "classical_eta_input": float(classical_fit_result.params.eta),
        }
        if isinstance(fit_result.model, ThreeComponentLatentWeibullModel):
            pending_updates.update(
                {
                    "latent_beta_3_input": float(fit_result.model.component_3.beta),
                    "latent_eta_3_input": float(fit_result.model.component_3.eta),
                    "latent_weight_2_input": float(np.clip(fit_result.model.weight_2, 0.001, 0.998)),
                }
            )
        st.session_state["latent_pending_widget_updates"] = pending_updates
        st.session_state["latent_last_fit"] = {
            "scope": selected_scope,
            "mode_count": int(latent_mode_count),
            "success": bool(fit_result.success),
            "message": str(fit_result.message),
            "optimized_parameters": ", ".join(
                [
                    label
                    for label, enabled in [
                        ("weight_1", optimize_weight_1),
                        ("weight_2", optimize_weight_2 and int(latent_mode_count) == 3),
                        ("beta_1", optimize_beta_1),
                        ("eta_1", optimize_eta_1),
                        ("beta_2", optimize_beta_2),
                        ("eta_2", optimize_eta_2),
                        ("beta_3", optimize_beta_3 and int(latent_mode_count) == 3),
                        ("eta_3", optimize_eta_3 and int(latent_mode_count) == 3),
                    ]
                    if enabled
                ]
            )
            or "(none)",
            "objective_value": float(fit_result.objective_value),
            "rmse": float(fit_result.rmse),
            "weighted_rmse": float(fit_result.weighted_rmse),
            "n_iter": int(fit_result.n_iter),
            "nfev": int(fit_result.nfev),
            "n_starts": int(fit_result.n_starts),
            "best_start_index": int(fit_result.best_start_index),
            "weight_scheme": weight_scheme_code,
            "weight_gamma": float(weight_gamma),
            "weight_tail_lambda": float(weight_tail_lambda),
            "weight_tail_power": float(weight_tail_power),
            "min_n_risk": int(min_n_risk),
        }
        st.session_state["classical_last_fit"] = {
            "scope": selected_scope,
            "success": bool(classical_fit_result.success),
            "message": str(classical_fit_result.message),
            "optimized_parameters": ", ".join(
                [
                    label
                    for label, enabled in [
                        ("beta", optimize_classical_beta),
                        ("eta", optimize_classical_eta),
                    ]
                    if enabled
                ]
            )
            or "(none)",
            "objective_value": float(classical_fit_result.objective_value),
            "rmse": float(classical_fit_result.rmse),
            "weighted_rmse": float(classical_fit_result.weighted_rmse),
            "n_iter": int(classical_fit_result.n_iter),
            "nfev": int(classical_fit_result.nfev),
            "n_starts": int(classical_fit_result.n_starts),
            "best_start_index": int(classical_fit_result.best_start_index),
            "weight_scheme": weight_scheme_code,
            "weight_gamma": float(weight_gamma),
            "weight_tail_lambda": float(weight_tail_lambda),
            "weight_tail_power": float(weight_tail_power),
            "min_n_risk": int(min_n_risk),
        }
        st.rerun()

    latent_model, effective_w1, effective_w2, effective_w3 = _build_latent_model_from_inputs(
        int(st.session_state["latent_mode_count"]),
        class_1_label=st.session_state["latent_class_1_label"],
        class_2_label=st.session_state["latent_class_2_label"],
        class_3_label=st.session_state["latent_class_3_label"],
        weight_1_input=float(st.session_state["latent_weight_1_input"]),
        weight_2_input=float(st.session_state["latent_weight_2_input"]),
        beta_1_input=float(st.session_state["latent_beta_1_input"]),
        eta_1_input=float(st.session_state["latent_eta_1_input"]),
        beta_2_input=float(st.session_state["latent_beta_2_input"]),
        eta_2_input=float(st.session_state["latent_eta_2_input"]),
        beta_3_input=float(st.session_state["latent_beta_3_input"]),
        eta_3_input=float(st.session_state["latent_eta_3_input"]),
    )
    classical_params = WeibullParameters(
        beta=float(st.session_state["classical_beta_input"]),
        eta=float(st.session_state["classical_eta_input"]),
        label="Classical Weibull",
    )

    horizon = max(float(max_time), float(observed_km["time"].max()) if observed_km is not None and not observed_km.empty else 0.0)
    latent_frame = latent_curve_frame(latent_model, max_time=horizon, num_points=num_points)
    classical_frame = _weibull_curve_frame(classical_params, max_time=horizon, num_points=num_points)
    survival_now = float(latent_survival(current_age, latent_model))
    failure_now = 1.0 - survival_now
    next_window_failure = float(latent_next_window_failure_probability(current_age, prediction_horizon, latent_model))
    posterior_survival = tuple(float(value) for value in latent_posterior_given_survival(current_age, latent_model))
    posterior_failure = tuple(float(value) for value in latent_posterior_given_failure(max(current_age, 1e-6), latent_model))
    b10 = float(latent_life_quantile(0.10, latent_model))
    b50 = float(latent_life_quantile(0.50, latent_model))
    b90 = float(latent_life_quantile(0.90, latent_model))
    mean_life = _latent_mean_life(latent_model)
    remaining_life = float(latent_remaining_life_quantile(current_age, quantile_probability, latent_model))
    live_fit_metrics = _fit_error_metrics(
        latent_model,
        observed_km,
        weight_scheme=weight_scheme_code,
        weight_gamma=float(weight_gamma),
        weight_tail_lambda=float(weight_tail_lambda),
        weight_tail_power=float(weight_tail_power),
        min_n_risk=int(min_n_risk),
    )
    classical_survival_now = float(weibull_survival(current_age, classical_params))
    classical_next_window_failure = float(
        np.clip(
            1.0
            - (
                float(weibull_survival(current_age + prediction_horizon, classical_params))
                / max(classical_survival_now, 1e-12)
            ),
            0.0,
            1.0,
        )
    )
    classical_b10 = _weibull_life_quantile(0.10, classical_params)
    classical_b50 = _weibull_life_quantile(0.50, classical_params)
    classical_b90 = _weibull_life_quantile(0.90, classical_params)
    classical_mean_life = _weibull_mean_life(classical_params)
    classical_fit_metrics = _weibull_fit_error_metrics(
        classical_params,
        observed_km,
        weight_scheme=weight_scheme_code,
        weight_gamma=float(weight_gamma),
        weight_tail_lambda=float(weight_tail_lambda),
        weight_tail_power=float(weight_tail_power),
        min_n_risk=int(min_n_risk),
    )

    top_metrics = st.columns(9)
    top_metrics[0].metric("Effective w1", f"{effective_w1:.3f}")
    top_metrics[1].metric("Effective w2", f"{effective_w2:.3f}")
    top_metrics[2].metric("Effective w3", "n/a" if effective_w3 is None else f"{effective_w3:.3f}")
    top_metrics[3].metric("S(current age)", f"{survival_now:.3f}")
    top_metrics[4].metric("F(current age)", f"{failure_now:.3f}")
    top_metrics[5].metric(f"Fail in next {int(prediction_horizon)}d", f"{next_window_failure:.3f}")
    top_metrics[6].metric("B50", f"{b50:.1f}")
    top_metrics[7].metric("Mean life", f"{mean_life:.1f}")
    if live_fit_metrics is not None:
        top_metrics[8].metric("Field RMSE", f"{live_fit_metrics['rmse']:.3f}")
    else:
        top_metrics[8].metric("Field RMSE", "n/a")

    if observed_km is not None and field_subset is not None and not field_subset.empty:
        field_metrics = st.columns(5)
        field_metrics[0].metric("Observed scope", selected_scope)
        field_metrics[1].metric("Rows", f"{len(field_subset):,}")
        field_metrics[2].metric("Failures", f"{int(field_subset['event'].sum()):,}")
        field_metrics[3].metric("Censored", f"{int((field_subset['event'] == 0).sum()):,}")
        field_metrics[4].metric("Observed max time", f"{float(observed_km['time'].max()):.1f}")

    summary_cols = st.columns(2)
    with summary_cols[0]:
        suffix = selected_scope if selected_scope != MANUAL_SCOPE else ""
        st.plotly_chart(_latent_dashboard_figure(latent_frame, latent_model, observed_km=observed_km, title_suffix=suffix), width="stretch")
        if observed_km is not None and not observed_km.empty:
            st.plotly_chart(_latent_fit_comparison_figure(latent_frame, observed_km, classical_curve=classical_frame), width="stretch")
    with summary_cols[1]:
        st.subheader("Latent Summary")
        optimize_flags = [
            bool(optimize_weight_1 or optimize_beta_1 or optimize_eta_1),
            bool((optimize_weight_2 if int(st.session_state["latent_mode_count"]) == 3 else False) or optimize_beta_2 or optimize_eta_2),
            bool(optimize_beta_3 or optimize_eta_3),
        ]
        summary_rows: list[dict[str, object]] = []
        for index, (component, weight) in enumerate(_latent_component_rows(latent_model), start=1):
            summary_rows.append(
                {
                    "class": component.label,
                    "prior_weight": float(weight),
                    "beta": component.beta,
                    "eta": component.eta,
                    "optimized_in_fit": optimize_flags[index - 1],
                    "posterior_given_survival_at_age": posterior_survival[index - 1],
                    "posterior_given_failure_at_age": posterior_failure[index - 1],
                }
            )
        summary_frame = pd.DataFrame(summary_rows)
        st.dataframe(summary_frame, width="stretch")
        st.dataframe(
            pd.DataFrame(
                [
                    {"quantity": "B10", "value": b10},
                    {"quantity": "B50", "value": b50},
                    {"quantity": "B90", "value": b90},
                    {"quantity": "Mean life", "value": mean_life},
                    {"quantity": "Median remaining life", "value": remaining_life},
                    {"quantity": f"Failure in next {int(prediction_horizon)} days", "value": next_window_failure},
                ]
            ),
            width="stretch",
        )
        if live_fit_metrics is not None:
            st.subheader("Current Fit vs Observed Survival")
            comparison_metrics = pd.DataFrame(
                [
                    {"model": "Latent mixture", **live_fit_metrics},
                    {"model": "Classical Weibull", **(classical_fit_metrics or {})},
                ]
            )
            st.dataframe(comparison_metrics, width="stretch")

        st.subheader("Classical Weibull Summary")
        st.dataframe(
            pd.DataFrame(
                [
                    {"parameter": "beta", "value": classical_params.beta},
                    {"parameter": "eta", "value": classical_params.eta},
                    {"parameter": "beta optimized in fit", "value": bool(optimize_classical_beta)},
                    {"parameter": "eta optimized in fit", "value": bool(optimize_classical_eta)},
                    {"parameter": "B10", "value": classical_b10},
                    {"parameter": "B50", "value": classical_b50},
                    {"parameter": "B90", "value": classical_b90},
                    {"parameter": "Mean life", "value": classical_mean_life},
                    {"parameter": f"Failure in next {int(prediction_horizon)} days", "value": classical_next_window_failure},
                    {"parameter": "S(current age)", "value": classical_survival_now},
                ]
            ),
            width="stretch",
        )

        last_fit = st.session_state.get("latent_last_fit")
        if last_fit and last_fit.get("scope") == selected_scope and int(last_fit.get("mode_count", 3)) == int(st.session_state["latent_mode_count"]):
            st.subheader("Last Latent Optimizer Run")
            st.dataframe(pd.DataFrame([last_fit]), width="stretch")
        classical_last_fit = st.session_state.get("classical_last_fit")
        if classical_last_fit and classical_last_fit.get("scope") == selected_scope:
            st.subheader("Last Classical Optimizer Run")
            st.dataframe(pd.DataFrame([classical_last_fit]), width="stretch")

        if observed_km is not None and not observed_km.empty:
            merged_export = observed_km.copy()
            merged_export["mixture_survival"] = np.asarray(latent_survival(merged_export["time"].to_numpy(dtype=float), latent_model), dtype=float)
            merged_export["classical_survival"] = np.asarray(weibull_survival(merged_export["time"].to_numpy(dtype=float), classical_params), dtype=float)
            st.download_button(
                "Download field-vs-models CSV",
                data=_csv_bytes(merged_export),
                file_name=f"survival_model_fit_{selected_scope.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
        st.download_button(
            "Download latent curve CSV",
            data=_csv_bytes(latent_frame),
            file_name="latent_weibull_curves.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download classical Weibull CSV",
            data=_csv_bytes(classical_frame),
            file_name="classical_weibull_curve.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("Cause-Specific Weibull Parameters")
    cause_count = st.slider("Number of competing causes", min_value=2, max_value=4, value=2, step=1)
    cause_columns = st.columns(cause_count)
    causes: list[WeibullParameters] = []
    default_labels = ["Electrical", "Mechanical", "Solids", "Corrosion"]
    default_betas = [1.35, 1.80, 0.95, 1.15]
    default_etas = [520.0, 360.0, 700.0, 900.0]
    for index in range(cause_count):
        with cause_columns[index]:
            label = st.text_input(f"Cause {index + 1} label", value=default_labels[index], key=f"cause_label_{index}")
            beta = st.number_input(
                f"Cause {index + 1} beta",
                min_value=0.05,
                value=default_betas[index],
                step=0.05,
                key=f"cause_beta_{index}",
            )
            eta = st.number_input(
                f"Cause {index + 1} eta",
                min_value=1.0,
                value=default_etas[index],
                step=10.0,
                key=f"cause_eta_{index}",
            )
            causes.append(WeibullParameters(beta=beta, eta=eta, label=label))

    competing_model = CompetingRiskModel(causes=tuple(causes))
    competing_frame = competing_curve_frame(competing_model, max_time=max_time, num_points=num_points)
    cause_windows = competing_next_window_probabilities(current_age, prediction_horizon, competing_model)
    overall_survival_now = float(competing_overall_survival(current_age, competing_model))
    overall_failure_now = 1.0 - overall_survival_now
    all_cause_window = float(cause_windows.attrs["all_cause_window_probability"])
    max_identity_error = float(cause_windows.shape[0] and competing_frame["identity_error"].abs().max())

    metrics = st.columns(5)
    metrics[0].metric("S(current age)", f"{overall_survival_now:.3f}")
    metrics[1].metric("All-cause failure by age", f"{overall_failure_now:.3f}")
    metrics[2].metric(f"Any failure in next {int(prediction_horizon)}d", f"{all_cause_window:.3f}")
    metrics[3].metric("Causes in model", f"{len(competing_model.causes)}")
    metrics[4].metric("Max identity error", f"{max_identity_error:.2e}")

    figure_cols = st.columns(2)
    with figure_cols[0]:
        st.plotly_chart(_competing_dashboard_figure(competing_frame, competing_model), width="stretch")
    with figure_cols[1]:
        st.subheader("Current-Age Predictions")
        enriched_windows = cause_windows.copy()
        for index, cause in enumerate(competing_model.causes, start=1):
            enriched_windows.loc[enriched_windows["cause"] == cause.label, "cif_at_current_age"] = float(
                competing_frame.loc[competing_frame["time"].sub(current_age).abs().idxmin(), f"cif_{index}"]
            )
        st.dataframe(enriched_windows, width="stretch")
        st.download_button(
            "Download competing-risk CSV",
            data=_csv_bytes(competing_frame),
            file_name="competing_risks_curves.csv",
            mime="text/csv",
        )

    st.plotly_chart(
        _line_figure(
            competing_frame,
            [(f"cif_{index}", cause.label) for index, cause in enumerate(competing_model.causes, start=1)],
            "Cause-Specific Cumulative Incidence",
            "Cumulative incidence",
        ),
        width="stretch",
    )

with tabs[2]:
    st.subheader("Interpretation Guardrails")
    latent_component_labels = [component.label for component, _ in _latent_component_rows(latent_model)]
    latent_component_label_text = ", ".join(f"`{label}`" for label in latent_component_labels)
    latent_mode_text = "two-component" if len(latent_component_labels) == 2 else "three-component"
    latent_eta_rule = "no eta ordering" if len(latent_component_labels) == 2 else "`eta1 <= eta2 <= eta3`"
    st.markdown(
        f"""
        - In the latent Weibull model, each unit belongs probabilistically to one hidden reliability class.
        - In the competing-risks model, every unit is exposed to all listed causes at the same time.
        - {latent_component_label_text} are not failure causes unless you separately justify that interpretation.
        - The latent model here is a simple {latent_mode_text}, two-parameter Weibull mixture with editable `beta`, `eta`, and weights.
        - The fit uses {latent_eta_rule} in the active latent-mode configuration; it does not force any `beta` ordering.
        - The classical benchmark is a single two-parameter Weibull with editable `beta` and `eta`.
        - When `Global` or a field is selected, the app fits both models to the observed Kaplan–Meier survival using weighted least squares and can try multiple deterministic starting points.
        - After fitting, you can keep varying all parameters manually and the fit-quality metrics update live against the observed field curve.
        """
    )

    comparison = pd.DataFrame(
        {
            "time": latent_frame["time"],
            "latent_survival": latent_frame["survival"],
            "classical_survival": classical_frame["survival"],
            "competing_survival": competing_frame.set_index("time").reindex(latent_frame["time"], method="nearest")["survival"].to_numpy(),
        }
    )
    comparison_figure = go.Figure()
    comparison_figure.add_trace(go.Scatter(x=comparison["time"], y=comparison["latent_survival"], mode="lines", name="Latent mixture survival"))
    comparison_figure.add_trace(go.Scatter(x=comparison["time"], y=comparison["classical_survival"], mode="lines", name="Classical Weibull survival"))
    comparison_figure.add_trace(
        go.Scatter(x=comparison["time"], y=comparison["competing_survival"], mode="lines", name="Competing-risk survival")
    )
    comparison_figure.update_layout(title="Overall Survival Curves", height=420, xaxis_title="Time", yaxis_title="Survival")
    st.plotly_chart(comparison_figure, width="stretch")

    comparison_rows = [
        {"model": "Latent Weibull", "S(current age)": survival_now, f"Next {int(prediction_horizon)}d failure": next_window_failure},
        {"model": "Classical Weibull", "S(current age)": classical_survival_now, f"Next {int(prediction_horizon)}d failure": classical_next_window_failure},
        {"model": "Competing risks", "S(current age)": overall_survival_now, f"Next {int(prediction_horizon)}d failure": all_cause_window},
    ]
    if live_fit_metrics is not None:
        comparison_rows[0]["Field RMSE"] = live_fit_metrics["rmse"]
    if classical_fit_metrics is not None:
        comparison_rows[1]["Field RMSE"] = classical_fit_metrics["rmse"]
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch")
