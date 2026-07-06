from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from analysis import predict_regularized_logistic_from_table
from scripts.analyze_field_mode_multivariate import (
    augment_model_features,
    load_feature_table,
    load_population_posteriors,
    merge_features_with_posteriors,
    resolve_feature_table,
    resolve_posterior_root,
    _numeric,
)


st.set_page_config(
    page_title="Vt Mode Interaction Surface",
    page_icon=":material/insights:",
    layout="wide",
)


DEFAULT_POPULATION = "vt"
DEFAULT_MODEL_NAME = "failure_only_with_category_interactions"
H2S_THRESHOLD_MG_L = 3.0


@st.cache_data(show_spinner=False)
def load_multivariate_artifacts(posterior_root_str: str, feature_table_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posterior_root = Path(posterior_root_str)
    feature_table = Path(feature_table_str)
    features = load_feature_table(feature_table)
    posteriors = load_population_posteriors(posterior_root)
    merged = augment_model_features(merge_features_with_posteriors(features, posteriors))
    coefficients = pd.read_csv(posterior_root / "field_mode_multivariate_coefficients.csv")
    summary_path = posterior_root / "field_mode_multivariate_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return merged, coefficients, summary


def select_model_frame(merged: pd.DataFrame, population: str, model_name: str) -> pd.DataFrame:
    frame = merged.loc[merged["population"] == population].copy()
    if "failure_only" in model_name:
        frame = frame.loc[_numeric(frame["event_model"]).eq(1)].copy()
    return frame


def load_model_coefficients(coefficients: pd.DataFrame, population: str, model_name: str) -> pd.DataFrame:
    subset = coefficients.loc[
        (coefficients["population"] == population)
        & (coefficients["model_name"] == model_name)
    ].copy()
    if subset.empty:
        raise ValueError(f"No coefficient rows found for population={population}, model={model_name}")
    return subset.reset_index(drop=True)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _mode_string(series: pd.Series) -> str:
    clean = series.fillna("<missing>").astype(str)
    if clean.empty:
        return "<missing>"
    return str(clean.mode(dropna=False).iloc[0])


def _select_or_default(label: str, options: list[str], default_value: str) -> str:
    if default_value in options:
        index = options.index(default_value)
    else:
        index = 0
    return st.selectbox(label, options=options, index=index)


def _scenario_h2s_value(model_frame: pd.DataFrame, scenario: str, custom_value: float) -> tuple[float, bool]:
    h2s = _numeric(model_frame["h2s_effective_mg_l"]) if "h2s_effective_mg_l" in model_frame.columns else pd.Series(dtype=float)
    low_mask = h2s.notna() & h2s.lt(H2S_THRESHOLD_MG_L)
    high_mask = h2s.notna() & h2s.ge(H2S_THRESHOLD_MG_L)
    low_value = float(h2s.loc[low_mask].median()) if low_mask.any() else 0.0
    high_value = float(h2s.loc[high_mask].median()) if high_mask.any() else max(low_value, H2S_THRESHOLD_MG_L)
    if scenario == "Low H2S":
        value = low_value
    elif scenario == "High H2S":
        value = high_value
    else:
        value = float(custom_value)
    return value, bool(value >= H2S_THRESHOLD_MG_L)


def build_baseline_row(
    model_frame: pd.DataFrame,
    contractor: str,
    pump_family: str,
    failure_category: str,
    h2s_scenario: str,
    custom_h2s_value: float,
) -> tuple[pd.DataFrame, int]:
    subset = model_frame.copy()
    if "contractor" in subset.columns:
        subset = subset.loc[subset["contractor"].fillna("<missing>").astype(str) == contractor]
    if "pump_family" in subset.columns:
        subset = subset.loc[subset["pump_family"].fillna("<missing>").astype(str) == pump_family]
    if "failure_category_raw" in subset.columns:
        subset = subset.loc[subset["failure_category_raw"].fillna("<missing>").astype(str) == failure_category]
    matched_rows = int(len(subset))
    if subset.empty:
        subset = model_frame.copy()

    h2s_value, high_h2s_flag = _scenario_h2s_value(model_frame, h2s_scenario, custom_h2s_value)

    baseline = pd.DataFrame(
        [
            {
                "contractor": contractor,
                "pump_family": pump_family,
                "failure_category_raw": failure_category,
                "high_h2s_flag": high_h2s_flag,
                "h2s_effective_mg_l": h2s_value,
                "mount_year": float(_numeric(subset["mount_year"]).median()),
                "avg_kpod": float(_numeric(subset["avg_kpod"]).median()),
                "freq_above_55hz_pct": float(_numeric(subset["freq_above_55hz_pct"]).median()),
                "freq_below_45hz_pct": float(_numeric(subset["freq_below_45hz_pct"]).median()),
                "freq_signed_exposure": float(_numeric(subset["freq_signed_exposure"]).median()),
                "freq_w_mean": float(_numeric(subset["freq_w_mean"]).median()),
                "tlf_per_day": float(_numeric(subset["tlf_per_day"]).median()),
                "calcium_load_per_day": float(_numeric(subset["calcium_load_per_day"]).median()),
                "chloride_load_per_day": float(_numeric(subset["chloride_load_per_day"]).median()),
                "sulfate_load_per_day": float(_numeric(subset["sulfate_load_per_day"]).median()),
                "gypsum_proxy_per_day": float(_numeric(subset["gypsum_proxy_per_day"]).median()),
            }
        ]
    )
    return baseline, matched_rows


def build_surface(
    baseline_row: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    model_frame: pd.DataFrame,
    year_min: float,
    year_max: float,
    year_points: int,
    gypsum_min: float,
    gypsum_max: float,
    gypsum_points: int,
) -> pd.DataFrame:
    year_grid = pd.Series(np.linspace(year_min, year_max, int(year_points)), dtype=float)
    gypsum_grid = pd.Series(np.geomspace(gypsum_min, gypsum_max, int(gypsum_points)), dtype=float)
    rows: list[dict[str, object]] = []
    base = baseline_row.iloc[0].to_dict()
    for mount_year in year_grid:
        for gypsum_proxy in gypsum_grid:
            row = dict(base)
            row["mount_year"] = float(mount_year)
            row["gypsum_proxy_per_day"] = float(gypsum_proxy)
            rows.append(row)
    surface = pd.DataFrame(rows)
    scored = predict_regularized_logistic_from_table(augment_model_features(surface.copy()), coefficient_table)
    surface["predicted_probability"] = scored["predicted_probability"].to_numpy(dtype=float)
    surface["linear_predictor"] = scored["linear_predictor"].to_numpy(dtype=float)
    return surface


def build_threshold_frame(surface: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, float | None]] = []
    for mount_year, subset in surface.groupby("mount_year", dropna=False):
        ordered = subset.sort_values("gypsum_proxy_per_day")
        crossed = ordered.loc[ordered["predicted_probability"] >= threshold]
        rows.append(
            {
                "mount_year": float(mount_year),
                "gypsum_proxy_at_threshold": float(crossed["gypsum_proxy_per_day"].iloc[0]) if not crossed.empty else None,
            }
        )
    return pd.DataFrame(rows)


def surface_figure(surface: pd.DataFrame, threshold_frame: pd.DataFrame, title: str) -> go.Figure:
    pivot = surface.pivot(index="gypsum_proxy_per_day", columns="mount_year", values="predicted_probability").sort_index()
    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=pivot.columns.to_numpy(dtype=float),
            y=pivot.index.to_numpy(dtype=float),
            z=pivot.to_numpy(dtype=float),
            colorscale="Viridis",
            contours={"start": 0.0, "end": 1.0, "size": 0.05, "showlabels": True},
            colorbar={"title": "P(short mode)"},
            hovertemplate="Mount year=%{x:.1f}<br>Gypsum/day=%{y:.1f}<br>P(short)=%{z:.3f}<extra></extra>",
        )
    )
    valid_threshold = threshold_frame.dropna(subset=["gypsum_proxy_at_threshold"])
    if not valid_threshold.empty:
        figure.add_trace(
            go.Scatter(
                x=valid_threshold["mount_year"],
                y=valid_threshold["gypsum_proxy_at_threshold"],
                mode="lines",
                name="50% threshold",
                line={"color": "white", "width": 3, "dash": "dash"},
                hovertemplate="Mount year=%{x:.1f}<br>Threshold gypsum/day=%{y:.1f}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        height=620,
        xaxis_title="Mount year",
        yaxis_title="Gypsum proxy per day",
        yaxis_type="log",
    )
    return figure


def cross_section_figure(surface: pd.DataFrame, year_choices: list[float]) -> go.Figure:
    figure = go.Figure()
    unique_years = sorted(surface["mount_year"].unique().tolist())
    for year in year_choices:
        nearest = min(unique_years, key=lambda value: abs(value - year))
        subset = surface.loc[surface["mount_year"] == nearest].sort_values("gypsum_proxy_per_day")
        figure.add_trace(
            go.Scatter(
                x=subset["gypsum_proxy_per_day"],
                y=subset["predicted_probability"],
                mode="lines",
                name=f"Year {nearest:.1f}",
            )
        )
    figure.update_layout(
        title="Gypsum Cross-Sections",
        height=420,
        xaxis_title="Gypsum proxy per day",
        xaxis_type="log",
        yaxis_title="Predicted short-mode probability",
        yaxis_range=[0.0, 1.0],
    )
    return figure


st.title("Vt Short-Mode Interaction Surface")
st.caption(
    "Interactive surface explorer for the latent short-life mode, using the fitted multivariate interaction model."
)

default_posterior_root = str(resolve_posterior_root(None))
default_feature_table = str(resolve_feature_table(None))

with st.sidebar:
    st.header("Data")
    posterior_root_str = st.text_input("Posterior output root", value=default_posterior_root)
    feature_table_str = st.text_input("Feature table", value=default_feature_table)

try:
    merged, coefficients, summary = load_multivariate_artifacts(posterior_root_str, feature_table_str)
except Exception as exc:  # pragma: no cover - Streamlit runtime guard
    st.error(f"Failed to load model artifacts: {exc}")
    st.stop()

population_options = sorted(coefficients["population"].dropna().astype(str).unique().tolist())
population_default = population_options.index(DEFAULT_POPULATION) if DEFAULT_POPULATION in population_options else 0

with st.sidebar:
    st.header("Model")
    population = st.selectbox("Population", options=population_options, index=population_default)
    model_options = (
        coefficients.loc[coefficients["population"] == population, "model_name"].dropna().astype(str).drop_duplicates().tolist()
    )
    model_default = model_options.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in model_options else 0
    model_name = st.selectbox("Model", options=model_options, index=model_default)

model_frame = select_model_frame(merged, population=population, model_name=model_name)
if model_frame.empty:
    st.warning("No rows are available for the selected population/model.")
    st.stop()
coefficient_table = load_model_coefficients(coefficients, population=population, model_name=model_name)

contractor_options = sorted(model_frame["contractor"].fillna("<missing>").astype(str).unique().tolist()) if "contractor" in model_frame.columns else ["<missing>"]
pump_family_options = sorted(model_frame["pump_family"].fillna("<missing>").astype(str).unique().tolist()) if "pump_family" in model_frame.columns else ["<missing>"]
if "failure_category_raw" in model_frame.columns:
    failure_category_options = sorted(model_frame["failure_category_raw"].fillna("<missing>").astype(str).unique().tolist())
else:
    failure_category_options = ["<missing>"]

with st.sidebar:
    st.header("Scenario")
    contractor = _select_or_default("Contractor", contractor_options, _mode_string(model_frame["contractor"]) if "contractor" in model_frame.columns else "<missing>")
    pump_family = _select_or_default("Pump family", pump_family_options, _mode_string(model_frame["pump_family"]) if "pump_family" in model_frame.columns else "<missing>")
    failure_category = _select_or_default("Failure category", failure_category_options, _mode_string(model_frame["failure_category_raw"]) if "failure_category_raw" in model_frame.columns else "<missing>")
    h2s_scenario = st.radio("H2S scenario", options=["Low H2S", "High H2S", "Custom mg/L"], index=0)
    custom_h2s_value = st.number_input("Custom H2S mg/L", min_value=0.0, value=3.0, step=1.0, disabled=h2s_scenario != "Custom mg/L")

baseline_row, matched_rows = build_baseline_row(
    model_frame=model_frame,
    contractor=contractor,
    pump_family=pump_family,
    failure_category=failure_category,
    h2s_scenario=h2s_scenario,
    custom_h2s_value=float(custom_h2s_value),
)

year_series = _numeric(model_frame["mount_year"]).dropna()
gypsum_series = _numeric(model_frame["gypsum_proxy_per_day"]).dropna()
gypsum_positive = gypsum_series.loc[gypsum_series > 0.0]
if year_series.empty or gypsum_positive.empty:
    st.error("The selected model frame does not have enough mount_year or positive gypsum data to build a surface.")
    st.stop()

default_year_min = int(year_series.min())
default_year_max = int(year_series.max())
default_gypsum_min = float(gypsum_positive.quantile(0.10))
default_gypsum_max = float(gypsum_positive.quantile(0.95))
default_gypsum_min = max(default_gypsum_min, float(gypsum_positive.min()))
default_gypsum_max = max(default_gypsum_max, default_gypsum_min * 1.01)

with st.sidebar:
    st.header("Grid")
    year_min, year_max = st.slider("Mount year range", min_value=default_year_min, max_value=default_year_max, value=(default_year_min, default_year_max))
    year_points = st.slider("Year grid points", min_value=30, max_value=180, value=100, step=10)
    gypsum_min = st.number_input("Gypsum min", min_value=float(gypsum_positive.min()), value=float(default_gypsum_min), step=float(max(default_gypsum_min * 0.1, 1.0)), format="%.3f")
    gypsum_max = st.number_input("Gypsum max", min_value=float(gypsum_min * 1.01), value=float(default_gypsum_max), step=float(max(default_gypsum_max * 0.1, 10.0)), format="%.3f")
    gypsum_points = st.slider("Gypsum grid points", min_value=30, max_value=180, value=100, step=10)

surface = build_surface(
    baseline_row=baseline_row,
    coefficient_table=coefficient_table,
    model_frame=model_frame,
    year_min=float(year_min),
    year_max=float(year_max),
    year_points=int(year_points),
    gypsum_min=float(gypsum_min),
    gypsum_max=float(gypsum_max),
    gypsum_points=int(gypsum_points),
)
threshold_frame = build_threshold_frame(surface, threshold=0.5)

year_choices = [float(year_min), float((year_min + year_max) / 2.0), float(year_max)]
top_metrics = st.columns(5)
top_metrics[0].metric("Rows in model frame", f"{len(model_frame):,}")
top_metrics[1].metric("Matched rows for baseline", f"{matched_rows:,}")
top_metrics[2].metric("Min P(short)", f"{surface['predicted_probability'].min():.3f}")
top_metrics[3].metric("Median P(short)", f"{surface['predicted_probability'].median():.3f}")
top_metrics[4].metric("Max P(short)", f"{surface['predicted_probability'].max():.3f}")

tab_surface, tab_baseline, tab_downloads = st.tabs(["Surface", "Baseline", "Downloads"])

with tab_surface:
    st.plotly_chart(
        surface_figure(
            surface,
            threshold_frame,
            title=f"{population.upper()} | {model_name} | {h2s_scenario}",
        ),
        width="stretch",
    )
    st.plotly_chart(cross_section_figure(surface, year_choices), width="stretch")
    st.dataframe(threshold_frame.head(25), width="stretch")

with tab_baseline:
    st.subheader("Baseline Covariates Used")
    st.dataframe(baseline_row, width="stretch")
    st.caption(
        "The app uses medians from the matching modeled subset when available. "
        "If the exact contractor/pump/failure-category combination is absent, it falls back to the full selected model frame."
    )
    if not summary.empty:
        summary_subset = summary.loc[
            (summary["population"] == population)
            & (summary["model_name"] == model_name)
        ].copy()
        if not summary_subset.empty:
            st.subheader("Model Fit Summary")
            st.dataframe(summary_subset, width="stretch")

with tab_downloads:
    st.download_button(
        "Download surface CSV",
        data=_csv_bytes(surface),
        file_name=f"{population}_{model_name}_surface.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download threshold CSV",
        data=_csv_bytes(threshold_frame),
        file_name=f"{population}_{model_name}_thresholds.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download baseline CSV",
        data=_csv_bytes(baseline_row),
        file_name=f"{population}_{model_name}_baseline.csv",
        mime="text/csv",
    )
