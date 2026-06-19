from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for path_text in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from scripts.db import WAREHOUSE_PATH, get_warehouse_conn, get_last_pipeline_run


st.set_page_config(
    page_title="TTF vs Frequency Exposure",
    page_icon=":material/show_chart:",
    layout="wide",
)

FREQ_THRESHOLD_HZ = 55.0
LOW_FREQ_THRESHOLD_HZ = 45.0
MIN_FREQ_DAYS = 7
DEFAULT_BOUNDARIES_TEXT = "-0.5, -0.2, 0, 0.2, 0.5"
DEFAULT_GLF_BOUNDARIES_TEXT = "100, 300, 500, 1000"
DEFAULT_KPOD_BOUNDARIES_TEXT = "0.3, 0.5, 0.7, 1.0"
DEFAULT_KPOD_FREQ_BOUNDARIES_TEXT = "0.3, 0.5, 0.7, 1.0"
DEFAULT_PRECIPITATE_PROXY_BOUNDARIES_TEXT = "1000, 5000, 10000, 50000"


def parse_boundaries(text: str) -> list[float]:
    parts = [item.strip() for item in str(text).split(",")]
    values: list[float] = []
    for part in parts:
        if not part:
            continue
        values.append(float(part))
    unique_sorted = sorted(set(values))
    if len(unique_sorted) < 2:
        raise ValueError("Provide at least two boundary values.")
    return unique_sorted


def build_bin_labels(boundaries: list[float], symbol: str = "y") -> list[str]:
    labels = [f"{symbol} <= {boundaries[0]:.3f}"]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
        labels.append(f"{left:.3f} < {symbol} <= {right:.3f}")
    labels.append(f"{symbol} > {boundaries[-1]:.3f}")
    return labels


def assign_bin(exposure: float, boundaries: list[float], symbol: str = "y") -> str:
    labels = build_bin_labels(boundaries, symbol=symbol)
    if pd.isna(exposure):
        return "<missing>"
    if exposure <= boundaries[0]:
        return labels[0]
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False), start=1):
        if left < exposure <= right:
            return labels[index]
    return labels[-1]


def palette_for_labels(labels: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Dark24
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    event_times = np.unique(durations[events == 1])
    if len(event_times) == 0:
        return np.array([0.0]), np.array([1.0])

    times = np.concatenate([[0.0], event_times])
    survival = np.ones(len(times), dtype=float)
    current = 1.0
    for index, current_time in enumerate(event_times, start=1):
        at_risk = int(np.sum(durations >= current_time))
        n_events = int(np.sum((durations == current_time) & (events == 1)))
        if at_risk > 0:
            current *= 1.0 - (n_events / at_risk)
        survival[index] = current
    return times, survival


@st.cache_resource
def _warehouse_conn():
    """Cached warehouse connection (one per Streamlit session)."""
    if not WAREHOUSE_PATH.exists():
        return None
    return get_warehouse_conn()


@st.cache_data(show_spinner=False, ttl=3600)
def load_run_stats(field_code: str) -> pd.DataFrame:
    """Load pre-computed run stats from mart__vt_freq55 in the warehouse.

    Falls back to an empty DataFrame if the warehouse is not built yet.
    Column names are normalized to the legacy names used by the rest of this app.
    """
    conn = _warehouse_conn()
    if conn is None:
        st.warning("Warehouse not found. Run `python scripts/pipeline.py` to build it.")
        return pd.DataFrame()

    try:
        df = pd.read_sql("SELECT * FROM mart__vt_freq55", conn, parse_dates=["install_date", "stop_date"])
    except Exception as exc:
        st.warning(f"mart__vt_freq55 not available yet: {exc}. Run the pipeline first.")
        return pd.DataFrame()

    if field_code != "GLOBAL":
        df = df.loc[df["field"].astype("string").str.strip() == field_code].copy()

    # Normalize warehouse column names → legacy names used by the rest of this file.
    # avg_glf and avg_kpod come directly from feat__run_freq_exposure (whole-run means).
    # kpod_m_mean and glf_m_mean from feat__run_mature are kept under their original names
    # for reference but are NOT mapped to avg_glf/avg_kpod.
    df = df.rename(columns={
        "well": "well_id",
        "install_date": "mount_date",
        "run_days": "ttf_days",
        "freq_above_55hz_pct": "frac_freq_above_55",
        "freq_below_45hz_pct": "frac_freq_below_45",
        "freq_signed_exposure": "signed_freq_exposure",
        "freq_w_mean": "avg_freq_hz",
        "n_freq_valid_days": "n_freq_days",
        "n_freq_above_55hz": "n_freq_days_above_55",
        "n_freq_below_45hz": "n_freq_days_below_45",
        "total_liquid_m3": "tlf_total_liquid_m3",
        "total_freq_hz_days": "trf_total_frequency_hz_days",
        "cum_salt_load_kg": "salt_proxy_total_kg",
    })

    # avg_kpod_freq not stored in mart; leave as NaN placeholder.
    if "avg_kpod_freq" not in df.columns:
        df["avg_kpod_freq"] = float("nan")

    # salt_proxy_total_kg fallback: use integrated_salt_proxy_m if cum column missing.
    if "salt_proxy_total_kg" not in df.columns and "integrated_salt_proxy_m" in df.columns:
        df["salt_proxy_total_kg"] = df["integrated_salt_proxy_m"]

    if "mount_date" in df.columns:
        df["mount_year"] = pd.to_datetime(df["mount_date"], errors="coerce").dt.year.astype("Int64")

    return df.reset_index(drop=True)


def filtered_view(
    df: pd.DataFrame,
    infant_days: int,
    year0: int,
    boundaries: list[float],
    glf_boundaries: list[float],
    kpod_boundaries: list[float],
    kpod_freq_boundaries: list[float],
    salt_proxy_boundaries: list[float],
    selected_failed_nodes: list[str],
    selected_glf_bins: list[str],
    selected_kpod_bins: list[str],
    selected_kpod_freq_bins: list[str],
    ttf_column: str,
) -> pd.DataFrame:
    filtered = df.loc[
        (df["mount_year"] > year0)
        & (df[ttf_column].notna())
        & (df[ttf_column] >= infant_days)
        & (df["signed_freq_exposure"].notna())
    ].copy()
    filtered["event_label"] = np.where(filtered["event"].eq(1), "Failure", "Censored")
    filtered["freq_bin"] = filtered["signed_freq_exposure"].apply(lambda value: assign_bin(value, boundaries, symbol="y"))
    filtered["glf_bin"] = filtered["avg_glf"].apply(lambda value: assign_bin(value, glf_boundaries, symbol="GLF") if pd.notna(value) else "<missing>")
    filtered["kpod_bin"] = filtered["avg_kpod"].apply(lambda value: assign_bin(value, kpod_boundaries, symbol="Kpod") if pd.notna(value) else "<missing>")
    filtered["kpod_freq_bin"] = filtered["avg_kpod_freq"].apply(lambda value: assign_bin(value, kpod_freq_boundaries, symbol="Kpod_freq") if pd.notna(value) else "<missing>")
    filtered["salt_proxy_bin"] = filtered["salt_proxy_total_kg"].apply(lambda value: assign_bin(value, salt_proxy_boundaries, symbol="Salt") if pd.notna(value) else "<missing>")
    filtered["failed_node_label"] = filtered["failed_node"].astype("string").fillna("<missing>")
    if selected_failed_nodes:
        filtered = filtered.loc[filtered["failed_node_label"].isin(selected_failed_nodes)].copy()
    if selected_glf_bins:
        filtered = filtered.loc[filtered["glf_bin"].isin(selected_glf_bins)].copy()
    if selected_kpod_bins:
        filtered = filtered.loc[filtered["kpod_bin"].isin(selected_kpod_bins)].copy()
    if selected_kpod_freq_bins:
        filtered = filtered.loc[filtered["kpod_freq_bin"].isin(selected_kpod_freq_bins)].copy()
    return filtered


def color_group_frame(df: pd.DataFrame, color_mode: str) -> tuple[pd.DataFrame, str]:
    plotting = df.copy().reset_index(drop=True)
    if color_mode == "Mount year":
        plotting["color_group"] = plotting["mount_year"].astype(str)
    elif color_mode == "Y exposure bin":
        plotting["color_group"] = plotting["freq_bin"].astype(str)
    elif color_mode == "GLF bin":
        plotting["color_group"] = plotting["glf_bin"].astype(str)
    elif color_mode == "Kpod bin":
        plotting["color_group"] = plotting["kpod_bin"].astype(str)
    elif color_mode == "Precipitate proxy bin":
        plotting["color_group"] = plotting["salt_proxy_bin"].astype(str)
    else:
        plotting["color_group"] = plotting["kpod_freq_bin"].astype(str)
    return plotting, "color_group"


def ordered_group_labels(plotting: pd.DataFrame, color_column: str, color_mode: str, boundaries: list[float] | None = None) -> list[str]:
    if color_mode == "Y exposure bin" and boundaries is not None:
        desired = build_bin_labels(boundaries, symbol="y")
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Precipitate proxy bin" and boundaries is not None:
        desired = build_bin_labels(boundaries, symbol="Salt")
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    return sorted(plotting[color_column].dropna().astype(str).unique().tolist())


def x_axis_config(x_mode: str, ttf_column: str, ttf_label: str) -> tuple[str, str]:
    if x_mode == "TRF":
        return "trf_total_frequency_hz_days", "TRF / cumulative frequency to failure (Hz-days)"
    if x_mode == "TLF":
        return "tlf_total_liquid_m3", "TLF / cumulative liquid to failure (m3)"
    return ttf_column, ttf_label


def scatter_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], color_mode: str, x_mode: str, ttf_column: str, ttf_label: str) -> go.Figure:
    plotting, color_column = color_group_frame(df, color_mode)
    plotting["marker_symbol"] = np.where(plotting["event"].eq(1), "x", "circle")
    plotting["marker_size"] = np.where(plotting["event"].eq(1), 11, 7)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label)
    fig = go.Figure()
    groups = ordered_group_labels(plotting, color_column, color_mode, boundaries)
    color_map = palette_for_labels(groups)
    for group in groups:
        sub = plotting.loc[plotting[color_column].astype(str) == str(group)].copy()
        fig.add_trace(
            go.Scattergl(
                x=sub[x_column],
                y=sub["signed_freq_exposure"],
                mode="markers",
                name=str(group),
                marker={
                    "color": color_map[str(group)],
                    "size": sub["marker_size"],
                    "symbol": sub["marker_symbol"],
                    "line": {"width": 0.6, "color": "black"},
                    "opacity": 0.8,
                },
                customdata=sub[
                    [
                        "well_id",
                        "contractor",
                        "mount_date",
                        "stop_date",
                        "event_label",
                        "n_freq_days",
                        "n_freq_days_above_55",
                        "n_freq_days_below_45",
                        "frac_freq_above_55",
                        "frac_freq_below_45",
                        "salt_proxy_total_kg",
                    ]
                ].astype(str).to_numpy(),
                hovertemplate=(
                    "Well: %{customdata[0]}<br>"
                    "Contractor: %{customdata[1]}<br>"
                    "Mount: %{customdata[2]}<br>"
                    "Stop: %{customdata[3]}<br>"
                    "Outcome: %{customdata[4]}<br>"
                    f"{x_mode}: " + "%{x:.1f}<br>"
                    "Signed exposure y: %{y:.3f}<br>"
                    "Share >55 Hz: %{customdata[8]}<br>"
                    "Share <45 Hz: %{customdata[9]}<br>"
                    "Freq days: %{customdata[5]}<br>"
                    "Days >55 Hz: %{customdata[6]}<br>"
                    "Days <45 Hz: %{customdata[7]}<br>"
                    "Salt proxy total: %{customdata[10]} kg<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{field_label}: TTF vs signed frequency exposure",
        xaxis_title=x_label,
        yaxis_title="y = share(time > 55 Hz) - share(time < 45 Hz)",
        height=520,
        legend_title=color_mode,
    )
    fig.update_yaxes(range=[-1.03, 1.03], zeroline=True, zerolinewidth=1.2)
    for boundary in boundaries:
        fig.add_hline(y=float(boundary), line_dash="dot", line_color="#999")
    return fig


def histogram_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], group_mode: str, x_mode: str, ttf_column: str, ttf_label: str) -> go.Figure:
    fig = go.Figure()
    failures = df.loc[df["event"].eq(1)].copy()
    if failures.empty:
        return fig
    failures, color_column = color_group_frame(failures, group_mode)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label)
    max_value = float(failures[x_column].max())
    if x_mode in {"TLF", "TRF"}:
        step = max(1000.0, round(max_value / 30.0, -2))
        bins = np.arange(0.0, max(step * 2.0, max_value + step), step)
    else:
        bins = np.arange(0.0, max(210.0, min(max_value + 30.0, 1800.0)), 30.0)
    labels = ordered_group_labels(failures, color_column, group_mode, boundaries)
    color_map = palette_for_labels(labels)
    for label in labels:
        sub = failures.loc[failures[color_column].astype(str) == str(label), x_column]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=f"{label} (n={len(sub)})",
                marker_color=color_map[str(label)],
                opacity=0.65,
                xbins={"start": float(bins.min()), "end": float(bins.max()), "size": float(bins[1] - bins[0]) if len(bins) > 1 else 1.0},
                hovertemplate=f"{x_mode} bin: " + "%{x}<br>Failures: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=f"{field_label}: failure-event histogram grouped by {group_mode}",
        xaxis_title=x_label,
        yaxis_title="Failure count",
        height=460,
        legend_title=group_mode,
    )
    return fig


def survival_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], group_mode: str, x_mode: str, ttf_column: str, ttf_label: str) -> go.Figure:
    fig = go.Figure()
    plotting, color_column = color_group_frame(df, group_mode)
    labels = ordered_group_labels(plotting, color_column, group_mode, boundaries)
    color_map = palette_for_labels(labels)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label)
    for label in labels:
        sub = plotting.loc[plotting[color_column].astype(str) == str(label)].copy()
        if len(sub) < 3:
            continue
        t = sub[x_column].to_numpy(dtype=float)
        e = sub["event"].to_numpy(dtype=float)
        km_t, km_s = kaplan_meier(t, e)
        fig.add_trace(
            go.Scatter(
                x=km_t,
                y=km_s,
                mode="lines",
                line={"shape": "hv", "width": 2.5, "color": color_map[str(label)]},
                name=f"{label} (n={len(sub)}, events={int(e.sum())})",
                hovertemplate="Time: %{x:.0f} d<br>Survival: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{field_label}: Kaplan-Meier survival grouped by {group_mode}",
        xaxis_title=x_label,
        yaxis_title="Survival probability",
        height=460,
        legend_title=group_mode,
    )
    fig.update_yaxes(range=[0.0, 1.02])
    return fig


def summary_table(df: pd.DataFrame, group_mode: str, ttf_column: str, ttf_label: str) -> pd.DataFrame:
    plotting, color_column = color_group_frame(df, group_mode)
    rows: list[dict[str, object]] = []
    for label in sorted(plotting[color_column].dropna().astype(str).unique().tolist()):
        sub = plotting.loc[plotting[color_column].astype(str) == str(label)].copy()
        failures = sub.loc[sub["event"].eq(1)]
        rows.append(
            {
                group_mode: label,
                "Runs": int(len(sub)),
                "Failures": int(len(failures)),
                "Censored": int(sub["event"].eq(0).sum()),
                f"Median {ttf_label} all": None if sub.empty else round(float(sub[ttf_column].median()), 1),
                f"Median {ttf_label} failures": None if failures.empty else round(float(failures[ttf_column].median()), 1),
                "Mean y": None if sub.empty else round(float(sub["signed_freq_exposure"].mean()), 3),
                "Mean share >55 Hz": None if sub.empty else round(float(sub["frac_freq_above_55"].mean()), 3),
                "Mean share <45 Hz": None if sub.empty else round(float(sub["frac_freq_below_45"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def correlation_figure(df: pd.DataFrame, field_label: str) -> tuple[go.Figure, float | None, int]:
    subset = df.loc[df["avg_freq_hz"].notna() & df["avg_kpod_freq"].notna()].copy()
    if subset.empty:
        return go.Figure(), None, 0
    corr = float(subset["avg_freq_hz"].corr(subset["avg_kpod_freq"])) if len(subset) >= 2 else None
    fig = px.scatter(
        subset,
        x="avg_freq_hz",
        y="avg_kpod_freq",
        color="mount_year",
        symbol="event_label",
        hover_data=["well_id", "contractor", "ttf_days", "avg_kpod", "frac_freq_above_55", "frac_freq_below_45"],
        title=f"{field_label}: average Kpod_freq vs average frequency",
        labels={
            "avg_freq_hz": "Average frequency over run (Hz)",
            "avg_kpod_freq": "Average Kpod_freq over run",
            "mount_year": "Mount year",
            "event_label": "Outcome",
        },
        height=460,
    )
    return fig, corr, int(len(subset))


def pair_correlation_figure(
    df: pd.DataFrame,
    field_label: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    color_mode: str,
    swap_axes: bool,
) -> tuple[go.Figure, float | None, int]:
    subset = df.loc[df[x_column].notna() & df[y_column].notna()].copy()
    if subset.empty:
        return go.Figure(), None, 0
    corr = float(subset[x_column].corr(subset[y_column])) if len(subset) >= 2 else None
    plotting, color_column = color_group_frame(subset, color_mode)
    actual_x = y_column if swap_axes else x_column
    actual_y = x_column if swap_axes else y_column
    actual_x_label = y_label if swap_axes else x_label
    actual_y_label = x_label if swap_axes else y_label
    fig = px.scatter(
        plotting,
        x=actual_x,
        y=actual_y,
        color=color_column,
        symbol="event_label",
        symbol_map={"Failure": "circle", "Censored": "x"},
        hover_data=["well_id", "contractor"],
        title=f"{field_label}: {title}",
        labels={
            actual_x: actual_x_label,
            actual_y: actual_y_label,
            color_column: color_mode,
            "event_label": "Outcome",
        },
        height=420,
    )

    if x_column == "trf_total_frequency_hz_days" and y_column in {"ttf_days", "ttf_true_best_days"}:
        trf_max = float(subset["trf_total_frequency_hz_days"].max())
        ttf_max = float(subset[y_column].max())
        ttf_values = np.linspace(0.0, max(ttf_max, 1.0), 120)
        reference_specs = [
            (40.0, "#4c78a8"),
            (50.0, "#f58518"),
            (60.0, "#54a24b"),
        ]
        for frequency_hz, color in reference_specs:
            trf_values = frequency_hz * ttf_values
            visible_mask = trf_values <= (trf_max * 1.05)
            if not np.any(visible_mask):
                visible_mask = np.ones_like(trf_values, dtype=bool)
            x_values = ttf_values[visible_mask] if swap_axes else trf_values[visible_mask]
            y_values = trf_values[visible_mask] if swap_axes else ttf_values[visible_mask]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{int(frequency_hz)} Hz reference",
                    line={"color": color, "width": 2.0, "dash": "dash"},
                    hovertemplate=(
                        f"{int(frequency_hz)} Hz reference<br>"
                        + f"{actual_x_label}: %{{x:.1f}}<br>"
                        + f"{actual_y_label}: %{{y:.1f}}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )
    return fig, corr, int(len(subset))


def ttf_mode_config(ttf_mode: str) -> tuple[str, str]:
    if ttf_mode == "TTF_true_best":
        return "ttf_true_best_days", "TTF_true_best (operating days)"
    return "ttf_days", "TTF (days)"


def main() -> None:
    st.title("TTF vs Signed Frequency Exposure")
    st.caption("Simple local GUI for checking frequency-exposure histograms globally or by field.")
    st.info("Signed exposure is defined as `y = share(time > 55 Hz) - share(time < 45 Hz)`. Positive y means more high-frequency exposure; negative y means more low-frequency exposure.")

    # Show warehouse freshness in the sidebar.
    conn = _warehouse_conn()
    if conn is not None:
        last = get_last_pipeline_run("mart__vt_freq55", conn=conn)
        if last:
            st.sidebar.caption(f"Data last built: {last['run_at']} ({last['row_count']:,} runs)")
        else:
            st.sidebar.warning("Warehouse exists but mart__vt_freq55 not built yet. Run `python scripts/pipeline.py`.")
    else:
        st.sidebar.error("Warehouse not found. Run `python scripts/pipeline.py --step ingest` to start.")

    # Load field list from warehouse for the filter dropdown.
    fields: list[str] = []
    year_min, year_max = 2010, 2025
    if conn is not None:
        try:
            meta = pd.read_sql(
                "SELECT DISTINCT field, strftime('%Y', install_date) AS yr FROM raw__v03_runs WHERE field IS NOT NULL",
                conn,
            )
            fields = sorted(meta["field"].dropna().astype(str).str.strip().unique().tolist())
            valid_years = pd.to_numeric(meta["yr"], errors="coerce").dropna()
            if not valid_years.empty:
                year_min = int(valid_years.min())
                year_max = int(valid_years.max())
        except Exception:
            pass

    with st.sidebar:
        st.subheader("Filters")
        field_choice = st.selectbox("Field", ["GLOBAL", *fields], index=0)
        infant_days = st.slider("Infant mortality threshold (days)", min_value=0, max_value=180, value=30, step=5)
        year0 = st.slider("Use runs with mount year > year0", min_value=year_min, max_value=max(year_min, year_max - 1), value=max(2023, year_min), step=1)
        st.subheader("Y Bin Boundaries")
        boundaries_text = st.text_input("Boundary list", value=DEFAULT_BOUNDARIES_TEXT, help="Comma-separated signed y boundaries, for example: -0.5, -0.2, 0, 0.2, 0.5")
        st.subheader("GLF Filters")
        glf_boundaries_text = st.text_input("GLF boundaries", value=DEFAULT_GLF_BOUNDARIES_TEXT)
        st.subheader("Kpod Filters")
        kpod_boundaries_text = st.text_input("Kpod boundaries", value=DEFAULT_KPOD_BOUNDARIES_TEXT)
        st.subheader("Kpod_freq Filters")
        kpod_freq_boundaries_text = st.text_input("Kpod_freq boundaries", value=DEFAULT_KPOD_FREQ_BOUNDARIES_TEXT)
        st.subheader("Precipitate Proxy")
        salt_proxy_boundaries_text = st.text_input("Precipitate proxy boundaries", value=DEFAULT_PRECIPITATE_PROXY_BOUNDARIES_TEXT)
        ttf_mode = st.selectbox("TTF definition", ["TTF", "TTF_true_best"], index=0, help="TTF uses calendar run days from the workbook. TTF_true_best uses operational days from techregime status, falling back to telemetry qliq > 0.")
        x_mode = st.selectbox("Plot x-axis", ["TTF", "TLF", "TRF"], index=0, help="TTF = time to failure in days. TLF = cumulative liquid to failure. TRF = cumulative frequency to failure.")
        color_mode = st.selectbox("Color scatter by", ["Mount year", "Y exposure bin", "GLF bin", "Kpod bin", "Kpod_freq bin", "Precipitate proxy bin"], index=0)
        group_mode = st.selectbox("Group histogram / survival by", ["Y exposure bin", "Mount year", "GLF bin", "Kpod bin", "Kpod_freq bin", "Precipitate proxy bin"], index=0)
        swap_pair_axes = st.checkbox("Swap X-Y on TTF pair correlations", value=False)

    try:
        boundaries = parse_boundaries(boundaries_text)
        glf_boundaries = parse_boundaries(glf_boundaries_text)
        kpod_boundaries = parse_boundaries(kpod_boundaries_text)
        kpod_freq_boundaries = parse_boundaries(kpod_freq_boundaries_text)
        salt_proxy_boundaries = parse_boundaries(salt_proxy_boundaries_text)
    except Exception as exc:
        st.error(f"Could not parse boundary list: {exc}")
        return

    with st.spinner("Loading frequency-exposure dataset..."):
        stats = load_run_stats(field_choice)
    ttf_column, ttf_label = ttf_mode_config(ttf_mode)

    failed_node_labels = sorted(stats["failed_node"].astype("string").fillna("<missing>").unique().tolist())
    glf_labels = build_bin_labels(glf_boundaries, symbol="GLF")
    kpod_labels = build_bin_labels(kpod_boundaries, symbol="Kpod")
    kpod_freq_labels = build_bin_labels(kpod_freq_boundaries, symbol="Kpod_freq")

    with st.sidebar:
        selected_failed_nodes = st.multiselect("Keep failed nodes", failed_node_labels, default=failed_node_labels)
        glf_options = glf_labels + ["<missing>"]
        selected_glf_bins = st.multiselect("Keep GLF bins", glf_options, default=glf_options)
        kpod_options = kpod_labels + ["<missing>"]
        selected_kpod_bins = st.multiselect("Keep Kpod bins", kpod_options, default=kpod_options)
        kpod_freq_options = kpod_freq_labels + ["<missing>"]
        selected_kpod_freq_bins = st.multiselect("Keep Kpod_freq bins", kpod_freq_options, default=kpod_freq_options)

    view = filtered_view(
        stats,
        infant_days=infant_days,
        year0=year0,
        boundaries=boundaries,
        glf_boundaries=glf_boundaries,
        kpod_boundaries=kpod_boundaries,
        kpod_freq_boundaries=kpod_freq_boundaries,
        salt_proxy_boundaries=salt_proxy_boundaries,
        selected_failed_nodes=selected_failed_nodes,
        selected_glf_bins=selected_glf_bins,
        selected_kpod_bins=selected_kpod_bins,
        selected_kpod_freq_bins=selected_kpod_freq_bins,
        ttf_column=ttf_column,
    )

    field_label = "Global" if field_choice == "GLOBAL" else f"Field={field_choice}"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Runs", len(view))
    k2.metric("Failures", int(view["event"].eq(1).sum()))
    k3.metric("Usable wells", int(view["well_id"].nunique()))
    if x_mode == "TRF":
        median_value = "n/a" if view.empty else f"{float(view['trf_total_frequency_hz_days'].median()):.0f} Hz-d"
        k4.metric("Median TRF", median_value)
    elif x_mode == "TLF":
        median_value = "n/a" if view.empty else f"{float(view['tlf_total_liquid_m3'].median()):.0f} m3"
        k4.metric("Median TLF", median_value)
    else:
        k4.metric(f"Median {ttf_mode}", "n/a" if view.empty else f"{float(view[ttf_column].median()):.0f} d")

    if view.empty:
        st.warning("No rows remain after the selected filters.")
        return

    st.plotly_chart(scatter_figure(view, field_label, boundaries, color_mode, x_mode, ttf_column, ttf_label), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(histogram_figure(view, field_label, boundaries, group_mode, x_mode, ttf_column, ttf_label), use_container_width=True)
    with col2:
        st.plotly_chart(survival_figure(view, field_label, boundaries, group_mode, x_mode, ttf_column, ttf_label), use_container_width=True)

    st.subheader("Average Kpod_freq vs Average Frequency")
    corr_fig, corr_value, corr_rows = correlation_figure(view, field_label)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(corr_fig, use_container_width=True)
    with c2:
        st.metric("Rows in correlation", corr_rows)
        st.metric("Pearson r", "n/a" if corr_value is None or np.isnan(corr_value) else f"{corr_value:.3f}")
        st.caption("Computed on the currently filtered runs using run-average frequency and run-average Kpod_freq.")

    st.subheader("TTF Pair Correlations")
    tlf_fig, tlf_corr, tlf_rows = pair_correlation_figure(
        view,
        field_label,
        x_column="tlf_total_liquid_m3",
        y_column=ttf_column,
        x_label="TLF / cumulative liquid to failure (m3)",
        y_label=ttf_label,
        title=f"TLF vs {ttf_mode}",
        color_mode=color_mode,
        swap_axes=swap_pair_axes,
    )
    trf_fig, trf_corr, trf_rows = pair_correlation_figure(
        view,
        field_label,
        x_column="trf_total_frequency_hz_days",
        y_column=ttf_column,
        x_label="TRF / cumulative frequency to failure (Hz-days)",
        y_label=ttf_label,
        title=f"TRF vs {ttf_mode}",
        color_mode=color_mode,
        swap_axes=swap_pair_axes,
    )
    proxy_ttf_fig, proxy_ttf_corr, proxy_ttf_rows = pair_correlation_figure(
        view,
        field_label,
        x_column="salt_proxy_total_kg",
        y_column=ttf_column,
        x_label="Precipitate proxy total (kg)",
        y_label=ttf_label,
        title=f"{ttf_mode} vs precipitate proxy",
        color_mode=color_mode,
        swap_axes=swap_pair_axes,
    )
    proxy_trf_fig, proxy_trf_corr, proxy_trf_rows = pair_correlation_figure(
        view,
        field_label,
        x_column="salt_proxy_total_kg",
        y_column="trf_total_frequency_hz_days",
        x_label="Precipitate proxy total (kg)",
        y_label="TRF / cumulative frequency to failure (Hz-days)",
        title="TRF vs precipitate proxy",
        color_mode=color_mode,
        swap_axes=swap_pair_axes,
    )
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(tlf_fig, use_container_width=True)
        st.caption(f"TLF vs {ttf_mode}: rows={tlf_rows}, Pearson r={'n/a' if tlf_corr is None or np.isnan(tlf_corr) else f'{tlf_corr:.3f}'}")
    with p2:
        st.plotly_chart(trf_fig, use_container_width=True)
        st.caption(f"TRF vs {ttf_mode}: rows={trf_rows}, Pearson r={'n/a' if trf_corr is None or np.isnan(trf_corr) else f'{trf_corr:.3f}'}")

    p3, p4 = st.columns(2)
    with p3:
        st.plotly_chart(proxy_ttf_fig, use_container_width=True)
        st.caption(
            f"{ttf_mode} vs precipitate proxy: "
            + f"rows={proxy_ttf_rows}, Pearson r={'n/a' if proxy_ttf_corr is None or np.isnan(proxy_ttf_corr) else f'{proxy_ttf_corr:.3f}'}"
        )
    with p4:
        st.plotly_chart(proxy_trf_fig, use_container_width=True)
        st.caption(
            "TRF vs precipitate proxy: "
            + f"rows={proxy_trf_rows}, Pearson r={'n/a' if proxy_trf_corr is None or np.isnan(proxy_trf_corr) else f'{proxy_trf_corr:.3f}'}"
        )

    st.subheader("Grouped Summary")
    st.dataframe(summary_table(view, group_mode, ttf_column, ttf_label), use_container_width=True, height=220)

    csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download filtered dataset CSV",
        data=csv_bytes,
        file_name=f"freq55_{field_choice.lower()}_year_gt_{year0}_infant_{infant_days}d.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
