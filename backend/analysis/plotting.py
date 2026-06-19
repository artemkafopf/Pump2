from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .weibull_model import WeibullStressFitResult


def _curve_frame_for_parameters(
    group_df: pd.DataFrame,
    beta: float,
    eta: float,
    duration_column: str,
    num_points: int = 200,
) -> pd.DataFrame:
    max_duration = float(max(group_df[duration_column].max(), eta * 3.0))
    times = np.linspace(1e-6, max_duration * 1.1, num_points)
    cdf = 1.0 - np.exp(-np.power(times / eta, beta))
    survival = 1.0 - cdf
    pdf = (beta / eta) * np.power(times / eta, beta - 1.0) * np.exp(-np.power(times / eta, beta))
    return pd.DataFrame({"time": times, "pdf": pdf, "survival": survival, "cdf": cdf})


def _kaplan_meier_frame(group_df: pd.DataFrame, duration_column: str, event_column: str) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame(columns=["time", "survival"])

    ordered = group_df[[duration_column, event_column]].copy()
    ordered[duration_column] = pd.to_numeric(ordered[duration_column], errors="coerce")
    ordered[event_column] = pd.to_numeric(ordered[event_column], errors="coerce")
    ordered = ordered.dropna().sort_values(duration_column)
    if ordered.empty:
        return pd.DataFrame(columns=["time", "survival"])

    rows: list[dict[str, float]] = []
    survival = 1.0
    unique_times = ordered[duration_column].drop_duplicates().tolist()
    for time in unique_times:
        at_risk = int((ordered[duration_column] >= time).sum())
        failed = int(((ordered[duration_column] == time) & (ordered[event_column] == 1)).sum())
        if at_risk > 0 and failed > 0:
            survival *= 1.0 - (failed / at_risk)
        rows.append({"time": float(time), "survival": float(survival)})
    return pd.DataFrame(rows)


def _empirical_bin_frame(group_df: pd.DataFrame, duration_column: str, event_column: str) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame(columns=["left", "right", "mid", "survival", "failure"])

    durations = pd.to_numeric(group_df[duration_column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) < 2:
        return pd.DataFrame(columns=["left", "right", "mid", "survival", "failure"])

    max_duration = float(np.max(durations))
    upper_edge = max(30.0, np.ceil(max_duration / 30.0) * 30.0)
    edges = np.arange(0.0, upper_edge + 30.0, 30.0, dtype=float)
    if len(edges) < 2:
        return pd.DataFrame(columns=["left", "right", "mid", "survival", "failure"])

    km = _kaplan_meier_frame(group_df, duration_column, event_column)
    if km.empty:
        return pd.DataFrame(columns=["left", "right", "mid", "survival", "failure"])

    durations_series = pd.to_numeric(group_df[duration_column], errors="coerce")
    total_rows = int(durations_series.notna().sum())
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        km_slice = km.loc[km["time"] <= right]
        survival = float(km_slice["survival"].iloc[-1]) if not km_slice.empty else 1.0
        wells_remaining = int((durations_series > right).sum())
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "mid": float((left + right) / 2.0),
                "width": float(right - left),
                "survival": survival,
                "failure": float(1.0 - survival),
                "wells_remaining": wells_remaining,
                "initial_wells": total_rows,
            }
        )
    return pd.DataFrame(rows)


def _failure_histogram_frame(group_df: pd.DataFrame, duration_column: str, event_column: str) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame(columns=["left", "right", "mid", "width", "count", "density"])

    failures = group_df.loc[pd.to_numeric(group_df[event_column], errors="coerce") == 1].copy()
    durations = pd.to_numeric(failures[duration_column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) == 0:
        return pd.DataFrame(columns=["left", "right", "mid", "width", "count", "density"])

    max_duration = float(np.max(durations))
    upper_edge = max(30.0, np.ceil(max_duration / 30.0) * 30.0)
    edges = np.arange(0.0, upper_edge + 30.0, 30.0, dtype=float)
    counts, edges = np.histogram(durations, bins=edges)
    total = int(np.sum(counts))
    rows: list[dict[str, float]] = []
    for left, right, count in zip(edges[:-1], edges[1:], counts, strict=False):
        width = float(right - left)
        density = float(count / (total * width)) if total > 0 and width > 0 else 0.0
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "mid": float((left + right) / 2.0),
                "width": width,
                "count": int(count),
                "density": density,
            }
        )
    return pd.DataFrame(rows)


def _factual_histogram_frame(group_df: pd.DataFrame, duration_column: str, event_column: str) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame(
            columns=[
                "left",
                "right",
                "mid",
                "width",
                "failures",
                "censored_zero",
                "other_flags",
                "total",
            ]
        )

    working = group_df[[duration_column, event_column]].copy()
    working[duration_column] = pd.to_numeric(working[duration_column], errors="coerce")
    working[event_column] = pd.to_numeric(working[event_column], errors="coerce")
    working = working.dropna(subset=[duration_column])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "left",
                "right",
                "mid",
                "width",
                "failures",
                "censored_zero",
                "other_flags",
                "total",
            ]
        )

    durations = working[duration_column].to_numpy(dtype=float)
    max_duration = float(np.max(durations))
    upper_edge = max(30.0, np.ceil(max_duration / 30.0) * 30.0)
    edges = np.arange(0.0, upper_edge + 30.0, 30.0, dtype=float)

    failure_mask = working[event_column] == 1
    censored_zero_mask = working[event_column] == 0
    other_mask = ~(failure_mask | censored_zero_mask)

    failure_counts, edges = np.histogram(working.loc[failure_mask, duration_column].to_numpy(dtype=float), bins=edges)
    censored_counts, _ = np.histogram(working.loc[censored_zero_mask, duration_column].to_numpy(dtype=float), bins=edges)
    other_counts, _ = np.histogram(working.loc[other_mask, duration_column].to_numpy(dtype=float), bins=edges)

    rows: list[dict[str, float]] = []
    for left, right, failure_count, censored_count, other_count in zip(
        edges[:-1],
        edges[1:],
        failure_counts,
        censored_counts,
        other_counts,
        strict=False,
    ):
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "mid": float((left + right) / 2.0),
                "width": float(right - left),
                "failures": int(failure_count),
                "censored_zero": int(censored_count),
                "other_flags": int(other_count),
                "total": int(failure_count + censored_count + other_count),
            }
        )
    return pd.DataFrame(rows)


def build_distribution_figure(
    result: WeibullStressFitResult,
    group_key: str,
    num_points: int = 200,
    beta_override: float | None = None,
    eta_override: float | None = None,
    group_df_override: pd.DataFrame | None = None,
    title_label: str | None = None,
) -> go.Figure:
    group_df = group_df_override.copy() if group_df_override is not None else result.group_source_rows(group_key)
    if group_df.empty:
        group_df = result.prepared_df.copy()
    row = result.representative_row(group_key) if group_df_override is None else group_df.iloc[0]
    plot_beta = float(beta_override if beta_override is not None else result.beta_for_group(group_key))
    plot_eta = float(eta_override if eta_override is not None else result.adjusted_eta(group_key, row=row))
    curves = _curve_frame_for_parameters(
        group_df,
        beta=plot_beta,
        eta=plot_eta,
        duration_column=result.duration_column,
        num_points=num_points,
    )
    empirical_bins = _empirical_bin_frame(group_df, result.duration_column, result.event_column)
    failure_hist = _failure_histogram_frame(group_df, result.duration_column, result.event_column)
    factual_hist = _factual_histogram_frame(group_df, result.duration_column, result.event_column)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Weibull PDF + Failure Histogram", "Stacked Factual Histogram", "Survival Curve", "Cumulative Failure Probability"),
    )
    fig.add_trace(go.Scatter(x=curves["time"], y=curves["pdf"], mode="lines", name="PDF"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curves["time"], y=curves["survival"], mode="lines", name="Survival"), row=3, col=1)
    fig.add_trace(go.Scatter(x=curves["time"], y=curves["cdf"], mode="lines", name="CDF"), row=4, col=1)

    if not failure_hist.empty:
        fig.add_trace(
            go.Bar(
                x=failure_hist["mid"],
                y=failure_hist["density"],
                width=failure_hist["width"],
                name="Failure histogram",
                marker={"color": "rgba(192, 57, 43, 0.35)", "line": {"color": "rgba(192, 57, 43, 0.7)", "width": 1}},
                opacity=0.75,
                customdata=failure_hist[["left", "right", "count"]].to_numpy(),
                hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>Wells: %{customdata[2]}<br>Density: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not factual_hist.empty:
        fig.add_trace(
            go.Bar(
                x=factual_hist["mid"],
                y=factual_hist["failures"],
                width=factual_hist["width"],
                name="Factual failures (event=1)",
                marker={"color": "rgba(192, 57, 43, 0.55)", "line": {"color": "rgba(192, 57, 43, 0.9)", "width": 1}},
                customdata=factual_hist[["left", "right", "total"]].to_numpy(),
                hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>Failures: %{y}<br>Total wells in bin: %{customdata[2]}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=factual_hist["mid"],
                y=factual_hist["censored_zero"],
                width=factual_hist["width"],
                name="Factual event=0",
                marker={"color": "rgba(41, 128, 185, 0.5)", "line": {"color": "rgba(41, 128, 185, 0.85)", "width": 1}},
                customdata=factual_hist[["left", "right", "total"]].to_numpy(),
                hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>event=0 wells: %{y}<br>Total wells in bin: %{customdata[2]}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        if int(factual_hist["other_flags"].sum()) > 0:
            fig.add_trace(
                go.Bar(
                    x=factual_hist["mid"],
                    y=factual_hist["other_flags"],
                    width=factual_hist["width"],
                    name="Factual other event flags",
                    marker={"color": "rgba(107, 114, 128, 0.5)", "line": {"color": "rgba(107, 114, 128, 0.85)", "width": 1}},
                    customdata=factual_hist[["left", "right", "total"]].to_numpy(),
                    hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>Other-flag wells: %{y}<br>Total wells in bin: %{customdata[2]}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    if not empirical_bins.empty:
        fig.add_trace(
            go.Bar(
                x=empirical_bins["mid"],
                y=empirical_bins["survival"],
                width=empirical_bins["width"],
                name="Empirical survival",
                marker={"color": "rgba(15, 118, 110, 0.28)", "line": {"color": "rgba(15, 118, 110, 0.6)", "width": 1}},
                opacity=0.8,
                customdata=empirical_bins[["left", "right", "wells_remaining", "initial_wells"]].to_numpy(),
                hovertemplate=(
                    "Range: %{customdata[0]:g} - %{customdata[1]:g}<br>"
                    "Wells remaining: %{customdata[2]} of %{customdata[3]}<br>"
                    "Survival: %{y:.4f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=empirical_bins["mid"],
                y=empirical_bins["failure"],
                width=empirical_bins["width"],
                name="Empirical failure probability",
                marker={"color": "rgba(180, 83, 9, 0.28)", "line": {"color": "rgba(180, 83, 9, 0.6)", "width": 1}},
                opacity=0.8,
            ),
            row=4,
            col=1,
        )

    fig.update_xaxes(title_text="Duration", row=4, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Well count", row=2, col=1)
    fig.update_yaxes(title_text="S(t)", row=3, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="F(t)", row=4, col=1, range=[0, 1.05])
    fig.update_layout(
        height=1120,
        legend_title_text="Series",
        barmode="stack",
        title=f"{title_label or group_key}: beta={plot_beta:.4f}, eta={plot_eta:.2f}",
    )
    return fig


def _palette(count: int) -> list[str]:
    base = ["#0f766e", "#1d4ed8", "#b45309", "#be123c", "#6d28d9", "#15803d", "#475569"]
    if count <= len(base):
        return base[:count]
    return [base[index % len(base)] for index in range(count)]


def _range_labels(values: list[float]) -> list[str]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    labels = [f"x <= {ordered[0]:g}"]
    for left, right in zip(ordered[:-1], ordered[1:], strict=False):
        labels.append(f"{left:g} < x <= {right:g}")
    labels.append(f"x > {ordered[-1]:g}")
    return labels


def build_sensitivity_range_split_frame(group_df: pd.DataFrame, column: str, values: Iterable[float]) -> pd.DataFrame:
    ordered = sorted(float(value) for value in values)
    split_df = group_df.copy()
    split_df[column] = pd.to_numeric(split_df[column], errors="coerce")
    split_df = split_df.loc[split_df[column].notna()].copy()
    if split_df.empty or not ordered:
        return pd.DataFrame()

    labels = _range_labels(ordered)

    def assign_label(item: float) -> str:
        if item <= ordered[0]:
            return labels[0]
        for index in range(1, len(ordered)):
            if item <= ordered[index]:
                return labels[index]
        return labels[-1]

    split_df["sensitivity_range_label"] = split_df[column].map(assign_label)
    return split_df


def build_sensitivity_figure(
    result: WeibullStressFitResult,
    group_key: str,
    stress_term_name: str,
    values: Iterable[float],
    num_points: int = 200,
    term_overrides: dict[str, float] | None = None,
) -> go.Figure:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("At least one sensitivity value is required.")

    term = next((item for item in result.stress_terms if item.name == stress_term_name), None)
    if term is None:
        raise ValueError(f"Stress term '{stress_term_name}' was not found in the fitted model.")

    baseline_row = result.representative_row(group_key)
    group_df = result.group_source_rows(group_key)
    if group_df.empty:
        group_df = result.prepared_df.copy()

    colors = _palette(max(len(numeric_values), 1))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Sensitivity Survival", "Sensitivity PDF / Failure Histogram"),
    )

    for color, value in zip(colors, numeric_values, strict=False):
        row_override = baseline_row.copy()
        row_override[term.column] = value
        adjusted_eta = result.adjusted_eta_with_term_overrides(
            group_key,
            row=row_override,
            term_overrides={term.name: term_overrides or {}},
        )
        curves = _curve_frame_for_parameters(
            group_df,
            beta=result.beta_for_group(group_key),
            eta=adjusted_eta,
            duration_column=result.duration_column,
            num_points=num_points,
        )
        fig.add_trace(
            go.Scatter(
                x=curves["time"],
                y=curves["survival"],
                mode="lines",
                name=f"Model survival: {term.column}={value:g}",
                line={"color": color, "width": 3},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=curves["time"],
                y=curves["pdf"],
                mode="lines",
                name=f"Model PDF: {term.column}={value:g}",
                line={"color": color, "width": 3},
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    split_df = build_sensitivity_range_split_frame(group_df, term.column, numeric_values)
    range_labels = _range_labels(numeric_values)
    range_colors = _palette(max(len(range_labels), 1))

    for color, label in zip(range_colors, range_labels, strict=False):
        subset = split_df.loc[split_df["sensitivity_range_label"] == label] if not split_df.empty else pd.DataFrame()
        if subset.empty:
            continue

        empirical_bins = _empirical_bin_frame(subset, result.duration_column, result.event_column)
        if not empirical_bins.empty:
            fig.add_trace(
                go.Bar(
                    x=empirical_bins["mid"],
                    y=empirical_bins["survival"],
                    width=empirical_bins["width"],
                    name=f"Empirical survival: {label}",
                    marker={"color": color, "line": {"color": color, "width": 1}},
                    opacity=0.18,
                    customdata=empirical_bins[["left", "right", "wells_remaining", "initial_wells"]].to_numpy(),
                    hovertemplate=(
                        "Range: %{customdata[0]:g} - %{customdata[1]:g}<br>"
                        "Wells remaining: %{customdata[2]} of %{customdata[3]}<br>"
                        "Survival: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        failure_hist = _failure_histogram_frame(subset, result.duration_column, result.event_column)
        if not failure_hist.empty:
            fig.add_trace(
                go.Bar(
                    x=failure_hist["mid"],
                    y=failure_hist["density"],
                    width=failure_hist["width"],
                    name=f"Empirical failure histogram: {label}",
                    marker={"color": color, "line": {"color": color, "width": 1}},
                    opacity=0.22,
                    showlegend=False,
                    customdata=failure_hist[["left", "right", "count"]].to_numpy(),
                    hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>Wells: %{customdata[2]}<br>Density: %{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Duration", row=2, col=1)
    fig.update_yaxes(title_text="S(t)", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_layout(height=760, legend_title_text=f"{term.column} scenarios", barmode="overlay")
    return fig


def build_model_sensitivity_frame(
    result: WeibullStressFitResult,
    group_key: str,
    stress_term_name: str,
    values: Iterable[float],
    term_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("At least one sensitivity value is required.")

    term = next((item for item in result.stress_terms if item.name == stress_term_name), None)
    if term is None:
        raise ValueError(f"Stress term '{stress_term_name}' was not found in the fitted model.")

    baseline_row = result.representative_row(group_key)
    group_df = result.group_source_rows(group_key)
    if group_df.empty:
        group_df = result.prepared_df.copy()

    rows: list[dict[str, float]] = []
    for value in numeric_values:
        row_override = baseline_row.copy()
        row_override[term.column] = value
        adjusted_eta = result.adjusted_eta_with_term_overrides(
            group_key,
            row=row_override,
            term_overrides={term.name: term_overrides or {}},
        )
        rows.append(
            {
                "scenario_value": float(value),
                "beta": float(result.beta_for_group(group_key)),
                "eta": float(adjusted_eta),
                "group_rows": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows)


def build_model_sensitivity_figure(
    result: WeibullStressFitResult,
    group_key: str,
    stress_term_name: str,
    values: Iterable[float],
    num_points: int = 200,
    term_overrides: dict[str, float] | None = None,
) -> go.Figure:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("At least one sensitivity value is required.")

    term = next((item for item in result.stress_terms if item.name == stress_term_name), None)
    if term is None:
        raise ValueError(f"Stress term '{stress_term_name}' was not found in the fitted model.")

    baseline_row = result.representative_row(group_key)
    group_df = result.group_source_rows(group_key)
    if group_df.empty:
        group_df = result.prepared_df.copy()

    colors = _palette(len(numeric_values))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Model Scenario Survival", "Model Scenario PDF"),
    )

    for color, value in zip(colors, numeric_values, strict=False):
        row_override = baseline_row.copy()
        row_override[term.column] = value
        adjusted_eta = result.adjusted_eta_with_term_overrides(
            group_key,
            row=row_override,
            term_overrides={term.name: term_overrides or {}},
        )
        curves = _curve_frame_for_parameters(
            group_df,
            beta=result.beta_for_group(group_key),
            eta=adjusted_eta,
            duration_column=result.duration_column,
            num_points=num_points,
        )
        fig.add_trace(
            go.Scatter(
                x=curves["time"],
                y=curves["survival"],
                mode="lines",
                name=f"Model {term.column}={value:g}",
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=curves["time"],
                y=curves["pdf"],
                mode="lines",
                name=f"Model PDF {value:g}",
                line={"color": color},
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Duration", row=2, col=1)
    fig.update_yaxes(title_text="S(t)", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_layout(height=700, legend_title_text=f"{term.column} model scenario", barmode="overlay")
    return fig


def build_empirical_sensitivity_figure(
    result: WeibullStressFitResult,
    group_key: str,
    stress_term_name: str,
    values: Iterable[float],
) -> go.Figure:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("At least one sensitivity value is required.")

    term = next((item for item in result.stress_terms if item.name == stress_term_name), None)
    if term is None:
        raise ValueError(f"Stress term '{stress_term_name}' was not found in the fitted model.")

    group_df = result.group_source_rows(group_key)
    if group_df.empty:
        group_df = result.prepared_df.copy()

    split_df = build_sensitivity_range_split_frame(group_df, term.column, numeric_values)
    range_labels = _range_labels(numeric_values)
    range_colors = _palette(max(len(range_labels), 1))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Empirical Scenario Survival", "Empirical Failure Histogram"),
    )

    for color, label in zip(range_colors, range_labels, strict=False):
        subset = split_df.loc[split_df["sensitivity_range_label"] == label] if not split_df.empty else pd.DataFrame()
        if subset.empty:
            continue

        empirical_bins = _empirical_bin_frame(subset, result.duration_column, result.event_column)
        if not empirical_bins.empty:
            fig.add_trace(
                go.Bar(
                    x=empirical_bins["mid"],
                    y=empirical_bins["survival"],
                    width=empirical_bins["width"],
                    name=f"Empirical survival: {label}",
                    marker={"color": color, "line": {"color": color, "width": 1}},
                    opacity=0.2,
                    customdata=empirical_bins[["left", "right", "wells_remaining", "initial_wells"]].to_numpy(),
                    hovertemplate=(
                        "Range: %{customdata[0]:g} - %{customdata[1]:g}<br>"
                        "Wells remaining: %{customdata[2]} of %{customdata[3]}<br>"
                        "Survival: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        failure_hist = _failure_histogram_frame(subset, result.duration_column, result.event_column)
        if not failure_hist.empty:
            fig.add_trace(
                go.Bar(
                    x=failure_hist["mid"],
                    y=failure_hist["density"],
                    width=failure_hist["width"],
                    name=f"Empirical failure histogram: {label}",
                    marker={"color": color, "line": {"color": color, "width": 1}},
                    opacity=0.25,
                    showlegend=False,
                    customdata=failure_hist[["left", "right", "count"]].to_numpy(),
                    hovertemplate="Range: %{customdata[0]:g} - %{customdata[1]:g}<br>Wells: %{customdata[2]}<br>Density: %{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Duration", row=2, col=1)
    fig.update_yaxes(title_text="S(t)", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_layout(height=700, legend_title_text=f"{term.column} empirical scenario", barmode="overlay")
    return fig
