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
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis import build_ttf_correlation_frame, prepare_presentation_dataset, read_excel_sheets


st.set_page_config(
    page_title="ESP Presentation Verification",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return read_excel_sheets(file_bytes)


@st.cache_data(show_spinner=False)
def analyze_workbook(file_bytes: bytes, infant_threshold_days: float, include_regime_last_30d_ratios: bool):
    workbook = read_excel_sheets(file_bytes)
    return prepare_presentation_dataset(
        workbook,
        infant_threshold_days=infant_threshold_days,
        include_regime_last_30d_ratios=include_regime_last_30d_ratios,
    )


def _mapping_frame(report) -> pd.DataFrame:
    if not report.mapped_columns:
        return pd.DataFrame(columns=["canonical_column", "source_column"])
    return pd.DataFrame(
        [{"canonical_column": canonical, "source_column": source} for canonical, source in report.mapped_columns.items()]
    ).sort_values("canonical_column")


def _sheet_report_frame(result) -> pd.DataFrame:
    return pd.DataFrame([report.to_row() for report in result.sheet_reports])


def _download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _correlation_scatter_figure(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str = "TTF_days",
) -> go.Figure:
    plot_df = df.copy()
    plot_df[feature_column] = pd.to_numeric(plot_df[feature_column], errors="coerce")
    plot_df[target_column] = pd.to_numeric(plot_df[target_column], errors="coerce")
    plot_df = plot_df.loc[plot_df[feature_column].notna() & plot_df[target_column].notna()].copy()
    if plot_df.empty:
        return go.Figure()

    if "event" in plot_df.columns:
        event_numeric = pd.to_numeric(plot_df["event"], errors="coerce")
        plot_df["event_label"] = event_numeric.map({1: "Failure", 0: "Censored"}).fillna("Unknown")
    else:
        plot_df["event_label"] = "Unknown"

    hover_columns = [
        column
        for column in ["run_id", "well", "field", "contractor", "event_label"]
        if column in plot_df.columns
    ]
    fig = px.scatter(
        plot_df,
        x=feature_column,
        y=target_column,
        color="event_label",
        hover_data=hover_columns,
        opacity=0.8,
    )

    if plot_df[feature_column].nunique() >= 2:
        x_values = plot_df[feature_column].to_numpy(dtype=float)
        y_values = plot_df[target_column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_line = intercept + (slope * x_line)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear trend",
                line={"color": "#111827", "width": 2},
            )
        )

    fig.update_layout(
        height=520,
        xaxis_title=feature_column,
        yaxis_title=target_column,
        legend_title_text="Run type",
    )
    return fig


def render_app() -> None:
    st.title("ESP Presentation Verification")
    st.caption(
        "Automated workbook inspection for the presentation-specific ESP reliability workflow. "
        "Upload one workbook and the app will detect sheets, map columns, and build a run-level merged dataset."
    )

    uploaded = st.file_uploader("Upload Excel workbook", type=["xlsx", "xls"])
    if uploaded is None:
        st.info("Upload a workbook to begin the automated presentation analysis setup.")
        return

    threshold_col, info_col = st.columns([1, 2])
    infant_threshold_days = threshold_col.number_input(
        "Infant mortality threshold (days)",
        min_value=1.0,
        value=30.0,
        step=1.0,
    )
    include_regime_last_30d_ratios = threshold_col.checkbox(
        "Include regime last-30-day / full-life ratios",
        value=False,
    )
    info_col.caption(
        "This threshold is used immediately while preparing the merged run-level dataset. "
        "Optionally, the app can also append tech-regime last-30-day average ratios next to each full-run average column."
    )

    file_bytes = uploaded.getvalue()
    try:
        workbook = load_workbook(file_bytes)
        result = analyze_workbook(
            file_bytes,
            infant_threshold_days=float(infant_threshold_days),
            include_regime_last_30d_ratios=bool(include_regime_last_30d_ratios),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    summary_cols = st.columns(5)
    detected_count = sum(1 for report in result.sheet_reports if report.detected_sheet)
    summary_cols[0].metric("Workbook sheets", f"{len(workbook):,}")
    summary_cols[1].metric("Detected roles", f"{detected_count:,} / {len(result.sheet_reports):,}")
    summary_cols[2].metric("Runs rows", f"{len(result.runs_df):,}")
    summary_cols[3].metric("Merged rows", f"{len(result.merged_df):,}")
    summary_cols[4].metric("Merged columns", f"{len(result.merged_df.columns):,}")

    if result.notes:
        for note in result.notes:
            st.caption(note)

    tab_detect, tab_merged, tab_correlations, tab_ready = st.tabs(
        ["Upload & Auto-Detect", "Merged Run-Level Dataset", "Feature Correlations", "Modeling Readiness"]
    )

    with tab_detect:
        st.subheader("Available Sheets")
        st.dataframe(
            pd.DataFrame({"sheet_name": list(workbook.keys()), "rows": [len(frame) for frame in workbook.values()]}),
            use_container_width=True,
        )

        st.subheader("Detected Sheet Roles")
        st.dataframe(_sheet_report_frame(result), use_container_width=True)

        for report in result.sheet_reports:
            with st.expander(f"{report.role} mapping", expanded=False):
                st.write(f"Detected sheet: `{report.detected_sheet or 'not detected'}`")
                st.write(f"Method: `{report.detection_method}`")
                if report.join_key:
                    st.write(f"Join key: `{report.join_key}`")
                if report.notes:
                    for note in report.notes:
                        st.caption(note)
                st.dataframe(_mapping_frame(report), use_container_width=True)

    with tab_merged:
        st.subheader("Run-Level Dataset")
        st.caption(
            "This table is the automated modeling base for the future presentation checks. "
            "It starts from `runs`, then left-joins latest design values and aggregated regime/telemetry summaries."
        )
        st.download_button(
            "Download merged dataset as CSV",
            data=_download_csv(result.merged_df),
            file_name="presentation_run_level_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(result.merged_df.head(500), use_container_width=True)

    with tab_correlations:
        st.subheader("Feature Correlations vs TTF")
        st.caption(
            "This is an exploratory view of how numeric merged features move with observed `TTF_days`. "
            "By default it includes all runs, including censored runs."
        )

        if "TTF_days" not in result.merged_df.columns:
            st.info("`TTF_days` is not available in the merged dataset, so correlations cannot be plotted yet.")
        else:
            filter_cols = st.columns(3)
            event_scope = filter_cols[0].selectbox(
                "Run scope",
                options=["All runs", "Failures only", "Censored only"],
                index=0,
            )
            min_pairs = int(
                filter_cols[1].number_input(
                    "Minimum paired rows",
                    min_value=3,
                    value=5,
                    step=1,
                )
            )
            rank_method = filter_cols[2].selectbox(
                "Rank by",
                options=["Spearman", "Pearson"],
                index=0,
            )

            correlation_df = result.merged_df.copy()
            if event_scope != "All runs" and "event" in correlation_df.columns:
                event_numeric = pd.to_numeric(correlation_df["event"], errors="coerce")
                if event_scope == "Failures only":
                    correlation_df = correlation_df.loc[event_numeric == 1].copy()
                else:
                    correlation_df = correlation_df.loc[event_numeric == 0].copy()

            correlation_frame = build_ttf_correlation_frame(
                correlation_df,
                target_column="TTF_days",
                min_pairs=min_pairs,
            )

            if correlation_frame.empty:
                st.warning("No numeric features had enough valid paired values to compute correlations with `TTF_days`.")
            else:
                metric_cols = st.columns(3)
                metric_cols[0].metric("Rows in scope", f"{len(correlation_df):,}")
                metric_cols[1].metric("Ranked features", f"{len(correlation_frame):,}")
                top_value = (
                    correlation_frame["abs_spearman_corr"].iloc[0]
                    if rank_method == "Spearman"
                    else correlation_frame["abs_pearson_corr"].iloc[0]
                )
                metric_cols[2].metric(f"Top |{rank_method}|", f"{float(top_value):.3f}")

                display_frame = correlation_frame.copy()
                display_frame["rank_metric"] = (
                    display_frame["abs_spearman_corr"]
                    if rank_method == "Spearman"
                    else display_frame["abs_pearson_corr"]
                )
                display_frame = display_frame.sort_values(
                    ["rank_metric", "valid_pairs", "feature"],
                    ascending=[False, False, True],
                ).reset_index(drop=True)
                st.dataframe(display_frame, use_container_width=True)

                default_feature = str(display_frame.iloc[0]["feature"])
                selected_feature = st.selectbox(
                    "Feature to plot",
                    options=display_frame["feature"].tolist(),
                    index=0,
                )
                if default_feature != selected_feature:
                    st.caption(f"Top-ranked feature for the current filter is `{default_feature}`.")

                selected_row = display_frame.loc[display_frame["feature"] == selected_feature].iloc[0]
                info_cols = st.columns(4)
                info_cols[0].metric("Valid pairs", f"{int(selected_row['valid_pairs']):,}")
                info_cols[1].metric("Pearson", f"{float(selected_row['pearson_corr']):.3f}")
                info_cols[2].metric("Spearman", f"{float(selected_row['spearman_corr']):.3f}")
                info_cols[3].metric("Feature group", str(selected_row["feature_group"]))

                st.plotly_chart(
                    _correlation_scatter_figure(correlation_df, selected_feature, target_column="TTF_days"),
                    use_container_width=True,
                )

    with tab_ready:
        st.subheader("Suggested Defaults")
        defaults = pd.DataFrame(
            [
                {"setting": "Event column", "value": result.suggested_event_column or "Not detected"},
                {"setting": "Duration column", "value": result.suggested_duration_column or "Not detected"},
                {
                    "setting": "Group columns",
                    "value": ", ".join(result.suggested_group_columns) if result.suggested_group_columns else "Not detected",
                },
            ]
        )
        st.dataframe(defaults, use_container_width=True)

        st.subheader("Gas Limits In Use")
        st.dataframe(result.gas_limits_df, use_container_width=True)

        present_sources = pd.DataFrame(
            [
                {
                    "source": "runs",
                    "detected": bool(result.runs_df is not None and not result.runs_df.empty),
                    "rows": len(result.runs_df),
                },
                {
                    "source": "design",
                    "detected": bool(result.design_df is not None and not result.design_df.empty),
                    "rows": 0 if result.design_df is None else len(result.design_df),
                },
                {
                    "source": "daily_or_monthly_regime",
                    "detected": bool(result.regime_df is not None and not result.regime_df.empty),
                    "rows": 0 if result.regime_df is None else len(result.regime_df),
                },
                {
                    "source": "telemetry",
                    "detected": bool(result.telemetry_df is not None and not result.telemetry_df.empty),
                    "rows": 0 if result.telemetry_df is None else len(result.telemetry_df),
                },
            ]
        )
        st.subheader("Source Coverage")
        st.dataframe(present_sources, use_container_width=True)

        with st.expander("Merged column list", expanded=False):
            st.write(pd.DataFrame({"column": result.merged_df.columns.tolist()}))


render_app()
