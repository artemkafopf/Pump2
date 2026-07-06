from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis import BayesianLatentWeibullConfig, fit_bayesian_latent_weibull, kaplan_meier_frame


st.set_page_config(
    page_title="Bayesian Latent Weibull Mixture",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_uploaded_sheets(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return {"CSV": pd.read_csv(BytesIO(file_bytes))}
    engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
    workbook = pd.ExcelFile(BytesIO(file_bytes), engine=engine)
    return {sheet_name: workbook.parse(sheet_name=sheet_name) for sheet_name in workbook.sheet_names}


def _curve_figure(survival_frame: pd.DataFrame, km_frame: pd.DataFrame, n_components: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=survival_frame["time"],
            y=survival_frame["survival_mean"],
            mode="lines",
            name="Posterior survival mean",
            line={"width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=survival_frame["time"],
            y=survival_frame["survival_q97_5"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=survival_frame["time"],
            y=survival_frame["survival_q2_5"],
            mode="lines",
            fill="tonexty",
            name="95% credible interval",
            line={"width": 0},
            hoverinfo="skip",
        )
    )
    if not km_frame.empty:
        figure.add_trace(
            go.Scatter(
                x=km_frame["time"],
                y=km_frame["survival"],
                mode="lines+markers",
                name="Kaplan-Meier",
                line={"dash": "dash"},
            )
        )
    for component in range(n_components):
        figure.add_trace(
            go.Scatter(
                x=survival_frame["time"],
                y=survival_frame[f"component_{component + 1}_survival_mean"],
                mode="lines",
                name=f"Component {component + 1} survival",
            )
        )
    figure.update_layout(height=460, title="Kaplan-Meier vs Posterior Weibull Mixture Survival", xaxis_title="Time", yaxis_title="Survival")
    return figure


def _hazard_figure(survival_frame: pd.DataFrame, n_components: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=survival_frame["time"], y=survival_frame["hazard_mean"], mode="lines", name="Mixture hazard"))
    for component in range(n_components):
        figure.add_trace(
            go.Scatter(
                x=survival_frame["time"],
                y=survival_frame[f"component_{component + 1}_hazard_mean"],
                mode="lines",
                name=f"Component {component + 1} hazard",
            )
        )
    figure.update_layout(height=420, title="Posterior Mean Hazards", xaxis_title="Time", yaxis_title="Hazard")
    return figure


def _trace_figure(trace_frame: pd.DataFrame, parameter: str) -> go.Figure:
    filtered = trace_frame.loc[trace_frame["parameter"] == parameter].copy()
    figure = go.Figure()
    for (chain, component), chunk in filtered.groupby(["chain", "component"], sort=True):
        figure.add_trace(
            go.Scatter(
                x=chunk["sample"],
                y=chunk["value"],
                mode="lines",
                name=f"chain {chain} | {component}",
            )
        )
    figure.update_layout(height=420, title=f"Trace Plot: {parameter}", xaxis_title="Saved sample", yaxis_title=parameter)
    return figure


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.title("Bayesian Latent Weibull Mixture")
st.caption(
    "Fits a fixed-K Bayesian Weibull mixture to right-censored data. This pass focuses on latent heterogeneity, not competing risks."
)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()

sheets = load_uploaded_sheets(uploaded_file.getvalue(), uploaded_file.name)
sheet_name = st.selectbox("Worksheet", options=list(sheets.keys()))
source_df = sheets[sheet_name]
columns = source_df.columns.tolist()

with st.sidebar:
    st.header("Column Mapping")
    duration_mode = st.radio("Duration source", options=["Existing duration column", "Calculate from dates"], index=0)
    duration_column = None
    start_date_column = None
    end_date_column = None
    if duration_mode == "Existing duration column":
        duration_column = st.selectbox("Duration column", options=columns)
    else:
        start_date_column = st.selectbox("Start date column", options=columns)
        end_date_column = st.selectbox("End date column", options=columns)
    event_column = st.selectbox("Event column", options=columns)
    id_column = st.selectbox("ID column", options=["<auto>"] + columns, index=0)

    st.header("MCMC Setup")
    use_unknown_k = st.toggle("Infer K with birth/death MCMC", value=False)
    n_components = st.slider("Initial K" if use_unknown_k else "Fixed K", min_value=1, max_value=6, value=2, step=1)
    k_max = st.slider("K max", min_value=1, max_value=8, value=max(4, n_components), step=1, disabled=not use_unknown_k)
    lambda_k = st.number_input("Lambda_K", min_value=0.1, value=3.0, step=0.1, disabled=not use_unknown_k)
    birth_death_probability = st.slider("Birth/death move probability", min_value=0.05, max_value=0.95, value=0.25, step=0.05, disabled=not use_unknown_k)
    n_iter = st.number_input("Iterations per chain", min_value=400, value=1600, step=200)
    burn_in = st.number_input("Burn-in", min_value=100, value=800, step=100)
    thin = st.number_input("Thin", min_value=1, value=4, step=1)
    n_chains = st.slider("Chains", min_value=1, max_value=4, value=2, step=1)
    random_seed = st.number_input("Random seed", min_value=1, value=42, step=1)
    run_fit = st.button("Fit Bayesian Mixture", type="primary")

if not run_fit:
    st.dataframe(source_df.head(20), width="stretch")
    st.stop()

config = BayesianLatentWeibullConfig(
    n_components=int(n_components),
    n_iter=int(n_iter),
    burn_in=int(burn_in),
    thin=int(thin),
    n_chains=int(n_chains),
    random_seed=int(random_seed),
    use_unknown_k=bool(use_unknown_k),
    k_max=int(max(k_max, n_components)),
    lambda_k=float(lambda_k),
    birth_death_probability=float(birth_death_probability),
)

with st.spinner("Running MCMC..."):
    result = fit_bayesian_latent_weibull(
        source_df,
        duration_column=duration_column,
        event_column=event_column,
        id_column=None if id_column == "<auto>" else id_column,
        start_date_column=start_date_column,
        end_date_column=end_date_column,
        config=config,
    )

km_frame = kaplan_meier_frame(
    result.cleaned_df[result.duration_column].to_numpy(dtype=float),
    result.cleaned_df[result.event_column].to_numpy(dtype=int),
)
survival_frame = result.posterior_survival_frame()
posterior_summary = result.posterior_summary_frame()
latent_probs = result.observation_posterior_probabilities()
active_predictions = result.active_unit_predictions()
trace_frame = result.trace_frame()
diagnostics = result.diagnostics_frame()

top_metrics = st.columns(5)
top_metrics[0].metric("Rows used", f"{result.validation_report.rows_after_validation:,}")
top_metrics[1].metric("Failures", f"{result.validation_report.failure_count:,}")
top_metrics[2].metric("Censored", f"{result.validation_report.censored_count:,}")
top_metrics[3].metric("Saved draws / chain", f"{result.chains[0].weights_relabeled.shape[0]:,}")
top_metrics[4].metric("Mode", "Unknown K" if result.config.use_unknown_k else f"Fixed K={result.config.n_components}")

tab_data, tab_fit, tab_posteriors, tab_predictions, tab_export = st.tabs(
    ["Validation", "Model Fit", "Posterior Diagnostics", "Predictions", "Export"]
)

with tab_data:
    st.subheader("Validation Report")
    st.dataframe(result.validation_report.to_frame(), width="stretch")
    st.subheader("Cleaned Data Preview")
    st.dataframe(result.cleaned_df.head(50), width="stretch")

with tab_fit:
    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.plotly_chart(_curve_figure(survival_frame, km_frame, result.max_components), width="stretch")
    with plot_cols[1]:
        st.plotly_chart(_hazard_figure(survival_frame, result.max_components), width="stretch")
    if result.config.use_unknown_k:
        st.subheader("Posterior Distribution of K")
        st.dataframe(result.posterior_k_frame(), width="stretch")
    st.subheader("Posterior Summary")
    st.dataframe(posterior_summary, width="stretch")

with tab_posteriors:
    st.subheader("Chain Diagnostics")
    st.dataframe(diagnostics, width="stretch")
    parameter = st.selectbox("Trace parameter", options=["K", "log_likelihood", "log_posterior", "weight", "eta", "beta"])
    st.plotly_chart(_trace_figure(trace_frame, parameter), width="stretch")
    st.subheader("Interpretation Safeguards")
    for limitation in result.limitations:
        st.write(f"- {limitation}")

with tab_predictions:
    st.subheader("Observation-Level Latent Probabilities")
    st.dataframe(latent_probs.head(200), width="stretch")
    st.subheader("Active Unit Predictions")
    if active_predictions.empty:
        st.info("No censored observations were found, so there are no active-unit predictions.")
    else:
        st.dataframe(active_predictions.head(200), width="stretch")

with tab_export:
    st.download_button(
        "Download posterior summary CSV",
        data=_csv_bytes(posterior_summary),
        file_name="posterior_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download latent probabilities CSV",
        data=_csv_bytes(latent_probs),
        file_name="latent_probabilities.csv",
        mime="text/csv",
    )
    if not active_predictions.empty:
        st.download_button(
            "Download active predictions CSV",
            data=_csv_bytes(active_predictions),
            file_name="active_unit_predictions.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download posterior survival CSV",
        data=_csv_bytes(survival_frame),
        file_name="posterior_survival.csv",
        mime="text/csv",
    )
