from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis import (
    TRANSFORM_LIBRARY,
    apply_derived_columns,
    apply_filters,
    build_grouped_stress_correlation_frame,
    build_stress_term_observation_frame,
    fit_weibull_stress_model,
    inspect_derived_presets,
    prepare_modeling_dataframe,
    rank_transform_candidates,
    suggest_stress_term_presets,
    suggest_derived_presets,
)
from analysis.plotting import (
    build_distribution_figure,
    build_model_sensitivity_frame,
    build_sensitivity_figure,
    build_sensitivity_range_split_frame,
)
from analysis.weibull_model import evaluate_weibull_nll, fit_basic_weibull


st.set_page_config(
    page_title="Weibull Stress Model",
    page_icon=":material/monitoring:",
    layout="wide",
)


FILTER_OPERATORS = ["==", "!=", "<", "<=", ">", ">=", "contains", "not_contains", "is_null", "not_null"]
DEFAULT_PRESET = {
    "event_column": "Failure Flag",
    "duration_column": "Наработка (сут)",
    "group_columns": ["Месторождение", "Принадлежность"],
    "filters": [
        {
            "column": "Наработка (сут)",
            "operator": ">",
            "value": "30",
        }
    ],
}


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    workbook = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return {sheet_name: workbook.parse(sheet_name=sheet_name) for sheet_name in workbook.sheet_names}


def build_processed_workbook(
    sheets: dict[str, pd.DataFrame],
    selected_sheet: str,
    processed_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            output = processed_df if sheet_name == selected_sheet else frame
            output.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def parse_numeric_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def suggest_sensitivity_values(group_df: pd.DataFrame, column: str) -> list[float]:
    numeric = pd.to_numeric(group_df[column], errors="coerce").dropna()
    if numeric.empty:
        return []
    return [round(float(numeric.median()), 6)]


def format_sensitivity_values(values: list[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _streamlit_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for column in safe_df.columns:
        series = safe_df[column]
        if series.dtype != "object":
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        python_types = {type(value) for value in non_null.tolist()}
        if len(python_types) > 1:
            safe_df[column] = series.astype("string")
    return safe_df


def render_dataframe(df: pd.DataFrame) -> None:
    st.dataframe(_streamlit_safe_dataframe(df), width="stretch")


def build_hierarchy_details(result) -> pd.DataFrame:
    df = result.prepared_df.copy()
    if df.empty:
        return pd.DataFrame()

    duration_column = result.duration_column
    event_column = result.event_column
    group_columns = [column for column in result.group_columns if column in df.columns]
    frames: list[pd.DataFrame] = []

    if group_columns:
        first_column = group_columns[0]
        first_df = df.copy()
        first_df["tree_level"] = first_column
        first_df["tree_key"] = first_df[first_column].astype("string").fillna("<missing>")
        frames.append(first_df)

    if len(group_columns) >= 2:
        second_column = group_columns[1]
        second_df = df.copy()
        second_df["tree_level"] = second_column
        second_df["tree_key"] = second_df[second_column].astype("string").fillna("<missing>")
        frames.append(second_df)

        combo_df = df.copy()
        combo_df["tree_level"] = f"{group_columns[0]} + {group_columns[1]}"
        combo_df["tree_key"] = (
            combo_df[group_columns[0]].astype("string").fillna("<missing>")
            + " | "
            + combo_df[group_columns[1]].astype("string").fillna("<missing>")
        )
        frames.append(combo_df)

    if not frames:
        fallback_df = df.copy()
        fallback_df["tree_level"] = "GLOBAL"
        fallback_df["tree_key"] = "GLOBAL"
        frames.append(fallback_df)

    stacked = pd.concat(frames, ignore_index=True)
    stacked["_duration_numeric"] = pd.to_numeric(stacked[duration_column], errors="coerce")
    stacked["_event_numeric"] = pd.to_numeric(stacked[event_column], errors="coerce")

    summary = (
        stacked.groupby(["tree_level", "tree_key"], dropna=False)
        .agg(
            rows=("_duration_numeric", "size"),
            failures=("_event_numeric", "sum"),
            mean_duration=("_duration_numeric", "mean"),
            median_duration=("_duration_numeric", "median"),
            std_duration=("_duration_numeric", "std"),
            min_duration=("_duration_numeric", "min"),
            max_duration=("_duration_numeric", "max"),
        )
        .reset_index()
    )
    summary["failures"] = summary["failures"].fillna(0).astype(int)
    summary["censored"] = summary["rows"] - summary["failures"]
    return summary.sort_values(["tree_level", "rows", "tree_key"], ascending=[True, False, True]).reset_index(drop=True)


def build_hierarchy_rows(result, tree_level: str, tree_key: str) -> pd.DataFrame:
    df = result.prepared_df.copy()
    if df.empty:
        return df

    group_columns = [column for column in result.group_columns if column in df.columns]
    if tree_level == "GLOBAL" or not group_columns:
        return df
    if tree_level == group_columns[0]:
        return df.loc[df[group_columns[0]].astype("string").fillna("<missing>") == str(tree_key)].copy()
    if len(group_columns) >= 2 and tree_level == group_columns[1]:
        return df.loc[df[group_columns[1]].astype("string").fillna("<missing>") == str(tree_key)].copy()
    if len(group_columns) >= 2 and tree_level == f"{group_columns[0]} + {group_columns[1]}":
        combo_key = (
            df[group_columns[0]].astype("string").fillna("<missing>")
            + " | "
            + df[group_columns[1]].astype("string").fillna("<missing>")
        )
        return df.loc[combo_key == str(tree_key)].copy()
    return pd.DataFrame(columns=df.columns)




def maybe_apply_default_preset(df: pd.DataFrame, sheet_name: str) -> None:
    columns = [str(column) for column in df.columns.tolist()]
    signature = (sheet_name, tuple(columns))
    if st.session_state.get("applied_default_preset_signature") == signature:
        return

    required_columns = {
        DEFAULT_PRESET["event_column"],
        DEFAULT_PRESET["duration_column"],
        *DEFAULT_PRESET["group_columns"],
    }
    if not required_columns.issubset(set(columns)):
        return

    st.session_state["event_mode"] = "Use existing event column"
    st.session_state["event_column"] = DEFAULT_PRESET["event_column"]
    st.session_state["duration_column"] = DEFAULT_PRESET["duration_column"]
    st.session_state["group_columns"] = [
        column for column in DEFAULT_PRESET["group_columns"] if column in columns
    ]
    st.session_state["filter_count"] = len(DEFAULT_PRESET["filters"])

    for index, filter_config in enumerate(DEFAULT_PRESET["filters"]):
        st.session_state[f"filter_column_{index}"] = filter_config["column"]
        st.session_state[f"filter_operator_{index}"] = filter_config["operator"]
        st.session_state[f"filter_value_{index}"] = filter_config["value"]

    st.session_state["applied_default_preset_signature"] = signature


def normalize_widget_state_for_columns(df: pd.DataFrame) -> None:
    columns = [str(column) for column in df.columns.tolist()]
    if not columns:
        return

    if st.session_state.get("event_column") not in columns:
        st.session_state["event_column"] = columns[0]
    if st.session_state.get("duration_column") not in columns:
        st.session_state["duration_column"] = columns[0]

    selected_groups = st.session_state.get("group_columns", [])
    st.session_state["group_columns"] = [column for column in selected_groups if column in columns]

    filter_count = int(st.session_state.get("filter_count", 0))
    for index in range(filter_count):
        if st.session_state.get(f"filter_column_{index}") not in columns:
            st.session_state[f"filter_column_{index}"] = columns[0]
        if st.session_state.get(f"filter_operator_{index}") not in FILTER_OPERATORS:
            st.session_state[f"filter_operator_{index}"] = FILTER_OPERATORS[0]


def build_group_option_labels(result) -> dict[str, str]:
    labels: dict[str, str] = {}
    for group_key in result.groups:
        stats = result.group_stats[group_key]
        labels[group_key] = (
            f"{group_key} "
            f"(original rows: {stats['source_rows']}, eta-bucket rows: {stats['bucket_rows']}, level: {stats['group_level']})"
        )
    return labels


def _preset_table_frame(presets) -> pd.DataFrame:
    rows: list[dict] = []
    for preset in presets:
        rows.append(
            {
                "include": bool(preset.auto_include),
                "auto_stress": bool(preset.auto_stress and preset.stress_preset is not None),
                "derived_column": preset.name,
                "category": preset.category,
                "description": preset.description,
                "formula": preset.formula,
                "stress_preset": "" if preset.stress_preset is None else preset.stress_preset.summary,
                "preset_key": preset.key,
            }
        )
    return pd.DataFrame(rows)


def _preset_catalog_frame(presets) -> pd.DataFrame:
    rows: list[dict] = []
    for preset in presets:
        rows.append(
            {
                "status": preset.availability_note,
                "available_now": bool(preset.available),
                "derived_column": preset.name,
                "category": preset.category,
                "description": preset.description,
                "required_inputs": ", ".join(preset.required_inputs),
                "formula_if_available": preset.formula,
                "stress_preset": "" if preset.stress_preset is None else preset.stress_preset.summary,
            }
        )
    return pd.DataFrame(rows)


def _preset_editor_key(selected_sheet: str, df: pd.DataFrame) -> str:
    signature_text = "||".join([selected_sheet, *[str(column) for column in df.columns.tolist()]])
    digest = hashlib.md5(signature_text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"derived_preset_table_{digest}"


def _stress_editor_key(df: pd.DataFrame, auto_stress_terms: list[dict] | None = None) -> str:
    parts = [str(column) for column in df.columns.tolist()]
    for term in auto_stress_terms or []:
        parts.extend(
            [
                str(term.get("name", "")),
                str(term.get("column", "")),
                str(term.get("transform", "")),
                str(term.get("reference_mode", "")),
            ]
        )
    digest = hashlib.md5("||".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"stress_term_table_{digest}"


def _to_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return float(value)


def _default_numeric_value(df: pd.DataFrame, column: str, fallback: float = 1.0) -> float:
    if column not in df.columns:
        return fallback
    series = pd.to_numeric(df[column], errors="coerce")
    value = series.median()
    return fallback if pd.isna(value) else float(value)


def _default_quantile_value(df: pd.DataFrame, column: str, q: float, fallback: float) -> float:
    if column not in df.columns:
        return fallback
    series = pd.to_numeric(df[column], errors="coerce")
    value = series.quantile(q)
    return fallback if pd.isna(value) else float(value)


def _default_min_value(df: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    if column not in df.columns:
        return fallback
    series = pd.to_numeric(df[column], errors="coerce")
    value = series.min()
    return fallback if pd.isna(value) else float(value)


def _default_max_value(df: pd.DataFrame, column: str, fallback: float = 1.0) -> float:
    if column not in df.columns:
        return fallback
    series = pd.to_numeric(df[column], errors="coerce")
    value = series.max()
    return fallback if pd.isna(value) else float(value)


def _auto_fit_reference_payload(df: pd.DataFrame, column: str) -> tuple[float, float, float]:
    reference_init = _default_numeric_value(df, column, 1.0)
    reference_lower = _default_min_value(df, column, reference_init)
    reference_upper = _default_max_value(df, column, reference_init)
    return reference_init, reference_lower, reference_upper


def _stress_preset_row_from_term(
    term: dict,
    *,
    df: pd.DataFrame,
    source: str,
    category: str,
    description: str,
    include: bool = True,
) -> dict:
    coefficient_bounds = term.get("coefficient_bounds", [0.0, None]) or [0.0, None]
    reference_multiplier_bounds = term.get("reference_multiplier_bounds", [0.1, 10.0]) or [0.1, 10.0]
    column = str(term.get("column", ""))
    reference_init, reference_lower, reference_upper = _auto_fit_reference_payload(df, column)
    return {
        "include": include,
        "source": source,
        "category": category,
        "name": term.get("name", ""),
        "column": column,
        "transform": term.get("transform", "relative_squared_deviation"),
        "reference_mode": "fit",
        "reference_value": None,
        "reference_column": "",
        "reference_init": reference_init,
        "reference_lower_bound": reference_lower,
        "reference_upper_bound": reference_upper,
        "reference_multiplier_mode": "fixed",
        "reference_multiplier_value": 1.0,
        "reference_multiplier_init": 1.0,
        "reference_multiplier_lower_bound": _to_optional_float(reference_multiplier_bounds[0]),
        "reference_multiplier_upper_bound": _to_optional_float(reference_multiplier_bounds[1]),
        "coefficient_mode": term.get("coefficient_mode", "fit"),
        "coefficient_value": _to_optional_float(term.get("coefficient_value", 0.05)),
        "coefficient_non_negative": bool(term.get("coefficient_non_negative", True)),
        "coefficient_lower_bound": _to_optional_float(coefficient_bounds[0]),
        "coefficient_upper_bound": _to_optional_float(coefficient_bounds[1]),
        "scale": _to_optional_float(term.get("scale", 1.0)) or 1.0,
        "description": description,
    }


def _collect_stress_term_rows(df: pd.DataFrame, auto_stress_terms: list[dict] | None = None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()

    for term in auto_stress_terms or []:
        row = _stress_preset_row_from_term(
            term,
            df=df,
            source="Derived preset",
            category="Derived",
            description=str(term.get("description", "Recommended from selected derived feature.")),
            include=True,
        )
        identity = (str(row["name"]), str(row["column"]), str(row["transform"]), str(row["reference_mode"]))
        if identity not in seen:
            seen.add(identity)
            rows.append(row)

    for preset in suggest_stress_term_presets(df):
        row = _stress_preset_row_from_term(
            preset.term.to_term_dict(),
            df=df,
            source=preset.source_type,
            category=preset.category,
            description=preset.description,
            include=True,
        )
        identity = (str(row["name"]), str(row["column"]), str(row["transform"]), str(row["reference_mode"]))
        if identity not in seen:
            seen.add(identity)
            rows.append(row)

    return rows


def _manual_stress_term_row(df: pd.DataFrame, numeric_candidates: list[str]) -> dict:
    default_column = numeric_candidates[0] if numeric_candidates else ""
    return {
        "include": True,
        "source": "Manual",
        "category": "Manual",
        "name": "stress_manual",
        "column": default_column,
        "transform": "relative_squared_deviation" if "relative_squared_deviation" in TRANSFORM_LIBRARY else sorted(TRANSFORM_LIBRARY)[0],
        "reference_mode": "fixed",
        "reference_value": _default_numeric_value(df, default_column, 1.0),
        "reference_column": default_column,
        "reference_init": _default_numeric_value(df, default_column, 1.0),
        "reference_lower_bound": _default_quantile_value(df, default_column, 0.1, 0.0),
        "reference_upper_bound": _default_quantile_value(df, default_column, 0.9, 1.0),
        "reference_multiplier_mode": "fixed",
        "reference_multiplier_value": 1.0,
        "reference_multiplier_init": 1.0,
        "reference_multiplier_lower_bound": 0.1,
        "reference_multiplier_upper_bound": 10.0,
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_lower_bound": 0.0,
        "coefficient_upper_bound": 10.0,
        "scale": 1.0,
        "description": "User-defined stress term.",
    }


def _stress_terms_from_editor_frame(editor_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    terms: list[dict] = []
    errors: list[str] = []
    for index, row in editor_df.iterrows():
        if not bool(row.get("include", False)):
            continue

        name = str(row.get("name", "")).strip()
        column = str(row.get("column", "")).strip()
        transform = str(row.get("transform", "")).strip()
        reference_mode = str(row.get("reference_mode", "fixed")).strip() or "fixed"
        if not name or not column or not transform:
            errors.append(f"Stress row {index + 1}: name, column, and transform are required.")
            continue

        term = {
            "name": name,
            "column": column,
            "transform": transform,
            "reference_mode": reference_mode,
            "coefficient_mode": str(row.get("coefficient_mode", "fit")).strip() or "fit",
            "coefficient_value": _to_optional_float(row.get("coefficient_value")) or 0.05,
            "coefficient_non_negative": bool(row.get("coefficient_non_negative", True)),
            "coefficient_bounds": [
                _to_optional_float(row.get("coefficient_lower_bound")),
                _to_optional_float(row.get("coefficient_upper_bound")),
            ],
            "scale": _to_optional_float(row.get("scale")) or 1.0,
        }

        if reference_mode == "fixed":
            reference_value = _to_optional_float(row.get("reference_value"))
            if reference_value is None:
                errors.append(f"Stress row '{name}': fixed reference mode requires `reference_value`.")
                continue
            term["reference_value"] = reference_value
        elif reference_mode == "column":
            reference_column = str(row.get("reference_column", "")).strip()
            if not reference_column:
                errors.append(f"Stress row '{name}': column reference mode requires `reference_column`.")
                continue
            term["reference_column"] = reference_column
            multiplier_mode = str(row.get("reference_multiplier_mode", "fixed")).strip() or "fixed"
            term["reference_multiplier_mode"] = multiplier_mode
            if multiplier_mode == "fit":
                term["reference_multiplier_init"] = _to_optional_float(row.get("reference_multiplier_init")) or 1.0
                term["reference_multiplier_bounds"] = [
                    _to_optional_float(row.get("reference_multiplier_lower_bound")),
                    _to_optional_float(row.get("reference_multiplier_upper_bound")),
                ]
            else:
                term["reference_multiplier_value"] = _to_optional_float(row.get("reference_multiplier_value")) or 1.0
        else:
            reference_init = _to_optional_float(row.get("reference_init"))
            if reference_init is not None:
                term["reference_init"] = reference_init
            term["reference_bounds"] = [
                _to_optional_float(row.get("reference_lower_bound")),
                _to_optional_float(row.get("reference_upper_bound")),
            ]

        terms.append(term)
    return terms, errors


def _event_label_series(df: pd.DataFrame, event_column: str) -> pd.Series:
    numeric = pd.to_numeric(df[event_column], errors="coerce") if event_column in df.columns else pd.Series(np.nan, index=df.index)
    return numeric.map({1: "Failure", 0: "Censored"}).fillna("Unknown")


def render_input_postprocessor(
    df: pd.DataFrame,
    sheets: dict[str, pd.DataFrame],
    selected_sheet: str,
) -> tuple[pd.DataFrame, list[dict], list[dict], list[str], str | None]:
    st.subheader("Input Postprocessor")
    st.caption(
        "Add derived columns before modeling. Reference source columns with square brackets, "
        "for example `[Qliq_m3d] / [pump_nominal_rate_m3d]` or `max(0, 0.7 - [Kpod])`."
    )

    with st.expander("Formula help", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "`[Column Name]` references a column, even if the name contains spaces or Cyrillic characters.",
                    "Supported arithmetic: `+`, `-`, `*`, `/`, `**`, `%`.",
                    "Supported functions: `abs`, `clip`, `exp`, `log`, `max`, `maximum`, `min`, `minimum`, `pow`, `sqrt`.",
                    "Derived columns are calculated in order, so later formulas can use earlier derived columns such as `[Kpod]`.",
                ]
            )
        )
        examples = pd.DataFrame(
            [
                {"derived_column": "Kpod", "formula_example": "[Qliq_m3d] / [pump_nominal_rate_m3d]"},
                {
                    "derived_column": "Kpod_freq_adjusted",
                    "formula_example": "([Qliq_m3d] / [pump_nominal_rate_m3d]) * ([reference_frequency_hz] / [frequency_hz])",
                },
                {"derived_column": "stress_low_Kpod", "formula_example": "max(0, 0.7 - [Kpod])"},
            ]
        )
        render_dataframe(examples)
        render_dataframe(pd.DataFrame({"reference_syntax": [f"[{column}]" for column in df.columns.tolist()]}))

    available_presets = suggest_derived_presets(df)
    preset_catalog = inspect_derived_presets(df)

    st.subheader("Preset Derived Features")
    st.caption(
        "Select from the currently derivable engineering ratios below. The full catalog appears underneath, "
        "including items that are not available yet for this sheet."
    )

    if available_presets:
        st.caption(
            "Recommended engineering ratios and design-mismatch features are shown below. "
            "Select them directly in the table. If `Auto stress` stays checked, the feature is automatically added "
            "to the stress model using a recommended preset."
        )
        preset_map = {preset.key: preset for preset in available_presets}
        default_preset_frame = _preset_table_frame(available_presets)
        edited_preset_frame = st.data_editor(
            default_preset_frame,
            key=_preset_editor_key(selected_sheet, df),
            width="stretch",
            hide_index=True,
            disabled=["derived_column", "category", "description", "formula", "stress_preset", "preset_key"],
            column_config={
                "include": st.column_config.CheckboxColumn("Include", default=True),
                "auto_stress": st.column_config.CheckboxColumn("Auto stress", default=True),
                "derived_column": st.column_config.TextColumn("Derived column"),
                "category": st.column_config.TextColumn("Category"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "formula": st.column_config.TextColumn("Formula", width="large"),
                "stress_preset": st.column_config.TextColumn("Stress preset", width="large"),
                "preset_key": st.column_config.TextColumn("Preset key"),
            },
        )
        selected_rows = edited_preset_frame.loc[edited_preset_frame["include"]].copy()
        selected_presets = [preset_map[str(key)] for key in selected_rows["preset_key"].tolist() if str(key) in preset_map]
        auto_stress_presets = [
            preset_map[str(key)].stress_preset.to_term_dict()
            for _, row in selected_rows.iterrows()
            for key in [row["preset_key"]]
            if bool(row["auto_stress"]) and str(key) in preset_map and preset_map[str(key)].stress_preset is not None
        ]
    else:
        selected_presets = []
        auto_stress_presets = []
        st.info("No preset derived features are currently calculable from the recognized columns on this sheet.")

    with st.expander("Preset catalog and input requirements", expanded=not bool(available_presets)):
        render_dataframe(_preset_catalog_frame(preset_catalog))

    derived_count = st.number_input(
        "Number of derived columns",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        key="derived_column_count",
    )
    derived_specs: list[dict] = []
    for index in range(int(derived_count)):
        with st.expander(f"Derived column {index + 1}", expanded=True):
            name = st.text_input("Column name", value=f"derived_{index + 1}", key=f"derived_name_{index}")
            formula = st.text_input("Formula", key=f"derived_formula_{index}")
            derived_specs.append({"name": name, "formula": formula})

    processed_df = df.copy()
    notes: list[str] = []
    error: str | None = None
    active_specs = (
        [{"name": preset.name, "formula": preset.formula} for preset in selected_presets]
        + [spec for spec in derived_specs if str(spec.get("name", "")).strip() or str(spec.get("formula", "")).strip()]
    )
    if active_specs:
        try:
            processed_df, notes = apply_derived_columns(df, active_specs)
        except Exception as exc:
            error = str(exc)
            st.error(error)
            st.warning("Fix the derived column definitions above before fitting the model.")
        else:
            for note in notes:
                st.caption(note)
            st.download_button(
                "Download workbook with processed selected sheet",
                data=build_processed_workbook(sheets, selected_sheet, processed_df),
                file_name=f"processed_{selected_sheet}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
            with st.expander("Processed data preview", expanded=True):
                render_dataframe(processed_df.head(50))

    return processed_df, derived_specs, auto_stress_presets, notes, error


def build_event_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    event_mode = st.radio(
        "Failure Flag source",
        options=["Use existing Failure Flag column", "Create Failure Flag column from another field"],
        horizontal=True,
        key="event_mode",
    )

    if event_mode == "Use existing Failure Flag column":
        event_column = st.selectbox("Failure Flag column", options=df.columns.tolist(), key="event_column")
        return df.copy(), event_column

    source_column = st.selectbox("Status/source column", options=df.columns.tolist(), key="event_source_column")
    raw_values = df[source_column].dropna().astype(str).sort_values().unique().tolist()
    default_failure_values = [value for value in raw_values if "fail" in value.casefold()]
    failure_values = st.multiselect(
        "Values that mean failure",
        options=raw_values,
        default=default_failure_values,
        key="failure_values",
    )

    derived = df.copy()
    derived["_event"] = derived[source_column].astype(str).isin({str(value) for value in failure_values}).astype(int)
    return derived, "_event"


def render_filters(df: pd.DataFrame) -> list[dict]:
    st.subheader("Filters")
    filter_count = st.number_input("Number of filters", min_value=0, max_value=8, value=0, step=1, key="filter_count")
    filters: list[dict] = []
    columns = df.columns.tolist()

    for index in range(int(filter_count)):
        filter_columns = st.columns([2, 1.3, 1.7])
        column = filter_columns[0].selectbox("Column", options=columns, key=f"filter_column_{index}")
        operator = filter_columns[1].selectbox("Operator", options=FILTER_OPERATORS, key=f"filter_operator_{index}")
        if operator in {"is_null", "not_null"}:
            value = None
            filter_columns[2].markdown("Value not required")
        else:
            value = filter_columns[2].text_input("Value", key=f"filter_value_{index}")
        filters.append({"column": column, "operator": operator, "value": value})

    if filters:
        preview_df = apply_filters(df, filters)
        st.caption(f"Rows after current filter preview: {len(preview_df):,}")
    return filters


def render_stress_terms(df: pd.DataFrame, auto_stress_terms: list[dict] | None = None) -> list[dict]:
    st.subheader("Stress Terms")
    auto_stress_terms = auto_stress_terms or []
    numeric_candidates = [
        column for column in df.columns if pd.to_numeric(df[column], errors="coerce").notna().sum() > 0
    ]

    st.caption(
        "Recommended stress terms from both raw input columns and derived features are listed below. "
        "You can edit the table directly, disable rows, or add fully custom rows."
    )

    default_rows = _collect_stress_term_rows(df, auto_stress_terms=auto_stress_terms)
    if not default_rows:
        default_rows = [_manual_stress_term_row(df, numeric_candidates)]
    editor_key = _stress_editor_key(df, auto_stress_terms=auto_stress_terms)
    edited_terms = st.data_editor(
        pd.DataFrame(default_rows),
        key=editor_key,
        width="stretch",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "include": st.column_config.CheckboxColumn("Include", default=True),
            "source": st.column_config.TextColumn("Source"),
            "category": st.column_config.TextColumn("Category"),
            "name": st.column_config.TextColumn("Name"),
            "column": st.column_config.SelectboxColumn("Data column", options=[""] + numeric_candidates),
            "transform": st.column_config.SelectboxColumn("Transform", options=sorted(TRANSFORM_LIBRARY)),
            "reference_mode": st.column_config.SelectboxColumn("Reference mode", options=["fixed", "column", "fit"]),
            "reference_value": st.column_config.NumberColumn("Reference value", format="%.6f"),
            "reference_column": st.column_config.SelectboxColumn("Reference column", options=[""] + numeric_candidates),
            "reference_init": st.column_config.NumberColumn("Reference init", format="%.6f"),
            "reference_lower_bound": st.column_config.NumberColumn("Reference lower", format="%.6f"),
            "reference_upper_bound": st.column_config.NumberColumn("Reference upper", format="%.6f"),
            "reference_multiplier_mode": st.column_config.SelectboxColumn("Ref multiplier mode", options=["fixed", "fit"]),
            "reference_multiplier_value": st.column_config.NumberColumn("Ref multiplier", format="%.6f"),
            "reference_multiplier_init": st.column_config.NumberColumn("Ref mult init", format="%.6f"),
            "reference_multiplier_lower_bound": st.column_config.NumberColumn("Ref mult lower", format="%.6f"),
            "reference_multiplier_upper_bound": st.column_config.NumberColumn("Ref mult upper", format="%.6f"),
            "coefficient_mode": st.column_config.SelectboxColumn("Coeff mode", options=["fit", "fixed"]),
            "coefficient_value": st.column_config.NumberColumn("Coeff value", format="%.6f"),
            "coefficient_non_negative": st.column_config.CheckboxColumn("Coeff >= 0", default=True),
            "coefficient_lower_bound": st.column_config.NumberColumn("Coeff lower", format="%.6f"),
            "coefficient_upper_bound": st.column_config.NumberColumn("Coeff upper", format="%.6f"),
            "scale": st.column_config.NumberColumn("Scale", format="%.6f"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    terms, errors = _stress_terms_from_editor_frame(edited_terms)
    for error in errors:
        st.warning(error)
    return terms


def render_transform_selection(
    df: pd.DataFrame,
    *,
    duration_column: str,
    event_column: str,
    group_columns: list[str],
    filters: list[dict],
    min_group_size: int,
) -> None:
    st.subheader("Transform Selection")
    st.caption(
        "Score candidate transforms for one stress variable at a time. "
        "Each candidate is fit with `reference_mode=fit`, `reference_init=median`, `reference_lower=min`, and `reference_upper=max` on the current filtered dataset."
    )

    numeric_candidates = [
        column for column in df.columns if pd.to_numeric(df[column], errors="coerce").notna().sum() > 0
    ]
    if not numeric_candidates:
        st.info("No numeric columns are available for transform selection.")
        return

    control_cols = st.columns(4)
    scan_column = control_cols[0].selectbox("Stress column", options=numeric_candidates, key="transform_scan_column")
    transform_candidates = control_cols[1].multiselect(
        "Candidate transforms",
        options=sorted(TRANSFORM_LIBRARY),
        default=sorted(TRANSFORM_LIBRARY),
        key="transform_scan_candidates",
    )
    coefficient_non_negative = control_cols[2].checkbox(
        "Keep coefficient non-negative",
        value=True,
        key="transform_scan_non_negative",
    )
    scale_value = control_cols[3].number_input(
        "Scale",
        min_value=0.000001,
        value=1.0,
        key="transform_scan_scale",
    )

    if not transform_candidates:
        st.info("Choose at least one candidate transform to run the ranking.")
        return

    if st.button("Rank transforms", key="transform_scan_run"):
        try:
            ranked = rank_transform_candidates(
                df,
                duration_column=duration_column,
                event_column=event_column,
                column=scan_column,
                group_columns=group_columns,
                filters=filters,
                min_group_size=int(min_group_size),
                candidate_transforms=transform_candidates,
                coefficient_non_negative=coefficient_non_negative,
                scale=float(scale_value),
            )
        except Exception as exc:
            st.error(str(exc))
            return

        result_frame = pd.DataFrame(
            [
                {
                    "transform": item.transform,
                    "success": item.success,
                    "nll": item.nll,
                    "aic": item.aic,
                    "bic": item.bic,
                    "delta_nll_vs_baseline": item.delta_nll_vs_baseline,
                    "delta_aic_vs_baseline": item.delta_aic_vs_baseline,
                    "delta_bic_vs_baseline": item.delta_bic_vs_baseline,
                    "coefficient": item.coefficient,
                    "reference_value": item.reference_value,
                    "message": item.message,
                }
                for item in ranked
            ]
        )
        render_dataframe(result_frame)
        if not result_frame.empty:
            best_row = result_frame.iloc[0]
            st.caption(
                "Preferred transform by this ranking: "
                f"`{best_row['transform']}` with delta AIC `{best_row['delta_aic_vs_baseline']:.2f}` "
                f"and delta NLL `{best_row['delta_nll_vs_baseline']:.2f}` versus the no-stress baseline."
            )


def render_model_setup() -> None:
    st.title("Interpretable Weibull Reliability Model")
    st.caption("Base reliability by category multiplied by an operating stress penalty.")

    uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded is None:
        st.info("Upload a workbook to begin.")
        return

    file_bytes = uploaded.getvalue()
    sheets = load_workbook(file_bytes)
    sheet_name = st.selectbox("Sheet", options=list(sheets.keys()))
    raw_df = sheets[sheet_name].copy()
    raw_df.columns = [str(column) for column in raw_df.columns]
    maybe_apply_default_preset(raw_df, sheet_name)

    st.subheader("Preview")
    render_dataframe(raw_df.head(50))

    processed_df, derived_specs, auto_stress_terms, _, postprocess_error = render_input_postprocessor(raw_df, sheets, sheet_name)
    analysis_source_df = processed_df if postprocess_error is None else raw_df
    normalize_widget_state_for_columns(analysis_source_df)

    working_df, event_column = build_event_frame(analysis_source_df)
    duration_column = st.selectbox("Duration column", options=working_df.columns.tolist(), key="duration_column")
    group_columns = st.multiselect("Group columns", options=working_df.columns.tolist(), key="group_columns")
    min_group_size = st.number_input("Minimum group size before fallback", min_value=1, value=20, step=1)

    filters = render_filters(working_df)
    stress_terms = render_stress_terms(working_df, auto_stress_terms=auto_stress_terms)
    if duration_column and event_column:
        render_transform_selection(
            working_df,
            duration_column=duration_column,
            event_column=event_column,
            group_columns=group_columns,
            filters=filters,
            min_group_size=int(min_group_size),
        )

    if postprocess_error is None and duration_column and event_column:
        try:
            preview_df, preview_summary = prepare_modeling_dataframe(
                working_df,
                duration_column=duration_column,
                event_column=event_column,
                group_columns=group_columns,
                stress_terms=stress_terms,
                filters=filters,
                min_group_size=int(min_group_size),
            )
            metric_cols = st.columns(4)
            metric_cols[0].metric("Rows before filtering", f"{preview_summary['rows_before_filtering']:,}")
            metric_cols[1].metric("Rows after filtering", f"{preview_summary['rows_after_filtering']:,}")
            metric_cols[2].metric("Failures", f"{preview_summary['failure_count']:,}")
            metric_cols[3].metric("Censored", f"{preview_summary['censored_count']:,}")
            if preview_summary["notes"]:
                for note in preview_summary["notes"]:
                    st.caption(note)
            with st.expander("Prepared data preview"):
                render_dataframe(preview_df.head(50))
        except Exception as exc:
            st.warning(str(exc))
    elif postprocess_error is not None:
        st.info("Model preparation is paused until the derived column formulas are valid.")

    if st.button("Fit Weibull-Stress Model", type="primary", width="stretch"):
        if postprocess_error is not None:
            st.session_state["weibull_fit_error"] = "Fix derived column formulas before fitting the model."
            st.session_state.pop("weibull_fit_result", None)
        else:
            try:
                result = fit_weibull_stress_model(
                    working_df,
                    duration_column=duration_column,
                    event_column=event_column,
                    group_columns=group_columns,
                    stress_terms=stress_terms,
                    filters=filters,
                    min_group_size=int(min_group_size),
                )
            except Exception as exc:
                st.session_state["weibull_fit_error"] = str(exc)
                st.session_state.pop("weibull_fit_result", None)
            else:
                st.session_state["weibull_fit_result"] = result
                st.session_state["weibull_fit_error"] = None
                st.success("Model fit completed and stored in the current session.")

    if st.session_state.get("weibull_fit_error"):
        st.error(st.session_state["weibull_fit_error"])


def render_model_results() -> None:
    st.title("Model Results")
    result = st.session_state.get("weibull_fit_result")
    if result is None:
        st.info("Fit a model in the first tab to inspect results.")
        return

    hierarchy_summary = build_hierarchy_details(result)
    levels = hierarchy_summary["tree_level"].drop_duplicates().tolist()
    selected_level = st.selectbox("Hierarchy level", options=levels, key="results_tree_level")
    level_summary = hierarchy_summary.loc[hierarchy_summary["tree_level"] == selected_level].copy()
    selected_key = st.selectbox("Node", options=level_summary["tree_key"].tolist(), key="results_tree_key")
    selected_summary = level_summary.loc[level_summary["tree_key"] == selected_key].iloc[0]
    node_rows = build_hierarchy_rows(result, str(selected_level), str(selected_key))
    if node_rows.empty:
        st.warning("No rows were found for the selected node.")
        return

    group_columns = [column for column in result.group_columns if column in result.prepared_df.columns]
    original_combo_level = (
        f"{group_columns[0]} + {group_columns[1]}"
        if len(group_columns) >= 2
        else (group_columns[0] if group_columns else "GLOBAL")
    )
    shared_available = selected_level == original_combo_level and selected_key in result.groups

    durations = pd.to_numeric(node_rows[result.duration_column], errors="coerce").to_numpy(dtype=float)
    events = pd.to_numeric(node_rows[result.event_column], errors="coerce").to_numpy(dtype=int)

    if shared_available:
        stats = result.group_stats[selected_key]
        default_curve_eta = result.adjusted_eta(selected_key)
        local_fit = result.fit_group_only_weibull(selected_key)
        shared_eval = result.evaluate_group_parameters(selected_key, result.beta_for_group(selected_key), default_curve_eta)
        display_mode = st.radio(
            "Parameter display mode",
            options=["Shared grouped model", "Selection-only Weibull refit"],
            horizontal=True,
            key=f"display_mode_{selected_key}",
        )
        if display_mode == "Shared grouped model":
            displayed_beta = float(result.beta_for_group(selected_key))
            displayed_eta = float(default_curve_eta)
            displayed_eta_baseline = float(result.eta_by_group[stats["eta_group"]])
            displayed_nll = float(shared_eval["nll"])
            displayed_aic = float(shared_eval["aic"])
            displayed_bic = float(shared_eval["bic"])
            displayed_caption = (
                "Showing the fitted original group model: beta belongs to this exact category combination, "
                "while eta comes from its fallback eta bucket."
            )
        else:
            displayed_beta = float(local_fit["beta"])
            displayed_eta = float(local_fit["eta"])
            displayed_eta_baseline = float(local_fit["eta"])
            displayed_nll = float(local_fit["nll"])
            displayed_aic = float(local_fit["aic"])
            displayed_bic = float(local_fit["bic"])
            displayed_caption = (
                "Showing a Weibull refit using only the rows inside the selected node."
            )
    else:
        local_fit = fit_basic_weibull(durations, events)
        display_mode = "Selection-only Weibull refit"
        displayed_beta = float(local_fit["beta"])
        displayed_eta = float(local_fit["eta"])
        displayed_eta_baseline = float(local_fit["eta"])
        displayed_nll = float(local_fit["nll"])
        displayed_aic = float(local_fit["aic"])
        displayed_bic = float(local_fit["bic"])
        displayed_caption = (
            "This hierarchy node aggregates multiple original groups, so the page shows a Weibull refit on the selected rows."
        )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Beta", f"{displayed_beta:.4f}")
    metric_cols[1].metric("Eta", f"{displayed_eta_baseline:.2f}")
    metric_cols[2].metric("Rows", f"{int(selected_summary['rows']):,}")
    metric_cols[3].metric("Failures", f"{int(selected_summary['failures']):,}")
    metric_cols[4].metric("Censored", f"{int(selected_summary['censored']):,}")
    metric_cols[5].metric("NLL", f"{displayed_nll:.2f}")

    if shared_available:
        st.caption(
            "For an exact original group, beta belongs to that exact combination and eta comes from the fallback eta bucket."
        )
        st.caption(f"Eta bucket: {stats['eta_group']} ({stats['bucket_rows']} rows)")
    st.caption(displayed_caption)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.metric("AIC", f"{displayed_aic:.2f}")
        st.metric("BIC", f"{displayed_bic:.2f}")
        st.metric("Curve eta", f"{displayed_eta:.2f}")
        st.metric("Mean duration", f"{float(selected_summary['mean_duration']):.2f}")
        st.metric("Median duration", f"{float(selected_summary['median_duration']):.2f}")
        st.metric("Std duration", f"{0.0 if pd.isna(selected_summary['std_duration']) else float(selected_summary['std_duration']):.2f}")
        if shared_available and display_mode == "Shared grouped model":
            st.write(f"Optimizer success: `{result.success}`")
            st.write(result.message)
        else:
            st.write(f"Optimizer success: `{bool(local_fit['success'])}`")
            st.write(str(local_fit["message"]))
    with detail_cols[1]:
        coefficients = pd.DataFrame(
            [{"term": key, "coefficient": value} for key, value in result.stress_coefficients.items()]
        )
        references = pd.DataFrame(
            [{"term": key, "reference_value": value} for key, value in result.reference_values.items()]
        )
        multipliers = pd.DataFrame(
            [{"term": key, "reference_multiplier": value} for key, value in result.reference_multipliers.items()]
        )
        if not coefficients.empty:
            render_dataframe(coefficients)
        if not references.empty:
            render_dataframe(references)
        if not multipliers.empty:
            render_dataframe(multipliers)

    if not node_rows.empty:
        with st.expander("Rows behind this selected node"):
            visible_columns = [
                column
                for column in [
                    "analysis_group_key",
                    "analysis_original_group",
                    "analysis_group_level",
                    result.duration_column,
                    result.event_column,
                ]
                if column in node_rows.columns
            ]
            render_dataframe(node_rows[visible_columns].head(200))

    stage_table = pd.DataFrame([stage for stage in result.to_dict()["stage_summaries"]])
    st.subheader("Staged fit summary")
    render_dataframe(stage_table)

    st.subheader("Distribution plots")
    plot_cols = st.columns(2)
    plot_beta = plot_cols[0].number_input(
        "Plot beta",
        min_value=0.0001,
        value=float(displayed_beta),
        step=0.05,
        key=f"plot_beta_{selected_level}_{selected_key}_{display_mode}",
    )
    plot_eta = plot_cols[1].number_input(
        "Plot eta",
        min_value=0.0001,
        value=float(displayed_eta),
        step=max(float(displayed_eta) * 0.05, 1.0),
        key=f"plot_eta_{selected_level}_{selected_key}_{display_mode}",
    )
    st.caption("These controls only replot the current Weibull curves for this selection. They do not refit the model.")
    st.plotly_chart(
        build_distribution_figure(
            result,
            selected_key if shared_available else result.groups[0],
            beta_override=float(plot_beta),
            eta_override=float(plot_eta),
            group_df_override=node_rows,
            title_label=f"{selected_level}: {selected_key}",
        ),
        width="stretch",
    )

    st.subheader("Selection Diagnostic")
    plotted_nll = evaluate_weibull_nll(durations, events, float(plot_beta), float(plot_eta))

    if shared_available:
        st.caption(
            "The diagnostics below compare the fitted original-group model against a Weibull refit using only the currently selected rows."
        )
        diag_cols = st.columns(3)
        with diag_cols[0]:
            st.write("Fitted original-group model")
            st.metric("Beta", f"{result.beta_for_group(selected_key):.4f}")
            st.metric("Eta", f"{default_curve_eta:.2f}")
            st.metric("Selection NLL", f"{shared_eval['nll']:.2f}")
        with diag_cols[1]:
            st.write("Currently plotted values")
            st.metric("Beta", f"{float(plot_beta):.4f}")
            st.metric("Eta", f"{float(plot_eta):.2f}")
            st.metric("Selection NLL", f"{plotted_nll:.2f}")
        with diag_cols[2]:
            st.write("Selection-only Weibull fit")
            st.metric("Beta", f"{float(local_fit['beta']):.4f}")
            st.metric("Eta", f"{float(local_fit['eta']):.2f}")
            st.metric("Selection NLL", f"{float(local_fit['nll']):.2f}")

        if float(plotted_nll) < float(shared_eval["nll"]):
            st.info(
                "For this selected node, the currently plotted beta/eta has a lower negative log-likelihood than the fitted original-group model."
            )
        else:
            st.info(
                "For this selected node, the fitted original-group model still has a lower negative log-likelihood than the currently plotted beta/eta."
            )
    else:
        st.caption("For aggregated hierarchy nodes, diagnostics compare the current plot values against the node-only Weibull refit.")
        diag_cols = st.columns(2)
        with diag_cols[0]:
            st.write("Currently plotted values")
            st.metric("Beta", f"{float(plot_beta):.4f}")
            st.metric("Eta", f"{float(plot_eta):.2f}")
            st.metric("Selection NLL", f"{plotted_nll:.2f}")
        with diag_cols[1]:
            st.write("Node-only Weibull fit")
            st.metric("Beta", f"{float(local_fit['beta']):.4f}")
            st.metric("Eta", f"{float(local_fit['eta']):.2f}")
            st.metric("Selection NLL", f"{float(local_fit['nll']):.2f}")


def render_sensitivity() -> None:
    st.title("Stress Sensitivity")
    result = st.session_state.get("weibull_fit_result")
    if result is None:
        st.info("Fit a model in the first tab to inspect sensitivity curves.")
        return
    if not result.stress_terms:
        st.info("Sensitivity analysis becomes available once the fitted model includes at least one stress term.")
        return

    group_labels = build_group_option_labels(result)
    group_key = st.selectbox(
        "Original group",
        options=result.groups,
        format_func=lambda key: group_labels[key],
        key="sensitivity_group_key",
    )
    stress_term_name = st.selectbox(
        "Stress term",
        options=[term.name for term in result.stress_terms],
        key="sensitivity_term_name",
    )
    term = next((item for item in result.stress_terms if item.name == stress_term_name), None)
    if term is None:
        st.error("Selected stress term was not found.")
        return

    group_df = result.group_source_rows(group_key)
    suggested_values = suggest_sensitivity_values(group_df, term.column)
    suggested_text = format_sensitivity_values(suggested_values) if suggested_values else ""
    value_text = st.text_input(
        "Scenario values",
        value=suggested_text,
        key=f"sensitivity_values_{group_key}_{stress_term_name}",
    )
    if suggested_values:
        st.caption(
            f"Default value is the selected group's median `{term.column}`."
        )

    try:
        values = parse_numeric_list(value_text)
    except ValueError:
        st.error("Scenario values must be a comma-separated list of numbers.")
        return

    if not values:
        st.warning("Enter at least one scenario value.")
        return

    baseline = result.representative_row(group_key)
    model_coefficient, model_reference_value, model_reference_multiplier = result.resolve_term_parameters(
        stress_term_name,
        row=baseline,
    )

    st.subheader("Sensitivity Parameters")
    parameter_cols = st.columns(3)
    used_coefficient = parameter_cols[0].number_input(
        "Stress coefficient",
        value=float(model_coefficient),
        step=max(abs(float(model_coefficient)) * 0.1, 0.01),
        key=f"sensitivity_coefficient_{group_key}_{stress_term_name}",
    )
    used_reference_value = parameter_cols[1].number_input(
        "Reference value",
        value=float(model_reference_value),
        step=max(abs(float(model_reference_value)) * 0.05, 0.1),
        key=f"sensitivity_reference_value_{group_key}_{stress_term_name}",
    )
    used_reference_multiplier = model_reference_multiplier
    if term.reference_mode == "column":
        used_reference_multiplier = parameter_cols[2].number_input(
            "Reference multiplier",
            value=float(model_reference_multiplier),
            step=max(abs(float(model_reference_multiplier)) * 0.05, 0.01),
            key=f"sensitivity_reference_multiplier_{group_key}_{stress_term_name}",
        )
    else:
        parameter_cols[2].metric("Reference mode", term.reference_mode)

    parameter_report = pd.DataFrame(
        [
            {
                "parameter": "stress_coefficient",
                "model_value": float(model_coefficient),
                "used_value": float(used_coefficient),
            },
            {
                "parameter": "reference_value",
                "model_value": float(model_reference_value),
                "used_value": float(used_reference_value),
            },
            {
                "parameter": "reference_multiplier",
                "model_value": float(model_reference_multiplier),
                "used_value": float(used_reference_multiplier),
            },
        ]
    )
    render_dataframe(parameter_report)

    term_overrides = {
        "coefficient": float(used_coefficient),
        "reference_value": float(used_reference_value),
        "reference_multiplier": float(used_reference_multiplier),
    }
    st.subheader("Scenario Plot")
    st.caption(
        "Model scenario curves and empirical range splits are overlaid on the same figure. "
        "Model scenarios use the exact selected values, while empirical scenarios use factual ranges around those values."
    )
    model_frame = build_model_sensitivity_frame(
        result,
        group_key,
        stress_term_name,
        values,
        term_overrides=term_overrides,
    )
    combined_fig = build_sensitivity_figure(
        result,
        group_key,
        stress_term_name,
        values,
        term_overrides=term_overrides,
    )
    st.plotly_chart(combined_fig, width="stretch")

    st.subheader("Model Scenario Parameters")
    render_dataframe(model_frame.rename(columns={"scenario_value": term.column}))

    split_df = build_sensitivity_range_split_frame(group_df, term.column, values)
    if not split_df.empty:
        st.subheader("Sensitivity Split")
        split_summary = (
            split_df.groupby("sensitivity_range_label", sort=True)
            .agg(
                rows=(term.column, "size"),
                min_value=(term.column, "min"),
                median_value=(term.column, "median"),
                max_value=(term.column, "max"),
                failures=(result.event_column, "sum"),
            )
            .reset_index()
            .rename(columns={"sensitivity_range_label": "range"})
        )
        render_dataframe(split_summary)

        with st.expander("Rows assigned to sensitivity ranges"):
            visible_columns = [
                column
                for column in [
                    "analysis_original_group",
                    "analysis_group_key",
                    term.column,
                    "sensitivity_range_label",
                    result.duration_column,
                    result.event_column,
                ]
                if column in split_df.columns
            ]
            render_dataframe(split_df[visible_columns].sort_values(["sensitivity_range_label", term.column]))

    with st.expander("Representative baseline row"):
        render_dataframe(pd.DataFrame([baseline]))


def render_distribution_details() -> None:
    st.title("Distribution Details")
    result = st.session_state.get("weibull_fit_result")
    if result is None:
        st.info("Fit a model in the first tab to inspect hierarchy-level distribution details.")
        return

    summary = build_hierarchy_details(result)
    if summary.empty:
        st.info("No distribution details are available for the current model.")
        return

    st.subheader("Hierarchy Summary")
    render_dataframe(summary)

    levels = summary["tree_level"].drop_duplicates().tolist()
    selected_level = st.selectbox("Hierarchy level", options=levels, key="distribution_tree_level")
    level_summary = summary.loc[summary["tree_level"] == selected_level].copy()
    selected_key = st.selectbox("Node", options=level_summary["tree_key"].tolist(), key="distribution_tree_key")

    selected_summary = level_summary.loc[level_summary["tree_key"] == selected_key].iloc[0]
    metric_cols = st.columns(7)
    metric_cols[0].metric("Rows", f"{int(selected_summary['rows']):,}")
    metric_cols[1].metric("Failures", f"{int(selected_summary['failures']):,}")
    metric_cols[2].metric("Censored", f"{int(selected_summary['censored']):,}")
    metric_cols[3].metric("Mean", f"{float(selected_summary['mean_duration']):.2f}")
    metric_cols[4].metric("Median", f"{float(selected_summary['median_duration']):.2f}")
    metric_cols[5].metric("Std", f"{0.0 if pd.isna(selected_summary['std_duration']) else float(selected_summary['std_duration']):.2f}")
    metric_cols[6].metric("Range", f"{float(selected_summary['min_duration']):.2f} - {float(selected_summary['max_duration']):.2f}")

    rows = build_hierarchy_rows(result, str(selected_level), str(selected_key))
    if not rows.empty:
        st.subheader("Node Rows")
        visible_columns = [
            column
            for column in (
                result.group_columns
                + [
                    "analysis_original_group",
                    "analysis_group_key",
                    result.duration_column,
                    result.event_column,
                ]
            )
            if column in rows.columns
        ]
        render_dataframe(rows[visible_columns].head(500))


def render_stress_correlations() -> None:
    st.title("Stress Correlations")
    result = st.session_state.get("weibull_fit_result")
    if result is None:
        st.info("Fit a model in the first tab to inspect stress-term correlations.")
        return
    if not result.stress_terms:
        st.info("Stress-term correlations become available once the fitted model includes at least one stress term.")
        return

    numeric_targets = [
        column
        for column in result.prepared_df.columns
        if pd.to_numeric(result.prepared_df[column], errors="coerce").notna().sum() > 0
    ]
    if not numeric_targets:
        st.info("No numeric target columns are available in the prepared model dataframe.")
        return

    control_cols = st.columns(4)
    term_name = control_cols[0].selectbox(
        "Stress term",
        options=[term.name for term in result.stress_terms],
        key="stress_corr_term_name",
    )
    default_target = result.duration_column if result.duration_column in numeric_targets else numeric_targets[0]
    target_column = control_cols[1].selectbox(
        "Target column",
        options=numeric_targets,
        index=numeric_targets.index(default_target),
        key="stress_corr_target_column",
    )
    value_column = control_cols[2].selectbox(
        "Stress value",
        options=["weighted_stress_value", "stress_value"],
        index=0,
        key="stress_corr_value_column",
    )
    group_options = ["<none>"] + [column for column in result.group_columns if column in result.prepared_df.columns]
    group_column = control_cols[3].selectbox("Group column", options=group_options, key="stress_corr_group_column")
    selected_group_column = None if group_column == "<none>" else group_column

    try:
        observation_df = build_stress_term_observation_frame(result, term_name, target_column=target_column)
    except Exception as exc:
        st.error(str(exc))
        return

    if observation_df.empty:
        st.warning("No valid rows were available to compute this stress-term correlation view.")
        return

    correlation_frame = build_grouped_stress_correlation_frame(
        observation_df,
        value_column=value_column,
        group_column=selected_group_column,
    )
    if correlation_frame.empty:
        st.warning("The selected stress term and target column do not have enough variation to compute correlations.")
        return

    metric_cols = st.columns(4)
    overall_row = correlation_frame.iloc[0]
    metric_cols[0].metric("Rows", f"{int(overall_row['rows']):,}")
    metric_cols[1].metric("Pearson", f"{float(overall_row['pearson_corr']):.3f}")
    metric_cols[2].metric("Spearman", f"{float(overall_row['spearman_corr']):.3f}")
    metric_cols[3].metric("Mean stress", f"{float(overall_row['mean_stress']):.4f}")

    render_dataframe(correlation_frame)

    plot_df = observation_df.copy()
    if selected_group_column is not None and selected_group_column in plot_df.columns:
        plot_df["plot_group"] = plot_df[selected_group_column].astype("string").fillna("<missing>")
    else:
        plot_df["plot_group"] = _event_label_series(plot_df, result.event_column)

    hover_columns = [
        column
        for column in [
            "analysis_original_group",
            "analysis_group_key",
            result.duration_column,
            result.event_column,
            "stress_reference_value",
            "stress_coefficient",
        ]
        if column in plot_df.columns
    ]
    fig = px.scatter(
        plot_df,
        x=value_column,
        y="target_value",
        color="plot_group",
        hover_data=hover_columns,
        opacity=0.8,
        labels={
            value_column: "Stress term value" if value_column == "stress_value" else "Weighted stress contribution",
            "target_value": target_column,
            "plot_group": selected_group_column or result.event_column,
        },
    )
    if plot_df[value_column].nunique() >= 2 and plot_df["target_value"].nunique() >= 2:
        x_values = plot_df[value_column].to_numpy(dtype=float)
        y_values = plot_df["target_value"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_line = intercept + (slope * x_line)
        fig.add_scatter(x=x_line, y=y_line, mode="lines", name="Linear trend", line={"color": "#111827", "width": 2})
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")


tab_setup, tab_results, tab_sensitivity, tab_correlations, tab_distribution = st.tabs(
    ["Data & Model Setup", "Model Results", "Stress Sensitivity", "Stress Correlations", "Distribution Details"]
)

with tab_setup:
    render_model_setup()

with tab_results:
    render_model_results()

with tab_sensitivity:
    render_sensitivity()

with tab_correlations:
    render_stress_correlations()

with tab_distribution:
    render_distribution_details()
