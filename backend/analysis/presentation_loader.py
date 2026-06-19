from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import normalize_event_series
from .presentation_schema import (
    ROLE_COLUMN_ALIASES,
    ROLE_DATE_COLUMNS,
    ROLE_NUMERIC_COLUMNS,
    SHEET_SPECS,
    build_default_gas_limits_frame,
    normalize_gas_handling_type,
    normalize_name,
)


@dataclass(slots=True)
class PresentationSheetReport:
    role: str
    required: bool
    detected_sheet: str | None
    detection_method: str
    detection_score: float
    row_count: int
    mapped_columns: dict[str, str]
    missing_required_columns: list[str]
    missing_recommended_columns: list[str]
    join_key: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "required": self.required,
            "detected_sheet": self.detected_sheet or "",
            "detection_method": self.detection_method,
            "detection_score": round(float(self.detection_score), 3),
            "row_count": int(self.row_count),
            "mapped_columns": len(self.mapped_columns),
            "missing_required_columns": ", ".join(self.missing_required_columns),
            "missing_recommended_columns": ", ".join(self.missing_recommended_columns),
            "join_key": self.join_key or "",
            "notes": "; ".join(self.notes),
        }


@dataclass(slots=True)
class PresentationPreparationResult:
    available_sheets: list[str]
    sheet_reports: list[PresentationSheetReport]
    merged_df: pd.DataFrame = field(repr=False)
    runs_df: pd.DataFrame = field(repr=False)
    design_df: pd.DataFrame | None = field(default=None, repr=False)
    regime_df: pd.DataFrame | None = field(default=None, repr=False)
    telemetry_df: pd.DataFrame | None = field(default=None, repr=False)
    gas_limits_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    notes: list[str] = field(default_factory=list)
    suggested_event_column: str = "event"
    suggested_duration_column: str = "TTF_days"
    suggested_group_columns: list[str] = field(default_factory=list)


def _find_matching_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized_candidates = {normalize_name(alias): alias for alias in aliases}
    for column in columns:
        if normalize_name(column) in normalized_candidates:
            return column
    return None


def map_role_columns(df: pd.DataFrame, role: str) -> dict[str, str]:
    aliases = ROLE_COLUMN_ALIASES[role]
    columns = [str(column) for column in df.columns.tolist()]
    mapping: dict[str, str] = {}
    used_columns: set[str] = set()

    for canonical, options in aliases.items():
        if canonical in columns and canonical not in used_columns:
            mapping[canonical] = canonical
            used_columns.add(canonical)
            continue
        matched = _find_matching_column(columns, options)
        if matched is not None and matched not in used_columns:
            mapping[canonical] = matched
            used_columns.add(matched)
    return mapping


def standardize_role_dataframe(df: pd.DataFrame, role: str) -> tuple[pd.DataFrame, dict[str, str]]:
    standardized = df.copy()
    standardized.columns = [str(column) for column in standardized.columns]
    mapping = map_role_columns(standardized, role)
    rename_map = {actual: canonical for canonical, actual in mapping.items() if actual != canonical}
    if rename_map:
        standardized = standardized.rename(columns=rename_map)

    for column in ROLE_DATE_COLUMNS.get(role, []):
        if column in standardized.columns:
            standardized[column] = pd.to_datetime(standardized[column], errors="coerce")

    for column in ROLE_NUMERIC_COLUMNS.get(role, []):
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    if "gas_handling_type" in standardized.columns:
        standardized["gas_handling_type"] = standardized["gas_handling_type"].map(normalize_gas_handling_type)

    return standardized, mapping


def _sheet_name_score(sheet_name: str, role: str) -> float:
    normalized_sheet = normalize_name(sheet_name)
    spec = SHEET_SPECS[role]
    if normalized_sheet in {normalize_name(alias) for alias in spec.aliases}:
        return 1.0
    for alias in spec.aliases:
        normalized_alias = normalize_name(alias)
        if normalized_alias and normalized_alias in normalized_sheet:
            return 0.5
    return 0.0


def detect_sheet_roles(workbook: dict[str, pd.DataFrame]) -> dict[str, tuple[str, str, float]]:
    remaining = list(workbook.keys())
    detected: dict[str, tuple[str, str, float]] = {}

    for role, spec in SHEET_SPECS.items():
        exact_match = next(
            (
                sheet_name
                for sheet_name in remaining
                if normalize_name(sheet_name) in {normalize_name(alias) for alias in spec.aliases}
            ),
            None,
        )
        if exact_match is not None:
            detected[role] = (exact_match, "sheet_name", 1.0)
            remaining.remove(exact_match)

    for role, spec in SHEET_SPECS.items():
        if role in detected:
            continue
        best_sheet = None
        best_score = 0.0
        best_overlap = 0
        best_name_score = 0.0
        for sheet_name in remaining:
            sheet_df = workbook[sheet_name]
            mapped = map_role_columns(sheet_df, role)
            overlap = len(set(mapped).intersection(spec.key_columns))
            column_score = overlap / max(len(spec.key_columns), 1)
            name_score = _sheet_name_score(sheet_name, role)
            total_score = column_score + (0.25 * name_score)
            if total_score > best_score:
                best_score = total_score
                best_sheet = sheet_name
                best_overlap = overlap
                best_name_score = name_score
        if best_sheet is not None and (best_overlap >= 2 or best_name_score >= 0.5) and best_score >= 0.2:
            detected[role] = (best_sheet, "column_profile", float(best_score))
            remaining.remove(best_sheet)

    return detected


def _missing_role_columns(role: str, mapping: dict[str, str]) -> tuple[list[str], list[str]]:
    present = set(mapping)
    if role == "runs":
        missing_required = []
        if not {"run_id", "well"}.intersection(present):
            missing_required.append("run_id or well")
        if "event" not in present:
            missing_required.append("event")
        if "TTF_days" not in present and not ({"start_date", "stop_date"}.issubset(present) or {"start_date", "failure_date"}.issubset(present)):
            missing_required.append("TTF_days or start/stop dates")
        missing_recommended = [column for column in SHEET_SPECS[role].recommended_columns if column not in present]
        return missing_required, missing_recommended

    if role == "gas_limits":
        required = ["gas_handling_type", "free_gas_limit_fraction"]
    elif role == "design":
        required = ["run_id"]
    elif role == "daily_or_monthly_regime":
        required = ["run_id", "date"]
    elif role == "telemetry":
        required = ["run_id", "timestamp"]
    else:
        required = []
    missing_required = [column for column in required if column not in present]
    missing_recommended = [column for column in SHEET_SPECS[role].recommended_columns if column not in present]
    return missing_required, missing_recommended


def inspect_presentation_workbook(workbook: dict[str, pd.DataFrame]) -> list[PresentationSheetReport]:
    detections = detect_sheet_roles(workbook)
    reports: list[PresentationSheetReport] = []

    for role, spec in SHEET_SPECS.items():
        detected_sheet, detection_method, detection_score = detections.get(role, (None, "not_detected", 0.0))
        mapped_columns: dict[str, str] = {}
        row_count = 0
        missing_required: list[str] = []
        missing_recommended: list[str] = []
        notes: list[str] = []
        if detected_sheet is not None:
            standardized, mapped_columns = standardize_role_dataframe(workbook[detected_sheet], role)
            row_count = int(len(standardized))
            missing_required, missing_recommended = _missing_role_columns(role, mapped_columns)
            if standardized.empty:
                notes.append("Sheet is empty.")
        else:
            missing_required = ["sheet not detected"] if spec.required else []

        reports.append(
            PresentationSheetReport(
                role=role,
                required=spec.required,
                detected_sheet=detected_sheet,
                detection_method=detection_method,
                detection_score=float(detection_score),
                row_count=row_count,
                mapped_columns=mapped_columns,
                missing_required_columns=missing_required,
                missing_recommended_columns=missing_recommended,
                notes=notes,
            )
        )
    return reports


def _normalize_flag_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    truthy = {"1", "true", "yes", "y"}
    falsy = {"0", "false", "no", "n"}
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(truthy)] = True
    result.loc[normalized.isin(falsy)] = False
    return result


def _prepare_runs_frame(runs_df: pd.DataFrame, infant_threshold_days: float, notes: list[str]) -> pd.DataFrame:
    prepared = runs_df.copy()

    if "event" in prepared.columns:
        try:
            prepared["event"] = normalize_event_series(prepared["event"]).astype("Int64")
        except Exception as exc:
            notes.append(f"Event normalization warning: {exc}")

    if "TTF_days" not in prepared.columns:
        if "start_date" in prepared.columns and ("stop_date" in prepared.columns or "failure_date" in prepared.columns):
            end_series = (
                prepared["failure_date"]
                if "failure_date" in prepared.columns
                else pd.Series(pd.NaT, index=prepared.index, dtype="datetime64[ns]")
            )
            if "stop_date" in prepared.columns:
                end_series = end_series.combine_first(prepared["stop_date"])
            prepared["TTF_days"] = (end_series - prepared["start_date"]).dt.total_seconds() / 86400.0
            notes.append("Derived TTF_days from start_date and stop/failure date.")
        else:
            notes.append("TTF_days could not be derived because required date columns were not found.")
    else:
        prepared["TTF_days"] = pd.to_numeric(prepared["TTF_days"], errors="coerce")

    if "infant_mortality_flag" in prepared.columns:
        normalized_flags = _normalize_flag_series(prepared["infant_mortality_flag"])
        prepared["infant_mortality_flag"] = normalized_flags.fillna(
            pd.to_numeric(prepared.get("TTF_days"), errors="coerce") < float(infant_threshold_days)
        )
    elif "TTF_days" in prepared.columns:
        prepared["infant_mortality_flag"] = pd.to_numeric(prepared["TTF_days"], errors="coerce") < float(infant_threshold_days)
        notes.append(f"Derived infant_mortality_flag using TTF_days < {float(infant_threshold_days):g}.")

    for column in ("field", "contractor", "well", "run_id"):
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string")

    return prepared


def _choose_join_key(runs_df: pd.DataFrame, other_df: pd.DataFrame) -> str | None:
    for candidate in ("run_id", "well"):
        if candidate in runs_df.columns and candidate in other_df.columns:
            if runs_df[candidate].notna().any() and other_df[candidate].notna().any():
                return candidate
    return None


def _restrict_to_target_entities(other_df: pd.DataFrame, runs_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    if join_key not in other_df.columns or join_key not in runs_df.columns:
        return other_df.copy()
    target_keys = runs_df[join_key].dropna().astype("string").unique().tolist()
    if not target_keys:
        return other_df.iloc[0:0].copy()
    working = other_df.copy()
    return working.loc[working[join_key].astype("string").isin(target_keys)].copy()


def _deduplicate_design_frame(df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    working = df.copy()
    if "design_date" in working.columns:
        working = working.sort_values([join_key, "design_date"], kind="stable")
    return working.groupby(join_key, dropna=False, as_index=False).tail(1).reset_index(drop=True)


def _aggregate_timeseries(df: pd.DataFrame, join_key: str, time_column: str, prefix: str) -> pd.DataFrame:
    working = df.copy()
    working = working.loc[working[join_key].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key])

    if time_column in working.columns:
        working = working.sort_values([join_key, time_column], kind="stable")

    result = working.groupby(join_key, dropna=False).size().rename(f"{prefix}_record_count").reset_index()

    if time_column in working.columns:
        time_summary = (
            working.groupby(join_key, dropna=False)[time_column]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={"min": f"{prefix}_{time_column}_min", "max": f"{prefix}_{time_column}_max"})
        )
        result = result.merge(time_summary, on=join_key, how="left")

    excluded = {join_key, time_column, "status", "well"}
    numeric_candidates = []
    for column in working.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        if numeric.notna().sum() > 0:
            working[column] = numeric
            numeric_candidates.append(column)

    for column in numeric_candidates:
        stats = (
            working.groupby(join_key, dropna=False)[column]
            .agg(["mean", "min", "max", "median"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{prefix}_{column}_mean",
                    "min": f"{prefix}_{column}_min",
                    "max": f"{prefix}_{column}_max",
                    "median": f"{prefix}_{column}_median",
                }
            )
        )
        result = result.merge(stats, on=join_key, how="left")
        if time_column in working.columns:
            last_values = (
                working[[join_key, time_column, column]]
                .dropna(subset=[column])
                .groupby(join_key, dropna=False, as_index=False)
                .tail(1)[[join_key, column]]
                .rename(columns={column: f"{prefix}_{column}_last"})
            )
            result = result.merge(last_values, on=join_key, how="left")

    if "status" in working.columns:
        status_frame = working[[join_key, "status"]].copy()
        status_frame["status"] = status_frame["status"].astype("string").str.strip().str.casefold().fillna("unknown")
        counts = pd.crosstab(status_frame[join_key], status_frame["status"]).reset_index()
        renamed = {
            column: f"{prefix}_status_{normalize_name(column) or 'unknown'}_count"
            for column in counts.columns
            if column != join_key
        }
        counts = counts.rename(columns=renamed)
        result = result.merge(counts, on=join_key, how="left")

    return result


def _insert_column_after(df: pd.DataFrame, new_column: str, after_column: str) -> pd.DataFrame:
    columns = df.columns.tolist()
    if new_column not in columns or after_column not in columns:
        return df
    columns.remove(new_column)
    insert_index = columns.index(after_column) + 1
    columns.insert(insert_index, new_column)
    return df[columns]


def _add_last_30_day_ratio_columns(df: pd.DataFrame, join_key: str, time_column: str, prefix: str) -> pd.DataFrame:
    working = df.copy()
    working = working.loc[working[join_key].notna()].copy()
    if working.empty or time_column not in working.columns:
        return pd.DataFrame(columns=[join_key])

    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.loc[working[time_column].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key])

    excluded = {join_key, time_column, "status", "well"}
    numeric_candidates: list[str] = []
    for column in working.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        if numeric.notna().sum() > 0:
            working[column] = numeric
            numeric_candidates.append(column)

    if not numeric_candidates:
        return pd.DataFrame(columns=[join_key])

    result = pd.DataFrame({join_key: working[join_key].drop_duplicates().tolist()})
    for column in numeric_candidates:
        full_mean = (
            working.groupby(join_key, dropna=False)[column]
            .mean()
            .rename(f"{prefix}_{column}_mean")
            .reset_index()
        )
        max_dates = working.groupby(join_key, dropna=False)[time_column].max().rename("_max_date").reset_index()
        last30_frame = working.merge(max_dates, on=join_key, how="left")
        last30_frame = last30_frame.loc[
            last30_frame[time_column] >= (last30_frame["_max_date"] - pd.Timedelta(days=30))
        ].copy()
        last30_mean = (
            last30_frame.groupby(join_key, dropna=False)[column]
            .mean()
            .rename(f"{prefix}_{column}_last30d_mean")
            .reset_index()
        )
        merged = full_mean.merge(last30_mean, on=join_key, how="left")
        ratio_column = f"{prefix}_{column}_last30d_to_mean_ratio"
        denominator = pd.to_numeric(merged[f"{prefix}_{column}_mean"], errors="coerce")
        numerator = pd.to_numeric(merged[f"{prefix}_{column}_last30d_mean"], errors="coerce")
        merged[ratio_column] = np.where(
            denominator.notna() & numerator.notna() & (denominator.abs() > 1e-12),
            numerator / denominator,
            np.nan,
        )
        result = result.merge(merged[[join_key, ratio_column]], on=join_key, how="left")
        result = _insert_column_after(result, ratio_column, join_key)
    return result


def _prepare_gas_limits_frame(df: pd.DataFrame | None, notes: list[str]) -> pd.DataFrame:
    defaults = build_default_gas_limits_frame()
    if df is None:
        notes.append("Gas limits sheet not found. Using built-in default gas limits.")
        return defaults

    prepared = df.copy()
    if "gas_handling_type" in prepared.columns:
        prepared["gas_handling_type"] = prepared["gas_handling_type"].map(normalize_gas_handling_type)
    if "free_gas_limit_fraction" in prepared.columns:
        prepared["free_gas_limit_fraction"] = pd.to_numeric(prepared["free_gas_limit_fraction"], errors="coerce")
    merged = defaults.merge(prepared, on="gas_handling_type", how="left", suffixes=("_default", ""))
    merged["free_gas_limit_fraction"] = merged["free_gas_limit_fraction"].combine_first(merged["free_gas_limit_fraction_default"])
    merged["description"] = merged["description"].combine_first(merged["description_default"])
    merged = merged[["gas_handling_type", "free_gas_limit_fraction", "description"]]

    extras = prepared.loc[~prepared["gas_handling_type"].isin(merged["gas_handling_type"])].copy()
    if not extras.empty:
        merged = pd.concat([merged, extras[["gas_handling_type", "free_gas_limit_fraction", "description"]]], ignore_index=True)
    return merged.drop_duplicates(subset=["gas_handling_type"], keep="last").reset_index(drop=True)


def prepare_presentation_dataset(
    workbook: dict[str, pd.DataFrame],
    infant_threshold_days: float = 30.0,
    include_regime_last_30d_ratios: bool = False,
) -> PresentationPreparationResult:
    notes: list[str] = []
    detections = detect_sheet_roles(workbook)
    reports = inspect_presentation_workbook(workbook)
    report_by_role = {report.role: report for report in reports}

    if "runs" not in detections:
        raise ValueError("A runs sheet could not be detected. Rename the main run table to 'runs' or include core run columns.")

    runs_sheet = detections["runs"][0]
    runs_df, runs_mapping = standardize_role_dataframe(workbook[runs_sheet], "runs")
    report_by_role["runs"].mapped_columns = runs_mapping
    runs_df = _prepare_runs_frame(runs_df, infant_threshold_days=infant_threshold_days, notes=notes)

    design_df = None
    regime_df = None
    telemetry_df = None
    gas_limits_df = None

    if "design" in detections:
        design_df, design_mapping = standardize_role_dataframe(workbook[detections["design"][0]], "design")
        report_by_role["design"].mapped_columns = design_mapping
    if "daily_or_monthly_regime" in detections:
        regime_df, regime_mapping = standardize_role_dataframe(workbook[detections["daily_or_monthly_regime"][0]], "daily_or_monthly_regime")
        report_by_role["daily_or_monthly_regime"].mapped_columns = regime_mapping
    if "telemetry" in detections:
        telemetry_df, telemetry_mapping = standardize_role_dataframe(workbook[detections["telemetry"][0]], "telemetry")
        report_by_role["telemetry"].mapped_columns = telemetry_mapping
    if "gas_limits" in detections:
        gas_limits_df, gas_mapping = standardize_role_dataframe(workbook[detections["gas_limits"][0]], "gas_limits")
        report_by_role["gas_limits"].mapped_columns = gas_mapping

    merged_df = runs_df.copy()

    if design_df is not None:
        join_key = _choose_join_key(runs_df, design_df)
        report_by_role["design"].join_key = join_key
        if join_key is None:
            report_by_role["design"].notes.append("No shared join key with runs. Design sheet was not merged.")
        else:
            deduped_design = _deduplicate_design_frame(design_df, join_key)
            design_payload = deduped_design[[column for column in deduped_design.columns if column == join_key or column != "well"]]
            merged_df = merged_df.merge(design_payload, on=join_key, how="left")
            notes.append(f"Merged design sheet using `{join_key}`.")

    if regime_df is not None:
        join_key = _choose_join_key(runs_df, regime_df)
        report_by_role["daily_or_monthly_regime"].join_key = join_key
        if join_key is None:
            report_by_role["daily_or_monthly_regime"].notes.append("No shared join key with runs. Regime sheet was not merged.")
        else:
            filtered_regime_df = _restrict_to_target_entities(regime_df, runs_df, join_key)
            aggregated_regime = _aggregate_timeseries(filtered_regime_df, join_key, "date", "regime")
            merged_df = merged_df.merge(aggregated_regime, on=join_key, how="left")
            if include_regime_last_30d_ratios:
                ratio_regime = _add_last_30_day_ratio_columns(filtered_regime_df, join_key, "date", "regime")
                merged_df = merged_df.merge(ratio_regime, on=join_key, how="left")
                for column in ratio_regime.columns:
                    if not column.endswith("_last30d_to_mean_ratio"):
                        continue
                    mean_column = column.replace("_last30d_to_mean_ratio", "_mean")
                    if mean_column in merged_df.columns:
                        merged_df = _insert_column_after(merged_df, column, mean_column)
                notes.append("Merged regime last-30-day average to full-run average ratio columns.")
            notes.append(f"Merged aggregated regime sheet using `{join_key}` for target runs/wells only.")

    if telemetry_df is not None:
        join_key = _choose_join_key(runs_df, telemetry_df)
        report_by_role["telemetry"].join_key = join_key
        if join_key is None:
            report_by_role["telemetry"].notes.append("No shared join key with runs. Telemetry sheet was not merged.")
        else:
            filtered_telemetry_df = _restrict_to_target_entities(telemetry_df, runs_df, join_key)
            aggregated_telemetry = _aggregate_timeseries(filtered_telemetry_df, join_key, "timestamp", "telemetry")
            merged_df = merged_df.merge(aggregated_telemetry, on=join_key, how="left")
            notes.append(f"Merged aggregated telemetry sheet using `{join_key}` for target runs/wells only.")

    gas_limits_df = _prepare_gas_limits_frame(gas_limits_df, notes)
    if "gas_handling_type" in merged_df.columns:
        merged_df = merged_df.merge(gas_limits_df[["gas_handling_type", "free_gas_limit_fraction"]], on="gas_handling_type", how="left")

    suggested_group_columns = [column for column in ["field", "contractor"] if column in merged_df.columns]

    return PresentationPreparationResult(
        available_sheets=list(workbook.keys()),
        sheet_reports=[report_by_role[role] for role in SHEET_SPECS],
        merged_df=merged_df,
        runs_df=runs_df,
        design_df=design_df,
        regime_df=regime_df,
        telemetry_df=telemetry_df,
        gas_limits_df=gas_limits_df,
        notes=notes,
        suggested_event_column="event" if "event" in merged_df.columns else "",
        suggested_duration_column="TTF_days" if "TTF_days" in merged_df.columns else "",
        suggested_group_columns=suggested_group_columns,
    )
