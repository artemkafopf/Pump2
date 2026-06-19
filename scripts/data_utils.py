from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.sqlite_paths import resolve_lab_db_path, resolve_techregime_db_path, resolve_telemetry_db_path


TELEMETRY_DB_PATH = resolve_telemetry_db_path()
TECHREGIME_DB_PATH = resolve_techregime_db_path()
LAB_DB_PATH = resolve_lab_db_path()

CANONICAL_COLUMNS = [
    "freq",
    "load",
    "rpl",
    "rpump_intake",
    "rzab",
    "qliq",
    "watercut",
    "gas_factor",
    "qgas",
    "kprod",
]

LAB_CHEMISTRY_COLUMNS = [
    "chloride_mg_l",
    "sulfate_mg_l",
    "calcium_mg_l",
    "bicarbonate_mg_l",
    "magnesium_mg_l",
    "sodium_potassium_mg_l",
    "total_mineralization_g_l",
    "ph",
]

TECHREGIME_QUERY_COLUMNS = {
    "freq": "col_0047",
    "load": "col_0048",
    "rpl": "col_0061",
    "rpump_intake": "col_0062",
    "rzab": "col_0063",
    "qliq": "col_0064",
    "watercut": "col_0065",
    "gas_factor": "col_0067",
    "qgas": "col_0068",
    "kprod": "col_0069",
}

TELEMETRY_DAILY_COLUMN_CANDIDATES = {
    "freq": ("frequency_hz",),
    "load": ("motor_load_percent",),
    "rpl": (),
    "rpump_intake": ("P_intake_atm",),
    "rzab": ("P_bhp_atm",),
    "qliq": ("Qliq_m3d",),
    "watercut": ("watercut_percent",),
    "gas_factor": ("GLF_m3m3",),
    "qgas": ("Qgas_m3d",),
    "kprod": (),
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_well_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["well_id", "dt", *CANONICAL_COLUMNS, "source"])


def _build_date_clause(date_from: pd.Timestamp | None, date_to: pd.Timestamp | None, date_column: str) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if date_from is not None:
        clauses.append(f"{date_column} >= ?")
        params.append(pd.Timestamp(date_from).strftime("%Y-%m-%d"))
    if date_to is not None:
        clauses.append(f"{date_column} <= ?")
        params.append(pd.Timestamp(date_to).strftime("%Y-%m-%d"))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _load_telemetry_daily(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    well_map = {normalize_well_key(well): str(well).strip() for well in wells if normalize_well_key(well)}
    normalized_wells = sorted(key for key in well_map if key)
    if not normalized_wells:
        return _empty_daily_frame()

    with sqlite3.connect(TELEMETRY_DB_PATH) as connection:
        schema = pd.read_sql_query("PRAGMA table_info(telemetry_daily)", connection)
        available = set(schema["name"].astype(str))
        select_parts = ["_meta_normalized_well as well_key", "_meta_record_date as dt"]
        for alias in CANONICAL_COLUMNS:
            source_column = next((candidate for candidate in TELEMETRY_DAILY_COLUMN_CANDIDATES[alias] if candidate in available), None)
            if source_column is None:
                select_parts.append(f"NULL as {alias}")
            else:
                select_parts.append(f"{source_column} as {alias}")
        query_select = ", ".join(select_parts)

        frames: list[pd.DataFrame] = []
        date_clause, date_params = _build_date_clause(date_from, date_to, "_meta_record_date")
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT {query_select}
                FROM telemetry_daily
                WHERE _meta_normalized_well IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))

    df = pd.concat(frames, ignore_index=True) if frames else _empty_daily_frame()
    if df.empty:
        return _empty_daily_frame()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for column in CANONICAL_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_key"].notna() & df["dt"].notna()].copy()
    df["well_id"] = df["well_key"].map(well_map).fillna(df["well_key"])
    df["source"] = "telemetry"
    result = (
        df.groupby(["well_id", "dt", "source"], as_index=False)[CANONICAL_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return result


def _load_techregime_daily(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    normalized_wells = [str(well).strip() for well in wells if str(well).strip()]
    if not normalized_wells:
        return _empty_daily_frame()

    query_select = ", ".join([f"{storage} as {alias}" for alias, storage in TECHREGIME_QUERY_COLUMNS.items()])
    date_clause, date_params = _build_date_clause(date_from, date_to, "col_0010")
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TECHREGIME_DB_PATH) as connection:
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT col_0003 as well_id, col_0010 as dt, {query_select}
                FROM techregime_records
                WHERE col_0003 IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))
    df = pd.concat(frames, ignore_index=True) if frames else _empty_daily_frame()
    if df.empty:
        return _empty_daily_frame()
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True, errors="coerce")
    df["well_id"] = df["well_id"].astype("string").str.strip()
    for column in CANONICAL_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_id"].notna() & df["dt"].notna()].copy()
    df["source"] = "techregime"
    result = (
        df.groupby(["well_id", "dt", "source"], as_index=False)[CANONICAL_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return result


def _empty_lab_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["well_id", "sample_date", *LAB_CHEMISTRY_COLUMNS, "source_file"])


def _load_lab_samples(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    if not LAB_DB_PATH.exists():
        return _empty_lab_frame()
    well_map = {normalize_well_key(well): str(well).strip() for well in wells if normalize_well_key(well)}
    normalized_wells = sorted(key for key in well_map if key)
    if not normalized_wells:
        return _empty_lab_frame()

    with sqlite3.connect(LAB_DB_PATH) as connection:
        schema = pd.read_sql_query("PRAGMA table_info(lab_samples)", connection)
        if schema.empty:
            return _empty_lab_frame()
        available = set(schema["name"].astype(str))
        select_parts = ["well_key", "sample_date", "source_file"]
        for column in LAB_CHEMISTRY_COLUMNS:
            if column in available:
                select_parts.append(column)
            else:
                select_parts.append(f"NULL as {column}")
        query_select = ", ".join(select_parts)
        date_clause, date_params = _build_date_clause(date_from, date_to, "sample_date")
        frames: list[pd.DataFrame] = []
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT {query_select}
                FROM lab_samples
                WHERE well_key IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))
    df = pd.concat(frames, ignore_index=True) if frames else _empty_lab_frame()
    if df.empty:
        return _empty_lab_frame()
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    for column in LAB_CHEMISTRY_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_key"].notna() & df["sample_date"].notna()].copy()
    df["well_id"] = df["well_key"].map(well_map).fillna(df["well_key"])
    return (
        df.groupby(["well_id", "sample_date", "source_file"], as_index=False)[LAB_CHEMISTRY_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "sample_date", "source_file"])
        .reset_index(drop=True)
    )


def attach_lab_chemistry(
    daily_df: pd.DataFrame,
    *,
    fill_mode: str = "step",
    extend_backward: bool = True,
) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df.copy()
    if fill_mode != "step":
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    wells = sorted(daily_df["well_id"].dropna().astype(str).unique().tolist())
    lab = _load_lab_samples(wells)
    result = daily_df.copy().sort_values(["well_id", "dt"]).reset_index(drop=True)

    if lab.empty:
        for column in LAB_CHEMISTRY_COLUMNS:
            result[column] = np.nan
        result["lab_sample_date"] = pd.NaT
        result["lab_source_file"] = pd.Series(pd.NA, index=result.index, dtype="string")
        return result

    frames: list[pd.DataFrame] = []
    for well_id, group in result.groupby("well_id", sort=False):
        group = group.sort_values("dt").reset_index(drop=True)
        chemistry = lab.loc[lab["well_id"].astype(str) == str(well_id)].copy()
        if chemistry.empty:
            frames.append(group)
            continue
        chemistry = chemistry.sort_values("sample_date").reset_index(drop=True)

        backward = pd.merge_asof(
            group,
            chemistry[["sample_date", "source_file", *LAB_CHEMISTRY_COLUMNS]],
            left_on="dt",
            right_on="sample_date",
            direction="backward",
        )
        if extend_backward:
            forward = pd.merge_asof(
                group,
                chemistry[["sample_date", "source_file", *LAB_CHEMISTRY_COLUMNS]],
                left_on="dt",
                right_on="sample_date",
                direction="forward",
            )
            for column in LAB_CHEMISTRY_COLUMNS:
                backward[column] = backward[column].combine_first(forward[column])
            backward["sample_date"] = backward["sample_date"].combine_first(forward["sample_date"])
            backward["source_file"] = backward["source_file"].combine_first(forward["source_file"])

        backward = backward.rename(columns={"sample_date": "lab_sample_date", "source_file": "lab_source_file"})
        frames.append(backward)

    filled = pd.concat(frames, ignore_index=True).sort_values(["well_id", "dt"]).reset_index(drop=True)
    if "lab_source_file" in filled.columns:
        filled["lab_source_file"] = filled["lab_source_file"].astype("string")
    return filled


def add_dynamic_salt_proxies(
    daily_df: pd.DataFrame,
    *,
    qliq_column: str = "qliq",
    fill_mode: str = "step",
    extend_backward: bool = True,
) -> pd.DataFrame:
    working = attach_lab_chemistry(daily_df, fill_mode=fill_mode, extend_backward=extend_backward)
    qliq = _numeric(working.get(qliq_column, pd.Series(np.nan, index=working.index, dtype=float))).clip(lower=0)
    calcium = _numeric(working.get("calcium_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))
    chloride = _numeric(working.get("chloride_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))
    sulfate = _numeric(working.get("sulfate_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))

    # mg/L * m3/day / 1000 -> kg/day
    working["daily_calcium_load_kg"] = (calcium * qliq) / 1000.0
    working["daily_chloride_load_kg"] = (chloride * qliq) / 1000.0
    working["daily_sulfate_load_kg"] = (sulfate * qliq) / 1000.0
    working["daily_salt_load_kg"] = ((calcium + chloride + sulfate) * qliq) / 1000.0
    # Exposure proxy, not direct precipitated mass.
    working["daily_gypsum_scale_proxy"] = (calcium * sulfate * qliq) / 1_000_000.0

    grouped = working.groupby("well_id", sort=False)
    for daily_col, cum_col in [
        ("daily_calcium_load_kg", "cum_calcium_load_kg_dynamic"),
        ("daily_chloride_load_kg", "cum_chloride_load_kg_dynamic"),
        ("daily_sulfate_load_kg", "cum_sulfate_load_kg_dynamic"),
        ("daily_salt_load_kg", "cum_salt_load_kg_dynamic"),
        ("daily_gypsum_scale_proxy", "cum_gypsum_scale_proxy_dynamic"),
    ]:
        working[cum_col] = grouped[daily_col].cumsum()
    return working


def load_daily_merged(
    wells: list[str],
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
) -> pd.DataFrame:
    telemetry = _load_telemetry_daily(wells, date_from=date_from, date_to=date_to)
    techregime = _load_techregime_daily(wells, date_from=date_from, date_to=date_to)

    telemetry = telemetry.copy()
    techregime = techregime.copy()
    telemetry["well_key"] = telemetry["well_id"].map(normalize_well_key)
    techregime["well_key"] = techregime["well_id"].map(normalize_well_key)

    merged = telemetry.merge(
        techregime,
        on=["well_key", "dt"],
        how="outer",
        suffixes=("_tel", "_tr"),
    )

    well_id_tel = merged.get("well_id_tel", pd.Series(index=merged.index, dtype="string")).astype("string")
    well_id_tr = merged.get("well_id_tr", pd.Series(index=merged.index, dtype="string")).astype("string")
    result = pd.DataFrame(
        {
            "well_id": well_id_tel.fillna(well_id_tr),
            "dt": merged["dt"],
        }
    )
    for column in CANONICAL_COLUMNS:
        tel_series = _numeric(merged.get(f"{column}_tel", pd.Series(np.nan, index=merged.index, dtype=float)))
        tr_series = _numeric(merged.get(f"{column}_tr", pd.Series(np.nan, index=merged.index, dtype=float)))
        result[column] = tel_series.combine_first(tr_series)
    source = pd.Series(pd.NA, index=merged.index, dtype="string")
    any_tel = pd.Series(False, index=merged.index)
    for column in CANONICAL_COLUMNS:
        any_tel = any_tel | _numeric(merged.get(f"{column}_tel", pd.Series(np.nan, index=merged.index, dtype=float))).notna()
    source.loc[any_tel] = "telemetry"
    source.loc[~any_tel] = "techregime"
    result["source"] = source
    result = result.loc[result["well_id"].notna() & result["dt"].notna()].copy()
    result = (
        result.groupby(["well_id", "dt", "source"], as_index=False)[CANONICAL_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return result


def source_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return pd.DataFrame(columns=["well_id", "telemetry_rows", "techregime_rows", "total_rows", "telemetry_fraction"])
    working = df.copy()
    rows: list[dict[str, object]] = []
    for well_id, frame in working.groupby("well_id", dropna=False):
        telemetry_rows = int(frame["source"].eq("telemetry").sum())
        techregime_rows = int(frame["source"].eq("techregime").sum())
        total_rows = int(len(frame))
        rows.append(
            {
                "well_id": str(well_id),
                "telemetry_rows": telemetry_rows,
                "techregime_rows": techregime_rows,
                "total_rows": total_rows,
                "telemetry_fraction": float(telemetry_rows / max(total_rows, 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["telemetry_fraction", "well_id"], ascending=[False, True]).reset_index(drop=True)


def split_by_run_id(
    df: pd.DataFrame,
    run_id_column: str = "row_id",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if run_id_column not in df.columns:
        raise ValueError(f"Run id column '{run_id_column}' not found.")
    from analysis.modeling_config import stable_hash_test_mask

    test_mask = stable_hash_test_mask(df[run_id_column], test_fraction=test_fraction)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()
    return train_df, test_df


__all__ = [
    "CANONICAL_COLUMNS",
    "LAB_CHEMISTRY_COLUMNS",
    "LAB_DB_PATH",
    "TECHREGIME_DB_PATH",
    "TELEMETRY_DB_PATH",
    "_numeric",
    "add_dynamic_salt_proxies",
    "attach_lab_chemistry",
    "load_daily_merged",
    "normalize_well_key",
    "source_coverage_report",
    "split_by_run_id",
]
