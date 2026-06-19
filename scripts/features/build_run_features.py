"""Build feat__run_mature and feat__run_whole_life from the processed warehouse tables.

Ports build_run_features() from scripts/build_run_features.py but reads from the warehouse
instead of raw SQLite sources.

Grain: 1 row per run in raw__v03_runs that has enough data in the mature window (≥14 obs days).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.modeling_config import (
    DEFAULT_GLF_THRESHOLD,
    DEFAULT_INFANT_MORTALITY_DAYS,
    DEFAULT_MATURE_WINDOW_DAYS,
    DEFAULT_MIN_MATURE_OBS,
)
from scripts.build_run_features import (
    _aggregate_window_features,
    _daily_run_frame,
    _static_feature_frame,
    split_infant_vs_mature_runs,
)
from scripts.data_utils import _numeric, normalize_well_key
from scripts.db import StepTimer, get_warehouse_conn, upsert_df


def _load_daily_from_warehouse(well_keys: list[str], conn) -> pd.DataFrame:
    """Load proc__daily_merged + proc__daily_precipitate joined for a set of well_keys."""
    placeholders = ",".join([f"'{wk}'" for wk in well_keys])
    merged = pd.read_sql(
        f"""
        SELECT well_key, dt, freq, load, qliq, watercut, gas_factor, qgas, kprod, rzab, source
        FROM proc__daily_merged
        WHERE well_key IN ({placeholders})
        """,
        conn,
        parse_dates=["dt"],
    )
    ppt = pd.read_sql(
        f"""
        SELECT well_key, dt, daily_salt_load_kg, daily_gypsum_scale_proxy
        FROM proc__daily_precipitate
        WHERE well_key IN ({placeholders})
        """,
        conn,
        parse_dates=["dt"],
    )
    if ppt.empty:
        daily = merged.copy()
    else:
        daily = merged.merge(ppt, on=["well_key", "dt"], how="left")
    daily["well_id"] = daily["well_key"]  # _daily_run_frame expects "well_id"
    return daily


def run(
    conn=None,
    glf_threshold: float = DEFAULT_GLF_THRESHOLD,
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS,
    mature_window_days: int = DEFAULT_MATURE_WINDOW_DAYS,
    min_mature_obs: int = DEFAULT_MIN_MATURE_OBS,
) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        runs_df = pd.read_sql(
            """
            SELECT row_id, well, well_key, field, pad, contractor, event, run_days,
                   install_date, stop_date, pump_type, nominal_flow_m3d, nominal_head_50hz_m,
                   nominal_freq_hz, motor_power_kw, stages, submergence_depth_m, pbubble_atm,
                   curvature_flag, h2s_mg_l
            FROM raw__v03_runs WHERE well IS NOT NULL
            """,
            conn,
            parse_dates=["install_date", "stop_date"],
        )
        if runs_df.empty:
            print("[build_run_features] raw__v03_runs is empty — run ingest_v03 first.")
            return

        # Adapt to Russian column names for split_infant_vs_mature_runs.
        adapted = pd.DataFrame({
            "row_id": runs_df["row_id"],
            "Скв.": runs_df["well"],
            "Дата монтажа": runs_df["install_date"],
            "Дата остановки": runs_df["stop_date"],
            "event": runs_df["event"],
            "run_days": runs_df["run_days"],
            "Месторождение": runs_df["field"],
            "Принадлежность": runs_df["contractor"],
            "Куст": runs_df["pad"],
            "Тип УЭЦН": runs_df["pump_type"],
            "Ном. Произв. м₃/сут": runs_df["nominal_flow_m3d"],
            "Ном.напор (50Гц)": runs_df["nominal_head_50hz_m"],
            "Номинальная частота, Гц": runs_df["nominal_freq_hz"],
            "Мощность, кВт": runs_df["motor_power_kw"],
            "Кол.ступеней": runs_df["stages"],
            "Глубина спуска УЭЦН, по НКТ": runs_df["submergence_depth_m"],
            "Дав. Нас": runs_df["pbubble_atm"],
            "Работа в кривизне": runs_df["curvature_flag"],
            "Массовая доля сероводорода, мг/дм³": runs_df["h2s_mg_l"],
            "Failure Flag": runs_df["event"],
        })

        # Normalize Скв. to well_key format (casefolded) so it matches proc__daily_merged.well_id.
        # _daily_run_frame compares daily_df["well_id"] == str(run_row["Скв."]).strip(),
        # and proc__daily_merged stores well_key (casefolded) as well_id.
        adapted["Скв."] = adapted["Скв."].map(normalize_well_key)

        mature_runs, _, _, _ = split_infant_vs_mature_runs(
            adapted, infant_mortality_days=infant_mortality_days, mature_window_days=mature_window_days
        )
        if mature_runs.empty:
            print("[build_run_features] No mature runs found.")
            return

        print(f"[build_run_features] {len(mature_runs)} mature runs to process.")
        all_wkeys = sorted(mature_runs["Скв."].dropna().astype(str).unique().tolist())
        daily_merged = _load_daily_from_warehouse(all_wkeys, conn)

        mature_rows: list[dict] = []
        whole_rows: list[dict] = []
        for run_dict_raw in mature_runs.to_dict(orient="records"):
            # run_dict_raw already has Russian column names from adapted; pass directly.
            interval = _daily_run_frame(daily_merged, run_dict_raw)
            if interval.empty:
                continue

            post_infant_start = run_dict_raw["Дата монтажа"] + pd.Timedelta(days=infant_mortality_days)
            mature_end = post_infant_start + pd.Timedelta(days=mature_window_days)
            mature_window = interval.loc[(interval["dt"] >= post_infant_start) & (interval["dt"] <= mature_end)]
            whole_life = interval.loc[interval["dt"] >= post_infant_start]

            row_id = int(run_dict_raw["row_id"])
            if int(mature_window["dt"].nunique()) >= min_mature_obs:
                row_m = {"row_id": row_id}
                row_m.update(_aggregate_window_features(mature_window, "m", glf_threshold))
                row_m["mature_obs_days"] = int(mature_window["dt"].nunique())
                mature_rows.append(row_m)

            if int(whole_life["dt"].nunique()) >= min_mature_obs:
                row_w = {"row_id": row_id}
                row_w.update(_aggregate_window_features(whole_life, "w", glf_threshold))
                row_w["whole_life_obs_days"] = int(whole_life["dt"].nunique())
                whole_rows.append(row_w)

        with StepTimer("feat__run_mature", conn) as timer:
            mature_df = pd.DataFrame(mature_rows) if mature_rows else pd.DataFrame(columns=["row_id"])
            timer.row_count = upsert_df(mature_df, "feat__run_mature", conn, pk_cols=["row_id"])

        with StepTimer("feat__run_whole_life", conn) as timer:
            whole_df = pd.DataFrame(whole_rows) if whole_rows else pd.DataFrame(columns=["row_id"])
            timer.row_count = upsert_df(whole_df, "feat__run_whole_life", conn, pk_cols=["row_id"])

    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
