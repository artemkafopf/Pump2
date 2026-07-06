"""Ingest V03_all and V03_failures workbooks into raw__v03_runs and raw__v03_failures."""
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

from analysis.data.label_hygiene import FIELD_ALIAS_MAP, trim_failed_node
from analysis.input_paths import resolve_v03_all_path, resolve_v03_failures_path
from scripts.db import StepTimer, file_sha256, get_warehouse_conn, log_source_hash, upsert_df
from scripts.data_utils import _numeric, normalize_well_key


# Columns to extract from the "Свод" sheet.
# Russian originals are mapped to clean English names for the warehouse.
_COLUMN_MAP = {
    "Скв.": "well",
    "Месторождение": "field",
    "Куст": "pad",
    "Принадлежность": "contractor",
    "Дата монтажа": "install_date",
    "Дата остановки": "stop_date",
    "Тип УЭЦН": "pump_type",
    "Ном. Произв. м₃/сут": "nominal_flow_m3d",
    "Ном.напор (50Гц)": "nominal_head_50hz_m",
    "Номинальная частота, Гц": "nominal_freq_hz",
    "Мощность, кВт": "motor_power_kw",
    "Кол.ступеней": "stages",
    "Глубина спуска УЭЦН, по НКТ": "submergence_depth_m",
    "Дав. Нас": "pbubble_atm",
    "Работа в кривизне": "curvature_flag",
    "ВГ, м": "vg_m",
    "Ном. ток/ A": "nominal_current_a",
    "Массовая доля сероводорода, мг/дм³": "h2s_mg_l",
    "Отказавший узел": "failed_node",
    "Failure Flag": "failure_flag",
}


def _load_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Свод").reset_index(names="_row_idx")
    df["event"] = _numeric(df["Failure Flag"])
    df["Дата монтажа"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    df["Дата остановки"] = pd.to_datetime(df["Дата остановки"], errors="coerce")
    df["Скв."] = df["Скв."].astype("string").str.strip()
    df = df.loc[df["event"].isin([0, 1])].copy()
    df = df.loc[df["Скв."].notna() & df["Дата монтажа"].notna() & df["Дата остановки"].notna()].copy()
    df = df.loc[df["Дата остановки"] > df["Дата монтажа"]].copy()
    df = df.drop_duplicates(subset=["Скв.", "Дата монтажа", "Дата остановки", "event"], keep="first")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # stable 0-based integer key, matches load_runs() output
    return df


def _build_warehouse_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["row_id"] = df["row_id"].astype(int)
    out["well"] = df["Скв."].astype("string").str.strip()
    out["well_key"] = out["well"].map(normalize_well_key)
    out["event"] = df["event"].astype(int)
    out["run_days"] = (df["Дата остановки"] - df["Дата монтажа"]).dt.days.astype(float)
    out["install_date"] = df["Дата монтажа"]
    out["stop_date"] = df["Дата остановки"]

    for src_col, dst_col in _COLUMN_MAP.items():
        if dst_col in ("well", "install_date", "stop_date", "failure_flag"):
            continue  # already handled
        if src_col in df.columns:
            out[dst_col] = df[src_col]
        else:
            out[dst_col] = np.nan

    # Coerce known numeric columns
    for col in ("nominal_flow_m3d", "nominal_head_50hz_m", "nominal_freq_hz", "motor_power_kw",
                "stages", "submergence_depth_m", "pbubble_atm", "vg_m", "nominal_current_a", "h2s_mg_l"):
        if col in out.columns:
            out[col] = _numeric(out[col])

    # Label hygiene (Phase A T5 wiring): trim failed_node so 'НКТ ' merges with 'НКТ',
    # and collapse confirmed field aliases onto their codes. Unmapped field names are
    # kept as-is (needs_review flow lives in scripts/run/phase_a_label_hygiene.py).
    out["failed_node"] = out["failed_node"].map(trim_failed_node)
    out["field"] = out["field"].astype("string").str.strip()
    out["field"] = out["field"].map(lambda v: FIELD_ALIAS_MAP.get(v, v) if pd.notna(v) else v)

    return out


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        v03_all_path = resolve_v03_all_path()
        v03_failures_path = resolve_v03_failures_path()

        # --- raw__v03_runs ---
        with StepTimer("raw__v03_runs", conn) as timer:
            df = _load_sheet(v03_all_path)
            out = _build_warehouse_frame(df)
            timer.row_count = upsert_df(out, "raw__v03_runs", conn, pk_cols=["row_id"])
            log_source_hash("v03_all", file_sha256(v03_all_path), conn=conn)

        # --- raw__v03_failures ---
        if v03_failures_path.exists():
            with StepTimer("raw__v03_failures", conn) as timer:
                df_f = _load_sheet(v03_failures_path)
                out_f = _build_warehouse_frame(df_f)
                timer.row_count = upsert_df(out_f, "raw__v03_failures", conn, pk_cols=["row_id"])
                log_source_hash("v03_failures", file_sha256(v03_failures_path), conn=conn)
        else:
            print(f"[ingest_v03] V03_failures not found at {v03_failures_path}, skipping.")
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
