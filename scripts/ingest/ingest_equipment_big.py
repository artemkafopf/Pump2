"""Ingest WellsArtificialLiftBig.xlsx → raw__equipment_big + feat__run_equipment.

raw__equipment_big  — one row per спуск (8k+ rows, ESP and non-ESP), normalized
                      selected columns (see analysis.data.equipment_big).
feat__run_equipment — one row per raw__v03_runs row that matches on
                      (well_key, install_date): the equipment/corrosion profile
                      joined onto the survival run inventory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.equipment_big import load_equipment_big
from analysis.paths import resolve_equipment_big_path
from scripts.db import StepTimer, file_sha256, get_warehouse_conn, log_source_hash, upsert_df

# Columns carried into the per-run feature table (all already normalized).
_RUN_FEATURE_COLS = [
    "gno_type", "pump_gabarit", "pump_od_mm", "pump_length_m", "curvature",
    "q_nom_m3d", "head_nom_m", "freq_nom_hz", "stages",
    "pump_exec_group", "pump_exec_lch", "pump_napornost",
    "type_wear_resistant", "type_corr_resistant",
    "pump_corr_class", "gassep_corr_class", "protector_corr_class",
    "ped_corr_class", "tms_corr_class", "nkt_corr_class", "cable_corr_class",
    "pump_coating", "gassep_coating", "protector_coating", "ped_coating",
    "n_components_monel", "max_corr_class", "any_corr_protection",
    "gassep_type", "intake_module", "multiphase_type", "protector_type",
    "ped_type", "ped_oil", "ped_max_temp_c", "ped_power_kw", "tms_type",
    "pump_depth_m", "vg_m", "nkt_diam_mm", "nkt_marka",
    "cable_model", "cable_section_mm2", "cable_max_temp_c", "cable_length_m",
    "run_no", "nno_days", "pull_reason",
    "launch_date", "fail_date", "pull_date", "pull_fail_gap_d", "launch_delay_d",
]


def _build_run_join(eq: pd.DataFrame, conn) -> pd.DataFrame:
    runs = pd.read_sql("SELECT row_id, well_key, install_date FROM raw__v03_runs", conn)
    runs["install_date"] = pd.to_datetime(runs["install_date"])

    cand = eq[eq["well_key"].notna() & eq["install_date"].notna()].copy()
    # Prefer ESP rows when a (well, date) has several записи; deterministic order.
    cand = cand.sort_values(["is_esp", "row_id"], ascending=[False, True])
    cand = cand.drop_duplicates(subset=["well_key", "install_date"], keep="first")

    joined = runs.merge(
        cand[["well_key", "install_date", "is_esp"] + _RUN_FEATURE_COLS],
        on=["well_key", "install_date"],
        how="inner",
    )
    return joined


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        src = resolve_equipment_big_path()
        if not src.exists():
            print(f"[ingest_equipment_big] source not found at {src}, skipping.")
            return

        with StepTimer("raw__equipment_big", conn) as timer:
            eq = load_equipment_big(src)
            timer.row_count = upsert_df(eq, "raw__equipment_big", conn, pk_cols=["row_id"])
            log_source_hash("equipment_big", file_sha256(src), conn=conn)

        with StepTimer("feat__run_equipment", conn) as timer:
            joined = _build_run_join(eq, conn)
            timer.row_count = upsert_df(joined, "feat__run_equipment", conn, pk_cols=["row_id"])
            print(f"[ingest_equipment_big] matched {len(joined)} of "
                  f"{conn.execute('SELECT COUNT(*) FROM raw__v03_runs').fetchone()[0]} runs")
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
