"""Build proc__h2s_proxy: H2S proxy per run with well → pad fallback.

Ports _h2s_proxy_frame() from scripts/build_run_features.py.
Reads raw__v03_runs from the warehouse.
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

from scripts.build_run_features import _h2s_proxy_frame
from scripts.db import StepTimer, get_warehouse_conn, upsert_df


def _adapt_for_h2s_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Map warehouse column names back to the Russian originals expected by _h2s_proxy_frame."""
    adapted = runs.copy()
    adapted["Скв."] = adapted["well"]
    adapted["Месторождение"] = adapted.get("field", pd.Series(index=adapted.index))
    adapted["Куст"] = adapted.get("pad", pd.Series(index=adapted.index))
    adapted["Массовая доля сероводорода, мг/дм³"] = adapted.get("h2s_mg_l", pd.Series(index=adapted.index))
    return adapted


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        runs = pd.read_sql(
            "SELECT row_id, well, field, pad, h2s_mg_l FROM raw__v03_runs WHERE well IS NOT NULL",
            conn,
        )
        if runs.empty:
            print("[build_h2s_proxy] raw__v03_runs is empty — run ingest_v03 first.")
            return

        with StepTimer("proc__h2s_proxy", conn) as timer:
            adapted = _adapt_for_h2s_frame(runs)
            proxy = _h2s_proxy_frame(adapted)
            # _h2s_proxy_frame returns: row_id, pad, h2s_well_median, h2s_pad_median, h2s_proxy_mg_l, h2s_proxy_source
            proxy = proxy.rename(columns={
                "h2s_well_median": "h2s_well_median_mg_l",
                "h2s_pad_median": "h2s_pad_median_mg_l",
            })
            keep = ["row_id", "h2s_proxy_mg_l", "h2s_proxy_source", "h2s_well_median_mg_l", "h2s_pad_median_mg_l"]
            result = proxy[[c for c in keep if c in proxy.columns]]
            timer.row_count = upsert_df(result, "proc__h2s_proxy", conn, pk_cols=["row_id"])
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
