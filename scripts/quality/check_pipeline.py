"""Pipeline quality checks: row counts, null rates, date coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.db import get_warehouse_conn


_CHECKS = {
    "raw__v03_runs": {
        "min_rows": 10,
        "required_not_null": ["row_id", "well", "well_key", "install_date", "stop_date", "event"],
    },
    "proc__ttf_true": {
        "min_rows": 10,
        "required_not_null": ["row_id", "ttf_true_source"],
        "coverage_column": "ttf_true_best_days",
    },
    "proc__daily_merged": {
        "min_rows": 1000,
        "required_not_null": ["well_key", "dt"],
    },
    "proc__daily_lab": {
        "min_rows": 100,
        "required_not_null": ["well_key", "dt"],
    },
    "proc__daily_operating": {
        "min_rows": 1000,
        "required_not_null": ["well_key", "dt", "in_operation"],
    },
    "proc__daily_precipitate": {
        "min_rows": 100,
        "required_not_null": ["well_key", "dt"],
    },
    "proc__h2s_proxy": {
        "min_rows": 10,
        "required_not_null": ["row_id", "h2s_proxy_source"],
        "coverage_column": "h2s_proxy_mg_l",
    },
    "feat__run_mature": {
        "min_rows": 1,
        "required_not_null": ["row_id"],
        "coverage_column": "kpod_m_mean",
    },
    "feat__run_freq_exposure": {
        "min_rows": 10,
        "required_not_null": ["row_id"],
    },
    "mart__vt_freq55": {
        "min_rows": 10,
        "required_not_null": ["row_id", "well", "event"],
    },
}


def check_table(table: str, spec: dict, conn) -> list[str]:
    issues: list[str] = []
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception as e:
        return [f"{table}: table missing or unreadable — {e}"]

    row_count = len(df)
    min_rows = spec.get("min_rows", 0)
    if row_count < min_rows:
        issues.append(f"{table}: only {row_count} rows (expected ≥{min_rows})")

    for col in spec.get("required_not_null", []):
        if col not in df.columns:
            issues.append(f"{table}.{col}: column missing entirely")
        elif df[col].isna().all():
            issues.append(f"{table}.{col}: all values are null")

    cov_col = spec.get("coverage_column")
    if cov_col and cov_col in df.columns:
        coverage = float(df[cov_col].notna().mean())
        if coverage < 0.05:
            issues.append(f"{table}.{cov_col}: very low coverage ({coverage:.1%})")

    return issues


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        all_issues: list[str] = []
        for table, spec in _CHECKS.items():
            issues = check_table(table, spec, conn)
            all_issues.extend(issues)

        # Also check pipeline run log.
        runs_log = pd.read_sql(
            "SELECT table_name, status, run_at, row_count FROM meta__pipeline_runs ORDER BY id DESC",
            conn,
        )
        if not runs_log.empty:
            failed = runs_log.loc[runs_log["status"] != "ok"]
            for _, row in failed.iterrows():
                all_issues.append(f"Pipeline: {row['table_name']} last run FAILED at {row['run_at']}")

        if all_issues:
            print(f"\n[check_pipeline] {len(all_issues)} issue(s) found:")
            for issue in all_issues:
                print(f"  ⚠ {issue}")
        else:
            print("[check_pipeline] All checks passed.")

        # Print a summary of table row counts.
        print("\n[check_pipeline] Table inventory:")
        for table in _CHECKS:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count:,} rows")
            except Exception:
                print(f"  {table}: not found")

        # Print last pipeline run per table.
        if not runs_log.empty:
            print("\n[check_pipeline] Last pipeline run per table:")
            latest = runs_log.drop_duplicates(subset=["table_name"], keep="first")
            for _, row in latest.iterrows():
                print(f"  {row['table_name']}: {row['status']} at {row['run_at']} ({row['row_count']:,} rows)")
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
