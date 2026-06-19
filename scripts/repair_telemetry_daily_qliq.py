from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.sqlite_paths import resolve_telemetry_db_path  # noqa: E402

DEFAULT_DB_PATH = resolve_telemetry_db_path()


def count_non_null(connection: sqlite3.Connection, column: str) -> int:
    query = f"SELECT SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) FROM telemetry_daily"
    return int(connection.execute(query).fetchone()[0] or 0)


def repair_qliq(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        before_qliq = count_non_null(connection, "Qliq_m3d")
        before_qgas = count_non_null(connection, "Qgas_m3d")
        total_rows = int(connection.execute("SELECT COUNT(*) FROM telemetry_daily").fetchone()[0] or 0)

        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("DROP TABLE IF EXISTS temp.telemetry_raw_daily_agg")
        connection.execute(
            """
            CREATE TEMP TABLE telemetry_raw_daily_agg AS
            SELECT
                _meta_normalized_well AS well_key,
                _meta_record_date AS record_date,
                AVG(col_0007) AS qliq_m3d,
                AVG(CASE WHEN col_0007 IS NOT NULL AND col_0013 IS NOT NULL THEN col_0007 * col_0013 END) AS qgas_m3d
            FROM telemetry_raw
            GROUP BY _meta_normalized_well, _meta_record_date
            """
        )
        connection.execute("CREATE INDEX temp.idx_telemetry_raw_daily_agg ON telemetry_raw_daily_agg(well_key, record_date)")
        connection.execute(
            """
            UPDATE telemetry_daily
            SET
                Qliq_m3d = (
                    SELECT CAST(a.qliq_m3d AS TEXT)
                    FROM telemetry_raw_daily_agg a
                    WHERE a.well_key = telemetry_daily._meta_normalized_well
                      AND a.record_date = telemetry_daily._meta_record_date
                ),
                Qgas_m3d = COALESCE(
                    Qgas_m3d,
                    (
                        SELECT CAST(a.qgas_m3d AS TEXT)
                        FROM telemetry_raw_daily_agg a
                        WHERE a.well_key = telemetry_daily._meta_normalized_well
                          AND a.record_date = telemetry_daily._meta_record_date
                    )
                )
            WHERE EXISTS (
                SELECT 1
                FROM telemetry_raw_daily_agg a
                WHERE a.well_key = telemetry_daily._meta_normalized_well
                  AND a.record_date = telemetry_daily._meta_record_date
            )
            """
        )
        connection.commit()

        after_qliq = count_non_null(connection, "Qliq_m3d")
        after_qgas = count_non_null(connection, "Qgas_m3d")
        return {
            "total_rows": total_rows,
            "before_qliq_non_null": before_qliq,
            "after_qliq_non_null": after_qliq,
            "before_qgas_non_null": before_qgas,
            "after_qgas_non_null": after_qgas,
        }
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair telemetry_daily liquid-rate normalization from telemetry_raw.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Path to telemetry.sqlite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = repair_qliq(Path(args.db_path))
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
