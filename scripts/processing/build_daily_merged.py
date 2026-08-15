"""Build proc__daily_merged: telemetry-first daily operational data with techregime fallback.

Ports load_daily_merged() from scripts/data_utils.py into the warehouse.
Processes wells in chunks to avoid loading all data into memory at once.
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

from scripts.data_utils import (
    CANONICAL_COLUMNS,
    GAS_FACTOR_COLUMN,
    GAS_LIQUID_RATIO_COLUMN,
    OIL_DENSITY_COLUMN,
    SOURCE_SUFFIX,
    column_provenance_report,
    load_daily_merged,
    normalize_well_key,
)
from scripts.db import StepTimer, get_warehouse_conn, upsert_df, log_quality_flag

_CHUNK_SIZE = 200  # wells per pipeline chunk (smaller than 400 to keep memory reasonable)

#: Витрина собирается здесь и подменяет боевую таблицу одной транзакцией — см.
#: развёрнутую оговорку в :func:`run`. Читатель никогда не видит недособранное.
_STAGING_TABLE = "proc__daily_merged__staging"


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        wells_df = pd.read_sql("SELECT DISTINCT well, well_key FROM raw__v03_runs WHERE well IS NOT NULL", conn)
        if wells_df.empty:
            print("[build_daily_merged] raw__v03_runs is empty — run ingest_v03 first.")
            return

        all_wells = sorted(wells_df["well"].dropna().astype(str).str.strip().unique().tolist())
        print(f"[build_daily_merged] Processing {len(all_wells)} wells in chunks of {_CHUNK_SIZE}...")

        with StepTimer("proc__daily_merged", conn) as timer:
            # ⚠⚠ Собираем в СТОРОНЕ и подменяем одной транзакцией. Раньше здесь стоял
            # `DROP TABLE proc__daily_merged` с последующей дозаписью чанками в
            # АЛФАВИТНОМ порядке скважин (см. `sorted(all_wells)` выше): пока сборка
            # шла, таблица существовала и читалась, но содержала только начало
            # алфавита. Читатель получал кадр в 725 пусков вместо 2306, без Ya и Vt
            # вовсе — и без единого признака поломки: ни исключения, ни пустых
            # колонок, 99.9 % заполненность, правдоподобные распределения. Такой
            # обрезок неотличим от «модель на новых данных поехала».
            # База в WAL, поэтому до COMMIT читатель видит целую СТАРУЮ витрину,
            # после — целую НОВУЮ, а промежуточного состояния не существует.
            conn.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")
            conn.commit()

            total_rows = 0
            first_chunk = True
            for start in range(0, len(all_wells), _CHUNK_SIZE):
                chunk_wells = all_wells[start : start + _CHUNK_SIZE]
                chunk_df = load_daily_merged(chunk_wells)
                if chunk_df.empty:
                    continue

                # Normalize well_id → well_key for consistent joins downstream.
                chunk_df["well_key"] = chunk_df["well_id"].map(normalize_well_key)
                chunk_df = chunk_df.drop(columns=["well_id"])

                # Reorder columns: значение и рядом его провенанс.
                # ⚠⚠ `row_source` — ПОСТРОЧНАЯ метка («в строке есть что-то от
                # телеметрии»), а не ответ на «откуда это значение». Ответ дают
                # колонки `<имя>_src`. `source` оставлен синонимом row_source для
                # существующих потребителей.
                provenance = [f"{c}{SOURCE_SUFFIX}" for c in CANONICAL_COLUMNS]
                # ⚠ Обе газовые оси, плотность и её ключ — ЯВНЫМИ колонками: выбор
                # базы принимает модель по кросс-проверке, данные обязаны дать
                # возможность выбрать, а не решить за неё.
                # ``field``/``lu`` — КОДЫ справочника плотностей (``Bt``/``Vt``),
                # ``field_name``/``lu_name`` — полные названия из телеметрии.
                gas_axes = [
                    GAS_FACTOR_COLUMN, f"{GAS_FACTOR_COLUMN}{SOURCE_SUFFIX}",
                    f"{GAS_LIQUID_RATIO_COLUMN}{SOURCE_SUFFIX}",
                    OIL_DENSITY_COLUMN, "field", "lu", "field_name", "lu_name",
                ]
                # dict.fromkeys дедуплицирует: `gas_liquid_ratio_m3m3_src` попадает и
                # в общий провенанс, и в газовый блок.
                cols = list(dict.fromkeys(
                    ["well_key", "dt", *CANONICAL_COLUMNS, *provenance, *gas_axes,
                     "row_source", "source"]
                ))
                chunk_df = chunk_df[[c for c in cols if c in chunk_df.columns]]

                if_exists = "replace" if first_chunk else "append"
                upsert_df(chunk_df, _STAGING_TABLE, conn, if_exists=if_exists)
                total_rows += len(chunk_df)
                first_chunk = False
                print(f"  wells {start}–{start + len(chunk_wells) - 1}: {len(chunk_df):,} rows (total so far: {total_rows:,})")

            # ⚠⚠ Контроль провенанса В ЛОГЕ СБОРКИ, а не «потом посмотрим по витрине».
            # Ноль телеметрии у колонки — это либо правда (частоты в телеметрии до
            # 2025 года нет вовсе), либо сломанный ключ/формат даты, и различить их
            # постфактум нельзя.
            report = pd.read_sql(
                "SELECT " + ", ".join(f"{c}{SOURCE_SUFFIX}" for c in CANONICAL_COLUMNS)
                + f" FROM {_STAGING_TABLE}",
                conn,
            )
            print("\n[build_daily_merged] Провенанс по колонкам:")
            print(column_provenance_report(report).to_string(index=False))

            # ⚠⚠ Плотность — В ЛОГЕ СБОРКИ. Пропуск ρ означает пропуск пересчёта
            # ГФ→ГЖФ, то есть тихую потерю газовой оси на целых стратах; постфактум
            # это читается как «слой не подтвердился», а не как «данных нет».
            density = pd.read_sql(
                f"SELECT field, lu, {OIL_DENSITY_COLUMN} rho FROM {_STAGING_TABLE}", conn
            )
            filled = density["rho"].notna()
            print(
                f"[build_daily_merged] Плотность: {filled.sum():,} из {len(density):,} строк "
                f"({filled.mean():.1%}), пар (месторождение, ЛУ) "
                f"{density.loc[filled, ['field', 'lu']].drop_duplicates().shape[0]} из "
                f"{density[['field', 'lu']].drop_duplicates().shape[0]}"
            )
            if not filled.all():
                gaps = (density.loc[~filled, ["field", "lu"]].value_counts().head(10))
                print("  ⚠ без плотности:")
                print("   " + gaps.to_string().replace("\n", "\n   "))

            # Подмена одной транзакцией: индекс строится уже на подменённой таблице,
            # внутри той же транзакции, поэтому «таблица без индекса» читателю тоже
            # не видна. commit() до BEGIN закрывает неявную транзакцию sqlite3.
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DROP TABLE IF EXISTS proc__daily_merged")
            conn.execute(f"ALTER TABLE {_STAGING_TABLE} RENAME TO proc__daily_merged")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_proc_daily_merged_key_dt "
                "ON proc__daily_merged (well_key, dt)"
            )
            conn.execute("COMMIT")
            timer.row_count = total_rows

            # Quality check: flag wells with zero rows.
            merged_wells = set(pd.read_sql("SELECT DISTINCT well_key FROM proc__daily_merged", conn)["well_key"].tolist())
            for w in all_wells:
                if normalize_well_key(w) not in merged_wells:
                    log_quality_flag("proc__daily_merged", "no_daily_rows", f"Well {w!r} has no rows in merged daily data", conn=conn, well_key=normalize_well_key(w))
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
