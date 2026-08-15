"""Core data loading: mart + failure categories → one merged DataFrame per run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.db import get_warehouse_conn
from analysis.paths import resolve_svod_main_path
from scripts.data_utils import normalize_well_key
from analysis.ingest.svod.causes import (
    NODE_CATEGORY_COLUMN,
    classify_failure_row as _classify_failure_row,
    has_node_diagnosis,
)
from .config import (
    FAILURE_CATEGORIES, FREQ_LOW_MAX, FREQ_HIGH_MIN, FREQ_VERY_HIGH_MIN,
    FREQ_HIGH_PCT_THRESHOLD, VT_FIELD,
)


# ---------------------------------------------------------------------------
# Failure category classifier
#
# ⚠ Классификатор ЖИВЁТ ОДИН и лежит в сборщике (`analysis.ingest.svod.causes`),
# потому что теперь его применяет САМ Свод: колонка «узел_категория» приезжает
# уже заполненной. Здесь остался только импорт — копия правил в двух местах и
# была тем разрывом, ради которого сборщик переносили в этот репозиторий.
#
# ⚠⚠ У общего классификатора ``default=None``: когда не сработало ни одно
# правило, категории НЕТ. Прежний дефолт «Износ РО» на полном регистре
# приписывал +109 выдуманных износов (278 вместо 169).
# ---------------------------------------------------------------------------

def _s(val) -> str:
    return str(val).strip().lower() if pd.notna(val) else ""


def load_failure_categories() -> pd.DataFrame:
    """Load and classify failures from the Excel workbook.

    Returns one row per Excel row with columns:
    well_key, mount_date, stop_date, failure_category, h2s_class_excel

    ⚠ Источник — ОСНОВНОЙ Свод, а не выгрузка ``_БДА_V03_failures``. В той выгрузке
    отказавший узел заполнен у 1573 строк из 1573, то есть она отфильтрована ПО НАЛИЧИЮ
    диагноза: пуск без узла в неё просто не попадает, и отличить «узел неизвестен» от
    «строки нет» по ней нельзя. Основной регистр покрывает 1577 отказов нашей популяции
    против 1452 и снимает 18 категорий «не определено».

    ⚠⚠ У полного регистра появляется то, чего в выгрузке отказов не было: НЕСКОЛЬКО строк
    на один пуск — например подъём по ГТМ рядом со строкой отказа. Дедупликация поэтому
    сначала сортирует определённые категории вперёд, иначе строка без диагноза может
    вытеснить строку с ним.

    ⚠ Категория БЕРЁТСЯ ИЗ РЕГИСТРА, если он её несёт. Свод, собранный
    ``analysis.ingest.svod``, приезжает уже с колонкой ``узел_категория`` — в этом и был
    смысл переноса сборщика — и пересчитывать её здесь значит снова заводить второе место,
    где живут те же правила. Регистр без этой колонки (собранный прежним сборщиком)
    классифицируется на лету тем же самым классификатором.
    """
    path = Path(resolve_svod_main_path())
    base_columns = [
        "Скв.", "Дата монтажа", "Дата остановки",
        "Отказавший узел", "Отказавший элемент",
        "Характер неисправности", "Причина отказа УЭЦН",
        "Кислый/Некислый",
    ]
    available = set(pd.read_excel(path, sheet_name="Свод", header=0, nrows=0).columns)
    has_precomputed = NODE_CATEGORY_COLUMN in available
    df = pd.read_excel(
        path,
        sheet_name="Свод",
        header=0,
        usecols=base_columns + ([NODE_CATEGORY_COLUMN] if has_precomputed else []),
    )
    df["well_key"]   = df["Скв."].astype("string").str.strip().map(normalize_well_key)
    df["mount_date"] = pd.to_datetime(df["Дата монтажа"], errors="coerce").dt.normalize()
    df["stop_date"]  = pd.to_datetime(df["Дата остановки"], errors="coerce").dt.normalize()
    if has_precomputed:
        df["failure_category"] = df[NODE_CATEGORY_COLUMN].where(
            df[NODE_CATEGORY_COLUMN].notna(), None
        )
    else:
        # default=None: пустая категория вместо молчаливого «Износ РО» — см. оговорку там
        df["failure_category"] = df.apply(_classify_failure_row, default=None, axis=1)
    df["h2s_class_excel"]  = (
        df["Кислый/Некислый"].astype("string").str.strip()
        .replace({"<NA>": "<missing>", "nan": "<missing>"})
        .fillna("<missing>")
    )
    # ⚠⚠ БЕЗ ЭТОГО ПЕРЕКЛЮЧЕНИЕ ИСТОЧНИКА ФАБРИКУЕТ ИЗНОС. `_classify_failure_row`
    # возвращает «Износ РО» ДЕФОЛТОМ, когда не сработало ни одно правило. В выгрузке
    # отказов это было безобидно (там узел заполнен у всех строк), а в полном регистре
    # почти половина строк диагноза не несёт — и все они получили бы «Износ РО»:
    # на нашей популяции это 278 отказов вместо 169, то есть +109 выдуманных износов.
    # Нет диагноза — значит категории нет, и она должна остаться пустой.
    _empty = ["", "нет", "-", "—", "н/д", "nan", "<na>", "none"]
    no_node = (df["Отказавший узел"].astype("string").str.strip().str.casefold()
               .isin(_empty).fillna(True))
    no_elem = (df["Отказавший элемент"].astype("string").str.strip().str.casefold()
               .isin(_empty).fillna(True))
    df["_нет_диагноза"] = (no_node & no_elem).astype(int)
    df.loc[no_node & no_elem, "failure_category"] = pd.NA
    result = (
        df.sort_values("_нет_диагноза", kind="stable")
        [["well_key", "mount_date", "stop_date", "failure_category", "h2s_class_excel"]]
        .dropna(subset=["well_key", "mount_date", "stop_date"])
        .drop_duplicates(subset=["well_key", "mount_date", "stop_date"], keep="first")
        .reset_index(drop=True)
    )
    return result


def load_mart() -> pd.DataFrame:
    """Load mart__vt_freq55 with normalised date columns."""
    conn = get_warehouse_conn()
    try:
        df = pd.read_sql(
            "SELECT * FROM mart__vt_freq55",
            conn,
            parse_dates=["install_date", "stop_date"],
        )
    finally:
        conn.close()

    df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce").dt.normalize()
    df["stop_date"]    = pd.to_datetime(df["stop_date"],    errors="coerce").dt.normalize()
    return df


def load_analysis_df() -> pd.DataFrame:
    """Full merged analysis dataset: mart + failure categories + derived columns."""
    mart = load_mart()
    cats = load_failure_categories()

    # Merge: match on (well_key, install_date ↔ mount_date, stop_date)
    mart = mart.rename(columns={"install_date": "mount_date"})
    merged = mart.merge(
        cats,
        on=["well_key", "mount_date", "stop_date"],
        how="left",
    )
    # Censored runs correctly get NaN failure_category → replace with "<missing>"
    merged["failure_category"] = merged["failure_category"].fillna("<missing>")
    merged["h2s_class_excel"]  = merged["h2s_class_excel"].fillna("<missing>")

    # --- Derived columns ---
    # Duration: prefer ttf_true_best_days, fall back to run_days
    merged["duration"] = np.where(
        merged["ttf_true_best_days"].notna(),
        merged["ttf_true_best_days"],
        merged["run_days"],
    )
    merged["using_true_ttf"] = merged["ttf_true_best_days"].notna()

    # Frequency group
    merged["freq_group"] = pd.cut(
        merged["freq_w_mean"],
        bins=[-np.inf, FREQ_LOW_MAX, FREQ_HIGH_MIN, np.inf],
        labels=["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"],
        right=True,
    ).astype("string").fillna("<no freq data>")

    # Proportion-based frequency group:
    # "High" = majority of run time (>50%) was above 55 Hz
    # Falls back to mean-based definition when freq_above_55hz_pct is missing
    pct_col = "freq_above_55hz_pct"
    has_pct = merged[pct_col].notna()
    merged["freq_group_pct"] = np.where(
        ~has_pct,
        merged["freq_group"],  # fallback to mean-based when telemetry missing
        np.where(
            merged[pct_col] > FREQ_HIGH_PCT_THRESHOLD,
            "High (>55 Hz)",
            np.where(
                merged["freq_w_mean"].notna() & (merged["freq_w_mean"] <= FREQ_LOW_MAX),
                "Low (≤50 Hz)",
                "Normal (50–55 Hz)",
            ),
        ),
    )
    merged["freq_group_pct"] = merged["freq_group_pct"].astype("string").fillna("<no freq data>")

    # Mount year
    merged["mount_year"] = merged["mount_date"].dt.year

    # Very-high frequency flag
    merged["very_high_freq"] = (
        merged["freq_w_mean"].notna() & (merged["freq_w_mean"] > FREQ_VERY_HIGH_MIN)
    )

    # H2S label (proxy threshold + excel label combined)
    # Excel label takes precedence where available; proxy fills the rest
    H2S_THRESHOLD = 3.0
    merged["h2s_label"] = np.where(
        merged["h2s_class_excel"].isin(["Кислый", "Некислый"]),
        merged["h2s_class_excel"],
        np.where(
            merged["h2s_proxy_mg_l"].notna(),
            np.where(merged["h2s_proxy_mg_l"] >= H2S_THRESHOLD, "Кислый", "Некислый"),
            "<missing>",
        ),
    )

    # is_vt flag
    merged["is_vt"] = merged["field"] == VT_FIELD

    # Normalised duty (only where run > 0)
    dur_safe = merged["duration"].replace(0, np.nan)
    merged["trf_per_day"] = merged["total_freq_hz_days"] / dur_safe
    merged["tlf_per_day"] = merged["total_liquid_m3"] / dur_safe

    return merged


__all__ = ["load_analysis_df", "load_failure_categories", "load_mart"]
