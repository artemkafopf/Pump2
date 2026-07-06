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
from analysis.paths import resolve_v03_failures_path
from scripts.data_utils import normalize_well_key
from .config import (
    FAILURE_CATEGORIES, FREQ_LOW_MAX, FREQ_HIGH_MIN, FREQ_VERY_HIGH_MIN,
    FREQ_HIGH_PCT_THRESHOLD, VT_FIELD,
)


# ---------------------------------------------------------------------------
# Failure category classifier (mirrors streamlit_apps/freq_exposure_app.py)
# ---------------------------------------------------------------------------

def _s(val) -> str:
    return str(val).strip().lower() if pd.notna(val) else ""


def _classify_failure_row(row) -> str:
    uzl  = _s(row.get("Отказавший узел", ""))
    elem = _s(row.get("Отказавший элемент", ""))
    char = _s(row.get("Характер неисправности", ""))
    prch = _s(row.get("Причина отказа УЭЦН", ""))

    cable_elems = {
        "кабельный удлинитель", "основная длина", "кабельный сросток",
        "кабельная муфта", "термовставка", "сальниковая разделка",
    }
    motor_elems = {"статор с обмоткой", "верхнее лобовое", "ротор", "выводные концы", "колодка токоввода"}
    nkt_uzly    = {"нкт", "нкт ", "клапан сливной", "подвесной патрубок", "клапан обратный",
                   "мандрель", "переводник", "подвесной патрубок "}
    clog_words  = ("засорен", "твердые отложения", "солеотложени")
    wear_words  = ("разрушен", "износ", "осевой", "пар трения", "радиальный", "промыв", "трещин", "эрозион")

    if uzl == "кабельная линия": return "КЛ (R-0)"
    if uzl == "тмс":             return "КЛ (R-0)"
    if elem in cable_elems and any(w in char for w in ("изоляц", "прогар", "оплавл", "механич", "разрушен")):
        return "КЛ (R-0)"
    if uzl == "пэд" and elem not in ("узел пяты", "шлицевая муфта"):
        return "ПЭД (R-0)"
    if elem in motor_elems and any(w in char for w in ("замыкание", "электропробой", "прогар", "перегрев", "изоляц")):
        return "ПЭД (R-0)"
    if uzl == "гидрозащита": return "Износ/негермет.гидрозащиты"
    if "вал" in elem and "слом" in char: return "Слом вала"
    if "шлицевая муфта" in elem and any(w in char for w in ("слом", "разрушен")) and uzl != "гидрозащита":
        return "Слом вала"
    if "корпус" in elem and "слом" in char: return "Слом вала"
    if uzl in nkt_uzly or "нкт" in uzl: return "НКТ"
    if "подвеска нкт" in elem: return "НКТ"

    pump_uzly = {"эцн", "газосепаратор", "диспергатор", "входной модуль"}
    if uzl in pump_uzly:
        if "рабочие органы" in elem:
            if any(w in char for w in wear_words):  return "Износ РО"
            if any(w in char for w in clog_words):  return "Засорение РО"
            if any(w in prch for w in ("засорен", "солеотложени")): return "Засорение РО"
            return "Засорение РО"
        if "вал" in elem: return "Слом вала"
        if any(w in char for w in wear_words) or any(w in prch for w in ("коррозия", "эрозион")):
            return "Износ РО"
        if any(w in char for w in clog_words) or any(w in prch for w in ("засорен", "солеотложени")):
            return "Засорение РО"
        return "Износ РО"

    if uzl == "пэд" and "узел пяты" in elem: return "Износ РО"
    if uzl == "пэд": return "ПЭД (R-0)"

    scores = {cat: 0 for cat in FAILURE_CATEGORIES}
    if elem in cable_elems:                          scores["КЛ (R-0)"] += 2
    if any(w in char for w in ("кабел", "изоляц")): scores["КЛ (R-0)"] += 1
    if elem in motor_elems:                          scores["ПЭД (R-0)"] += 2
    if "замыкание" in char:                          scores["ПЭД (R-0)"] += 2
    if "вал" in elem:                                scores["Слом вала"] += 2
    if "слом" in char:                               scores["Слом вала"] += 2
    if "рабочие органы" in elem:
        if any(w in char for w in clog_words): scores["Засорение РО"] += 3
        if any(w in char for w in wear_words): scores["Износ РО"] += 3
    if any(w in char for w in clog_words): scores["Засорение РО"] += 1
    if any(w in char for w in wear_words): scores["Износ РО"] += 1
    if "нкт" in uzl or "подвеска нкт" in elem: scores["НКТ"] += 2
    if "обрыв" in char:                        scores["НКТ"] += 1
    if any(w in elem for w in ("уплотнен", "пяты")) or "пята" in char:
        scores["Износ/негермет.гидрозащиты"] += 2
    if "негермет" in char: scores["Износ/негермет.гидрозащиты"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Износ РО"


def load_failure_categories() -> pd.DataFrame:
    """Load and classify failures from the Excel workbook.

    Returns one row per Excel row with columns:
    well_key, mount_date, stop_date, failure_category, h2s_class_excel
    """
    path = Path(resolve_v03_failures_path())
    df = pd.read_excel(
        path,
        sheet_name="Свод",
        header=0,
        usecols=[
            "Скв.", "Дата монтажа", "Дата остановки",
            "Отказавший узел", "Отказавший элемент",
            "Характер неисправности", "Причина отказа УЭЦН",
            "Кислый/Некислый",
        ],
    )
    df["well_key"]   = df["Скв."].astype("string").str.strip().map(normalize_well_key)
    df["mount_date"] = pd.to_datetime(df["Дата монтажа"], errors="coerce").dt.normalize()
    df["stop_date"]  = pd.to_datetime(df["Дата остановки"], errors="coerce").dt.normalize()
    df["failure_category"] = df.apply(_classify_failure_row, axis=1)
    df["h2s_class_excel"]  = (
        df["Кислый/Некислый"].astype("string").str.strip()
        .replace({"<NA>": "<missing>", "nan": "<missing>"})
        .fillna("<missing>")
    )
    result = (
        df[["well_key", "mount_date", "stop_date", "failure_category", "h2s_class_excel"]]
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
