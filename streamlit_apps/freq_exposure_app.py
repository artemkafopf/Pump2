from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for path_text in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from scripts.db import WAREHOUSE_PATH, get_warehouse_conn, get_last_pipeline_run
from scripts.data_utils import normalize_well_key


st.set_page_config(
    page_title="TTF vs Frequency Exposure",
    page_icon=":material/show_chart:",
    layout="wide",
)

FREQ_THRESHOLD_HZ = 55.0
LOW_FREQ_THRESHOLD_HZ = 45.0
MIN_FREQ_DAYS = 7
DEFAULT_BOUNDARIES_TEXT = "-0.5, -0.2, 0.2, 0.5"
DEFAULT_GLF_BOUNDARIES_TEXT = "100, 300, 500, 1000"
DEFAULT_KPOD_BOUNDARIES_TEXT = "0.3, 0.5, 0.7, 1.0"
DEFAULT_KPOD_FREQ_BOUNDARIES_TEXT = "0.3, 0.5, 0.7, 1.0"
DEFAULT_PRECIPITATE_PROXY_BOUNDARIES_TEXT = "1000, 5000, 10000, 50000"
DEFAULT_CA_BOUNDARIES_TEXT = "500000, 2000000, 10000000, 30000000"
DEFAULT_CL_BOUNDARIES_TEXT = "1000000, 5000000, 20000000, 80000000"
DEFAULT_SO4_BOUNDARIES_TEXT = "5000, 50000, 200000, 1000000"
DEFAULT_H2S_THRESHOLD_MG_L = 3.0
DEFAULT_H2S_PROXY_BOUNDARIES_TEXT = "1, 3, 10, 30"
DEFAULT_MEAN_FREQ_BOUNDARIES_TEXT = "45, 50, 55, 60"
COLOR_GROUP_OPTIONS = [
    "Mount year",
    "Y exposure bin",
    "GLF bin",
    "Kpod bin",
    "Kpod_freq bin",
    "Precipitate proxy bin",
    "Failure category",
    "H2S class",
    "H2S proxy bin",
    "Ca proxy bin",
    "Cl proxy bin",
    "SO₄ proxy bin",
    "Mean freq bin",
]

FAILURE_CATEGORIES = [
    "КЛ (R-0)",
    "ПЭД (R-0)",
    "Слом вала",
    "Засорение РО",
    "НКТ",
    "Износ РО",
    "Износ/негермет.гидрозащиты",
]

# Stable colour palette for H2S class.
_H2S_PALETTE = {
    "Кислый": "#d62728",    # red   — acidic
    "Некислый": "#2ca02c",  # green — non-acidic
    "<missing>": "#B6B6B6",
}

_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "TTF vs Frequency Exposure",
        "main_title": "TTF vs Signed Frequency Exposure",
        "main_caption": "Simple local GUI for checking frequency-exposure histograms globally or by field.",
        "info_text": (
            "Signed exposure: `y = share(time > upper Hz) - share(time < lower Hz)`. "
            "Configure thresholds under **Frequency Bins** in the sidebar. "
            "Positive y = more high-frequency exposure; negative y = more low-frequency exposure."
        ),
        "lang_label": "Language / Язык",
        "filters_header": "Filters",
        "field_label": "Field",
        "contractor_label": "Contractor",
        "infant_label": "Infant mortality threshold (days)",
        "year0_label": "Use runs with mount year > year0",
        "freq_bins_header": "Frequency Bins",
        "upper_hz_label": "Upper threshold (Hz)",
        "lower_hz_label": "Lower threshold (Hz)",
        "boundary_label": "Boundary list",
        "boundary_help": "Comma-separated signed y boundaries, e.g.: -0.5, -0.2, 0.2, 0.5",
        "glf_header": "GLF Filters",
        "glf_boundaries_label": "GLF boundaries",
        "kpod_header": "Kpod Filters",
        "kpod_boundaries_label": "Kpod boundaries",
        "kpod_freq_header": "Kpod_freq Filters",
        "kpod_freq_boundaries_label": "Kpod_freq boundaries",
        "precipitate_header": "Precipitate Proxy",
        "precipitate_boundaries_label": "Precipitate proxy boundaries",
        "ions_header": "Ca / Cl / SO₄ Proxies",
        "ca_boundaries_label": "Ca boundaries (kg)",
        "cl_boundaries_label": "Cl boundaries (kg)",
        "so4_boundaries_label": "SO₄ boundaries (kg)",
        "h2s_header": "H2S",
        "h2s_method_label": "H2S classification method",
        "h2s_threshold_label": "H2S proxy threshold (mg/L)",
        "h2s_threshold_help": "Runs with h2s_proxy_mg_l ≥ threshold → Кислый, else → Некислый",
        "h2s_proxy_boundaries_label": "H2S proxy bin boundaries (mg/L)",
        "h2s_proxy_help": "Used when coloring/grouping by 'H2S proxy bin'",
        "ttf_def_label": "TTF definition",
        "ttf_def_help": "TTF uses calendar run days from the workbook. TTF_true_best uses operational days from techregime status, falling back to telemetry qliq > 0.",
        "xmode_label": "Plot x-axis",
        "xmode_help": "TTF = time to failure in days. TLF = cumulative liquid to failure. TRF = cumulative frequency to failure.",
        "colormode_label": "Color scatter by",
        "groupmode_label": "Group histogram / survival by",
        "swap_axes_label": "Swap X-Y on TTF pair correlations",
        "logscale_label": "Log-log scale on TRF vs TTF",
        "failed_nodes_label": "Keep failed nodes",
        "failure_cats_label": "Failure categories",
        "h2s_class_label": "H2S class",
        "glf_bins_label": "Keep GLF bins",
        "kpod_bins_label": "Keep Kpod bins",
        "kpod_freq_bins_label": "Keep Kpod_freq bins",
        "ca_bins_label": "Ca proxy bins",
        "cl_bins_label": "Cl proxy bins",
        "so4_bins_label": "SO₄ proxy bins",
        "mean_freq_bins_label": "Mean frequency bins",
        "mean_freq_boundaries_label": "Mean frequency boundaries",
        "mean_freq_boundaries_help": "Comma-separated average-frequency boundaries, e.g.: 45, 50, 55, 60",
        "tab_freq": "Frequency Exposure",
        "tab_infant": "Infant Mortality",
        "runs_metric": "Runs (≥ threshold)",
        "failures_metric": "Failures (≥ threshold)",
        "infant_metric": "Infant failures (< {n} d)",
        "median_trf": "Median TRF",
        "median_tlf": "Median TLF",
        "scatter_title": "{field}: TTF vs signed frequency exposure",
        "scatter_yaxis": "y = share(time > {upper:.0f} Hz) − share(time < {lower:.0f} Hz)",
        "hist_title": "{field}: failure-event histogram grouped by {group}",
        "survival_title": "{field}: Kaplan-Meier survival grouped by {group}",
        "pair_corr_header": "TTF Pair Correlations",
        "grouped_summary_header": "Grouped Summary",
        "download_btn": "Download filtered dataset CSV",
        "trf_label": "TRF / cumulative frequency to failure (Hz-days)",
        "tlf_label": "TLF / cumulative liquid to failure (m3)",
        "hz_limit": "{hz:.0f} Hz limit",
        "hz_reference": "{hz:.0f} Hz reference",
        "infant_tab_header": "Infant Mortality — {field}",
        "infant_tab_caption": (
            "True failures (event = 1) with run time **< {n} days** and mount year > {year}. "
            "Use the **Failure categories** filter in the sidebar to drill down."
        ),
        "infant_failures_metric": "Infant failures",
        "all_failures_metric": "All failures (same period)",
        "infant_share_metric": "Infant share",
        "cat_breakdown_header": "Category breakdown",
        "ttf_dist_header": "TTF distribution of infant failures",
    },
    "ru": {
        "page_title": "ВРО vs Частотная экспозиция",
        "main_title": "ВРО vs Знаковая частотная экспозиция",
        "main_caption": "Локальный GUI для анализа частотной экспозиции глобально или по месторождению.",
        "info_text": (
            "Знаковая экспозиция: `y = доля(время > верх. Гц) − доля(время < нижн. Гц)`. "
            "Настройте пороги в разделе **Частотные бины** на панели. "
            "y > 0 — больше высокочастотной нагрузки; y < 0 — больше низкочастотной."
        ),
        "lang_label": "Language / Язык",
        "filters_header": "Фильтры",
        "field_label": "Месторождение",
        "contractor_label": "Подрядчик",
        "infant_label": "Порог инфантильной смертности (дни)",
        "year0_label": "Запуски с годом монтажа > year0",
        "freq_bins_header": "Частотные бины",
        "upper_hz_label": "Верхний порог (Гц)",
        "lower_hz_label": "Нижний порог (Гц)",
        "boundary_label": "Список границ",
        "boundary_help": "Границы y через запятую, напр.: -0.5, -0.2, 0.2, 0.5",
        "glf_header": "Фильтры GLF",
        "glf_boundaries_label": "Границы GLF",
        "kpod_header": "Фильтры Kpod",
        "kpod_boundaries_label": "Границы Kpod",
        "kpod_freq_header": "Фильтры Kpod_freq",
        "kpod_freq_boundaries_label": "Границы Kpod_freq",
        "precipitate_header": "Прокси осадков",
        "precipitate_boundaries_label": "Границы прокси осадков",
        "ions_header": "Прокси Ca / Cl / SO₄",
        "ca_boundaries_label": "Границы Ca (кг)",
        "cl_boundaries_label": "Границы Cl (кг)",
        "so4_boundaries_label": "Границы SO₄ (кг)",
        "h2s_header": "H2S",
        "h2s_method_label": "Метод классификации H2S",
        "h2s_threshold_label": "Порог прокси H2S (мг/л)",
        "h2s_threshold_help": "Запуски с h2s_proxy_mg_l ≥ порога → Кислый, иначе → Некислый",
        "h2s_proxy_boundaries_label": "Границы бинов прокси H2S (мг/л)",
        "h2s_proxy_help": "Используется при раскраске/группировке по 'H2S proxy bin'",
        "ttf_def_label": "Определение ВРО",
        "ttf_def_help": "ВРО — календарные дни из журнала. TTF_true_best — операционные дни из техрежима или телеметрии (qliq > 0).",
        "xmode_label": "Ось X графика",
        "xmode_help": "ВРО = время до отказа (дни). ОПЖ = накопленная добыча жидкости. НЧЖ = накопленная частота·дни.",
        "colormode_label": "Раскраска точек по",
        "groupmode_label": "Группировка гистограммы / выживаемости по",
        "swap_axes_label": "Поменять X-Y на графиках ВРО",
        "logscale_label": "Лог-лог шкала на НЧЖ vs ВРО",
        "failed_nodes_label": "Отказавшие узлы",
        "failure_cats_label": "Категории отказов",
        "h2s_class_label": "Класс H2S",
        "glf_bins_label": "Бины GLF",
        "kpod_bins_label": "Бины Kpod",
        "kpod_freq_bins_label": "Бины Kpod_freq",
        "ca_bins_label": "Бины прокси Ca",
        "cl_bins_label": "Бины прокси Cl",
        "so4_bins_label": "Бины прокси SO₄",
        "mean_freq_bins_label": "Бины средней частоты",
        "mean_freq_boundaries_label": "Границы средней частоты",
        "mean_freq_boundaries_help": "Границы средней частоты через запятую, напр.: 45, 50, 55, 60",
        "tab_freq": "Частотная экспозиция",
        "tab_infant": "Инфантильная смертность",
        "runs_metric": "Запуски (≥ порога)",
        "failures_metric": "Отказы (≥ порога)",
        "infant_metric": "Инфантильные отказы (< {n} д)",
        "median_trf": "Медиана НЧЖ",
        "median_tlf": "Медиана ОПЖ",
        "scatter_title": "{field}: ВРО vs знаковая частотная экспозиция",
        "scatter_yaxis": "y = доля(время > {upper:.0f} Гц) − доля(время < {lower:.0f} Гц)",
        "hist_title": "{field}: гистограмма отказов, группировка по {group}",
        "survival_title": "{field}: выживаемость Каплана-Мейера, группировка по {group}",
        "pair_corr_header": "Парные корреляции ВРО",
        "grouped_summary_header": "Сводная таблица",
        "download_btn": "Скачать отфильтрованные данные CSV",
        "trf_label": "НЧЖ / накопленная частота до отказа (Гц·дни)",
        "tlf_label": "ОПЖ / накопленная жидкость до отказа (м³)",
        "hz_limit": "предел {hz:.0f} Гц",
        "hz_reference": "реф. {hz:.0f} Гц",
        "infant_tab_header": "Инфантильная смертность — {field}",
        "infant_tab_caption": (
            "Истинные отказы (event = 1) при наработке **< {n} дней** и годе монтажа > {year}. "
            "Используйте фильтр **Категории отказов** на панели для детализации."
        ),
        "infant_failures_metric": "Инфантильные отказы",
        "all_failures_metric": "Все отказы (тот же период)",
        "infant_share_metric": "Доля инфантильных",
        "cat_breakdown_header": "Разбивка по категориям",
        "ttf_dist_header": "Распределение ВРО инфантильных отказов",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    label = _LABELS.get(lang, _LABELS["en"]).get(key) or _LABELS["en"].get(key, key)
    if kwargs:
        try:
            return label.format(**kwargs)
        except (KeyError, ValueError):
            return label
    return label


# Colour palette for failure categories — kept stable across tabs.
_CATEGORY_PALETTE = {
    "КЛ (R-0)": "#636EFA",
    "ПЭД (R-0)": "#EF553B",
    "Слом вала": "#00CC96",
    "Засорение РО": "#AB63FA",
    "НКТ": "#FFA15A",
    "Износ РО": "#19D3F3",
    "Износ/негермет.гидрозащиты": "#FF6692",
    "<missing>": "#B6B6B6",
}


def parse_boundaries(text: str) -> list[float]:
    parts = [item.strip() for item in str(text).split(",")]
    values: list[float] = []
    for part in parts:
        if not part:
            continue
        values.append(float(part))
    unique_sorted = sorted(set(values))
    if len(unique_sorted) < 2:
        raise ValueError("Provide at least two boundary values.")
    return unique_sorted


def build_bin_labels(boundaries: list[float], symbol: str = "y") -> list[str]:
    labels = [f"{symbol} <= {boundaries[0]:.3f}"]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
        labels.append(f"{left:.3f} < {symbol} <= {right:.3f}")
    labels.append(f"{symbol} > {boundaries[-1]:.3f}")
    return labels


def assign_bin(exposure: float, boundaries: list[float], symbol: str = "y") -> str:
    labels = build_bin_labels(boundaries, symbol=symbol)
    if pd.isna(exposure):
        return "<missing>"
    if exposure <= boundaries[0]:
        return labels[0]
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False), start=1):
        if left < exposure <= right:
            return labels[index]
    return labels[-1]


def palette_for_labels(labels: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Dark24
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def mean_freq_bin_labels(boundaries: list[float]) -> list[str]:
    labels = [f"<= {boundaries[0]:.3f} Hz"]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
        labels.append(f"{left:.3f}-{right:.3f} Hz")
    labels.append(f"> {boundaries[-1]:.3f} Hz")
    return labels


def assign_mean_freq_bin(avg_freq_hz: float, boundaries: list[float]) -> str:
    labels = mean_freq_bin_labels(boundaries)
    if pd.isna(avg_freq_hz):
        return "<missing>"
    if avg_freq_hz <= boundaries[0]:
        return labels[0]
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False), start=1):
        if left < avg_freq_hz <= right:
            return labels[index]
    return labels[-1]


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    event_times = np.unique(durations[events == 1])
    if len(event_times) == 0:
        return np.array([0.0]), np.array([1.0])

    times = np.concatenate([[0.0], event_times])
    survival = np.ones(len(times), dtype=float)
    current = 1.0
    for index, current_time in enumerate(event_times, start=1):
        at_risk = int(np.sum(durations >= current_time))
        n_events = int(np.sum((durations == current_time) & (events == 1)))
        if at_risk > 0:
            current *= 1.0 - (n_events / at_risk)
        survival[index] = current
    return times, survival


def _classify_failure_row(row: "pd.Series") -> str:
    """Classify a failure row into one of the 7 standard categories.

    Uses four columns (may be from the raw failures Excel or renamed copies):
      uzl_col, elem_col, char_col, prch_col.
    Column names are resolved via the ``_CLASSIFY_COLS`` mapping below.
    """
    def _s(val) -> str:
        return str(val).strip().lower() if pd.notna(val) else ""

    uzl  = _s(row.get("_uzl",  row.get("Отказавший узел", "")))
    elem = _s(row.get("_elem", row.get("Отказавший элемент", "")))
    char = _s(row.get("_char", row.get("Характер неисправности", "")))
    prch = _s(row.get("_prch", row.get("Причина отказа УЭЦН", "")))

    cable_elems = {"кабельный удлинитель", "основная длина", "кабельный сросток",
                   "кабельная муфта", "термовставка", "сальниковая разделка"}
    motor_elems = {"статор с обмоткой", "верхнее лобовое", "ротор", "выводные концы",
                   "колодка токоввода"}
    nkt_uzly    = {"нкт", "нкт ", "клапан сливной", "подвесной патрубок", "клапан обратный",
                   "мандрель", "переводник", "подвесной патрубок "}
    clog_words  = ("засорен", "твердые отложения", "солеотложени")
    wear_words  = ("разрушен", "износ", "осевой", "пар трения", "радиальный", "промыв", "трещин", "эрозион")

    # 1. Cable line (КЛ)
    if uzl == "кабельная линия":
        return "КЛ (R-0)"
    if uzl == "тмс":
        return "КЛ (R-0)"
    if elem in cable_elems and any(w in char for w in ("изоляц", "прогар", "оплавл", "механич", "разрушен")):
        return "КЛ (R-0)"

    # 2. Motor (ПЭД)
    if uzl == "пэд" and elem not in ("узел пяты", "шлицевая муфта"):
        return "ПЭД (R-0)"
    if elem in motor_elems and any(w in char for w in ("замыкание", "электропробой", "прогар", "перегрев", "изоляц")):
        return "ПЭД (R-0)"

    # 3. Hydro-protection — takes priority over shaft checks
    if uzl == "гидрозащита":
        return "Износ/негермет.гидрозащиты"

    # 4. Shaft break
    if "вал" in elem and "слом" in char:
        return "Слом вала"
    if "шлицевая муфта" in elem and any(w in char for w in ("слом", "разрушен")) and uzl != "гидрозащита":
        return "Слом вала"
    if "корпус" in elem and "слом" in char:
        return "Слом вала"

    # 5. NKT group
    if uzl in nkt_uzly or "нкт" in uzl:
        return "НКТ"
    if "подвеска нкт" in elem:
        return "НКТ"

    # 6. Pump unit (ЭЦН, gas separator, disperser, inlet module)
    pump_uzly = {"эцн", "газосепаратор", "диспергатор", "входной модуль"}
    if uzl in pump_uzly:
        if "рабочие органы" in elem:
            if any(w in char for w in wear_words):
                return "Износ РО"
            if any(w in char for w in clog_words):
                return "Засорение РО"
            if any(w in prch for w in ("засорен", "солеотложени")):
                return "Засорение РО"
            return "Засорение РО"
        if "вал" in elem:
            return "Слом вала"
        # Pump body / head / other element
        if any(w in char for w in wear_words) or any(w in prch for w in ("коррозия", "эрозион")):
            return "Износ РО"
        if any(w in char for w in clog_words) or any(w in prch for w in ("засорен", "солеотложени")):
            return "Засорение РО"
        return "Износ РО"

    # ПЭД thrust bearing → wear
    if uzl == "пэд" and "узел пяты" in elem:
        return "Износ РО"
    if uzl == "пэд":
        return "ПЭД (R-0)"

    # --- Scoring fallback ---
    scores: dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}
    if elem in cable_elems: scores["КЛ (R-0)"] += 2
    if any(w in char for w in ("кабел", "изоляц")): scores["КЛ (R-0)"] += 1
    if elem in motor_elems: scores["ПЭД (R-0)"] += 2
    if "замыкание" in char: scores["ПЭД (R-0)"] += 2
    if "вал" in elem: scores["Слом вала"] += 2
    if "слом" in char: scores["Слом вала"] += 2
    if "рабочие органы" in elem:
        if any(w in char for w in clog_words): scores["Засорение РО"] += 3
        if any(w in char for w in wear_words): scores["Износ РО"] += 3
    if any(w in char for w in clog_words): scores["Засорение РО"] += 1
    if any(w in char for w in wear_words): scores["Износ РО"] += 1
    if "нкт" in uzl or "подвеска нкт" in elem: scores["НКТ"] += 2
    if "обрыв" in char: scores["НКТ"] += 1
    if any(w in elem for w in ("уплотнен", "пяты")) or "пята" in char:
        scores["Износ/негермет.гидрозащиты"] += 2
    if "негермет" in char: scores["Износ/негермет.гидрозащиты"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Износ РО"


@st.cache_resource
def _warehouse_conn():
    """Cached warehouse connection (one per Streamlit session)."""
    if not WAREHOUSE_PATH.exists():
        return None
    return get_warehouse_conn()


@st.cache_data(show_spinner=False, ttl=3600)
def load_failure_categories() -> pd.DataFrame:
    """Load and classify all failures from the raw failures workbook.

    Returns a DataFrame with columns [well_key, mount_date, stop_date, failure_category]
    suitable for left-joining onto mart data.
    """
    try:
        from analysis.input_paths import resolve_v03_failures_path
        path = Path(resolve_v03_failures_path())
    except Exception:
        return pd.DataFrame()

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_excel(
            path,
            sheet_name="Свод",
            header=0,
            usecols=["Скв.", "Дата монтажа", "Дата остановки",
                     "Отказавший узел", "Отказавший элемент",
                     "Характер неисправности", "Причина отказа УЭЦН",
                     "Кислый/Некислый"],
        )
    except Exception:
        return pd.DataFrame()

    df["well_key"] = df["Скв."].astype("string").str.strip().map(normalize_well_key)
    df["mount_date"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    df["stop_date"] = pd.to_datetime(df["Дата остановки"], errors="coerce")
    df["failure_category"] = df.apply(_classify_failure_row, axis=1)
    df["h2s_class_excel"] = df["Кислый/Некислый"].astype("string").str.strip().fillna("<missing>")

    return (
        df[["well_key", "mount_date", "stop_date", "failure_category", "h2s_class_excel"]]
        .dropna(subset=["well_key", "mount_date", "stop_date"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_run_stats(field_code: str) -> pd.DataFrame:
    """Load pre-computed run stats from mart__vt_freq55 in the warehouse."""
    conn = _warehouse_conn()
    if conn is None:
        st.warning("Warehouse not found. Run `python scripts/pipeline.py` to build it.")
        return pd.DataFrame()

    try:
        df = pd.read_sql("SELECT * FROM mart__vt_freq55", conn, parse_dates=["install_date", "stop_date"])
    except Exception as exc:
        st.warning(f"mart__vt_freq55 not available yet: {exc}. Run the pipeline first.")
        return pd.DataFrame()

    if field_code != "GLOBAL":
        df = df.loc[df["field"].astype("string").str.strip() == field_code].copy()

    df = df.rename(columns={
        "well": "well_id",
        "install_date": "mount_date",
        "run_days": "ttf_days",
        "freq_above_55hz_pct": "frac_freq_above_55",
        "freq_below_45hz_pct": "frac_freq_below_45",
        "freq_signed_exposure": "signed_freq_exposure",
        "freq_w_mean": "avg_freq_hz",
        "n_freq_valid_days": "n_freq_days",
        "n_freq_above_55hz": "n_freq_days_above_55",
        "n_freq_below_45hz": "n_freq_days_below_45",
        "total_liquid_m3": "tlf_total_liquid_m3",
        "total_freq_hz_days": "trf_total_frequency_hz_days",
        "cum_salt_load_kg": "salt_proxy_total_kg",
        "cum_calcium_load_kg": "calcium_proxy_kg",
        "cum_chloride_load_kg": "chloride_proxy_kg",
        "cum_sulfate_load_kg": "sulfate_proxy_kg",
    })

    if "avg_kpod_freq" not in df.columns:
        df["avg_kpod_freq"] = float("nan")
    if "salt_proxy_total_kg" not in df.columns and "integrated_salt_proxy_m" in df.columns:
        df["salt_proxy_total_kg"] = df["integrated_salt_proxy_m"]
    if "mount_date" in df.columns:
        df["mount_year"] = pd.to_datetime(df["mount_date"], errors="coerce").dt.year.astype("Int64")

    return df.reset_index(drop=True)


def filtered_view(
    df: pd.DataFrame,
    infant_days: int,
    year0: int,
    boundaries: list[float],
    glf_boundaries: list[float],
    kpod_boundaries: list[float],
    kpod_freq_boundaries: list[float],
    mean_freq_boundaries: list[float],
    salt_proxy_boundaries: list[float],
    ca_boundaries: list[float],
    cl_boundaries: list[float],
    so4_boundaries: list[float],
    h2s_proxy_boundaries: list[float],
    selected_contractors: list[str],
    selected_failed_nodes: list[str],
    selected_failure_categories: list[str],
    selected_h2s_classes: list[str],
    selected_glf_bins: list[str],
    selected_kpod_bins: list[str],
    selected_kpod_freq_bins: list[str],
    selected_mean_freq_bins: list[str],
    selected_ca_bins: list[str],
    selected_cl_bins: list[str],
    selected_so4_bins: list[str],
    ttf_column: str,
) -> pd.DataFrame:
    filtered = df.loc[
        (df["mount_year"] > year0)
        & (df[ttf_column].notna())
        & (df[ttf_column] >= infant_days)
        & (df["signed_freq_exposure"].notna())
    ].copy()
    filtered["event_label"] = np.where(filtered["event"].eq(1), "Failure", "Censored")
    filtered["freq_bin"] = filtered["signed_freq_exposure"].apply(lambda v: assign_bin(v, boundaries, symbol="y"))
    filtered["glf_bin"] = filtered["avg_glf"].apply(lambda v: assign_bin(v, glf_boundaries, symbol="GLF") if pd.notna(v) else "<missing>")
    filtered["kpod_bin"] = filtered["avg_kpod"].apply(lambda v: assign_bin(v, kpod_boundaries, symbol="Kpod") if pd.notna(v) else "<missing>")
    filtered["kpod_freq_bin"] = filtered["avg_kpod_freq"].apply(lambda v: assign_bin(v, kpod_freq_boundaries, symbol="Kpod_freq") if pd.notna(v) else "<missing>")
    filtered["mean_freq_bin"] = filtered["avg_freq_hz"].apply(lambda v: assign_mean_freq_bin(v, mean_freq_boundaries)) if "avg_freq_hz" in filtered.columns else "<missing>"
    filtered["salt_proxy_bin"] = filtered["salt_proxy_total_kg"].apply(lambda v: assign_bin(v, salt_proxy_boundaries, symbol="Salt") if pd.notna(v) else "<missing>")
    filtered["ca_bin"] = filtered["calcium_proxy_kg"].apply(lambda v: assign_bin(v, ca_boundaries, symbol="Ca") if pd.notna(v) else "<missing>") if "calcium_proxy_kg" in filtered.columns else "<missing>"
    filtered["cl_bin"] = filtered["chloride_proxy_kg"].apply(lambda v: assign_bin(v, cl_boundaries, symbol="Cl") if pd.notna(v) else "<missing>") if "chloride_proxy_kg" in filtered.columns else "<missing>"
    filtered["so4_bin"] = filtered["sulfate_proxy_kg"].apply(lambda v: assign_bin(v, so4_boundaries, symbol="SO₄") if pd.notna(v) else "<missing>") if "sulfate_proxy_kg" in filtered.columns else "<missing>"
    filtered["h2s_proxy_bin"] = filtered["h2s_proxy_mg_l"].apply(lambda v: assign_bin(v, h2s_proxy_boundaries, symbol="H2S") if pd.notna(v) else "<missing>") if "h2s_proxy_mg_l" in filtered.columns else "<missing>"
    filtered["contractor_label"] = filtered["contractor"].astype("string").fillna("<missing>")
    filtered["failed_node_label"] = filtered["failed_node"].astype("string").fillna("<missing>")
    filtered["failure_category_label"] = filtered["failure_category"].fillna("<missing>") if "failure_category" in filtered.columns else "<missing>"
    filtered["h2s_label_f"] = filtered["h2s_label"].fillna("<missing>") if "h2s_label" in filtered.columns else "<missing>"

    if selected_contractors:
        filtered = filtered.loc[filtered["contractor_label"].isin(selected_contractors)].copy()
    if selected_failed_nodes:
        filtered = filtered.loc[filtered["failed_node_label"].isin(selected_failed_nodes)].copy()
    if selected_failure_categories:
        filtered = filtered.loc[filtered["failure_category_label"].isin(selected_failure_categories)].copy()
    if selected_h2s_classes:
        filtered = filtered.loc[filtered["h2s_label_f"].isin(selected_h2s_classes)].copy()
    if selected_glf_bins:
        filtered = filtered.loc[filtered["glf_bin"].isin(selected_glf_bins)].copy()
    if selected_kpod_bins:
        filtered = filtered.loc[filtered["kpod_bin"].isin(selected_kpod_bins)].copy()
    if selected_kpod_freq_bins:
        filtered = filtered.loc[filtered["kpod_freq_bin"].isin(selected_kpod_freq_bins)].copy()
    if selected_mean_freq_bins:
        filtered = filtered.loc[filtered["mean_freq_bin"].isin(selected_mean_freq_bins)].copy()
    if selected_ca_bins:
        filtered = filtered.loc[filtered["ca_bin"].isin(selected_ca_bins)].copy()
    if selected_cl_bins:
        filtered = filtered.loc[filtered["cl_bin"].isin(selected_cl_bins)].copy()
    if selected_so4_bins:
        filtered = filtered.loc[filtered["so4_bin"].isin(selected_so4_bins)].copy()
    return filtered


def color_group_frame(df: pd.DataFrame, color_mode: str) -> tuple[pd.DataFrame, str]:
    plotting = df.copy().reset_index(drop=True)
    if color_mode == "Mount year":
        plotting["color_group"] = plotting["mount_year"].astype(str)
    elif color_mode == "Y exposure bin":
        plotting["color_group"] = plotting["freq_bin"].astype(str)
    elif color_mode == "GLF bin":
        plotting["color_group"] = plotting["glf_bin"].astype(str)
    elif color_mode == "Kpod bin":
        plotting["color_group"] = plotting["kpod_bin"].astype(str)
    elif color_mode == "Precipitate proxy bin":
        plotting["color_group"] = plotting["salt_proxy_bin"].astype(str)
    elif color_mode == "Failure category":
        plotting["color_group"] = plotting["failure_category_label"].astype(str) if "failure_category_label" in plotting.columns else "<missing>"
    elif color_mode == "H2S class":
        plotting["color_group"] = plotting["h2s_label_f"].astype(str) if "h2s_label_f" in plotting.columns else "<missing>"
    elif color_mode == "Ca proxy bin":
        plotting["color_group"] = plotting["ca_bin"].astype(str) if "ca_bin" in plotting.columns else "<missing>"
    elif color_mode == "Cl proxy bin":
        plotting["color_group"] = plotting["cl_bin"].astype(str) if "cl_bin" in plotting.columns else "<missing>"
    elif color_mode == "SO₄ proxy bin":
        plotting["color_group"] = plotting["so4_bin"].astype(str) if "so4_bin" in plotting.columns else "<missing>"
    elif color_mode == "H2S proxy bin":
        plotting["color_group"] = plotting["h2s_proxy_bin"].astype(str) if "h2s_proxy_bin" in plotting.columns else "<missing>"
    elif color_mode == "Mean freq bin":
        plotting["color_group"] = plotting["mean_freq_bin"].astype(str) if "mean_freq_bin" in plotting.columns else "<missing>"
    else:
        plotting["color_group"] = plotting["kpod_freq_bin"].astype(str)
    return plotting, "color_group"


def ordered_group_labels(plotting: pd.DataFrame, color_column: str, color_mode: str, boundaries: list[float] | None = None, mean_freq_boundaries: list[float] | None = None, ca_boundaries: list[float] | None = None, cl_boundaries: list[float] | None = None, so4_boundaries: list[float] | None = None, h2s_proxy_boundaries: list[float] | None = None) -> list[str]:
    if color_mode == "Y exposure bin" and boundaries is not None:
        desired = build_bin_labels(boundaries, symbol="y")
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Precipitate proxy bin" and boundaries is not None:
        desired = build_bin_labels(boundaries, symbol="Salt")
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Ca proxy bin" and ca_boundaries is not None:
        desired = build_bin_labels(ca_boundaries, symbol="Ca") + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Cl proxy bin" and cl_boundaries is not None:
        desired = build_bin_labels(cl_boundaries, symbol="Cl") + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "SO₄ proxy bin" and so4_boundaries is not None:
        desired = build_bin_labels(so4_boundaries, symbol="SO₄") + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Failure category":
        desired = FAILURE_CATEGORIES + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "H2S class":
        desired = ["Кислый", "Некислый", "<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "H2S proxy bin" and h2s_proxy_boundaries is not None:
        desired = build_bin_labels(h2s_proxy_boundaries, symbol="H2S") + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    if color_mode == "Mean freq bin":
        desired = (mean_freq_bin_labels(mean_freq_boundaries) if mean_freq_boundaries is not None else []) + ["<missing>"]
        present = set(plotting[color_column].dropna().astype(str).tolist())
        return [label for label in desired if label in present]
    return sorted(plotting[color_column].dropna().astype(str).unique().tolist())


def _color_map_for_mode(labels: list[str], color_mode: str) -> dict[str, str]:
    if color_mode == "Failure category":
        return {label: _CATEGORY_PALETTE.get(label, "#B6B6B6") for label in labels}
    if color_mode == "H2S class":
        return {label: _H2S_PALETTE.get(label, "#B6B6B6") for label in labels}
    return palette_for_labels(labels)


def x_axis_config(x_mode: str, ttf_column: str, ttf_label: str, lang: str = "en") -> tuple[str, str]:
    if x_mode == "TRF":
        return "trf_total_frequency_hz_days", t(lang, "trf_label")
    if x_mode == "TLF":
        return "tlf_total_liquid_m3", t(lang, "tlf_label")
    return ttf_column, ttf_label


def scatter_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], color_mode: str, x_mode: str, ttf_column: str, ttf_label: str, upper_hz: float = 55.0, lower_hz: float = 45.0, mean_freq_boundaries: list[float] | None = None, ca_boundaries: list[float] | None = None, cl_boundaries: list[float] | None = None, so4_boundaries: list[float] | None = None, h2s_proxy_boundaries: list[float] | None = None, lang: str = "en") -> go.Figure:
    plotting, color_column = color_group_frame(df, color_mode)
    plotting["marker_symbol"] = np.where(plotting["event"].eq(1), "x", "circle")
    plotting["marker_size"] = np.where(plotting["event"].eq(1), 11, 7)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label, lang)
    fig = go.Figure()
    groups = ordered_group_labels(plotting, color_column, color_mode, boundaries, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries)
    color_map = _color_map_for_mode(groups, color_mode)
    for group in groups:
        sub = plotting.loc[plotting[color_column].astype(str) == str(group)].copy()
        fig.add_trace(
            go.Scattergl(
                x=sub[x_column],
                y=sub["signed_freq_exposure"],
                mode="markers",
                name=str(group),
                marker={
                    "color": color_map[str(group)],
                    "size": sub["marker_size"],
                    "symbol": sub["marker_symbol"],
                    "line": {"width": 0.6, "color": "black"},
                    "opacity": 0.8,
                },
                customdata=sub[
                    [
                        "well_id",
                        "contractor",
                        "mount_date",
                        "stop_date",
                        "event_label",
                        "n_freq_days",
                        "n_freq_days_above_55",
                        "n_freq_days_below_45",
                        "frac_freq_above_55",
                        "frac_freq_below_45",
                        "salt_proxy_total_kg",
                    ]
                ].astype(str).to_numpy(),
                hovertemplate=(
                    "Well: %{customdata[0]}<br>"
                    "Contractor: %{customdata[1]}<br>"
                    "Mount: %{customdata[2]}<br>"
                    "Stop: %{customdata[3]}<br>"
                    "Outcome: %{customdata[4]}<br>"
                    f"{x_mode}: " + "%{x:.1f}<br>"
                    "Signed exposure y: %{y:.3f}<br>"
                    f"Share >{upper_hz:.0f} Hz: %{{customdata[8]}}<br>"
                    f"Share <{lower_hz:.0f} Hz: %{{customdata[9]}}<br>"
                    "Freq days: %{customdata[5]}<br>"
                    f"Days >{upper_hz:.0f} Hz: %{{customdata[6]}}<br>"
                    f"Days <{lower_hz:.0f} Hz: %{{customdata[7]}}<br>"
                    "Salt proxy total: %{customdata[10]} kg<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=t(lang, "scatter_title", field=field_label),
        xaxis_title=x_label,
        yaxis_title=t(lang, "scatter_yaxis", upper=upper_hz, lower=lower_hz),
        height=520,
        legend_title=color_mode,
    )
    fig.update_yaxes(range=[-1.03, 1.03], zeroline=True, zerolinewidth=1.2)
    for boundary in boundaries:
        fig.add_hline(y=float(boundary), line_dash="dot", line_color="#999")
    return fig


def histogram_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], group_mode: str, x_mode: str, ttf_column: str, ttf_label: str, mean_freq_boundaries: list[float] | None = None, ca_boundaries: list[float] | None = None, cl_boundaries: list[float] | None = None, so4_boundaries: list[float] | None = None, h2s_proxy_boundaries: list[float] | None = None, lang: str = "en") -> go.Figure:
    fig = go.Figure()
    failures = df.loc[df["event"].eq(1)].copy()
    if failures.empty:
        return fig
    failures, color_column = color_group_frame(failures, group_mode)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label, lang)
    max_value = float(failures[x_column].max())
    if x_mode in {"TLF", "TRF"}:
        step = max(1000.0, round(max_value / 30.0, -2))
        bins = np.arange(0.0, max(step * 2.0, max_value + step), step)
    else:
        bins = np.arange(0.0, max(210.0, min(max_value + 30.0, 1800.0)), 30.0)
    labels = ordered_group_labels(failures, color_column, group_mode, boundaries, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries)
    color_map = _color_map_for_mode(labels, group_mode)
    for label in labels:
        sub = failures.loc[failures[color_column].astype(str) == str(label), x_column]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=f"{label} (n={len(sub)})",
                marker_color=color_map[str(label)],
                opacity=0.65,
                xbins={"start": float(bins.min()), "end": float(bins.max()), "size": float(bins[1] - bins[0]) if len(bins) > 1 else 1.0},
                hovertemplate=f"{x_mode} bin: " + "%{x}<br>Failures: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=t(lang, "hist_title", field=field_label, group=group_mode),
        xaxis_title=x_label,
        yaxis_title="Failure count" if lang == "en" else "Количество отказов",
        height=460,
        legend_title=group_mode,
    )
    return fig


def survival_figure(df: pd.DataFrame, field_label: str, boundaries: list[float], group_mode: str, x_mode: str, ttf_column: str, ttf_label: str, mean_freq_boundaries: list[float] | None = None, ca_boundaries: list[float] | None = None, cl_boundaries: list[float] | None = None, so4_boundaries: list[float] | None = None, h2s_proxy_boundaries: list[float] | None = None, lang: str = "en") -> go.Figure:
    fig = go.Figure()
    plotting, color_column = color_group_frame(df, group_mode)
    labels = ordered_group_labels(plotting, color_column, group_mode, boundaries, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries)
    color_map = _color_map_for_mode(labels, group_mode)
    x_column, x_label = x_axis_config(x_mode, ttf_column, ttf_label, lang)
    for label in labels:
        sub = plotting.loc[plotting[color_column].astype(str) == str(label)].copy()
        if len(sub) < 3:
            continue
        dur = sub[x_column].to_numpy(dtype=float)
        e = sub["event"].to_numpy(dtype=float)
        km_t, km_s = kaplan_meier(dur, e)
        fig.add_trace(
            go.Scatter(
                x=km_t,
                y=km_s,
                mode="lines",
                line={"shape": "hv", "width": 2.5, "color": color_map[str(label)]},
                name=f"{label} (n={len(sub)}, events={int(e.sum())})",
                hovertemplate="Time: %{x:.0f} d<br>Survival: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=t(lang, "survival_title", field=field_label, group=group_mode),
        xaxis_title=x_label,
        yaxis_title="Survival probability" if lang == "en" else "Вероятность выживания",
        height=460,
        legend_title=group_mode,
    )
    fig.update_yaxes(range=[0.0, 1.02])
    return fig


def summary_table(df: pd.DataFrame, group_mode: str, ttf_column: str, ttf_label: str) -> pd.DataFrame:
    plotting, color_column = color_group_frame(df, group_mode)
    rows: list[dict[str, object]] = []
    for label in sorted(plotting[color_column].dropna().astype(str).unique().tolist()):
        sub = plotting.loc[plotting[color_column].astype(str) == str(label)].copy()
        failures = sub.loc[sub["event"].eq(1)]
        rows.append(
            {
                group_mode: label,
                "Runs": int(len(sub)),
                "Failures": int(len(failures)),
                "Censored": int(sub["event"].eq(0).sum()),
                f"Median {ttf_label} all": None if sub.empty else round(float(sub[ttf_column].median()), 1),
                f"Median {ttf_label} failures": None if failures.empty else round(float(failures[ttf_column].median()), 1),
                "Mean y": None if sub.empty else round(float(sub["signed_freq_exposure"].mean()), 3),
                "Mean share >55 Hz": None if sub.empty else round(float(sub["frac_freq_above_55"].mean()), 3),
                "Mean share <45 Hz": None if sub.empty else round(float(sub["frac_freq_below_45"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def pair_correlation_figure(
    df: pd.DataFrame,
    field_label: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    color_mode: str,
    swap_axes: bool,
    log_scale: bool = False,
    lang: str = "en",
) -> tuple[go.Figure, float | None, int]:
    subset = df.loc[df[x_column].notna() & df[y_column].notna()].copy()
    if subset.empty:
        return go.Figure(), None, 0
    corr = float(subset[x_column].corr(subset[y_column])) if len(subset) >= 2 else None
    plotting, color_column = color_group_frame(subset, color_mode)
    actual_x = y_column if swap_axes else x_column
    actual_y = x_column if swap_axes else y_column
    actual_x_label = y_label if swap_axes else x_label
    actual_y_label = x_label if swap_axes else y_label
    fig = px.scatter(
        plotting,
        x=actual_x,
        y=actual_y,
        color=color_column,
        symbol="event_label",
        symbol_map={"Failure": "circle", "Censored": "x"},
        hover_data=["well_id", "contractor"],
        title=f"{field_label}: {title}",
        labels={
            actual_x: actual_x_label,
            actual_y: actual_y_label,
            color_column: color_mode,
            "event_label": "Outcome",
        },
        height=420,
    )

    if x_column == "trf_total_frequency_hz_days" and y_column in {"ttf_days", "ttf_true_best_days"}:
        trf_vals_pos = subset["trf_total_frequency_hz_days"].loc[subset["trf_total_frequency_hz_days"] > 0]
        ttf_vals_pos = subset[y_column].loc[subset[y_column] > 0]
        trf_max = float(subset["trf_total_frequency_hz_days"].max())
        ttf_max = float(subset[y_column].max())

        if log_scale and not trf_vals_pos.empty and not ttf_vals_pos.empty:
            trf_min = float(trf_vals_pos.min())
            ttf_min = float(ttf_vals_pos.min())
            ttf_values = np.logspace(np.log10(ttf_min * 0.8), np.log10(ttf_max * 1.25), 120)
            x_ext = trf_max * 1.25
            y_ext = max(ttf_max * 1.25, x_ext / 30.0)

            def _poly(trf_pts: list[float], ttf_pts: list[float]) -> tuple[np.ndarray, np.ndarray]:
                t = np.array(trf_pts)
                d = np.array(ttf_pts)
                return (d, t) if swap_axes else (t, d)

            # In log scale, clip polygons to data range (no zero vertices)
            lo_trf = trf_min * 0.5
            lo_ttf = ttf_min * 0.5

            # Fill: freq > 70 Hz (TRF > 70*TTF, below 70Hz line)
            fx, fy = _poly(
                [lo_trf, x_ext,       x_ext,       lo_trf * 70, lo_trf],
                [lo_ttf, lo_ttf,      x_ext / 70.0, lo_ttf,      lo_ttf],
            )
            fig.add_trace(go.Scatter(
                x=fx, y=fy, fill="toself",
                fillcolor="rgba(210, 30, 30, 0.09)",
                line={"width": 0}, mode="lines",
                showlegend=False, hoverinfo="skip",
            ))

            # Fill: freq < 30 Hz (TRF < 30*TTF, above 30Hz line)
            fx, fy = _poly(
                [lo_trf, lo_trf, x_ext,  x_ext,        lo_trf],
                [x_ext / 30.0, y_ext, y_ext, x_ext / 30.0, x_ext / 30.0],
            )
            fig.add_trace(go.Scatter(
                x=fx, y=fy, fill="toself",
                fillcolor="rgba(210, 30, 30, 0.09)",
                line={"width": 0}, mode="lines",
                showlegend=False, hoverinfo="skip",
            ))
        else:
            ttf_values = np.linspace(0.0, max(ttf_max, 1.0), 120)
            x_ext = trf_max * 1.25
            y_ext = max(ttf_max * 1.25, x_ext / 30.0)

            def _poly(trf_pts: list[float], ttf_pts: list[float]) -> tuple[np.ndarray, np.ndarray]:
                t = np.array(trf_pts)
                d = np.array(ttf_pts)
                return (d, t) if swap_axes else (t, d)

            # -- Red fill: freq > 70 Hz --
            fx, fy = _poly(
                [0.0, x_ext, x_ext, 0.0],
                [0.0, 0.0,   x_ext / 70.0, 0.0],
            )
            fig.add_trace(go.Scatter(
                x=fx, y=fy, fill="toself",
                fillcolor="rgba(210, 30, 30, 0.09)",
                line={"width": 0}, mode="lines",
                showlegend=False, hoverinfo="skip",
            ))

            # -- Red fill: freq < 30 Hz --
            fx, fy = _poly(
                [0.0, 0.0,   x_ext, x_ext,        0.0],
                [0.0, y_ext, y_ext, x_ext / 30.0, 0.0],
            )
            fig.add_trace(go.Scatter(
                x=fx, y=fy, fill="toself",
                fillcolor="rgba(210, 30, 30, 0.09)",
                line={"width": 0}, mode="lines",
                showlegend=False, hoverinfo="skip",
            ))

        # -- Reference lines (work in both linear and log scale) --
        reference_specs = [
            (30.0, "#d62728", "solid", 2.5),
            (40.0, "#e67e00", "dash",  2.0),
            (50.0, "#000000", "dash",  2.0),
            (60.0, "#2ca02c", "dash",  2.0),
            (70.0, "#d62728", "solid", 2.5),
        ]
        for frequency_hz, color, dash, width in reference_specs:
            trf_line = frequency_hz * ttf_values
            visible_mask = trf_line <= (trf_max * 1.05)
            if not np.any(visible_mask):
                visible_mask = np.ones_like(trf_line, dtype=bool)
            x_values = ttf_values[visible_mask] if swap_axes else trf_line[visible_mask]
            y_values = trf_line[visible_mask] if swap_axes else ttf_values[visible_mask]
            label = t(lang, "hz_limit" if frequency_hz in (30.0, 70.0) else "hz_reference", hz=frequency_hz)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": width, "dash": dash},
                    hovertemplate=(
                        f"{label}<br>"
                        + f"{actual_x_label}: %{{x:.1f}}<br>"
                        + f"{actual_y_label}: %{{y:.1f}}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    # Force equal visual scale so the 50 Hz diagonal appears at 45°.
    # When x=TRF, y=TTF: 1 TTF-day needs 50× more pixels than 1 TRF-Hz-day → scaleratio=50.
    # When swapped (x=TTF, y=TRF): inverse → scaleratio=1/50.
    if x_column == "trf_total_frequency_hz_days" and y_column in {"ttf_days", "ttf_true_best_days"}:
        scale_ratio = 1.0 / 50.0 if swap_axes else 50.0
        fig.update_yaxes(scaleanchor="x", scaleratio=scale_ratio, constrain="domain")
        trf_max_display = 40_000.0
        if swap_axes:
            fig.update_yaxes(range=[0, trf_max_display])
        else:
            fig.update_xaxes(range=[0, trf_max_display])

    return fig, corr, int(len(subset))


def ttf_mode_config(ttf_mode: str, lang: str = "en") -> tuple[str, str]:
    if ttf_mode == "TTF_true_best":
        lbl = "ВРО_опт (операционные дни)" if lang == "ru" else "TTF_true_best (operating days)"
        return "ttf_true_best_days", lbl
    lbl = "ВРО (дни)" if lang == "ru" else "TTF (days)"
    return "ttf_days", lbl


def _infant_category_pie(infant_df: pd.DataFrame, title: str) -> go.Figure:
    cat_counts = (
        infant_df["failure_category_label"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )
    color_map = [_CATEGORY_PALETTE.get(c, "#B6B6B6") for c in cat_counts["Category"]]
    fig = go.Figure(
        go.Pie(
            labels=cat_counts["Category"],
            values=cat_counts["Count"],
            marker_colors=color_map,
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="Category: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=420, showlegend=True)
    return fig


def _infant_node_pie(infant_df: pd.DataFrame) -> go.Figure:
    node_counts = (
        infant_df["failed_node"].fillna("<missing>")
        .value_counts()
        .rename_axis("Failed node")
        .reset_index(name="Count")
    )
    fig = go.Figure(
        go.Pie(
            labels=node_counts["Failed node"],
            values=node_counts["Count"],
            marker_colors=px.colors.qualitative.Pastel,
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="Node: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(title="Infant failures by failed node", height=420, showlegend=True)
    return fig


def main() -> None:
    # Language selector must come before any t() call so state is set on first render.
    with st.sidebar:
        lang_choice = st.selectbox("Language / Язык", ["English", "Русский"], index=0, key="lang_display")
    lang = "ru" if lang_choice == "Русский" else "en"

    st.title(t(lang, "main_title"))
    st.caption(t(lang, "main_caption"))
    st.info(t(lang, "info_text"))

    conn = _warehouse_conn()
    if conn is not None:
        last = get_last_pipeline_run("mart__vt_freq55", conn=conn)
        if last:
            st.sidebar.caption(f"Data last built: {last['run_at']} ({last['row_count']:,} runs)")
        else:
            st.sidebar.warning("Warehouse exists but mart__vt_freq55 not built yet. Run `python scripts/pipeline.py`.")
    else:
        st.sidebar.error("Warehouse not found. Run `python scripts/pipeline.py --step ingest` to start.")

    fields: list[str] = []
    year_min, year_max = 2010, 2025
    if conn is not None:
        try:
            meta = pd.read_sql(
                "SELECT DISTINCT field, strftime('%Y', install_date) AS yr FROM raw__v03_runs WHERE field IS NOT NULL",
                conn,
            )
            fields = sorted(meta["field"].dropna().astype(str).str.strip().unique().tolist())
            valid_years = pd.to_numeric(meta["yr"], errors="coerce").dropna()
            if not valid_years.empty:
                year_min = int(valid_years.min())
                year_max = int(valid_years.max())
        except Exception:
            pass

    with st.sidebar:
        st.subheader(t(lang, "filters_header"))
        field_choice = st.selectbox(t(lang, "field_label"), ["GLOBAL", *fields], index=0)
        infant_days = st.slider(t(lang, "infant_label"), min_value=0, max_value=180, value=30, step=5)
        year0 = st.slider(
            t(lang, "year0_label"),
            min_value=year_min,
            max_value=max(year_min, year_max - 1),
            value=max(2023, year_min),
            step=1,
        )
        st.subheader(t(lang, "freq_bins_header"))
        upper_hz = st.number_input(t(lang, "upper_hz_label"), value=55.0, step=1.0, format="%.1f")
        lower_hz = st.number_input(t(lang, "lower_hz_label"), value=45.0, step=1.0, format="%.1f")
        boundaries_text = st.text_input(
            t(lang, "boundary_label"),
            value=DEFAULT_BOUNDARIES_TEXT,
            help=t(lang, "boundary_help"),
        )
        st.subheader(t(lang, "glf_header"))
        glf_boundaries_text = st.text_input(t(lang, "glf_boundaries_label"), value=DEFAULT_GLF_BOUNDARIES_TEXT)
        st.subheader(t(lang, "kpod_header"))
        kpod_boundaries_text = st.text_input(t(lang, "kpod_boundaries_label"), value=DEFAULT_KPOD_BOUNDARIES_TEXT)
        st.subheader(t(lang, "kpod_freq_header"))
        kpod_freq_boundaries_text = st.text_input(t(lang, "kpod_freq_boundaries_label"), value=DEFAULT_KPOD_FREQ_BOUNDARIES_TEXT)
        mean_freq_boundaries_text = st.text_input(
            t(lang, "mean_freq_boundaries_label"),
            value=DEFAULT_MEAN_FREQ_BOUNDARIES_TEXT,
            help=t(lang, "mean_freq_boundaries_help"),
        )
        st.subheader(t(lang, "precipitate_header"))
        salt_proxy_boundaries_text = st.text_input(
            t(lang, "precipitate_boundaries_label"), value=DEFAULT_PRECIPITATE_PROXY_BOUNDARIES_TEXT
        )
        st.subheader(t(lang, "ions_header"))
        ca_boundaries_text = st.text_input(t(lang, "ca_boundaries_label"), value=DEFAULT_CA_BOUNDARIES_TEXT)
        cl_boundaries_text = st.text_input(t(lang, "cl_boundaries_label"), value=DEFAULT_CL_BOUNDARIES_TEXT)
        so4_boundaries_text = st.text_input(t(lang, "so4_boundaries_label"), value=DEFAULT_SO4_BOUNDARIES_TEXT)
        st.subheader(t(lang, "h2s_header"))
        h2s_method = st.radio(
            t(lang, "h2s_method_label"),
            ["Кислый/Некислый (Excel)", "H2S proxy (threshold)"],
            index=0,
        )
        h2s_threshold = st.number_input(
            t(lang, "h2s_threshold_label"),
            value=DEFAULT_H2S_THRESHOLD_MG_L,
            step=0.5,
            format="%.1f",
            help=t(lang, "h2s_threshold_help"),
        )
        h2s_proxy_boundaries_text = st.text_input(
            t(lang, "h2s_proxy_boundaries_label"),
            value=DEFAULT_H2S_PROXY_BOUNDARIES_TEXT,
            help=t(lang, "h2s_proxy_help"),
        )
        ttf_mode = st.selectbox(
            t(lang, "ttf_def_label"),
            ["TTF", "TTF_true_best"],
            index=0,
            help=t(lang, "ttf_def_help"),
        )
        x_mode = st.selectbox(
            t(lang, "xmode_label"),
            ["TTF", "TLF", "TRF"],
            index=0,
            help=t(lang, "xmode_help"),
        )
        color_mode = st.selectbox(
            t(lang, "colormode_label"),
            COLOR_GROUP_OPTIONS,
            index=0,
        )
        group_mode = st.selectbox(
            t(lang, "groupmode_label"),
            COLOR_GROUP_OPTIONS,
            index=0,
        )
        swap_pair_axes = st.checkbox(t(lang, "swap_axes_label"), value=True)
        log_scale_pairs = st.checkbox(t(lang, "logscale_label"), value=False)

    try:
        boundaries = parse_boundaries(boundaries_text)
        glf_boundaries = parse_boundaries(glf_boundaries_text)
        kpod_boundaries = parse_boundaries(kpod_boundaries_text)
        kpod_freq_boundaries = parse_boundaries(kpod_freq_boundaries_text)
        mean_freq_boundaries = parse_boundaries(mean_freq_boundaries_text)
        salt_proxy_boundaries = parse_boundaries(salt_proxy_boundaries_text)
        ca_boundaries = parse_boundaries(ca_boundaries_text)
        cl_boundaries = parse_boundaries(cl_boundaries_text)
        so4_boundaries = parse_boundaries(so4_boundaries_text)
        h2s_proxy_boundaries = parse_boundaries(h2s_proxy_boundaries_text)
    except Exception as exc:
        st.error(f"Could not parse boundary list: {exc}")
        return

    with st.spinner("Loading frequency-exposure dataset..."):
        stats = load_run_stats(field_choice)

    # Merge failure categories from raw failures workbook
    cats = load_failure_categories()
    if not cats.empty and not stats.empty and "well_key" in stats.columns:
        stats = stats.merge(cats, on=["well_key", "mount_date", "stop_date"], how="left")
    if "failure_category" not in stats.columns:
        stats["failure_category"] = None

    # Compute H2S label from chosen method
    if h2s_method == "H2S proxy (threshold)":
        if "h2s_proxy_mg_l" in stats.columns:
            stats["h2s_label"] = np.where(
                stats["h2s_proxy_mg_l"].isna(), "<missing>",
                np.where(stats["h2s_proxy_mg_l"] >= h2s_threshold, "Кислый", "Некислый"),
            )
        else:
            stats["h2s_label"] = "<missing>"
    else:
        stats["h2s_label"] = stats["h2s_class_excel"].fillna("<missing>") if "h2s_class_excel" in stats.columns else "<missing>"

    ttf_column, ttf_label = ttf_mode_config(ttf_mode, lang)

    contractor_labels = sorted(stats["contractor"].astype("string").fillna("<missing>").unique().tolist())
    failed_node_labels = sorted(stats["failed_node"].astype("string").fillna("<missing>").unique().tolist())
    glf_labels = build_bin_labels(glf_boundaries, symbol="GLF")
    kpod_labels = build_bin_labels(kpod_boundaries, symbol="Kpod")
    kpod_freq_labels = build_bin_labels(kpod_freq_boundaries, symbol="Kpod_freq")
    mean_freq_labels = mean_freq_bin_labels(mean_freq_boundaries)
    category_options = FAILURE_CATEGORIES + ["<missing>"]

    h2s_options = ["Кислый", "Некислый", "<missing>"]
    ca_labels = build_bin_labels(ca_boundaries, symbol="Ca") + ["<missing>"]
    cl_labels = build_bin_labels(cl_boundaries, symbol="Cl") + ["<missing>"]
    so4_labels = build_bin_labels(so4_boundaries, symbol="SO₄") + ["<missing>"]

    with st.sidebar:
        selected_contractors = st.multiselect(t(lang, "contractor_label"), contractor_labels, default=contractor_labels)
        selected_failed_nodes = st.multiselect(t(lang, "failed_nodes_label"), failed_node_labels, default=failed_node_labels)
        selected_failure_categories = st.multiselect(
            t(lang, "failure_cats_label"), category_options, default=category_options
        )
        selected_h2s_classes = st.multiselect(t(lang, "h2s_class_label"), h2s_options, default=h2s_options)
        glf_options = glf_labels + ["<missing>"]
        selected_glf_bins = st.multiselect(t(lang, "glf_bins_label"), glf_options, default=glf_options)
        kpod_options = kpod_labels + ["<missing>"]
        selected_kpod_bins = st.multiselect(t(lang, "kpod_bins_label"), kpod_options, default=kpod_options)
        kpod_freq_options = kpod_freq_labels + ["<missing>"]
        selected_kpod_freq_bins = st.multiselect(t(lang, "kpod_freq_bins_label"), kpod_freq_options, default=kpod_freq_options)
        mean_freq_options = mean_freq_labels + ["<missing>"]
        selected_mean_freq_bins = st.multiselect(t(lang, "mean_freq_bins_label"), mean_freq_options, default=mean_freq_options)
        selected_ca_bins = st.multiselect(t(lang, "ca_bins_label"), ca_labels, default=ca_labels)
        selected_cl_bins = st.multiselect(t(lang, "cl_bins_label"), cl_labels, default=cl_labels)
        selected_so4_bins = st.multiselect(t(lang, "so4_bins_label"), so4_labels, default=so4_labels)

    view = filtered_view(
        stats,
        infant_days=infant_days,
        year0=year0,
        boundaries=boundaries,
        glf_boundaries=glf_boundaries,
        kpod_boundaries=kpod_boundaries,
        kpod_freq_boundaries=kpod_freq_boundaries,
        mean_freq_boundaries=mean_freq_boundaries,
        salt_proxy_boundaries=salt_proxy_boundaries,
        ca_boundaries=ca_boundaries,
        cl_boundaries=cl_boundaries,
        so4_boundaries=so4_boundaries,
        h2s_proxy_boundaries=h2s_proxy_boundaries,
        selected_contractors=selected_contractors,
        selected_failed_nodes=selected_failed_nodes,
        selected_failure_categories=selected_failure_categories,
        selected_h2s_classes=selected_h2s_classes,
        selected_glf_bins=selected_glf_bins,
        selected_kpod_bins=selected_kpod_bins,
        selected_kpod_freq_bins=selected_kpod_freq_bins,
        selected_mean_freq_bins=selected_mean_freq_bins,
        selected_ca_bins=selected_ca_bins,
        selected_cl_bins=selected_cl_bins,
        selected_so4_bins=selected_so4_bins,
        ttf_column=ttf_column,
    )

    # Infant mortality view: true failures with TTF below threshold
    infant_base = stats.loc[
        (stats["mount_year"] > year0)
        & stats[ttf_column].notna()
        & (stats[ttf_column] < infant_days)
        & stats["event"].eq(1)
    ].copy()
    infant_base["failure_category_label"] = infant_base["failure_category"].fillna("<missing>")
    if selected_failure_categories:
        infant_base = infant_base.loc[
            infant_base["failure_category_label"].isin(selected_failure_categories)
        ].copy()

    field_label = ("Global" if lang == "en" else "Глобально") if field_choice == "GLOBAL" else f"Field={field_choice}"

    # Top-level metrics shared across tabs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t(lang, "runs_metric"), len(view))
    k2.metric(t(lang, "failures_metric"), int(view["event"].eq(1).sum()))
    k3.metric(t(lang, "infant_metric", n=infant_days), len(infant_base))
    if x_mode == "TRF":
        median_val = "n/a" if view.empty else f"{float(view['trf_total_frequency_hz_days'].median()):.0f} Hz-d"
        k4.metric(t(lang, "median_trf"), median_val)
    elif x_mode == "TLF":
        median_val = "n/a" if view.empty else f"{float(view['tlf_total_liquid_m3'].median()):.0f} m³"
        k4.metric(t(lang, "median_tlf"), median_val)
    else:
        k4.metric(f"Median {ttf_mode}", "n/a" if view.empty else f"{float(view[ttf_column].median()):.0f} d")

    tab_freq, tab_infant = st.tabs([t(lang, "tab_freq"), t(lang, "tab_infant")])

    # ── Tab 1: Frequency Exposure ─────────────────────────────────────────────
    with tab_freq:
        if view.empty:
            st.warning("No rows remain after the selected filters." if lang == "en" else "Нет строк после применения фильтров.")
        else:
            st.plotly_chart(
                scatter_figure(view, field_label, boundaries, color_mode, x_mode, ttf_column, ttf_label, upper_hz=upper_hz, lower_hz=lower_hz, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries, lang=lang),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(histogram_figure(view, field_label, boundaries, group_mode, x_mode, ttf_column, ttf_label, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries, lang=lang), use_container_width=True)
            with col2:
                st.plotly_chart(survival_figure(view, field_label, boundaries, group_mode, x_mode, ttf_column, ttf_label, mean_freq_boundaries=mean_freq_boundaries, ca_boundaries=ca_boundaries, cl_boundaries=cl_boundaries, so4_boundaries=so4_boundaries, h2s_proxy_boundaries=h2s_proxy_boundaries, lang=lang), use_container_width=True)

            st.subheader(t(lang, "pair_corr_header"))
            tlf_x_label = t(lang, "tlf_label")
            trf_x_label = t(lang, "trf_label")
            tlf_fig, tlf_corr, tlf_rows = pair_correlation_figure(
                view, field_label,
                x_column="tlf_total_liquid_m3", y_column=ttf_column,
                x_label=tlf_x_label, y_label=ttf_label,
                title=f"TLF vs {ttf_mode}", color_mode=color_mode, swap_axes=swap_pair_axes,
                lang=lang,
            )
            trf_fig, trf_corr, trf_rows = pair_correlation_figure(
                view, field_label,
                x_column="trf_total_frequency_hz_days", y_column=ttf_column,
                x_label=trf_x_label, y_label=ttf_label,
                title=f"TRF vs {ttf_mode}", color_mode=color_mode, swap_axes=swap_pair_axes,
                log_scale=log_scale_pairs, lang=lang,
            )
            proxy_label = "Precipitate proxy total (kg)" if lang == "en" else "Прокси осадков итого (кг)"
            proxy_ttf_fig, proxy_ttf_corr, proxy_ttf_rows = pair_correlation_figure(
                view, field_label,
                x_column="salt_proxy_total_kg", y_column=ttf_column,
                x_label=proxy_label, y_label=ttf_label,
                title=f"{ttf_mode} vs precipitate proxy", color_mode=color_mode, swap_axes=swap_pair_axes,
                lang=lang,
            )
            proxy_trf_fig, proxy_trf_corr, proxy_trf_rows = pair_correlation_figure(
                view, field_label,
                x_column="salt_proxy_total_kg", y_column="trf_total_frequency_hz_days",
                x_label=proxy_label, y_label=trf_x_label,
                title="TRF vs precipitate proxy", color_mode=color_mode, swap_axes=swap_pair_axes,
                lang=lang,
            )
            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(tlf_fig, use_container_width=True)
                st.caption(f"TLF vs {ttf_mode}: rows={tlf_rows}, Pearson r={'n/a' if tlf_corr is None or np.isnan(tlf_corr) else f'{tlf_corr:.3f}'}")
            with p2:
                st.plotly_chart(trf_fig, use_container_width=True)
                st.caption(f"TRF vs {ttf_mode}: rows={trf_rows}, Pearson r={'n/a' if trf_corr is None or np.isnan(trf_corr) else f'{trf_corr:.3f}'}")
            p3, p4 = st.columns(2)
            with p3:
                st.plotly_chart(proxy_ttf_fig, use_container_width=True)
                st.caption(
                    f"{ttf_mode} vs precipitate proxy: "
                    + f"rows={proxy_ttf_rows}, Pearson r={'n/a' if proxy_ttf_corr is None or np.isnan(proxy_ttf_corr) else f'{proxy_ttf_corr:.3f}'}"
                )
            with p4:
                st.plotly_chart(proxy_trf_fig, use_container_width=True)
                st.caption(
                    "TRF vs precipitate proxy: "
                    + f"rows={proxy_trf_rows}, Pearson r={'n/a' if proxy_trf_corr is None or np.isnan(proxy_trf_corr) else f'{proxy_trf_corr:.3f}'}"
                )

            st.subheader(t(lang, "grouped_summary_header"))
            st.dataframe(summary_table(view, group_mode, ttf_column, ttf_label), use_container_width=True, height=220)

            csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                t(lang, "download_btn"),
                data=csv_bytes,
                file_name=f"freq55_{field_choice.lower()}_year_gt_{year0}_infant_{infant_days}d.csv",
                mime="text/csv",
            )

    # ── Tab 2: Infant Mortality ───────────────────────────────────────────────
    with tab_infant:
        st.subheader(t(lang, "infant_tab_header", field=field_label))
        st.caption(t(lang, "infant_tab_caption", n=infant_days, year=year0))

        if infant_base.empty:
            no_inf = f"No infant mortality failures found for the current filters (threshold = {infant_days} days)." if lang == "en" else f"Инфантильных отказов не найдено (порог = {infant_days} дней)."
            st.info(no_inf)
        else:
            # Summary metrics for the infant tab
            im1, im2, im3 = st.columns(3)
            all_fails = stats.loc[stats["event"].eq(1) & (stats["mount_year"] > year0)]
            im1.metric(t(lang, "infant_failures_metric"), len(infant_base))
            im2.metric(t(lang, "all_failures_metric"), len(all_fails))
            pct = 100.0 * len(infant_base) / len(all_fails) if len(all_fails) > 0 else 0.0
            im3.metric(t(lang, "infant_share_metric"), f"{pct:.1f}%")

            # Pie charts
            pc1, pc2 = st.columns(2)
            pie_title = f"Failure categories ({len(infant_base)} infant failures)" if lang == "en" else f"Категории отказов ({len(infant_base)} инфантильных)"
            node_pie_title = "Infant failures by failed node" if lang == "en" else "Инфантильные отказы по узлу"
            with pc1:
                st.plotly_chart(
                    _infant_category_pie(infant_base, title=pie_title),
                    use_container_width=True,
                )
            with pc2:
                fig_node = _infant_node_pie(infant_base)
                fig_node.update_layout(title=node_pie_title)
                st.plotly_chart(fig_node, use_container_width=True)

            # Category breakdown table
            st.subheader(t(lang, "cat_breakdown_header"))
            cat_col = "Failure category" if lang == "en" else "Категория отказа"
            cat_tbl = (
                infant_base["failure_category_label"]
                .value_counts()
                .rename_axis(cat_col)
                .reset_index(name="Count" if lang == "en" else "Кол-во")
            )
            cat_tbl["Share %" if lang == "en" else "Доля %"] = (cat_tbl.iloc[:, 1] / cat_tbl.iloc[:, 1].sum() * 100).round(1)
            st.dataframe(cat_tbl, use_container_width=True, height=280)

            # TTF distribution for infant failures
            st.subheader(t(lang, "ttf_dist_header"))
            ttf_hist = go.Figure()
            bins_inf = np.arange(0.0, infant_days + 5.0, max(1.0, infant_days / 20.0))
            for cat in FAILURE_CATEGORIES + ["<missing>"]:
                sub = infant_base.loc[infant_base["failure_category_label"] == cat, ttf_column]
                if sub.empty:
                    continue
                ttf_hist.add_trace(
                    go.Histogram(
                        x=sub,
                        name=cat,
                        marker_color=_CATEGORY_PALETTE.get(cat, "#B6B6B6"),
                        opacity=0.7,
                        xbins={"start": 0.0, "end": float(bins_inf.max()), "size": float(bins_inf[1] - bins_inf[0]) if len(bins_inf) > 1 else 1.0},
                        hovertemplate="TTF: %{x:.0f} d<br>Count: %{y}<extra></extra>",
                    )
                )
            ttf_hist.update_layout(
                barmode="overlay",
                xaxis_title=ttf_label,
                yaxis_title="Count" if lang == "en" else "Кол-во",
                height=380,
                legend_title="Failure category" if lang == "en" else "Категория отказа",
            )
            st.plotly_chart(ttf_hist, use_container_width=True)


if __name__ == "__main__":
    main()
