from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor, Pool


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_ASSETS_DIR = REPO_ROOT / "frontend" / "src" / "assets"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns


st.set_page_config(
    page_title="CatBoost Analysis",
    page_icon=":material/insights:",
    layout="wide",
)


NUMERIC_PARSE_THRESHOLD = 0.8
DATETIME_PARSE_THRESHOLD = 0.8
MIN_NON_NULL_FOR_ANALYSIS = 5
DATETIME_NAME_TOKENS = (
    "date",
    "datetime",
    "time",
    "timestamp",
    "year",
    "month",
    "day",
    "дата",
    "время",
    "год",
    "месяц",
    "день",
)

DEFAULT_EXCLUDED_FEATURE_PATTERNS = (
    "причина остановки - группа наработок",
    "failure flag",
    "группа наработок",
    "дата монтажа",
    "дата остановки",
    "причина остановки",
    "дата демонтажа",
    "признак отказа",
    "отказавший узел",
    "отказавший элемент",
    "характер неисправности",
    "причина отказа уэцн",
)


@dataclass
class Overview:
    total_rows: int
    total_columns: int
    valid_target_rows: int
    missing_values_share: float
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int
    target_mean: float | None
    target_median: float | None
    target_min: float | None
    target_max: float | None


@dataclass
class TrainedForecast:
    model: CatBoostRegressor
    prepared: pd.DataFrame
    actual: pd.Series
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    metrics: dict[str, float | int | None]
    baseline_values: dict[str, str | float]
    categorical_columns: list[str]


CATBOOST_PRIMARY_VIEWS = [
    "Факт ЭПУ",
    "Факторный анализ CatBoost",
    "Корреляции",
    "Номограмма",
]

CATBOOST_DERIVED_DISPLAY_MAP: dict[str, tuple[str, str]] = {
    "Kpod": (
        "Kпод",
        "Коэффициент подачи: дебит жидкости / номинальная производительность насоса.",
    ),
    "Kpod_freq_adjusted": (
        "Kпод_freq",
        "Кпод с поправкой на фактическую и опорную частоту.",
    ),
    "pressure_ratio": (
        "Pзаб / Дав. Нас",
        "Отношение забойного давления к давлению насыщения.",
    ),
    "frequency_to_reference_ratio": (
        "Частота / Номинальная частота",
        "Отношение фактической частоты к номинальной частоте.",
    ),
    "frequency_over_reference_hz": (
        "Частота - Номинальная частота",
        "Превышение фактической частоты над номинальной в Гц.",
    ),
    "qliq_per_hz": (
        "Qж / Частота",
        "Удельная подача на единицу частоты.",
    ),
    "qliq_per_kw": (
        "Qж / Мощность",
        "Удельная подача на единицу потребляемой мощности.",
    ),
    "current_to_nominal_ratio": (
        "Ток / Номинальный ток",
        "Отношение фактического тока к номинальному току двигателя.",
    ),
    "qliq_per_current": (
        "Qж / Ток",
        "Удельная подача на единицу фактического тока.",
    ),
    "bhp_to_reservoir_ratio": (
        "Pзаб / Pпл",
        "Отношение забойного давления к пластовому давлению.",
    ),
    "pressure_margin_to_bubble": (
        "Pзаб - Дав. Нас",
        "Абсолютный запас по давлению относительно давления насыщения.",
    ),
    "submergence_margin_m": (
        "Запас погружения, м",
        "Разница между глубиной спуска и динамическим уровнем.",
    ),
    "submergence_margin_ratio": (
        "Запас погружения / Глубина",
        "Нормированный запас погружения относительно глубины спуска.",
    ),
    "nominal_head_per_stage": (
        "Ном. напор / ступень",
        "Номинальный напор насоса на одну ступень.",
    ),
    "motor_load_per_hz": (
        "Загрузка / Частота",
        "Процент загрузки двигателя на 1 Гц рабочей частоты.",
    ),
    "curve_work_per_meter": (
        "Кривизна / длина",
        "Работа в кривизне, нормированная на длину УЭЦН.",
    ),
    "cum_calcium_load_kg": (
        "Cum Ca load",
        "Кумулятивный прокси потока кальция за наработку: Ca × Qж × длительность.",
    ),
    "cum_chloride_load_kg": (
        "Cum Cl load",
        "Кумулятивный прокси потока хлоридов за наработку: Cl × Qж × длительность.",
    ),
    "cum_sulfate_load_kg": (
        "Cum SO4 load",
        "Кумулятивный прокси потока сульфатов за наработку: SO4 × Qж × длительность.",
    ),
    "cum_salt_load_kg": (
        "Cum salt load",
        "Суммарный солевой throughput-прокси на базе Ca + Cl + SO4, Qж и длительности.",
    ),
    "gypsum_scale_proxy": (
        "Gypsum proxy",
        "Прокси гипсовой/ангидритной нагрузки на базе Ca × SO4 × Qж × длительность.",
    ),
}


CATBOOST_FILTER_SPECS: list[dict[str, str]] = [
    {"key": "field", "column": "Месторождение", "label": "Месторождение", "type": "exact"},
    {"key": "contractor", "column": "Принадлежность", "label": "Принадлежность", "type": "exact"},
    {"key": "install_date_min", "column": "Дата монтажа", "label": "Дата монтажа > выбранная дата", "type": "date_min"},
    {"key": "stop_reason", "column": "Причина остановки", "label": "Причина остановки", "type": "exact"},
    {"key": "failure_flag", "column": "Признак отказа", "label": "Признак отказа", "type": "exact"},
    {"key": "failed_node", "column": "Отказавший узел", "label": "Отказавший узел", "type": "exact"},
    {"key": "failed_element", "column": "Отказавший элемент", "label": "Отказавший элемент", "type": "exact"},
    {"key": "failure_character", "column": "Характер неисправности", "label": "Характер неисправности", "type": "exact"},
    {"key": "failure_cause", "column": "Причина отказа УЭЦН", "label": "Причина отказа УЭЦН", "type": "exact"},
    {"key": "runtime_group", "column": "группа наработок", "label": "Группа наработок", "type": "exact"},
    {"key": "duration_min", "column": "Наработка (сут)", "label": "Наработка > выбранное значение", "type": "numeric_min"},
    {"key": "duration_max", "column": "Наработка (сут)", "label": "Наработка < выбранное значение", "type": "numeric_max"},
]


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg-app: #0f1115;
          --bg-shell: #121212;
          --bg-surface: rgba(21, 24, 33, 0.86);
          --bg-surface-2: rgba(26, 31, 41, 0.9);
          --bg-elevated: #1f2430;
          --text-primary: #e5e7eb;
          --text-secondary: #9ca3af;
          --text-muted: #7f8896;
          --accent: #6366f1;
          --accent-2: #8b5cf6;
          --accent-soft: rgba(99, 102, 241, 0.16);
          --accent-glow: rgba(99, 102, 241, 0.34);
          --border-soft: rgba(255, 255, 255, 0.08);
          --border-strong: rgba(255, 255, 255, 0.10);
          --shadow-soft: 0 20px 50px rgba(0, 0, 0, 0.28);
          --shadow-elevated: 0 16px 36px rgba(0, 0, 0, 0.24);
          --radius-xl: 24px;
          --radius-lg: 20px;
          --radius-md: 16px;
          --radius-sm: 12px;
          --gradient-accent: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          --gradient-surface: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(99, 102, 241, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.10), transparent 24%),
            linear-gradient(180deg, #0f1115 0%, #121212 45%, #151821 100%);
          color: var(--text-primary);
        }
        .block-container {
          max-width: 1520px;
          padding-top: 1.4rem;
          padding-bottom: 2.4rem;
        }
        h1, h2, h3, .stMarkdown, label, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
          color: var(--text-primary);
        }
        .cat-shell {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }
        .cat-hero, .cat-panel {
          border: 1px solid var(--border-soft);
          border-radius: var(--radius-lg);
          background: var(--gradient-surface), var(--bg-surface);
          box-shadow: var(--shadow-soft);
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
          padding: 1.5rem;
          transition:
            transform 180ms ease,
            border-color 180ms ease,
            background-color 180ms ease,
            box-shadow 180ms ease;
        }
        .cat-panel:hover, .cat-metric-card:hover {
          border-color: var(--border-strong);
          box-shadow: var(--shadow-elevated);
        }
        .cat-hero {
          padding: 1.6rem 1.6rem 1.3rem;
          position: relative;
          overflow: hidden;
        }
        .cat-hero::after {
          content: "";
          position: absolute;
          inset: auto -10% -35% auto;
          width: 280px;
          height: 280px;
          background: radial-gradient(circle, rgba(99,102,241,0.24), transparent 60%);
          pointer-events: none;
        }
        .cat-kicker {
          color: var(--accent-2);
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-size: 0.74rem;
          font-weight: 700;
          margin-bottom: 0.55rem;
        }
        .cat-title {
          font-size: 2rem;
          font-weight: 700;
          margin: 0;
          line-height: 1.05;
        }
        .cat-subtitle {
          margin: 0.55rem 0 0;
          color: var(--text-secondary);
          max-width: 820px;
          line-height: 1.45;
        }
        .cat-panel-title {
          font-size: clamp(1.12rem, 1.4vw, 1.32rem);
          font-weight: 600;
          letter-spacing: -0.02em;
          margin: 0;
        }
        .cat-panel-subtitle {
          color: var(--text-secondary);
          margin: 0.35rem 0 1.1rem;
          max-width: 820px;
          font-size: 0.94rem;
          line-height: 1.6;
        }
        .cat-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.85rem;
        }
        .cat-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.55rem 0.85rem;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 999px;
          background: rgba(255,255,255,0.04);
          color: var(--text-primary);
          font-size: 0.88rem;
          font-weight: 500;
          line-height: 1;
          white-space: nowrap;
        }
        .cat-chip-active {
          background:
            linear-gradient(180deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.12)),
            rgba(255, 255, 255, 0.03);
          border-color: rgba(139, 92, 246, 0.28);
          box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.16), 0 16px 36px rgba(76, 29, 149, 0.18);
          color: #f5f3ff;
          font-weight: 600;
        }
        .cat-metric-card {
          border: 1px solid var(--border-soft);
          border-radius: var(--radius-md);
          background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
          padding: 1.05rem 1.1rem;
          min-height: 104px;
          transition:
            transform 180ms ease,
            border-color 180ms ease,
            background-color 180ms ease,
            box-shadow 180ms ease;
        }
        .cat-metric-label {
          color: var(--text-secondary);
          font-size: 0.84rem;
          font-weight: 500;
          margin-bottom: 0.6rem;
        }
        .cat-metric-value {
          font-size: 1.3rem;
          font-weight: 600;
          line-height: 1.2;
        }
        .cat-metric-meta {
          color: var(--text-secondary);
          font-size: 0.86rem;
          line-height: 1.55;
          margin-top: 0.55rem;
        }
        [data-testid="stTabs"] > div > div {
          gap: 0.5rem;
        }
        [data-testid="stTabs"] {
          margin-top: 0.2rem;
        }
        [data-testid="stTabs"] button {
          border-radius: 999px;
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.06);
          color: var(--text-secondary);
          padding: 0.4rem 0.95rem;
          transition:
            transform 160ms ease,
            background-color 160ms ease,
            border-color 160ms ease,
            box-shadow 160ms ease;
        }
        [data-testid="stTabs"] button:hover {
          transform: translateY(-1px);
          border-color: rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.065);
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
          background: var(--gradient-accent);
          color: white;
          border-color: transparent;
          box-shadow: 0 14px 28px rgba(99, 102, 241, 0.24);
        }
        div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input,
        .stMultiSelect [data-baseweb="tag"],
        .stDataFrame, .stPlotlyChart, [data-testid="stDataEditor"] {
          border-radius: 14px !important;
        }
        div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
          background: rgba(31, 36, 48, 0.92) !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          color: var(--text-primary) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
          background:
            linear-gradient(180deg, rgba(76, 93, 122, 0.12), rgba(21, 24, 33, 0.18)),
            rgba(18, 22, 30, 0.72) !important;
          border: 1px dashed rgba(139, 92, 246, 0.28) !important;
          border-radius: 16px !important;
          min-height: 156px;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: rgba(139, 92, 246, 0.42) !important;
          background:
            linear-gradient(180deg, rgba(99, 102, 241, 0.12), rgba(21, 24, 33, 0.2)),
            rgba(18, 22, 30, 0.78) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] div {
          color: var(--text-secondary) !important;
        }
        .stButton button {
          background: var(--gradient-accent);
          border: none;
          border-radius: 14px;
          color: white;
          font-weight: 600;
          min-height: 42px;
          box-shadow: 0 14px 28px rgba(99, 102, 241, 0.24);
          transition:
            transform 160ms ease,
            background-color 160ms ease,
            border-color 160ms ease,
            box-shadow 160ms ease,
            opacity 160ms ease;
        }
        .stButton button:hover {
          transform: translateY(-1px);
          opacity: 0.96;
          box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.18), 0 20px 34px rgba(99, 102, 241, 0.22);
        }
        .stButton button[kind="secondary"] {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: none;
        }
        .stButton button[kind="secondary"]:hover {
          transform: translateY(-1px);
          border-color: rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.065);
          box-shadow: none;
        }
        .cat-chip-toolbar {
          display: flex;
          gap: 0.6rem;
          flex-wrap: wrap;
          margin: 0.2rem 0 0.85rem;
        }
        [data-testid="stMetric"] {
          background: transparent;
          border: none;
          padding: 0;
        }
        .stAlert {
          border-radius: 16px;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(18,18,18,0.95), rgba(21,24,33,0.95));
          border-right: 1px solid rgba(255,255,255,0.06);
        }
        .cat-sidebar-brand {
          border: 1px solid var(--border-soft);
          border-radius: 18px;
          background: var(--gradient-surface), rgba(21, 24, 33, 0.86);
          padding: 1rem 1rem 0.9rem;
          margin-bottom: 1rem;
          box-shadow: var(--shadow-soft);
        }
        .cat-sidebar-title {
          font-size: 1.1rem;
          font-weight: 700;
          margin: 0 0 0.35rem;
        }
        .cat-sidebar-subtitle {
          color: var(--text-secondary);
          font-size: 0.88rem;
          line-height: 1.4;
          margin: 0;
        }
        .cat-control-box {
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 18px;
          background: rgba(255,255,255,0.025);
          padding: 1rem 1rem 0.4rem;
          margin: 0.5rem 0 1rem;
        }
        .cat-control-title {
          font-size: 0.92rem;
          font-weight: 600;
          margin: 0 0 0.2rem;
        }
        .cat-control-subtitle {
          color: var(--text-secondary);
          font-size: 0.86rem;
          line-height: 1.55;
          margin: 0 0 0.7rem;
        }
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {
          border: 1px solid rgba(255,255,255,0.06);
          background: rgba(255,255,255,0.02);
          padding: 0.35rem;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataEditor"] > div {
          border-radius: 12px !important;
        }
        [data-testid="stMarkdownContainer"] p {
          line-height: 1.55;
        }
        [data-testid="stSlider"] [role="slider"] {
          box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }
        [data-testid="stSliderTickBarMin"],
        [data-testid="stSliderTickBarMax"] {
          background: rgba(255,255,255,0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f"<span class='cat-chip cat-chip-active'>{chip}</span>" for chip in chips if chip)
    st.markdown(
        f"""
        <div class="cat-shell">
          <section class="cat-hero">
            <div class="cat-kicker">WOWPUMP · CatBoost</div>
            <h1 class="cat-title">{title}</h1>
            <p class="cat-subtitle">{subtitle}</p>
            <div class="cat-chip-row">{chip_html}</div>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        logo_path = FRONTEND_ASSETS_DIR / "wowpumpLOGO.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        st.markdown(
            """
            <div class="cat-sidebar-brand">
              <div class="cat-sidebar-title">Модули</div>
              <p class="cat-sidebar-subtitle">Навигация в стиле исходного интерфейса. В локальном Streamlit сейчас активен модуль CatBoost.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_module = st.radio(
            "Раздел",
            options=[
                "Домашняя страница",
                "Хранение данных",
                "Словарь и отчеты",
                "Анализ работы насосов",
                "Настройка модели отказов",
                "Прогноз ремонтов",
            ],
            index=3,
            key="catboost_sidebar_module",
        )
        if selected_module != "Анализ работы насосов":
            st.info('В локальной Streamlit-версии сейчас реализован только модуль "Анализ работы насосов".')
        current_view = st.session_state.get("catboost_primary_view", CATBOOST_PRIMARY_VIEWS[0])
        if current_view not in CATBOOST_PRIMARY_VIEWS:
            current_view = CATBOOST_PRIMARY_VIEWS[0]
        if st.session_state.get("catboost_sidebar_primary_view") != current_view:
            st.session_state["catboost_sidebar_primary_view"] = current_view
        selected_view = st.radio(
            "Экран анализа",
            options=CATBOOST_PRIMARY_VIEWS,
            key="catboost_sidebar_primary_view",
        )
        st.session_state["catboost_primary_view"] = selected_view
        st.caption('Локальная Streamlit-версия повторяет модуль "Анализ работы насосов".')


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p class='cat-panel-subtitle'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <section class="cat-panel">
          <h3 class="cat-panel-title">{title}</h3>
          {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def close_panel() -> None:
    st.markdown("</section>", unsafe_allow_html=True)


def render_control_box(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='cat-control-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="cat-control-box">
          <div class="cat-control-title">{title}</div>
          {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def close_control_box() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, meta: str | None = None) -> str:
    meta_html = f"<div class='cat-metric-meta'>{meta}</div>" if meta else ""
    return f"""
    <div class="cat-metric-card">
      <div class="cat-metric-label">{label}</div>
      <div class="cat-metric-value">{value}</div>
      {meta_html}
    </div>
    """


def render_note_chips(notes: list[str]) -> None:
    if not notes:
        return
    chip_html = "".join(f"<span class='cat-chip'>{note}</span>" for note in notes)
    st.markdown(f"<div class='cat-chip-row'>{chip_html}</div>", unsafe_allow_html=True)


def render_stat_strip(items: list[tuple[str, str]]) -> None:
    chip_html = "".join(
        f"<span class='cat-chip'><strong>{label}:</strong> {value}</span>"
        for label, value in items
        if value
    )
    if chip_html:
        st.markdown(f"<div class='cat-chip-row'>{chip_html}</div>", unsafe_allow_html=True)


def render_button_selector(
    *,
    title: str,
    options: list[str],
    selected: list[str] | str | None,
    key_prefix: str,
    single: bool,
) -> str | list[str]:
    st.markdown(f"**{title}**")
    current_single = str(selected) if single and selected is not None else ""
    current_multi = list(selected) if isinstance(selected, list) else []
    cols_per_row = 4
    for start in range(0, len(options), cols_per_row):
        row = st.columns(cols_per_row)
        for column_container, option in zip(row, options[start : start + cols_per_row], strict=False):
            if single:
                active = current_single == option
                if column_container.button(
                    option,
                    key=f"{key_prefix}_{option}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                ):
                    current_single = option
            else:
                active = option in current_multi
                if column_container.button(
                    option,
                    key=f"{key_prefix}_{option}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                ):
                    if option in current_multi:
                        current_multi = [item for item in current_multi if item != option]
                    else:
                        current_multi = [*current_multi, option]
    return current_single if single else current_multi


def render_labeled_single_selector(
    *,
    title: str,
    options: list[str],
    selected: str,
    key_prefix: str,
    empty_label: str | None = None,
) -> str:
    selector_options = options[:]
    if empty_label is not None:
        selector_options = [""] + selector_options
    selected_value = render_button_selector(
        title=title,
        options=selector_options,
        selected=selected,
        key_prefix=key_prefix,
        single=True,
    )
    if empty_label is not None:
        return "" if selected_value == "" else str(selected_value)
    return str(selected_value)


def render_labeled_multi_selector(
    *,
    title: str,
    options: list[str],
    selected: list[str],
    key_prefix: str,
) -> list[str]:
    return list(
        render_button_selector(
            title=title,
            options=options,
            selected=selected,
            key_prefix=key_prefix,
            single=False,
        )
    )


def render_slider_metric(label: str, value: str) -> str:
    return f"""
    <div class="cat-metric-card">
      <div class="cat-metric-label">{label}</div>
      <div class="cat-metric-value">{value}</div>
    </div>
    """


def render_inline_section_title(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='cat-control-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style="margin: 0.45rem 0 0.55rem;">
          <div class="cat-control-title">{title}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21,24,33,0.22)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=28, r=20, t=36, b=28),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    if height is not None:
        fig.update_layout(height=height)
    return fig


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    lower_name = file_name.lower()
    engine = "openpyxl" if lower_name.endswith(".xlsx") else "xlrd"
    workbook = pd.ExcelFile(BytesIO(file_bytes), engine=engine)
    return {sheet_name: workbook.parse(sheet_name=sheet_name) for sheet_name in workbook.sheet_names}


def _streamlit_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for column in safe_df.columns:
        series = safe_df[column]
        if series.dtype != "object":
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        python_types = {type(value) for value in non_null.tolist()}
        if len(python_types) > 1:
            safe_df[column] = series.astype("string")
    return safe_df


def render_dataframe(df: pd.DataFrame, *, height: int | None = None) -> None:
    st.dataframe(_streamlit_safe_dataframe(df), use_container_width=True, height=height)


def normalize_lookup_value(value: str) -> str:
    return str(value).strip().casefold()


def should_exclude_default_feature(column_name: str) -> bool:
    normalized = normalize_lookup_value(column_name)
    return any(pattern in normalized for pattern in DEFAULT_EXCLUDED_FEATURE_PATTERNS)


def is_datetime_like_column_name(column_name: str) -> bool:
    normalized = normalize_lookup_value(column_name)
    return any(token in normalized for token in DATETIME_NAME_TOKENS)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    normalized = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")

    comma_mask = normalized.notna() & normalized.str.contains(",", regex=False) & ~normalized.str.contains(".", regex=False)
    if comma_mask.any():
        numeric.loc[comma_mask] = pd.to_numeric(
            normalized.loc[comma_mask].str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return numeric.replace([np.inf, -np.inf], np.nan)


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, errors="coerce", unit="D", origin="1899-12-30")

    string_series = series.astype("string").str.strip()
    parsed = pd.to_datetime(string_series, errors="coerce", dayfirst=True)
    numeric_series = coerce_numeric_series(series)
    numeric_mask = parsed.isna() & numeric_series.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric_series.loc[numeric_mask],
            errors="coerce",
            unit="D",
            origin="1899-12-30",
        )
    return parsed


def classify_columns(
    df: pd.DataFrame,
    target_column: str | None,
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    feature_df = df.drop(columns=[target_column], errors="ignore").copy()
    if selected_columns is not None:
        filtered = [column for column in selected_columns if column in feature_df.columns]
        feature_df = feature_df[filtered]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for column in feature_df.columns:
        raw_series = feature_df[column]
        non_null_count = int(raw_series.notna().sum())
        if non_null_count == 0:
            feature_df[column] = raw_series.astype("string")
            categorical_columns.append(column)
            continue

        if is_datetime_like_column_name(column):
            datetime_series = coerce_datetime_series(raw_series)
            datetime_count = int(datetime_series.notna().sum())
            if datetime_count / non_null_count >= min(DATETIME_PARSE_THRESHOLD, 0.5):
                feature_df[column] = datetime_series
                datetime_columns.append(column)
                continue

        numeric_series = coerce_numeric_series(raw_series)
        numeric_count = int(numeric_series.notna().sum())
        if non_null_count >= MIN_NON_NULL_FOR_ANALYSIS and numeric_count / non_null_count >= NUMERIC_PARSE_THRESHOLD:
            feature_df[column] = numeric_series
            numeric_columns.append(column)
            continue

        datetime_series = coerce_datetime_series(raw_series)
        datetime_count = int(datetime_series.notna().sum())
        if non_null_count >= MIN_NON_NULL_FOR_ANALYSIS and datetime_count / non_null_count >= DATETIME_PARSE_THRESHOLD:
            feature_df[column] = datetime_series
            datetime_columns.append(column)
            continue

        feature_df[column] = raw_series.astype("string").str.strip()
        categorical_columns.append(column)

    return feature_df, numeric_columns, categorical_columns, datetime_columns


def build_target_series(df: pd.DataFrame, target_column: str | None) -> pd.Series:
    if not target_column or target_column not in df.columns:
        return pd.Series(dtype=float)
    return coerce_numeric_series(df[target_column])


def build_overview(df: pd.DataFrame, target_column: str | None, selected_columns: list[str]) -> tuple[Overview, list[str], list[str], list[str]]:
    target = build_target_series(df, target_column)
    _, numeric_columns, categorical_columns, datetime_columns = classify_columns(df, target_column, selected_columns)
    denominator = df.shape[0] * max(df.shape[1], 1)
    missing_share = float(df.isna().sum().sum() / denominator) if denominator else 0.0
    valid_target = target.dropna()
    return (
        Overview(
            total_rows=int(df.shape[0]),
            total_columns=int(df.shape[1]),
            valid_target_rows=int(valid_target.shape[0]),
            missing_values_share=missing_share,
            numeric_column_count=len(numeric_columns),
            categorical_column_count=len(categorical_columns),
            datetime_column_count=len(datetime_columns),
            target_mean=float(valid_target.mean()) if not valid_target.empty else None,
            target_median=float(valid_target.median()) if not valid_target.empty else None,
            target_min=float(valid_target.min()) if not valid_target.empty else None,
            target_max=float(valid_target.max()) if not valid_target.empty else None,
        ),
        numeric_columns,
        categorical_columns,
        datetime_columns,
    )


def build_correlations(df: pd.DataFrame, target_column: str | None, selected_columns: list[str]) -> pd.DataFrame:
    target = build_target_series(df, target_column)
    if target.empty:
        return pd.DataFrame(columns=["feature", "correlation"])

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty:
        return pd.DataFrame(columns=["feature", "correlation"])

    feature_df, numeric_columns, _, datetime_columns = classify_columns(clean, target_column, selected_columns)
    correlations: list[dict[str, float | str | None]] = []
    correlation_map: dict[str, float | None] = {column: None for column in feature_df.columns}

    for column in numeric_columns:
        series = coerce_numeric_series(feature_df[column])
        if int(series.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = series.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlation_map[column] = float(corr)

    for column in datetime_columns:
        series = pd.to_datetime(feature_df[column], errors="coerce")
        numeric_time = series.map(lambda value: value.timestamp() if pd.notna(value) else pd.NA)
        numeric_time = pd.to_numeric(numeric_time, errors="coerce")
        if int(numeric_time.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = numeric_time.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlation_map[column] = float(corr)

    for column in selected_columns:
        if column in feature_df.columns:
            correlations.append({"feature": column, "correlation": correlation_map.get(column)})

    corr_df = pd.DataFrame(correlations)
    if corr_df.empty:
        return corr_df
    return corr_df.sort_values("correlation", key=lambda series: series.abs(), ascending=False, na_position="last").reset_index(drop=True)


def build_feature_importance(df: pd.DataFrame, target_column: str | None, selected_columns: list[str]) -> pd.DataFrame:
    target = build_target_series(df, target_column)
    if target.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty or clean["_target_numeric"].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(clean, target_column, selected_columns)
    if feature_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    prepared = pd.DataFrame(index=feature_df.index)
    for column in numeric_columns:
        prepared[column] = coerce_numeric_series(feature_df[column])
    for column in datetime_columns:
        dt_series = pd.to_datetime(feature_df[column], errors="coerce")
        prepared[column] = dt_series.map(lambda value: value.timestamp() if pd.notna(value) else None)
    for column in categorical_columns:
        prepared[column] = feature_df[column].astype("string").fillna("__missing__")

    if prepared.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    categorical_feature_indices = [prepared.columns.get_loc(column) for column in categorical_columns]
    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(prepared, clean["_target_numeric"], cat_features=categorical_feature_indices))
    importance_df = pd.DataFrame(
        {
            "feature": prepared.columns.tolist(),
            "importance": [float(value) for value in model.get_feature_importance()],
        }
    )
    importance_map = importance_df.set_index("feature")["importance"].to_dict()
    ordered = pd.DataFrame(
        [{"feature": column, "importance": float(importance_map.get(column, 0.0))} for column in selected_columns if column in prepared.columns]
    )
    return ordered.sort_values("importance", ascending=False).reset_index(drop=True)


def build_notes(overview: Overview, target_column: str | None, selected_columns: list[str]) -> list[str]:
    notes: list[str] = []
    if not target_column:
        notes.append("Целевая переменная не выбрана.")
    elif overview.valid_target_rows == 0:
        notes.append("В целевой переменной недостаточно числовых значений для анализа CatBoost.")
    if not selected_columns:
        notes.append("Зависимые переменные не выбраны.")
    if overview.total_rows < 20:
        notes.append("Набор данных небольшой, поэтому зависимости могут быть нестабильными.")
    return notes


def build_numeric_or_datetime_value(series: pd.Series) -> pd.Series:
    numeric = coerce_numeric_series(series)
    if int(numeric.notna().sum()) >= max(3, int(series.notna().sum() * 0.5)):
        return numeric
    dt = coerce_datetime_series(series)
    return pd.to_numeric(dt.map(lambda value: value.timestamp() if pd.notna(value) else pd.NA), errors="coerce")


def resolve_years(series: pd.Series) -> pd.Series:
    dt = coerce_datetime_series(series)
    return dt.dt.year


def build_time_summary_frame(df: pd.DataFrame, target_column: str, x_column: str, group_column: str | None, selected_groups: list[str], year_min: int | None, year_max: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df[[target_column, x_column] + ([group_column] if group_column else [])].copy()
    working["_target"] = coerce_numeric_series(working[target_column])
    working["_year"] = resolve_years(working[x_column])
    working = working[working["_target"].notna() & (working["_target"] != 0) & working["_year"].notna()].copy()
    if year_min is not None:
        working = working.loc[working["_year"] >= year_min]
    if year_max is not None:
        working = working.loc[working["_year"] <= year_max]
    if group_column:
        working["_group"] = working[group_column].astype("string").fillna("Пусто")
        if selected_groups:
            working = working.loc[working["_group"].isin(selected_groups)]
    else:
        working["_group"] = "Все данные"

    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        working.groupby(["_group", "_year"], dropna=False)
        .agg(target_mean=("_target", "mean"), rows=("_target", "size"))
        .reset_index()
        .sort_values(["_group", "_year"])
    )

    pivot = summary.pivot(index="_group", columns="_year", values="target_mean").round(1)
    pivot = pivot.reset_index().rename(columns={"_group": group_column or "Группа"})
    return summary, pivot


def build_scatter_frame(df: pd.DataFrame, x_column: str, y_column: str, color_column: str | None) -> pd.DataFrame:
    working = df.copy()
    working["_x"] = build_numeric_or_datetime_value(working[x_column])
    working["_y"] = build_numeric_or_datetime_value(working[y_column])
    working = working[working["_x"].notna() & working["_y"].notna()].copy()
    if color_column:
        working["_color"] = working[color_column].astype("string").fillna("Пусто")
    else:
        working["_color"] = "Все точки"
    return working


def filter_scatter_ranges(scatter_df: pd.DataFrame, x_min: float, x_max: float, y_min: float, y_max: float) -> pd.DataFrame:
    if scatter_df.empty:
        return scatter_df
    low_x, high_x = sorted([x_min, x_max])
    low_y, high_y = sorted([y_min, y_max])
    return scatter_df.loc[
        scatter_df["_x"].between(low_x, high_x, inclusive="both")
        & scatter_df["_y"].between(low_y, high_y, inclusive="both")
    ].copy()


def build_window_average_frame(scatter_df: pd.DataFrame, y_column: str) -> pd.DataFrame:
    if scatter_df.empty:
        return pd.DataFrame(columns=["window_label", "avg_x", "avg_y", "count", "x_start", "x_end"])
    x_min = float(scatter_df["_x"].min())
    x_max = float(scatter_df["_x"].max())
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min == x_max:
        return pd.DataFrame(columns=["window_label", "avg_x", "avg_y", "count", "x_start", "x_end"])

    step = (x_max - x_min) * 0.1
    if step <= 0:
        return pd.DataFrame(columns=["window_label", "avg_x", "avg_y", "count", "x_start", "x_end"])

    rows: list[dict[str, float | int | str]] = []
    current = x_min
    while current < x_max + (step * 0.001):
        next_edge = min(current + step, x_max)
        window = scatter_df.loc[scatter_df["_x"].between(current, next_edge, inclusive="both")].copy()
        window_y = coerce_numeric_series(window[y_column]) if y_column in window.columns else window["_y"]
        window = window.loc[window_y.notna()].copy()
        window["_window_y"] = window_y.loc[window.index]
        window = window.loc[window["_window_y"] != 0]
        if not window.empty:
            rows.append(
                {
                    "window_label": f"{current:.2f} - {next_edge:.2f}",
                    "avg_x": float(window["_x"].mean()),
                    "avg_y": float(window["_window_y"].mean()),
                    "count": int(len(window)),
                    "x_start": float(current),
                    "x_end": float(next_edge),
                }
            )
        if next_edge >= x_max:
            break
        current = next_edge
    return pd.DataFrame(rows)


def build_heatmap_frame(scatter_df: pd.DataFrame, color_mode: str, value_column: str | None) -> pd.DataFrame:
    if scatter_df.empty:
        return pd.DataFrame()
    working = scatter_df.copy()
    x_bins = np.linspace(float(working["_x"].min()), float(working["_x"].max()), 13)
    y_bins = np.linspace(float(working["_y"].min()), float(working["_y"].max()), 13)
    if len(np.unique(x_bins)) < 2 or len(np.unique(y_bins)) < 2:
        return pd.DataFrame()
    working["_x_bin"] = pd.cut(working["_x"], bins=x_bins, include_lowest=True, duplicates="drop")
    working["_y_bin"] = pd.cut(working["_y"], bins=y_bins, include_lowest=True, duplicates="drop")
    if color_mode == "__count__":
        grouped = working.groupby(["_y_bin", "_x_bin"], observed=False).size().reset_index(name="value")
    else:
        if not value_column or value_column not in working.columns:
            return pd.DataFrame()
        numeric = build_numeric_or_datetime_value(working[value_column])
        working["_value"] = numeric
        grouped = (
            working.loc[working["_value"].notna()]
            .groupby(["_y_bin", "_x_bin"], observed=False)["_value"]
            .mean()
            .reset_index(name="value")
        )
    grouped["_x_label"] = grouped["_x_bin"].astype("string")
    grouped["_y_label"] = grouped["_y_bin"].astype("string")
    return grouped


def build_histogram_frame(df: pd.DataFrame, value_column: str, group_column: str | None, selected_group_values: list[str], aggregate_selected: bool) -> pd.DataFrame:
    working = df.copy()
    value_numeric = coerce_numeric_series(working[value_column])
    value_datetime = coerce_datetime_series(working[value_column])
    numeric_share = value_numeric.notna().mean() if len(working) else 0
    datetime_share = value_datetime.notna().mean() if len(working) else 0

    if datetime_share >= max(0.5, numeric_share):
        working = working.loc[value_datetime.notna()].copy()
        working["_hist_value"] = value_datetime.loc[working.index].dt.year.astype("int64")
    else:
        working = working.loc[value_numeric.notna()].copy()
        working["_hist_value"] = value_numeric.loc[working.index]

    if group_column:
        if group_column == "__year__":
            year_values = resolve_years(working[value_column]).astype("Int64").astype("string")
            working["_group"] = year_values.fillna("Пусто")
        else:
            working["_group"] = working[group_column].astype("string").fillna("Пусто")
        if selected_group_values:
            working = working.loc[working["_group"].isin(selected_group_values)]
            if aggregate_selected:
                working["_group"] = "Выбранные вместе"
    else:
        working["_group"] = "Все данные"

    return working


def prepare_forecast_frame(df: pd.DataFrame, target_column: str | None, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str], pd.Series]:
    target = build_target_series(df, target_column)
    if target.empty:
        return pd.DataFrame(), [], [], pd.Series(dtype=float)

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty or clean["_target_numeric"].nunique() < 2:
        return pd.DataFrame(), [], [], pd.Series(dtype=float)

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(clean, target_column, feature_columns)
    prepared = pd.DataFrame(index=feature_df.index)
    numeric_feature_columns: list[str] = []

    for column in numeric_columns:
        prepared[column] = coerce_numeric_series(feature_df[column])
        numeric_feature_columns.append(column)
    for column in datetime_columns:
        dt_series = pd.to_datetime(feature_df[column], errors="coerce")
        prepared[column] = dt_series.map(lambda value: value.timestamp() if pd.notna(value) else None)
        numeric_feature_columns.append(column)
    for column in categorical_columns:
        prepared[column] = feature_df[column].astype("string").fillna("__missing__")

    prepared = prepared.dropna(axis=0, how="all")
    if prepared.empty:
        return pd.DataFrame(), [], [], pd.Series(dtype=float)

    target_series = clean.loc[prepared.index, "_target_numeric"]
    return prepared, numeric_feature_columns, categorical_columns, target_series


def train_forecast_model(df: pd.DataFrame, target_column: str | None, feature_columns: list[str], test_fraction: float, random_seed: int) -> TrainedForecast | None:
    prepared, numeric_feature_columns, categorical_columns, target_series = prepare_forecast_frame(df, target_column, feature_columns)
    if prepared.empty or target_series.empty:
        return None

    test_fraction = min(max(float(test_fraction), 0.05), 0.4)
    total_rows = len(prepared)
    test_rows = max(1, int(total_rows * test_fraction)) if total_rows > 4 else max(1, total_rows // 3)
    test_rows = min(test_rows, max(total_rows - 1, 1))
    shuffled_indices = prepared.sample(frac=1.0, random_state=int(random_seed)).index.tolist()
    test_index = shuffled_indices[:test_rows]
    train_index = shuffled_indices[test_rows:] or shuffled_indices[: max(total_rows - test_rows, 1)]

    train_frame = prepared.loc[train_index]
    train_target = target_series.loc[train_index]
    test_frame = prepared.loc[test_index]
    test_target = target_series.loc[test_index]
    categorical_feature_indices = [prepared.columns.get_loc(column) for column in categorical_columns]

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(random_seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(train_frame, train_target, cat_features=categorical_feature_indices))

    predicted = model.predict(test_frame)
    residual = test_target.to_numpy(dtype=float) - predicted
    rmse = float(np.sqrt(np.mean(np.square(residual)))) if len(test_target) else None
    mae = float(np.mean(np.abs(residual))) if len(test_target) else None
    if len(test_target) and test_target.nunique() > 1:
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(test_target.to_numpy(dtype=float) - test_target.mean())))
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot else None
    else:
        r2 = None

    baseline_values: dict[str, str | float] = {}
    for column in prepared.columns:
        if column in numeric_feature_columns:
            baseline_values[column] = float(pd.to_numeric(prepared[column], errors="coerce").median())
        else:
            mode = prepared[column].mode(dropna=True)
            baseline_values[column] = str(mode.iloc[0]) if not mode.empty else "__missing__"

    return TrainedForecast(
        model=model,
        prepared=prepared,
        actual=target_series.loc[prepared.index],
        feature_columns=prepared.columns.tolist(),
        numeric_feature_columns=numeric_feature_columns,
        metrics={
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        baseline_values=baseline_values,
        categorical_columns=categorical_columns,
    )


def predict_forecast_rows(trained: TrainedForecast, rows: list[dict]) -> list[float | None]:
    prediction_frame = pd.DataFrame(index=range(len(rows)))
    numeric_feature_set = set(trained.numeric_feature_columns)

    for column in trained.feature_columns:
        source_values = [row.get(column, trained.baseline_values.get(column)) for row in rows]
        series = pd.Series(source_values, dtype="object")
        if column in numeric_feature_set:
            prediction_frame[column] = coerce_numeric_series(series)
        else:
            prediction_frame[column] = series.astype("string").fillna("__missing__")

    result: list[float | None] = [None] * len(rows)
    valid_mask = prediction_frame.notna().any(axis=1)
    if valid_mask.any():
        predicted = trained.model.predict(prediction_frame.loc[valid_mask])
        for index, value in zip(prediction_frame.loc[valid_mask].index.tolist(), predicted, strict=False):
            result[index] = float(value)
    return result


def build_forecast_contour(trained: TrainedForecast, x_feature: str, y_feature: str, slice_overrides: dict[str, str | float]) -> pd.DataFrame:
    numeric_feature_set = set(trained.numeric_feature_columns)
    if x_feature not in numeric_feature_set or y_feature not in numeric_feature_set or x_feature == y_feature:
        return pd.DataFrame()

    x_series = pd.to_numeric(trained.prepared[x_feature], errors="coerce").dropna()
    y_series = pd.to_numeric(trained.prepared[y_feature], errors="coerce").dropna()
    if x_series.empty or y_series.empty:
        return pd.DataFrame()

    x_values = np.linspace(float(x_series.min()), float(x_series.max()), 24)
    y_values = np.linspace(float(y_series.min()), float(y_series.max()), 24)
    grid_rows: list[dict[str, str | float]] = []

    for y_value in y_values:
        for x_value in x_values:
            row = {column: trained.baseline_values.get(column) for column in trained.feature_columns}
            for column, value in slice_overrides.items():
                if column in row:
                    row[column] = value
            row[x_feature] = float(x_value)
            row[y_feature] = float(y_value)
            grid_rows.append(row)

    predictions = predict_forecast_rows(trained, grid_rows)
    contour_rows: list[dict[str, float]] = []
    offset = 0
    for y_value in y_values:
        row_predictions = predictions[offset : offset + len(x_values)]
        for x_value, prediction in zip(x_values, row_predictions, strict=False):
            contour_rows.append(
                {
                    x_feature: float(x_value),
                    y_feature: float(y_value),
                    "prediction": float(prediction) if prediction is not None else np.nan,
                }
            )
        offset += len(x_values)
    return pd.DataFrame(contour_rows)


def format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def format_percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def pick_default_target(df: pd.DataFrame) -> str:
    numeric_candidates: list[tuple[str, int]] = []
    for column in df.columns:
        numeric = coerce_numeric_series(df[column])
        numeric_candidates.append((str(column), int(numeric.notna().sum())))
    numeric_candidates.sort(key=lambda item: item[1], reverse=True)
    return numeric_candidates[0][0] if numeric_candidates else str(df.columns[0])


def current_dataset_signature(file_name: str, sheet_name: str, columns: list[str], rows: int) -> str:
    return f"{file_name}|{sheet_name}|{rows}|{'|'.join(columns)}"


def ensure_derived_defaults(signature: str, available_names: list[str]) -> None:
    if st.session_state.get("catboost_derived_signature") == signature:
        return
    st.session_state["catboost_enabled_derived_columns"] = available_names[:]
    st.session_state["catboost_derived_signature"] = signature


def ensure_filter_defaults(signature: str, available_keys: list[str]) -> None:
    if st.session_state.get("catboost_filter_signature") == signature:
        return
    st.session_state["catboost_active_filters"] = []
    for key in available_keys:
        st.session_state.pop(f"catboost_filter_value_{key}", None)
    st.session_state["catboost_filter_signature"] = signature


def ensure_selection_defaults(signature: str, columns: list[str], default_target: str) -> None:
    if st.session_state.get("catboost_selection_signature") == signature:
        return
    st.session_state["catboost_target_column"] = default_target
    st.session_state["catboost_feature_columns"] = [
        column
        for column in columns
        if column != default_target and not should_exclude_default_feature(column)
    ]
    st.session_state["catboost_selection_signature"] = signature
    st.session_state.pop("catboost_trained_model", None)
    st.session_state.pop("catboost_trained_signature", None)


def apply_feature_default_exclusions_once(signature: str, target_column: str) -> None:
    if st.session_state.get("catboost_feature_defaults_applied_signature") == signature:
        return
    current_features = list(st.session_state.get("catboost_feature_columns", []))
    st.session_state["catboost_feature_columns"] = [
        column
        for column in current_features
        if column != target_column and not should_exclude_default_feature(column)
    ]
    st.session_state["catboost_feature_defaults_applied_signature"] = signature


def maybe_reset_trained_model(signature: str, target_column: str, feature_columns: list[str]) -> None:
    trained_signature = f"{signature}|{target_column}|{'|'.join(feature_columns)}"
    if st.session_state.get("catboost_trained_signature") != trained_signature:
        st.session_state.pop("catboost_trained_model", None)
    st.session_state["catboost_expected_signature"] = trained_signature


def build_catboost_derived_specs(df: pd.DataFrame) -> tuple[list[dict[str, str]], list[str]]:
    inspected = inspect_derived_presets(df)
    available_specs: list[dict[str, str]] = []
    unavailable_notes: list[str] = []
    for preset in inspected:
        if not preset.available or not preset.formula:
            unavailable_notes.append(f"{preset.name}: {preset.availability_note}")
            continue
        display_name, description = CATBOOST_DERIVED_DISPLAY_MAP.get(
            preset.name,
            (preset.name, preset.description),
        )
        available_specs.append(
            {
                "name": display_name,
                "source_name": preset.name,
                "formula": preset.formula,
                "description": description,
                "category": preset.category,
            }
        )
    available_specs.sort(key=lambda item: (str(item.get("category", "")), str(item["name"])))
    return available_specs, unavailable_notes


def apply_catboost_derived_layer(df: pd.DataFrame, selected_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    derived_specs, _ = build_catboost_derived_specs(df)
    selected_specs = [spec for spec in derived_specs if spec["name"] in selected_names]
    if not selected_specs:
        return df.copy(), pd.DataFrame(index=df.index), []
    processed, notes = apply_derived_columns(
        df.copy(),
        [{"name": spec["name"], "formula": spec["formula"]} for spec in selected_specs],
    )
    derived_columns = [spec["name"] for spec in selected_specs if spec["name"] in processed.columns]
    derived_frame = processed[derived_columns].copy() if derived_columns else pd.DataFrame(index=df.index)
    return processed, derived_frame, notes


def build_selection_signature(target_column: str, feature_columns: list[str], derived_columns: list[str]) -> str:
    return f"{target_column}|{'|'.join(sorted(feature_columns))}|{'|'.join(sorted(derived_columns))}"


def build_filter_signature(active_filters: list[str], filter_values: dict[str, str]) -> str:
    parts = [f"{key}={filter_values.get(key, '')}" for key in sorted(active_filters)]
    return "|".join(parts)


def available_filter_specs(df: pd.DataFrame) -> list[dict[str, str]]:
    return [spec for spec in CATBOOST_FILTER_SPECS if spec["column"] in df.columns]


def get_filter_choice_values(df: pd.DataFrame, column: str) -> list[str]:
    series = df[column].astype("string").fillna("Пусто").str.strip()
    values = sorted(value for value in series.unique().tolist() if value)
    return values


def apply_catboost_filters(df: pd.DataFrame, active_filters: list[str], filter_values: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    filtered = df.copy()
    notes: list[str] = []
    spec_by_key = {spec["key"]: spec for spec in available_filter_specs(df)}
    for key in active_filters:
        spec = spec_by_key.get(key)
        if spec is None:
            continue
        column = spec["column"]
        raw_value = filter_values.get(key, "")
        if spec["type"] == "exact":
            if raw_value == "":
                continue
            series = filtered[column].astype("string").fillna("Пусто").str.strip()
            filtered = filtered.loc[series == raw_value].copy()
            notes.append(f"{spec['label']} = {raw_value}")
        elif spec["type"] == "date_min":
            if raw_value == "":
                continue
            threshold = pd.to_datetime(raw_value, errors="coerce")
            if pd.isna(threshold):
                continue
            series = coerce_datetime_series(filtered[column])
            filtered = filtered.loc[series > threshold].copy()
            notes.append(f"{spec['label']}: {threshold.date().isoformat()}")
        elif spec["type"] == "numeric_min":
            if raw_value == "":
                continue
            threshold_numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(threshold_numeric):
                continue
            series = coerce_numeric_series(filtered[column])
            filtered = filtered.loc[series > float(threshold_numeric)].copy()
            notes.append(f"{spec['label']}: {float(threshold_numeric):g}")
        elif spec["type"] == "numeric_max":
            if raw_value == "":
                continue
            threshold_numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(threshold_numeric):
                continue
            series = coerce_numeric_series(filtered[column])
            filtered = filtered.loc[series < float(threshold_numeric)].copy()
            notes.append(f"{spec['label']}: {float(threshold_numeric):g}")
    return filtered, notes


def main() -> None:
    inject_app_styles()
    render_sidebar()
    render_hero(
        "Анализ работы насосов",
        "Локальный модуль CatBoost с тем же характером интерфейса: тёмные панели, обзор метрик, факторный анализ, временные графики и номограмма прогноза.",
        CATBOOST_PRIMARY_VIEWS,
    )
    primary_view = render_labeled_single_selector(
        title="Быстрый переход",
        options=CATBOOST_PRIMARY_VIEWS,
        selected=st.session_state.get("catboost_primary_view", CATBOOST_PRIMARY_VIEWS[0]),
        key_prefix="primary_view_button",
    )
    st.session_state["catboost_primary_view"] = primary_view

    render_panel_header("Источник данных", "Загрузите Excel-файл и переключайтесь между листами без лишних шагов. После выбора листа обзор таблицы обновляется автоматически.")
    upload_cols = st.columns([1.3, 1.7])
    with upload_cols[0]:
        uploaded = st.file_uploader("Загрузить Excel-файл", type=["xlsx", "xls"], key="catboost_file_uploader")
    with upload_cols[1]:
        st.markdown(
            render_metric_card(
                "Режим работы",
                "Автоанализ",
                "После загрузки файла приложение сразу строит обзор, выбор признаков и прогнозные вкладки.",
            ),
            unsafe_allow_html=True,
        )
    if not uploaded:
        close_panel()
        st.info("Загрузите Excel-файл, чтобы начать анализ.")
        return

    workbook = load_workbook(uploaded.getvalue(), uploaded.name)
    sheet_names = list(workbook.keys())
    default_sheet = st.session_state.get("catboost_sheet_name", sheet_names[0])
    if default_sheet not in sheet_names:
        default_sheet = sheet_names[0]
    sheet_name = render_labeled_single_selector(
        title="Лист",
        options=sheet_names,
        selected=default_sheet,
        key_prefix="sheet_button",
    )
    st.session_state["catboost_sheet_name"] = sheet_name
    render_stat_strip(
        [
            ("Файл", uploaded.name),
            ("Листов", str(len(workbook))),
            ("Текущий лист", sheet_name),
        ]
    )
    close_panel()

    df_raw = workbook[sheet_name].copy()
    df_raw.columns = [str(column) for column in df_raw.columns]

    raw_signature = current_dataset_signature(uploaded.name, sheet_name, df_raw.columns.tolist(), len(df_raw))
    available_derived_specs, unavailable_derived_notes = build_catboost_derived_specs(df_raw)
    available_derived_names = [spec["name"] for spec in available_derived_specs]
    filter_specs = available_filter_specs(df_raw)
    available_filter_keys = [spec["key"] for spec in filter_specs]
    ensure_derived_defaults(raw_signature, available_derived_names)
    ensure_filter_defaults(raw_signature, available_filter_keys)
    initial_enabled_derived_columns = [
        column
        for column in st.session_state.get("catboost_enabled_derived_columns", [])
        if column in available_derived_names
    ]
    active_filters = [
        key
        for key in st.session_state.get("catboost_active_filters", [])
        if key in available_filter_keys
    ]
    enabled_derived_columns = initial_enabled_derived_columns[:]
    df_draft, derived_df, derived_notes = apply_catboost_derived_layer(df_raw, enabled_derived_columns)

    signature = current_dataset_signature(
        uploaded.name,
        sheet_name,
        [*df_draft.columns.tolist(), *enabled_derived_columns],
        len(df_draft),
    )
    ensure_selection_defaults(signature, df_draft.columns.tolist(), pick_default_target(df_draft))
    target_column = st.session_state.get("catboost_target_column", df_draft.columns.tolist()[0] if len(df_draft.columns) else "")
    apply_feature_default_exclusions_once(signature, target_column)
    feature_columns = [
        column
        for column in st.session_state.get("catboost_feature_columns", [])
        if column != target_column and column in df_draft.columns
    ]

    render_panel_header("Предпросмотр источника", "Быстрый обзор текущего листа с основными размерами набора данных.")
    preview_cols = st.columns([2, 1])
    with preview_cols[0]:
        render_dataframe(df_raw.head(300), height=320)
    with preview_cols[1]:
        metric_html = "".join(
            [
                render_metric_card("Строк", f"{len(df_raw):,}"),
                render_metric_card("Колонок", f"{len(df_raw.columns):,}"),
                render_metric_card("Листов", f"{len(workbook):,}"),
            ]
        )
        st.markdown(metric_html, unsafe_allow_html=True)
    close_panel()

    render_panel_header("Выбор переменных", "После выбора целевой переменной CatBoost использует все включённые зависимые переменные для основного анализа и прогноза.")
    st.markdown("<div class='cat-chip-toolbar'></div>", unsafe_allow_html=True)
    target_column = render_button_selector(
        title="Целевая переменная",
        options=df_draft.columns.tolist(),
        selected=target_column,
        key_prefix="target_button",
        single=True,
    )
    feature_candidates = [column for column in df_draft.columns if column != target_column]
    feature_columns = [column for column in feature_columns if column in feature_candidates]
    action_cols = st.columns([1, 1, 4])
    if action_cols[0].button("Выбрать все", key="features_select_all", type="secondary", use_container_width=True):
        feature_columns = feature_candidates[:]
    if action_cols[1].button("Очистить", key="features_clear_all", type="secondary", use_container_width=True):
        feature_columns = []
    feature_columns = render_button_selector(
        title="Зависимые переменные",
        options=feature_candidates,
        selected=feature_columns,
        key_prefix="feature_button",
        single=False,
    )
    st.session_state["catboost_target_column"] = target_column
    st.session_state["catboost_feature_columns"] = feature_columns
    close_panel()

    render_panel_header("Производные переменные", "Промежуточный слой расширяет исходный dataframe расчётными признаками. Включённые производные переменные дальше доступны наравне с обычными колонками.")
    if not available_derived_specs:
        st.info("Для выбранного листа пока не удалось вычислить ни одну производную переменную из каталога пресетов.")
    else:
        derived_action_cols = st.columns([1, 1, 4])
        if derived_action_cols[0].button("Выбрать все", key="derived_select_all", type="secondary", use_container_width=True):
            enabled_derived_columns = available_derived_names[:]
        if derived_action_cols[1].button("Очистить", key="derived_clear_all", type="secondary", use_container_width=True):
            enabled_derived_columns = []
        enabled_derived_columns = list(
            render_button_selector(
                title="Производные признаки",
                options=available_derived_names,
                selected=enabled_derived_columns,
                key_prefix="derived_feature_button",
                single=False,
            )
        )
        st.session_state["catboost_enabled_derived_columns"] = enabled_derived_columns
        render_stat_strip(
            [
                ("Доступно", str(len(available_derived_names))),
                ("Включено", str(len(enabled_derived_columns))),
                ("Итоговых колонок", str(len(df_draft.columns))),
            ]
        )
        if unavailable_derived_notes:
            render_note_chips(unavailable_derived_notes)
        if derived_notes:
            render_note_chips(derived_notes)
        if enabled_derived_columns and not derived_df.empty:
            render_inline_section_title("Предпросмотр расширенного слоя", "Добавленные производные колонки для текущего листа.")
            render_dataframe(derived_df.head(200), height=240)
            formula_rows = pd.DataFrame(
                [
                    {
                        "Производная переменная": spec["name"],
                        "Категория": spec.get("category", ""),
                        "Формула": spec["formula"],
                        "Описание": spec["description"],
                    }
                    for spec in available_derived_specs
                    if spec["name"] in enabled_derived_columns
                ]
            )
            if not formula_rows.empty:
                render_dataframe(formula_rows, height=220)
    close_panel()

    st.session_state["catboost_enabled_derived_columns"] = enabled_derived_columns
    render_panel_header("Фильтры данных", "Включайте только нужные фильтры. Категориальные фильтры используют точное совпадение, а дата монтажа и наработка работают как пороги `>`.")
    if not filter_specs:
        st.info("Для текущего листа доступные фильтры не найдены.")
    else:
        filter_options = [spec["label"] for spec in filter_specs]
        label_to_key = {spec["label"]: spec["key"] for spec in filter_specs}
        key_to_spec = {spec["key"]: spec for spec in filter_specs}
        selected_filter_labels = [spec["label"] for spec in filter_specs if spec["key"] in active_filters]
        filter_action_cols = st.columns([1, 1, 4])
        if filter_action_cols[0].button("Выбрать все", key="filters_select_all", type="secondary", use_container_width=True):
            selected_filter_labels = filter_options[:]
        if filter_action_cols[1].button("Очистить", key="filters_clear_all", type="secondary", use_container_width=True):
            selected_filter_labels = []
        selected_filter_labels = list(
            render_button_selector(
                title="Активные фильтры",
                options=filter_options,
                selected=selected_filter_labels,
                key_prefix="filter_button",
                single=False,
            )
        )
        active_filters = [label_to_key[label] for label in selected_filter_labels if label in label_to_key]
        st.session_state["catboost_active_filters"] = active_filters
        draft_filter_values: dict[str, str] = {}
        for key in active_filters:
            spec = key_to_spec[key]
            column = spec["column"]
            stored_value = st.session_state.get(f"catboost_filter_value_{key}", "")
            if spec["type"] == "exact":
                options = get_filter_choice_values(df_raw, column)
                current_value = stored_value if stored_value in options else (options[0] if options else "")
                selected_value = st.selectbox(
                    spec["label"],
                    options=options,
                    index=options.index(current_value) if current_value in options else 0,
                    key=f"catboost_filter_select_{key}",
                ) if options else ""
                st.session_state[f"catboost_filter_value_{key}"] = selected_value
                draft_filter_values[key] = str(selected_value)
            elif spec["type"] == "date_min":
                series = coerce_datetime_series(df_raw[column]).dropna()
                if not series.empty:
                    default_date = pd.to_datetime(stored_value, errors="coerce")
                    chosen_date = default_date.date() if pd.notna(default_date) else series.min().date()
                    selected_date = st.date_input(spec["label"], value=chosen_date, key=f"catboost_filter_date_{key}")
                    selected_date_value = selected_date.isoformat() if isinstance(selected_date, date) else str(selected_date)
                    st.session_state[f"catboost_filter_value_{key}"] = selected_date_value
                    draft_filter_values[key] = selected_date_value
            elif spec["type"] in {"numeric_min", "numeric_max"}:
                series = coerce_numeric_series(df_raw[column]).dropna()
                if spec["type"] == "numeric_min":
                    default_numeric = float(series.min()) if not series.empty else 0.0
                else:
                    default_numeric = float(series.max()) if not series.empty else 0.0
                stored_numeric = pd.to_numeric(pd.Series([stored_value]), errors="coerce").iloc[0]
                current_numeric = float(stored_numeric) if pd.notna(stored_numeric) else default_numeric
                selected_numeric = st.number_input(spec["label"], value=float(current_numeric), step=1.0, key=f"catboost_filter_number_{key}")
                st.session_state[f"catboost_filter_value_{key}"] = str(selected_numeric)
                draft_filter_values[key] = str(selected_numeric)
        filtered_preview_df, active_filter_notes = apply_catboost_filters(df_raw, active_filters, draft_filter_values)
        render_stat_strip(
            [
                ("Активно фильтров", str(len(active_filters))),
                ("Строк после фильтра", str(len(filtered_preview_df))),
                ("Исходных строк", str(len(df_raw))),
            ]
        )
        if active_filter_notes:
            render_note_chips(active_filter_notes)
    close_panel()

    st.session_state["catboost_enabled_derived_columns"] = enabled_derived_columns
    filter_value_snapshot = {
        key: str(st.session_state.get(f"catboost_filter_value_{key}", ""))
        for key in active_filters
    }
    draft_selection_signature = (
        build_selection_signature(target_column, feature_columns, enabled_derived_columns)
        + "|filters|"
        + build_filter_signature(active_filters, filter_value_snapshot)
    )

    render_panel_header("Запуск расчёта", "Изменения в переключателях не запускают анализ сразу. Нажмите `Calculate`, чтобы применить текущий набор переменных.")
    calculate_clicked = st.button("Calculate", type="primary", use_container_width=True, key="catboost_calculate")
    if calculate_clicked:
        st.session_state["catboost_applied_raw_signature"] = raw_signature
        st.session_state["catboost_applied_target_column"] = target_column
        st.session_state["catboost_applied_feature_columns"] = feature_columns[:]
        st.session_state["catboost_applied_derived_columns"] = enabled_derived_columns[:]
        st.session_state["catboost_applied_active_filters"] = active_filters[:]
        st.session_state["catboost_applied_filter_values"] = filter_value_snapshot.copy()
        st.session_state["catboost_applied_selection_signature"] = draft_selection_signature
        st.session_state.pop("catboost_trained_model", None)
        st.session_state.pop("catboost_trained_signature", None)

    applied_ready = (
        st.session_state.get("catboost_applied_raw_signature") == raw_signature
        and bool(st.session_state.get("catboost_applied_selection_signature"))
    )
    if applied_ready:
        applied_target_column = st.session_state.get("catboost_applied_target_column", target_column)
        applied_feature_columns = list(st.session_state.get("catboost_applied_feature_columns", []))
        applied_derived_columns = list(st.session_state.get("catboost_applied_derived_columns", []))
        applied_active_filters = list(st.session_state.get("catboost_applied_active_filters", []))
        render_stat_strip(
            [
                ("Целевая", applied_target_column),
                ("Зависимых", str(len(applied_feature_columns))),
                ("Производных", str(len(applied_derived_columns))),
                ("Фильтров", str(len(applied_active_filters))),
            ]
        )
        if st.session_state.get("catboost_applied_selection_signature") != draft_selection_signature:
            st.info("Ниже показан последний рассчитанный вариант. Текущие изменения в переключателях будут применены только после нажатия `Calculate`.")
    else:
        st.info("Выберите нужные переменные и нажмите `Calculate`, чтобы построить анализ.")
    close_panel()

    if not applied_ready:
        return

    applied_active_filters = list(st.session_state.get("catboost_applied_active_filters", []))
    applied_filter_values = dict(st.session_state.get("catboost_applied_filter_values", {}))
    filtered_raw_df, applied_filter_notes = apply_catboost_filters(df_raw, applied_active_filters, applied_filter_values)
    applied_derived_columns = list(st.session_state.get("catboost_applied_derived_columns", []))
    df, derived_df_applied, derived_notes_applied = apply_catboost_derived_layer(
        filtered_raw_df,
        applied_derived_columns,
    )
    if df.empty:
        render_panel_header("Результат фильтрации", "После применения фильтров в выборке не осталось строк.")
        render_note_chips(applied_filter_notes)
        close_panel()
        return
    target_column = st.session_state.get("catboost_applied_target_column", target_column)
    feature_columns = [
        column
        for column in st.session_state.get("catboost_applied_feature_columns", [])
        if column != target_column and column in df.columns
    ]
    applied_signature = current_dataset_signature(
        uploaded.name,
        sheet_name,
        [*df.columns.tolist(), *applied_derived_columns],
        len(df),
    )

    maybe_reset_trained_model(applied_signature, target_column, feature_columns)
    overview, numeric_columns, categorical_columns, datetime_columns = build_overview(df, target_column, feature_columns)
    _, all_numeric_columns, all_categorical_columns, all_datetime_columns = classify_columns(df, target_column, None)
    notes = build_notes(overview, target_column, feature_columns)
    if applied_filter_notes:
        notes = [*notes, *applied_filter_notes]
    if derived_notes_applied:
        notes = [*notes, *derived_notes_applied]
    correlation_df = build_correlations(df, target_column, feature_columns)
    importance_df = build_feature_importance(df, target_column, feature_columns) if target_column and feature_columns else pd.DataFrame(columns=["feature", "importance"])

    tab_definitions = {
        "dashboard": "Сводка",
        "time": "Во времени",
        "scatter": "Scatter и Heatmap",
        "hist": "Гистограммы",
        "forecast": "Прогноз",
    }
    tab_order = ["dashboard", "time", "scatter", "hist", "forecast"]
    primary_view = st.session_state.get("catboost_primary_view", CATBOOST_PRIMARY_VIEWS[0])
    if primary_view == "Факторный анализ CatBoost":
        tab_order = ["forecast", "dashboard", "scatter", "hist", "time"]
    elif primary_view == "Корреляции":
        tab_order = ["scatter", "dashboard", "hist", "time", "forecast"]
    elif primary_view == "Номограмма":
        tab_order = ["forecast", "scatter", "dashboard", "hist", "time"]
    tab_objects = {
        key: tab
        for key, tab in zip(
            tab_order,
            st.tabs([tab_definitions[key] for key in tab_order]),
            strict=False,
        )
    }
    tab_dashboard = tab_objects["dashboard"]
    tab_time = tab_objects["time"]
    tab_scatter = tab_objects["scatter"]
    tab_hist = tab_objects["hist"]
    tab_forecast = tab_objects["forecast"]

    with tab_dashboard:
        render_panel_header("Сводка", "Ключевые метрики по таблице и выбранной целевой переменной.")
        metric_cols = st.columns(6)
        metric_specs = [
            ("Строк", f"{overview.total_rows:,}", None),
            ("Колонок в анализе", f"{overview.numeric_column_count + overview.categorical_column_count + overview.datetime_column_count:,}", None),
            ("Валидный target", f"{overview.valid_target_rows:,}", None),
            ("Среднее target", format_metric(overview.target_mean), None),
            ("Медиана target", format_metric(overview.target_median), None),
            ("Пропуски", format_percent(overview.missing_values_share), None),
        ]
        for column, (label, value, meta) in zip(metric_cols, metric_specs, strict=False):
            column.markdown(render_metric_card(label, value, meta), unsafe_allow_html=True)
        render_note_chips(notes)
        close_panel()

        left, right = st.columns(2)
        with left:
            render_panel_header("Важность факторов CatBoost", "Важность зависимых переменных относительно целевой переменной.")
            if importance_df.empty:
                st.info("Для текущего выбора важности признаков пока недоступны.")
            else:
                fig = px.bar(
                    importance_df.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    text="importance",
                )
                fig.update_traces(texttemplate="%{text:.2f}")
                apply_plot_style(fig, height=max(360, 32 * len(importance_df)))
                st.plotly_chart(fig, use_container_width=True)
            close_panel()

        with right:
            render_panel_header("Корреляционная матрица", "Корреляция признаков с целевой переменной.")
            if correlation_df.empty:
                st.info("Для текущего выбора нет валидных корреляций между признаками и целевой переменной.")
            else:
                heatmap_df = correlation_df.copy()
                heatmap_df["target"] = target_column
                z_values = [heatmap_df["correlation"].fillna(0.0).tolist()]
                text_values = [[("" if pd.isna(value) else f"{value:.2f}") for value in heatmap_df["correlation"].tolist()]]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z_values,
                        x=heatmap_df["feature"].tolist(),
                        y=[target_column],
                        text=text_values,
                        texttemplate="%{text}",
                        colorscale="RdBu",
                        zmid=0,
                        hovertemplate="Feature=%{x}<br>Correlation=%{z:.3f}<extra></extra>",
                    )
                )
                apply_plot_style(fig, height=320)
                st.plotly_chart(fig, use_container_width=True)
                render_dataframe(correlation_df.round(4), height=300)
            close_panel()

    with tab_time:
        render_panel_header("Целевая переменная во времени", "Выбирайте любую временную колонку. Усреднение по Y считается без нулей и пустых значений.")
        available_time_columns = [column for column in df.columns if column != target_column]
        if not available_time_columns:
            st.info("Нет доступных кандидатов для оси X.")
        else:
            render_control_box("Управление временным графиком", "Выберите временную колонку, группировку и диапазон лет.")
            selected_time_column = render_labeled_single_selector(
                title="Колонка по оси X",
                options=available_time_columns,
                selected=st.session_state.get("catboost_time_column", available_time_columns[0]),
                key_prefix="time_column_button",
            )
            st.session_state["catboost_time_column"] = selected_time_column
            selected_group_column = render_labeled_single_selector(
                title="Группировка",
                options=[column for column in df.columns if column != selected_time_column],
                selected=st.session_state.get("catboost_time_group", ""),
                key_prefix="time_group_button",
                empty_label="Без группировки",
            )
            st.session_state["catboost_time_group"] = selected_group_column
            available_years = sorted([int(year) for year in resolve_years(df[selected_time_column]).dropna().unique().tolist()])
            if available_years:
                time_range_cols = st.columns([1.2, 1.2, 0.8, 0.8])
                year_min = time_range_cols[0].slider(
                    "Год от",
                    min_value=available_years[0],
                    max_value=available_years[-1],
                    value=available_years[0],
                    key="catboost_time_year_min",
                )
                year_max = time_range_cols[1].slider(
                    "Год до",
                    min_value=available_years[0],
                    max_value=available_years[-1],
                    value=available_years[-1],
                    key="catboost_time_year_max",
                )
                time_range_cols[2].markdown(render_slider_metric("Старт", str(year_min)), unsafe_allow_html=True)
                time_range_cols[3].markdown(render_slider_metric("Финиш", str(year_max)), unsafe_allow_html=True)
            else:
                year_min = None
                year_max = None
                st.write("Годы не распознаны.")

            group_values: list[str] = []
            selected_groups: list[str] = []
            if selected_group_column:
                group_values = sorted(df[selected_group_column].astype("string").fillna("Пусто").unique().tolist())
                selected_groups = render_labeled_multi_selector(
                    title="Срезы",
                    options=group_values,
                    selected=st.session_state.get("catboost_time_group_values", group_values[: min(5, len(group_values))]),
                    key_prefix="time_group_value_button",
                )
                st.session_state["catboost_time_group_values"] = selected_groups
            close_control_box()

            time_summary, pivot_df = build_time_summary_frame(
                df,
                target_column,
                selected_time_column,
                selected_group_column or None,
                selected_groups,
                year_min,
                year_max,
            )
            if time_summary.empty:
                st.info("Для этого представления недостаточно валидных значений времени и target.")
            else:
                fig = px.line(
                    time_summary,
                    x="_year",
                    y="target_mean",
                    color="_group",
                    markers=True,
                    hover_data={"rows": True, "_year": True, "target_mean": ":.2f"},
                )
                fig.update_layout(xaxis_title="Год", yaxis_title=f"Среднее {target_column}")
                apply_plot_style(fig, height=420)
                st.plotly_chart(fig, use_container_width=True)
                render_inline_section_title("Сводная таблица по годам")
                render_dataframe(pivot_df, height=320)
        close_panel()

    with tab_scatter:
        render_panel_header("Диаграмма рассеяния", "Настраивайте оси, диапазоны и смотрите точки вместе с осреднением по окнам X с шагом 10%.")
        scatter_candidates = sorted(set(all_numeric_columns + all_datetime_columns))
        if len(scatter_candidates) < 2:
            st.info("Для scatter-анализа нужны как минимум две числовые или datetime-подобные колонки.")
        else:
            render_control_box("Управление scatter", "Выберите оси, раскраску и сузьте диапазоны для сравнения.")
            scatter_x = render_labeled_single_selector(
                title="Ось X",
                options=scatter_candidates,
                selected=st.session_state.get("catboost_scatter_x", scatter_candidates[0]),
                key_prefix="scatter_x_button",
            )
            st.session_state["catboost_scatter_x"] = scatter_x
            scatter_y_candidates = [column for column in scatter_candidates if column != scatter_x]
            scatter_y = render_labeled_single_selector(
                title="Ось Y",
                options=scatter_y_candidates,
                selected=st.session_state.get("catboost_scatter_y", scatter_y_candidates[0] if scatter_y_candidates else ""),
                key_prefix="scatter_y_button",
            )
            st.session_state["catboost_scatter_y"] = scatter_y
            color_options = [""] + [column for column in df.columns if column not in {scatter_x, scatter_y}]
            scatter_color = render_labeled_single_selector(
                title="Раскраска точек",
                options=[column for column in df.columns if column not in {scatter_x, scatter_y}],
                selected=st.session_state.get("catboost_scatter_color", ""),
                key_prefix="scatter_color_button",
                empty_label="Один цвет",
            )
            st.session_state["catboost_scatter_color"] = scatter_color

            scatter_df = build_scatter_frame(df, scatter_x, scatter_y, scatter_color or None)
            if scatter_df.empty:
                st.info("Для выбранных осей нет валидных строк scatter.")
            else:
                x_min_full = float(scatter_df["_x"].min())
                x_max_full = float(scatter_df["_x"].max())
                y_min_full = float(scatter_df["_y"].min())
                y_max_full = float(scatter_df["_y"].max())
                range_cols = st.columns(4)
                x_min_pct = range_cols[0].slider("Минимум X %", min_value=0, max_value=100, value=0, key="catboost_scatter_x_min_pct")
                x_max_pct = range_cols[1].slider("Максимум X %", min_value=0, max_value=100, value=100, key="catboost_scatter_x_max_pct")
                y_min_pct = range_cols[2].slider("Минимум Y %", min_value=0, max_value=100, value=0, key="catboost_scatter_y_min_pct")
                y_max_pct = range_cols[3].slider("Максимум Y %", min_value=0, max_value=100, value=100, key="catboost_scatter_y_max_pct")

                x_min_value = x_min_full + (x_max_full - x_min_full) * (x_min_pct / 100)
                x_max_value = x_min_full + (x_max_full - x_min_full) * (x_max_pct / 100)
                y_min_value = y_min_full + (y_max_full - y_min_full) * (y_min_pct / 100)
                y_max_value = y_min_full + (y_max_full - y_min_full) * (y_max_pct / 100)
                render_stat_strip(
                    [
                        ("X диапазон", f"{x_min_value:.2f} .. {x_max_value:.2f}"),
                        ("Y диапазон", f"{y_min_value:.2f} .. {y_max_value:.2f}"),
                    ]
                )
                close_control_box()
                filtered_scatter = filter_scatter_ranges(scatter_df, x_min_value, x_max_value, y_min_value, y_max_value)

                average_df = build_window_average_frame(filtered_scatter, scatter_y)
                scatter_fig = px.scatter(
                    filtered_scatter,
                    x="_x",
                    y="_y",
                    color="_color",
                    opacity=0.7,
                    hover_data={scatter_x: True, scatter_y: True},
                )
                if not average_df.empty:
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=average_df["avg_x"],
                            y=average_df["avg_y"],
                            mode="lines+markers+text",
                            name="Осреднение по окнам 10%",
                            text=[f"{x:.1f}, {y:.1f}" for x, y in zip(average_df["avg_x"], average_df["avg_y"], strict=False)],
                            textposition="top center",
                        )
                    )
                scatter_fig.update_layout(xaxis_title=scatter_x, yaxis_title=scatter_y)
                apply_plot_style(scatter_fig, height=430)
                st.plotly_chart(scatter_fig, use_container_width=True)

                avg_cols = st.columns([1.1, 1.3])
                with avg_cols[0]:
                    render_inline_section_title("Осреднённые точки")
                    if average_df.empty:
                        st.info("Недостаточно данных для расчёта 10% осреднения по окнам.")
                    else:
                        avg_fig = px.scatter(average_df, x="avg_x", y="avg_y", size="count", text="count")
                        avg_fig.update_traces(mode="markers+lines+text", textposition="top center")
                        apply_plot_style(avg_fig, height=340)
                        st.plotly_chart(avg_fig, use_container_width=True)
                with avg_cols[1]:
                    render_inline_section_title("Таблица осреднения")
                    if average_df.empty:
                        st.info("Таблица осреднения недоступна.")
                    else:
                        display_avg = average_df.rename(
                            columns={"window_label": "X window", "avg_x": "Average X", "avg_y": "Average Y", "count": "Points"}
                        )[["X window", "Average X", "Average Y", "Points"]].round(1)
                        render_dataframe(display_avg, height=320)

                render_inline_section_title("Тепловая карта")
                render_control_box("Управление heatmap", "Переключайтесь между плотностью точек и усреднением выбранной переменной.")
                heatmap_mode = render_labeled_single_selector(
                    title="Режим heatmap",
                    options=["__count__", "__mean__"],
                    selected=st.session_state.get("catboost_heatmap_mode", "__count__"),
                    key_prefix="heatmap_mode_button",
                )
                st.session_state["catboost_heatmap_mode"] = heatmap_mode
                heatmap_value_column = None
                if heatmap_mode == "__mean__":
                    heatmap_value_column = render_labeled_single_selector(
                        title="Переменная для среднего значения",
                        options=scatter_candidates,
                        selected=st.session_state.get("catboost_heatmap_value_column", scatter_candidates[0]),
                        key_prefix="heatmap_value_button",
                    )
                    st.session_state["catboost_heatmap_value_column"] = heatmap_value_column
                close_control_box()
                heatmap_df = build_heatmap_frame(filtered_scatter, heatmap_mode, heatmap_value_column)
                if heatmap_df.empty:
                    st.info("Для текущего scatter-выбора heatmap построить не удалось.")
                else:
                    pivot = heatmap_df.pivot(index="_y_label", columns="_x_label", values="value")
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            text=np.round(pivot.values, 1),
                            texttemplate="%{text}",
                            colorscale="YlOrRd",
                        )
                    )
                    fig.update_layout(xaxis_title=scatter_x, yaxis_title=scatter_y)
                    apply_plot_style(fig, height=380)
                    st.plotly_chart(fig, use_container_width=True)
        close_panel()

    with tab_hist:
        render_panel_header("Гистограммы", "Для любой переменной можно быстро переключать срезы и собирать выбранные категории вместе, как в исходном модуле.")
        render_control_box("Управление гистограммой", "Выберите переменную, тип среза и способ агрегации выбранных категорий.")
        histogram_candidates = df.columns.tolist()
        hist_column = render_labeled_single_selector(
            title="Переменная",
            options=histogram_candidates,
            selected=st.session_state.get("catboost_hist_column", histogram_candidates[0] if len(histogram_candidates) else ""),
            key_prefix="hist_column_button",
        )
        st.session_state["catboost_hist_column"] = hist_column
        hist_group_options = [column for column in df.columns if column != hist_column] + ["__year__"]
        hist_group = render_labeled_single_selector(
            title="Срезы",
            options=hist_group_options,
            selected=st.session_state.get("catboost_hist_group", ""),
            key_prefix="hist_group_button",
            empty_label="Без срезов",
        )
        st.session_state["catboost_hist_group"] = hist_group
        aggregate_mode = render_labeled_single_selector(
            title="Режим отображения",
            options=["Объединить выбранные", "Показать по отдельности"],
            selected=st.session_state.get("catboost_hist_aggregate_mode", "Объединить выбранные"),
            key_prefix="hist_aggregate_mode_button",
        )
        st.session_state["catboost_hist_aggregate_mode"] = aggregate_mode
        aggregate_selected = aggregate_mode == "Объединить выбранные"
        st.session_state["catboost_hist_aggregate"] = aggregate_selected

        hist_group_values: list[str] = []
        selected_hist_values: list[str] = []
        if hist_group:
            if hist_group == "__year__":
                year_values = sorted([str(int(year)) for year in resolve_years(df[hist_column]).dropna().unique().tolist()])
                hist_group_values = year_values
            else:
                hist_group_values = sorted(df[hist_group].astype("string").fillna("Пусто").unique().tolist())
            selected_hist_values = render_labeled_multi_selector(
                title="Значения срезов",
                options=hist_group_values,
                selected=st.session_state.get("catboost_hist_group_values", hist_group_values[: min(5, len(hist_group_values))]),
                key_prefix="hist_group_value_button",
            )
            st.session_state["catboost_hist_group_values"] = selected_hist_values
        close_control_box()

        hist_df = build_histogram_frame(df, hist_column, hist_group or None, selected_hist_values, aggregate_selected)
        if hist_df.empty:
            st.info("Для этой гистограммы нет валидных строк.")
        else:
            fig = px.histogram(
                hist_df,
                x="_hist_value",
                color="_group",
                opacity=0.6,
                barmode="overlay" if aggregate_selected else "group",
            )
            fig.update_layout(xaxis_title=hist_column, yaxis_title="Строки")
            apply_plot_style(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
        close_panel()

    with tab_forecast:
        render_panel_header("Настройка прогноза", "Используем текущий датасет и CatBoost для оценки риска отказов и прогноза целевой переменной.")
        render_control_box("Параметры обучения", "Настройте размер тестовой выборки и случайное зерно перед запуском анализа.")
        train_cols = st.columns([1.2, 1.2, 0.8, 0.8])
        test_fraction = train_cols[0].slider("Тестовая выборка", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="catboost_test_fraction")
        random_seed = int(train_cols[1].number_input("Случайное зерно", min_value=1, value=42, step=1, key="catboost_random_seed"))
        train_cols[2].markdown(render_slider_metric("Test", f"{test_fraction:.0%}"), unsafe_allow_html=True)
        train_cols[3].markdown(render_slider_metric("Seed", str(random_seed)), unsafe_allow_html=True)
        close_control_box()

        auto_train_needed = (
            st.session_state.get("catboost_trained_model") is None
            and bool(target_column)
            and bool(feature_columns)
        )
        if (st.button("Запустить аналитику CatBoost", type="primary", use_container_width=True) or auto_train_needed) and target_column and feature_columns:
            trained = train_forecast_model(df, target_column, feature_columns, test_fraction, random_seed)
            st.session_state["catboost_trained_model"] = trained
            st.session_state["catboost_trained_signature"] = st.session_state.get("catboost_expected_signature")

        trained: TrainedForecast | None = st.session_state.get("catboost_trained_model")
        if trained is None:
            st.info("Запустите локальную модель CatBoost, чтобы увидеть метрики, прогноз и номограмму.")
            close_panel()
        else:
            metric_cols = st.columns(5)
            forecast_specs = [
                ("Строк train", f"{trained.metrics['train_rows']:,}", "Обучающая выборка"),
                ("Строк test", f"{trained.metrics['test_rows']:,}", "Тестовая выборка"),
                ("RMSE", format_metric(trained.metrics["rmse"]), None),
                ("MAE", format_metric(trained.metrics["mae"]), None),
                ("R²", format_metric(trained.metrics["r2"], digits=3), None),
            ]
            for column, (label, value, meta) in zip(metric_cols, forecast_specs, strict=False):
                column.markdown(render_metric_card(label, value, meta), unsafe_allow_html=True)

            actual_pred_df = trained.prepared.copy()
            actual_pred_df["Actual"] = trained.actual.loc[trained.prepared.index].to_numpy(dtype=float)
            actual_pred_df["Predicted"] = trained.model.predict(trained.prepared)
            compare_cols = st.columns([1.2, 1])
            with compare_cols[0]:
                fig = px.scatter(actual_pred_df, x="Actual", y="Predicted", opacity=0.7)
                diagonal_min = min(float(actual_pred_df["Actual"].min()), float(actual_pred_df["Predicted"].min()))
                diagonal_max = max(float(actual_pred_df["Actual"].max()), float(actual_pred_df["Predicted"].max()))
                fig.add_trace(
                    go.Scatter(
                        x=[diagonal_min, diagonal_max],
                        y=[diagonal_min, diagonal_max],
                        mode="lines",
                        name="Идеальное совпадение",
                    )
                )
                fig.update_layout(xaxis_title="Факт", yaxis_title="Прогноз")
                apply_plot_style(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)
            with compare_cols[1]:
                render_inline_section_title("Предпросмотр прогноза")
                display_predictions = actual_pred_df[["Actual", "Predicted"]].copy().round(2).head(200)
                display_predictions = display_predictions.rename(columns={"Actual": "Факт", "Predicted": "Прогноз"})
                render_dataframe(display_predictions, height=320)

            close_panel()
            render_panel_header("Сценарная строка", "Отредактируйте базовый набор признаков и сразу оцените прогноз для одного сценария.")
            render_stat_strip(
                [
                    ("Числовых признаков", str(len(trained.numeric_feature_columns))),
                    ("Категориальных признаков", str(len(trained.categorical_columns))),
                    ("Всего признаков", str(len(trained.feature_columns))),
                ]
            )
            baseline_row = pd.DataFrame([trained.baseline_values])
            scenario_row = st.data_editor(
                _streamlit_safe_dataframe(baseline_row),
                num_rows="fixed",
                use_container_width=True,
                key="catboost_scenario_editor",
            )
            scenario_prediction = predict_forecast_rows(trained, scenario_row.to_dict(orient="records"))
            if scenario_prediction:
                st.markdown(
                    render_metric_card("Сценарный прогноз", format_metric(scenario_prediction[0]), "Оценка модели для текущей сценарной строки."),
                    unsafe_allow_html=True,
                )
            close_panel()

            render_panel_header("Номограмма", "Двухфакторная карта прогноза с фиксированным срезом по дополнительному признаку.")
            render_control_box("Управление номограммой", "Выберите две числовые оси и при необходимости задайте фиксированный срез по третьему признаку.")
            contour_candidates = trained.numeric_feature_columns
            if len(contour_candidates) < 2:
                st.info("Для номограммы нужны как минимум два числовых признака.")
            else:
                contour_x = render_labeled_single_selector(
                    title="Ось X",
                    options=contour_candidates,
                    selected=st.session_state.get("catboost_contour_x", contour_candidates[0]),
                    key_prefix="contour_x_button",
                )
                st.session_state["catboost_contour_x"] = contour_x
                contour_y = render_labeled_single_selector(
                    title="Ось Y",
                    options=[column for column in contour_candidates if column != contour_x],
                    selected=st.session_state.get("catboost_contour_y", next((column for column in contour_candidates if column != contour_x), "")),
                    key_prefix="contour_y_button",
                )
                st.session_state["catboost_contour_y"] = contour_y
                slice_feature_options = [""] + [column for column in trained.feature_columns if column not in {contour_x, contour_y}]
                slice_feature = render_labeled_single_selector(
                    title="Срез",
                    options=[column for column in trained.feature_columns if column not in {contour_x, contour_y}],
                    selected=st.session_state.get("catboost_contour_slice_feature", ""),
                    key_prefix="contour_slice_button",
                    empty_label="Без среза",
                )
                st.session_state["catboost_contour_slice_feature"] = slice_feature
                slice_value = ""
                if slice_feature:
                    default_value = trained.baseline_values.get(slice_feature, "")
                    slice_value = st.text_input("Значение среза", value=str(default_value), key="catboost_contour_slice_value")
                slice_overrides: dict[str, str | float] = {}
                if slice_feature:
                    if slice_feature in trained.numeric_feature_columns:
                        numeric_value = coerce_numeric_series(pd.Series([slice_value])).iloc[0]
                        if pd.notna(numeric_value):
                            slice_overrides[slice_feature] = float(numeric_value)
                    else:
                        slice_overrides[slice_feature] = str(slice_value)
                contour_df = build_forecast_contour(trained, contour_x, contour_y, slice_overrides)
                if contour_df.empty:
                    st.info("Для выбранных признаков номограмму построить не удалось.")
                else:
                    contour_pivot = contour_df.pivot(index=contour_y, columns=contour_x, values="prediction")
                    fig = go.Figure(
                        data=go.Contour(
                            z=contour_pivot.values,
                            x=contour_pivot.columns.to_numpy(dtype=float),
                            y=contour_pivot.index.to_numpy(dtype=float),
                            colorscale="Viridis",
                            contours=dict(showlabels=True),
                            colorbar=dict(title="Прогноз"),
                        )
                    )
                    fig.update_layout(xaxis_title=contour_x, yaxis_title=contour_y)
                    apply_plot_style(fig, height=430)
                    st.plotly_chart(fig, use_container_width=True)
                    render_dataframe(contour_df.head(300).round(3), height=280)
            close_control_box()
            close_panel()


if __name__ == "__main__":
    main()
