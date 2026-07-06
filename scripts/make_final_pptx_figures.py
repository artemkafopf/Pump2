"""
Generate static PNG figures for docs/Vt_frequency_analysis_final_ru.pptx.

Outputs:  analysis_outputs/final_pptx_figures/
  trf_vs_ttf_by_category.png      — ННО vs Накопленная частота, по категориям (без Прочее/<missing>)
  trf_vs_ttf_by_h2s.png           — то же, Кислый/Некислый
  trf_vs_ttf_by_h2s_proxy.png     — то же, H2S-прокси 4 бина (<missing>/<1/1-20/>20 мг/л)
  trf_vs_ttf_by_precipitate.png   — то же, прокси осадков
  trf_vs_ttf_by_glf.png           — то же, газовый фактор
  glf_vs_nno_boxplot.png          — ГФ-бины vs ННО (ящики)
  hf_runs_km_global.png           — KM: высокочастотные прогоны vs базис (Global)
  hf_runs_category_mix.png        — структура отказов: >55 Гц vs базис
  cycling_near60_vs_baseline.png  — цикличность near-60 vs базис

Usage:
    python -X utf8 scripts/make_final_pptx_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analysis.paths import results_dir

REPO = Path(__file__).resolve().parents[1]
_SLUG = "final_pptx_figures"
OUT_DIR = results_dir(_SLUG)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

try:
    from scripts.db import WAREHOUSE_PATH, get_warehouse_conn
except ImportError:
    WAREHOUSE_PATH = REPO / "data" / "warehouse" / "pump2.db"
    import sqlite3
    def get_warehouse_conn():
        return sqlite3.connect(str(WAREHOUSE_PATH))

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
})

# ── palettes ───────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "КЛ (R-0)":      "#1f77b4",   # blue
    "ПЭД (R-0)":     "#d62728",   # red
    "Слом вала":     "#2ca02c",   # green
    "Засорение РО":  "#ff7f0e",   # orange
    "НКТ":           "#9467bd",   # purple
    "Износ РО":      "#17becf",   # teal
    "Гидрозащита":   "#e377c2",   # pink
}
# Only these categories are shown in scatter (no missing/Прочее)
CATEGORY_ORDER = list(CATEGORY_COLORS.keys())

H2S_BIN_COLORS = {
    "<нет данных>": "#AAAAAA",
    "<1 мг/л":      "#2ca02c",
    "1–20 мг/л":    "#ff7f0e",
    ">20 мг/л":     "#d62728",
}
H2S_BIN_ORDER = ["<нет данных>", "<1 мг/л", "1–20 мг/л", ">20 мг/л"]

H2S_BINARY_COLORS = {
    "Кислый":    "#d62728",
    "Некислый":  "#2ca02c",
    "<нет данных>": "#AAAAAA",
}

GLF_BIN_COLORS = {
    "ГФ ≤100":    "#1f77b4",
    "ГФ 100–300": "#2ca02c",
    "ГФ 300–700": "#ff7f0e",
    "ГФ >700":    "#d62728",
}
GLF_BIN_ORDER = list(GLF_BIN_COLORS.keys())

PRECIP_PALETTE = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

HZ_SPECS = [
    (30, "#c0392b", ":",  1.2, "30 Гц"),
    (40, "#e67e00", "--", 1.5, "40 Гц"),
    (50, "#2ca02c", "--", 1.8, "50 Гц"),
    (60, "#1a6eb0", "-",  2.2, "60 Гц"),
    (70, "#c0392b", ":",  1.2, "70 Гц"),
]

OUT_OF_RANGE_COLOR = "#FF9999"
OUT_OF_RANGE_ALPHA = 0.28

NNO_LABEL   = "ННО — наработка на отказ (дней)"
НАКЧ_LABEL  = "Накопленная частота (Гц·дней)"

# ── helpers ────────────────────────────────────────────────────────────────────

def classify_node(val) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return None  # will be filtered
    n = str(val).strip().lower()
    if "кабел" in n or "тмс" in n:
        return "КЛ (R-0)"
    if "пэд" in n:
        return "ПЭД (R-0)"
    if "гидрозащита" in n:
        return "Гидрозащита"
    if any(w in n for w in ("эцн", "газосепар", "диспергатор", "входной модуль")):
        return "Засорение РО"
    if "нкт" in n:
        return "НКТ"
    if "вал" in n:
        return "Слом вала"
    return None  # Прочее — exclude

def classify_h2s_proxy(val) -> str:
    if pd.isna(val):
        return "<нет данных>"
    v = float(val)
    if v < 1:
        return "<1 мг/л"
    if v < 20:
        return "1–20 мг/л"
    return ">20 мг/л"

def classify_h2s_binary(val, threshold: float = 3.0) -> str:
    if pd.isna(val):
        return "<нет данных>"
    return "Кислый" if float(val) >= threshold else "Некислый"

def classify_glf(val) -> str | None:
    if pd.isna(val) or float(val) <= 0:
        return None
    v = float(val)
    if v <= 100:
        return "ГФ ≤100"
    if v <= 300:
        return "ГФ 100–300"
    if v <= 700:
        return "ГФ 300–700"
    return "ГФ >700"

def add_hz_reflines(ax, ttf_max: float, trf_max: float) -> list:
    # Shade out-of-range zones (below 30 Hz and above 70 Hz) before drawing lines
    x_edge = np.array([0.0, ttf_max * 1.1])
    zone_kw = dict(color=OUT_OF_RANGE_COLOR, alpha=OUT_OF_RANGE_ALPHA,
                   zorder=0, linewidth=0)
    ax.fill_between(x_edge, 0, 30 * x_edge, **zone_kw)                   # below 30 Hz
    ax.fill_between(x_edge, 70 * x_edge, trf_max * 1.1, **zone_kw)       # above 70 Hz

    ttf_vals = np.linspace(1, ttf_max, 300)
    handles = []
    for hz, color, ls, lw, label in HZ_SPECS:
        trf_line = hz * ttf_vals
        mask = trf_line <= trf_max * 1.08
        ax.plot(ttf_vals[mask], trf_line[mask],
                color=color, ls=ls, lw=lw, alpha=0.85, zorder=1)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw,
                               label=label, alpha=0.85))
    # Zone indicator in legend
    handles.append(mpatches.Patch(color=OUT_OF_RANGE_COLOR,
                                  alpha=min(OUT_OF_RANGE_ALPHA * 2, 0.9),
                                  label="<30 / >70 Гц (аномалия)"))
    return handles

def scatter_colored(ax, sub: pd.DataFrame, color_col: str,
                    color_map: dict, order: list,
                    alpha: float = 0.5, s: int = 16) -> list:
    handles = []
    for cat in order:
        m = sub[color_col].astype(str) == str(cat)
        if not m.any():
            continue
        fc = color_map.get(cat, "#AAAAAA")
        ax.scatter(sub.loc[m, "ttf"], sub.loc[m, "trf"],
                   c=fc, s=s, alpha=alpha, edgecolors="none",
                   zorder=2, rasterized=True)
        handles.append(mpatches.Patch(color=fc, label=f"{cat} (n={m.sum()})"))
    return handles

def base_scatter_axes(ax, df_plot: pd.DataFrame, extra_pct: float = 1.05):
    ttf_max = float(df_plot["ttf"].quantile(0.98))
    trf_max = float(df_plot["trf"].quantile(0.98))
    ax.set_xlim(0, ttf_max * extra_pct)
    ax.set_ylim(0, trf_max * extra_pct)
    ax.set_xlabel(NNO_LABEL, fontsize=12)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=12)
    return ttf_max, trf_max

def kaplan_meier(t: np.ndarray, e: np.ndarray):
    order = np.argsort(t)
    t, e = t[order], e[order]
    event_times = np.unique(t[e == 1])
    if len(event_times) == 0:
        return np.array([0.0]), np.array([1.0])
    times = np.concatenate([[0.0], event_times])
    surv = np.ones(len(times))
    cur = 1.0
    for i, et in enumerate(event_times, 1):
        at_risk = int(np.sum(t >= et))
        n_ev = int(np.sum((t == et) & (e == 1)))
        if at_risk > 0:
            cur *= 1.0 - n_ev / at_risk
        surv[i] = cur
    return times, surv

# ── load & prepare data ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame | None:
    if not WAREHOUSE_PATH.exists():
        print(f"[ERROR] Warehouse not found: {WAREHOUSE_PATH}")
        return None
    try:
        conn = get_warehouse_conn()
        df = pd.read_sql("SELECT * FROM mart__vt_freq55", conn)
        conn.close()
        print(f"Загружено mart__vt_freq55: {len(df)} строк")
        return df
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ttf"] = df["run_days"]
    trf_col = next((c for c in ["total_freq_hz_days"] if c in df.columns), None)
    df["trf"] = df[trf_col] if trf_col else np.nan
    df["cat"] = df["failed_node"].apply(classify_node) if "failed_node" in df.columns else None
    df["h2s_proxy_bin"] = df["h2s_proxy_mg_l"].apply(classify_h2s_proxy) if "h2s_proxy_mg_l" in df.columns else "<нет данных>"
    df["h2s_binary"] = df["h2s_proxy_mg_l"].apply(classify_h2s_binary) if "h2s_proxy_mg_l" in df.columns else "<нет данных>"
    df["glf_bin"] = df["avg_glf"].apply(classify_glf) if "avg_glf" in df.columns else None
    salt_col = next((c for c in ["cum_salt_load_kg", "integrated_salt_proxy_m"] if c in df.columns), None)
    df["salt"] = df[salt_col] if salt_col else np.nan
    df["freq_mean"] = df.get("freq_w_mean", pd.Series(np.nan, index=df.index))
    df["freq_share55"] = df.get("freq_above_55hz_pct", pd.Series(np.nan, index=df.index))
    df["n_below_45"] = df.get("n_freq_below_45hz", pd.Series(np.nan, index=df.index))
    df["n_valid"] = df.get("n_freq_valid_days", pd.Series(np.nan, index=df.index))
    kpod = pd.to_numeric(df.get("kpod_m_mean", np.nan), errors="coerce")
    if "avg_kpod" in df.columns:
        kpod = kpod.fillna(pd.to_numeric(df["avg_kpod"], errors="coerce"))
    df["kpod"] = kpod
    df["kpod_source"] = np.where(
        pd.to_numeric(df.get("kpod_m_mean", np.nan), errors="coerce").notna(), "mature",
        np.where(pd.to_numeric(df.get("avg_kpod", np.nan), errors="coerce").notna(), "whole_run", None)
    )
    return df

def trf_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["trf"].notna() & df["ttf"].notna() & (df["trf"] > 0)].copy()

# ── Figure 1: ННО vs Накопленная частота — категории ──────────────────────────

def fig_trf_category(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["cat"].notna()]  # remove missing / Прочее
    if sub.empty:
        print("  [skip] нет данных для категорий")
        return

    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50          # 50 Hz line hits the top-right corner → 45° with set_aspect

    fig, ax = plt.subplots(figsize=(9, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)
    cat_h = scatter_colored(ax, sub, "cat", CATEGORY_COLORS, CATEGORY_ORDER, alpha=0.55, s=17)
    ax.set_xlim(0, xlim_max * 1.04); ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title("ННО vs Накопленная частота — категории отказов\n"
                 "Диагональные линии: средняя рабочая частота", fontsize=14, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=cat_h, loc="lower right", title="Категория отказа",
              fontsize=8.5, title_fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_category.png")

# ── Figure 2: ННО vs Накопленная частота — H₂S бинарный ──────────────────────

def fig_trf_h2s_binary(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50

    fig, ax = plt.subplots(figsize=(9, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)
    h_order = ["Некислый", "Кислый", "<нет данных>"]
    h2s_h = scatter_colored(ax, sub, "h2s_binary", H2S_BINARY_COLORS, h_order, alpha=0.5, s=17)
    ax.set_xlim(0, xlim_max * 1.04); ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title("ННО vs Накопленная частота — кислотность H₂S (порог 3 мг/л)", fontsize=14, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=h2s_h, loc="lower right", title="Класс H₂S",
              fontsize=10, title_fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_h2s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_h2s.png")

# ── Figure 3: ННО vs Накопленная частота — H₂S прокси 4 бина ─────────────────

def fig_trf_h2s_proxy(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50

    fig, ax = plt.subplots(figsize=(9, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)
    h2s_h = scatter_colored(ax, sub, "h2s_proxy_bin", H2S_BIN_COLORS, H2S_BIN_ORDER,
                             alpha=0.5, s=17)
    ax.set_xlim(0, xlim_max * 1.04); ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title("ННО vs Накопленная частота — H₂S прокси (мг/л)\n"
                 "Зелёный: некислый (<1); Оранжевый: умеренный (1–20); Красный: высокий (>20)", fontsize=13, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=h2s_h, loc="lower right", title="H₂S прокси",
              fontsize=10, title_fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_h2s_proxy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_h2s_proxy.png")

# ── Figure 4: ННО vs Накопленная частота — прокси осадков ────────────────────

def fig_trf_precipitate(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["salt"].notna() & (sub["salt"] > 0)]
    if sub.empty:
        print("  [skip] нет данных об осадках")
        return

    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50

    log_salt = np.log10(sub["salt"].clip(lower=1))
    bins = np.quantile(log_salt, np.linspace(0, 1, len(PRECIP_PALETTE) + 1))
    sub = sub.copy()
    sub["salt_bin"] = pd.cut(log_salt, bins=bins, labels=range(len(PRECIP_PALETTE)),
                              include_lowest=True).astype("Int64")

    fig, ax = plt.subplots(figsize=(9, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)
    edges_10 = 10.0 ** bins
    p_h = []
    for i, color in enumerate(PRECIP_PALETTE):
        m = sub["salt_bin"] == i
        if not m.any():
            continue
        ax.scatter(sub.loc[m, "ttf"], sub.loc[m, "trf"],
                   c=color, s=17, alpha=0.55, edgecolors="none", zorder=2, rasterized=True)
        lo, hi = edges_10[i], edges_10[i + 1]
        p_h.append(mpatches.Patch(color=color,
                                   label=f"{lo:.0f}–{hi:.0f} кг (n={m.sum()})"))
    ax.set_xlim(0, xlim_max * 1.04); ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title("ННО vs Накопленная частота — прокси осадков (накопленная соль, кг)\n"
                 "Тёмно-красный: высокая осадкообразующая нагрузка", fontsize=13, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=p_h, loc="lower right", title="Прокси осадков (кг)",
              fontsize=9, title_fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_precipitate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_precipitate.png")

# ── Figure 5: ННО vs Накопленная частота — ГФ (газовый фактор) ───────────────

def fig_trf_glf(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["glf_bin"].notna()]
    if sub.empty:
        print("  [skip] нет данных ГФ")
        return

    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50

    fig, ax = plt.subplots(figsize=(9, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)
    glf_h = scatter_colored(ax, sub, "glf_bin", GLF_BIN_COLORS, GLF_BIN_ORDER,
                             alpha=0.55, s=17)
    ax.set_xlim(0, xlim_max * 1.04); ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title("ННО vs Накопленная частота — газовый фактор (ГФ)\n"
                 "Синий: низкий ГФ; Красный: высокий ГФ", fontsize=14, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=glf_h, loc="lower right", title="Газовый фактор (м³/м³)",
              fontsize=9.5, title_fontsize=9.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_glf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_glf.png")

# ── Figure 6b: ННО vs Накопленная частота — Кпод ─────────────────────────────

KPOD_BINS = [
    (None,  0.50, "#d62728", "<0.50"),
    (0.50,  0.70, "#ff7f0e", "0.50–0.70"),
    (0.70,  1.00, "#2ca02c", "0.70–1.00"),
    (1.00,  1.20, "#1f77b4", "1.00–1.20"),
    (1.20,  None, "#9467bd", ">1.20"),
]
KPOD_NO_DATA_COLOR = "#CCCCCC"

def _classify_kpod(val) -> str:
    if pd.isna(val):
        return "<нет данных>"
    v = float(val)
    if v < 0.50:
        return "<0.50"
    if v < 0.70:
        return "0.50–0.70"
    if v < 1.00:
        return "0.70–1.00"
    if v < 1.20:
        return "1.00–1.20"
    return ">1.20"

def fig_trf_kpod(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    if sub.empty:
        print("  [skip] нет данных для Кпод")
        return

    sub = sub.copy()
    sub["kpod_bin"] = sub["kpod"].apply(_classify_kpod)

    trf_max = float(sub["trf"].quantile(0.98))
    xlim_max = trf_max / 50

    fig, ax = plt.subplots(figsize=(10, 9))
    hz_h = add_hz_reflines(ax, xlim_max, trf_max)

    handles = []
    # No-data points first (bottom layer)
    no_data = sub[sub["kpod_bin"] == "<нет данных>"]
    if not no_data.empty:
        ax.scatter(no_data["ttf"], no_data["trf"],
                   c=KPOD_NO_DATA_COLOR, s=10, alpha=0.30, edgecolors="none",
                   zorder=2, rasterized=True)
        handles.append(mpatches.Patch(color=KPOD_NO_DATA_COLOR,
                                      label=f"нет данных Кпод (n={len(no_data)})"))

    for lo, hi, color, label in KPOD_BINS:
        grp = sub[sub["kpod_bin"] == label]
        if grp.empty:
            continue
        ax.scatter(grp["ttf"], grp["trf"],
                   c=color, s=18, alpha=0.65, edgecolors="none",
                   zorder=3, rasterized=True)
        handles.append(mpatches.Patch(color=color, label=f"{label} (n={len(grp)})"))

    ax.set_xlim(0, xlim_max * 1.04)
    ax.set_ylim(0, trf_max * 1.04)
    ax.set_aspect(1 / 50, adjustable="box")
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    n_total = len(sub)
    n_kpod = int((sub["kpod_bin"] != "<нет данных>").sum())
    n_mature = int((sub.get("kpod_source", pd.Series()) == "mature").sum()) if "kpod_source" in sub.columns else "?"
    n_whole  = int((sub.get("kpod_source", pd.Series()) == "whole_run").sum()) if "kpod_source" in sub.columns else "?"
    ax.set_title(
        f"ННО vs Накопленная частота: коэффициент подачи (Кпод)\n"
        f"n={n_total}  с Кпод: {n_kpod}  (зрелое окно: {n_mature}, весь прогон: {n_whole})  без данных: {n_total - n_kpod}",
        fontsize=13, pad=10,
    )
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=handles, loc="lower right", title="Кпод (коэф. подачи)",
              fontsize=9, title_fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_kpod.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_kpod.png")

def fig_trf_kpod_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    if sub.empty:
        return
    sub = sub.copy()
    sub["kpod_bin"] = sub["kpod"].apply(_classify_kpod)

    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)

    handles = []
    no_data = sub[sub["kpod_bin"] == "<нет данных>"]
    if not no_data.empty:
        ax.scatter(no_data["ttf"], no_data["trf"],
                   c=KPOD_NO_DATA_COLOR, s=10, alpha=0.25, edgecolors="none",
                   zorder=2, rasterized=True)
        handles.append(mpatches.Patch(color=KPOD_NO_DATA_COLOR,
                                      label=f"нет данных Кпод (n={len(no_data)})"))

    for lo, hi, color, label in KPOD_BINS:
        grp = sub[sub["kpod_bin"] == label]
        if grp.empty:
            continue
        ax.scatter(grp["ttf"], grp["trf"],
                   c=color, s=18, alpha=0.65, edgecolors="none",
                   zorder=3, rasterized=True)
        handles.append(mpatches.Patch(color=color, label=f"{label} (n={len(grp)})"))

    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(
        _zoom_suffix("ННО vs Накопленная частота — коэффициент подачи (Кпод)"),
        fontsize=12, pad=10,
    )
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=handles, loc="lower right", title="Кпод (коэф. подачи)",
              fontsize=9, title_fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_kpod_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_kpod_zoom.png")

# ── Zoom variants (ylim 0–20k, 50 Hz at 45°) ─────────────────────────────────
#
# For the 50 Hz diagonal to appear at exactly 45° we need:
#   pixel_height_per_y_unit / pixel_width_per_x_unit = 1/50
# → ax.set_aspect(1/50, adjustable='box') enforces this regardless of figure shape.
# With ylim=20000 the natural 45°-consistent xlim is 20000/50 = 400 days.
# We add 5 % right padding → ZOOM_XLIM = 420.

ZOOM_YLIM = 20_000.0
ZOOM_XLIM = ZOOM_YLIM / 50 * 1.05   # 420 days

def _apply_zoom(ax) -> None:
    ax.set_xlim(0, ZOOM_XLIM)
    ax.set_ylim(0, ZOOM_YLIM)
    ax.set_aspect(1 / 50, adjustable="box")

def _zoom_suffix(title: str) -> str:
    return title + "\n(zoom: ННО 0–420 дн., накопл. частота 0–20 000 Гц·дн., 50 Гц = 45°)"

def fig_trf_category_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["cat"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)
    cat_h = scatter_colored(ax, sub, "cat", CATEGORY_COLORS, CATEGORY_ORDER, alpha=0.55, s=17)
    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(_zoom_suffix("ННО vs Накопленная частота — категории отказов"), fontsize=12, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=cat_h, loc="lower right", title="Категория отказа",
              fontsize=8.5, title_fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_category_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_category_zoom.png")

def fig_trf_h2s_binary_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)
    h_order = ["Некислый", "Кислый", "<нет данных>"]
    h2s_h = scatter_colored(ax, sub, "h2s_binary", H2S_BINARY_COLORS, h_order, alpha=0.5, s=17)
    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(_zoom_suffix("ННО vs Накопленная частота — кислотность H₂S"), fontsize=12, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=h2s_h, loc="lower right", title="Класс H₂S",
              fontsize=10, title_fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_h2s_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_h2s_zoom.png")

def fig_trf_h2s_proxy_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)
    h2s_h = scatter_colored(ax, sub, "h2s_proxy_bin", H2S_BIN_COLORS, H2S_BIN_ORDER,
                             alpha=0.5, s=17)
    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(_zoom_suffix("ННО vs Накопленная частота — H₂S прокси (мг/л)"), fontsize=12, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=h2s_h, loc="lower right", title="H₂S прокси",
              fontsize=10, title_fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_h2s_proxy_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_h2s_proxy_zoom.png")

def fig_trf_precipitate_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["salt"].notna() & (sub["salt"] > 0)]
    if sub.empty:
        return
    log_salt = np.log10(sub["salt"].clip(lower=1))
    bins = np.quantile(log_salt, np.linspace(0, 1, len(PRECIP_PALETTE) + 1))
    sub = sub.copy()
    sub["salt_bin"] = pd.cut(log_salt, bins=bins, labels=range(len(PRECIP_PALETTE)),
                              include_lowest=True).astype("Int64")
    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)
    edges_10 = 10.0 ** bins
    p_h = []
    for i, color in enumerate(PRECIP_PALETTE):
        m = sub["salt_bin"] == i
        if not m.any():
            continue
        ax.scatter(sub.loc[m, "ttf"], sub.loc[m, "trf"],
                   c=color, s=17, alpha=0.55, edgecolors="none", zorder=2, rasterized=True)
        lo, hi = edges_10[i], edges_10[i + 1]
        p_h.append(mpatches.Patch(color=color,
                                   label=f"{lo:.0f}–{hi:.0f} кг (n={m.sum()})"))
    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(_zoom_suffix("ННО vs Накопленная частота — прокси осадков"), fontsize=12, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=p_h, loc="lower right", title="Прокси осадков (кг)",
              fontsize=9, title_fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_precipitate_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_precipitate_zoom.png")

def fig_trf_glf_zoom(df: pd.DataFrame) -> None:
    sub = trf_subset(df)
    sub = sub[sub["glf_bin"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    hz_h = add_hz_reflines(ax, ZOOM_XLIM, ZOOM_YLIM)
    glf_h = scatter_colored(ax, sub, "glf_bin", GLF_BIN_COLORS, GLF_BIN_ORDER,
                             alpha=0.55, s=17)
    _apply_zoom(ax)
    ax.set_xlabel(NNO_LABEL, fontsize=13)
    ax.set_ylabel(НАКЧ_LABEL, fontsize=13)
    ax.set_title(_zoom_suffix("ННО vs Накопленная частота — газовый фактор (ГФ)"), fontsize=12, pad=10)
    l1 = ax.legend(handles=hz_h, loc="upper left", title="Частота", fontsize=9, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=glf_h, loc="lower right", title="Газовый фактор (м³/м³)",
              fontsize=9.5, title_fontsize=9.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trf_vs_ttf_by_glf_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: trf_vs_ttf_by_glf_zoom.png")

# ── Figure 6: ГФ-бины vs ННО — ящики + KM ─────────────────────────────────────

def fig_glf_vs_nno(df: pd.DataFrame) -> None:
    sub = df[df["glf_bin"].notna() & df["ttf"].notna() & (df["ttf"] > 0)].copy()
    if sub.empty:
        print("  [skip] нет данных ГФ для boxplot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Газовый фактор (ГФ): влияние на ННО и накопленную частоту",
                 fontsize=14, y=1.01)

    # Panel A: ящики ГФ vs ННО
    ax = axes[0]
    order = [g for g in GLF_BIN_ORDER if g in sub["glf_bin"].values]
    data_a = [sub.loc[sub["glf_bin"] == g, "ttf"].dropna().values for g in order]
    colors_a = [GLF_BIN_COLORS[g] for g in order]
    bp = ax.boxplot(data_a, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="white", linewidth=2.0))
    for patch, color in zip(bp["boxes"], colors_a):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel(NNO_LABEL, fontsize=11)
    ax.set_title("А. Распределение ННО по бинам ГФ", fontsize=11)
    # Аннотации медиан
    for i, (g, d) in enumerate(zip(order, data_a), 1):
        med = np.median(d) if len(d) > 0 else 0
        ax.text(i, med + 5, f"{med:.0f}", ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")
        n = len(d)
        ax.text(i, ax.get_ylim()[0] + 2, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")

    # Panel B: KM кривые по ГФ-бинам
    ax = axes[1]
    km_colors = list(GLF_BIN_COLORS.values())
    max_t = float(sub["ttf"].quantile(0.97))
    for i, g in enumerate(order):
        sg = sub[sub["glf_bin"] == g]
        t = sg["ttf"].to_numpy(float)
        e = sg["event"].to_numpy(float)
        km_t, km_s = kaplan_meier(t, e)
        mask = km_t <= max_t
        median_ttf = km_t[np.searchsorted(-km_s, -0.5)] if np.any(km_s <= 0.5) else float("nan")
        label = f"{g} (n={len(sg)}, мед.={median_ttf:.0f} дн.)" if not np.isnan(median_ttf) else f"{g} (n={len(sg)})"
        ax.step(km_t[mask], km_s[mask], where="post",
                color=km_colors[i], lw=2.0, label=label)
    ax.set_xlabel(NNO_LABEL, fontsize=11)
    ax.set_ylabel("Вероятность выживания", fontsize=11)
    ax.set_title("Б. Kaplan-Meier: ННО по бинам ГФ", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.legend(fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "glf_vs_nno_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: glf_vs_nno_boxplot.png")

# ── Figure 7: Высокочастотные прогоны: KM и профиль отказов ──────────────────

def fig_hf_runs(df: pd.DataFrame) -> None:
    has_share = df["freq_share55"].notna().any()
    has_freq  = df["freq_mean"].notna().any()
    if not has_share and not has_freq:
        print("  [skip] нет данных о частоте")
        return

    sub = df[df["ttf"].notna() & (df["ttf"] > 0)].copy()

    # Groups: HF (>50% > 55 Hz), baseline (50–55 Hz mean), rest
    hf_mask   = sub["freq_share55"] > 0.5
    bl_mask   = sub["freq_mean"].between(50, 55) & ~hf_mask
    sub["grp_hf"] = "Остальные"
    sub.loc[bl_mask, "grp_hf"] = "Базис (50–55 Гц)"
    sub.loc[hf_mask,   "grp_hf"] = "ВЧ-прогоны (>55 Гц >50% вр.)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Высокочастотные прогоны: >50% рабочего времени выше 55 Гц\nKaplan-Meier и структура отказов",
                 fontsize=13, y=1.01)

    grp_colors = {
        "Базис (50–55 Гц)":         "#1f77b4",
        "ВЧ-прогоны (>55 Гц >50% вр.)": "#d62728",
    }

    # Panel A: KM кривые
    ax = axes[0]
    max_t = float(sub.loc[sub["grp_hf"] != "Остальные", "ttf"].quantile(0.97))
    for grp, color in grp_colors.items():
        sg = sub[sub["grp_hf"] == grp]
        if len(sg) < 3:
            continue
        t = sg["ttf"].to_numpy(float)
        e = sg["event"].to_numpy(float)
        km_t, km_s = kaplan_meier(t, e)
        mask = km_t <= max_t
        ev_idx = np.where(km_s <= 0.5)[0]
        med = km_t[ev_idx[0]] if len(ev_idx) else float("nan")
        lbl = f"{grp}\n(n={len(sg)}, мед.={med:.0f} дн.)" if not np.isnan(med) else f"{grp} (n={len(sg)})"
        ax.step(km_t[mask], km_s[mask], where="post", color=color, lw=2.2, label=lbl)
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel(NNO_LABEL, fontsize=11)
    ax.set_ylabel("Вероятность выживания", fontsize=11)
    ax.set_title("А. Kaplan-Meier: ВЧ-прогоны vs Базис (глобальный фонд)", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")

    # Panel B: профиль отказов
    ax = axes[1]
    show_cats = CATEGORY_ORDER
    cat_col = "cat"
    counts = {}
    for grp in ["Базис (50–55 Гц)", "ВЧ-прогоны (>55 Гц >50% вр.)"]:
        sg = sub[(sub["grp_hf"] == grp) & (sub["event"] == 1) & sub[cat_col].notna()]
        vc = sg[cat_col].value_counts()
        total = vc.sum() if vc.sum() > 0 else 1
        counts[grp] = {c: vc.get(c, 0) / total * 100 for c in show_cats}

    x = np.arange(len(show_cats))
    w = 0.35
    for j, (grp, color) in enumerate(grp_colors.items()):
        vals = [counts.get(grp, {}).get(c, 0) for c in show_cats]
        ax.bar(x + j * w, vals, w, color=color, alpha=0.75, label=grp)
    ax.set_xticks(x + w / 2)
    short_cats = [c.replace(" (R-0)", "") for c in show_cats]
    ax.set_xticklabels(short_cats, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Доля от отказов группы (%)", fontsize=11)
    ax.set_title("Б. Структура отказов: ВЧ-прогоны vs Базис", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "hf_runs_km_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: hf_runs_km_global.png")

# ── Figure 8: Cycling analysis ─────────────────────────────────────────────────

def fig_cycling(df: pd.DataFrame) -> None:
    has_freq = df["freq_mean"].notna().any()
    if not has_freq:
        print("  [skip] нет данных о частоте для cycling")
        return

    sub = df[df["freq_mean"].notna() & df["ttf"].notna() & (df["ttf"] >= 10)].copy()
    sub["grp"] = "Прочие"
    sub.loc[sub["freq_mean"].between(50, 55, inclusive="both"), "grp"] = "Базис (50–55 Гц)"
    sub.loc[sub["freq_mean"] > 58, "grp"] = "Near-60 (mean >58 Гц)"

    grp_order = ["Базис (50–55 Гц)", "Near-60 (mean >58 Гц)"]
    palette = {"Базис (50–55 Гц)": "#1f77b4", "Near-60 (mean >58 Гц)": "#d62728"}

    if sub["n_valid"].notna().any() and sub["n_below_45"].notna().any():
        sub["frac_below45"] = sub["n_below_45"] / sub["n_valid"].replace(0, np.nan)
        sub["frac_no_tele"] = (1.0 - sub["n_valid"] / sub["ttf"].replace(0, np.nan)).clip(0, 1)
    else:
        sub["frac_below45"] = np.nan
        sub["frac_no_tele"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Стабильность режима: near-60 Гц vs базис (50–55 Гц)\n"
                 "Прокси цикличности: доля дней <45 Гц и дней без телеметрии",
                 fontsize=13, y=1.01)

    for axi, (metric, title, xlabel) in enumerate([
        ("frac_below45", "А. Дни ниже 45 Гц (остановки)", "Доля дней <45 Гц от валидных (%)"),
        ("frac_no_tele", "Б. Дни без частотной телеметрии", "Доля дней без телеметрии от ННО (%)"),
    ]):
        ax = axes[axi]
        has_data = sub[metric].notna().any()
        if has_data:
            for g in grp_order:
                vals = sub.loc[(sub["grp"] == g) & sub[metric].notna(), metric]
                if vals.empty:
                    continue
                med = vals.median()
                ax.hist(vals * 100, bins=25, alpha=0.55, color=palette[g], density=True,
                        label=f"{g}\nМедиана: {med*100:.1f}% (n={len(vals)})")
                ax.axvline(med * 100, color=palette[g], lw=2, ls="--")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Плотность", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=11)

    # Summary footnote
    rows = []
    for g in grp_order:
        sg = sub[sub["grp"] == g]
        rows.append(f"{g}: n={len(sg)}, медиана ННО={sg['ttf'].median():.0f} дн.")
    fig.text(0.5, -0.04, "  |  ".join(rows), ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "cycling_near60_vs_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: cycling_near60_vs_baseline.png")

# ── Figure 9: Classical Weibull — Global fund ─────────────────────────────────

def fig_weibull_classical_global(df: pd.DataFrame) -> None:
    from scipy.optimize import minimize
    from scipy.special import gamma as gamma_fn

    t_col = "ttf"
    sub = df[df[t_col].notna() & (df[t_col] > 0)].copy()
    t = sub[t_col].values.astype(float)
    e = sub["event"].values.astype(float)

    # ── KM ────────────────────────────────────────────────────────────────────
    order = np.argsort(t)
    t_s, e_s = t[order], e[order]
    km_times, km_surv = [0.0], [1.0]
    S = 1.0
    n = len(t_s)
    for i in range(n):
        at_risk = n - i
        if e_s[i] == 1:
            S *= 1 - 1 / at_risk
        if i == n - 1 or t_s[i] != t_s[i + 1]:
            km_times.append(float(t_s[i]))
            km_surv.append(S)
    km_times = np.array(km_times)
    km_surv  = np.array(km_surv)

    # KM median (empirical)
    km_median = float(km_times[np.searchsorted(-km_surv, -0.5)])

    # ── Weibull MLE with censoring ────────────────────────────────────────────
    def neg_loglik(params):
        b, n_ = np.exp(params)
        x = (t / n_) ** b
        ll = np.sum(e * (np.log(b / n_) + (b - 1) * np.log(np.clip(t / n_, 1e-9, None))) - x)
        return -ll

    med_ev = float(np.median(t[e == 1]))
    res = minimize(neg_loglik, [np.log(0.9), np.log(med_ev)], method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 80000})
    beta, eta = float(np.exp(res.x[0])), float(np.exp(res.x[1]))

    wb_median = eta * np.log(2) ** (1 / beta)
    wb_mean   = eta * float(gamma_fn(1 + 1 / beta))
    wb_b90    = eta * (-np.log(0.1)) ** (1 / beta)

    # RMST(365) — area under KM up to 365
    tau = 365.0
    mask365 = km_times <= tau
    t_rm = np.append(km_times[mask365], tau)
    s_rm = np.append(km_surv[mask365], km_surv[mask365][-1])
    rmst365 = float(np.trapz(s_rm, t_rm))

    # ── Figure ───────────────────────────────────────────────────────────────
    import matplotlib.transforms as mtransforms

    t_plot = np.linspace(0, 900, 800)
    wb_sf  = np.exp(-(t_plot / eta) ** beta)
    s_at_median = 0.5
    s_at_mean   = float(np.exp(-(wb_mean / eta) ** beta))

    C_MED  = "#B36A00"   # amber  — median
    C_MEAN = "#1a6eb0"   # blue   — mean
    C_B90  = "#d62728"   # red    — B90
    C_KM   = "#444444"   # dark   — KM
    C_RMST = "#1F5C99"   # navy   — RMST area

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # RMST shaded area (under KM, 0-365)
    ax.fill_between(t_rm, 0, s_rm, step="post",
                    color=C_RMST, alpha=0.12, zorder=0,
                    label=f"RMST(365) = {rmst365:.0f} дн.")
    ax.axvline(tau, color=C_RMST, ls=":", lw=1.2, alpha=0.5)

    # KM curve
    ax.step(km_times, km_surv, where="post",
            color=C_KM, lw=1.6, alpha=0.8,
            label=f"KM (эмпир.),  B50={km_median:.0f} дн.")

    # Weibull fit
    ax.plot(t_plot, wb_sf, color=C_MED, lw=2.4, ls="-",
            label=f"Weibull (MLE),  β={beta:.3f},  η={eta:.0f} дн.")

    # ── Median crosshair (horizontal to y-axis + vertical to x-axis) ─────
    ax.plot([0, wb_median], [s_at_median, s_at_median],
            color=C_MED, ls="--", lw=1.2, alpha=0.8)
    ax.plot([wb_median, wb_median], [0, s_at_median],
            color=C_MED, ls="--", lw=1.4, alpha=0.85)
    ax.annotate(f"Медиана\n{wb_median:.0f} дн.",
                xy=(wb_median, s_at_median),
                xytext=(wb_median + 35, s_at_median + 0.07),
                fontsize=9.5, color=C_MED,
                arrowprops=dict(arrowstyle="-", color=C_MED, lw=0.9))

    # ── Mean crosshair ────────────────────────────────────────────────────
    ax.plot([0, wb_mean], [s_at_mean, s_at_mean],
            color=C_MEAN, ls="--", lw=1.2, alpha=0.8)
    ax.plot([wb_mean, wb_mean], [0, s_at_mean],
            color=C_MEAN, ls="--", lw=1.4, alpha=0.85)
    ax.annotate(f"Среднее\n{wb_mean:.0f} дн.",
                xy=(wb_mean, s_at_mean),
                xytext=(wb_mean + 30, s_at_mean + 0.07),
                fontsize=9.5, color=C_MEAN,
                arrowprops=dict(arrowstyle="-", color=C_MEAN, lw=0.9))

    # ── Y-axis labels for median and mean ────────────────────────────────
    blended = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.01, s_at_median, f"{s_at_median:.2f}",
            transform=blended, fontsize=8.5, color=C_MED,
            ha="right", va="center", fontweight="bold")
    ax.text(-0.01, s_at_mean, f"{s_at_mean:.2f}",
            transform=blended, fontsize=8.5, color=C_MEAN,
            ha="right", va="center", fontweight="bold")

    # ── B90 ───────────────────────────────────────────────────────────────
    ax.axvline(wb_b90, color=C_B90, ls=":", lw=1.3, alpha=0.75)
    ax.annotate(f"B90 = {wb_b90:.0f} дн.",
                xy=(wb_b90, 0.10), xytext=(wb_b90 - 280, 0.16),
                fontsize=9.5, color=C_B90,
                arrowprops=dict(arrowstyle="-", color=C_B90, lw=0.9))

    # ── Parameter box ─────────────────────────────────────────────────────
    box_txt = (
        f"n = {len(t):,}   отказов = {int(e.sum()):,}\n"
        f"β = {beta:.3f}  (β<1 → убывающий риск)\n"
        f"η = {eta:.0f} дн.  (63.2% фонда откажут к этому моменту)\n"
        f"Среднее (теор.) = {wb_mean:.0f} дн.  →  S = {s_at_mean:.2f}\n"
        f"RMST(365) = {rmst365:.0f} дн."
    )
    ax.text(0.97, 0.97, box_txt, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#AAAAAA", alpha=0.92))

    ax.set_xlim(0, 900)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Наработка на отказ, ННО (дней)", fontsize=13)
    ax.set_ylabel("Вероятность выживания S(t)", fontsize=13)
    ax.set_title("Классическая модель Вейбулла — Глобальный фонд\n"
                 f"S(t) = exp(−(t/η)^β),   KM медиана = {km_median:.0f} дн., "
                 f"n = {len(t):,}", fontsize=13, pad=10)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "weibull_classical_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранено: weibull_classical_global.png")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Выходная директория: {OUT_DIR}")
    df_raw = load_data()
    if df_raw is None:
        return
    df = prepare(df_raw)
    print(f"  ttf: {df['ttf'].notna().sum()}  trf: {df['trf'].notna().sum()}"
          f"  h2s_proxy: {(df['h2s_proxy_bin']!='<нет данных>').sum()}"
          f"  glf_bin: {df['glf_bin'].notna().sum()}")

    print("\n1. ННО vs Накопленная частота — категории")
    fig_trf_category(df)
    print("1z. zoom — категории")
    fig_trf_category_zoom(df)
    print("2. ННО vs Накопленная частота — H₂S бинарный")
    fig_trf_h2s_binary(df)
    print("2z. zoom — H₂S бинарный")
    fig_trf_h2s_binary_zoom(df)
    print("3. ННО vs Накопленная частота — H₂S прокси (4 бина)")
    fig_trf_h2s_proxy(df)
    print("3z. zoom — H₂S прокси")
    fig_trf_h2s_proxy_zoom(df)
    print("4. ННО vs Накопленная частота — прокси осадков")
    fig_trf_precipitate(df)
    print("4z. zoom — прокси осадков")
    fig_trf_precipitate_zoom(df)
    print("5. ННО vs Накопленная частота — ГФ")
    fig_trf_glf(df)
    print("5z. zoom — ГФ")
    fig_trf_glf_zoom(df)
    print("6. ГФ-бины vs ННО: boxplot + KM")
    fig_glf_vs_nno(df)
    print("7. Высокочастотные прогоны: KM + структура отказов")
    fig_hf_runs(df)
    print("8. Цикличность: near-60 vs базис")
    fig_cycling(df)
    print("9. Классическая Weibull — Global")
    fig_weibull_classical_global(df)
    print("10. ННО vs Накопленная частота — Кпод")
    fig_trf_kpod(df)
    print("10z. zoom — Кпод")
    fig_trf_kpod_zoom(df)

    print(f"\nГотово. Файлы в {OUT_DIR}")

if __name__ == "__main__":
    main()
