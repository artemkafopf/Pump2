from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.paths import results_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
PROMPT_TABLES = PROMPT_DIR / "tables"
_SLUG = "vt_60hz_presentation_assets"
OUT_DIR = results_dir(_SLUG)
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
DB_PATH = REPO_ROOT / "data" / "warehouse" / "pump2.db"
LATENT_PATH = REPO_ROOT / "analysis" / "vt_ya_global_survival_comparison.csv"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

CATEGORY_COLORS = {
    "КЛ (R-0)": "#4C78A8",
    "ПЭД (R-0)": "#F58518",
    "Слом вала": "#54A24B",
    "Засорение РО": "#E45756",
    "НКТ": "#8F63B8",
    "Износ РО": "#9C755F",
    "Износ/негермет.гидрозащиты": "#E377C2",
    "<missing>": "#BAB0AC",
}

GROUP_COLORS = {
    "50-55 Гц": "#54A24B",
    "mean > 58 Гц": "#E45756",
    "Y > 0.5": "#4C78A8",
}

REFERENCE_LINES = [
    (40.0, "#E67E22", "--"),
    (50.0, "#2CA02C", "--"),
    (60.0, "#1F77B4", "--"),
    (70.0, "#D62728", "-"),
]


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(PROMPT_TABLES / "analysis_dataset.csv", low_memory=False)
    numeric_cols = [
        "row_id",
        "event",
        "freq_w_mean",
        "freq_above_55hz_pct",
        "total_freq_hz_days",
        "ttf_true_best_days",
        "gypsum_proxy_per_day",
        "h2s_effective_mg_l",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["population"] = np.where(df["field"].eq("Vt"), "Vt", "Global")
    df["failure_category_plot"] = df["failure_category_raw"].fillna(df.get("failure_category", "<missing>")).fillna("<missing>")
    df["baseline_50_55"] = df["freq_w_mean"].gt(50.0) & df["freq_w_mean"].le(55.0)
    df["true60_mean58"] = df["freq_w_mean"].gt(58.0)
    df["hf_share55"] = df["freq_above_55hz_pct"].gt(0.5)
    return df


def build_failure_mix_pies(df: pd.DataFrame) -> None:
    failures = df.loc[df["event"].eq(1)].copy()
    categories = [
        c for c in CATEGORY_COLORS
        if c != "<missing>" and (failures["failure_category_plot"] == c).any()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"aspect": "equal"})
    for ax, label, subset in [
        (axes[0], "Global", failures),
        (axes[1], "Vt", failures.loc[failures["field"].eq("Vt")]),
    ]:
        counts = subset["failure_category_plot"].value_counts()
        vals = [counts.get(cat, 0) for cat in categories]
        colors = [CATEGORY_COLORS[cat] for cat in categories]
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=categories,
            colors=colors,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 4 else "",
            pctdistance=0.78,
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        for t in texts:
            t.set_fontsize(10)
        for t in autotexts:
            t.set_fontsize(10)
            t.set_color("white")
            t.set_weight("bold")
        ax.set_title(f"{label}\nотказы = {len(subset):,}")

    fig.suptitle("Структура отказов по категориям", fontsize=18)
    _save(fig, "failure_mix_pies_vt_global.png")


def build_latent_summary() -> None:
    df = pd.read_csv(LATENT_PATH)
    pops = ["Global", "Vt", "Ya"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    y_pos = np.arange(len(pops))
    for i, pop in enumerate(pops):
        b10 = df.loc[(df["group"] == pop) & (df["component"] == "mixture") & (df["parameter"] == "B10"), "q50"].iloc[0]
        b50 = df.loc[(df["group"] == pop) & (df["component"] == "mixture") & (df["parameter"] == "B50"), "q50"].iloc[0]
        b90 = df.loc[(df["group"] == pop) & (df["component"] == "mixture") & (df["parameter"] == "B90"), "q50"].iloc[0]
        ax.hlines(i, b10, b90, color="#4C78A8", linewidth=6, alpha=0.75)
        ax.scatter(b50, i, s=90, color="#E45756", zorder=3)
        ax.text(b90 + 18, i, f"B10={b10:.0f}  B50={b50:.0f}  B90={b90:.0f}", va="center", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pops)
    ax.set_xlabel("Сутки")
    ax.set_title("Латентная смесь: интервалы жизни")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1]
    bar_h = 0.55
    for i, pop in enumerate(pops):
        w1 = df.loc[(df["group"] == pop) & (df["component"] == "component_1") & (df["parameter"] == "weight"), "q50"].iloc[0]
        w2 = df.loc[(df["group"] == pop) & (df["component"] == "component_2") & (df["parameter"] == "weight"), "q50"].iloc[0]
        b1 = df.loc[(df["group"] == pop) & (df["component"] == "component_1") & (df["parameter"] == "beta"), "q50"].iloc[0]
        b2 = df.loc[(df["group"] == pop) & (df["component"] == "component_2") & (df["parameter"] == "beta"), "q50"].iloc[0]
        ax.barh(i, w1, height=bar_h, color="#F2CF5B", label="Ранний/случайный" if i == 0 else None)
        ax.barh(i, w2, left=w1, height=bar_h, color="#54A24B", label="Износ" if i == 0 else None)
        ax.text(0.02, i, f"β1={b1:.2f}", va="center", ha="left", fontsize=10)
        ax.text(min(w1 + 0.02, 0.92), i, f"β2={b2:.2f}", va="center", ha="left", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pops)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Вес скрытого компонента")
    ax.set_title("Состав скрытой смеси")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Латентная Weibull-модель по полям", fontsize=18)
    _save(fig, "latent_mixture_summary.png")


def _add_reference_lines(ax: plt.Axes, xmax: float) -> None:
    xs = np.linspace(0, xmax, 200)
    for freq_hz, color, style in REFERENCE_LINES:
        ys = freq_hz * xs
        ax.plot(xs, ys, linestyle=style, linewidth=1.8, color=color, alpha=0.8, label=f"{int(freq_hz)} Гц")


def _cloud_axes_limits(sub: pd.DataFrame) -> tuple[float, float]:
    xmax = float(sub["ttf_true_best_days"].quantile(0.99))
    ymax = float(sub["total_freq_hz_days"].quantile(0.99))
    return max(xmax, 50.0), max(ymax, 2000.0)


def build_trf_clouds(df: pd.DataFrame) -> None:
    fails = df.loc[
        df["event"].eq(1)
        & df["ttf_true_best_days"].gt(0)
        & df["total_freq_hz_days"].gt(0)
    ].copy()

    build_trf_cloud_category(fails)
    build_trf_cloud_continuous(
        fails,
        color_col="gypsum_proxy_per_day",
        cmap="viridis",
        title="TRF vs TTF_true_best: окраска по гипсовому прокси",
        cbar_label="Gypsum proxy / day",
        out_name="trf_ttf_cloud_precip.png",
    )
    build_trf_cloud_continuous(
        fails,
        color_col="h2s_effective_mg_l",
        cmap="magma",
        title="TRF vs TTF_true_best: окраска по H₂S",
        cbar_label="H₂S effective, mg/L",
        out_name="trf_ttf_cloud_h2s.png",
    )


def build_trf_cloud_category(fails: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, label, sub in [
        (axes[0], "Global", fails),
        (axes[1], "Vt", fails.loc[fails["field"].eq("Vt")]),
    ]:
        xmax, ymax = _cloud_axes_limits(sub)
        cats = [c for c in CATEGORY_COLORS if (sub["failure_category_plot"] == c).any()]
        for cat in cats:
            grp = sub.loc[sub["failure_category_plot"] == cat]
            ax.scatter(
                grp["ttf_true_best_days"],
                grp["total_freq_hz_days"],
                s=24,
                alpha=0.65,
                label=cat,
                color=CATEGORY_COLORS.get(cat, "#999999"),
                edgecolors="none",
            )
        _add_reference_lines(ax, xmax)
        ax.set_xlim(0, xmax * 1.03)
        ax.set_ylim(0, ymax * 1.03)
        ax.set_title(label)
        ax.set_xlabel("TTF_true_best, сутки")
        ax.set_ylabel("TRF, Гц·сут")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
    )
    fig.suptitle("TRF vs TTF_true_best: облако отказов по категориям", fontsize=18, y=1.08)
    _save(fig, "trf_ttf_cloud_category.png")


def build_trf_cloud_continuous(
    fails: pd.DataFrame,
    *,
    color_col: str,
    cmap: str,
    title: str,
    cbar_label: str,
    out_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    values = fails[color_col].replace([np.inf, -np.inf], np.nan)
    vmax = float(values.quantile(0.95))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(values.max()) if np.isfinite(values.max()) else 1.0

    for ax, label, sub in [
        (axes[0], "Global", fails),
        (axes[1], "Vt", fails.loc[fails["field"].eq("Vt")]),
    ]:
        xmax, ymax = _cloud_axes_limits(sub)
        clean = sub.loc[sub[color_col].notna()].copy()
        scatter = ax.scatter(
            clean["ttf_true_best_days"],
            clean["total_freq_hz_days"],
            c=clean[color_col].clip(upper=vmax),
            cmap=cmap,
            s=26,
            alpha=0.75,
            edgecolors="none",
        )
        _add_reference_lines(ax, xmax)
        ax.set_xlim(0, xmax * 1.03)
        ax.set_ylim(0, ymax * 1.03)
        ax.set_title(label)
        ax.set_xlabel("TTF_true_best, сутки")
        ax.set_ylabel("TRF, Гц·сут")
        ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=18, y=1.04)
    _save(fig, out_name)


def build_stop_metrics(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        runs = pd.read_sql_query(
            "SELECT row_id, well_key, install_date, stop_date FROM raw__v03_runs",
            conn,
            parse_dates=["install_date", "stop_date"],
        )
        daily = pd.read_sql_query(
            "SELECT well_key, dt, freq, qliq FROM proc__daily_merged",
            conn,
            parse_dates=["dt"],
        )

    daily["freq"] = pd.to_numeric(daily["freq"], errors="coerce")
    daily["qliq"] = pd.to_numeric(daily["qliq"], errors="coerce")
    by_well = {
        well_key: sub.sort_values("dt").reset_index(drop=True)
        for well_key, sub in daily.groupby("well_key", sort=False)
    }

    merged = df.merge(runs, on="row_id", how="left")
    rows: list[dict[str, float | int]] = []
    for rec in merged[["row_id", "well_key", "install_date", "stop_date"]].itertuples(index=False):
        sub = by_well.get(rec.well_key)
        if sub is None or pd.isna(rec.install_date) or pd.isna(rec.stop_date):
            rows.append({"row_id": rec.row_id, "observed_days": np.nan, "inactive_share": np.nan, "stop_ep_per_100d": np.nan})
            continue
        run = sub.loc[(sub["dt"] >= rec.install_date) & (sub["dt"] <= rec.stop_date)].copy()
        run = run.loc[run["freq"].notna() | run["qliq"].notna()].copy()
        if run.empty:
            rows.append({"row_id": rec.row_id, "observed_days": np.nan, "inactive_share": np.nan, "stop_ep_per_100d": np.nan})
            continue
        inactive = ((run["freq"].fillna(0.0) <= 0.0) | (run["qliq"].fillna(0.0) <= 0.0)).astype(int)
        episodes = int(((inactive == 1) & (inactive.shift(fill_value=0) == 0)).sum())
        observed_days = int(len(run))
        rows.append(
            {
                "row_id": rec.row_id,
                "observed_days": observed_days,
                "inactive_share": float(inactive.mean()),
                "stop_ep_per_100d": float(episodes / observed_days * 100.0),
            }
        )

    metric_df = pd.DataFrame(rows)
    out = merged.merge(metric_df, on="row_id", how="left")

    summary_rows = []
    for population, subset in [("Global", out), ("Vt", out.loc[out["field"].eq("Vt")])]:
        for label, mask in [
            ("50-55 Гц", subset["baseline_50_55"]),
            ("mean > 58 Гц", subset["true60_mean58"]),
            ("Y > 0.5", subset["hf_share55"]),
        ]:
            grp = subset.loc[mask & subset["observed_days"].notna()].copy()
            if grp.empty:
                continue
            summary_rows.append(
                {
                    "population": population,
                    "group": label,
                    "n_runs": int(len(grp)),
                    "inactive_share_mean": float(grp["inactive_share"].mean()),
                    "inactive_share_median": float(grp["inactive_share"].median()),
                    "stop_ep_per_100d_mean": float(grp["stop_ep_per_100d"].mean()),
                    "stop_ep_per_100d_median": float(grp["stop_ep_per_100d"].median()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "stability_stop_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for col_idx, population in enumerate(["Global", "Vt"]):
        sub = summary.loc[summary["population"] == population].copy()
        x = np.arange(len(sub))
        colors = [GROUP_COLORS[g] for g in sub["group"]]

        ax = axes[0, col_idx]
        vals = sub["inactive_share_mean"] * 100.0
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.4, f"{yi:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_title(population)
        ax.set_ylabel("Неактивные дни, %")
        ax.set_ylim(0, max(40, vals.max() * 1.25))
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[1, col_idx]
        vals = sub["stop_ep_per_100d_mean"]
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.08, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Стоп-эпизоды / 100 дней")
        ax.set_ylim(0, max(4.5, vals.max() * 1.28))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=0)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Телеметрия стабильности режима: mean-прокси vs Y > 0.5", fontsize=18)
    _save(fig, "stability_stop_metrics.png")


def main() -> None:
    _ensure_dirs()
    dataset = load_dataset()
    build_failure_mix_pies(dataset)
    build_latent_summary()
    build_trf_clouds(dataset)
    build_stop_metrics(dataset)
    print(f"[vt-60hz-final-assets] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
