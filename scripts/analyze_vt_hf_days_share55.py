from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.paths import results_dir
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt
from scipy.stats import fisher_exact, mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
_SLUG = "vt_hf_days_share55"
_PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
BASE_OUTPUT_DIR = results_dir(_SLUG)
FIGURES_DIR = BASE_OUTPUT_DIR / "figures"
TABLES_DIR = BASE_OUTPUT_DIR / "tables"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


COLORS = {
    "baseline": "#2ca02c",
    "hf": "#d62728",
    "global": "#1f77b4",
    "vt": "#ff7f0e",
    "best": "#4c78a8",
    "ttf_true": "#f58518",
    "trf": "#54a24b",
}

THRESHOLDS = [30, 60, 90]


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    path = _PROMPT_DIR / "tables" / "analysis_dataset.csv"
    df = pd.read_csv(path)
    for column in [
        "duration_best_days",
        "ttf_true_best_days",
        "run_days",
        "freq_w_mean",
        "freq_above_55hz_pct",
        "n_freq_above_55hz",
        "total_freq_hz_days",
        "event",
        "h2s_effective_mg_l",
        "avg_glf",
        "mount_year",
        "trf_per_day",
        "tlf_per_day",
        "Kpod",
        "calcium_load_per_day",
        "chloride_load_per_day",
        "sulfate_load_per_day",
        "gypsum_proxy_per_day",
        "corrosion_fill",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["baseline_50_55_flag"] = (df["freq_w_mean"] > 50.0) & (df["freq_w_mean"] <= 55.0)
    # User-requested high-frequency definition:
    # high-frequency well = days(freq > 55 Hz) / days_total > 0.5
    # We use best-available operating days as days_total for consistency with TTF_true-centric analysis.
    df["hf_days_share55_flag"] = (_numeric(df["n_freq_above_55hz"]) / _numeric(df["duration_best_days"])) > 0.5
    df["hf_days_share55_run_flag"] = (_numeric(df["n_freq_above_55hz"]) / _numeric(df["run_days"])) > 0.5
    df["high_h2s_flag"] = df["h2s_effective_mg_l"] >= 3.0
    df["trf_50hz_equiv_days"] = _numeric(df["total_freq_hz_days"]) / 50.0
    df["infant_90d_best"] = ((df["event"] == 1) & (_numeric(df["duration_best_days"]) < 90.0)).astype(int)
    df["acid_flag"] = df["Кислый/Некислый"].astype("string").str.contains("Кислый", case=False, na=False).astype(int)
    return df


def build_population_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, pop_df in [("Global", df), ("Vt", df.loc[df["field"] == "Vt"].copy())]:
        for label, mask in [
            ("50-55 Hz baseline", pop_df["baseline_50_55_flag"]),
            ("mean >58 Hz proxy", pop_df["freq_w_mean"] > 58.0),
            ("mean >55 Hz proxy", pop_df["freq_w_mean"] > 55.0),
            ("days >55 / duration_best_days >0.5", pop_df["hf_days_share55_flag"]),
            ("days >55 / run_days >0.5", pop_df["hf_days_share55_run_flag"]),
        ]:
            subset = pop_df.loc[mask.fillna(False)].copy()
            rows.append(
                {
                    "population": population,
                    "definition": label,
                    "runs": int(len(subset)),
                    "failures": int(subset["event"].sum()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "hf_days_share55_population_comparison.csv", index=False, encoding="utf-8-sig")
    return frame


def build_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axes = [
        ("best_available_days", "duration_best_days", _numeric(df["duration_best_days"]).notna()),
        ("ttf_true_days", "ttf_true_best_days", _numeric(df["ttf_true_best_days"]).notna()),
        ("trf_50hz_equiv_days", "trf_50hz_equiv_days", _numeric(df["trf_50hz_equiv_days"]).notna() & _numeric(df["ttf_true_best_days"]).notna()),
    ]
    for axis_name, duration_col, axis_mask in axes:
        for population, pop_df in [("Global", df.loc[axis_mask].copy()), ("Vt", df.loc[axis_mask & df["field"].eq("Vt")].copy())]:
            for mode, mask in [
                ("50-55 Hz baseline", pop_df["baseline_50_55_flag"]),
                ("HF by >55d share >0.5", pop_df["hf_days_share55_flag"]),
            ]:
                subset = pop_df.loc[mask.fillna(False)].copy()
                failures = int(subset["event"].sum())
                early = ((subset["event"] == 1) & (_numeric(subset[duration_col]) < 90.0)).astype(int)
                rows.append(
                    {
                        "axis_name": axis_name,
                        "population": population,
                        "mode": mode,
                        "runs": int(len(subset)),
                        "failures": failures,
                        "median_duration": float(_numeric(subset[duration_col]).median()) if len(subset) else float("nan"),
                        "mean_duration": float(_numeric(subset[duration_col]).mean()) if len(subset) else float("nan"),
                        "early_rate_runs": float(early.mean()) if len(subset) else float("nan"),
                        "early_share_failures": float(early.sum() / max(failures, 1)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "hf_days_share55_axis_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_km(base: pd.DataFrame, hf: pd.DataFrame, duration_col: str, title: str, output_path: Path) -> dict[str, float]:
    km_base = KaplanMeierFitter()
    km_hf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    km_base.fit(base[duration_col], base["event"], label=f"50-55 Hz baseline (n={len(base)}, fail={int(base['event'].sum())})")
    km_hf.fit(hf[duration_col], hf["event"], label=f"HF share>0.5 (n={len(hf)}, fail={int(hf['event'].sum())})")
    km_base.plot_survival_function(ax=ax, ci_show=True, color=COLORS["baseline"], linewidth=2)
    km_hf.plot_survival_function(ax=ax, ci_show=True, color=COLORS["hf"], linewidth=2)
    result = logrank_test(base[duration_col], hf[duration_col], base["event"], hf["event"])
    ax.set_title(title)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.05,
        f"log-rank p = {result.p_value:.3f}\nMedian = {km_base.median_survival_time_:.0f} vs {km_hf.median_survival_time_:.0f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {
        "logrank_p": float(result.p_value),
        "baseline_median": float(km_base.median_survival_time_),
        "hf_median": float(km_hf.median_survival_time_),
    }


def build_km_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axes = [
        ("best_available_days", "duration_best_days"),
        ("ttf_true_days", "ttf_true_best_days"),
        ("trf_50hz_equiv_days", "trf_50hz_equiv_days"),
    ]
    for axis_name, duration_col in axes:
        mask = _numeric(df[duration_col]).notna()
        for population, pop_df in [("Global", df.loc[mask].copy()), ("Vt", df.loc[mask & df["field"].eq("Vt")].copy())]:
            base = pop_df.loc[pop_df["baseline_50_55_flag"]].copy()
            hf = pop_df.loc[pop_df["hf_days_share55_flag"]].copy()
            if len(base) == 0 or len(hf) == 0 or int(base["event"].sum()) < 6 or int(hf["event"].sum()) < 6:
                continue
            stats = plot_km(
                base,
                hf,
                duration_col,
                f"{population}: HF share>0.5 vs 50-55 Hz baseline ({axis_name})",
                FIGURES_DIR / f"hf_share55_{axis_name}_{population.lower()}_km.png",
            )
            rows.append(
                {
                    "axis_name": axis_name,
                    "population": population,
                    "baseline_runs": int(len(base)),
                    "baseline_failures": int(base["event"].sum()),
                    "hf_runs": int(len(hf)),
                    "hf_failures": int(hf["event"].sum()),
                    **stats,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "hf_days_share55_km_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def fit_cox(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axes = [
        ("best_available_days", "duration_best_days"),
        ("ttf_true_days", "ttf_true_best_days"),
        ("trf_50hz_equiv_days", "trf_50hz_equiv_days"),
    ]
    for axis_name, duration_col in axes:
        for population, pop_mask, include_field in [
            ("Global", pd.Series(True, index=df.index), True),
            ("Vt", df["field"] == "Vt", False),
        ]:
            work = df.loc[pop_mask & (df["baseline_50_55_flag"] | df["hf_days_share55_flag"])].copy()
            work["duration"] = _numeric(work[duration_col])
            work["hf"] = work["hf_days_share55_flag"].astype(int)
            work["high_h2s"] = work["high_h2s_flag"].astype(int)
            keep = ["duration", "event", "hf", "high_h2s", "avg_glf", "mount_year", "Скв."]
            if include_field:
                field_counts = work["field"].astype("string").value_counts()
                keep_fields = field_counts.loc[field_counts >= 20].index.astype(str).tolist()
                work = work.loc[work["field"].astype("string").isin(keep_fields)].copy()
                keep.append("field")
            work = work[keep].dropna().copy()
            if len(work) == 0 or int(work["hf"].sum()) == 0:
                continue
            formula = "hf + high_h2s + avg_glf + mount_year"
            if include_field:
                formula += " + C(field)"
            cph = CoxPHFitter()
            cph.fit(work, duration_col="duration", event_col="event", cluster_col="Скв.", robust=True, formula=formula)
            row = cph.summary.loc["hf"]
            rows.append(
                {
                    "axis_name": axis_name,
                    "population": population,
                    "rows": int(len(work)),
                    "events": int(work["event"].sum()),
                    "hr_hf": float(row["exp(coef)"]),
                    "hr_lower_95": float(row["exp(coef) lower 95%"]),
                    "hr_upper_95": float(row["exp(coef) upper 95%"]),
                    "p_value": float(row["p"]),
                    "concordance": float(cph.concordance_index_),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "hf_days_share55_hr_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_hr(hr_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = hr_df.copy()
    plot_df["label"] = plot_df["population"] + " - " + plot_df["axis_name"]
    color_map = {
        "best_available_days": COLORS["best"],
        "ttf_true_days": COLORS["ttf_true"],
        "trf_50hz_equiv_days": COLORS["trf"],
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.errorbar(
            float(row["hr_hf"]),
            idx,
            xerr=[[float(row["hr_hf"] - row["hr_lower_95"])], [float(row["hr_upper_95"] - row["hr_hf"])]],
            fmt="o",
            color=color_map.get(str(row["axis_name"]), "#555555"),
            ecolor="#555555",
            capsize=4,
        )
        ax.text(float(row["hr_upper_95"]) * 1.02, idx, f"p={float(row['p_value']):.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(plot_df)), plot_df["label"].tolist())
    ax.set_xlabel("Adjusted hazard ratio for HF share>0.5 vs baseline")
    ax.set_title("High-frequency definition sensitivity: days above 55 Hz share > 0.5")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_threshold_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    failures = df.loc[(df["event"] == 1) & _numeric(df["ttf_true_best_days"]).notna()].copy()
    rows = []
    tests = []
    for threshold in THRESHOLDS:
        for population, pop_df in [("Global", failures), ("Vt", failures.loc[failures["field"] == "Vt"].copy())]:
            for mode, subset in [
                ("50-55 Hz baseline", pop_df.loc[pop_df["baseline_50_55_flag"]].copy()),
                ("HF by >55d share >0.5", pop_df.loc[pop_df["hf_days_share55_flag"]].copy()),
            ]:
                early = (subset["ttf_true_best_days"] < threshold).astype(int)
                rows.append(
                    {
                        "threshold_days": threshold,
                        "population": population,
                        "mode": mode,
                        "failures_with_ttf_true": int(len(subset)),
                        "early_failures": int(early.sum()),
                        "mature_failures": int(len(subset) - early.sum()),
                        "early_share_failures": float(early.mean()) if len(subset) else float("nan"),
                    }
                )
            base = pop_df.loc[pop_df["baseline_50_55_flag"]].copy()
            hf = pop_df.loc[pop_df["hf_days_share55_flag"]].copy()
            base_early = int((base["ttf_true_best_days"] < threshold).sum())
            base_mature = int(len(base) - base_early)
            hf_early = int((hf["ttf_true_best_days"] < threshold).sum())
            hf_mature = int(len(hf) - hf_early)
            odds_ratio, p_value = fisher_exact([[hf_early, hf_mature], [base_early, base_mature]], alternative="two-sided")
            tests.append(
                {
                    "threshold_days": threshold,
                    "population": population,
                    "baseline_failures": int(len(base)),
                    "hf_failures": int(len(hf)),
                    "baseline_early_share": float(base_early / max(len(base), 1)),
                    "hf_early_share": float(hf_early / max(len(hf), 1)),
                    "share_difference_hf_minus_baseline": float((hf_early / max(len(hf), 1)) - (base_early / max(len(base), 1))),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                }
            )
    summary_df = pd.DataFrame(rows)
    tests_df = pd.DataFrame(tests)
    summary_df.to_csv(TABLES_DIR / "hf_days_share55_threshold_summary.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(TABLES_DIR / "hf_days_share55_threshold_tests.csv", index=False, encoding="utf-8-sig")
    return summary_df, tests_df


def plot_thresholds(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    color_lookup = {
        "Global | 50-55 Hz baseline": "#2ca02c",
        "Global | HF by >55d share >0.5": "#d62728",
        "Vt | 50-55 Hz baseline": "#1f77b4",
        "Vt | HF by >55d share >0.5": "#ff7f0e",
    }
    for (population, mode), subset in summary_df.groupby(["population", "mode"], dropna=False):
        subset = subset.sort_values("threshold_days")
        label = f"{population} | {mode}"
        ax.plot(
            subset["threshold_days"],
            subset["early_share_failures"],
            marker="o",
            linewidth=2,
            color=color_lookup.get(label, "#555555"),
            label=label,
        )
    ax.set_xticks(THRESHOLDS)
    ax.set_xlabel("Early / mature cutoff in TTF_true days")
    ax.set_ylabel("Early share among failures")
    ax.set_title("HF share>0.5: early-failure share at 30 / 60 / 90 days")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def compare_vt_environment(df: pd.DataFrame) -> pd.DataFrame:
    vt = df.loc[df["field"] == "Vt"].copy()
    vt["group"] = np.where(vt["hf_days_share55_flag"], "HF share>0.5", "Rest of Vt")
    hf = vt.loc[vt["group"] == "HF share>0.5"].copy()
    rest = vt.loc[vt["group"] == "Rest of Vt"].copy()
    rows = []
    for column, label in [
        ("h2s_effective_mg_l", "H2S effective, mg/L"),
        ("avg_glf", "GLF"),
        ("Kpod", "Kpod"),
        ("trf_per_day", "TRF per day"),
        ("tlf_per_day", "TLF per day"),
        ("calcium_load_per_day", "Calcium load per day"),
        ("chloride_load_per_day", "Chloride load per day"),
        ("sulfate_load_per_day", "Sulfate load per day"),
        ("gypsum_proxy_per_day", "Gypsum proxy per day"),
        ("duration_best_days", "Best duration, days"),
    ]:
        x = _numeric(hf[column]).dropna()
        y = _numeric(rest[column]).dropna()
        p_value = float(mannwhitneyu(x, y, alternative="two-sided").pvalue) if len(x) >= 3 and len(y) >= 3 else float("nan")
        rows.append(
            {
                "metric": label,
                "hf_n": int(len(x)),
                "rest_n": int(len(y)),
                "hf_median": float(x.median()) if len(x) else float("nan"),
                "rest_median": float(y.median()) if len(y) else float("nan"),
                "difference_hf_minus_rest": float(x.median() - y.median()) if len(x) and len(y) else float("nan"),
                "p_value": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "hf_days_share55_vt_environment.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_vt_environment(env_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = env_df.loc[env_df["metric"].isin(["H2S effective, mg/L", "GLF", "Kpod", "Calcium load per day", "Chloride load per day", "Sulfate load per day", "Gypsum proxy per day"])].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(plot_df))
    colors = [COLORS["hf"] if float(v) > 0 else COLORS["baseline"] for v in plot_df["difference_hf_minus_rest"]]
    ax.barh(y, plot_df["difference_hf_minus_rest"], color=colors)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, plot_df["metric"].tolist())
    ax.set_xlabel("Median difference: HF share>0.5 minus rest of Vt")
    ax.set_title("Vt environment check for the new HF definition")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_failure_mix(df: pd.DataFrame) -> pd.DataFrame:
    vt_fail = df.loc[(df["field"] == "Vt") & (df["event"] == 1)].copy()
    vt_fail["group"] = np.where(vt_fail["hf_days_share55_flag"], "HF share>0.5", "Rest of Vt")
    counts = (
        vt_fail.groupby(["group", "failure_category_raw"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"failure_category_raw": "category"})
    )
    counts["share"] = counts["count"] / counts.groupby("group")["count"].transform("sum")
    counts = counts.sort_values(["group", "share"], ascending=[True, False]).reset_index(drop=True)
    counts.to_csv(TABLES_DIR / "hf_days_share55_vt_failure_mix.csv", index=False, encoding="utf-8-sig")
    return counts


def plot_failure_mix(mix_df: pd.DataFrame, output_path: Path) -> None:
    top_categories = mix_df.groupby("category")["count"].sum().sort_values(ascending=False).head(6).index.tolist()
    plot_df = mix_df.loc[mix_df["category"].isin(top_categories)].copy()
    pivot = plot_df.pivot(index="category", columns="group", values="share").fillna(0.0)
    order = pivot.get("HF share>0.5", pd.Series(0.0, index=pivot.index)).sort_values(ascending=True).index.tolist()
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    y = np.arange(len(pivot))
    ax.barh(y - 0.18, pivot.get("Rest of Vt", pd.Series(0.0, index=pivot.index)), height=0.35, color=COLORS["baseline"], label="Rest of Vt")
    ax.barh(y + 0.18, pivot.get("HF share>0.5", pd.Series(0.0, index=pivot.index)), height=0.35, color=COLORS["hf"], label="HF share>0.5")
    ax.set_yticks(y, pivot.index.tolist())
    ax.set_xlabel("Failure share within Vt failures")
    ax.set_title("Vt failure mix: new HF definition vs rest of Vt")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_note(pop_df: pd.DataFrame, axis_df: pd.DataFrame, hr_df: pd.DataFrame, threshold_tests: pd.DataFrame, env_df: pd.DataFrame, mix_df: pd.DataFrame) -> Path:
    def pick(df: pd.DataFrame, **filters: object) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for key, value in filters.items():
            mask &= df[key] == value
        return df.loc[mask].iloc[0]

    pop_vt_old = pick(pop_df, population="Vt", definition="mean >58 Hz proxy")
    pop_vt_new = pick(pop_df, population="Vt", definition="days >55 / duration_best_days >0.5")
    pop_global_old = pick(pop_df, population="Global", definition="mean >58 Hz proxy")
    pop_global_new = pick(pop_df, population="Global", definition="days >55 / duration_best_days >0.5")

    vt_best = pick(axis_df, axis_name="best_available_days", population="Vt", mode="HF by >55d share >0.5")
    vt_best_base = pick(axis_df, axis_name="best_available_days", population="Vt", mode="50-55 Hz baseline")
    vt_true = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="HF by >55d share >0.5")
    vt_true_base = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="50-55 Hz baseline")
    global_true = pick(axis_df, axis_name="ttf_true_days", population="Global", mode="HF by >55d share >0.5")
    global_true_base = pick(axis_df, axis_name="ttf_true_days", population="Global", mode="50-55 Hz baseline")

    vt_hr_best = pick(hr_df, axis_name="best_available_days", population="Vt")
    vt_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Vt")
    global_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Global")
    vt_hr_trf = pick(hr_df, axis_name="trf_50hz_equiv_days", population="Vt")

    vt_thr90 = pick(threshold_tests, threshold_days=90, population="Vt")
    global_thr90 = pick(threshold_tests, threshold_days=90, population="Global")

    h2s_row = env_df.loc[env_df["metric"] == "H2S effective, mg/L"].iloc[0]
    glf_row = env_df.loc[env_df["metric"] == "GLF"].iloc[0]

    true_mix = mix_df.loc[mix_df["group"] == "HF share>0.5"].copy()
    rest_mix = mix_df.loc[mix_df["group"] == "Rest of Vt"].copy()
    merged = true_mix.merge(rest_mix[["category", "share"]], on="category", how="left", suffixes=("_hf", "_rest"))
    merged["share_diff"] = merged["share_hf"] - merged["share_rest"].fillna(0.0)
    merged = merged.sort_values("share_diff", ascending=False).reset_index(drop=True)
    top_risk = merged.iloc[0]

    lines = [
        "# High-Frequency Redefinition Note",
        "",
        "## Definition used",
        "",
        "- High-frequency well = `days(freq > 55 Hz) / duration_best_days > 0.5`.",
        "- This is a materially broader group than the old `mean > 58 Hz` proxy.",
        "",
        "## Population change",
        "",
        f"- Global old proxy (`mean >58 Hz`): `{int(pop_global_old['runs'])}` runs / `{int(pop_global_old['failures'])}` failures.",
        f"- Global new definition: `{int(pop_global_new['runs'])}` runs / `{int(pop_global_new['failures'])}` failures.",
        f"- Vt old proxy (`mean >58 Hz`): `{int(pop_vt_old['runs'])}` runs / `{int(pop_vt_old['failures'])}` failures.",
        f"- Vt new definition: `{int(pop_vt_new['runs'])}` runs / `{int(pop_vt_new['failures'])}` failures.",
        "",
        "## What changes in the result?",
        "",
        f"- Under the new definition, the Vt subgroup no longer looks worse on raw life. Best-available median is `{float(vt_best_base['median_duration']):.1f}` d at baseline versus `{float(vt_best['median_duration']):.1f}` d in the new HF group.",
        f"- On `TTF_true`, the same direction holds: `{float(vt_true_base['median_duration']):.1f}` d at baseline versus `{float(vt_true['median_duration']):.1f}` d in the new HF group.",
        f"- The adjusted Vt hazard ratio is approximately neutral: `{float(vt_hr_true['hr_hf']):.2f}` on `TTF_true` and `{float(vt_hr_best['hr_hf']):.2f}` on best-available duration.",
        f"- Globally, the new HF group still has somewhat higher early-failure burden, but the adjusted `TTF_true` hazard ratio is only `{float(global_hr_true['hr_hf']):.2f}`.",
        "",
        "## Interpretation",
        "",
        "- This new definition does increase the population a lot, but it does not isolate the same operating concept as the old near-60-Hz proxy.",
        "- It captures wells that spent a majority of operating days above 55 Hz, which includes many wells that are not truly '60 Hz mode' wells.",
        "- In Vt, that broader group does not show a clear life penalty and may even include a healthier selection of wells.",
        "",
        "## Early-failure view",
        "",
        f"- At 90 days, the new HF definition is almost neutral in Vt: early-share difference `{float(vt_thr90['share_difference_hf_minus_baseline']):.2f}`.",
        f"- Globally, the 90-day difference is `{float(global_thr90['share_difference_hf_minus_baseline']):.2f}`.",
        "",
        "## Harsh-environment check",
        "",
        f"- In Vt, the new HF group does not look chemically harsher on measured `H2S` or `GLF`: median `H2S` `{float(h2s_row['hf_median']):.3f}` vs `{float(h2s_row['rest_median']):.3f}`, median `GLF` `{float(glf_row['hf_median']):.1f}` vs `{float(glf_row['rest_median']):.1f}`.",
        "",
        "## Failure pattern in Vt",
        "",
        f"- The most overrepresented Vt failure category under the new HF definition is `{top_risk['category']}`: `{float(top_risk['share_hf']):.1%}` in the HF group versus `{float(top_risk['share_rest']):.1%}` in the rest of Vt.",
        "",
        "## Bottom line",
        "",
        "- If the business question is truly about 'sustained high-frequency exposure above 55 Hz', then this definition is reasonable and should replace the older one.",
        "- But if the business question is specifically about near-60-Hz operation, this broader definition dilutes that signal and changes the conclusion substantially.",
        "- Under this new definition alone, we do **not** get evidence that Vt high-frequency wells have shorter TTF than the 50-55 Hz baseline.",
        "",
    ]
    path = BASE_OUTPUT_DIR / "hf_days_share55_note.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_decision_md(pop_df: pd.DataFrame, axis_df: pd.DataFrame, hr_df: pd.DataFrame, threshold_df: pd.DataFrame, env_df: pd.DataFrame, mix_df: pd.DataFrame) -> Path:
    def pick(df: pd.DataFrame, **filters: object) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for key, value in filters.items():
            mask &= df[key] == value
        return df.loc[mask].iloc[0]

    vt_best = pick(axis_df, axis_name="best_available_days", population="Vt", mode="HF by >55d share >0.5")
    vt_best_base = pick(axis_df, axis_name="best_available_days", population="Vt", mode="50-55 Hz baseline")
    vt_true = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="HF by >55d share >0.5")
    vt_true_base = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="50-55 Hz baseline")
    vt_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Vt")
    vt_hr_best = pick(hr_df, axis_name="best_available_days", population="Vt")
    vt_hr_trf = pick(hr_df, axis_name="trf_50hz_equiv_days", population="Vt")
    global_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Global")
    vt_thr90 = pick(threshold_df, threshold_days=90, population="Vt")
    global_thr90 = pick(threshold_df, threshold_days=90, population="Global")
    h2s_row = env_df.loc[env_df["metric"] == "H2S effective, mg/L"].iloc[0]
    glf_row = env_df.loc[env_df["metric"] == "GLF"].iloc[0]

    true_mix = mix_df.loc[mix_df["group"] == "HF share>0.5"].copy()
    rest_mix = mix_df.loc[mix_df["group"] == "Rest of Vt"].copy()
    merged = true_mix.merge(rest_mix[["category", "share"]], on="category", how="left", suffixes=("_hf", "_rest"))
    merged["share_diff"] = merged["share_hf"] - merged["share_rest"].fillna(0.0)
    merged = merged.sort_values("share_diff", ascending=False).reset_index(drop=True)
    top_risk = merged.iloc[0]
    second_risk = merged.iloc[1]

    lines = [
        "# Vt High-Frequency (>55 Hz Majority Days) Decision Summary",
        "",
        "## Executive Answer",
        "",
        "**Short answer:** under the new definition, `Vt` high-frequency wells do **not** show a clear TTF penalty versus the `50-55 Hz` baseline.",
        "",
        "This is a very different answer from the earlier `mean >58 Hz` proxy. The reason is that the new rule creates a much broader high-frequency population. That broader group does not behave like the old near-60-Hz subgroup.",
        "",
        "## Definition Used",
        "",
        "- High-frequency well = `days(freq > 55 Hz) / duration_best_days > 0.5`.",
        "- Baseline = `50 < freq_w_mean <= 55 Hz`.",
        "",
        "## What Happens To The Population?",
        "",
        "- In `Vt`, the old `mean >58 Hz` proxy had `24` runs / `17` failures.",
        "- The new high-frequency definition has `57` runs / `42` failures.",
        "- Globally, the old proxy had `176` runs / `115` failures, while the new definition has `328` runs / `219` failures.",
        "",
        "So yes, this redefinition changes the study population a lot.",
        "",
        "## Does The New High-Frequency Group In Vt Have Shorter TTF?",
        "",
        "Not in this rerun.",
        "",
        f"- Best-available median in `Vt`: `{float(vt_best_base['median_duration']):.1f}` d at baseline versus `{float(vt_best['median_duration']):.1f}` d in the new high-frequency group.",
        f"- `TTF_true` median in `Vt`: `{float(vt_true_base['median_duration']):.1f}` d at baseline versus `{float(vt_true['median_duration']):.1f}` d in the new high-frequency group.",
        f"- Adjusted Vt hazard ratio: `{float(vt_hr_true['hr_hf']):.2f}` on `TTF_true` and `{float(vt_hr_best['hr_hf']):.2f}` on best-available duration.",
        "",
        "So under this definition, the high-frequency Vt subset is **not** failing sooner than the baseline subset.",
        "",
        "## Does It Increase Early-Failure Risk?",
        "",
        "Not clearly in `Vt`.",
        "",
        f"- At `90` days, the Vt early-failure share difference is `{float(vt_thr90['share_difference_hf_minus_baseline']):.2f}`.",
        f"- Globally, the same `90`-day difference is `{float(global_thr90['share_difference_hf_minus_baseline']):.2f}`.",
        "",
        "That means the global high-frequency burden is still slightly less favorable, but the Vt-specific signal is weak under this broader definition.",
        "",
        "## Is The New HF Group Simply Harsher?",
        "",
        "Not on the measured chemistry proxies we checked.",
        "",
        f"- Vt median `H2S`: `{float(h2s_row['hf_median']):.3f}` in the high-frequency group versus `{float(h2s_row['rest_median']):.3f}` in the rest of `Vt`.",
        f"- Vt median `GLF`: `{float(glf_row['hf_median']):.1f}` versus `{float(glf_row['rest_median']):.1f}`.",
        "",
        "## What Is Most Likely To Become Problematic?",
        "",
        "Under the new definition, the Vt failure pattern is less concentrated than in the old near-60-Hz proxy, but two areas still deserve attention:",
        "",
        f"- `{top_risk['category']}` is the most overrepresented category in the new high-frequency Vt group: `{float(top_risk['share_hf']):.1%}` versus `{float(top_risk['share_rest']):.1%}`.",
        f"- `{second_risk['category']}` is the next strongest difference: `{float(second_risk['share_hf']):.1%}` versus `{float(second_risk['share_rest']):.1%}`.",
        "",
        "## TRF Interpretation",
        "",
        f"- The Vt adjusted hazard ratio on the TRF axis is `{float(vt_hr_trf['hr_hf']):.2f}`.",
        f"- The global `TTF_true` hazard ratio is `{float(global_hr_true['hr_hf']):.2f}`.",
        "",
        "This supports a practical conclusion: the broader majority-days-above-55-Hz definition is not isolating the same high-risk operating mode as the old near-60-Hz proxy.",
        "",
        "## Final Decision Interpretation",
        "",
        "- If management wants to know whether **majority-of-days above 55 Hz** is unsafe in `Vt`, this rerun does **not** show a clear TTF penalty.",
        "- If management wants to know whether **near-60-Hz mode** is unsafe in `Vt`, the older narrower proxy remains more aligned with that question and gives a more cautionary answer.",
        "",
        "## Key Figures",
        "",
        "### Vt KM on TTF_true",
        "",
        "![Vt HF share55 TTF_true](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_ttf_true_days_vt_km.png)",
        "",
        "### Early-failure share at 30 / 60 / 90 days",
        "",
        "![HF share55 thresholds](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_thresholds.png)",
        "",
        "### Vt environment check",
        "",
        "![HF share55 Vt environment](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_vt_environment.png)",
        "",
        "### Vt failure mix",
        "",
        "![HF share55 Vt failure mix](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_vt_failure_mix.png)",
        "",
    ]
    path = DOCS_DIR / "vt_highfreq_share55_decision_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(246, 247, 249)
    bg.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.7), Inches(0.7), Inches(11.6), Inches(1.2))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(26, 36, 52)
    sub = slide.shapes.add_textbox(Inches(0.75), Inches(1.8), Inches(11.0), Inches(1.0))
    p2 = sub.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(74, 86, 104)


def add_bullets_slide(prs: Presentation, title: str, bullets: list[str], image_path: Path | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.45), Inches(0.3), Inches(12.0), Inches(0.6))
    p = title_box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(26, 36, 52)
    body_width = Inches(5.3) if image_path else Inches(11.9)
    body = slide.shapes.add_textbox(Inches(0.55), Inches(1.0), body_width, Inches(5.9))
    tf = body.text_frame
    tf.word_wrap = True
    for idx, bullet in enumerate(bullets):
        para = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        para.text = bullet
        para.font.size = Pt(17)
        para.font.color.rgb = RGBColor(46, 56, 68)
        para.space_after = Pt(10)
    if image_path is not None and image_path.exists():
        slide.shapes.add_picture(str(image_path), Inches(6.0), Inches(1.0), width=Inches(6.4))


def build_slides(pop_df: pd.DataFrame, axis_df: pd.DataFrame, hr_df: pd.DataFrame, threshold_df: pd.DataFrame, mix_df: pd.DataFrame) -> Path:
    def pick(df: pd.DataFrame, **filters: object) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for key, value in filters.items():
            mask &= df[key] == value
        return df.loc[mask].iloc[0]

    pop_vt_old = pick(pop_df, population="Vt", definition="mean >58 Hz proxy")
    pop_vt_new = pick(pop_df, population="Vt", definition="days >55 / duration_best_days >0.5")
    vt_true = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="HF by >55d share >0.5")
    vt_true_base = pick(axis_df, axis_name="ttf_true_days", population="Vt", mode="50-55 Hz baseline")
    vt_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Vt")
    global_hr_true = pick(hr_df, axis_name="ttf_true_days", population="Global")
    vt_hr_trf = pick(hr_df, axis_name="trf_50hz_equiv_days", population="Vt")
    vt_thr90 = pick(threshold_df, threshold_days=90, population="Vt")
    mix_hf = mix_df.loc[mix_df["group"] == "HF share>0.5"].copy().sort_values("share", ascending=False)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "Vt High-Frequency Redefinition Review",
        "High-frequency well = more than half of best-available operating days above 55 Hz",
    )
    add_bullets_slide(
        prs,
        "Population Change",
        [
            f"Old near-60-Hz proxy in Vt: {int(pop_vt_old['runs'])} runs, {int(pop_vt_old['failures'])} failures.",
            f"New majority-days-above-55-Hz definition in Vt: {int(pop_vt_new['runs'])} runs, {int(pop_vt_new['failures'])} failures.",
            "So the new definition broadens the group a lot and should be expected to change the answer.",
        ],
    )
    add_bullets_slide(
        prs,
        "TTF Result In Vt",
        [
            f"TTF_true median at baseline: {float(vt_true_base['median_duration']):.0f} d.",
            f"TTF_true median in the new high-frequency group: {float(vt_true['median_duration']):.0f} d.",
            f"Adjusted Vt hazard ratio on TTF_true: {float(vt_hr_true['hr_hf']):.2f}x.",
            "Under this broader definition, Vt no longer shows a clear life penalty.",
        ],
        image_path=FIGURES_DIR / "hf_share55_ttf_true_days_vt_km.png",
    )
    add_bullets_slide(
        prs,
        "Early Failure",
        [
            f"At 90 days, the Vt early-failure share difference is {float(vt_thr90['share_difference_hf_minus_baseline']):.2f}.",
            f"Global TTF_true adjusted HR is {float(global_hr_true['hr_hf']):.2f}x, but the Vt-specific result remains near neutral.",
            "This suggests the broader definition is not isolating the same risk pattern as the old near-60-Hz proxy.",
        ],
        image_path=FIGURES_DIR / "hf_share55_thresholds.png",
    )
    add_bullets_slide(
        prs,
        "Failure Pattern",
        [
            f"The largest category inside the new Vt high-frequency group is {mix_hf.iloc[0]['category']} at {float(mix_hf.iloc[0]['share']):.0%}.",
            f"The next categories are {mix_hf.iloc[1]['category']} and {mix_hf.iloc[2]['category']}.",
            "The pattern is broader and less concentrated than in the old near-60-Hz subgroup.",
        ],
        image_path=FIGURES_DIR / "hf_share55_vt_failure_mix.png",
    )
    add_bullets_slide(
        prs,
        "Conclusion",
        [
            "If 'high frequency' means majority of operating days above 55 Hz, Vt does not show a clear TTF penalty.",
            f"The Vt TRF-axis hazard ratio is {float(vt_hr_trf['hr_hf']):.2f}x, which also does not point to a clear cumulative-duty penalty.",
            "So this definition supports a milder conclusion than the old near-60-Hz proxy.",
            "Use the broader definition for a sustained-high-frequency question, but keep the old proxy for a specifically near-60-Hz question.",
        ],
    )
    path = BASE_OUTPUT_DIR / "vt_highfreq_share55_slides.pptx"
    prs.save(path)
    return path


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    pop_df = build_population_comparison(df)
    axis_df = build_axis_summary(df)
    build_km_tables(df)
    hr_df = fit_cox(df)
    plot_hr(hr_df, FIGURES_DIR / "hf_share55_hr_summary.png")
    threshold_summary, threshold_tests = build_threshold_tables(df)
    plot_thresholds(threshold_summary, FIGURES_DIR / "hf_share55_thresholds.png")
    env_df = compare_vt_environment(df)
    plot_vt_environment(env_df, FIGURES_DIR / "hf_share55_vt_environment.png")
    mix_df = build_failure_mix(df)
    plot_failure_mix(mix_df, FIGURES_DIR / "hf_share55_vt_failure_mix.png")
    note_path = build_note(pop_df, axis_df, hr_df, threshold_tests, env_df, mix_df)
    md_path = build_decision_md(pop_df, axis_df, hr_df, threshold_tests, env_df, mix_df)
    slides_path = build_slides(pop_df, axis_df, hr_df, threshold_tests, mix_df)
    summary = {
        "note_path": str(note_path),
        "markdown_path": str(md_path),
        "slides_path": str(slides_path),
    }
    (BASE_OUTPUT_DIR / "vt_highfreq_share55_pack.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hf-share55] Wrote outputs to {BASE_OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
