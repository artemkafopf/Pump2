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


REPO_ROOT = Path(__file__).resolve().parents[1]
_SLUG = "true60_axis_sensitivity"
_PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
BASE_OUTPUT_DIR = results_dir(_SLUG)
FIGURES_DIR = BASE_OUTPUT_DIR / "figures"
TABLES_DIR = BASE_OUTPUT_DIR / "tables"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


COLORS = {
    "baseline": "#2ca02c",
    "true60": "#d62728",
    "global": "#1f77b4",
    "vt": "#ff7f0e",
    "best_available_days": "#4c78a8",
    "ttf_true_days": "#f58518",
    "trf_50hz_equiv_days": "#54a24b",
}


AXES = [
    {
        "axis_name": "best_available_days",
        "duration_col": "duration_best_days",
        "label": "Best-available days",
        "early_threshold": 90.0,
        "mask_name": "all_non_missing",
    },
    {
        "axis_name": "ttf_true_days",
        "duration_col": "ttf_true_best_days",
        "label": "TTF_true days only",
        "early_threshold": 90.0,
        "mask_name": "ttf_true_non_missing",
    },
    {
        "axis_name": "trf_50hz_equiv_days",
        "duration_col": "trf_50hz_equiv_days",
        "label": "TRF axis (50 Hz-equivalent days)",
        "early_threshold": 90.0,
        "mask_name": "trf_cov50",
    },
]


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    path = _PROMPT_DIR / "tables" / "analysis_dataset.csv"
    df = pd.read_csv(path)
    for column in [
        "duration_best_days",
        "run_days",
        "ttf_true_best_days",
        "freq_w_mean",
        "h2s_effective_mg_l",
        "avg_glf",
        "mount_year",
        "total_freq_hz_days",
        "n_freq_valid_days",
        "event",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["true60_flag"] = df["freq_w_mean"] > 58.0
    df["baseline_50_55_flag"] = (df["freq_w_mean"] > 50.0) & (df["freq_w_mean"] <= 55.0)
    df["high_h2s_flag"] = df["h2s_effective_mg_l"] >= 3.0
    df["trf_50hz_equiv_days"] = df["total_freq_hz_days"] / 50.0
    df["freq_coverage_to_ttf_true"] = _numeric(df["n_freq_valid_days"]) / _numeric(df["ttf_true_best_days"])
    return df


def axis_mask(df: pd.DataFrame, axis_name: str) -> pd.Series:
    if axis_name == "best_available_days":
        return _numeric(df["duration_best_days"]).notna()
    if axis_name == "ttf_true_days":
        return _numeric(df["ttf_true_best_days"]).notna()
    if axis_name == "trf_50hz_equiv_days":
        return (
            _numeric(df["trf_50hz_equiv_days"]).notna()
            & _numeric(df["ttf_true_best_days"]).notna()
            & (_numeric(df["freq_coverage_to_ttf_true"]) >= 0.5)
        )
    raise ValueError(f"Unsupported axis: {axis_name}")


def describe_mode(subset: pd.DataFrame, duration_col: str, early_threshold: float, axis_name: str, population: str, mode: str) -> dict[str, object]:
    failures = int(subset["event"].sum())
    runs = int(len(subset))
    early = ((subset["event"] == 1) & (_numeric(subset[duration_col]) < early_threshold)).astype(int)
    return {
        "axis_name": axis_name,
        "population": population,
        "mode": mode,
        "runs": runs,
        "failures": failures,
        "failure_rate": float(failures / max(runs, 1)),
        "median_duration": float(_numeric(subset[duration_col]).median()) if runs else float("nan"),
        "mean_duration": float(_numeric(subset[duration_col]).mean()) if runs else float("nan"),
        "early_rate_runs": float(early.mean()) if runs else float("nan"),
        "early_share_failures": float(early.sum() / max(failures, 1)),
    }


def build_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_spec in AXES:
        axis_name = axis_spec["axis_name"]
        duration_col = axis_spec["duration_col"]
        threshold = axis_spec["early_threshold"]
        mask = axis_mask(df, axis_name)
        compare = df.loc[mask].copy()
        for population, pop_df in [("Global", compare), ("Vt", compare.loc[compare["field"] == "Vt"].copy())]:
            rows.append(
                describe_mode(
                    pop_df.loc[pop_df["baseline_50_55_flag"]].copy(),
                    duration_col,
                    threshold,
                    axis_name,
                    population,
                    "50-55 Hz baseline",
                )
            )
            rows.append(
                describe_mode(
                    pop_df.loc[pop_df["true60_flag"]].copy(),
                    duration_col,
                    threshold,
                    axis_name,
                    population,
                    ">58 Hz true60 proxy",
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "true60_axis_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_pair_km(base: pd.DataFrame, true60: pd.DataFrame, duration_col: str, title: str, output_path: Path) -> dict[str, float]:
    km_base = KaplanMeierFitter()
    km_true = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    km_base.fit(base[duration_col], base["event"], label=f"50-55 Hz baseline (n={len(base)}, fail={int(base['event'].sum())})")
    km_true.fit(true60[duration_col], true60["event"], label=f">58 Hz true60 proxy (n={len(true60)}, fail={int(true60['event'].sum())})")
    km_base.plot_survival_function(ax=ax, ci_show=True, color=COLORS["baseline"], linewidth=2)
    km_true.plot_survival_function(ax=ax, ci_show=True, color=COLORS["true60"], linewidth=2)
    result = logrank_test(base[duration_col], true60[duration_col], base["event"], true60["event"])
    ax.set_title(title)
    ax.set_xlabel("Axis duration")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.05,
        f"log-rank p = {result.p_value:.3f}\nMedian = {km_base.median_survival_time_:.0f} vs {km_true.median_survival_time_:.0f}",
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
        "true60_median": float(km_true.median_survival_time_),
    }


def make_km_outputs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_spec in AXES:
        axis_name = axis_spec["axis_name"]
        duration_col = axis_spec["duration_col"]
        mask = axis_mask(df, axis_name)
        for population, pop_df in [("Global", df.loc[mask].copy()), ("Vt", df.loc[mask & df["field"].eq("Vt")].copy())]:
            base = pop_df.loc[pop_df["baseline_50_55_flag"]].copy()
            true60 = pop_df.loc[pop_df["true60_flag"]].copy()
            if len(base) == 0 or len(true60) == 0 or int(base["event"].sum()) < 6 or int(true60["event"].sum()) < 6:
                continue
            stats = plot_pair_km(
                base,
                true60,
                duration_col,
                f"{population}: {axis_spec['label']} — >58 Hz vs 50-55 Hz",
                FIGURES_DIR / f"{axis_name}_{population.lower()}_km.png",
            )
            rows.append(
                {
                    "axis_name": axis_name,
                    "population": population,
                    "baseline_runs": int(len(base)),
                    "baseline_failures": int(base["event"].sum()),
                    "true60_runs": int(len(true60)),
                    "true60_failures": int(true60["event"].sum()),
                    **stats,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "true60_axis_sensitivity_km.csv", index=False, encoding="utf-8-sig")
    return frame


def fit_axis_cox_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_spec in AXES:
        axis_name = axis_spec["axis_name"]
        duration_col = axis_spec["duration_col"]
        mask = axis_mask(df, axis_name)

        for model_name, pop_mask, include_field in [
            ("Global", pd.Series(True, index=df.index), True),
            ("Vt", df["field"] == "Vt", False),
        ]:
            work = df.loc[mask & pop_mask & (df["baseline_50_55_flag"] | df["true60_flag"])].copy()
            work["duration"] = _numeric(work[duration_col])
            work["true60"] = work["true60_flag"].astype(int)
            work["high_h2s"] = work["high_h2s_flag"].astype(int)
            work["avg_glf"] = _numeric(work["avg_glf"])
            work["mount_year"] = _numeric(work["mount_year"])

            keep = ["duration", "event", "true60", "high_h2s", "avg_glf", "mount_year", "Скв."]
            if include_field:
                field_counts = work["field"].astype("string").value_counts()
                keep_fields = field_counts.loc[field_counts >= 20].index.astype(str).tolist()
                work = work.loc[work["field"].astype("string").isin(keep_fields)].copy()
                keep.append("field")
            work = work[keep].dropna().copy()
            if len(work) == 0 or int(work["true60"].sum()) == 0:
                continue

            formula = "true60 + high_h2s + avg_glf + mount_year"
            if include_field:
                formula += " + C(field)"

            cph = CoxPHFitter()
            cph.fit(work, duration_col="duration", event_col="event", cluster_col="Скв.", robust=True, formula=formula)
            row = cph.summary.loc["true60"]
            rows.append(
                {
                    "axis_name": axis_name,
                    "population": model_name,
                    "rows": int(len(work)),
                    "events": int(work["event"].sum()),
                    "hr_true60": float(row["exp(coef)"]),
                    "hr_lower_95": float(row["exp(coef) lower 95%"]),
                    "hr_upper_95": float(row["exp(coef) upper 95%"]),
                    "p_value": float(row["p"]),
                    "concordance": float(cph.concordance_index_),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "true60_axis_sensitivity_hr.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_axis_hr(hr_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = hr_df.copy()
    plot_df["label"] = plot_df["population"] + " — " + plot_df["axis_name"]
    plot_df["color"] = plot_df["axis_name"].map(COLORS)
    y = np.arange(len(plot_df))
    hr = plot_df["hr_true60"].to_numpy(dtype=float)
    lo = plot_df["hr_lower_95"].to_numpy(dtype=float)
    hi = plot_df["hr_upper_95"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.errorbar(
            row["hr_true60"],
            idx,
            xerr=[[row["hr_true60"] - row["hr_lower_95"]], [row["hr_upper_95"] - row["hr_true60"]]],
            fmt="o",
            color=row["color"],
            ecolor="#555555",
            capsize=4,
        )
        ax.text(float(row["hr_upper_95"]) * 1.02, idx, f"p={float(row['p_value']):.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, plot_df["label"].tolist())
    ax.set_xlabel("Adjusted hazard ratio for >58 Hz vs 50-55 Hz")
    ax.set_title("60 Hz sensitivity: does the answer depend on the time axis?")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_true = df.loc[_numeric(df["ttf_true_best_days"]).notna()].copy()
    vt_true = all_true.loc[all_true["field"] == "Vt"].copy()
    for population, pop_df in [("Global", all_true), ("Vt", vt_true)]:
        coverage = _numeric(pop_df["freq_coverage_to_ttf_true"])
        rows.append(
            {
                "population": population,
                "rows_with_ttf_true": int(len(pop_df)),
                "rows_with_trf": int(_numeric(pop_df["trf_50hz_equiv_days"]).notna().sum()),
                "median_freq_coverage_to_ttf_true": float(coverage.median()) if coverage.notna().any() else float("nan"),
                "share_coverage_ge_0p5": float((coverage >= 0.5).mean()) if coverage.notna().any() else float("nan"),
                "share_coverage_ge_0p7": float((coverage >= 0.7).mean()) if coverage.notna().any() else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "true60_trf_coverage_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def build_note(summary_df: pd.DataFrame, km_df: pd.DataFrame, hr_df: pd.DataFrame, coverage_df: pd.DataFrame) -> Path:
    def get_summary(axis_name: str, population: str, mode: str) -> pd.Series:
        return summary_df.loc[
            (summary_df["axis_name"] == axis_name)
            & (summary_df["population"] == population)
            & (summary_df["mode"] == mode)
        ].iloc[0]

    def get_hr(axis_name: str, population: str) -> pd.Series:
        return hr_df.loc[(hr_df["axis_name"] == axis_name) & (hr_df["population"] == population)].iloc[0]

    def get_km(axis_name: str, population: str) -> pd.Series:
        return km_df.loc[(km_df["axis_name"] == axis_name) & (km_df["population"] == population)].iloc[0]

    best_global_hr = get_hr("best_available_days", "Global")
    true_global_hr = get_hr("ttf_true_days", "Global")
    trf_global_hr = get_hr("trf_50hz_equiv_days", "Global")
    best_vt_hr = get_hr("best_available_days", "Vt")
    true_vt_hr = get_hr("ttf_true_days", "Vt")
    trf_vt_hr = get_hr("trf_50hz_equiv_days", "Vt")
    true_global_km = get_km("ttf_true_days", "Global")
    trf_vt_km = get_km("trf_50hz_equiv_days", "Vt")

    global_cov = coverage_df.loc[coverage_df["population"] == "Global"].iloc[0]
    vt_cov = coverage_df.loc[coverage_df["population"] == "Vt"].iloc[0]

    lines = [
        "# 60 Hz Axis Sensitivity Note",
        "",
        "## Why this pass matters",
        "",
        "- The main analysis already used `duration_best_days`, which mixes `TTF_true` where available with calendar run days as fallback.",
        "- This sensitivity pass asks two separate questions:",
        "  1. Does the 60 Hz signal get clearer if we use only `TTF_true`?",
        "  2. Does the 60 Hz penalty remain if we switch from calendar time to cumulative frequency duty (`TRF`)?",
        "",
        "## TTF_true-only result",
        "",
        f"- Global adjusted HR for `>58 Hz` vs `50-55 Hz` moves from `{float(best_global_hr['hr_true60']):.2f}` on the mixed axis to `{float(true_global_hr['hr_true60']):.2f}` on pure `TTF_true` (`p={float(true_global_hr['p_value']):.3f}`).",
        f"- Global `TTF_true` KM medians also separate more clearly: `50-55 Hz` = `{float(true_global_km['baseline_median']):.0f}` d vs `>58 Hz` = `{float(true_global_km['true60_median']):.0f}` d.",
        f"- Inside `Vt`, the direction stays worse at `>58 Hz`, but the adjusted HR remains imprecise: `{float(true_vt_hr['hr_true60']):.2f}` (`p={float(true_vt_hr['p_value']):.3f}`).",
        "",
        "## TRF-axis result",
        "",
        f"- On the `TRF` axis expressed as `50 Hz-equivalent days`, the global adjusted HR shrinks to `{float(trf_global_hr['hr_true60']):.2f}` (`p={float(trf_global_hr['p_value']):.3f}`).",
        f"- In `Vt`, the adjusted HR on the TRF axis is `{float(trf_vt_hr['hr_true60']):.2f}` (`p={float(trf_vt_hr['p_value']):.3f}`).",
        f"- The Vt KM median flips direction on the TRF axis: baseline `{float(trf_vt_km['baseline_median']):.0f}` vs `>58 Hz` `{float(trf_vt_km['true60_median']):.0f}` equivalent days.",
        f"- That means `>58 Hz` pumps can look worse in calendar days while not looking worse in cumulative frequency-duty to failure.",
        "",
        "## Interpretation",
        "",
        "- The `TTF_true` pass strengthens the case that calendar-style operating life is shorter at `>58 Hz`, especially globally.",
        "- The `TRF` pass weakens the case that high-frequency pumps fail after less cumulative duty. In simpler terms, the pump seems to get through its usable work faster at higher frequency, rather than clearly failing after less total accumulated frequency work.",
        "- Operationally, both statements can be true at once: the pump may still fail sooner on the calendar, but not necessarily after fewer total Hz-days.",
        "",
        "## Coverage caveat",
        "",
        f"- TRF sensitivity was restricted to runs with both `TTF_true` and at least 50% telemetry coverage of the true operating interval. Coverage at that threshold is `{float(global_cov['share_coverage_ge_0p5']):.2%}` globally and `{float(vt_cov['share_coverage_ge_0p5']):.2%}` in `Vt`.",
        "- So the TRF result is informative, but it applies to the telemetry-covered subset rather than the full population.",
        "",
    ]
    path = BASE_OUTPUT_DIR / "vt_60hz_axis_sensitivity_note.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    summary_df = build_axis_summary(df)
    km_df = make_km_outputs(df)
    hr_df = fit_axis_cox_models(df)
    coverage_df = build_coverage_summary(df)
    plot_axis_hr(hr_df, FIGURES_DIR / "true60_axis_sensitivity_hr.png")
    note_path = build_note(summary_df, km_df, hr_df, coverage_df)

    summary = {
        "summary_table": str(TABLES_DIR / "true60_axis_sensitivity_summary.csv"),
        "km_table": str(TABLES_DIR / "true60_axis_sensitivity_km.csv"),
        "hr_table": str(TABLES_DIR / "true60_axis_sensitivity_hr.csv"),
        "coverage_table": str(TABLES_DIR / "true60_trf_coverage_summary.csv"),
        "note_path": str(note_path),
    }
    (BASE_OUTPUT_DIR / "vt_60hz_axis_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[axis-sensitivity] Wrote outputs to {BASE_OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
