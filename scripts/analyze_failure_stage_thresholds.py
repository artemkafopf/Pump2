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
from scipy.stats import fisher_exact


REPO_ROOT = Path(__file__).resolve().parents[1]
_SLUG = "failure_stage_thresholds"
_PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
BASE_OUTPUT_DIR = results_dir(_SLUG)
FIGURES_DIR = BASE_OUTPUT_DIR / "figures"
TABLES_DIR = BASE_OUTPUT_DIR / "tables"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


THRESHOLDS = [30, 60, 90]
COLORS = {
    "Global | 50-55 Hz baseline": "#2ca02c",
    "Global | >58 Hz true60 proxy": "#d62728",
    "Vt | 50-55 Hz baseline": "#1f77b4",
    "Vt | >58 Hz true60 proxy": "#ff7f0e",
}


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    path = _PROMPT_DIR / "tables" / "analysis_dataset.csv"
    df = pd.read_csv(path)
    for column in ["ttf_true_best_days", "freq_w_mean", "event"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["true60_flag"] = df["freq_w_mean"] > 58.0
    df["baseline_50_55_flag"] = (df["freq_w_mean"] > 50.0) & (df["freq_w_mean"] <= 55.0)
    return df


def build_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    failures = df.loc[(df["event"] == 1) & _numeric(df["ttf_true_best_days"]).notna()].copy()
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        for population, pop_df in [("Global", failures), ("Vt", failures.loc[failures["field"] == "Vt"].copy())]:
            for mode, mode_df in [
                ("50-55 Hz baseline", pop_df.loc[pop_df["baseline_50_55_flag"]].copy()),
                (">58 Hz true60 proxy", pop_df.loc[pop_df["true60_flag"]].copy()),
            ]:
                if mode_df.empty:
                    continue
                early = (mode_df["ttf_true_best_days"] < threshold).astype(int)
                rows.append(
                    {
                        "threshold_days": threshold,
                        "population": population,
                        "mode": mode,
                        "failures_with_ttf_true": int(len(mode_df)),
                        "early_failures": int(early.sum()),
                        "mature_failures": int(len(mode_df) - early.sum()),
                        "early_share_failures": float(early.mean()),
                    }
                )
    frame = pd.DataFrame(rows)

    test_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        for population, pop_df in [("Global", failures), ("Vt", failures.loc[failures["field"] == "Vt"].copy())]:
            base = pop_df.loc[pop_df["baseline_50_55_flag"]].copy()
            true60 = pop_df.loc[pop_df["true60_flag"]].copy()
            if base.empty or true60.empty:
                continue
            base_early = int((base["ttf_true_best_days"] < threshold).sum())
            base_mature = int(len(base) - base_early)
            true_early = int((true60["ttf_true_best_days"] < threshold).sum())
            true_mature = int(len(true60) - true_early)
            odds_ratio, p_value = fisher_exact([[true_early, true_mature], [base_early, base_mature]], alternative="two-sided")
            test_rows.append(
                {
                    "threshold_days": threshold,
                    "population": population,
                    "baseline_failures": int(len(base)),
                    "true60_failures": int(len(true60)),
                    "baseline_early_share": float(base_early / max(len(base), 1)),
                    "true60_early_share": float(true_early / max(len(true60), 1)),
                    "share_difference_true60_minus_baseline": float((true_early / max(len(true60), 1)) - (base_early / max(len(base), 1))),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                }
            )

    frame.to_csv(TABLES_DIR / "failure_stage_threshold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(TABLES_DIR / "failure_stage_threshold_tests.csv", index=False, encoding="utf-8-sig")
    return frame


def build_reason_mix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    failures = df.loc[(df["event"] == 1) & _numeric(df["ttf_true_best_days"]).notna()].copy()
    mix_rows: list[dict[str, object]] = []
    skew_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        for population, pop_df in [("Global", failures), ("Vt", failures.loc[failures["field"] == "Vt"].copy())]:
            work = pop_df.copy()
            work["stage"] = np.where(work["ttf_true_best_days"] < threshold, "early", "mature")
            counts = (
                work.groupby(["stage", "failure_category_raw"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
                .rename(columns={"failure_category_raw": "category"})
            )
            counts["threshold_days"] = threshold
            counts["population"] = population
            counts["stage_total"] = counts.groupby("stage")["count"].transform("sum")
            counts["share"] = counts["count"] / counts["stage_total"]
            mix_rows.extend(counts[["threshold_days", "population", "stage", "category", "count", "share"]].to_dict(orient="records"))

            pivot = counts.pivot(index="category", columns="stage", values="share").fillna(0.0)
            count_pivot = counts.pivot(index="category", columns="stage", values="count").fillna(0.0)
            for category in pivot.index.tolist():
                skew_rows.append(
                    {
                        "threshold_days": threshold,
                        "population": population,
                        "category": category,
                        "early_count": int(count_pivot.loc[category, "early"]) if "early" in count_pivot.columns else 0,
                        "mature_count": int(count_pivot.loc[category, "mature"]) if "mature" in count_pivot.columns else 0,
                        "early_share": float(pivot.loc[category, "early"]) if "early" in pivot.columns else 0.0,
                        "mature_share": float(pivot.loc[category, "mature"]) if "mature" in pivot.columns else 0.0,
                        "early_minus_mature_share": float(
                            (pivot.loc[category, "early"] if "early" in pivot.columns else 0.0)
                            - (pivot.loc[category, "mature"] if "mature" in pivot.columns else 0.0)
                        ),
                    }
                )
    mix_df = pd.DataFrame(mix_rows)
    skew_df = pd.DataFrame(skew_rows)
    mix_df.to_csv(TABLES_DIR / "failure_stage_reason_mix.csv", index=False, encoding="utf-8-sig")
    skew_df.to_csv(TABLES_DIR / "failure_stage_reason_skew.csv", index=False, encoding="utf-8-sig")
    return mix_df, skew_df


def plot_threshold_lines(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for (population, mode), subset in summary_df.groupby(["population", "mode"], dropna=False):
        subset = subset.sort_values("threshold_days")
        label = f"{population} | {mode}"
        ax.plot(
            subset["threshold_days"],
            subset["early_share_failures"],
            marker="o",
            linewidth=2,
            color=COLORS.get(label, "#555555"),
            label=label,
        )
        for _, row in subset.iterrows():
            ax.text(float(row["threshold_days"]), float(row["early_share_failures"]) + 0.01, f"{float(row['early_share_failures']):.2f}", fontsize=8, ha="center")
    ax.set_xticks(THRESHOLDS)
    ax.set_xlabel("Early / mature cutoff in TTF_true days")
    ax.set_ylabel("Early share among failures")
    ax.set_title("60 Hz sensitivity: early-failure share at 30 / 60 / 90 days")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_stage_skew_heatmap(skew_df: pd.DataFrame, population: str, output_path: Path) -> None:
    plot_df = skew_df.loc[skew_df["population"] == population].copy()
    pivot = plot_df.pivot(index="category", columns="threshold_days", values="early_minus_mature_share").fillna(0.0)
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    data = pivot.to_numpy(dtype=float)
    vmax = max(0.05, float(np.abs(data).max()))
    image = ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns.tolist())
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.tolist())
    ax.set_title(f"{population}: which reasons skew early vs mature?")
    ax.set_xlabel("Threshold days")
    ax.set_ylabel("Failure category")
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            ax.text(col_idx, row_idx, f"{data[row_idx, col_idx]:.02f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, shrink=0.85, label="Early share minus mature share")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_note(summary_df: pd.DataFrame, test_df: pd.DataFrame, skew_df: pd.DataFrame) -> Path:
    def metric(threshold: int, population: str, mode: str) -> pd.Series:
        return summary_df.loc[
            (summary_df["threshold_days"] == threshold)
            & (summary_df["population"] == population)
            & (summary_df["mode"] == mode)
        ].iloc[0]

    def test_metric(threshold: int, population: str) -> pd.Series:
        return test_df.loc[(test_df["threshold_days"] == threshold) & (test_df["population"] == population)].iloc[0]

    def top_skew(threshold: int, population: str, ascending: bool) -> pd.Series:
        subset = skew_df.loc[(skew_df["threshold_days"] == threshold) & (skew_df["population"] == population)].copy()
        subset = subset.sort_values("early_minus_mature_share", ascending=ascending).reset_index(drop=True)
        return subset.iloc[0]

    lines = [
        "# Failure Stage Threshold Note",
        "",
        "## 60 Hz sensitivity at 30 / 60 / 90 days",
        "",
        f"- Global early-failure share at `>58 Hz` versus `50-55 Hz` rises from `{float(metric(30, 'Global', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(30, 'Global', '50-55 Hz baseline')['early_share_failures']):.2f}` at 30 days, to `{float(metric(60, 'Global', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(60, 'Global', '50-55 Hz baseline')['early_share_failures']):.2f}` at 60 days, and `{float(metric(90, 'Global', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(90, 'Global', '50-55 Hz baseline')['early_share_failures']):.2f}` at 90 days.",
        f"- Inside `Vt`, the same direction holds: `{float(metric(30, 'Vt', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(30, 'Vt', '50-55 Hz baseline')['early_share_failures']):.2f}` at 30 days, `{float(metric(60, 'Vt', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(60, 'Vt', '50-55 Hz baseline')['early_share_failures']):.2f}` at 60 days, and `{float(metric(90, 'Vt', '>58 Hz true60 proxy')['early_share_failures']):.2f}` vs `{float(metric(90, 'Vt', '50-55 Hz baseline')['early_share_failures']):.2f}` at 90 days.",
        f"- The global 90-day gap is the clearest of the three thresholds: absolute difference `{float(test_metric(90, 'Global')['share_difference_true60_minus_baseline']):.2f}`. The Vt 90-day gap is larger in raw terms (`{float(test_metric(90, 'Vt')['share_difference_true60_minus_baseline']):.2f}`) but still sample-limited.",
        "",
        "## What makes an early failure different from a mature failure?",
        "",
        f"- Globally at 60 days, the most early-skewed category is `{top_skew(60, 'Global', ascending=False)['category']}` and the most mature-skewed category is `{top_skew(60, 'Global', ascending=True)['category']}`.",
        f"- In `Vt` at 60 days, the strongest early-skewed category is `{top_skew(60, 'Vt', ascending=False)['category']}`, while the strongest mature-skewed category is `{top_skew(60, 'Vt', ascending=True)['category']}`.",
        f"- At 90 days, `Vt` still leans early toward `{top_skew(90, 'Vt', ascending=False)['category']}` and away from `{top_skew(90, 'Vt', ascending=True)['category']}`.",
        "",
        "## Plain-language interpretation",
        "",
        "- Early failures are not just the same failures happening a bit sooner. The category mix shifts.",
        "- In Vt, earlier failures are relatively more concentrated in clogging / working-organ issues and some fast mechanical/electrical events, while more mature failures lean relatively more toward motor, tubing, and seal-protection related categories.",
        "- That means the 60 Hz question should be asked in two layers: does higher frequency increase the chance of failing early, and does it also change which mechanism fails first?",
        "",
    ]
    path = BASE_OUTPUT_DIR / "failure_stage_threshold_note.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    summary_df = build_threshold_summary(df)
    test_df = pd.read_csv(TABLES_DIR / "failure_stage_threshold_tests.csv")
    _, skew_df = build_reason_mix(df)
    plot_threshold_lines(summary_df, FIGURES_DIR / "true60_threshold_early_share_ttf_true.png")
    plot_stage_skew_heatmap(skew_df, "Global", FIGURES_DIR / "failure_stage_skew_global.png")
    plot_stage_skew_heatmap(skew_df, "Vt", FIGURES_DIR / "failure_stage_skew_vt.png")
    note_path = build_note(summary_df, test_df, skew_df)
    summary = {
        "threshold_summary_table": str(TABLES_DIR / "failure_stage_threshold_summary.csv"),
        "threshold_test_table": str(TABLES_DIR / "failure_stage_threshold_tests.csv"),
        "reason_mix_table": str(TABLES_DIR / "failure_stage_reason_mix.csv"),
        "reason_skew_table": str(TABLES_DIR / "failure_stage_reason_skew.csv"),
        "note_path": str(note_path),
    }
    (BASE_OUTPUT_DIR / "failure_stage_threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[failure-stage] Wrote outputs to {BASE_OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
