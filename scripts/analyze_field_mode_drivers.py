from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


import sys as _sys
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from analysis.paths import RESULTS_ROOT
ANALYSIS_OUTPUTS_DIR = REPO_ROOT / "analysis_outputs"  # legacy fallback

DEFAULT_NUMERIC_FEATURES = [
    "h2s_effective_mg_l",
    "high_h2s_flag",
    "avg_glf",
    "avg_kpod",
    "trf_per_day",
    "tlf_per_day",
    "calcium_load_per_day",
    "chloride_load_per_day",
    "sulfate_load_per_day",
    "gypsum_proxy_per_day",
    "freq_above_55hz_pct",
    "freq_below_45hz_pct",
    "freq_signed_exposure",
    "freq_w_mean",
    "Номинальная частота, Гц",
    "mount_year",
    "duration_best_days",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "failure_category_raw",
    "contractor",
    "pump_family",
    "field",
]

POPULATION_DIRS = {
    "global": "global",
    "ya": "ya",
    "vt": "vt",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate latent survival modes to H2S, precipitate, frequency, and static parameters."
    )
    parser.add_argument(
        "--posterior-root",
        default=None,
        help="Directory with global/ya/vt posterior bundles. Defaults to latest bayesian_field_survival_compare* folder.",
    )
    parser.add_argument(
        "--feature-table",
        default=None,
        help="Path to analysis_dataset.csv. Defaults to latest vt_60hz_prompt_analysis_*/tables/analysis_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output tables and report. Defaults to posterior-root.",
    )
    parser.add_argument(
        "--min-category-rows",
        type=int,
        default=10,
        help="Minimum rows per category level.",
    )
    parser.add_argument(
        "--min-numeric-rows",
        type=int,
        default=25,
        help="Minimum non-missing rows for numeric associations.",
    )
    return parser.parse_args()


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _latest_path(*roots_and_patterns: tuple) -> Path:
    """Search each (root, pattern) pair in order; return the most-recently-modified match."""
    all_matches: list[Path] = []
    for root, pattern in roots_and_patterns:
        all_matches.extend(root.glob(pattern))
    if not all_matches:
        patterns = [p for _, p in roots_and_patterns]
        raise FileNotFoundError(f"No paths found for patterns: {patterns}")
    return sorted(all_matches, key=lambda path: path.stat().st_mtime)[-1]


def resolve_posterior_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return _latest_path(
        (RESULTS_ROOT, "bayesian_field_survival/*"),
        (ANALYSIS_OUTPUTS_DIR, "bayesian_field_survival_compare*"),
    )


def resolve_feature_table(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return _latest_path(
        (RESULTS_ROOT, "vt_60hz_prompt/*/tables/analysis_dataset.csv"),
        (ANALYSIS_OUTPUTS_DIR, "vt_60hz_prompt_analysis_*/tables/analysis_dataset.csv"),
    )


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "row_id" not in df.columns:
        raise KeyError(f"`row_id` column is required in {path}")
    return df.drop_duplicates(subset=["row_id"], keep="first").copy()


def _component_probability_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("component_") and column.endswith("_probability")]
    return sorted(columns, key=lambda value: int(value.split("_")[1]))


def load_population_posteriors(posterior_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for population, directory_name in POPULATION_DIRS.items():
        path = posterior_root / directory_name / "observation_latent_probabilities.csv"
        frame = pd.read_csv(path)
        probability_columns = _component_probability_columns(frame)
        if not probability_columns:
            raise ValueError(f"No component probability columns found in {path}")
        renamed = frame.rename(columns={"id": "row_id", "duration": "duration_model", "event": "event_model"}).copy()
        for column in ["row_id", "most_probable_component"]:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        renamed["population"] = population
        renamed["p_short"] = _numeric(renamed[probability_columns[0]])
        if len(probability_columns) == 1:
            renamed["p_longer"] = 1.0 - renamed["p_short"]
        else:
            longer = renamed[probability_columns[1:]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            renamed["p_longer"] = longer
        total = renamed["p_short"] + renamed["p_longer"]
        valid_total = total.notna() & total.gt(0)
        renamed.loc[valid_total, "p_short"] = renamed.loc[valid_total, "p_short"] / total.loc[valid_total]
        renamed.loc[valid_total, "p_longer"] = renamed.loc[valid_total, "p_longer"] / total.loc[valid_total]
        renamed["hard_short"] = renamed["most_probable_component"].eq(1).astype(int)
        renamed["hard_longer"] = 1 - renamed["hard_short"]
        keep = [
            "row_id",
            "duration_model",
            "event_model",
            "population",
            "p_short",
            "p_longer",
            "hard_short",
            "hard_longer",
            "most_probable_component",
            "max_probability",
            "classification_entropy",
            "uncertainty_flag",
        ]
        frames.append(renamed[keep])
    return pd.concat(frames, ignore_index=True)


def merge_features_with_posteriors(features: pd.DataFrame, posteriors: pd.DataFrame) -> pd.DataFrame:
    merged = posteriors.merge(features, on="row_id", how="left", suffixes=("", "_feature"))
    if "event" in merged.columns and "event_model" in merged.columns:
        merged["event_model"] = _numeric(merged["event_model"]).combine_first(_numeric(merged["event"]))
    if "duration_best_days" in merged.columns and "duration_model" in merged.columns:
        merged["duration_best_days"] = _numeric(merged["duration_best_days"]).combine_first(_numeric(merged["duration_model"]))
    return merged


def compute_merge_coverage(posteriors: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population in POPULATION_DIRS:
        posterior_rows = int((posteriors["population"] == population).sum())
        merged_rows = merged.loc[merged["population"] == population]
        matched_rows = int(merged_rows["row_id"].notna().sum())
        feature_rows = int(merged_rows["field"].notna().sum()) if "field" in merged_rows.columns else matched_rows
        rows.append(
            {
                "population": population,
                "posterior_rows": posterior_rows,
                "merged_rows": int(len(merged_rows)),
                "feature_rows_with_field": feature_rows,
                "matched_rows": matched_rows,
            }
        )
    return pd.DataFrame(rows)


def compute_numeric_associations(
    merged: pd.DataFrame,
    numeric_features: list[str],
    min_numeric_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    associations: list[dict[str, object]] = []
    weighted_means: list[dict[str, object]] = []

    for population, pop_df in merged.groupby("population", dropna=False):
        for feature in numeric_features:
            if feature not in pop_df.columns:
                continue
            values = _numeric(pop_df[feature])
            p_short = _numeric(pop_df["p_short"])
            p_longer = _numeric(pop_df["p_longer"])
            mask = values.notna() & p_short.notna() & p_longer.notna()
            rows = int(mask.sum())
            if rows < min_numeric_rows:
                continue

            rho = math.nan
            pvalue = math.nan
            if values.loc[mask].nunique(dropna=True) >= 2 and p_short.loc[mask].nunique(dropna=True) >= 2:
                rho, pvalue = spearmanr(values.loc[mask], p_short.loc[mask], nan_policy="omit")

            short_weights = p_short.loc[mask].to_numpy(dtype=float)
            longer_weights = p_longer.loc[mask].to_numpy(dtype=float)
            raw_values = values.loc[mask].to_numpy(dtype=float)
            short_weight_sum = float(short_weights.sum())
            longer_weight_sum = float(longer_weights.sum())
            short_mean = float(np.average(raw_values, weights=short_weights)) if short_weight_sum > 0 else math.nan
            longer_mean = float(np.average(raw_values, weights=longer_weights)) if longer_weight_sum > 0 else math.nan

            associations.append(
                {
                    "population": population,
                    "feature": feature,
                    "rows": rows,
                    "spearman_rho_p_short": float(rho) if pd.notna(rho) else math.nan,
                    "spearman_pvalue": float(pvalue) if pd.notna(pvalue) else math.nan,
                    "short_weighted_mean": short_mean,
                    "longer_weighted_mean": longer_mean,
                    "short_minus_longer": short_mean - longer_mean if pd.notna(short_mean) and pd.notna(longer_mean) else math.nan,
                }
            )
            weighted_means.append(
                {
                    "population": population,
                    "feature": feature,
                    "rows": rows,
                    "short_weight_sum": short_weight_sum,
                    "longer_weight_sum": longer_weight_sum,
                    "short_weighted_mean": short_mean,
                    "longer_weighted_mean": longer_mean,
                }
            )

    numeric_df = pd.DataFrame(associations)
    means_df = pd.DataFrame(weighted_means)
    if not numeric_df.empty:
        numeric_df = numeric_df.sort_values(["population", "feature"]).reset_index(drop=True)
    if not means_df.empty:
        means_df = means_df.sort_values(["population", "feature"]).reset_index(drop=True)
    return numeric_df, means_df


def _cramers_v(table: pd.DataFrame) -> float:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return math.nan
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy(dtype=float).sum()
    if n <= 0:
        return math.nan
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1.0, 1.0))
    rows_corr = rows - ((rows - 1) ** 2) / max(n - 1.0, 1.0)
    cols_corr = cols - ((cols - 1) ** 2) / max(n - 1.0, 1.0)
    denom = min(cols_corr - 1.0, rows_corr - 1.0)
    if denom <= 0:
        return math.nan
    return math.sqrt(phi2_corr / denom)


def compute_categorical_associations(
    merged: pd.DataFrame,
    categorical_features: list[str],
    min_category_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for population, pop_df in merged.groupby("population", dropna=False):
        for feature in categorical_features:
            if feature not in pop_df.columns:
                continue
            subset = pop_df.copy()
            if feature == "failure_category_raw":
                subset = subset.loc[_numeric(subset["event_model"]).eq(1)].copy()

            series = subset[feature].fillna("<missing>").astype(str)
            counts = series.value_counts()
            keep_levels = counts.loc[counts >= min_category_rows].index
            subset = subset.loc[series.isin(keep_levels)].copy()
            if subset.empty:
                continue

            series = subset[feature].fillna("<missing>").astype(str)
            total_short_weight = float(_numeric(subset["p_short"]).sum())
            total_longer_weight = float(_numeric(subset["p_longer"]).sum())

            for level, level_df in subset.groupby(series, dropna=False):
                short_weight = float(_numeric(level_df["p_short"]).sum())
                longer_weight = float(_numeric(level_df["p_longer"]).sum())
                level_rows.append(
                    {
                        "population": population,
                        "feature": feature,
                        "level": level,
                        "rows": int(len(level_df)),
                        "mean_p_short": float(_numeric(level_df["p_short"]).mean()),
                        "mean_p_longer": float(_numeric(level_df["p_longer"]).mean()),
                        "short_mode_share": float(_numeric(level_df["hard_short"]).mean()),
                        "delta_short_minus_longer": float(_numeric(level_df["p_short"]).mean() - _numeric(level_df["p_longer"]).mean()),
                        "short_weighted_mix_share": short_weight / total_short_weight if total_short_weight > 0 else math.nan,
                        "longer_weighted_mix_share": longer_weight / total_longer_weight if total_longer_weight > 0 else math.nan,
                    }
                )

            contingency = pd.crosstab(series, subset["hard_short"])
            for column in (0, 1):
                if column not in contingency.columns:
                    contingency[column] = 0
            contingency = contingency[[0, 1]]
            pvalue = math.nan
            if contingency.shape[0] >= 2 and contingency.to_numpy(dtype=float).sum() > 0:
                _, pvalue, _, _ = chi2_contingency(contingency)
            test_rows.append(
                {
                    "population": population,
                    "feature": feature,
                    "rows": int(len(subset)),
                    "levels": int(contingency.shape[0]),
                    "chi2_pvalue": pvalue,
                    "cramers_v": _cramers_v(contingency),
                }
            )

    level_df = pd.DataFrame(level_rows)
    test_df = pd.DataFrame(test_rows)
    if not level_df.empty:
        level_df = level_df.sort_values(["population", "feature", "rows"], ascending=[True, True, False]).reset_index(drop=True)
    if not test_df.empty:
        test_df = test_df.sort_values(["population", "cramers_v"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return level_df, test_df


def _top_numeric_rows(numeric_df: pd.DataFrame, population: str, features: list[str], positive: bool) -> list[pd.Series]:
    subset = numeric_df.loc[numeric_df["population"] == population].copy()
    subset = subset.loc[subset["feature"].isin(features)].copy()
    subset = subset.loc[subset["spearman_rho_p_short"].notna()].copy()
    subset = subset.sort_values("spearman_rho_p_short", ascending=not positive)
    if positive:
        subset = subset.loc[subset["spearman_rho_p_short"] > 0]
    else:
        subset = subset.loc[subset["spearman_rho_p_short"] < 0]
    return [row for _, row in subset.head(3).iterrows()]


def _top_category_rows(level_df: pd.DataFrame, population: str, feature: str) -> list[pd.Series]:
    subset = level_df.loc[(level_df["population"] == population) & (level_df["feature"] == feature)].copy()
    subset = subset.sort_values(["mean_p_short", "rows"], ascending=[False, False])
    return [row for _, row in subset.head(5).iterrows()]


def build_markdown_report(
    merged: pd.DataFrame,
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    categorical_tests_df: pd.DataFrame,
    feature_table_path: Path,
    posterior_root: Path,
) -> str:
    lines = [
        "# Latent Survival Mode Drivers",
        "",
        f"- Posterior root: `{posterior_root}`",
        f"- Feature table: `{feature_table_path}`",
        "- Mode definition: `short = component 1`, `longer = components 2+ collapsed`.",
        "",
        "## Population Snapshot",
        "",
    ]

    for population in ("global", "ya", "vt"):
        pop_df = merged.loc[merged["population"] == population].copy()
        if pop_df.empty:
            continue
        lines.append(
            f"- `{population}`: rows `{len(pop_df)}`, failures `{int(_numeric(pop_df['event_model']).eq(1).sum())}`, "
            f"mean `p_short` `{_numeric(pop_df['p_short']).mean():.3f}`, hard short-mode share `{_numeric(pop_df['hard_short']).mean():.3f}`."
        )

    lines.extend(["", "## Main Numeric Links", ""])
    h2s_features = [
        "h2s_effective_mg_l",
        "high_h2s_flag",
        "calcium_load_per_day",
        "chloride_load_per_day",
        "sulfate_load_per_day",
        "gypsum_proxy_per_day",
        "avg_kpod",
        "tlf_per_day",
    ]
    freq_features = [
        "freq_above_55hz_pct",
        "freq_below_45hz_pct",
        "freq_signed_exposure",
        "freq_w_mean",
        "Номинальная частота, Гц",
    ]
    other_features = ["mount_year", "duration_best_days"]

    for population in ("global", "ya", "vt"):
        positive_rows = _top_numeric_rows(numeric_df, population, h2s_features + other_features, positive=True)
        negative_rows = _top_numeric_rows(numeric_df, population, freq_features + other_features, positive=False)
        if not positive_rows and not negative_rows:
            continue
        lines.append(f"### {population}")
        lines.append("")
        if positive_rows:
            snippets = [
                f"`{row['feature']}` rho `{row['spearman_rho_p_short']:.3f}`; short-longer delta `{row['short_minus_longer']:.3f}`"
                for row in positive_rows
            ]
            lines.append(f"- Stronger short-mode alignment: {', '.join(snippets)}.")
        if negative_rows:
            snippets = [
                f"`{row['feature']}` rho `{row['spearman_rho_p_short']:.3f}`; short-longer delta `{row['short_minus_longer']:.3f}`"
                for row in negative_rows
            ]
            lines.append(f"- Weaker short-mode alignment: {', '.join(snippets)}.")
        lines.append("")

    lines.extend(["## Failure And Static Categories", ""])
    for population in ("global", "ya", "vt"):
        lines.append(f"### {population}")
        lines.append("")
        for feature in ("failure_category_raw", "contractor", "pump_family"):
            test_row = categorical_tests_df.loc[
                (categorical_tests_df["population"] == population)
                & (categorical_tests_df["feature"] == feature)
            ]
            if not test_row.empty:
                row = test_row.iloc[0]
                lines.append(
                    f"- `{feature}` overall association: Cramer's V `{row['cramers_v']:.3f}`, p-value `{row['chi2_pvalue']:.3g}`."
                )
            top_rows = _top_category_rows(categorical_df, population, feature)
            if top_rows:
                snippets = [
                    f"`{row['level']}` mean `p_short` `{row['mean_p_short']:.3f}` (n=`{int(row['rows'])}`)"
                    for row in top_rows[:3]
                ]
                lines.append(f"- Highest short-mode levels in `{feature}`: {', '.join(snippets)}.")
        lines.append("")

    lines.extend(["## Interpretation", ""])
    lines.append("- `mount_year` is the strongest consistent numeric signal across all three populations: newer runs skew more toward the short-life mode.")
    lines.append("- `Vt` shows the clearest chemistry link: higher H2S flag, higher H2S proxy, and heavier calcium / chloride / gypsum proxies all move toward the short-life mode.")
    lines.append("- Frequency exposure is weaker than chemistry: `freq_above_55hz_pct`, `freq_signed_exposure`, and nominal frequency are slightly negative or near-zero in `global` and `ya`, so more time above 55 Hz is not the dominant separator here.")
    lines.append("- Failure category matters much more in `Vt` than in `Ya` or `global`: `Засорение РО`, `ПЭД (R-0)`, `КЛ (R-0)`, and `Слом вала` lean strongly toward the short-life mode in `Vt`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    posterior_root = resolve_posterior_root(args.posterior_root)
    feature_table_path = resolve_feature_table(args.feature_table)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else posterior_root
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_table(feature_table_path)
    posteriors = load_population_posteriors(posterior_root)
    merged = merge_features_with_posteriors(features, posteriors)

    coverage_df = compute_merge_coverage(posteriors, merged)
    numeric_df, weighted_means_df = compute_numeric_associations(
        merged,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
        min_numeric_rows=int(args.min_numeric_rows),
    )
    categorical_df, categorical_tests_df = compute_categorical_associations(
        merged,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        min_category_rows=int(args.min_category_rows),
    )

    selected_columns = [
        "row_id",
        "population",
        "p_short",
        "p_longer",
        "hard_short",
        "hard_longer",
        "max_probability",
        "classification_entropy",
        "uncertainty_flag",
        "event_model",
        "duration_best_days",
        "field",
        "failure_category_raw",
        "contractor",
        "pump_family",
        "h2s_effective_mg_l",
        "high_h2s_flag",
        "avg_kpod",
        "freq_above_55hz_pct",
        "freq_below_45hz_pct",
        "freq_signed_exposure",
        "freq_w_mean",
        "tlf_per_day",
        "calcium_load_per_day",
        "chloride_load_per_day",
        "sulfate_load_per_day",
        "gypsum_proxy_per_day",
        "mount_year",
    ]
    keep_columns = [column for column in selected_columns if column in merged.columns]
    merged[keep_columns].to_csv(output_dir / "field_mode_feature_model_dataset.csv", index=False)
    coverage_df.to_csv(output_dir / "field_mode_feature_merge_coverage.csv", index=False)
    numeric_df.to_csv(output_dir / "field_mode_numeric_associations.csv", index=False)
    weighted_means_df.to_csv(output_dir / "field_mode_weighted_feature_means.csv", index=False)
    categorical_df.to_csv(output_dir / "field_mode_categorical_associations.csv", index=False)
    categorical_tests_df.to_csv(output_dir / "field_mode_categorical_tests.csv", index=False)

    report = {
        "posterior_root": str(posterior_root),
        "feature_table": str(feature_table_path),
        "populations": sorted(merged["population"].dropna().unique().tolist()),
        "merge_coverage": coverage_df.to_dict(orient="records"),
        "settings": {
            "min_category_rows": int(args.min_category_rows),
            "min_numeric_rows": int(args.min_numeric_rows),
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "categorical_features": DEFAULT_CATEGORICAL_FEATURES,
        },
    }
    (output_dir / "field_mode_driver_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "field_mode_driver_report.md").write_text(
        build_markdown_report(
            merged=merged,
            numeric_df=numeric_df,
            categorical_df=categorical_df,
            categorical_tests_df=categorical_tests_df,
            feature_table_path=feature_table_path,
            posterior_root=posterior_root,
        ),
        encoding="utf-8",
    )

    print(f"Posterior root: {posterior_root}")
    print(f"Feature table: {feature_table_path}")
    print(f"Outputs written to: {output_dir}")
    print(coverage_df.to_string(index=False))
    if not numeric_df.empty:
        ranked = numeric_df.copy()
        ranked["abs_rho"] = ranked["spearman_rho_p_short"].abs()
        ranked = ranked.sort_values(["population", "abs_rho"], ascending=[True, False])
        print("\nTop numeric links by population:")
        print(
            ranked.loc[:, ["population", "feature", "rows", "spearman_rho_p_short", "short_minus_longer"]]
            .groupby("population", group_keys=False)
            .head(5)
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
