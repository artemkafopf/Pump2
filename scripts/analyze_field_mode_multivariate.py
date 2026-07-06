from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from analysis import RegularizedLogisticConfig, fit_regularized_logistic_model
from analysis.paths import RESULTS_ROOT

ANALYSIS_OUTPUTS_DIR = REPO_ROOT / "analysis_outputs"  # legacy fallback
POPULATION_DIRS = {
    "global": "global",
    "ya": "ya",
    "vt": "vt",
}

POSITIVE_LOG_FEATURES = [
    "h2s_effective_mg_l",
    "tlf_per_day",
    "calcium_load_per_day",
    "chloride_load_per_day",
    "sulfate_load_per_day",
    "gypsum_proxy_per_day",
]

INTERACTION_NUMERIC_FEATURES = [
    "low_h2s_flag_numeric",
    "mount_year_x_low_h2s",
    "mount_year_x_log1p_gypsum_proxy_per_day",
]

ALL_RUN_NUMERIC_FEATURES = [
    "mount_year",
    "log1p_h2s_effective_mg_l",
    "avg_kpod",
    "freq_above_55hz_pct",
    "freq_below_45hz_pct",
    "freq_signed_exposure",
    "freq_w_mean",
    "log1p_tlf_per_day",
    "log1p_calcium_load_per_day",
    "log1p_chloride_load_per_day",
    "log1p_sulfate_load_per_day",
    "log1p_gypsum_proxy_per_day",
]

ALL_RUN_CATEGORICAL_FEATURES = [
    "contractor",
    "pump_family",
]

FAILURE_ONLY_NUMERIC_FEATURES = ALL_RUN_NUMERIC_FEATURES
FAILURE_ONLY_CATEGORICAL_FEATURES = [
    "contractor",
    "pump_family",
    "failure_category_raw",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit multivariate ridge-logistic models for latent short-life mode drivers."
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
        help="Directory for outputs. Defaults to posterior-root.",
    )
    parser.add_argument(
        "--populations",
        nargs="*",
        default=["global", "ya", "vt"],
        help="Populations to fit: global ya vt.",
    )
    parser.add_argument(
        "--target-column",
        default="hard_short",
        choices=["hard_short", "p_short"],
        help="Mode target. `hard_short` uses the MAP short-mode label; `p_short` fits a fractional-logit sensitivity model.",
    )
    parser.add_argument(
        "--sample-weight-column",
        default="max_probability",
        help="Optional weight column. Default uses posterior classification confidence.",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=1.0,
        help="Ridge penalty strength.",
    )
    parser.add_argument(
        "--min-category-rows",
        type=int,
        default=10,
        help="Minimum rows per category level before pooling into <other>.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="Maximum Newton iterations.",
    )
    return parser.parse_args()


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _latest_path(*roots_and_patterns: tuple) -> Path:
    all_matches: list[Path] = []
    for root, pattern in roots_and_patterns:
        all_matches.extend(root.glob(pattern))
    if not all_matches:
        raise FileNotFoundError(f"No paths found for patterns: {[p for _, p in roots_and_patterns]}")
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
    frame = pd.read_csv(path, low_memory=False)
    if "row_id" not in frame.columns:
        raise KeyError(f"`row_id` column is required in {path}")
    return frame.drop_duplicates(subset=["row_id"], keep="first").copy()


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
        renamed["population"] = population
        renamed["row_id"] = pd.to_numeric(renamed["row_id"], errors="coerce")
        renamed["most_probable_component"] = pd.to_numeric(renamed["most_probable_component"], errors="coerce")
        renamed["p_short"] = _numeric(renamed[probability_columns[0]])
        if len(probability_columns) == 1:
            renamed["p_longer"] = 1.0 - renamed["p_short"]
        else:
            renamed["p_longer"] = renamed[probability_columns[1:]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        total = renamed["p_short"] + renamed["p_longer"]
        valid_total = total.notna() & total.gt(0.0)
        renamed.loc[valid_total, "p_short"] = renamed.loc[valid_total, "p_short"] / total.loc[valid_total]
        renamed.loc[valid_total, "p_longer"] = renamed.loc[valid_total, "p_longer"] / total.loc[valid_total]
        renamed["hard_short"] = renamed["most_probable_component"].eq(1).astype(int)
        keep = [
            "row_id",
            "duration_model",
            "event_model",
            "population",
            "p_short",
            "p_longer",
            "hard_short",
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
    if "field" not in merged.columns and "Месторождение" in merged.columns:
        merged["field"] = merged["Месторождение"].astype(str)
    return merged


def _safe_log1p(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    values = _numeric(series)
    mask = values.notna() & values.ge(0.0)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[mask] = np.log1p(values.loc[mask])
    return result


def _centered_numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    values = _numeric(series)
    observed = values.dropna()
    if observed.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return values - float(observed.median())


def augment_model_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for feature in POSITIVE_LOG_FEATURES:
        if feature in enriched.columns:
            enriched[f"log1p_{feature}"] = _safe_log1p(enriched[feature])
    enriched["mount_year_centered"] = _centered_numeric(enriched["mount_year"]) if "mount_year" in enriched.columns else np.nan
    if "high_h2s_flag" in enriched.columns:
        h2s_flag = enriched["high_h2s_flag"]
        if not isinstance(h2s_flag, pd.Series):
            h2s_flag = pd.Series(h2s_flag)
        normalized = h2s_flag.astype(str).str.strip().str.lower().map(
            {
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
            }
        )
        enriched["high_h2s_flag_numeric"] = _numeric(normalized)
        enriched["low_h2s_flag_numeric"] = 1.0 - enriched["high_h2s_flag_numeric"]
    else:
        enriched["high_h2s_flag_numeric"] = np.nan
        enriched["low_h2s_flag_numeric"] = np.nan
    if "mount_year_centered" in enriched.columns:
        enriched["mount_year_x_low_h2s"] = enriched["mount_year_centered"] * enriched["low_h2s_flag_numeric"]
        if "log1p_gypsum_proxy_per_day" in enriched.columns:
            gypsum_centered = _centered_numeric(enriched["log1p_gypsum_proxy_per_day"])
            enriched["log1p_gypsum_proxy_per_day_centered"] = gypsum_centered
            enriched["mount_year_x_log1p_gypsum_proxy_per_day"] = enriched["mount_year_centered"] * gypsum_centered
    return enriched


def _top_terms(frame: pd.DataFrame, direction: str, limit: int = 5) -> list[pd.Series]:
    subset = frame.loc[frame["term_type"].isin(["numeric", "categorical"])].copy()
    subset = subset.loc[subset["term"] != "intercept"].copy()
    ascending = direction == "negative"
    subset = subset.sort_values("coefficient", ascending=ascending)
    if direction == "positive":
        subset = subset.loc[subset["coefficient"] > 0.0]
    else:
        subset = subset.loc[subset["coefficient"] < 0.0]
    return [row for _, row in subset.head(limit).iterrows()]


def build_markdown_report(summary_df: pd.DataFrame, coefficient_df: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# Multivariate Latent Short-Mode Drivers",
        "",
        f"- Target: `{args.target_column}`",
        f"- Sample weight: `{args.sample_weight_column or '<none>'}`",
        f"- Ridge penalty: `{args.l2_penalty}`",
        "- Numeric terms are standardized inside the fit, so coefficients are comparable as roughly one-SD shifts.",
        "- Positive coefficients mean higher odds of the short-life mode after controlling for the other terms in the same model.",
        "",
    ]

    for population in summary_df["population"].drop_duplicates().tolist():
        lines.append(f"## {population}")
        lines.append("")
        population_summary = summary_df.loc[summary_df["population"] == population].copy()
        for row in population_summary.to_dict(orient="records"):
            model_name = str(row["model_name"])
            lines.append(
                f"- `{model_name}`: rows `{int(row['rows'])}`, positive rate `{row['positive_rate']:.3f}`, "
                f"pseudo-R2 `{row['mcfadden_pseudo_r2']:.3f}`, weighted AUC `{row['auc_weighted']:.3f}`, converged `{bool(row['converged'])}`."
            )
            coefficient_rows = coefficient_df.loc[
                (coefficient_df["population"] == population)
                & (coefficient_df["model_name"] == model_name)
            ].copy()
            positive_terms = _top_terms(coefficient_rows, "positive", limit=4)
            negative_terms = _top_terms(coefficient_rows, "negative", limit=4)
            if positive_terms:
                snippets = [
                    f"`{term['term']}` coef `{term['coefficient']:.3f}` OR `{term['odds_ratio']:.2f}`"
                    for term in positive_terms
                ]
                lines.append(f"- Higher short-mode odds: {', '.join(snippets)}.")
            if negative_terms:
                snippets = [
                    f"`{term['term']}` coef `{term['coefficient']:.3f}` OR `{term['odds_ratio']:.2f}`"
                    for term in negative_terms
                ]
                lines.append(f"- Lower short-mode odds: {', '.join(snippets)}.")
            interaction_rows = coefficient_rows.loc[
                coefficient_rows["term"].isin(
                    ["mount_year_x_low_h2s", "mount_year_x_log1p_gypsum_proxy_per_day"]
                )
            ].copy()
            if not interaction_rows.empty:
                snippets = [
                    f"`{row['term']}` coef `{row['coefficient']:.3f}` OR `{row['odds_ratio']:.2f}`"
                    for _, row in interaction_rows.sort_values("term").iterrows()
                ]
                lines.append(f"- Interaction check: {', '.join(snippets)}.")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    posterior_root = resolve_posterior_root(args.posterior_root)
    feature_table_path = resolve_feature_table(args.feature_table)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else posterior_root
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_table(feature_table_path)
    posteriors = load_population_posteriors(posterior_root)
    merged = augment_model_features(merge_features_with_posteriors(features, posteriors))

    config = RegularizedLogisticConfig(
        l2_penalty=float(args.l2_penalty),
        max_iter=int(args.max_iter),
        min_category_rows=int(args.min_category_rows),
        standardize_numeric=True,
        add_missing_indicators=True,
        include_intercept=True,
    )

    model_specs = [
        {
            "model_name": "all_runs_multivariate",
            "filter_name": "all_rows",
            "row_filter": lambda frame: frame.copy(),
            "numeric_features": ALL_RUN_NUMERIC_FEATURES,
            "categorical_features": ALL_RUN_CATEGORICAL_FEATURES,
            "description": "All modeled runs; no failure category to avoid censoring leakage.",
        },
        {
            "model_name": "all_runs_with_interactions",
            "filter_name": "all_rows",
            "row_filter": lambda frame: frame.copy(),
            "numeric_features": ALL_RUN_NUMERIC_FEATURES + INTERACTION_NUMERIC_FEATURES,
            "categorical_features": ALL_RUN_CATEGORICAL_FEATURES,
            "description": "All modeled runs with explicit mount_year x low_H2S and mount_year x gypsum interaction terms.",
        },
        {
            "model_name": "failure_only_with_category",
            "filter_name": "event_model_eq_1",
            "row_filter": lambda frame: frame.loc[_numeric(frame["event_model"]).eq(1)].copy(),
            "numeric_features": FAILURE_ONLY_NUMERIC_FEATURES,
            "categorical_features": FAILURE_ONLY_CATEGORICAL_FEATURES,
            "description": "Failure-only subset with failure category added as a co-driver.",
        },
        {
            "model_name": "failure_only_with_category_interactions",
            "filter_name": "event_model_eq_1",
            "row_filter": lambda frame: frame.loc[_numeric(frame["event_model"]).eq(1)].copy(),
            "numeric_features": FAILURE_ONLY_NUMERIC_FEATURES + INTERACTION_NUMERIC_FEATURES,
            "categorical_features": FAILURE_ONLY_CATEGORICAL_FEATURES,
            "description": "Failure-only subset with failure category plus mount_year interaction terms.",
        },
    ]

    summary_rows: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    populations = [str(population).lower() for population in args.populations]
    for population in populations:
        if population not in POPULATION_DIRS:
            raise ValueError(f"Unknown population: {population}")
        population_frame = merged.loc[merged["population"] == population].copy()
        population_dir = output_dir / population / "multivariate_short_mode"
        population_dir.mkdir(parents=True, exist_ok=True)

        for spec in model_specs:
            model_df = spec["row_filter"](population_frame)
            model_dir = population_dir / spec["model_name"]
            model_dir.mkdir(parents=True, exist_ok=True)

            result = fit_regularized_logistic_model(
                model_df,
                target_column=args.target_column,
                numeric_features=list(spec["numeric_features"]),
                categorical_features=list(spec["categorical_features"]),
                sample_weight_column=args.sample_weight_column if args.sample_weight_column else None,
                id_column="row_id",
                config=config,
            )

            coefficient_table = result.coefficient_table.copy()
            coefficient_table.insert(0, "model_name", spec["model_name"])
            coefficient_table.insert(0, "population", population)
            prediction_table = result.prediction_table.copy()
            prediction_table.insert(0, "model_name", spec["model_name"])
            prediction_table.insert(0, "population", population)

            coefficient_table.to_csv(model_dir / "coefficients.csv", index=False)
            prediction_table.to_csv(model_dir / "predictions.csv", index=False)
            (model_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        **result.metrics,
                        "population": population,
                        "model_name": spec["model_name"],
                        "target_column": args.target_column,
                        "sample_weight_column": args.sample_weight_column,
                        "description": spec["description"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary_row = {
                "population": population,
                "model_name": spec["model_name"],
                "target_column": args.target_column,
                "sample_weight_column": args.sample_weight_column,
                "description": spec["description"],
                **result.metrics,
            }
            summary_rows.append(summary_row)
            coefficient_frames.append(coefficient_table)
            prediction_frames.append(prediction_table)

    summary_df = pd.DataFrame(summary_rows)
    coefficient_df = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    summary_df.to_csv(output_dir / "field_mode_multivariate_summary.csv", index=False)
    coefficient_df.to_csv(output_dir / "field_mode_multivariate_coefficients.csv", index=False)
    prediction_df.to_csv(output_dir / "field_mode_multivariate_predictions.csv", index=False)
    (output_dir / "field_mode_multivariate_report.md").write_text(
        build_markdown_report(summary_df, coefficient_df, args),
        encoding="utf-8",
    )
    (output_dir / "field_mode_multivariate_report.json").write_text(
        json.dumps(
            {
                "posterior_root": str(posterior_root),
                "feature_table": str(feature_table_path),
                "target_column": args.target_column,
                "sample_weight_column": args.sample_weight_column,
                "summary": summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Posterior root: {posterior_root}")
    print(f"Feature table: {feature_table_path}")
    print(f"Outputs written to: {output_dir}")
    if not summary_df.empty:
        display_columns = [
            "population",
            "model_name",
            "rows",
            "positive_rate",
            "mcfadden_pseudo_r2",
            "auc_weighted",
            "converged",
            "iterations",
        ]
        print(summary_df[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
