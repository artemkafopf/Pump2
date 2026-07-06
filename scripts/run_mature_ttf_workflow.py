from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None
    Pool = None

from analysis.modeling_config import assert_no_explanatory_features
from analysis.transform_selection import rank_transform_candidates
from analysis.weibull_model import fit_weibull_stress_model
from analysis.paths import results_dir
from scripts.analyze_failure_horizon import ALL_PATH, prepare_feature_matrix
from scripts.build_run_features import build_run_features, load_runs
from scripts.data_utils import _numeric, split_by_run_id


_SLUG = "mature_ttf"

WEIBULL_STRESS_PRIORITY = [
    ("kpod_freq_w_mean", 0.10),
    ("frac_kpod_below_0p7_w", 0.10),
    ("frac_pzab_below_1_w", 0.10),
    ("frac_glf_above_thr_w", 0.10),
    ("h2s_proxy_mg_l", 100.0),
    ("salt_proxy_w_mean", 0.01),
    ("load_w_std", 2.0),
]

STRESS_TRANSFORMS = [
    "negative_excess",
    "positive_excess",
    "relative_abs_deviation",
]

PROVISIONAL_MESSAGES = (
    "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    "TOTAL NO. OF ITERATIONS REACHED LIMIT",
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = truth - pred
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    truth_mean = float(np.mean(truth))
    ss_tot = float(np.sum(np.square(truth - truth_mean)))
    ss_res = float(np.sum(np.square(residual)))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def mature_predictive_features(df: pd.DataFrame) -> list[str]:
    candidates = [
        "field",
        "contractor",
        "pump_family",
        "nominal_qliq",
        "nominal_head_50hz",
        "nominal_freq",
        "motor_power_kw",
        "stages",
        "submergence_depth_m",
        "pbubble",
        "curvature_flag",
        "h2s_proxy_mg_l",
        "h2s_proxy_source",
        "run_number_at_well",
        "season_of_install",
        "kpod_m_mean",
        "kpod_freq_m_mean",
        "frac_kpod_below_0p7_m",
        "pzab_over_pbubble_m_mean",
        "frac_pzab_below_1_m",
        "glf_m_mean",
        "frac_glf_above_thr_m",
        "load_m_mean",
        "load_m_std",
        "freq_m_mean",
        "mean_neg_excess_kpod_m",
        "salt_proxy_m_mean",
        "gypsum_proxy_m_mean",
    ]
    features = [column for column in candidates if column in df.columns]
    assert_no_explanatory_features(features, predictive=True)
    return features


def fit_catboost_ttf(df: pd.DataFrame, feature_columns: list[str]) -> tuple[CatBoostRegressor, dict[str, object], list[str], list[str]]:
    if CatBoostRegressor is None or Pool is None:
        raise RuntimeError("CatBoost is not available.")
    failures = df.loc[df["event"].eq(1)].copy()
    failures["duration_days"] = _numeric(failures["duration_days"])
    failures = failures.loc[failures["duration_days"].notna()].copy()
    prepared, categorical_columns = prepare_feature_matrix(failures, feature_columns)
    aligned = failures.loc[prepared.index].copy()
    train_df, test_df = split_by_run_id(aligned, run_id_column="row_id", test_fraction=0.2)
    train_index = train_df.index
    test_index = test_df.index
    if len(test_index) < max(30, int(len(aligned) * 0.1)) or len(train_index) < max(80, int(len(aligned) * 0.4)):
        fallback_mask = np.zeros(len(aligned), dtype=bool)
        fallback_mask[::5] = True
        train_index = aligned.index[~fallback_mask]
        test_index = aligned.index[fallback_mask]

    train_x = prepared.loc[train_index]
    test_x = prepared.loc[test_index]
    train_y = aligned.loc[train_index, "duration_days"].to_numpy(dtype=float)
    test_y = aligned.loc[test_index, "duration_days"].to_numpy(dtype=float)
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]

    eval_model = CatBoostRegressor(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    eval_model.fit(Pool(train_x, train_y, cat_features=cat_indices))
    metrics: dict[str, object] = {
        "rows": int(len(aligned)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "feature_count": int(prepared.shape[1]),
        "categorical_feature_count": int(len(categorical_columns)),
    }
    if len(test_x) > 0:
        metrics.update(regression_metrics(test_y, eval_model.predict(test_x)))

    final_model = CatBoostRegressor(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    final_model.fit(Pool(prepared, aligned["duration_days"].to_numpy(dtype=float), cat_features=cat_indices))
    return final_model, metrics, prepared.columns.tolist(), categorical_columns


def select_weibull_stress_terms(df: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for column, scale in WEIBULL_STRESS_PRIORITY:
        if column not in df.columns:
            continue
        numeric = _numeric(df[column]).dropna()
        if len(numeric) < 80:
            continue
        ranked = rank_transform_candidates(
            df,
            duration_column="duration_days",
            event_column="event",
            column=column,
            group_columns=["field", "contractor", "pump_family"],
            min_group_size=20,
            candidate_transforms=STRESS_TRANSFORMS,
            coefficient_non_negative=True,
            scale=float(scale),
        )
        if not ranked:
            continue
        best = ranked[0]
        rows.append(
            {
                "column": column,
                "scale": float(scale),
                "transform": best.transform,
                "aic": float(best.aic),
                "delta_aic_vs_baseline": float(best.delta_aic_vs_baseline),
                "success": bool(best.success),
                "message": str(best.message),
                "reference_init": float(numeric.median()),
                "reference_lower": float(numeric.min()),
                "reference_upper": float(numeric.max()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return [], ranking
    ranking = ranking.sort_values(["success", "delta_aic_vs_baseline", "column"], ascending=[False, True, True]).reset_index(drop=True)

    def is_provisional_candidate(row: pd.Series) -> bool:
        if bool(row["success"]):
            return True
        if not np.isfinite(float(row["aic"])):
            return False
        if float(row["delta_aic_vs_baseline"]) > -10.0:
            return False
        message = str(row["message"])
        return any(token in message for token in PROVISIONAL_MESSAGES)

    ranking["selection_status"] = ranking.apply(
        lambda row: (
            "selected"
            if bool(row["success"]) and float(row["delta_aic_vs_baseline"]) <= -2.0
            else (
                "provisional"
                if is_provisional_candidate(row)
                else "rejected"
            )
        ),
        axis=1,
    )
    selected = ranking.loc[ranking["selection_status"].isin(["selected", "provisional"])].copy()
    if not selected.empty:
        priority_map = {"selected": 0, "provisional": 1}
        selected["selection_priority"] = selected["selection_status"].map(priority_map).fillna(9)
        selected = selected.sort_values(
            ["selection_priority", "delta_aic_vs_baseline", "column"],
            ascending=[True, True, True],
        ).head(3)
    stress_terms = [
        {
            "name": f"stress_{row['column']}_{row['transform']}",
            "column": str(row["column"]),
            "transform": str(row["transform"]),
            # Phase 1: use the screened median reference to reduce optimizer complexity.
            "reference_mode": "fixed",
            "reference_value": float(row["reference_init"]),
            "coefficient_mode": "fit",
            "coefficient_value": 0.05,
            "coefficient_non_negative": True,
            "coefficient_bounds": [0.0, None],
            "scale": float(row["scale"]),
            "selection_status": str(row["selection_status"]),
            "screen_delta_aic": float(row["delta_aic_vs_baseline"]),
        }
        for _, row in selected.iterrows()
    ]
    return stress_terms, ranking


def fit_weibull_explanatory(df: pd.DataFrame, stress_terms: list[dict[str, object]]) -> object:
    attempts = [
        (["field", "contractor", "pump_family"], 20),
        (["field", "pump_family"], 20),
        (["contractor", "pump_family"], 20),
        (["pump_family"], 20),
        ([], 1),
    ]
    term_subsets: list[list[dict[str, object]]] = [stress_terms]
    if len(stress_terms) > 2:
        term_subsets.append(stress_terms[:2])
    if len(stress_terms) > 1:
        term_subsets.append(stress_terms[:1])

    best_successful: object | None = None
    best_successful_aic = float("inf")
    best_provisional: object | None = None
    best_provisional_aic = float("inf")
    last_error: Exception | None = None

    for group_columns, min_group_size in attempts:
        for term_subset in term_subsets:
            try:
                result = fit_weibull_stress_model(
                    df,
                    duration_column="duration_days",
                    event_column="event",
                    group_columns=group_columns,
                    stress_terms=term_subset,
                    min_group_size=min_group_size,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue
            if result.success and result.aic < best_successful_aic:
                best_successful = result
                best_successful_aic = float(result.aic)
            elif result.aic < best_provisional_aic:
                best_provisional = result
                best_provisional_aic = float(result.aic)

    if best_successful is not None:
        return best_successful
    if best_provisional is not None:
        return best_provisional
    raise RuntimeError(f"Weibull fit failed for all grouping attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mature-life TTF workflow on the Phase 1 run feature table.")
    parser.add_argument("--output-dir", default=None, help="Directory for output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir(_SLUG)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(ALL_PATH)
    mature_features, infant_runs, gap_runs, summary = build_run_features(runs, include_whole_life=True, predictive=False)
    feature_columns = mature_predictive_features(mature_features)
    ttf_model, ttf_metrics, prepared_columns, categorical_columns = fit_catboost_ttf(mature_features, feature_columns)
    stress_terms, stress_ranking = select_weibull_stress_terms(mature_features)
    weibull_result = fit_weibull_explanatory(mature_features, stress_terms)

    mature_failures = mature_features.loc[mature_features["event"].eq(1)].copy()
    prepared, _ = prepare_feature_matrix(mature_failures, feature_columns)
    aligned = mature_failures.loc[prepared.index].copy()
    predictions = ttf_model.predict(prepared)
    catboost_predictions = aligned[["row_id", "well_id", "duration_days", "field", "contractor", "pump_family"]].copy()
    catboost_predictions["predicted_duration_days"] = predictions
    catboost_predictions["prediction_error_days"] = predictions - _numeric(aligned["duration_days"]).to_numpy(dtype=float)

    stress_ranking.to_csv(output_dir / "weibull_stress_ranking.csv", index=False, encoding="utf-8-sig")
    if stress_terms:
        pd.DataFrame(stress_terms).to_csv(output_dir / "weibull_selected_stress_terms.csv", index=False, encoding="utf-8-sig")
    actual_weibull_terms = [asdict(term) for term in weibull_result.stress_terms]
    if actual_weibull_terms:
        pd.DataFrame(actual_weibull_terms).to_csv(output_dir / "weibull_fitted_stress_terms.csv", index=False, encoding="utf-8-sig")
    catboost_predictions.to_csv(output_dir / "catboost_ttf_predictions.csv", index=False, encoding="utf-8-sig")
    mature_features.to_csv(output_dir / "mature_run_features.csv", index=False, encoding="utf-8-sig")
    metrics_payload = {
        "catboost_ttf": ttf_metrics,
        "weibull_group_columns": weibull_result.group_columns,
        "weibull_stage_summaries": weibull_result.to_dict()["stage_summaries"],
        "weibull_candidate_stress_terms": stress_terms,
        "weibull_fitted_stress_terms": actual_weibull_terms,
        "feature_columns": feature_columns,
        "prepared_columns": prepared_columns,
        "categorical_columns": categorical_columns,
        "population_summary": summary,
    }
    (output_dir / "mature_ttf_metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Mature-Life TTF Workflow",
        "",
        f"- Mature feature rows: `{len(mature_features)}`",
        f"- Infant runs excluded: `{len(infant_runs)}`",
        f"- Gap runs excluded from predictive `_m_` workflow: `{len(gap_runs)}`",
        "",
        "## CatBoost TTF",
        "",
        f"- Metrics: `{json.dumps(ttf_metrics, ensure_ascii=False)}`",
        "",
        "## Weibull",
        "",
        f"- Group columns: `{weibull_result.group_columns}`",
        f"- Stage summaries: `{json.dumps(weibull_result.to_dict()['stage_summaries'], ensure_ascii=False)}`",
        f"- Fitted stress terms: `{json.dumps(actual_weibull_terms, ensure_ascii=False)}`",
        "",
        "## Stress Terms",
        "",
    ]
    if stress_ranking.empty:
        report_lines.append("- No explanatory stress terms were selected.")
    else:
        for row in stress_ranking.itertuples(index=False):
            report_lines.append(
                f"- `{row.column}` -> `{row.transform}` with delta AIC `{row.delta_aic_vs_baseline:.2f}` "
                f"(`{row.selection_status}`, success={row.success})"
            )
    (output_dir / "mature_ttf_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Mature TTF workflow written to: {output_dir}")
    print(f"Mature rows: {len(mature_features)}")
    print(f"Gap runs: {len(gap_runs)}")
    print(f"CatBoost TTF rows: {ttf_metrics['rows']}")


if __name__ == "__main__":
    main()
