from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


@dataclass(slots=True)
class RegularizedLogisticConfig:
    l2_penalty: float = 1.0
    max_iter: int = 200
    tolerance: float = 1e-8
    min_category_rows: int = 10
    standardize_numeric: bool = True
    add_missing_indicators: bool = True
    include_intercept: bool = True
    min_scale: float = 1e-8
    max_backtracking_steps: int = 30


@dataclass(slots=True)
class RegularizedLogisticResult:
    coefficients: np.ndarray
    coefficient_table: pd.DataFrame
    prediction_table: pd.DataFrame
    metrics: dict[str, float | int | bool]
    design_info: pd.DataFrame
    config: RegularizedLogisticConfig


def _coefficient_numeric_value(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default if pd.isna(numeric) else numeric)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    if denominator <= 0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / denominator)


def _safe_logit(probability: float, eps: float = 1e-8) -> float:
    clipped = min(max(probability, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _weighted_auc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    positive = y_true >= 0.5
    negative = ~positive
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positive[order]
    sorted_weights = weights[order]

    negative_weight_cumulative = 0.0
    concordant_weight = 0.0
    tie_positive_weight = 0.0
    tie_negative_weight = 0.0
    current_score = None

    for score, is_positive, weight in zip(sorted_scores, sorted_positive, sorted_weights, strict=False):
        if current_score is None or score != current_score:
            concordant_weight += tie_positive_weight * (negative_weight_cumulative + 0.5 * tie_negative_weight)
            negative_weight_cumulative += tie_negative_weight
            tie_positive_weight = 0.0
            tie_negative_weight = 0.0
            current_score = score
        if is_positive:
            tie_positive_weight += float(weight)
        else:
            tie_negative_weight += float(weight)

    concordant_weight += tie_positive_weight * (negative_weight_cumulative + 0.5 * tie_negative_weight)
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    denominator = positive_weight * negative_weight
    if denominator <= 0:
        return float("nan")
    return float(concordant_weight / denominator)


def _negative_log_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalized_mask: np.ndarray,
    l2_penalty: float,
) -> float:
    linear = X @ beta
    loss = np.logaddexp(0.0, linear) - y * linear
    penalty = 0.5 * l2_penalty * float(np.dot(beta[penalized_mask], beta[penalized_mask]))
    return float(np.dot(weights, loss) + penalty)


def _gradient_and_hessian(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalized_mask: np.ndarray,
    l2_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = X @ beta
    probabilities = expit(linear)
    weighted_residual = weights * (probabilities - y)
    gradient = X.T @ weighted_residual
    if np.any(penalized_mask):
        gradient = gradient + l2_penalty * beta * penalized_mask.astype(float)

    fisher_weights = weights * probabilities * (1.0 - probabilities)
    hessian = X.T @ (X * fisher_weights[:, None])
    if np.any(penalized_mask):
        hessian = hessian + np.diag(penalized_mask.astype(float) * l2_penalty)
    return gradient, hessian, probabilities


def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: RegularizedLogisticConfig,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    n_terms = X.shape[1]
    beta = np.zeros(n_terms, dtype=float)
    penalized_mask = np.ones(n_terms, dtype=bool)
    if config.include_intercept and n_terms > 0:
        penalized_mask[0] = False
        beta[0] = _safe_logit(_weighted_mean(y, weights))

    converged = False
    iterations = 0
    for iterations in range(1, config.max_iter + 1):
        gradient, hessian, probabilities = _gradient_and_hessian(
            beta=beta,
            X=X,
            y=y,
            weights=weights,
            penalized_mask=penalized_mask,
            l2_penalty=config.l2_penalty,
        )
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm < config.tolerance:
            converged = True
            break

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if float(np.max(np.abs(step))) < config.tolerance:
            converged = True
            break

        current_objective = _negative_log_likelihood(
            beta=beta,
            X=X,
            y=y,
            weights=weights,
            penalized_mask=penalized_mask,
            l2_penalty=config.l2_penalty,
        )
        accepted = False
        alpha = 1.0
        for _ in range(config.max_backtracking_steps):
            candidate = beta - alpha * step
            candidate_objective = _negative_log_likelihood(
                beta=candidate,
                X=X,
                y=y,
                weights=weights,
                penalized_mask=penalized_mask,
                l2_penalty=config.l2_penalty,
            )
            if candidate_objective <= current_objective:
                beta = candidate
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            beta = beta - 0.1 * step

    _, final_hessian, final_probabilities = _gradient_and_hessian(
        beta=beta,
        X=X,
        y=y,
        weights=weights,
        penalized_mask=penalized_mask,
        l2_penalty=config.l2_penalty,
    )
    return beta, final_hessian, iterations, converged


def _encode_design_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: RegularizedLogisticConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    if config.include_intercept:
        columns.append(np.ones(len(df), dtype=float))
        rows.append(
            {
                "term": "intercept",
                "term_type": "intercept",
                "source_feature": "intercept",
                "level": None,
                "reference_level": None,
                "impute_value": None,
                "center": None,
                "scale": None,
            }
        )

    for feature in numeric_features:
        if feature not in df.columns:
            continue
        values = _numeric(df[feature])
        missing = values.isna()
        observed = values.loc[~missing]
        if observed.empty:
            continue
        impute_value = float(observed.median())
        filled = values.fillna(impute_value).to_numpy(dtype=float)
        center = float(filled.mean()) if config.standardize_numeric else 0.0
        scale = float(filled.std(ddof=0)) if config.standardize_numeric else 1.0
        if abs(scale) < config.min_scale:
            scale = 1.0
        encoded = (filled - center) / scale if config.standardize_numeric else filled
        columns.append(encoded)
        rows.append(
            {
                "term": feature,
                "term_type": "numeric",
                "source_feature": feature,
                "level": None,
                "reference_level": None,
                "impute_value": impute_value,
                "center": center if config.standardize_numeric else None,
                "scale": scale if config.standardize_numeric else None,
            }
        )
        if config.add_missing_indicators and bool(missing.any()):
            columns.append(missing.astype(float).to_numpy(dtype=float))
            rows.append(
                {
                    "term": f"{feature}__missing",
                    "term_type": "missing_flag",
                    "source_feature": feature,
                    "level": "<missing>",
                    "reference_level": None,
                    "impute_value": None,
                    "center": None,
                    "scale": None,
                }
            )

    for feature in categorical_features:
        if feature not in df.columns:
            continue
        series = df[feature].fillna("<missing>").astype(str)
        counts = series.value_counts()
        if counts.empty:
            continue
        cleaned = series.where(series.isin(counts.loc[counts >= config.min_category_rows].index), "<other>")
        cleaned_counts = cleaned.value_counts()
        if cleaned_counts.shape[0] <= 1:
            continue
        reference_level = str(cleaned_counts.idxmax())
        levels = [level for level in cleaned_counts.index.tolist() if str(level) != reference_level]
        for level in levels:
            encoded = cleaned.eq(level).astype(float).to_numpy(dtype=float)
            columns.append(encoded)
            rows.append(
                {
                    "term": f"{feature}[{level}]",
                    "term_type": "categorical",
                    "source_feature": feature,
                    "level": str(level),
                    "reference_level": reference_level,
                    "impute_value": None,
                    "center": None,
                    "scale": None,
                }
            )

    if not columns:
        raise ValueError("No features available after encoding.")
    matrix = np.column_stack(columns).astype(float)
    design_info = pd.DataFrame(rows)
    return matrix, design_info


def design_matrix_from_coefficient_table(df: pd.DataFrame, coefficient_table: pd.DataFrame) -> np.ndarray:
    if coefficient_table.empty:
        raise ValueError("Coefficient table is empty.")

    working = coefficient_table.copy().reset_index(drop=True)
    columns: list[np.ndarray] = []
    categorical_levels: dict[str, set[str]] = {}
    categorical_reference: dict[str, str] = {}
    categorical_rows = working.loc[working["term_type"] == "categorical"].copy()
    for row in categorical_rows.itertuples(index=False):
        source_feature = str(row.source_feature)
        if source_feature not in categorical_levels:
            categorical_levels[source_feature] = set()
        level = "" if pd.isna(row.level) else str(row.level)
        if level:
            categorical_levels[source_feature].add(level)
        if pd.notna(row.reference_level):
            categorical_reference[source_feature] = str(row.reference_level)

    for row in working.itertuples(index=False):
        term_type = str(row.term_type)
        source_feature = str(row.source_feature)

        if term_type == "intercept":
            columns.append(np.ones(len(df), dtype=float))
            continue

        if term_type == "numeric":
            values = _numeric(df[source_feature]) if source_feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
            impute_value = _coefficient_numeric_value(row.impute_value, default=0.0)
            center = _coefficient_numeric_value(row.center, default=0.0)
            scale = _coefficient_numeric_value(row.scale, default=1.0)
            if abs(scale) < 1e-12:
                scale = 1.0
            encoded = (values.fillna(impute_value).to_numpy(dtype=float) - center) / scale
            columns.append(encoded)
            continue

        if term_type == "missing_flag":
            values = _numeric(df[source_feature]) if source_feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
            columns.append(values.isna().astype(float).to_numpy(dtype=float))
            continue

        if term_type == "categorical":
            if source_feature in df.columns:
                values = df[source_feature].fillna("<missing>").astype(str)
            else:
                values = pd.Series(["<missing>"] * len(df), index=df.index, dtype="string").astype(str)
            level = "" if pd.isna(row.level) else str(row.level)
            if level == "<other>":
                explicit_levels = categorical_levels.get(source_feature, set())
                reference_level = categorical_reference.get(source_feature, "")
                encoded = (~values.eq(reference_level) & ~values.isin(list(explicit_levels))).astype(float).to_numpy(dtype=float)
            else:
                encoded = values.eq(level).astype(float).to_numpy(dtype=float)
            columns.append(encoded)
            continue

        raise ValueError(f"Unsupported term_type: {term_type}")

    return np.column_stack(columns).astype(float)


def predict_regularized_logistic_from_table(df: pd.DataFrame, coefficient_table: pd.DataFrame) -> pd.DataFrame:
    if "coefficient" not in coefficient_table.columns:
        raise KeyError("Coefficient table must contain a `coefficient` column.")
    X = design_matrix_from_coefficient_table(df, coefficient_table)
    beta = pd.to_numeric(coefficient_table["coefficient"], errors="coerce").to_numpy(dtype=float)
    linear_predictor = X @ beta
    predicted_probability = expit(linear_predictor)
    return pd.DataFrame(
        {
            "linear_predictor": linear_predictor,
            "predicted_probability": predicted_probability,
        }
    )


def fit_regularized_logistic_model(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    sample_weight_column: str | None = None,
    id_column: str | None = None,
    config: RegularizedLogisticConfig | None = None,
) -> RegularizedLogisticResult:
    if config is None:
        config = RegularizedLogisticConfig()
    if target_column not in df.columns:
        raise KeyError(f"Target column `{target_column}` not found.")

    target = _numeric(df[target_column])
    valid = target.notna() & target.ge(0.0) & target.le(1.0)
    if sample_weight_column is not None and sample_weight_column in df.columns:
        sample_weights = _numeric(df[sample_weight_column])
        valid = valid & sample_weights.notna() & sample_weights.gt(0.0)
    else:
        sample_weights = pd.Series(np.ones(len(df), dtype=float), index=df.index)
    working = df.loc[valid].copy()
    if working.empty:
        raise ValueError("No valid rows remain after filtering target and weights.")

    y = _numeric(working[target_column]).to_numpy(dtype=float)
    if float(np.nanmax(y) - np.nanmin(y)) < 1e-12:
        raise ValueError(f"Target column `{target_column}` has no variation after filtering.")
    weights = _numeric(working[sample_weight_column]).to_numpy(dtype=float) if sample_weight_column else np.ones(len(working), dtype=float)
    X, design_info = _encode_design_matrix(
        working,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    beta, hessian, iterations, converged = _fit_newton(X=X, y=y, weights=weights, config=config)
    probabilities = expit(X @ beta)

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_values = beta / standard_errors
    p_values = 2.0 * norm.sf(np.abs(z_values))

    coefficient_table = design_info.copy()
    coefficient_table["coefficient"] = beta
    coefficient_table["odds_ratio"] = np.exp(np.clip(beta, -50.0, 50.0))
    coefficient_table["std_error"] = standard_errors
    coefficient_table["z_value"] = z_values
    coefficient_table["p_value"] = p_values

    null_probability = _weighted_mean(y, weights)
    null_linear = np.full(len(y), _safe_logit(null_probability), dtype=float)
    null_loss = np.logaddexp(0.0, null_linear) - y * null_linear
    model_loss = np.logaddexp(0.0, X @ beta) - y * (X @ beta)
    log_likelihood = float(-np.dot(weights, model_loss))
    null_log_likelihood = float(-np.dot(weights, null_loss))
    weighted_count = float(weights.sum())
    brier = float(np.dot(weights, np.square(probabilities - y)) / max(weighted_count, 1.0))
    hard_target = (y >= 0.5).astype(int)
    hard_prediction = (probabilities >= 0.5).astype(int)
    accuracy = float(np.dot(weights, (hard_target == hard_prediction).astype(float)) / max(weighted_count, 1.0))
    auc = _weighted_auc(hard_target.astype(float), probabilities, weights)

    metrics: dict[str, float | int | bool] = {
        "rows": int(len(working)),
        "weighted_rows": weighted_count,
        "positive_rate": float(_weighted_mean(y, weights)),
        "converged": bool(converged),
        "iterations": int(iterations),
        "l2_penalty": float(config.l2_penalty),
        "log_likelihood": log_likelihood,
        "null_log_likelihood": null_log_likelihood,
        "mcfadden_pseudo_r2": 1.0 - (log_likelihood / null_log_likelihood) if null_log_likelihood != 0 else float("nan"),
        "brier_score": brier,
        "accuracy_at_0p5": accuracy,
        "auc_weighted": auc,
    }

    prediction_table = pd.DataFrame(
        {
            "target": y,
            "predicted_probability": probabilities,
            "sample_weight": weights,
        },
        index=working.index,
    )
    if id_column is not None and id_column in working.columns:
        prediction_table.insert(0, id_column, working[id_column].to_numpy())

    return RegularizedLogisticResult(
        coefficients=beta,
        coefficient_table=coefficient_table.reset_index(drop=True),
        prediction_table=prediction_table.reset_index(drop=True),
        metrics=metrics,
        design_info=design_info.reset_index(drop=True),
        config=config,
    )
