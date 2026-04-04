from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy.orm import Session
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde, norm

from app.db.models import Dataset, DatasetColumnMatch, RepairForecastCalculation, TrainedModel
from app.schemas.analysis import (
    RepairForecastMonthlySummary,
    RepairForecastResponse,
    RepairForecastRow,
    RepairForecastSamplingLog,
    RepairForecastSourceDataset,
)
from app.services.analysis import (
    build_dataset_detail,
    coerce_numeric_series,
    load_saved_forecast_model,
    predict_forecast_rows,
    resolve_forecast_columns,
    train_forecast_model,
)

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_GROUP_SIZE = 20
DEFAULT_MAX_SAMPLING_ITER = 1000
TAIL_UPPER_QUANTILE = 0.995
TAIL_DISTRIBUTIONS = {"kde", "spline", "normal"}

FIELD_PATTERNS = ("месторожд", "field", "field_name")
CLUSTER_PATTERNS = ("куст", "cluster", "pad", "cluster_name")
WELL_PATTERNS = ("скваж", "well", "well id", "well_id", "id скв", "well_name", "скв.№", "скв №")
IDENTIFIER_PATTERNS = ("ун", "well_id", "well id", "id скв", "id well")
NNO_PATTERNS = ("нно", "mtbf", "наработка на отказ", "mean time")
RUNTIME_PATTERNS = ("наработ", "runtime", "отработ", "worked")
DATE_PATTERNS = ("дата", "date", "day", "period")
LIQUID_RATE_PATTERNS = ("дебж", "дебит жид", "liq", "liquid")


@dataclass
class PreparedDataset:
    dataset: Dataset
    rows: list[dict]
    canonical_map: dict[str, str]


@dataclass
class TailSamplingConfig:
    distribution: str = "kde"
    clip_min: float | None = None
    clip_max: float | None = None
    fit_to_fact: bool = True


def _build_rng(random_state: int | None) -> np.random.Generator:
    return np.random.default_rng(DEFAULT_RANDOM_STATE if random_state is None else random_state)


def _clean_runtime_values(values: Iterable[object]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    cleaned = numeric.astype(float).to_numpy()
    return cleaned[np.isfinite(cleaned) & (cleaned > 0)]


def _apply_tail_config(values: np.ndarray, config: TailSamplingConfig) -> np.ndarray:
    adjusted = np.asarray(values, dtype=float)
    adjusted = adjusted[np.isfinite(adjusted) & (adjusted > 0)]
    if adjusted.size == 0:
        return adjusted
    if config.clip_min is not None:
        adjusted = adjusted[adjusted >= config.clip_min]
    if config.clip_max is not None:
        adjusted = adjusted[adjusted <= config.clip_max]
    return adjusted


def _align_values_to_fact(reference_values: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    if reference_values.size < 2 or source_values.size < 2:
        return source_values
    ref_mean = float(np.mean(reference_values))
    ref_std = float(np.std(reference_values))
    src_mean = float(np.mean(source_values))
    src_std = float(np.std(source_values))
    if ref_std <= 1e-9 or src_std <= 1e-9:
        return source_values
    standardized = (source_values - src_mean) / src_std
    return standardized * ref_std + ref_mean


def _fallback_positive_delta(values: np.ndarray, rng: np.random.Generator) -> float:
    if values.size >= 2:
        sorted_values = np.sort(values)
        diffs = np.diff(sorted_values)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size:
            return float(rng.choice(positive_diffs))
    scale_source = np.median(values) if values.size else 1.0
    return float(max(scale_source * 0.05, 1.0))


def _sample_from_kde(values: np.ndarray, t_fact: float, upper_bound: float, rng: np.random.Generator, max_iter: int) -> float | None:
    if values.size < 2 or np.unique(values).size < 2:
        return None
    kde = gaussian_kde(values)
    for _ in range(max_iter):
        sampled = float(kde.resample(1, seed=rng).reshape(-1)[0])
        if t_fact < sampled <= upper_bound:
            return sampled
    return None


def _sample_from_spline(values: np.ndarray, t_fact: float, upper_bound: float, rng: np.random.Generator, max_iter: int) -> float | None:
    if values.size < 4 or np.unique(values).size < 4:
        return None
    bins = min(24, max(8, values.size // 2))
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.all(hist <= 0):
        return None
    spline = UnivariateSpline(centers, hist, s=max(len(centers) * 0.001, 1e-6))
    spline_min = max(float(np.min(centers)), t_fact)
    spline_max = min(float(np.max(centers)), upper_bound)
    if spline_max <= spline_min:
        return None
    max_density = max(float(np.max(np.clip(spline(centers), a_min=0, a_max=None))), 1e-9)
    for _ in range(max_iter):
        candidate = float(rng.uniform(spline_min, spline_max))
        density = max(float(spline(candidate)), 0.0)
        if density > 0 and rng.uniform(0, max_density) <= density and candidate > t_fact:
            return candidate
    return None


def _sample_from_normal(values: np.ndarray, t_fact: float, upper_bound: float, rng: np.random.Generator, max_iter: int) -> float | None:
    if values.size < 2:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-9:
        return None
    for _ in range(max_iter):
        candidate = float(rng.normal(mean, std))
        if t_fact < candidate <= upper_bound:
            return candidate
    return None


def _build_fitted_density_curve(
    values: np.ndarray,
    config: TailSamplingConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    adjusted = _apply_tail_config(values, config)
    if adjusted.size < 2:
        return None

    lower = float(np.min(adjusted))
    upper = float(np.max(adjusted))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None

    x_values = np.linspace(lower, upper, 240)

    try:
        if config.distribution == "kde":
            if np.unique(adjusted).size < 2:
                return None
            kde = gaussian_kde(adjusted)
            y_values = kde(x_values)
        elif config.distribution == "spline":
            if np.unique(adjusted).size < 4:
                return None
            bins = min(24, max(8, adjusted.size // 2))
            hist, edges = np.histogram(adjusted, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if np.all(hist <= 0):
                return None
            spline = UnivariateSpline(centers, hist, s=max(len(centers) * 0.001, 1e-6))
            y_values = np.clip(spline(x_values), a_min=0.0, a_max=None)
        else:
            mean = float(np.mean(adjusted))
            std = float(np.std(adjusted))
            if std <= 1e-9:
                return None
            y_values = norm.pdf(x_values, loc=mean, scale=std)
    except Exception:
        logger.exception("Tail diagnostic fit failed for distribution=%s", config.distribution)
        return None

    return x_values, np.asarray(y_values, dtype=float)


def sample_tail_runtime(
    group_values: np.ndarray,
    t_fact: float,
    random_state: int | None = None,
    min_group_size: int = 20,
    max_iter: int = 1000,
    distribution: str = "kde",
    clip_min: float | None = None,
    clip_max: float | None = None,
    fit_to_fact: bool = True,
) -> float:
    config = TailSamplingConfig(
        distribution=distribution if distribution in TAIL_DISTRIBUTIONS else "kde",
        clip_min=clip_min,
        clip_max=clip_max,
        fit_to_fact=fit_to_fact,
    )
    values = _apply_tail_config(_clean_runtime_values(group_values), config)
    rng = _build_rng(random_state)
    if values.size == 0:
        return float(t_fact + 1.0)

    upper_bound = float(max(np.quantile(values, TAIL_UPPER_QUANTILE), t_fact + 1.0))
    effective_values = values
    if effective_values.size < max(min_group_size, 2):
        repeats = int(np.ceil(max(min_group_size, 2) / max(effective_values.size, 1)))
        effective_values = np.tile(effective_values, repeats)

    tail_values = effective_values[effective_values > t_fact]
    kde_source = tail_values if tail_values.size >= 2 else effective_values

    sampled = None
    if config.distribution == "kde":
        try:
            sampled = _sample_from_kde(kde_source, t_fact, upper_bound, rng, max_iter)
        except Exception:
            logger.exception("Tail KDE sampling failed, switching to fallback.")
    elif config.distribution == "spline":
        sampled = _sample_from_spline(kde_source, t_fact, upper_bound, rng, max_iter)
    elif config.distribution == "normal":
        sampled = _sample_from_normal(kde_source, t_fact, upper_bound, rng, max_iter)

    if sampled is not None:
        return float(sampled)

    fallback_delta = _fallback_positive_delta(effective_values, rng)
    fallback_value = t_fact + fallback_delta
    if fallback_value <= t_fact:
        fallback_value = t_fact + 1.0
    return float(min(max(fallback_value, t_fact + 1e-6), upper_bound))


def postprocess_runtime_prediction(
    t_pred: float,
    t_fact: float,
    group_values: np.ndarray,
    random_state: int | None = None,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    max_iter: int = DEFAULT_MAX_SAMPLING_ITER,
    distribution: str = "kde",
    clip_min: float | None = None,
    clip_max: float | None = None,
    fit_to_fact: bool = True,
) -> float:
    if t_pred > t_fact:
        return float(t_pred)
    return sample_tail_runtime(
        group_values=np.asarray(group_values, dtype=float),
        t_fact=t_fact,
        random_state=random_state,
        min_group_size=min_group_size,
        max_iter=max_iter,
        distribution=distribution,
        clip_min=clip_min,
        clip_max=clip_max,
        fit_to_fact=fit_to_fact,
    )


def build_group_runtime_distribution(
    rows: list[dict],
    key_columns: dict[str, str | None],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    identifier_column = key_columns.get("identifier")
    runtime_column = key_columns.get("nno") or key_columns.get("runtime")
    if not identifier_column or not runtime_column:
        return {}, np.array([], dtype=float)

    distributions: dict[str, list[float]] = {}
    all_values: list[float] = []
    for row in rows:
        group_key = normalize_text(row.get(identifier_column))
        runtime_value = to_float(row.get(runtime_column))
        if not group_key or runtime_value is None or runtime_value <= 0:
            continue
        distributions.setdefault(group_key, []).append(float(runtime_value))
        all_values.append(float(runtime_value))

    return {key: np.asarray(values, dtype=float) for key, values in distributions.items()}, np.asarray(all_values, dtype=float)


def sample_group_runtime_tail(
    group_key: str,
    t_fact: float,
    group_distributions: dict[str, np.ndarray],
    global_values: np.ndarray,
    random_state: int | None,
    min_group_size: int,
    max_iter: int,
    config: TailSamplingConfig,
) -> tuple[float, str, bool, int]:
    group_values = _apply_tail_config(_clean_runtime_values(group_distributions.get(group_key, np.array([], dtype=float))), config)
    global_clean = _apply_tail_config(_clean_runtime_values(global_values), config)
    values_for_sampling = group_values
    method = f"{config.distribution}_group_tail"
    fallback_used = False

    if values_for_sampling.size < min_group_size and global_clean.size:
        values_for_sampling = global_clean
        if config.fit_to_fact and group_values.size >= 2:
            values_for_sampling = _align_values_to_fact(group_values, global_clean)
            values_for_sampling = _apply_tail_config(values_for_sampling, config)
            method = f"{config.distribution}_global_aligned_tail"
        else:
            method = f"{config.distribution}_global_tail"
        fallback_used = True

    sampled = sample_tail_runtime(
        group_values=values_for_sampling,
        t_fact=t_fact,
        random_state=random_state,
        min_group_size=min_group_size,
        max_iter=max_iter,
        distribution=config.distribution,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        fit_to_fact=config.fit_to_fact,
    )
    return sampled, method, fallback_used, int(values_for_sampling.size)


def build_tail_diagnostic_image(
    group_distributions: dict[str, np.ndarray],
    sampled_tail_values: dict[str, list[float]],
    config: TailSamplingConfig,
) -> str | None:
    groups = [group for group, values in group_distributions.items() if values.size]
    if not groups:
        return None

    groups = sorted(groups)[:9]
    figure, axes = plt.subplots(len(groups), 1, figsize=(10, max(3.2 * len(groups), 4.0)), squeeze=False)
    axes_flat = axes.flatten()

    for axis, group in zip(axes_flat, groups):
        actual = group_distributions[group]
        sampled = np.asarray(sampled_tail_values.get(group, []), dtype=float)
        axis.hist(
            actual,
            bins=min(16, max(actual.size // 2, 6)),
            density=True,
            alpha=0.32,
            color="#ef4444",
            edgecolor="#fca5a5",
            linewidth=1,
            label="Actual histogram (Fact EPU)",
        )

        fitted_curve = _build_fitted_density_curve(actual, config)
        if fitted_curve is not None:
            x_values, y_values = fitted_curve
            axis.plot(
                x_values,
                y_values,
                color="#60a5fa",
                linewidth=2.4,
                label=f"Fitted {config.distribution} curve",
            )

        if sampled.size:
            axis.hist(
                sampled,
                bins=min(12, max(sampled.size, 3)),
                density=True,
                alpha=0.3,
                color="#60a5fa",
                edgecolor="#93c5fd",
                linewidth=1,
                label="Sampled tail histogram",
            )
            axis.scatter(
                sampled,
                np.full(sampled.shape, 0.0),
                color="#22c55e",
                s=24,
                alpha=0.85,
                marker="x",
                label="Generated tail points",
                zorder=4,
            )

        axis.set_title(group)
        axis.set_ylabel("Density")
        axis.legend(loc="upper right")

    axes_flat[-1].set_xlabel("Runtime")
    figure.suptitle(f"Fact EPU runtime histogram vs fitted {config.distribution} distribution by group", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def build_preview_sampled_tail_values(
    group_distributions: dict[str, np.ndarray],
    random_state: int | None,
    min_group_size: int,
    max_sampling_iter: int,
    config: TailSamplingConfig,
) -> tuple[dict[str, list[float]], list[str]]:
    sampled_tail_values: dict[str, list[float]] = {}
    notes: list[str] = []

    for group in sorted(group_distributions):
        values = _apply_tail_config(_clean_runtime_values(group_distributions[group]), config)
        if values.size == 0:
            continue

        anchor_upper = float(np.quantile(values, 0.8)) if values.size > 2 else float(np.max(values))
        anchor_candidates = values[values < anchor_upper]
        if anchor_candidates.size == 0:
            anchor_candidates = values[values < float(np.max(values))]
        if anchor_candidates.size == 0:
            anchor_candidates = values

        sampled_group: list[float] = []
        for index, t_fact in enumerate(anchor_candidates[: min(24, anchor_candidates.size)]):
            sampled = sample_tail_runtime(
                group_values=values,
                t_fact=float(t_fact),
                random_state=(DEFAULT_RANDOM_STATE if random_state is None else random_state) + index,
                min_group_size=min_group_size,
                max_iter=max_sampling_iter,
                distribution=config.distribution,
                clip_min=config.clip_min,
                clip_max=config.clip_max,
                fit_to_fact=config.fit_to_fact,
            )
            if sampled > float(t_fact):
                sampled_group.append(float(sampled))

        if sampled_group:
            sampled_tail_values[group] = sampled_group
            notes.append(f"{group}: generated {len(sampled_group)} tail values")

    return sampled_tail_values, notes


def build_tail_preview(
    fact_dataset: Dataset,
    tail_fact_dataset: Dataset | None,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    max_sampling_iter: int = DEFAULT_MAX_SAMPLING_ITER,
    tail_distribution: str = "kde",
    tail_clip_min: float | None = None,
    tail_clip_max: float | None = None,
    tail_fit_to_fact: bool = True,
) -> tuple[str | None, list[str]]:
    reference_dataset = tail_fact_dataset or fact_dataset
    prepared = prepare_dataset(reference_dataset)
    if not prepared.rows:
        return None, ["В выбранном наборе Факт ЭПУ нет строк для построения диагностики."]

    key_columns = identify_key_columns(prepared.rows)
    group_distributions, _ = build_group_runtime_distribution(prepared.rows, key_columns)
    if not group_distributions:
        return None, ["В выбранном наборе Факт ЭПУ не найдены фактические значения ННО/наработки."]

    config = TailSamplingConfig(
        distribution=tail_distribution if tail_distribution in TAIL_DISTRIBUTIONS else "kde",
        clip_min=tail_clip_min,
        clip_max=tail_clip_max,
        fit_to_fact=tail_fit_to_fact,
    )
    sampled_tail_values, sampled_notes = build_preview_sampled_tail_values(
        group_distributions=group_distributions,
        random_state=random_state,
        min_group_size=min_group_size,
        max_sampling_iter=max_sampling_iter,
        config=config,
    )
    image = build_tail_diagnostic_image(group_distributions, sampled_tail_values, config)
    notes = [
        f"Preview dataset: {reference_dataset.name} (v{reference_dataset.storage_version})",
        f"Distribution: {config.distribution}",
    ]
    notes.extend(sampled_notes[:6])
    return image, notes


def normalize_text(value: object) -> str:
    return str(value or "").strip().casefold().replace("ё", "е")


def find_best_column(columns: Iterable[str], patterns: tuple[str, ...]) -> str | None:
    scored: list[tuple[int, str]] = []
    for column in columns:
        normalized = normalize_text(column)
        score = sum(len(pattern) for pattern in patterns if pattern in normalized)
        if score:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def parse_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        numeric_value = float(numeric)
        if 20000 <= numeric_value <= 80000:
            parsed = pd.to_datetime(numeric_value, errors="coerce", unit="D", origin="1899-12-30")
            if pd.notna(parsed):
                return parsed.date()
        if 1900 <= numeric_value <= 2500 and numeric_value.is_integer():
            return date(int(numeric_value), 1, 1)

    parsed = pd.to_datetime(pd.Series([value], dtype="object"), errors="coerce", dayfirst=True).iloc[0]
    if pd.notna(parsed):
        return parsed.date()
    return None


def to_float(value: object) -> float | None:
    numeric = coerce_numeric_series(pd.Series([value], dtype="object")).iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def stringify(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_canonical_map(matches: list[DatasetColumnMatch]) -> dict[str, str]:
    return {item.source_column: item.canonical_name for item in matches if item.canonical_name}


def canonicalize_row(row: dict, canonical_map: dict[str, str]) -> dict:
    prepared = dict(row)
    for source, canonical in canonical_map.items():
        if source in row and canonical:
            prepared.setdefault(canonical, row[source])
    return prepared


def prepare_dataset(dataset: Dataset) -> PreparedDataset:
    rows = [dict(record.payload) for record in dataset.records]
    canonical_map = build_canonical_map(list(dataset.column_matches))
    prepared_rows = [canonicalize_row(row, canonical_map) for row in rows]
    return PreparedDataset(dataset=dataset, rows=prepared_rows, canonical_map=canonical_map)


def resolve_model_bundle(
    dataset: Dataset,
    model_id: int | None,
    base_feature_columns: list[str],
) -> tuple[dict | None, TrainedModel | None]:
    selected_model = None
    if model_id is not None:
        selected_model = next((model for model in dataset.trained_models if model.id == model_id), None)
    else:
        selected_model = next((model for model in dataset.trained_models if model.is_active), None)

    if selected_model is not None:
        loaded = load_saved_forecast_model(selected_model, dataset, dataset.records)
        if loaded is not None:
            return loaded, selected_model

    trained = train_forecast_model(dataset, dataset.records, base_feature_columns)
    return trained, selected_model


def identify_key_columns(rows: list[dict]) -> dict[str, str | None]:
    columns = list({column for row in rows for column in row.keys()})
    return {
        "field": find_best_column(columns, FIELD_PATTERNS),
        "cluster": find_best_column(columns, CLUSTER_PATTERNS),
        "well": find_best_column(columns, WELL_PATTERNS),
        "identifier": find_best_column(columns, IDENTIFIER_PATTERNS),
        "nno": find_best_column(columns, NNO_PATTERNS),
        "runtime": find_best_column(columns, RUNTIME_PATTERNS),
        "date": find_best_column(columns, DATE_PATTERNS),
        "liquid_rate": find_best_column(columns, LIQUID_RATE_PATTERNS),
    }


def build_lookup_keys(row: dict, key_columns: dict[str, str | None]) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []

    identifier_column = key_columns.get("identifier")
    if identifier_column:
        identifier = normalize_text(row.get(identifier_column))
        if identifier:
            keys.append(("identifier", identifier))

    well = normalize_text(row.get(key_columns["well"])) if key_columns.get("well") else ""
    cluster = normalize_text(row.get(key_columns["cluster"])) if key_columns.get("cluster") else ""

    if well and cluster:
        keys.append(("well_cluster", well, cluster))
    if well:
        keys.append(("well", well))

    return keys


def primary_group_key(row: dict, key_columns: dict[str, str | None]) -> tuple[str, ...]:
    keys = build_lookup_keys(row, key_columns)
    for prefix in ("well_cluster", "well", "identifier"):
        for key in keys:
            if key and key[0] == prefix:
                return key
    if keys:
        return keys[0]
    return ("row", normalize_text(row))


def best_latest_rows(rows: list[dict], key_columns: dict[str, str | None]) -> dict[tuple[str, ...], dict]:
    dated_rows: dict[tuple[str, ...], tuple[date | None, int, dict]] = {}
    for index, row in enumerate(rows):
        row_date = parse_date(row.get(key_columns["date"])) if key_columns.get("date") else None
        for key in build_lookup_keys(row, key_columns):
            previous = dated_rows.get(key)
            if previous is None or ((row_date or date.min), index) >= ((previous[0] or date.min), previous[1]):
                dated_rows[key] = (row_date, index, row)
    return {key: item[2] for key, item in dated_rows.items()}


def find_matching_fact_row(
    source_row: dict,
    source_keys: dict[str, str | None],
    fact_latest_by_key: dict[tuple[str, ...], dict],
) -> dict | None:
    for key in build_lookup_keys(source_row, source_keys):
        if key in fact_latest_by_key:
            return fact_latest_by_key[key]
    return None


def build_identifier_stats(rows: list[dict], key_columns: dict[str, str | None], feature_columns: list[str]) -> dict[str, dict[str, float]]:
    identifier_column = key_columns.get("identifier")
    if not identifier_column:
        return {}

    frame = pd.DataFrame(rows)
    if frame.empty or identifier_column not in frame.columns:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for identifier_value, group in frame.groupby(identifier_column, dropna=True):
        metrics: dict[str, float] = {}
        for column in feature_columns:
            if column not in group.columns:
                continue
            numeric = coerce_numeric_series(group[column])
            if numeric.notna().any():
                metrics[column] = float(numeric.mean())
        stats[normalize_text(identifier_value)] = metrics
    return stats


def apply_feature_defaults(
    row: dict,
    feature_columns: list[str],
    fact_row: dict | None,
    identifier_stats: dict[str, dict[str, float]],
    key_columns: dict[str, str | None],
    manual_values: dict[str, object],
    baseline_values: dict[str, object],
) -> dict:
    prepared = dict(row)
    identifier_key = normalize_text(prepared.get(key_columns["identifier"])) if key_columns.get("identifier") else ""
    identifier_defaults = identifier_stats.get(identifier_key, {})

    for column in feature_columns:
        value = prepared.get(column)
        if value not in (None, ""):
            continue
        if column in manual_values and manual_values[column] not in (None, ""):
            prepared[column] = manual_values[column]
        elif fact_row and fact_row.get(column) not in (None, ""):
            prepared[column] = fact_row.get(column)
        elif column in identifier_defaults:
            prepared[column] = identifier_defaults[column]
        else:
            prepared[column] = baseline_values.get(column)
    return prepared


def predict_single_row(trained: dict, row: dict) -> float | None:
    prediction = predict_forecast_rows(trained, [row])
    if not prediction or prediction[0] is None:
        return None
    return max(float(prediction[0]), 5.0)


def calculate_first_failure_offset(
    predicted_nno: float | None,
    actual_nno: float | None,
    runtime_days: float | None,
    base_failure_coefficient: float,
) -> int:
    del runtime_days
    if predicted_nno is None:
        return 1
    if actual_nno is not None and predicted_nno > actual_nno:
        return max(int(round(predicted_nno * base_failure_coefficient)), 1)
    if actual_nno is not None and actual_nno >= predicted_nno:
        return max(int(round(predicted_nno * max(base_failure_coefficient - 1, 0))), 1)
    return max(int(round(predicted_nno * base_failure_coefficient)), 1)


def align_to_forecast_date(target: date, forecast_dates: list[date]) -> date | None:
    for item in forecast_dates:
        if item >= target:
            return item
    return None


def prediction_for_date(predictions: list[tuple[date, float | None]], target_date: date) -> float | None:
    current = None
    for row_date, prediction in predictions:
        if row_date <= target_date:
            current = prediction
        else:
            break
    if current is not None:
        return current
    return predictions[0][1] if predictions else None


def build_statuses_for_well(
    forecast_dates: list[date],
    prediction_series: list[tuple[date, float | None]],
    actual_nno: float | None,
    runtime_days: float | None,
    base_failure_coefficient: float,
) -> tuple[list[int], list[str], list[tuple[date, float]]]:
    if not forecast_dates:
        return [], [], []

    statuses = [1] * len(forecast_dates)
    if not prediction_series:
        return statuses, [], []

    first_prediction = prediction_series[0][1]
    if first_prediction is None:
        return statuses, [], []

    event_dates: list[str] = []
    event_nnos: list[tuple[date, float]] = []
    date_index = {item: index for index, item in enumerate(forecast_dates)}
    current_anchor = forecast_dates[0]
    next_failure = current_anchor + timedelta(
        days=calculate_first_failure_offset(
            first_prediction,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        )
    )

    while next_failure <= forecast_dates[-1]:
        aligned = align_to_forecast_date(next_failure, forecast_dates)
        if aligned is None:
            break

        index = date_index[aligned]
        statuses[index] = 0
        event_dates.append(aligned.isoformat())

        next_prediction = prediction_for_date(prediction_series, aligned)
        if next_prediction is None:
            break
        event_nnos.append((aligned, next_prediction))

        current_anchor = aligned
        next_failure = current_anchor + timedelta(days=max(int(round(next_prediction)), 1))
        actual_nno = None
        runtime_days = None

    return statuses, event_dates, event_nnos


def build_repair_forecast(
    db: Session,
    fact_dataset: Dataset,
    source_dataset: Dataset,
    tail_fact_dataset: Dataset | None,
    model_id: int | None,
    base_failure_coefficient: float,
    nominal_gap_coefficient: float,
    manual_feature_values: dict[str, object],
    random_state: int | None = DEFAULT_RANDOM_STATE,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    max_sampling_iter: int = DEFAULT_MAX_SAMPLING_ITER,
    tail_distribution: str = "kde",
    tail_clip_min: float | None = None,
    tail_clip_max: float | None = None,
    tail_fit_to_fact: bool = True,
) -> RepairForecastResponse:
    del db, nominal_gap_coefficient

    fact_prepared = prepare_dataset(fact_dataset)
    tail_fact_prepared = prepare_dataset(tail_fact_dataset or fact_dataset)
    source_prepared = prepare_dataset(source_dataset)
    if not source_prepared.rows:
        raise ValueError("В сводпрогнозе нет строк для расчета.")

    feature_columns = resolve_forecast_columns(
        fact_dataset,
        list(fact_dataset.selected_features_json or []),
        pd.DataFrame(fact_prepared.rows, columns=list(fact_dataset.columns_json)),
    )
    trained, selected_model = resolve_model_bundle(fact_dataset, model_id, feature_columns)
    if trained is None:
        raise ValueError("Не удалось подготовить модель CatBoost для прогноза ремонтов.")

    fact_keys = identify_key_columns(fact_prepared.rows)
    tail_fact_keys = identify_key_columns(tail_fact_prepared.rows)
    source_keys = identify_key_columns(source_prepared.rows)
    date_column = source_keys.get("date")
    liquid_rate_column = source_keys.get("liquid_rate")
    identifier_column = source_keys.get("identifier")
    well_column = source_keys.get("well")

    if not date_column:
        raise ValueError("В сводпрогнозе не найдена колонка с датой.")
    if not identifier_column:
        raise ValueError("В сводпрогнозе не найдена колонка УН.")
    if not well_column:
        raise ValueError("В сводпрогнозе не найдена колонка со скважиной.")

    source_rows_with_dates: list[tuple[date, dict]] = []
    for row in source_prepared.rows:
        row_date = parse_date(row.get(date_column))
        if row_date is not None:
            source_rows_with_dates.append((row_date, row))
    if not source_rows_with_dates:
        raise ValueError("В сводпрогнозе не удалось распознать даты для расчета.")

    source_rows_with_dates.sort(key=lambda item: item[0])
    today = date.today()
    end_date = date(today.year + 1, 12, 31)
    forecast_dates = [today + timedelta(days=offset) for offset in range((end_date - today).days + 1)]

    fact_latest_by_key = best_latest_rows(fact_prepared.rows, fact_keys)
    identifier_stats = build_identifier_stats(fact_prepared.rows, fact_keys, trained["feature_columns"])
    group_distributions, global_runtime_values = build_group_runtime_distribution(tail_fact_prepared.rows, tail_fact_keys)
    tail_config = TailSamplingConfig(
        distribution=tail_distribution if tail_distribution in TAIL_DISTRIBUTIONS else "kde",
        clip_min=tail_clip_min,
        clip_max=tail_clip_max,
        fit_to_fact=tail_fit_to_fact,
    )

    available_columns = {column for row in source_prepared.rows for column in row.keys()}
    missing_feature_columns = [column for column in trained["feature_columns"] if column not in available_columns]

    wells: dict[tuple[str, ...], list[tuple[date, dict]]] = {}
    for row_date, row in source_rows_with_dates:
        wells.setdefault(primary_group_key(row, source_keys), []).append((row_date, row))

    result_rows: list[RepairForecastRow] = []
    notes: list[str] = []
    monthly_summary_map: dict[str, dict[str, float | int]] = {}
    sampling_logs: list[RepairForecastSamplingLog] = []
    sampled_tail_values: dict[str, list[float]] = {}
    sample_counter = 0

    for _, items in sorted(wells.items(), key=lambda item: item[0]):
        first_row = items[0][1]
        fact_row = find_matching_fact_row(first_row, source_keys, fact_latest_by_key)
        prediction_series: list[tuple[date, float | None]] = []
        actual_nno = None
        runtime_days = None
        group_key = normalize_text(first_row.get(identifier_column))

        for row_date, source_row in items:
            prepared_row = apply_feature_defaults(
                source_row,
                trained["feature_columns"],
                fact_row,
                identifier_stats,
                source_keys,
                manual_feature_values,
                trained["baseline_values"],
            )
            if actual_nno is None and source_keys.get("nno"):
                actual_nno = to_float(prepared_row.get(source_keys["nno"]))
            if runtime_days is None and source_keys.get("runtime"):
                runtime_days = to_float(prepared_row.get(source_keys["runtime"]))

            raw_prediction = predict_single_row(trained, prepared_row)
            final_prediction = raw_prediction
            sampling_method = "catboost_direct"
            fallback_used = False
            sample_size = int(_clean_runtime_values(group_distributions.get(group_key, np.array([], dtype=float))).size)

            if raw_prediction is not None and actual_nno is not None:
                if raw_prediction <= actual_nno:
                    final_prediction, sampling_method, fallback_used, sample_size = sample_group_runtime_tail(
                        group_key=group_key,
                        t_fact=actual_nno,
                        group_distributions=group_distributions,
                        global_values=global_runtime_values,
                        random_state=(DEFAULT_RANDOM_STATE if random_state is None else random_state) + sample_counter,
                        min_group_size=min_group_size,
                        max_iter=max_sampling_iter,
                        config=tail_config,
                    )
                    sampled_tail_values.setdefault(group_key or "global", []).append(final_prediction)
                    sample_counter += 1
                else:
                    final_prediction = float(raw_prediction)
                sampling_logs.append(
                    RepairForecastSamplingLog(
                        group=stringify(first_row.get(identifier_column)) or "global",
                        sample_size=sample_size,
                        method=sampling_method,
                        fallback_used=fallback_used,
                        t_pred=float(raw_prediction),
                        t_fact=float(actual_nno),
                        t_final=float(final_prediction),
                    )
                )
                logger.info(
                    "repair_forecast_tail group=%s sample_size=%s method=%s fallback=%s t_pred=%.3f t_fact=%.3f t_final=%.3f",
                    stringify(first_row.get(identifier_column)) or "global",
                    sample_size,
                    sampling_method,
                    fallback_used,
                    float(raw_prediction),
                    float(actual_nno),
                    float(final_prediction),
                )

            prediction_series.append((row_date, final_prediction))

        if actual_nno is None and fact_row and fact_keys.get("nno"):
            actual_nno = to_float(fact_row.get(fact_keys["nno"]))
        if runtime_days is None and fact_row and fact_keys.get("runtime"):
            runtime_days = to_float(fact_row.get(fact_keys["runtime"]))

        statuses, event_dates, event_nnos = build_statuses_for_well(
            forecast_dates,
            prediction_series,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        )

        for event_date, event_nno in event_nnos:
            month_key = f"{event_date.year}-{event_date.month:02d}"
            bucket = monthly_summary_map.setdefault(month_key, {"total_nno": 0.0, "failure_count": 0})
            bucket["total_nno"] = float(bucket["total_nno"]) + float(event_nno)
            bucket["failure_count"] = int(bucket["failure_count"]) + 1

        predicted_values = [value for _, value in prediction_series if value is not None]
        average_prediction = float(sum(predicted_values) / len(predicted_values)) if predicted_values else None

        first_liquid_rate = to_float(first_row.get(liquid_rate_column)) if liquid_rate_column else None
        category = "База" if first_liquid_rate is not None and first_liquid_rate > 0 else "ВНС"

        result_rows.append(
            RepairForecastRow(
                category=category,
                field_name=stringify(first_row.get(source_keys["field"])) if source_keys.get("field") else None,
                license_area=stringify(first_row.get(identifier_column)),
                cluster_name=stringify(first_row.get(source_keys["cluster"])) if source_keys.get("cluster") else None,
                well_name=stringify(first_row.get(well_column)) or "Неизвестная скважина",
                predicted_nno=average_prediction,
                actual_nno=actual_nno,
                runtime_days=runtime_days,
                event_dates=event_dates,
                statuses=statuses,
            )
        )

    if selected_model is None:
        notes.append("Сохраненная модель не выбрана, используется текущая конфигурация CatBoost.")
    if missing_feature_columns:
        notes.append(
            "Часть признаков модели отсутствовала в сводпрогнозе и была заполнена из Факт ЭПУ, средними значениями по УН, вручную или базовыми значениями модели."
        )
    if tail_fact_dataset is not None and tail_fact_dataset.id != fact_dataset.id:
        notes.append(f"Tail distributions are compared against Fact EPU dataset '{tail_fact_dataset.name}' (v{tail_fact_dataset.storage_version}).")
    notes.append(
        "Tail distribution config: "
        f"distribution={tail_config.distribution}, "
        f"clip_min={tail_config.clip_min if tail_config.clip_min is not None else 'auto'}, "
        f"clip_max={tail_config.clip_max if tail_config.clip_max is not None else 'auto'}, "
        f"fit_to_fact={'yes' if tail_config.fit_to_fact else 'no'}."
    )
    if not monthly_summary_map:
        notes.append("В расчетном горизонте не возникло прогнозных отказов, поэтому месячная диаграмма показывает ноль.")

    if sampling_logs:
        notes.append(f"Tail postprocessing was applied to {len(sampling_logs)} forecast points.")

    result_rows.sort(
        key=lambda item: (
            item.category,
            item.license_area or "",
            item.cluster_name or "",
            item.well_name,
        )
    )

    monthly_summary = [
        RepairForecastMonthlySummary(
            month=month,
            total_nno=round(float(values["total_nno"]), 1),
            failure_count=int(values["failure_count"]),
        )
        for month, values in sorted(monthly_summary_map.items())
    ]
    tail_diagnostic_image = build_tail_diagnostic_image(group_distributions, sampled_tail_values, tail_config)

    return RepairForecastResponse(
        start_date=today.isoformat(),
        end_date=end_date.isoformat(),
        dates=[item.isoformat() for item in forecast_dates],
        used_feature_columns=list(trained["feature_columns"]),
        missing_feature_columns=missing_feature_columns,
        source_datasets=[
            RepairForecastSourceDataset(
                label="Сводпрогноз",
                dataset=build_dataset_detail(source_dataset, source_dataset.records).dataset,
            )
            ,
            RepairForecastSourceDataset(
                label="Факт ЭПУ для хвостов",
                dataset=build_dataset_detail(tail_fact_dataset or fact_dataset, (tail_fact_dataset or fact_dataset).records).dataset,
            )
        ],
        monthly_summary=monthly_summary,
        rows=result_rows,
        sampling_logs=sampling_logs,
        tail_diagnostic_image=tail_diagnostic_image,
        notes=notes,
    )


def repair_forecast_calculation_to_summary(item: RepairForecastCalculation) -> dict:
    return {
        "id": item.id,
        "dataset_id": item.dataset_id,
        "source_dataset_id": item.source_dataset_id,
        "trained_model_id": item.trained_model_id,
        "name": item.name,
        "created_at": item.created_at,
    }


def save_repair_forecast_calculation(
    db: Session,
    dataset: Dataset,
    source_dataset: Dataset | None,
    trained_model: TrainedModel | None,
    name: str | None,
    settings: dict,
    result: RepairForecastResponse,
) -> RepairForecastCalculation:
    item = RepairForecastCalculation(
        dataset_id=dataset.id,
        source_dataset_id=source_dataset.id if source_dataset else None,
        trained_model_id=trained_model.id if trained_model else None,
        name=(name or f"Repair forecast v{dataset.storage_version}").strip() or "Repair forecast",
        payload_json=result.model_dump(),
        settings_json=settings,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item
