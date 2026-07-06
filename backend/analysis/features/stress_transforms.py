from __future__ import annotations

from collections.abc import Callable

import numpy as np


EPSILON = 1e-12


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _safe_scale(scale: float | None) -> float:
    if scale is None or not np.isfinite(scale) or abs(scale) < EPSILON:
        return 1.0
    return float(scale)


def _safe_ratio(x, reference) -> np.ndarray:
    x_arr = _as_float_array(x)
    ref_arr = _as_float_array(reference)
    ratio = np.full(x_arr.shape, np.nan, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(ref_arr) & (np.abs(ref_arr) > EPSILON)
    ratio[valid] = x_arr[valid] / ref_arr[valid]
    return ratio


def relative_abs_deviation(x, reference, scale: float = 1.0) -> np.ndarray:
    ratio = _safe_ratio(x, reference)
    return np.abs(ratio - 1.0)


def relative_squared_deviation(x, reference, scale: float = 1.0) -> np.ndarray:
    ratio = _safe_ratio(x, reference)
    return np.square(ratio - 1.0)


def absolute_deviation(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.abs(_as_float_array(x) - _as_float_array(reference))


def squared_deviation(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.square(_as_float_array(x) - _as_float_array(reference))


def positive_excess(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.maximum(0.0, _as_float_array(x) - _as_float_array(reference))


def negative_excess(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.maximum(0.0, _as_float_array(reference) - _as_float_array(x))


def positive_excess_squared(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.square(positive_excess(x, reference, scale))


def negative_excess_squared(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.square(negative_excess(x, reference, scale))


def relative_positive_excess(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.maximum(0.0, _safe_ratio(x, reference) - 1.0)


def relative_negative_excess(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.maximum(0.0, 1.0 - _safe_ratio(x, reference))


def sigmoid_high(x, reference, scale: float = 1.0) -> np.ndarray:
    x_arr = _as_float_array(x)
    ref_arr = _as_float_array(reference)
    scale_value = _safe_scale(scale)
    return 1.0 / (1.0 + np.exp(-(x_arr - ref_arr) / scale_value))


def sigmoid_low(x, reference, scale: float = 1.0) -> np.ndarray:
    x_arr = _as_float_array(x)
    ref_arr = _as_float_array(reference)
    scale_value = _safe_scale(scale)
    return 1.0 / (1.0 + np.exp(-(ref_arr - x_arr) / scale_value))


def log_ratio(x, reference, scale: float = 1.0) -> np.ndarray:
    ratio = _safe_ratio(x, reference)
    result = np.full(ratio.shape, np.nan, dtype=float)
    valid = np.isfinite(ratio) & (ratio > 0.0)
    result[valid] = np.log(ratio[valid])
    return result


def abs_log_ratio(x, reference, scale: float = 1.0) -> np.ndarray:
    return np.abs(log_ratio(x, reference, scale))


def squared_log_ratio(x, reference, scale: float = 1.0) -> np.ndarray:
    values = log_ratio(x, reference, scale)
    return np.square(values)


TRANSFORM_LIBRARY: dict[str, Callable[..., np.ndarray]] = {
    "relative_abs_deviation": relative_abs_deviation,
    "relative_squared_deviation": relative_squared_deviation,
    "absolute_deviation": absolute_deviation,
    "squared_deviation": squared_deviation,
    "positive_excess": positive_excess,
    "negative_excess": negative_excess,
    "positive_excess_squared": positive_excess_squared,
    "negative_excess_squared": negative_excess_squared,
    "relative_positive_excess": relative_positive_excess,
    "relative_negative_excess": relative_negative_excess,
    "sigmoid_high": sigmoid_high,
    "sigmoid_low": sigmoid_low,
    "log_ratio": log_ratio,
    "abs_log_ratio": abs_log_ratio,
    "squared_log_ratio": squared_log_ratio,
}


def apply_transform(name: str, x, reference, scale: float = 1.0) -> np.ndarray:
    try:
        transform = TRANSFORM_LIBRARY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TRANSFORM_LIBRARY))
        raise ValueError(f"Unsupported transform '{name}'. Supported transforms: {supported}.") from exc
    return transform(x, reference, scale=scale)
