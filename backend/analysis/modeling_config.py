from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INFANT_MORTALITY_DAYS = 90
DEFAULT_MATURE_WINDOW_DAYS = 90
DEFAULT_MIN_MATURE_OBS = 14
DEFAULT_GLF_THRESHOLD = 300.0

FEATURE_SCOPES = {
    "_m_": "mature_fixed_window",
    "_w_": "whole_life_explanatory_only",
    "_30d_": "rolling_window",
    "_rtd": "run_to_date",
}


def assert_no_explanatory_features(columns: Iterable[str], predictive: bool) -> None:
    if not predictive:
        return
    offending = sorted({str(column) for column in columns if "_w_" in str(column)})
    if offending:
        raise ValueError(
            "Explanatory-only whole-life features are not allowed in predictive feature matrices: "
            + ", ".join(offending)
        )


def stable_hash_test_mask(series: pd.Series, test_fraction: float = 0.2) -> np.ndarray:
    modulus = max(2, int(round(1.0 / max(min(test_fraction, 0.5), 0.05))))

    def is_test(value: object) -> bool:
        key = "<missing>" if pd.isna(value) else str(value)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % modulus == 0

    return series.map(is_test).to_numpy(dtype=bool)


__all__ = [
    "DEFAULT_GLF_THRESHOLD",
    "DEFAULT_INFANT_MORTALITY_DAYS",
    "DEFAULT_MATURE_WINDOW_DAYS",
    "DEFAULT_MIN_MATURE_OBS",
    "FEATURE_SCOPES",
    "assert_no_explanatory_features",
    "stable_hash_test_mask",
]
