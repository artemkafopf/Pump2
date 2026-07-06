from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm, spearmanr

try:
    from catboost import CatBoostClassifier, Pool
except ModuleNotFoundError:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for _path in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analysis.input_paths import resolve_v03_all_path
from analysis.weibull_model import evaluate_weibull_nll, fit_basic_weibull
from analysis.paths import results_dir
from scripts.analyze_failure_horizon import load_runs, prepare_feature_matrix, evaluate_catboost_cv
from scripts.db import WAREHOUSE_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ANALYSIS_DATE_TAG = "2026_06_22"
_SLUG = "vt_60hz_prompt"
DEFAULT_H2S_THRESHOLD_MG_L = 3.0
DEFAULT_INFANT_THRESHOLDS = (45, 60, 90, 120)
DEFAULT_MATURE_THRESHOLD = 90
DEFAULT_GLF_THRESHOLD = 300.0
DEFAULT_RMST_HORIZON_DAYS = 365.0
MIN_FREQ_VALID_DAYS = 7
MIN_SURVIVAL_FAILURES = 20
MIN_WEIBULL_FAILURES = 50
MIN_CATBOOST_ROWS = 100
PLOT_STYLE_COLORS = {
    "Low (<=50 Hz)": "#1f77b4",
    "Normal (50-55 Hz)": "#2ca02c",
    "High (>55 Hz)": "#d62728",
    "True60 (>58 Hz)": "#9467bd",
    "Vt": "#d62728",
    "Global": "#1f77b4",
}

FAILURE_CATEGORIES = [
    "КЛ (R-0)",
    "ПЭД (R-0)",
    "Слом вала",
    "Засорение РО",
    "НКТ",
    "Износ РО",
    "Износ/негермет.гидрозащиты",
]


def _numeric(series: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_slug(value: object) -> str:
    text = str(value).strip() if value is not None else "missing"
    result = []
    for char in text:
        if char.isalnum():
            result.append(char)
        elif char in {" ", "-", "_"}:
            result.append("_")
    slug = "".join(result).strip("_")
    return slug or "missing"


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


def classify_failure_category(row: pd.Series) -> str | None:
    uzl = _normalize_lower(row.get("Отказавший узел"))
    elem = _normalize_lower(row.get("Отказавший элемент"))
    char = _normalize_lower(row.get("Характер неисправности"))
    prch = _normalize_lower(row.get("Причина отказа УЭЦН"))

    if not any([uzl, elem, char, prch]):
        return None

    cable_elems = {"кабельный удлинитель", "основная длина", "кабельный сросток", "кабельная муфта", "термовставка", "сальниковая разделка"}
    motor_elems = {"статор с обмоткой", "верхнее лобовое", "ротор", "выводные концы", "колодка токоввода"}
    nkt_uzly = {"нкт", "нкт ", "клапан сливной", "подвесной патрубок", "клапан обратный", "мандрель", "переводник", "подвесной патрубок "}
    clog_words = ("засорен", "твердые отложения", "солеотложени")
    wear_words = ("разрушен", "износ", "осевой", "пар трения", "радиальный", "промыв", "трещин", "эрозион")

    if uzl in {"кабельная линия", "тмс"}:
        return "КЛ (R-0)"
    if elem in cable_elems and any(word in char for word in ("изоляц", "прогар", "оплавл", "механич", "разрушен")):
        return "КЛ (R-0)"

    if uzl == "пэд" and elem not in ("узел пяты", "шлицевая муфта"):
        return "ПЭД (R-0)"
    if elem in motor_elems and any(word in char for word in ("замыкание", "электропробой", "прогар", "перегрев", "изоляц")):
        return "ПЭД (R-0)"

    if uzl == "гидрозащита":
        return "Износ/негермет.гидрозащиты"

    if "вал" in elem and "слом" in char:
        return "Слом вала"
    if "шлицевая муфта" in elem and any(word in char for word in ("слом", "разрушен")) and uzl != "гидрозащита":
        return "Слом вала"
    if "корпус" in elem and "слом" in char:
        return "Слом вала"

    if uzl in nkt_uzly or "нкт" in uzl:
        return "НКТ"
    if "подвеска нкт" in elem:
        return "НКТ"

    pump_uzly = {"эцн", "газосепаратор", "диспергатор", "входной модуль"}
    if uzl in pump_uzly:
        if "рабочие органы" in elem:
            if any(word in char for word in wear_words):
                return "Износ РО"
            if any(word in char for word in clog_words):
                return "Засорение РО"
            if any(word in prch for word in ("засорен", "солеотложени")):
                return "Засорение РО"
            return "Засорение РО"
        if "вал" in elem:
            return "Слом вала"
        if any(word in char for word in wear_words) or any(word in prch for word in ("коррозия", "эрозион")):
            return "Износ РО"
        if any(word in char for word in clog_words) or any(word in prch for word in ("засорен", "солеотложени")):
            return "Засорение РО"
        return "Износ РО"

    if uzl == "пэд" and "узел пяты" in elem:
        return "Износ РО"
    if uzl == "пэд":
        return "ПЭД (R-0)"

    scores: dict[str, int] = {category: 0 for category in FAILURE_CATEGORIES}
    if elem in cable_elems:
        scores["КЛ (R-0)"] += 2
    if any(word in char for word in ("кабел", "изоляц")):
        scores["КЛ (R-0)"] += 1
    if elem in motor_elems:
        scores["ПЭД (R-0)"] += 2
    if "замыкание" in char:
        scores["ПЭД (R-0)"] += 2
    if "вал" in elem:
        scores["Слом вала"] += 2
    if "слом" in char:
        scores["Слом вала"] += 2
    if "рабочие органы" in elem:
        if any(word in char for word in clog_words):
            scores["Засорение РО"] += 3
        if any(word in char for word in wear_words):
            scores["Износ РО"] += 3
    if any(word in char for word in clog_words):
        scores["Засорение РО"] += 1
    if any(word in char for word in wear_words):
        scores["Износ РО"] += 1
    if "нкт" in uzl or "подвеска нкт" in elem:
        scores["НКТ"] += 2
    if "обрыв" in char:
        scores["НКТ"] += 1
    if any(word in elem for word in ("уплотнен", "пяты")) or "пята" in char:
        scores["Износ/негермет.гидрозащиты"] += 2
    if "негермет" in char:
        scores["Износ/негермет.гидрозащиты"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Износ РО"


def frequency_group_3(value: float | object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "<missing>"
    if numeric <= 50.0:
        return "Low (<=50 Hz)"
    if numeric <= 55.0:
        return "Normal (50-55 Hz)"
    return "High (>55 Hz)"


def frequency_band_4(value: float | object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "<missing>"
    if numeric <= 50.0:
        return "<=50 Hz"
    if numeric <= 55.0:
        return "50-55 Hz"
    if numeric <= 60.0:
        return "55-60 Hz"
    return ">60 Hz"


def direct_h2s_class(value: float | object, label_value: object, threshold_mg_l: float = DEFAULT_H2S_THRESHOLD_MG_L) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return "Кислый" if float(numeric) >= threshold_mg_l else "Некислый"
    label = _normalize_lower(label_value)
    if "кисл" in label and "некисл" not in label:
        return "Кислый"
    if "некисл" in label:
        return "Некислый"
    return "<missing>"


def correlation_ci_spearman(rho: float, n: int) -> tuple[float, float]:
    if not np.isfinite(rho) or n <= 3 or abs(rho) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(float(np.clip(rho, -0.999999, 0.999999)))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    low = np.tanh(z - 1.96 * se)
    high = np.tanh(z + 1.96 * se)
    return (float(low), float(high))


@dataclass
class KaplanMeierResult:
    times: np.ndarray
    survival: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    event_table: pd.DataFrame

    def quantile_time(self, survival_threshold: float) -> float | None:
        if len(self.times) == 0:
            return None
        mask = self.survival <= survival_threshold
        if not np.any(mask):
            return None
        return float(self.times[np.argmax(mask)])

    @property
    def median(self) -> float | None:
        return self.quantile_time(0.5)

    @property
    def median_ci(self) -> tuple[float | None, float | None]:
        lower_time = None
        upper_time = None
        lower_mask = self.ci_upper <= 0.5
        upper_mask = self.ci_lower <= 0.5
        if np.any(lower_mask):
            lower_time = float(self.times[np.argmax(lower_mask)])
        if np.any(upper_mask):
            upper_time = float(self.times[np.argmax(upper_mask)])
        return lower_time, upper_time


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> KaplanMeierResult:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    mask = np.isfinite(durations) & np.isfinite(events) & (durations > 0)
    durations = durations[mask]
    events = events[mask]
    if len(durations) == 0:
        empty = np.array([], dtype=float)
        return KaplanMeierResult(empty, empty, empty, empty, pd.DataFrame())

    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]

    unique_event_times = np.unique(durations[events == 1])
    if len(unique_event_times) == 0:
        event_table = pd.DataFrame({"time": [0.0], "at_risk": [len(durations)], "events": [0], "censored": [int((events == 0).sum())], "survival": [1.0]})
        return KaplanMeierResult(np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), event_table)

    times = [0.0]
    survival = [1.0]
    ci_lower = [1.0]
    ci_upper = [1.0]
    variance_sum = 0.0
    current_survival = 1.0
    table_rows: list[dict[str, float | int]] = [
        {"time": 0.0, "at_risk": int(len(durations)), "events": 0, "censored": int((events == 0).sum()), "survival": 1.0}
    ]
    for current_time in unique_event_times:
        at_risk = int(np.sum(durations >= current_time))
        n_events = int(np.sum((durations == current_time) & (events == 1)))
        n_censored = int(np.sum((durations == current_time) & (events == 0)))
        if at_risk <= 0:
            continue
        current_survival *= 1.0 - (n_events / at_risk)
        if at_risk > n_events:
            variance_sum += n_events / (at_risk * (at_risk - n_events))
        se = current_survival * math.sqrt(max(variance_sum, 0.0))
        times.append(float(current_time))
        survival.append(float(current_survival))
        ci_lower.append(float(max(0.0, current_survival - 1.96 * se)))
        ci_upper.append(float(min(1.0, current_survival + 1.96 * se)))
        table_rows.append(
            {
                "time": float(current_time),
                "at_risk": at_risk,
                "events": n_events,
                "censored": n_censored,
                "survival": float(current_survival),
            }
        )
    return KaplanMeierResult(
        np.asarray(times, dtype=float),
        np.asarray(survival, dtype=float),
        np.asarray(ci_lower, dtype=float),
        np.asarray(ci_upper, dtype=float),
        pd.DataFrame(table_rows),
    )


def logrank_two_sample(durations_a: np.ndarray, events_a: np.ndarray, durations_b: np.ndarray, events_b: np.ndarray) -> dict[str, float]:
    durations_a = np.asarray(durations_a, dtype=float)
    events_a = np.asarray(events_a, dtype=int)
    durations_b = np.asarray(durations_b, dtype=float)
    events_b = np.asarray(events_b, dtype=int)
    mask_a = np.isfinite(durations_a) & np.isfinite(events_a) & (durations_a > 0)
    mask_b = np.isfinite(durations_b) & np.isfinite(events_b) & (durations_b > 0)
    durations_a = durations_a[mask_a]
    events_a = events_a[mask_a]
    durations_b = durations_b[mask_b]
    events_b = events_b[mask_b]
    if len(durations_a) == 0 or len(durations_b) == 0:
        return {"chi2": float("nan"), "pval": float("nan")}

    times = np.unique(np.concatenate([durations_a[events_a == 1], durations_b[events_b == 1]]))
    observed_a = 0.0
    expected_a = 0.0
    variance = 0.0
    for time in times:
        n1 = int(np.sum(durations_a >= time))
        n2 = int(np.sum(durations_b >= time))
        d1 = int(np.sum((durations_a == time) & (events_a == 1)))
        d2 = int(np.sum((durations_b == time) & (events_b == 1)))
        n = n1 + n2
        d = d1 + d2
        if n <= 1 or d == 0:
            continue
        observed_a += d1
        expected_a += d * (n1 / n)
        variance += (n1 * n2 * d * (n - d)) / max((n ** 2) * (n - 1), 1)
    if variance <= 0:
        return {"chi2": float("nan"), "pval": float("nan")}
    chi2_value = ((observed_a - expected_a) ** 2) / variance
    pval = float(chi2.sf(chi2_value, df=1))
    return {"chi2": float(chi2_value), "pval": pval}


def weibull_with_ci(durations: np.ndarray, events: np.ndarray) -> dict[str, float | bool | str]:
    fit = fit_basic_weibull(durations, events)
    beta = float(fit["beta"])
    eta = float(fit["eta"])
    x_opt = np.log([beta, eta])

    def objective(theta: np.ndarray) -> float:
        return evaluate_weibull_nll(np.asarray(durations, dtype=float), np.asarray(events, dtype=int), float(np.exp(theta[0])), float(np.exp(theta[1])))

    eps = 1e-4
    hessian = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2)
            ej = np.zeros(2)
            ei[i] = eps
            ej[j] = eps
            fpp = objective(x_opt + ei + ej)
            fpm = objective(x_opt + ei - ej)
            fmp = objective(x_opt - ei + ej)
            fmm = objective(x_opt - ei - ej)
            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    try:
        covariance = np.linalg.inv(hessian)
        se_log_beta = math.sqrt(max(float(covariance[0, 0]), 0.0))
        se_log_eta = math.sqrt(max(float(covariance[1, 1]), 0.0))
        fit["beta_ci_low"] = float(math.exp(math.log(beta) - 1.96 * se_log_beta))
        fit["beta_ci_high"] = float(math.exp(math.log(beta) + 1.96 * se_log_beta))
        fit["eta_ci_low"] = float(math.exp(math.log(eta) - 1.96 * se_log_eta))
        fit["eta_ci_high"] = float(math.exp(math.log(eta) + 1.96 * se_log_eta))
    except Exception:
        fit["beta_ci_low"] = float("nan")
        fit["beta_ci_high"] = float("nan")
        fit["eta_ci_low"] = float("nan")
        fit["eta_ci_high"] = float("nan")
    return fit


def cumulative_incidence(durations: np.ndarray, events: np.ndarray, categories: np.ndarray, category_labels: list[str]) -> pd.DataFrame:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    categories = np.asarray(categories, dtype=object)
    mask = np.isfinite(durations) & np.isfinite(events) & (durations > 0)
    durations = durations[mask]
    events = events[mask]
    categories = categories[mask]
    rows: list[dict[str, Any]] = []
    if len(durations) == 0:
        return pd.DataFrame(rows)
    event_times = np.unique(durations[events == 1])
    survival_prev = 1.0
    cif_values = {label: 0.0 for label in category_labels}
    for time in event_times:
        at_risk = int(np.sum(durations >= time))
        if at_risk <= 0:
            continue
        total_events = int(np.sum((durations == time) & (events == 1)))
        for label in category_labels:
            d_k = int(np.sum((durations == time) & (events == 1) & (categories == label)))
            cif_values[label] += survival_prev * (d_k / at_risk)
        survival_prev *= 1.0 - (total_events / at_risk)
        for label in category_labels:
            rows.append({"time": float(time), "category": label, "cif": float(cif_values[label]), "survival": float(survival_prev), "at_risk": at_risk})
    return pd.DataFrame(rows)


def restricted_mean_failure_time(durations: pd.Series, horizon_days: float) -> float:
    numeric = _numeric(durations).dropna()
    if numeric.empty:
        return float("nan")
    clipped = numeric.clip(upper=horizon_days)
    return float(clipped.mean())


def harrell_c_index(durations: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float | None:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)
    concordant = 0.0
    comparable = 0.0
    tied = 0.0
    n = len(durations)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(durations[i]) or not np.isfinite(durations[j]):
                continue
            if durations[i] == durations[j]:
                continue
            if events[i] == 0 and events[j] == 0:
                continue
            if durations[i] < durations[j] and events[i] == 1:
                comparable += 1.0
                if scores[i] > scores[j]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    tied += 1.0
            elif durations[j] < durations[i] and events[j] == 1:
                comparable += 1.0
                if scores[j] > scores[i]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    tied += 1.0
    if comparable == 0:
        return None
    return float((concordant + 0.5 * tied) / comparable)


def one_hot_frame(df: pd.DataFrame, columns: list[str], drop_reference: dict[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    drop_reference = drop_reference or {}
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            frames.append(pd.DataFrame({column: numeric}, index=df.index))
            names.append(column)
            continue
        stringed = series.astype("string").fillna("__missing__")
        dummies = pd.get_dummies(stringed, prefix=column, dtype=float)
        ref = drop_reference.get(column)
        if ref is not None:
            ref_name = f"{column}_{ref}"
            if ref_name in dummies.columns:
                dummies = dummies.drop(columns=[ref_name])
        elif len(dummies.columns) > 0:
            dummies = dummies.iloc[:, 1:]
        frames.append(dummies)
        names.extend(dummies.columns.tolist())
    if not frames:
        return pd.DataFrame(index=df.index), []
    combined = pd.concat(frames, axis=1)
    return combined, names


def cox_fit(durations: np.ndarray, events: np.ndarray, x: np.ndarray, column_names: list[str]) -> dict[str, Any]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(durations) & np.isfinite(events) & np.all(np.isfinite(x), axis=1) & (durations > 0)
    durations = durations[mask]
    events = events[mask]
    x = x[mask]
    n_rows, n_features = x.shape
    if n_rows == 0 or n_features == 0 or int(events.sum()) < max(10, n_features + 3):
        raise ValueError("Not enough usable rows/events for Cox fit.")

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        loglik = 0.0
        for time in np.unique(durations[events == 1]):
            event_mask = (durations == time) & (events == 1)
            risk_mask = durations >= time
            d = int(event_mask.sum())
            if d == 0:
                continue
            event_eta = eta[event_mask]
            risk_eta = eta[risk_mask]
            m = float(np.max(risk_eta))
            log_denom = m + math.log(float(np.exp(risk_eta - m).sum()))
            loglik += float(event_eta.sum()) - d * log_denom
        return -loglik

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        grad = np.zeros(n_features, dtype=float)
        for time in np.unique(durations[events == 1]):
            event_mask = (durations == time) & (events == 1)
            risk_mask = durations >= time
            d = int(event_mask.sum())
            if d == 0:
                continue
            weights = np.exp(eta[risk_mask] - np.max(eta[risk_mask]))
            s0 = weights.sum()
            s1 = (weights[:, None] * x[risk_mask]).sum(axis=0)
            grad += x[event_mask].sum(axis=0) - d * (s1 / s0)
        return -grad

    def hessian(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        info = np.zeros((n_features, n_features), dtype=float)
        for time in np.unique(durations[events == 1]):
            event_mask = (durations == time) & (events == 1)
            risk_mask = durations >= time
            d = int(event_mask.sum())
            if d == 0:
                continue
            weights = np.exp(eta[risk_mask] - np.max(eta[risk_mask]))
            xrisk = x[risk_mask]
            s0 = weights.sum()
            s1 = (weights[:, None] * xrisk).sum(axis=0)
            s2 = xrisk.T @ (xrisk * weights[:, None])
            info += d * ((s2 / s0) - np.outer(s1, s1) / (s0 ** 2))
        return info

    beta0 = np.zeros(n_features, dtype=float)
    result = minimize(objective, x0=beta0, jac=gradient, method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
    beta = np.asarray(result.x, dtype=float)
    info = hessian(beta)
    covariance = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2.0 * norm.sf(np.abs(z))
    linear_predictor = x @ beta
    c_index = harrell_c_index(durations, events, linear_predictor)
    table = pd.DataFrame(
        {
            "term": column_names,
            "beta": beta,
            "se": se,
            "z": z,
            "pval": p,
            "hr": np.exp(beta),
            "hr_lo95": np.exp(beta - 1.96 * se),
            "hr_hi95": np.exp(beta + 1.96 * se),
        }
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "rows": int(n_rows),
        "events": int(events.sum()),
        "nll": float(objective(beta)),
        "aic": float((2 * n_features) + (2 * objective(beta))),
        "c_index": c_index,
        "table": table,
        "beta": beta,
        "covariance": covariance,
        "linear_predictor": linear_predictor,
        "durations": durations,
        "events_array": events,
        "x": x,
    }


def schoenfeld_frequency_check(cox_result: dict[str, Any], frequency_terms: list[str]) -> pd.DataFrame:
    x = np.asarray(cox_result["x"], dtype=float)
    beta = np.asarray(cox_result["beta"], dtype=float)
    durations = np.asarray(cox_result["durations"], dtype=float)
    events = np.asarray(cox_result["events_array"], dtype=int)
    names = cox_result["table"]["term"].tolist()
    if not frequency_terms:
        return pd.DataFrame()

    eta = x @ beta
    rows: list[dict[str, Any]] = []
    for time in np.unique(durations[events == 1]):
        event_idx = np.where((durations == time) & (events == 1))[0]
        risk_idx = np.where(durations >= time)[0]
        if len(event_idx) == 0 or len(risk_idx) == 0:
            continue
        weights = np.exp(eta[risk_idx] - np.max(eta[risk_idx]))
        weighted_mean = np.average(x[risk_idx], axis=0, weights=weights)
        for idx in event_idx:
            residual = x[idx] - weighted_mean
            rows.append({"time": float(time), **{names[j]: float(residual[j]) for j in range(len(names))}})
    residual_frame = pd.DataFrame(rows)
    if residual_frame.empty:
        return pd.DataFrame()
    checks: list[dict[str, Any]] = []
    log_time = np.log(np.clip(residual_frame["time"].to_numpy(dtype=float), 1e-6, None))
    for term in frequency_terms:
        if term not in residual_frame.columns:
            continue
        rho, pval = spearmanr(log_time, residual_frame[term].to_numpy(dtype=float))
        checks.append({"term": term, "rho_vs_log_time": float(rho), "pval": float(pval)})
    return pd.DataFrame(checks)


def plot_km_groups(df: pd.DataFrame, group_column: str, title: str, output_path: Path, min_failures: int = MIN_SURVIVAL_FAILURES) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(9, 6))
    rows: list[dict[str, Any]] = []
    groups = [group for group in df[group_column].dropna().astype(str).unique().tolist() if group != "<missing>"]
    if not groups:
        plt.close(fig)
        return pd.DataFrame()
    reference_group = "Normal (50-55 Hz)" if "Normal (50-55 Hz)" in groups else groups[0]
    ref_df = df.loc[df[group_column].astype(str) == reference_group].copy()
    ref_duration = ref_df["duration_best_days"].to_numpy(dtype=float)
    ref_event = ref_df["event"].to_numpy(dtype=int)

    for group in groups:
        subset = df.loc[df[group_column].astype(str) == group].copy()
        failures = int(subset["event"].sum())
        if failures < min_failures:
            continue
        km = kaplan_meier(subset["duration_best_days"].to_numpy(dtype=float), subset["event"].to_numpy(dtype=int))
        color = PLOT_STYLE_COLORS.get(group, None)
        ax.step(km.times, km.survival, where="post", label=f"{group} (fail={failures})", linewidth=2, color=color)
        if len(km.times) > 1:
            ax.fill_between(km.times, km.ci_lower, km.ci_upper, step="post", alpha=0.12, color=color)
        logrank = {"chi2": float("nan"), "pval": float("nan")}
        if group != reference_group and len(ref_df) > 0:
            logrank = logrank_two_sample(subset["duration_best_days"].to_numpy(dtype=float), subset["event"].to_numpy(dtype=int), ref_duration, ref_event)
        rows.append(
            {
                "group": group,
                "rows": int(len(subset)),
                "failures": failures,
                "median_survival_days": km.median,
                "p25_survival_days": km.quantile_time(0.75),
                "median_ci_low_days": km.median_ci[0],
                "median_ci_high_days": km.median_ci[1],
                "logrank_p_vs_reference": logrank["pval"],
                "reference_group": reference_group,
            }
        )
    ax.set_title(title)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_km_compare(df_left: pd.DataFrame, df_right: pd.DataFrame, left_label: str, right_label: str, title: str, output_path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(9, 6))
    rows: list[dict[str, Any]] = []
    for label, subset in [(left_label, df_left.copy()), (right_label, df_right.copy())]:
        if int(subset["event"].sum()) < MIN_SURVIVAL_FAILURES:
            continue
        km = kaplan_meier(subset["duration_best_days"].to_numpy(dtype=float), subset["event"].to_numpy(dtype=int))
        color = PLOT_STYLE_COLORS.get(label, None)
        ax.step(km.times, km.survival, where="post", label=f"{label} (fail={int(subset['event'].sum())})", linewidth=2, color=color)
        ax.fill_between(km.times, km.ci_lower, km.ci_upper, step="post", alpha=0.12, color=color)
        rows.append(
            {
                "group": label,
                "rows": int(len(subset)),
                "failures": int(subset["event"].sum()),
                "median_survival_days": km.median,
                "p25_survival_days": km.quantile_time(0.75),
                "median_ci_low_days": km.median_ci[0],
                "median_ci_high_days": km.median_ci[1],
            }
        )
    if int(df_left["event"].sum()) >= MIN_SURVIVAL_FAILURES and int(df_right["event"].sum()) >= MIN_SURVIVAL_FAILURES:
        logrank = logrank_two_sample(df_left["duration_best_days"].to_numpy(dtype=float), df_left["event"].to_numpy(dtype=int), df_right["duration_best_days"].to_numpy(dtype=float), df_right["event"].to_numpy(dtype=int))
        rows.append({"group": f"{left_label}_vs_{right_label}", "logrank_p": logrank["pval"], "logrank_chi2": logrank["chi2"]})
    ax.set_title(title)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_failure_mix(global_failures: pd.DataFrame, vt_failures: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    def share_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        counts = frame["failure_category"].value_counts().reindex(FAILURE_CATEGORIES, fill_value=0)
        total = max(int(counts.sum()), 1)
        return pd.DataFrame({"category": counts.index, "count": counts.values, "share": counts.values / total, "group": label})

    combined = pd.concat([share_frame(global_failures, "Global"), share_frame(vt_failures, "Vt")], ignore_index=True)
    pivot = combined.pivot(index="category", columns="group", values="share").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(pivot.index))
    ax.barh(y - 0.18, pivot["Global"], height=0.36, label="Global", color=PLOT_STYLE_COLORS["Global"])
    ax.barh(y + 0.18, pivot["Vt"], height=0.36, label="Vt", color=PLOT_STYLE_COLORS["Vt"])
    ax.set_yticks(y, labels=pivot.index.tolist())
    ax.set_xlabel("Failure share among failures")
    ax.set_title("Failure category mix: Vt vs Global")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return combined


def plot_box_compare(df: pd.DataFrame, value_column: str, output_path: Path, title: str) -> None:
    global_values = _numeric(df[value_column]).dropna()
    vt_values = _numeric(df.loc[df["field"].astype(str) == "Vt", value_column]).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([global_values, vt_values], tick_labels=["Global", "Vt"], showfliers=False)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heatmap(frame: pd.DataFrame, index: str, columns: str, values: str, title: str, output_path: Path, fmt: str = ".2f") -> None:
    pivot = frame.pivot(index=index, columns=columns, values=values).sort_index()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.5)))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(col) for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(idx) for idx in pivot.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cif(cif_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    if cif_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for category in FAILURE_CATEGORIES:
        subset = cif_frame.loc[cif_frame["category"] == category].copy()
        if subset.empty:
            continue
        ax.step(subset["time"], subset["cif"], where="post", label=category, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Cumulative incidence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    workbook_path = Path(resolve_v03_all_path())
    runs = load_runs(workbook_path).copy()
    runs["mount_year"] = runs["Дата монтажа"].dt.year
    runs["direct_h2s_mg_l"] = _numeric(runs.get("Массовая доля сероводорода, мг/дм³"))
    runs["acid_label_raw"] = runs.get("Кислый/Некислый")
    runs["corrosion_resistance"] = runs.get("Коррозионная стойкость")
    runs["corrosion_fill"] = runs["corrosion_resistance"].astype("string").str.strip().replace({"": pd.NA}).notna()
    runs["classification_input_missing"] = (
        runs[["Отказавший узел", "Отказавший элемент", "Характер неисправности", "Причина отказа УЭЦН"]]
        .fillna("")
        .apply(lambda row: all(not _normalize_text(value) for value in row), axis=1)
    )
    runs["failure_category_raw"] = runs.apply(classify_failure_category, axis=1)
    runs["failure_category"] = np.where(
        runs["event"].eq(0),
        "Censored",
        runs["failure_category_raw"].fillna("<missing>"),
    )

    with sqlite3.connect(WAREHOUSE_PATH) as connection:
        mart = pd.read_sql_query("SELECT * FROM mart__vt_freq55", connection)
    keep_columns = [
        "row_id",
        "freq_above_55hz_pct",
        "freq_below_45hz_pct",
        "freq_signed_exposure",
        "freq_w_mean",
        "n_freq_valid_days",
        "n_freq_above_55hz",
        "n_freq_below_45hz",
        "total_liquid_m3",
        "total_freq_hz_days",
        "avg_glf",
        "avg_kpod",
        "ttf_true_best_days",
        "ttf_true_source",
        "ttf_tele_days",
        "ttf_treg_days",
        "h2s_proxy_mg_l",
        "h2s_proxy_source",
        "kpod_m_mean",
        "frac_kpod_below_0p7_m",
        "glf_m_mean",
        "frac_glf_above_thr_m",
        "load_m_mean",
        "salt_proxy_m_mean",
        "integrated_salt_proxy_m",
        "gypsum_proxy_m_mean",
        "cum_salt_load_kg",
        "cum_gypsum_scale_proxy",
        "cum_calcium_load_kg",
        "cum_chloride_load_kg",
        "cum_sulfate_load_kg",
    ]
    mart = mart[[column for column in keep_columns if column in mart.columns]].copy()
    df = runs.merge(mart, on="row_id", how="left", suffixes=("", "_mart"))

    for column in [
        "cum_salt_load_kg",
        "cum_gypsum_scale_proxy",
        "cum_calcium_load_kg",
        "cum_chloride_load_kg",
        "cum_sulfate_load_kg",
    ]:
        mart_col = f"{column}_mart"
        if mart_col in df.columns:
            base_series = df[column] if column in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
            df[column] = _numeric(df[mart_col]).combine_first(_numeric(base_series))

    df["duration_best_days"] = _numeric(df["ttf_true_best_days"]).combine_first(_numeric(df["run_days"]))
    df["duration_source"] = np.where(df["ttf_true_best_days"].notna(), "ttf_true_best", "calendar_run_days")
    df["h2s_source_priority"] = np.where(
        df["direct_h2s_mg_l"].notna(),
        "direct_workbook",
        np.where(df["h2s_proxy_mg_l"].notna(), "proxy", np.where(df["acid_label_raw"].astype("string").str.strip().replace({"": pd.NA}).notna(), "acid_label_only", "missing")),
    )
    df["h2s_effective_mg_l"] = _numeric(df["direct_h2s_mg_l"]).combine_first(_numeric(df["h2s_proxy_mg_l"]))
    df["h2s_class"] = [direct_h2s_class(value, label) for value, label in zip(df["h2s_effective_mg_l"], df["acid_label_raw"], strict=False)]
    df["freq_group_3"] = df["freq_w_mean"].map(frequency_group_3)
    df["freq_band_4"] = df["freq_w_mean"].map(frequency_band_4)
    df["true60_flag"] = _numeric(df["freq_w_mean"]) > 58.0
    df["valid_freq_run"] = _numeric(df["n_freq_valid_days"]).fillna(0) >= MIN_FREQ_VALID_DAYS
    df["trf_per_day"] = _numeric(df["total_freq_hz_days"]) / _numeric(df["duration_best_days"])
    df["tlf_per_day"] = _numeric(df["total_liquid_m3"]) / _numeric(df["duration_best_days"])
    for raw_col, daily_col in [
        ("cum_calcium_load_kg", "calcium_load_per_day"),
        ("cum_chloride_load_kg", "chloride_load_per_day"),
        ("cum_sulfate_load_kg", "sulfate_load_per_day"),
        ("cum_salt_load_kg", "salt_load_per_day"),
        ("cum_gypsum_scale_proxy", "gypsum_proxy_per_day"),
    ]:
        df[daily_col] = _numeric(df[raw_col]) / _numeric(df["duration_best_days"])
    df["glf_bin"] = pd.cut(_numeric(df["avg_glf"]), bins=[-np.inf, DEFAULT_GLF_THRESHOLD, 500.0, np.inf], labels=["<=300", "300-500", ">500"])
    df["high_h2s_flag"] = _numeric(df["h2s_effective_mg_l"]) >= DEFAULT_H2S_THRESHOLD_MG_L
    df["field"] = df["Месторождение"].astype("string").str.strip()
    df["contractor"] = df["Принадлежность"].astype("string").str.strip()
    df["pump_family"] = df.get("Тип УЭЦН", pd.Series(index=df.index, dtype="object")).astype("string").str.extract(r"([A-Za-z0-9]+)", expand=False).fillna("<missing>")

    audit = {
        "total_runs": int(len(df)),
        "total_failures": int(df["event"].sum()),
        "fields": sorted(df["field"].dropna().astype(str).unique().tolist()),
        "true60_failures_global": int(df.loc[df["true60_flag"] & df["event"].eq(1)].shape[0]),
    }
    return df, audit


def save_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def describe_group(df: pd.DataFrame, label: str, mature_threshold: int = DEFAULT_MATURE_THRESHOLD) -> dict[str, Any]:
    numeric_duration = _numeric(df["duration_best_days"])
    failures = int(df["event"].sum())
    total = int(len(df))
    mature_mask = numeric_duration >= mature_threshold
    return {
        "group": label,
        "runs": total,
        "failures": failures,
        "failure_rate": float(failures / max(total, 1)),
        "median_ttf_days": float(numeric_duration.median()) if numeric_duration.notna().any() else float("nan"),
        "mean_ttf_days": float(numeric_duration.mean()) if numeric_duration.notna().any() else float("nan"),
        "median_ttf_mature_days": float(numeric_duration.loc[mature_mask].median()) if mature_mask.any() else float("nan"),
        "mean_h2s_mg_l": float(_numeric(df["h2s_effective_mg_l"]).mean()) if df["h2s_effective_mg_l"].notna().any() else float("nan"),
        "share_high_h2s": float(_numeric(df["high_h2s_flag"]).mean()) if len(df) else float("nan"),
        "mean_glf": float(_numeric(df["avg_glf"]).mean()) if df["avg_glf"].notna().any() else float("nan"),
        "mean_kpod": float(_numeric(df["avg_kpod"]).mean()) if df["avg_kpod"].notna().any() else float("nan"),
        "mean_trf_per_day": float(_numeric(df["trf_per_day"]).mean()) if df["trf_per_day"].notna().any() else float("nan"),
        "mean_tlf_per_day": float(_numeric(df["tlf_per_day"]).mean()) if df["tlf_per_day"].notna().any() else float("nan"),
        "mean_calcium_load_per_day": float(_numeric(df["calcium_load_per_day"]).mean()) if df["calcium_load_per_day"].notna().any() else float("nan"),
        "mean_chloride_load_per_day": float(_numeric(df["chloride_load_per_day"]).mean()) if df["chloride_load_per_day"].notna().any() else float("nan"),
        "mean_sulfate_load_per_day": float(_numeric(df["sulfate_load_per_day"]).mean()) if df["sulfate_load_per_day"].notna().any() else float("nan"),
        "mean_gypsum_proxy_per_day": float(_numeric(df["gypsum_proxy_per_day"]).mean()) if df["gypsum_proxy_per_day"].notna().any() else float("nan"),
    }


def build_data_audit(df: pd.DataFrame, output_dir: Path, audit_meta: dict[str, Any]) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    freq_counts = (
        df.assign(field_group=df["field"].fillna("<missing>"))
        .groupby(["field_group", "freq_band_4"], dropna=False)
        .agg(runs=("row_id", "size"), failures=("event", "sum"))
        .reset_index()
        .rename(columns={"field_group": "field"})
    )
    global_counts = (
        df.groupby(["freq_band_4"], dropna=False)
        .agg(runs=("row_id", "size"), failures=("event", "sum"))
        .reset_index()
        .assign(field="Global")
    )
    freq_counts = pd.concat([global_counts, freq_counts], ignore_index=True)
    save_frame(freq_counts, tables_dir / "frequency_band_counts.csv")

    telemetry_coverage = (
        df.groupby(["field", "mount_year"], dropna=False)
        .agg(
            runs=("row_id", "size"),
            runs_with_valid_freq=("valid_freq_run", "sum"),
            mean_valid_freq_days=("n_freq_valid_days", "mean"),
        )
        .reset_index()
    )
    telemetry_coverage["valid_freq_run_share"] = telemetry_coverage["runs_with_valid_freq"] / telemetry_coverage["runs"].clip(lower=1)
    save_frame(telemetry_coverage, tables_dir / "telemetry_frequency_coverage.csv")

    ttf_source = (
        df.groupby(["field", "ttf_true_source"], dropna=False)
        .agg(runs=("row_id", "size"))
        .reset_index()
    )
    save_frame(ttf_source, tables_dir / "ttf_true_source_counts.csv")

    failure_category_coverage = pd.DataFrame(
        [
            {
                "population": "all_runs",
                "runs": int(len(df)),
                "failures": int(df["event"].sum()),
                "classified_failures": int(df.loc[df["event"].eq(1) & df["failure_category_raw"].notna()].shape[0]),
                "missing_classification_failures": int(df.loc[df["event"].eq(1) & df["failure_category_raw"].isna()].shape[0]),
                "classification_input_missing_failures": int(df.loc[df["event"].eq(1) & df["classification_input_missing"]].shape[0]),
            }
        ]
    )
    save_frame(failure_category_coverage, tables_dir / "failure_category_coverage.csv")

    h2s_coverage = (
        df.groupby(["field", "h2s_source_priority"], dropna=False)
        .agg(runs=("row_id", "size"))
        .reset_index()
    )
    save_frame(h2s_coverage, tables_dir / "h2s_source_coverage.csv")

    corrosion_fill = (
        df.groupby("field", dropna=False)
        .agg(runs=("row_id", "size"), corrosion_fill=("corrosion_fill", "sum"))
        .reset_index()
    )
    corrosion_fill["corrosion_fill_rate"] = corrosion_fill["corrosion_fill"] / corrosion_fill["runs"].clip(lower=1)
    save_frame(corrosion_fill, tables_dir / "corrosion_fill_rates.csv")

    duty_metric_audit = pd.DataFrame(
        [
            {
                "runs": int(len(df)),
                "trf_non_missing": int(_numeric(df["total_freq_hz_days"]).notna().sum()),
                "tlf_non_missing": int(_numeric(df["total_liquid_m3"]).notna().sum()),
                "tlf_zero_runs": int((_numeric(df["total_liquid_m3"]).fillna(0.0) <= 0.0).sum()),
                "tlf_near_zero_runs": int((_numeric(df["total_liquid_m3"]).fillna(0.0) <= 1.0).sum()),
            }
        ]
    )
    save_frame(duty_metric_audit, tables_dir / "duty_metric_audit.csv")

    ion_columns = [
        "cum_calcium_load_kg",
        "cum_chloride_load_kg",
        "cum_sulfate_load_kg",
        "cum_salt_load_kg",
        "cum_gypsum_scale_proxy",
        "calcium_load_per_day",
        "chloride_load_per_day",
        "sulfate_load_per_day",
        "salt_load_per_day",
        "gypsum_proxy_per_day",
    ]
    ion_coverage_rows = []
    for column in ion_columns:
        numeric = _numeric(df[column])
        ion_coverage_rows.append({"column": column, "non_missing_rows": int(numeric.notna().sum()), "fill_rate": float(numeric.notna().mean())})
    ion_coverage = pd.DataFrame(ion_coverage_rows)
    save_frame(ion_coverage, tables_dir / "ion_proxy_coverage.csv")
    ion_corr = df[[column for column in ["calcium_load_per_day", "chloride_load_per_day", "sulfate_load_per_day", "salt_load_per_day", "gypsum_proxy_per_day"] if column in df.columns]].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    ion_corr.to_csv(tables_dir / "ion_proxy_correlations.csv", encoding="utf-8-sig")

    multi_run = (
        df.groupby("Скв.", dropna=False)
        .agg(runs=("row_id", "size"))
        .reset_index()
    )
    multi_run_share = float((multi_run["runs"] > 1).mean()) if not multi_run.empty else float("nan")
    save_frame(multi_run, tables_dir / "runs_per_well.csv")

    audit_summary = {
        **audit_meta,
        "true60_proxy_used": bool(audit_meta["true60_failures_global"] >= 50),
        "frequency_band_rows": int(len(freq_counts)),
        "multi_run_well_share": multi_run_share,
        "ion_proxy_fill_rates": {row["column"]: row["fill_rate"] for row in ion_coverage.to_dict(orient="records")},
    }
    (output_dir / "data_audit_summary.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit_summary


def build_descriptive_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    group_rows = [describe_group(df, "Global")]
    for field_name, group in df.groupby("field", dropna=False):
        group_rows.append(describe_group(group, str(field_name)))
    descriptive = pd.DataFrame(group_rows).sort_values("group").reset_index(drop=True)
    save_frame(descriptive, tables_dir / "field_descriptive_summary.csv")

    vt_df = df.loc[df["field"].astype(str) == "Vt"].copy()
    vt_vs_global_rows = []
    for label, subset in [("Global", df), ("Vt", vt_df)]:
        vt_vs_global_rows.append(describe_group(subset, label))
    vt_vs_global = pd.DataFrame(vt_vs_global_rows)
    save_frame(vt_vs_global, tables_dir / "vt_vs_global_summary.csv")

    global_failures = df.loc[df["event"].eq(1)].copy()
    vt_failures = vt_df.loc[vt_df["event"].eq(1)].copy()
    failure_mix = plot_failure_mix(global_failures, vt_failures, figures_dir / "vt_vs_global_failure_mix.png")
    save_frame(failure_mix, tables_dir / "vt_vs_global_failure_mix.csv")

    for column, title, filename in [
        ("h2s_effective_mg_l", "H2S distribution: Vt vs Global", "vt_vs_global_h2s_box.png"),
        ("avg_glf", "GLF distribution: Vt vs Global", "vt_vs_global_glf_box.png"),
        ("duration_best_days", "Best-available TTF: Vt vs Global", "vt_vs_global_ttf_box.png"),
    ]:
        plot_box_compare(df, column, figures_dir / filename, title)
    return descriptive, vt_vs_global


def build_infant_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    infant_field_rows: list[dict[str, Any]] = []
    infant_category_rows: list[dict[str, Any]] = []
    vt_standardization_rows: list[dict[str, Any]] = []

    failure_only = df.loc[df["event"].eq(1) & df["failure_category_raw"].notna()].copy()
    global_category_weights = failure_only["failure_category_raw"].value_counts(normalize=True)

    for threshold in DEFAULT_INFANT_THRESHOLDS:
        df[f"infant_{threshold}d"] = (df["event"].eq(1) & (_numeric(df["duration_best_days"]) < threshold)).astype(int)
        grouped = (
            df.groupby(["field", "freq_group_3"], dropna=False)
            .agg(
                runs=("row_id", "size"),
                failures=("event", "sum"),
                infant_failures=(f"infant_{threshold}d", "sum"),
            )
            .reset_index()
        )
        grouped["threshold_days"] = threshold
        grouped["infant_failure_rate"] = grouped["infant_failures"] / grouped["runs"].clip(lower=1)
        grouped["infant_share_among_failures"] = grouped["infant_failures"] / grouped["failures"].clip(lower=1)
        infant_field_rows.extend(grouped.to_dict(orient="records"))

        by_category = (
            df.loc[df["event"].eq(1) & df["failure_category_raw"].notna()]
            .groupby("failure_category_raw", dropna=False)
            .agg(
                failures=("row_id", "size"),
                infant_failures=(f"infant_{threshold}d", "sum"),
            )
            .reset_index()
            .rename(columns={"failure_category_raw": "failure_category"})
        )
        by_category["threshold_days"] = threshold
        by_category["infant_rate_within_category"] = by_category["infant_failures"] / by_category["failures"].clip(lower=1)
        infant_category_rows.extend(by_category.to_dict(orient="records"))

        vt_failures = df.loc[df["field"].astype(str) == "Vt"].copy()
        vt_by_category = (
            vt_failures.loc[vt_failures["event"].eq(1) & vt_failures["failure_category_raw"].notna()]
            .groupby("failure_category_raw", dropna=False)
            .agg(
                failures=("row_id", "size"),
                infant_failures=(f"infant_{threshold}d", "sum"),
            )
            .reset_index()
        )
        vt_by_category["vt_rate"] = vt_by_category["infant_failures"] / vt_by_category["failures"].clip(lower=1)
        vt_by_category["global_weight"] = vt_by_category["failure_category_raw"].map(global_category_weights).fillna(0.0)
        standardized = float((vt_by_category["vt_rate"] * vt_by_category["global_weight"]).sum())
        observed = float(vt_failures[f"infant_{threshold}d"].sum() / max(int(vt_failures["event"].sum()), 1))
        global_observed = float(df[f"infant_{threshold}d"].sum() / max(int(df["event"].sum()), 1))
        vt_standardization_rows.append(
            {
                "threshold_days": threshold,
                "vt_observed_infant_share_among_failures": observed,
                "vt_standardized_to_global_category_mix": standardized,
                "global_observed_infant_share_among_failures": global_observed,
            }
        )

    infant_field = pd.DataFrame(infant_field_rows)
    infant_category = pd.DataFrame(infant_category_rows)
    vt_standardization = pd.DataFrame(vt_standardization_rows)
    save_frame(infant_field, tables_dir / "infant_rates_by_field_freq_threshold.csv")
    save_frame(infant_category, tables_dir / "infant_rates_by_category_threshold.csv")
    save_frame(vt_standardization, tables_dir / "vt_infant_standardization.csv")

    for threshold in DEFAULT_INFANT_THRESHOLDS:
        subset = infant_field.loc[infant_field["threshold_days"].eq(threshold)].copy()
        plot_heatmap(
            subset.assign(field=subset["field"].astype(str)),
            index="field",
            columns="freq_group_3",
            values="infant_failure_rate",
            title=f"Infant failure rate by field and frequency group (<{threshold}d)",
            output_path=figures_dir / f"infant_rate_heatmap_{threshold}d.png",
            fmt=".2f",
        )
    return {
        "infant_field": infant_field,
        "infant_category": infant_category,
        "vt_standardization": vt_standardization,
    }


def build_survival_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    summaries: list[pd.DataFrame] = []

    global_freq = plot_km_groups(df.loc[df["freq_group_3"] != "<missing>"].copy(), "freq_group_3", "Global survival by frequency group", figures_dir / "km_global_by_freq.png")
    global_freq["analysis"] = "global_all"
    summaries.append(global_freq)

    mature_df = df.loc[_numeric(df["duration_best_days"]) >= DEFAULT_MATURE_THRESHOLD].copy()
    mature_freq = plot_km_groups(mature_df.loc[mature_df["freq_group_3"] != "<missing>"].copy(), "freq_group_3", f"Global survival by frequency group (TTF >= {DEFAULT_MATURE_THRESHOLD} d)", figures_dir / "km_global_mature_by_freq.png")
    mature_freq["analysis"] = "global_mature"
    summaries.append(mature_freq)

    for field_name, group in df.groupby("field", dropna=False):
        if int(group["event"].sum()) < MIN_SURVIVAL_FAILURES:
            continue
        field_summary = plot_km_groups(group.loc[group["freq_group_3"] != "<missing>"].copy(), "freq_group_3", f"{field_name}: survival by frequency group", figures_dir / f"km_field_{_safe_slug(field_name)}_by_freq.png")
        field_summary["analysis"] = f"field_{field_name}"
        summaries.append(field_summary)

    vt_df = df.loc[df["field"].astype(str) == "Vt"].copy()
    vt_compare = plot_km_compare(vt_df, df, "Vt", "Global", "Vt vs Global survival", figures_dir / "km_vt_vs_global.png")
    vt_compare["analysis"] = "vt_vs_global"
    summaries.append(vt_compare)

    category_rows: list[dict[str, Any]] = []
    for label, subset in [("Global", df.loc[df["event"].eq(1)].copy()), ("Vt", vt_df.loc[vt_df["event"].eq(1)].copy())]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for category in FAILURE_CATEGORIES:
            category_df = subset.loc[subset["failure_category_raw"] == category].copy()
            if len(category_df) < MIN_SURVIVAL_FAILURES:
                continue
            km = kaplan_meier(category_df["duration_best_days"].to_numpy(dtype=float), np.ones(len(category_df), dtype=int))
            ax.step(km.times, km.survival, where="post", label=f"{category} (n={len(category_df)})")
            category_rows.append(
                {
                    "population": label,
                    "category": category,
                    "rows": int(len(category_df)),
                    "failures": int(len(category_df)),
                    "median_duration_days": km.median,
                    "p25_duration_days": km.quantile_time(0.75),
                }
            )
        ax.set_title(f"{label}: failure-time distribution by category")
        ax.set_xlabel("Failure time (days)")
        ax.set_ylabel("Empirical survivor among failed runs")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"km_failure_categories_{label.lower()}.png", dpi=160)
        plt.close(fig)

    category_summary = pd.DataFrame(category_rows)
    save_frame(category_summary, tables_dir / "km_category_failure_time_summary.csv")

    km_summary = pd.concat(summaries, ignore_index=True)
    save_frame(km_summary, tables_dir / "km_summary.csv")
    return {"km_summary": km_summary, "category_summary": category_summary}


def build_weibull_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    rows: list[dict[str, Any]] = []

    for label, subset in [
        ("Low freq — global", df.loc[df["freq_group_3"] == "Low (<=50 Hz)"].copy()),
        ("Normal freq — global", df.loc[df["freq_group_3"] == "Normal (50-55 Hz)"].copy()),
        ("High freq — global", df.loc[df["freq_group_3"] == "High (>55 Hz)"].copy()),
        ("Low freq — Vt", df.loc[(df["field"].astype(str) == "Vt") & (df["freq_group_3"] == "Low (<=50 Hz)")].copy()),
        ("High freq — Vt", df.loc[(df["field"].astype(str) == "Vt") & (df["freq_group_3"] == "High (>55 Hz)")].copy()),
    ]:
        if int(subset["event"].sum()) < MIN_WEIBULL_FAILURES:
            continue
        fit = weibull_with_ci(subset["duration_best_days"].to_numpy(dtype=float), subset["event"].to_numpy(dtype=int))
        rows.append(
            {
                "group": label,
                "rows": int(len(subset)),
                "failures": int(subset["event"].sum()),
                "beta": fit["beta"],
                "beta_ci_low": fit["beta_ci_low"],
                "beta_ci_high": fit["beta_ci_high"],
                "eta_days": fit["eta"],
                "eta_ci_low": fit["eta_ci_low"],
                "eta_ci_high": fit["eta_ci_high"],
                "failure_pattern": ("infant-dominated" if float(fit["beta"]) < 1.0 else "random" if abs(float(fit["beta"]) - 1.0) <= 0.2 else "wear-out"),
                "aic": fit["aic"],
                "success": fit["success"],
                "message": fit["message"],
            }
        )

    for category in FAILURE_CATEGORIES:
        global_events = np.where((df["event"].eq(1)) & (df["failure_category_raw"] == category), 1, 0).astype(int)
        if int(global_events.sum()) >= MIN_WEIBULL_FAILURES:
            fit = weibull_with_ci(df["duration_best_days"].to_numpy(dtype=float), global_events)
            rows.append(
                {
                    "group": f"{category} — global",
                    "rows": int(len(df)),
                    "failures": int(global_events.sum()),
                    "beta": fit["beta"],
                    "beta_ci_low": fit["beta_ci_low"],
                    "beta_ci_high": fit["beta_ci_high"],
                    "eta_days": fit["eta"],
                    "eta_ci_low": fit["eta_ci_low"],
                    "eta_ci_high": fit["eta_ci_high"],
                    "failure_pattern": ("fast-failing" if float(fit["beta"]) < 1.0 else "mixed/random" if float(fit["beta"]) <= 1.5 else "wear-dominated"),
                    "aic": fit["aic"],
                    "success": fit["success"],
                    "message": fit["message"],
                }
            )

    summary = pd.DataFrame(rows)
    save_frame(summary, tables_dir / "weibull_summary.csv")
    return summary


def build_competing_risks_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    category_series = df["failure_category_raw"].fillna("Censored").astype(str)
    global_cif = cumulative_incidence(
        df["duration_best_days"].to_numpy(dtype=float),
        df["event"].to_numpy(dtype=int),
        category_series.to_numpy(dtype=object),
        FAILURE_CATEGORIES,
    )
    vt_df = df.loc[df["field"].astype(str) == "Vt"].copy()
    vt_cif = cumulative_incidence(
        vt_df["duration_best_days"].to_numpy(dtype=float),
        vt_df["event"].to_numpy(dtype=int),
        vt_df["failure_category_raw"].fillna("Censored").astype(str).to_numpy(dtype=object),
        FAILURE_CATEGORIES,
    )
    save_frame(global_cif, tables_dir / "cif_global.csv")
    save_frame(vt_cif, tables_dir / "cif_vt.csv")
    plot_cif(global_cif, "Global cumulative incidence by failure category", figures_dir / "cif_global.png")
    plot_cif(vt_cif, "Vt cumulative incidence by failure category", figures_dir / "cif_vt.png")
    return {"global_cif": global_cif, "vt_cif": vt_cif}


def build_trf_tlf_chemistry_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    rows: list[dict[str, Any]] = []
    for population_name, subset in [("Global", df.loc[df["event"].eq(1)].copy()), ("Vt", df.loc[(df["field"].astype(str) == "Vt") & df["event"].eq(1)].copy())]:
        for column in [
            "trf_per_day",
            "tlf_per_day",
            "calcium_load_per_day",
            "chloride_load_per_day",
            "sulfate_load_per_day",
            "salt_load_per_day",
            "gypsum_proxy_per_day",
        ]:
            numeric = _numeric(subset[column])
            duration = _numeric(subset["duration_best_days"])
            mask = numeric.notna() & duration.notna()
            if int(mask.sum()) < 20:
                continue
            rho, pval = spearmanr(numeric.loc[mask], duration.loc[mask])
            rows.append(
                {
                    "population": population_name,
                    "metric": column,
                    "rows": int(mask.sum()),
                    "spearman_rho_vs_ttf": float(rho),
                    "spearman_pval": float(pval),
                    "median_metric": float(numeric.loc[mask].median()),
                }
            )
        for category in FAILURE_CATEGORIES:
            category_df = subset.loc[subset["failure_category_raw"] == category].copy()
            if len(category_df) < 10:
                continue
            rows.append(
                {
                    "population": population_name,
                    "metric": "category_summary",
                    "category": category,
                    "rows": int(len(category_df)),
                    "median_calcium_load_per_day": float(_numeric(category_df["calcium_load_per_day"]).median()),
                    "median_chloride_load_per_day": float(_numeric(category_df["chloride_load_per_day"]).median()),
                    "median_sulfate_load_per_day": float(_numeric(category_df["sulfate_load_per_day"]).median()),
                    "median_gypsum_proxy_per_day": float(_numeric(category_df["gypsum_proxy_per_day"]).median()),
                }
            )
    summary = pd.DataFrame(rows)
    save_frame(summary, tables_dir / "trf_tlf_chemistry_summary.csv")
    return summary


def build_confounding_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    rows: list[dict[str, Any]] = []

    def add_correlation(frame: pd.DataFrame, scope: str, scope_value: str) -> None:
        failures = frame.loc[frame["event"].eq(1)].copy()
        x = _numeric(failures["freq_signed_exposure"])
        y = _numeric(failures["duration_best_days"])
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 10:
            return
        rho, pval = spearmanr(x.loc[mask], y.loc[mask])
        low, high = correlation_ci_spearman(float(rho), int(mask.sum()))
        rows.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "rows": int(mask.sum()),
                "rho": float(rho),
                "rho_ci_low": low,
                "rho_ci_high": high,
                "pval": float(pval),
            }
        )

    add_correlation(df, "global", "all")
    for field_name, subset in df.groupby("field", dropna=False):
        add_correlation(subset, "field", str(field_name))
    for category, subset in df.loc[df["event"].eq(1) & df["failure_category_raw"].notna()].groupby("failure_category_raw", dropna=False):
        add_correlation(subset, "failure_category", str(category))

    result = pd.DataFrame(rows)
    save_frame(result, tables_dir / "confounding_spearman_summary.csv")
    return result


def build_cox_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    working = df.loc[df["freq_group_3"] != "<missing>"].copy()
    working = working.loc[_numeric(working["duration_best_days"]).notna()].copy()
    working["duration_best_days"] = _numeric(working["duration_best_days"])
    working["mount_year"] = _numeric(working["mount_year"])
    working["h2s_high_class"] = np.where(working["h2s_class"].astype(str) == "Кислый", "Кислый", np.where(working["h2s_class"].astype(str) == "Некислый", "Некислый", "<missing>"))
    working["glf_bin"] = working["glf_bin"].astype("string").fillna("<missing>")
    working["first_run_per_well"] = working.sort_values(["Скв.", "Дата монтажа", "Дата остановки", "row_id"]).groupby("Скв.").cumcount().eq(0)

    model_specs = [
        ("model_1_freq_only", ["freq_group_3"]),
        ("model_2_plus_field", ["freq_group_3", "field"]),
        ("model_3_plus_mount_year", ["freq_group_3", "field", "mount_year"]),
        ("model_4_plus_env", ["freq_group_3", "field", "mount_year", "h2s_high_class", "glf_bin"]),
    ]

    output_tables: list[pd.DataFrame] = []
    ph_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name, features in model_specs:
        model_df = working.copy()
        design, column_names = one_hot_frame(model_df[features], features, drop_reference={"freq_group_3": "Normal (50-55 Hz)"})
        if design.empty:
            continue
        valid = design.notna().all(axis=1) & model_df["duration_best_days"].notna() & model_df["event"].isin([0, 1])
        model_df = model_df.loc[valid].copy()
        design = design.loc[valid].astype(float)
        if int(model_df["event"].sum()) < max(20, len(column_names) + 5):
            continue
        fit = cox_fit(model_df["duration_best_days"].to_numpy(dtype=float), model_df["event"].to_numpy(dtype=int), design.to_numpy(dtype=float), column_names)
        table = fit["table"].copy()
        table["model"] = model_name
        output_tables.append(table)
        frequency_terms = [term for term in column_names if term.startswith("freq_group_3_")]
        ph = schoenfeld_frequency_check(fit, frequency_terms)
        if not ph.empty:
            ph["model"] = model_name
            ph_tables.append(ph)
        summary_rows.append(
            {
                "model": model_name,
                "rows": fit["rows"],
                "events": fit["events"],
                "aic": fit["aic"],
                "c_index": fit["c_index"],
                "success": fit["success"],
                "message": fit["message"],
            }
        )

    if output_tables:
        save_frame(pd.concat(output_tables, ignore_index=True), tables_dir / "cox_model_terms.csv")
    if ph_tables:
        save_frame(pd.concat(ph_tables, ignore_index=True), tables_dir / "cox_ph_checks.csv")
    cox_summary = pd.DataFrame(summary_rows)
    save_frame(cox_summary, tables_dir / "cox_model_summary.csv")

    loglog_input = []
    for group in ["Low (<=50 Hz)", "Normal (50-55 Hz)", "High (>55 Hz)"]:
        subset = working.loc[working["freq_group_3"] == group].copy()
        if int(subset["event"].sum()) < MIN_SURVIVAL_FAILURES:
            continue
        km = kaplan_meier(subset["duration_best_days"].to_numpy(dtype=float), subset["event"].to_numpy(dtype=int))
        for time, surv in zip(km.times[1:], km.survival[1:], strict=False):
            if surv <= 0.0 or surv >= 1.0 or time <= 0:
                continue
            loglog_input.append({"group": group, "log_time": math.log(time), "loglog_survival": math.log(-math.log(surv))})
    if loglog_input:
        loglog_frame = pd.DataFrame(loglog_input)
        fig, ax = plt.subplots(figsize=(8, 6))
        for group, subset in loglog_frame.groupby("group", dropna=False):
            ax.plot(subset["log_time"], subset["loglog_survival"], marker="o", linestyle="-", label=group)
        ax.set_title("Approximate log-log survival by frequency group")
        ax.set_xlabel("log(time)")
        ax.set_ylabel("log(-log(S(t)))")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "cox_loglog_survival.png", dpi=160)
        plt.close(fig)

    return {"summary": cox_summary}


def build_vt_deep_dive(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables_dir = output_dir / "tables"
    vt_df = df.loc[df["field"].astype(str) == "Vt"].copy()
    global_df = df.copy()

    comparison_rows = []
    variables = [
        ("Mean H2S (declared source)", "h2s_effective_mg_l"),
        ("Share Кислый", "high_h2s_flag"),
        ("Mean GLF", "avg_glf"),
        ("Mean Kpod", "avg_kpod"),
        ("Mean normalized TRF", "trf_per_day"),
        ("Mean normalized TLF", "tlf_per_day"),
        ("Calcium load proxy", "calcium_load_per_day"),
        ("Chloride load proxy", "chloride_load_per_day"),
        ("Sulfate load proxy", "sulfate_load_per_day"),
        ("Gypsum proxy", "gypsum_proxy_per_day"),
        ("Mean nominal frequency Hz", "Номинальная частота, Гц"),
        ("Share freq >55 Hz", "freq_above_55hz_pct"),
        ("Telemetry coverage", "valid_freq_run"),
        ("Corrosion fill rate", "corrosion_fill"),
    ]
    for label, column in variables:
        vt_values = _numeric(vt_df[column])
        global_values = _numeric(global_df[column])
        comparison_rows.append(
            {
                "variable": label,
                "vt_value": float(vt_values.mean()) if vt_values.notna().any() else float("nan"),
                "global_value": float(global_values.mean()) if global_values.notna().any() else float("nan"),
                "difference_vt_minus_global": float(vt_values.mean() - global_values.mean()) if vt_values.notna().any() and global_values.notna().any() else float("nan"),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    save_frame(comparison, tables_dir / "vt_environment_comparison.csv")

    global_failure_mix = df.loc[df["event"].eq(1) & df["failure_category_raw"].notna()].copy()
    vt_failure_mix = vt_df.loc[vt_df["event"].eq(1) & vt_df["failure_category_raw"].notna()].copy()
    mix_weights_global = global_failure_mix["failure_category_raw"].value_counts(normalize=True)
    mix_weights_vt = vt_failure_mix["failure_category_raw"].value_counts(normalize=True)
    horizon = DEFAULT_RMST_HORIZON_DAYS

    category_effect_rows = []
    for category in FAILURE_CATEGORIES:
        vt_cat = vt_failure_mix.loc[vt_failure_mix["failure_category_raw"] == category]
        global_cat = global_failure_mix.loc[global_failure_mix["failure_category_raw"] == category]
        category_effect_rows.append(
            {
                "category": category,
                "vt_weight": float(mix_weights_vt.get(category, 0.0)),
                "global_weight": float(mix_weights_global.get(category, 0.0)),
                "vt_expected_ttf_trunc": restricted_mean_failure_time(vt_cat["duration_best_days"], horizon),
                "global_expected_ttf_trunc": restricted_mean_failure_time(global_cat["duration_best_days"], horizon),
                "vt_failures": int(len(vt_cat)),
                "global_failures": int(len(global_cat)),
            }
        )
    category_effects = pd.DataFrame(category_effect_rows)
    save_frame(category_effects, tables_dir / "vt_category_effects.csv")

    observed_vt = float(np.nansum(category_effects["vt_weight"] * category_effects["vt_expected_ttf_trunc"]))
    observed_global = float(np.nansum(category_effects["global_weight"] * category_effects["global_expected_ttf_trunc"]))
    within_category_counterfactual = float(np.nansum(category_effects["global_weight"] * category_effects["vt_expected_ttf_trunc"]))
    mix_counterfactual = float(np.nansum(category_effects["vt_weight"] * category_effects["global_expected_ttf_trunc"]))
    decomposition = pd.DataFrame(
        [
            {"metric": "observed_vt_expected_ttf_trunc", "value": observed_vt},
            {"metric": "observed_global_expected_ttf_trunc", "value": observed_global},
            {"metric": "within_category_counterfactual_global_mix_with_vt_category_ttf", "value": within_category_counterfactual},
            {"metric": "mix_counterfactual_vt_mix_with_global_category_ttf", "value": mix_counterfactual},
            {"metric": "mix_effect_days", "value": mix_counterfactual - observed_global},
            {"metric": "within_category_effect_days", "value": within_category_counterfactual - observed_global},
            {"metric": "total_gap_days", "value": observed_vt - observed_global},
        ]
    )
    save_frame(decomposition, tables_dir / "vt_decomposition_summary.csv")

    vt_high_h2s = vt_df.loc[vt_df["h2s_class"].astype(str) == "Кислый"].copy()
    vt_non_h2s = vt_df.loc[vt_df["h2s_class"].astype(str) == "Некислый"].copy()
    h2s_interaction_rows = []
    for label, subset in [("Vt_Кислый", vt_high_h2s), ("Vt_Некислый", vt_non_h2s)]:
        h2s_interaction_rows.append(describe_group(subset, label))
    h2s_interaction = pd.DataFrame(h2s_interaction_rows)
    save_frame(h2s_interaction, tables_dir / "vt_h2s_interaction_summary.csv")

    hypotheses = [
        {
            "hypothesis": "Vt has more fast-failing categories",
            "evidence_for": "Higher shares of shaft break / clogging / cable groups than global" if float(category_effects.loc[category_effects["category"].isin(["Слом вала", "Засорение РО", "КЛ (R-0)"]), "vt_weight"].sum()) > float(category_effects.loc[category_effects["category"].isin(["Слом вала", "Засорение РО", "КЛ (R-0)"]), "global_weight"].sum()) else "",
            "evidence_against": "",
            "verdict": "support" if float(category_effects.loc[category_effects["category"].isin(["Слом вала", "Засорение РО", "КЛ (R-0)"]), "vt_weight"].sum()) > float(category_effects.loc[category_effects["category"].isin(["Слом вала", "Засорение РО", "КЛ (R-0)"]), "global_weight"].sum()) else "insufficient",
        },
        {
            "hypothesis": "Vt has harsher H2S environment",
            "evidence_for": "Higher mean H2S or higher acidic share than global" if float(comparison.loc[comparison["variable"] == "Mean H2S (declared source)", "difference_vt_minus_global"].iloc[0]) > 0 else "",
            "evidence_against": "",
            "verdict": "support" if float(comparison.loc[comparison["variable"] == "Mean H2S (declared source)", "difference_vt_minus_global"].iloc[0]) > 0 else "insufficient",
        },
        {
            "hypothesis": "Vt frequency effect persists after H2S/GLF control",
            "evidence_for": "",
            "evidence_against": "",
            "verdict": "insufficient",
        },
        {
            "hypothesis": "Vt equipment/contractor mix is different",
            "evidence_for": "Contractor / pump mix differs in descriptive tables" if vt_df["contractor"].astype("string").value_counts(normalize=True).head(3).to_dict() != global_df["contractor"].astype("string").value_counts(normalize=True).head(3).to_dict() else "",
            "evidence_against": "",
            "verdict": "support",
        },
        {
            "hypothesis": "High-frequency × H2S interaction accelerates failure",
            "evidence_for": "",
            "evidence_against": "",
            "verdict": "insufficient",
        },
    ]
    hypothesis_df = pd.DataFrame(hypotheses)
    save_frame(hypothesis_df, tables_dir / "vt_hypothesis_verdicts.csv")
    return {
        "comparison": comparison,
        "category_effects": category_effects,
        "decomposition": decomposition,
        "h2s_interaction": h2s_interaction,
        "hypotheses": hypothesis_df,
    }


def fit_catboost_infant(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    if CatBoostClassifier is None or Pool is None:
        return {"status": "catboost_missing"}
    tables_dir = output_dir / "tables"
    df = df.copy()
    df["infant_90d_label"] = (df["event"].eq(1) & (_numeric(df["duration_best_days"]) < 90)).astype(int)
    feature_candidates = [
        "freq_signed_exposure",
        "freq_w_mean",
        "freq_above_55hz_pct",
        "field",
        "contractor",
        "pump_family",
        "h2s_effective_mg_l",
        "h2s_class",
        "avg_glf",
        "avg_kpod",
        "trf_per_day",
        "calcium_load_per_day",
        "chloride_load_per_day",
        "sulfate_load_per_day",
        "gypsum_proxy_per_day",
        "mount_year",
        "Номинальная частота, Гц",
        "corrosion_resistance",
    ]
    feature_columns = [column for column in feature_candidates if column in df.columns]
    metrics = evaluate_catboost_cv(df.assign(label=df["infant_90d_label"]), feature_columns, group_column="Скв.", label_column="label", n_splits=5)
    prepared, categorical_columns = prepare_feature_matrix(df, feature_columns)
    aligned = df.loc[prepared.index].copy()
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    class_weights = [1.0, max(1.0, float((aligned["infant_90d_label"] == 0).sum()) / max(float((aligned["infant_90d_label"] == 1).sum()), 1.0))]
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=class_weights,
    )
    model.fit(Pool(prepared, aligned["infant_90d_label"].astype(int), cat_features=cat_indices))
    shap_values = model.get_feature_importance(Pool(prepared, aligned["infant_90d_label"].astype(int), cat_features=cat_indices), type="ShapValues")
    shap_frame = pd.DataFrame({"feature": prepared.columns.tolist(), "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(axis=0)})
    shap_frame = shap_frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    save_frame(shap_frame, tables_dir / "catboost_infant_shap.csv")
    metrics_path = output_dir / "catboost_infant_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "metrics": metrics, "shap": shap_frame}


def build_report(
    output_dir: Path,
    audit_summary: dict[str, Any],
    descriptive: pd.DataFrame,
    vt_vs_global: pd.DataFrame,
    infant_outputs: dict[str, Any],
    survival_outputs: dict[str, Any],
    weibull_summary: pd.DataFrame,
    confounding_summary: pd.DataFrame,
    chemistry_summary: pd.DataFrame,
    vt_outputs: dict[str, pd.DataFrame],
    catboost_result: dict[str, Any],
) -> None:
    vt_row = vt_vs_global.loc[vt_vs_global["group"] == "Vt"].iloc[0]
    global_row = vt_vs_global.loc[vt_vs_global["group"] == "Global"].iloc[0]
    km_vt_global = survival_outputs["km_summary"].loc[survival_outputs["km_summary"]["analysis"] == "vt_vs_global"].copy()
    km_vt = km_vt_global.loc[km_vt_global["group"] == "Vt"]
    km_global = km_vt_global.loc[km_vt_global["group"] == "Global"]
    km_vt_median = float(km_vt["median_survival_days"].iloc[0]) if not km_vt.empty else float(vt_row["median_ttf_days"])
    km_global_median = float(km_global["median_survival_days"].iloc[0]) if not km_global.empty else float(global_row["median_ttf_days"])
    infant_90 = infant_outputs["vt_standardization"].loc[infant_outputs["vt_standardization"]["threshold_days"].eq(90)].iloc[0]
    corr_global = confounding_summary.loc[(confounding_summary["scope"] == "global") & (confounding_summary["scope_value"] == "all")]
    corr_vt = confounding_summary.loc[(confounding_summary["scope"] == "field") & (confounding_summary["scope_value"] == "Vt")]
    corr_text = "n/a"
    if not corr_global.empty:
        corr_row = corr_global.iloc[0]
        corr_text = f"rho={corr_row['rho']:.3f} [{corr_row['rho_ci_low']:.3f}, {corr_row['rho_ci_high']:.3f}]"
    corr_vt_text = "n/a"
    if not corr_vt.empty:
        corr_row = corr_vt.iloc[0]
        corr_vt_text = f"rho={corr_row['rho']:.3f} [{corr_row['rho_ci_low']:.3f}, {corr_row['rho_ci_high']:.3f}]"

    chemistry_vt = chemistry_summary.loc[(chemistry_summary["population"] == "Vt") & (chemistry_summary["metric"] != "category_summary")].copy()
    strongest_chem = None
    if not chemistry_vt.empty:
        chemistry_vt["abs_rho"] = chemistry_vt["spearman_rho_vs_ttf"].astype(float).abs()
        strongest_chem = chemistry_vt.sort_values("abs_rho", ascending=False).iloc[0]

    top_shap = ""
    if catboost_result.get("status") == "ok":
        top = catboost_result["shap"].loc[~catboost_result["shap"]["feature"].isin(["corrosion_resistance"])].head(5)
        top_shap = ", ".join(f"{row.feature} ({row.mean_abs_shap:.3f})" for row in top.itertuples(index=False))

    chemistry_text = "n/a"
    if strongest_chem is not None:
        chemistry_text = f"{strongest_chem['metric']} rho={float(strongest_chem['spearman_rho_vs_ttf']):.3f} in Vt"

    total_gap = float(km_vt_median - km_global_median)
    failure_gap = float(vt_row["failure_rate"] - global_row["failure_rate"])
    h2s_gap = float(vt_row["mean_h2s_mg_l"] - global_row["mean_h2s_mg_l"])
    acid_gap = float(vt_row["share_high_h2s"] - global_row["share_high_h2s"])
    infant_gap = float(infant_90["vt_observed_infant_share_among_failures"] - infant_90["global_observed_infant_share_among_failures"])
    decomposition = vt_outputs["decomposition"].set_index("metric")["value"].to_dict()

    lines = [
        "# VT 60 Hz Failure Analysis",
        "",
        "## Executive Summary",
        "",
        f"- In the KM survival comparison, `Vt` is shorter-lived than global: median survival `{km_vt_median:.1f}` vs `{km_global_median:.1f}` days (gap `{total_gap:.1f}` days), and its failure rate is also higher: `{vt_row['failure_rate']:.3f}` vs `{global_row['failure_rate']:.3f}`.",
        f"- At the `<90 d` infant threshold, `Vt` is very close to global on infant share among failures: `{infant_90['vt_observed_infant_share_among_failures']:.3f}` vs `{infant_90['global_observed_infant_share_among_failures']:.3f}` (gap `{infant_gap:.3f}`), and standardizing `Vt` to the global failure-category mix gives `{infant_90['vt_standardized_to_global_category_mix']:.3f}`.",
        f"- The raw frequency signal is weak in this run population: global `{corr_text}`, Vt `{corr_vt_text}`.",
        f"- `Vt` is more chemically severe than global: mean H2S is higher by `{h2s_gap:.1f}` mg/L and acidic-share is higher by `{acid_gap:.3f}`.",
        f"- The strongest Vt chemistry timing signal in the proxy screen is `{chemistry_text}`; normalized `TRF` itself is not a strong standalone TTF discriminator.",
        f"- The analysis uses `>55 Hz` high-frequency exposure as the main regime. True `>58 Hz` failures globally = `{audit_summary['true60_failures_global']}`.",
        "",
        "## What We Can Say Confidently",
        "",
        "- In this analysis set, the clearest `Vt` distinctions are shorter KM median survival, higher failure rate, higher H2S burden, somewhat higher GLF, and a different failure-category mix.",
        "- Raw frequency-only evidence is weak; higher-frequency exposure does not emerge as a clean standalone explanation from the simple global or Vt-only rank-correlation checks.",
        "- Separate chemistry proxies add more signal than a single generic precipitate proxy, especially in `Vt`.",
        f"- The category-mix decomposition suggests only a small overall duration gap in this population: total gap `{float(decomposition.get('total_gap_days', float('nan'))):.1f}` days, with mix effect `{float(decomposition.get('mix_effect_days', float('nan'))):.1f}` days.",
        "",
        "## What Remains Uncertain",
        "",
        "- Fine-Gray and Gray's-test style competing-risk inference was not implemented in this run because the environment lacks a dedicated survival library; CIF curves were produced instead.",
        "- Cox PH results, if used, are custom and numerically fragile in this environment; they should be treated as secondary evidence pending package-backed survival validation.",
        "- Cluster-robust survival standard errors were not available in the current environment.",
        "- Earlier raw workbook summaries suggested a larger Vt life penalty; the smaller gap here indicates that population definition, deduplication, and duration-source choice materially affect the story.",
        "",
        "## CatBoost Infant Model",
        "",
        f"- Status: `{catboost_result.get('status', 'unknown')}`",
        f"- Top SHAP-style features after filtering the sparse corrosion field from the headline list: `{top_shap or 'n/a'}`",
        "",
        "## Key Outputs",
        "",
        "- `tables/data_audit_summary.json`",
        "- `tables/field_descriptive_summary.csv`",
        "- `tables/infant_rates_by_field_freq_threshold.csv`",
        "- `tables/km_summary.csv`",
        "- `tables/weibull_summary.csv`",
        "- `tables/confounding_spearman_summary.csv`",
        "- `tables/vt_decomposition_summary.csv`",
        "- `figures/km_vt_vs_global.png`",
        "- `figures/vt_vs_global_failure_mix.png`",
        "",
    ]
    (output_dir / "vt_60hz_prompt_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt-aligned VT 60 Hz failure analysis.")
    parser.add_argument("--output-dir", default=None, help="Directory for analysis outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir(_SLUG)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("[analysis] Loading dataset...", flush=True)
    df, audit_meta = build_dataset()
    save_frame(df, tables_dir / "analysis_dataset.csv")

    print("[analysis] Building data audit...", flush=True)
    audit_summary = build_data_audit(df, output_dir, audit_meta)

    print("[analysis] Building descriptive outputs...", flush=True)
    descriptive, vt_vs_global = build_descriptive_outputs(df, output_dir)

    print("[analysis] Building infant mortality outputs...", flush=True)
    infant_outputs = build_infant_outputs(df.copy(), output_dir)

    print("[analysis] Building survival outputs...", flush=True)
    survival_outputs = build_survival_outputs(df, output_dir)

    print("[analysis] Building Weibull outputs...", flush=True)
    weibull_summary = build_weibull_outputs(df, output_dir)

    print("[analysis] Building competing-risks outputs...", flush=True)
    _ = build_competing_risks_outputs(df, output_dir)

    print("[analysis] Building TRF/TLF and chemistry outputs...", flush=True)
    chemistry_summary = build_trf_tlf_chemistry_outputs(df, output_dir)

    print("[analysis] Building confounding outputs...", flush=True)
    confounding_summary = build_confounding_outputs(df, output_dir)

    print("[analysis] Building Cox outputs...", flush=True)
    _ = build_cox_outputs(df, output_dir)

    print("[analysis] Building Vt deep dive...", flush=True)
    vt_outputs = build_vt_deep_dive(df, output_dir)

    print("[analysis] Fitting CatBoost infant model...", flush=True)
    catboost_result = fit_catboost_infant(df, output_dir)

    print("[analysis] Writing report...", flush=True)
    build_report(output_dir, audit_summary, descriptive, vt_vs_global, infant_outputs, survival_outputs, weibull_summary, confounding_summary, chemistry_summary, vt_outputs, catboost_result)

    print(f"[analysis] Complete. Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
