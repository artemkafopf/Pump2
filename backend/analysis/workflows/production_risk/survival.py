"""Survival math and deterministic daily renewal projection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk.time_map import TimeMap


@dataclass
class WellState:
    code: str
    raw_id: str
    plan_field: str
    model_field: str | None
    stratum: str
    params: dict[str, float]
    age_pmf: dict[int, float]
    age_source: str
    age_mean: float
    state_label: str
    scope_label: str
    confidence: str
    contractor_group: str
    contractor_source: str
    sour_class: str
    sour_source: str
    covariates: dict[str, float]
    cov_source: str
    current_status: str | None
    current_in_operation: bool | None
    current_status_date: str | None
    include_primary: bool
    oil_volume: np.ndarray
    liquid_volume: np.ndarray
    op_days: np.ndarray
    cal_days: np.ndarray
    first_active_month: str | None
    first_gtm_date: pd.Timestamp | None
    event_90d_flag: bool


class StrataModel:
    """Registry-backed baseline survival model with VBA fallback resolution."""

    def __init__(self, registry_path: Path | None = None, bundle_date: str = C.BUNDLE_DATE):
        registry_path = Path(registry_path or C.model_registry_path(bundle_date))
        self.registry_path = registry_path
        df = pd.read_csv(registry_path, encoding="utf-8-sig")
        self.df = df.copy()
        self.by_stratum = df.set_index("stratum").to_dict("index")
        self._q_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.time_map = TimeMap.from_csv(C.time_map_path(bundle_date))

    def resolve(self, field: str | None, sour: str, ctr: str) -> tuple[dict[str, float], str]:
        keys: list[str] = []
        field_s = str(field).strip() if field else ""
        sour_s = str(sour or "nonsour").strip() or "nonsour"
        ctr_s = str(ctr or "Pooled").strip() or "Pooled"
        if field_s:
            keys.append(f"{field_s}_{sour_s}_{ctr_s}")
            keys.append(f"{field_s}_{sour_s}_Pooled")
        keys.append("Global_Pooled")
        for key in keys:
            if key in self.by_stratum:
                row = self.by_stratum[key]
                return {
                    "w1": float(row["w1"]),
                    "beta1": float(row["beta1"]),
                    "eta1": float(row["eta1"]),
                    "beta2": float(row["beta2"]),
                    "eta2": float(row["eta2"]),
                    "b50": float(row["b50"]) if pd.notna(row.get("b50")) else float("nan"),
                    "uptime_factor": float(row.get("uptime_factor", 1.0)) if pd.notna(row.get("uptime_factor", 1.0)) else 1.0,
                }, key
        raise KeyError("Global_Pooled missing from esp_models.csv")

    @staticmethod
    def S(t, params: dict[str, float]):
        t = np.asarray(t, dtype=float)
        s = (
            params["w1"] * np.exp(-((np.maximum(t, 0.0) / params["eta1"]) ** params["beta1"]))
            + (1.0 - params["w1"]) * np.exp(-((np.maximum(t, 0.0) / params["eta2"]) ** params["beta2"]))
        )
        return np.where(t <= 0, 1.0, s)

    def _param_key(self, params: dict[str, float]) -> tuple[float, ...]:
        return (
            round(float(params["w1"]), 10),
            round(float(params["beta1"]), 10),
            round(float(params["eta1"]), 10),
            round(float(params["beta2"]), 10),
            round(float(params["eta2"]), 10),
        )

    def daily_fail_prob_array(self, params: dict[str, float], max_age: int) -> np.ndarray:
        """Daily conditional failure probability q(a) = 1 - S(a+1)/S(a) for a=0..max_age.

        Cached per parameter set (not per max_age) so the many wells sharing a stratum
        reuse one array; the cache entry is extended on demand and sliced on return.
        """
        max_age = max(1, int(max_age))
        key = self._param_key(params)
        cached = self._q_cache.get(key)
        if cached is None or len(cached) < max_age + 1:
            ages = np.arange(0, max_age + 2, dtype=float)
            s0 = self.S(ages[:-1], params)
            s1 = self.S(ages[1:], params)
            out = np.zeros(max_age + 1, dtype=float)
            mask = s0 > 0
            out[mask] = 1.0 - np.clip(s1[mask] / s0[mask], 0.0, 1.0)
            # beyond numerical support of S (s0 == 0) the pump is certain to fail
            out[~mask] = 1.0
            self._q_cache[key] = np.clip(out, 0.0, 1.0)
        return self._q_cache[key][: max_age + 1]


class HazardLayer:
    """Direct stress-only hazard overlay mirroring the workbook math."""

    def __init__(self, coeff_path: Path | None = None, bundle_date: str = C.BUNDLE_DATE):
        coeff_path = Path(coeff_path or C.hazard_coeffs_path(bundle_date))
        self.coeff_path = coeff_path
        if coeff_path.exists():
            df = pd.read_csv(coeff_path, encoding="utf-8-sig")
            df["enabled_flag"] = df["enabled"].astype(str).str.upper() == "TRUE"
            self.coeffs = df[df["enabled_flag"]].copy()
        else:
            self.coeffs = pd.DataFrame(
                columns=["covariate", "beta", "reference_value", "ship_reason", "clock"]
            )

    def theta(self, covariates: dict[str, float]) -> float:
        if self.coeffs.empty or not covariates:
            return 1.0
        total = 0.0
        used = 0
        for _, row in self.coeffs.iterrows():
            cov_name = str(row["covariate"])
            if cov_name not in covariates:
                continue
            value = covariates[cov_name]
            if not np.isfinite(value):
                continue
            beta = float(row["beta"])
            ref = float(row["reference_value"])
            total += beta * (float(value) - ref)
            used += 1
        return float(math.exp(total)) if used else 1.0

    @staticmethod
    def apply_theta(params: dict[str, float], theta: float) -> dict[str, float]:
        if theta <= 0 or abs(theta - 1.0) < 1e-12:
            return dict(params)
        out = dict(params)
        out["eta1"] = float(params["eta1"]) / (theta ** (1.0 / float(params["beta1"])))
        out["eta2"] = float(params["eta2"]) / (theta ** (1.0 / float(params["beta2"])))
        return out


def compress_age_pmf(values: list[float], max_points: int = C.IMPUTED_AGE_PMF_MAX_POINTS) -> dict[int, float]:
    if not values:
        return {0: 1.0}
    arr = np.asarray([float(v) for v in values if np.isfinite(v) and v >= 0], dtype=float)
    if arr.size == 0:
        return {0: 1.0}
    arr.sort()
    if arr.size <= max_points:
        counts = pd.Series(np.round(arr).astype(int)).value_counts().sort_index()
        out = {int(age): float(cnt / arr.size) for age, cnt in counts.items()}
        return out or {0: 1.0}
    chunks = np.array_split(arr, max_points)
    out: dict[int, float] = {}
    total = float(arr.size)
    for chunk in chunks:
        if chunk.size == 0:
            continue
        age = int(round(float(np.median(chunk))))
        out[age] = out.get(age, 0.0) + float(chunk.size / total)
    norm = sum(out.values()) or 1.0
    return {age: weight / norm for age, weight in out.items()}


def weighted_age_mean(age_pmf: dict[int, float]) -> float:
    if not age_pmf:
        return 0.0
    return float(sum(float(age) * float(weight) for age, weight in age_pmf.items()))


def current_pump_p_fail(
    age_pmf: dict[int, float],
    params: dict[str, float],
    op_days: float,
    model: StrataModel,
) -> float:
    op_days = max(0.0, float(op_days))
    out = 0.0
    for age, weight in age_pmf.items():
        s0 = float(model.S(float(age), params))
        if s0 <= 0:
            out += float(weight)
            continue
        out += float(weight) * max(0.0, 1.0 - float(model.S(float(age) + op_days, params)) / s0)
    return float(np.clip(out, 0.0, 1.0))


def scenario_params(
    base_params: dict[str, float],
    covariates: dict[str, float],
    hazard: HazardLayer,
    hazard_mode: str,
) -> tuple[dict[str, float], float]:
    if hazard_mode != "stress":
        return dict(base_params), 1.0
    theta = hazard.theta(covariates)
    return hazard.apply_theta(base_params, theta), theta


def project_well(
    well: WellState,
    params: dict[str, float],
    downtime_days: int,
    model: StrataModel,
) -> tuple[np.ndarray, np.ndarray]:
    month_count = len(well.cal_days)
    failures = np.zeros(month_count, dtype=float)
    lost_op_days = np.zeros(month_count, dtype=float)
    if month_count == 0:
        return failures, lost_op_days

    start_ages = {int(age): float(w) for age, w in well.age_pmf.items() if w > C.MIN_STATE_MASS}
    total_active_days = int(math.ceil(float(np.maximum(well.op_days, 0.0).sum()))) + 2
    max_age = (max(start_ages) if start_ages else 0) + total_active_days + 2
    q_arr = model.daily_fail_prob_array(params, max_age)

    # Dense age-mass vector; identical recurrence to the previous dict-of-masses
    # implementation, ~50x faster.  Each operating day: mass at age a fails with q(a),
    # survivors advance one op-day, idle mass keeps its age; failed mass waits in the
    # downtime queue and re-enters at age 0 (renewal).
    up = np.zeros(max_age + 2, dtype=float)
    for age, w in start_ages.items():
        up[age] += w
    hi = (max(start_ages) + 1) if start_ages else 1  # exclusive upper bound of support
    q_pad = np.zeros(max_age + 2, dtype=float)
    q_pad[: len(q_arr)] = q_arr
    if len(q_arr):
        q_pad[len(q_arr):] = q_arr[-1]

    n_down = max(0, int(downtime_days))
    down = np.zeros(n_down, dtype=float)

    p_active = np.divide(
        well.op_days,
        well.cal_days,
        out=np.zeros_like(well.op_days, dtype=float),
        where=well.cal_days > 0,
    )
    p_active = np.clip(p_active, 0.0, 1.0)

    for mi in range(month_count):
        days_in_month = int(round(float(well.cal_days[mi])))
        if days_in_month <= 0:
            continue
        p = float(p_active[mi])
        for _ in range(days_in_month):
            if n_down:
                up[0] += down[0]
                if n_down > 1:
                    down[:-1] = down[1:]
                down[-1] = 0.0
                lost_op_days[mi] += float(down.sum()) * p

            if p <= 0.0:
                continue  # planned idle day: no ageing, no failures

            seg = up[:hi]
            fail_vec = seg * (p * q_pad[:hi])
            fail_mass_day = float(fail_vec.sum())
            moved = seg * p - fail_vec           # p * (1 - q) survivors advance one day
            seg *= 1.0 - p                        # idle mass keeps its age
            up[1 : hi + 1] += moved
            if hi < max_age + 1:
                hi += 1

            failures[mi] += fail_mass_day
            if n_down:
                down[-1] += fail_mass_day

    return failures, lost_op_days
