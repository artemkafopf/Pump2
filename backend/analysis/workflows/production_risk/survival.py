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
    water_rate_m3d: np.ndarray | None = None
    gtm_dates: tuple[pd.Timestamp, ...] = ()


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


def ql_hazard_theta(log_ql: float, model_field: str | None, ages: np.ndarray | float) -> np.ndarray | float:
    """Monthly Ql (IOR) hazard multiplier — stress-scenario sensitivity dial.

    ``log_ql`` is log1p(monthly liquid rate, m3/d).  The covariate is field-centered and
    capped at +/- log(5).  The multiplier is ``exp(z * (beta + gamma * log(max(age,1))))``;
    with the sign-corrected config (``beta=+0.07``, ``gamma=0``) this is a plain
    proportional hazard: higher-than-reference liquid rate raises the hazard at every age
    (physically forward), capped at ~+12% at 5x field Ql.  See ``config`` for the sign
    history and the reverse-causation caveat.
    """
    if not C.QL_HAZARD_ENABLED or not np.isfinite(float(log_ql)):
        return np.ones_like(ages, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    age_arr = np.asarray(ages, dtype=float)
    ref = C.QL_HAZARD_FIELD_REF_LOG.get(str(model_field or ""), C.QL_HAZARD_GLOBAL_REF_LOG)
    z = np.clip(float(log_ql) - float(ref), -C.QL_HAZARD_CAP_LOG_RATIO, C.QL_HAZARD_CAP_LOG_RATIO)
    coef = C.QL_HAZARD_BETA + C.QL_HAZARD_GAMMA * np.log(np.maximum(age_arr, 1.0))
    eta = z * coef
    theta = np.exp(np.clip(eta, -8.0, 8.0))
    if isinstance(ages, np.ndarray):
        return theta
    return float(theta)


def kpod_hazard_theta(kpod: float, ages: np.ndarray | float, field: str | None = None) -> np.ndarray | float:
    """Raw Kpod U-shape hazard multiplier, stress-scenario sensitivity dial.

    ``kpod`` is monthly ``Ql / Qnominal`` (not frequency-normalized).  Values inside
    ``[k_lo, k_hi]`` are neutral.  The underload and overload sides have separate
    non-negative beta terms plus optional age interactions, with side-specific caps.
    """
    if not C.KPOD_HAZARD_ENABLED or not np.isfinite(float(kpod)):
        return np.ones_like(ages, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    age_arr = np.asarray(ages, dtype=float)
    p = C.kpod_hazard_params(field)
    k = float(kpod)
    s_under = min(max(float(p["k_lo"]) - k, 0.0), float(p["cap_under"]))
    s_over = min(max(k - float(p["k_hi"]), 0.0), float(p["cap_over"]))
    if s_under <= 0.0 and s_over <= 0.0:
        return np.ones_like(age_arr, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    log_age = np.log(np.maximum(age_arr, 1.0))
    coef_u = float(p["beta_under"]) + float(p["gamma_under"]) * log_age
    coef_o = float(p["beta_over"]) + float(p["gamma_over"]) * log_age
    eta = s_under * coef_u + s_over * coef_o
    theta = np.exp(np.clip(eta, -8.0, 8.0))
    if isinstance(ages, np.ndarray):
        return theta
    return float(theta)


def infant_hazard_theta(
    ages: np.ndarray | float,
    field: str | None = None,
) -> np.ndarray | float:
    """Measured infant-mortality hazard multiplier: ``1 + u0*exp(-age/tau)``, capped.

    Supplies the early-life hazard the shipped single Weibull structurally cannot
    hold (it delivers ~65% of the real day-8 hazard in every stratum while being
    calibrated past ~45 d).  Applies to EVERY pump at every age — a renewed pump is
    a new pump — and in every scenario, because this is measured physics, not a
    stress dial.  See ``config.INFANT_HAZARD_ENABLED``.
    """
    if not C.INFANT_HAZARD_ENABLED:
        return np.ones_like(ages, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    p = C.infant_hazard_params(field)
    u0 = max(0.0, float(p["u0"]))
    tau = float(p["tau_days"])
    cap = max(1.0, float(p["cap"]))
    age_arr = np.asarray(ages, dtype=float)
    if u0 <= 0.0 or tau <= 0.0:
        theta = np.ones_like(age_arr, dtype=float)
    else:
        theta = 1.0 + u0 * np.exp(-np.maximum(age_arr, 0.0) / tau)
    theta = np.clip(theta, 1.0, cap)
    return theta if isinstance(ages, np.ndarray) else float(theta)


def uncertainty_hazard_theta(
    ages: np.ndarray | float,
    state_label: str | None,
    field: str | None = None,
) -> np.ndarray | float:
    """New-launch uncertainty hazard multiplier, stress-scenario sensitivity dial.

    The multiplier is only active for configured future/new-pump state labels.  It
    starts at ``1 + u0`` when operating age is zero and decays toward 1 as the pump
    earns operating days, capped by ``cap``.
    """
    if not C.UNCERTAINTY_HAZARD_ENABLED:
        return np.ones_like(ages, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    if str(state_label or "") not in C.UNCERTAINTY_HAZARD_APPLY_TO:
        return np.ones_like(ages, dtype=float) if isinstance(ages, np.ndarray) else 1.0
    p = C.uncertainty_hazard_params(field)
    u0 = max(0.0, float(p["u0"]))
    tau = max(0.0, float(p["tau_days"]))
    cap = max(1.0, float(p["cap"]))
    age_arr = np.asarray(ages, dtype=float)
    if u0 <= 0.0:
        theta = np.ones_like(age_arr, dtype=float)
    elif tau <= 0.0:
        theta = 1.0 + u0 * (age_arr <= 0.0)
    else:
        theta = 1.0 + u0 * np.exp(-np.maximum(age_arr, 0.0) / tau)
    theta = np.minimum(theta, cap)
    if isinstance(ages, np.ndarray):
        return theta
    return float(theta)


def current_pump_p_fail_monthly(
    age_pmf: dict[int, float],
    params: dict[str, float],
    op_days: np.ndarray,
    cal_days: np.ndarray,
    model: StrataModel,
    *,
    horizon_calendar_days: int,
    log_ql_monthly: np.ndarray | None = None,
    kpod_monthly: np.ndarray | None = None,
    model_field: str | None = None,
    kpod_field: str | None = None,
    uncertainty_state_label: str | None = None,
    uncertainty_field: str | None = None,
) -> float:
    """Failure probability over the first N calendar days, with optional stress dials."""
    horizon_calendar_days = max(0, int(horizon_calendar_days))
    if horizon_calendar_days <= 0:
        return 0.0
    start_ages = {int(age): float(w) for age, w in age_pmf.items() if w > C.MIN_STATE_MASS}
    max_active_days = int(math.ceil(float(np.maximum(op_days, 0.0).sum()))) + 2
    max_age = (max(start_ages) if start_ages else 0) + max_active_days + 2
    q_arr = model.daily_fail_prob_array(params, max_age)
    q_pad = np.zeros(max_age + 2, dtype=float)
    q_pad[: len(q_arr)] = q_arr
    if len(q_arr):
        q_pad[len(q_arr):] = q_arr[-1]

    up = np.zeros(max_age + 2, dtype=float)
    for age, weight in start_ages.items():
        up[age] += weight
    hi = (max(start_ages) + 1) if start_ages else 1
    p_active = np.divide(op_days, cal_days, out=np.zeros_like(op_days, dtype=float), where=cal_days > 0)
    p_active = np.clip(p_active, 0.0, 1.0)
    log_ql = log_ql_monthly if log_ql_monthly is not None else np.full(len(op_days), np.nan)
    kpod = kpod_monthly if kpod_monthly is not None else np.full(len(op_days), np.nan)

    # The infant dial is measured physics, not a stress sensitivity: it is on in
    # every scenario and for every pump, including each renewal (which re-enters the
    # infant window at age 0).
    use_infant = C.INFANT_HAZARD_ENABLED
    use_dials = (
        log_ql_monthly is not None
        or kpod_monthly is not None
        or uncertainty_state_label is not None
        or use_infant
    )
    ages_full = np.arange(max_age + 2, dtype=float) if use_dials else None
    failed = 0.0
    remaining = horizon_calendar_days
    for mi in range(len(cal_days)):
        if remaining <= 0:
            break
        days = min(int(round(float(cal_days[mi]))), remaining)
        p = float(p_active[mi])
        # q_eff is constant within a month (dials depend only on month-constant inputs +
        # age); compute once and slice per day (byte-identical to the per-day recompute).
        if use_dials:
            theta_full = np.ones(max_age + 2, dtype=float)
            if log_ql_monthly is not None:
                theta_full = theta_full * ql_hazard_theta(float(log_ql[mi]), model_field, ages_full)
            if kpod_monthly is not None:
                theta_full = theta_full * kpod_hazard_theta(float(kpod[mi]), ages_full, kpod_field or model_field)
            if use_infant:
                theta_full = theta_full * infant_hazard_theta(ages_full, model_field)
            if uncertainty_state_label is not None:
                theta_full = theta_full * uncertainty_hazard_theta(ages_full, uncertainty_state_label, uncertainty_field or kpod_field or model_field)
            q_eff_full = 1.0 - np.power(1.0 - q_pad, theta_full)
        for _ in range(days):
            remaining -= 1
            if p <= 0.0:
                continue
            q_eff = q_eff_full[:hi] if use_dials else q_pad[:hi]
            seg = up[:hi]
            fail_vec = seg * (p * q_eff)
            fail_day = float(fail_vec.sum())
            moved = seg * p - fail_vec
            seg *= 1.0 - p
            up[1 : hi + 1] += moved
            if hi < max_age + 1:
                hi += 1
            failed += fail_day
    return float(np.clip(failed, 0.0, 1.0))


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
    log_ql_monthly: np.ndarray | None = None,
    kpod_monthly: np.ndarray | None = None,
    model_field: str | None = None,
    kpod_field: str | None = None,
    uncertainty_state_label: str | None = None,
    uncertainty_field: str | None = None,
    gtm_reset_months: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project monthly expected failures and lost op-days for one well.

    ``gtm_reset_months`` maps a month index to the share of age mass a planned ГТМ
    sends back to age 0 (the pump is replaced).  Omitted → the pump renews on
    FAILURE only, which lets it age past the range the survival fit was estimated
    on wherever ГТМ drives most pulls.  See ``config.GTM_AGE_RESET_ENABLED``.
    """
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

    # The infant dial is measured physics, not a stress sensitivity: it is on in
    # every scenario and for every pump, including each renewal (which re-enters the
    # infant window at age 0).
    use_infant = C.INFANT_HAZARD_ENABLED
    use_dials = (
        log_ql_monthly is not None
        or kpod_monthly is not None
        or uncertainty_state_label is not None
        or use_infant
    )
    ages_full = np.arange(max_age + 2, dtype=float) if use_dials else None

    for mi in range(month_count):
        days_in_month = int(round(float(well.cal_days[mi])))
        if days_in_month <= 0:
            continue
        # A planned ГТМ swaps the pump out: `share` of the age mass restarts at 0,
        # mirroring the failure renewal below (which also re-enters at age 0).
        share = float((gtm_reset_months or {}).get(mi, 0.0))
        if share > 0.0:
            moved_mass = float(up[:hi].sum()) * share
            up[:hi] *= 1.0 - share
            up[0] += moved_mass
        p = float(p_active[mi])
        # The dial multipliers depend only on month-constant inputs and age, so q_eff is
        # constant within the month.  Compute it once over the full age range and slice per
        # day (byte-identical to the previous per-day recompute, ~day_count x cheaper).
        if use_dials:
            theta_full = np.ones(max_age + 2, dtype=float)
            if log_ql_monthly is not None:
                theta_full = theta_full * ql_hazard_theta(float(log_ql_monthly[mi]), model_field, ages_full)
            if kpod_monthly is not None:
                theta_full = theta_full * kpod_hazard_theta(float(kpod_monthly[mi]), ages_full, kpod_field or model_field)
            if use_infant:
                theta_full = theta_full * infant_hazard_theta(ages_full, model_field)
            if uncertainty_state_label is not None:
                theta_full = theta_full * uncertainty_hazard_theta(
                    ages_full, uncertainty_state_label, uncertainty_field or kpod_field or model_field
                )
            q_eff_full = 1.0 - np.power(1.0 - q_pad, theta_full)
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
            q_eff = q_eff_full[:hi] if use_dials else q_pad[:hi]
            fail_vec = seg * (p * q_eff)
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
