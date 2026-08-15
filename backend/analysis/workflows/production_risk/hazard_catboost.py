"""Discrete-time hazard CatBoost — the covariate counterpart to ``esp_hazard_fit``.

Why not TTF regression
----------------------
``models/ml/regression_ttf.py`` predicts TTF quantiles and bridges them to a survival
curve.  Two structural problems fall out of that choice:

* **IPCW zeroes every censored run** (``ipcw_weights`` -> weight 0, and ``fit`` drops
  them), so Mc trains on 43 rows rather than its whole population.
* **It predicts the wrong object.**  The empirical hazard is spike-then-flat with no
  wear-out arm, so TTF is near-exponential: b50/b90 are pure scale and carry no shape to
  learn, and the post-b90 exponential tail then covers ~23% of evaluations.

This module models the **hazard rate directly** on the age axis:

    lambda(age, x) = exp( f(log_age, x) )         [failures per operating day]

fitted by Poisson regression over person-period bins with ``log(exposure)`` as an offset.
That means:

* censored runs contribute their at-risk bins natively — no IPCW, nothing discarded;
* age is a feature, so the **infant spike** is expressed non-parametrically.  That is the
  one form defect the shipped Weibull still has (it delivers 0.78 of the empirical 0-30 d
  hazard) and the one a 5-column parametric schema cannot hold;
* the model predicts the monthly failure probability the failure-rate line actually
  scores, so it plugs into ``failure_rate._append_interval_predictions``'s injected
  ``p_fail_fn`` with no survival-curve bridge.

The covariate-availability trap (read before changing ``attach_covariates``)
---------------------------------------------------------------------------
The covariate frame (``build_regression_frame``) is built on the **warehouse** population
and joins to Свод runs only — **no Big-only run has covariates**.  Big-only runs are what
the 2026-07-17 population fix added, and their hazard meaning is field-specific:

    Ya_brt   covariates missing -> 0.0578 fail/mo vs 0.0288 present  (2.01x HOTTER)
    Mc_brt   covariates missing -> 0.0245 fail/mo vs 0.0480 present  (0.51x COLDER)

Ya's Big-only runs are closed runs carrying failures Свод never recorded; Mc's are open,
still-running pumps.  CatBoost treats NaN as a split direction, so a model that sees the
missing rows learns "missing => hot" on Ya and applies it to an Mc whose missing rows are
cold — the sign inverts.  So fit and evaluate on covariate-present runs only.

**But that restriction is not free, and the cost is not recoverable.**  Availability is
itself strongly confounded with hazard, in opposite directions per field, so the
covariate-bearing subset is not representative of either: on it Mc/Ya = 0.0480/0.0288 =
**1.66**, against the population-level age-standardised SMR of **1.23**.  Selecting on
covariate availability therefore *amplifies* the very gap the covariates are asked to
explain.  There is no subset that is both trap-free and representative — that is a data
limitation, not a modelling choice, and any conclusion drawn here inherits it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import crosswalk

# The four strongest *numeric* covariates from the 2026-07-13 mean-TTF importance audit
# (nominal_freq_hz 18.5, pbubble_atm 17.3, freq_std_early 5.9, log_glf_mean_opdays 5.4).
# ``field`` ranked 5th (4.6) but is deliberately absent: in a single-field fit it is
# constant, and at transfer time the target field is an unseen level — it cannot carry
# signal across fields, only absorb it within one.
COVARIATES: list[str] = [
    "nominal_freq_hz",
    "pbubble_atm",
    "freq_std_early",
    "log_glf_mean_opdays",
]

# **Do not fit a cross-field model on ``COVARIATES`` without running ``support_overlap``.**
# The importance ranking above is measured *within the pooled fleet*, where a covariate
# that merely labels the field scores highly — and the top two do exactly that:
#
#     pbubble_atm      Ya_brt 230.0-259.5 atm   vs  Mc_brt 0-158 atm   -> 0% overlap
#     nominal_freq_hz  Ya_brt 0-220 (non-physical values present)  vs  Mc_brt == 50.0 flat
#
# `pbubble_atm` is a field label in disguise (which is why `field` itself scored only 4.6 —
# pbubble had already absorbed it), and `nominal_freq_hz` has no within-Mc variance at all.
# Fitting on them and predicting Mc extrapolates an exp() link far outside its support: the
# Ya->Mc expected failure count blows up to 181 against 21 observed (ratio 8.6). High
# importance means "separates this fleet's runs", NOT "transports to another field".
TRANSFERABLE_COVARIATES: list[str] = [
    "freq_std_early",
    "log_glf_mean_opdays",
]

# Loading covariates, tested after the first four came back null.  All are computed over
# the **first 30 operating days** (``run_covariates`` aggregates under ``oprank <= n_days``)
# despite `kpod_mean`'s whole-run-sounding name — so none of them leak the outcome.
#
# Kpod is the interesting one: it is dimensionless (``qliq / nominal_flow_m3d``), so unlike
# `pbubble_atm` it is comparable across fields by construction, and all of these clear
# ``support_overlap`` on Ya->Mc.  ``qliq_early_m3d`` is reconstructed as
# ``kpod_mean * nominal_flow_m3d`` (nominal flow is a per-run design constant, so the
# product recovers the early-window mean rate) — the "Ql directly" comparison against the
# normalised form.
LOADING_COVARIATES: list[str] = [
    "kpod_mean",
    "kpod_freq_mean",
    "frac_kpod_below_0p7",
    "qliq_early_m3d",
]

ALL_COVARIATES: list[str] = [*COVARIATES, *LOADING_COVARIATES]

# Registry-imputed columns must be returned to NaN before use: the registry mean-imputes
# well->pad->stratum over the full dataset, so an imputed value silently carries other
# runs' information — and here it would also fake support overlap.
_RESTORE_NAN: tuple[str, ...] = (
    "nominal_freq_hz", "pbubble_atm", "freq_std_early", "glf_mean_opdays",
    "kpod_mean", "kpod_freq_mean", "nominal_flow_m3d",
)

# Age-bin edges in operating days.  Deliberately dense below 30 d: that is where the
# infant spike lives and where the parametric fits fail.  Above ~650 d the hazard is flat
# and events are thin, so the bins widen.
AGE_EDGES: tuple[float, ...] = (
    0.0, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0,
    365.0, 450.0, 550.0, 650.0, 800.0, 1000.0, 1250.0, 1500.0, 2000.0, 3000.0,
)

_JOIN_TOL_DAYS = 7
_LAMBDA_MAX = 1.0        # failures/day — a numerical guard, far above anything physical
_MIN_EXPOSURE = 1e-6


def build_covariate_frame(tte_col: str = "ttf_mix") -> pd.DataFrame:
    """Run-level covariate frame: ``well_key``, ``install_date`` and ``ALL_COVARIATES``.

    Built straight off ``build_competing_risks_df`` rather than ``build_regression_frame``,
    which keeps only its own ``VERIFIED_NUMERICS`` and so drops ``kpod_mean`` and
    ``nominal_flow_m3d``.  Registry imputation is undone (see ``_RESTORE_NAN``).
    """
    from analysis.data.competing_risks_loader import build_competing_risks_df

    df = build_competing_risks_df(tte_col=tte_col).copy()
    for col in _RESTORE_NAN:
        flag = f"{col}_missing"
        if col in df.columns and flag in df.columns:
            df.loc[df[flag].astype(float).eq(1.0), col] = np.nan
    if "log_glf_mean_opdays" in df.columns and "glf_mean_opdays_missing" in df.columns:
        df.loc[df["glf_mean_opdays_missing"].astype(float).eq(1.0), "log_glf_mean_opdays"] = np.nan
    df["qliq_early_m3d"] = df["kpod_mean"] * df["nominal_flow_m3d"]
    return df[["well_key", "install_date", *ALL_COVARIATES]].copy()


def attach_covariates(
    pop: pd.DataFrame,
    *,
    frame: pd.DataFrame | None = None,
    tolerance_days: int = _JOIN_TOL_DAYS,
    required: list[str] | None = None,
) -> pd.DataFrame:
    """Join ``COVARIATES`` onto an ``esp_population`` frame by well + nearest install.

    Adds the covariate columns plus ``covariates_present`` — all of ``required``
    (default: ``TRANSFERABLE_COVARIATES``) non-null.  Requiring only the covariates a
    model will actually use matters: demanding all four costs 3.3x the population
    (Ya_brt 810 -> 245 runs) to gate on two that no honest transfer can use anyway.

    Nothing is dropped here — the caller decides, so that coverage stays reportable
    rather than silently applied.
    """
    required = list(TRANSFERABLE_COVARIATES) if required is None else list(required)
    frame = build_covariate_frame() if frame is None else frame

    cols = [c for c in ALL_COVARIATES if c in frame.columns]
    right = frame.copy()
    right["code"] = right["well_key"].map(crosswalk.norm_well)
    right["install"] = pd.to_datetime(right["install_date"], errors="coerce")
    right = right.dropna(subset=["code", "install"])[["code", "install", *cols]]

    left = pop.copy()
    left["_row"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("install"),
        right.sort_values("install"),
        on="install",
        by="code",
        tolerance=pd.Timedelta(days=tolerance_days),
        direction="nearest",
    )
    out = merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)
    out["covariates_present"] = out[required].notna().all(axis=1)
    return out


def coverage_report(pop: pd.DataFrame) -> pd.DataFrame:
    """Per-covariate presence and the hazard split by availability.

    The hazard split is the load-bearing diagnostic, not the presence rate: if
    ``fail_per_month`` differs between the present and missing halves, covariate
    missingness is informative, and a model that sees both halves will encode it.  That
    association is field-specific and does not transport (see the module docstring).
    """
    rows = []
    for present, grp in pop.groupby(pop["covariates_present"], sort=True):
        days = float(pd.to_numeric(grp["tte"], errors="coerce").sum())
        events = int(pd.to_numeric(grp["event"], errors="coerce").sum())
        rows.append({
            "covariates_present": bool(present),
            "n_runs": int(len(grp)),
            "n_events": events,
            "pump_days": days,
            "fail_per_month": 30.4 * events / days if days > 0 else float("nan"),
        })
    out = pd.DataFrame(rows)
    for col in ALL_COVARIATES:
        if col in pop.columns:
            out[f"{col}_present"] = float(pop[col].notna().mean())
    return out


def support_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Is each covariate usable for a ``train`` -> ``test`` transfer?

    A gradient-boosted model with an exp() link cannot honestly extrapolate a covariate
    outside its training support — it will emit a confident, arbitrary number.  Two
    failure modes are checked, and either one disqualifies a covariate:

    * ``share_in_train_range`` — fraction of test values inside the train min..max.  At 0
      the model has never seen the region it is being asked about; the covariate is a
      field label, not a physical driver.
    * ``test_is_constant`` — the covariate has no variance within the test field, so it
      cannot discriminate anything there regardless of what it did in training.
    """
    covariates = list(ALL_COVARIATES) if covariates is None else list(covariates)
    rows = []
    for col in covariates:
        a = pd.to_numeric(train[col], errors="coerce").dropna()
        b = pd.to_numeric(test[col], errors="coerce").dropna()
        if a.empty or b.empty:
            rows.append({"covariate": col, "n_train": len(a), "n_test": len(b),
                         "share_in_train_range": float("nan"), "test_is_constant": True,
                         "transferable": False})
            continue
        inside = float(((b >= a.min()) & (b <= a.max())).mean())
        constant = bool(b.nunique() <= 1)
        rows.append({
            "covariate": col,
            "n_train": int(len(a)), "n_test": int(len(b)),
            "train_min": float(a.min()), "train_max": float(a.max()),
            "test_min": float(b.min()), "test_max": float(b.max()),
            "share_in_train_range": inside,
            "test_is_constant": constant,
            "transferable": bool(inside >= 0.5 and not constant),
        })
    return pd.DataFrame(rows)


def person_period(
    pop: pd.DataFrame,
    *,
    edges: tuple[float, ...] = AGE_EDGES,
    entry_col: str | None = "entry",
) -> pd.DataFrame:
    """Expand runs into at-risk age bins.

    One row per (run, age bin) with any exposure.  ``exposure`` is operating days spent in
    the bin; ``y`` is 1 only in the bin containing the failure.  A run censored at age T
    simply stops contributing after T — that is how censored runs enter the likelihood
    without any weighting scheme.

    ``entry_col`` supports left truncation: a run observed only from age ``entry`` onward
    contributes no exposure below it, which is what fitting a recent regime on an older
    fleet requires.
    """
    edges_arr = np.asarray(edges, dtype=float)
    tte = pd.to_numeric(pop["tte"], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(pop["event"], errors="coerce").fillna(0).to_numpy(dtype=int)
    if entry_col is not None and entry_col in pop.columns:
        entry = pd.to_numeric(pop[entry_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        entry = np.zeros_like(tte)

    carry = [c for c in (*ALL_COVARIATES, "code", "field", "contractor_group", "stratum", "source")
             if c in pop.columns]

    rows = []
    for i in range(len(pop)):
        t_end, e_i, t_start = tte[i], event[i], entry[i]
        if not np.isfinite(t_end) or t_end <= t_start:
            continue
        for k in range(len(edges_arr) - 1):
            lo, hi = edges_arr[k], edges_arr[k + 1]
            a = max(lo, t_start)
            b = min(hi, t_end)
            if b <= a:
                continue
            # The failure lands in the bin whose half-open span contains t_end.
            failed = int(e_i == 1 and lo < t_end <= hi)
            rows.append({
                "_run": i,
                "bin": k,
                "age_lo": lo,
                "age_hi": hi,
                "exposure": float(b - a),
                "y": failed,
            })
    pp = pd.DataFrame(rows)
    if pp.empty:
        return pp
    # Bin midpoint of the *exposed* span is not needed — the hazard is piecewise constant
    # per bin, so the bin's own lower edge identifies it. log1p keeps age 0 finite.
    pp["log_age"] = np.log1p(pp["age_lo"].to_numpy(dtype=float))
    for col in carry:
        pp[col] = pop[col].to_numpy()[pp["_run"].to_numpy()]
    pp["exposure"] = pp["exposure"].clip(lower=_MIN_EXPOSURE)
    return pp.reset_index(drop=True)


@dataclass
class DiscreteHazardModel:
    """Poisson hazard-rate model over person-period bins.

    ``lambda(age, x) = exp(f(log_age, x))`` in failures per operating day, fitted with
    ``log(exposure)`` as a fixed offset (CatBoost ``baseline``), so the model estimates a
    *rate*, not a count — bins of unequal width are handled exactly.
    """

    covariates: list[str] = dc_field(default_factory=lambda: list(COVARIATES))
    random_seed: int = 7
    iterations: int = 600
    depth: int = 4
    learning_rate: float = 0.05
    features_: list[str] = dc_field(default_factory=list)
    model_: object | None = None

    def _features(self) -> list[str]:
        return ["log_age", *self.covariates]

    def fit(self, pp: pd.DataFrame) -> "DiscreteHazardModel":
        from catboost import CatBoostRegressor, Pool

        self.features_ = self._features()
        missing = [c for c in self.features_ if c not in pp.columns]
        if missing:
            raise AssertionError(f"person-period frame missing features: {missing}")
        if int(pp["y"].sum()) < 3:
            raise ValueError("need at least 3 events to fit")

        offset = np.log(pp["exposure"].to_numpy(dtype=float))
        pool = Pool(pp[self.features_], pp["y"].to_numpy(dtype=float), baseline=offset)
        model = CatBoostRegressor(
            loss_function="Poisson",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(pool)
        self.model_ = model
        return self

    def rate(self, ages, covariates: dict[str, float] | pd.Series) -> np.ndarray:
        """Hazard rate (failures per operating day) at each age in ``ages``."""
        if self.model_ is None:
            raise RuntimeError("DiscreteHazardModel is not fitted")
        from catboost import Pool

        ages_arr = np.maximum(np.asarray(ages, dtype=float), 0.0)
        x = pd.DataFrame({"log_age": np.log1p(ages_arr)})
        for col in self.covariates:
            x[col] = float(covariates[col]) if pd.notna(covariates[col]) else np.nan
        # baseline 0 => the raw score is log(rate) per day, exposure factored out.
        # RawFormulaVal is requested explicitly: CatBoost's Poisson `predict` defaults to
        # prediction_type="Exponent" and would return the rate already, so relying on the
        # default and exponentiating would square it.
        raw = self.model_.predict(
            Pool(x[self.features_], baseline=np.zeros(len(x))),
            prediction_type="RawFormulaVal",
        )
        return np.clip(np.exp(raw), 0.0, _LAMBDA_MAX)

    def cumulative_hazard(self, t: float, covariates, *, entry: float = 0.0, step: float = 1.0) -> float:
        """Integral of the rate from ``entry`` to ``t`` on a daily grid."""
        if t <= entry:
            return 0.0
        grid = np.arange(entry, t, step, dtype=float)
        return float(np.sum(self.rate(grid, covariates) * step))

    def survival(self, t: float, covariates, *, entry: float = 0.0) -> float:
        return float(np.exp(-self.cumulative_hazard(t, covariates, entry=entry)))

    def p_fail(self, age: float, delta: float, covariates) -> float:
        """Conditional failure probability over ``delta`` operating days given age.

        ``1 - exp(-integral of lambda over [age, age+delta])`` — the same conditional
        quantity as ``survival.current_pump_p_fail``, computed straight off the hazard
        with no survival-curve bridge.
        """
        if delta <= 0:
            return 0.0
        h = self.cumulative_hazard(age + delta, covariates, entry=max(age, 0.0))
        return float(np.clip(1.0 - np.exp(-h), 0.0, 1.0))


def empirical_hazard(pp: pd.DataFrame) -> pd.DataFrame:
    """Observed failures / exposure per age bin — the thing a fit must reproduce."""
    g = pp.groupby(["bin", "age_lo", "age_hi"], as_index=False).agg(
        events=("y", "sum"), pump_days=("exposure", "sum")
    )
    g["fail_per_month"] = 30.4 * g["events"] / g["pump_days"].clip(lower=_MIN_EXPOSURE)
    return g


__all__ = [
    "AGE_EDGES",
    "COVARIATES",
    "DiscreteHazardModel",
    "attach_covariates",
    "coverage_report",
    "empirical_hazard",
    "person_period",
]
