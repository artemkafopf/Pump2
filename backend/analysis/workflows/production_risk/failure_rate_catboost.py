"""CatBoost failure-rate line — the ML counterpart to the Weibull survival line.

This module is the CatBoost side of the CatBoost-vs-Weibull failure-rate comparison
(``docs/notes/production_risk_catboost_vs_weibull_prompt.md``).  It is a **parallel model
line**, not a replacement: it plugs a gradient-boosted TTF survival curve into the *exact*
same monthly replay that the Weibull line uses (``failure_rate._append_interval_predictions``
with an injected ``p_fail_fn``), so the only thing that differs between the two lines is
where the survival curve comes from — never the exposure/aging/denominator plumbing.

Bridge (parallel to ``survival.current_pump_p_fail``)
-----------------------------------------------------
* ``TTFRegressor.predict_ttf`` gives per-run op-day quantiles ``(b10, b50, b90)`` — one
  inference per run.  These anchor a per-run CDF ``F(b10)=.1, F(b50)=.5, F(b90)=.9`` (plus
  ``F(0)=0``); ``build_cb_survival`` turns them into a monotone ``S_cb(t)=1-F(t)`` with an
  explicit linear pre-b10 ramp and an exponential post-b90 tail whose rate is the implied
  hazard of the ``b50→b90`` segment (hazard continuity at the joint).
* ``catboost_p_fail`` is *byte-for-byte* the conditional formula of ``current_pump_p_fail``:
  ``clip(1 - S_cb(age+Δ)/S_cb(age), 0, 1)`` — only the survival curve is swapped.

Capacity fairness
-----------------
A depth-5 boosted ensemble over 2,634 runs can partially memorise per-run failure times, so
an all-data in-sample CatBoost line is upper-bounded by memorisation.  The headline line is
therefore **cross-fitted**: K-fold, well-clustered (``well_key`` never spans folds), each
history run predicted by a model that never saw it (nor any run of its well).  Both lines
are produced; the caller leads the verdict with the cross-fit line.

No Weibull anywhere in the survival curve: matched runs use their full covariate row; runs
with no covariate row still get a *pure CatBoost* prediction from their known categoricals
(operational numerics left NaN — CatBoost handles them natively).  The one thing shared with
the Weibull path is exposure plumbing (month slicing, plan op-days, the registry
``uptime_factor`` used only to age runs with no observed total op-days); that parameterises
exposure, not the survival curve.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import crosswalk

# Quantile probabilities anchoring the per-run CDF (must match TTFRegressor QUANTILES).
_Q10, _Q50, _Q90 = 0.10, 0.50, 0.90
_ANCHOR_FLOOR = 1.0        # op-days: smallest admissible b10 (keeps CDF finite, never a step)
_ANCHOR_EPS = 1.0          # op-days: strict-increase separation between tied anchors
_SURV_EPS = 1e-6           # S_cb is clipped to (eps, 1]
_JOIN_TOL_DAYS = 7         # install-date tolerance for the covariate join (§6)


# --------------------------------------------------------------------------- #
# The bridge: TTF quantiles -> monotone survival curve -> conditional p_fail   #
# --------------------------------------------------------------------------- #
def _guard_anchors(b10: float, b50: float, b90: float) -> tuple[float, float, float]:
    """Clip raw quantile anchors to a positive floor and force strict monotonicity.

    ``predict_ttf`` only row-sorts its outputs; they can be ``<= 0`` or tied.  A degenerate
    row (b10 == b50 == b90) must still yield a valid steep-but-finite CDF, so we separate
    ties by ``_ANCHOR_EPS`` rather than collapsing to a step function / dividing by zero.
    """
    a10 = max(float(b10) if np.isfinite(b10) else _ANCHOR_FLOOR, _ANCHOR_FLOOR)
    a50 = max(float(b50) if np.isfinite(b50) else a10 + _ANCHOR_EPS, a10 + _ANCHOR_EPS)
    a90 = max(float(b90) if np.isfinite(b90) else a50 + _ANCHOR_EPS, a50 + _ANCHOR_EPS)
    return a10, a50, a90


def build_cb_survival(b10: float, b50: float, b90: float) -> tuple[Callable[[float], float], float]:
    """Monotone per-run survival ``S_cb(t)`` from op-day TTF quantiles; returns ``(S, b90)``.

    * ``t <= 0``            -> 1.0
    * ``0 < t <= b90``      -> ``1 - F`` with ``F`` piecewise-linear through
      ``(0,0),(b10,.1),(b50,.5),(b90,.9)`` (linear pre-b10 ramp included by construction).
    * ``t > b90``           -> exponential tail ``S(b90)·exp(-λ (t-b90))`` with
      ``λ = ln(S(b50)/S(b90)) / (b90-b50) = ln(5)/(b90-b50)`` — the implied hazard of the
      ``b50→b90`` segment, so the hazard is continuous at the ``b90`` joint.
    """
    a10, a50, a90 = _guard_anchors(b10, b50, b90)
    ts = np.array([0.0, a10, a50, a90], dtype=float)
    fs = np.array([0.0, _Q10, _Q50, _Q90], dtype=float)
    s90 = 1.0 - _Q90                                    # 0.10
    lam = math.log((1.0 - _Q50) / (1.0 - _Q90)) / (a90 - a50)  # ln(5)/(b90-b50)

    def surv(t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 1.0
        if t <= a90:
            f = float(np.interp(t, ts, fs))
            s = 1.0 - f
        else:
            s = s90 * math.exp(-lam * (t - a90))
        return float(min(max(s, _SURV_EPS), 1.0))

    return surv, a90


def catboost_p_fail(surv: Callable[[float], float], age_op_d: float, op_days: float) -> float:
    """Conditional monthly failure probability off a CatBoost survival curve.

    ``clip(1 - S_cb(age+Δ)/S_cb(age), 0, 1)`` — identical conditional math to
    ``survival.current_pump_p_fail``; only ``surv`` differs.
    """
    age = max(0.0, float(age_op_d))
    delta = max(0.0, float(op_days))
    s0 = surv(age)
    if s0 <= 0.0:
        return 1.0
    return float(np.clip(1.0 - surv(age + delta) / s0, 0.0, 1.0))


def catboost_p_fail_pmf(surv: Callable[[float], float], age_pmf: dict[int, float], op_days: float) -> float:
    """Age-PMF wrapper mirroring ``current_pump_p_fail``'s pmf handling exactly."""
    out = 0.0
    for age, weight in age_pmf.items():
        out += float(weight) * catboost_p_fail(surv, float(age), op_days)
    return float(np.clip(out, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Covariate join + per-run curve cache + cross-fit                            #
# --------------------------------------------------------------------------- #
def _join_key(code: str) -> str:
    return str(code).strip().casefold()


def _install_period(dt: datetime) -> str:
    """Same 3-bucket install cut the regression frame uses (``install_period``)."""
    year = dt.year
    if year <= 2019:
        return "<=2019"
    if year <= 2022:
        return "2020-2022"
    return "2023+"


@dataclass
class _ReplayDiag:
    """Diagnostics accumulated during one replay pass (per variant)."""
    # predicted-failure mass by field, split full-covariate vs categorical-only
    mass_full: dict[str, float] = dc_field(default_factory=lambda: defaultdict(float))
    mass_partial: dict[str, float] = dc_field(default_factory=lambda: defaultdict(float))
    # pump-month evaluation counts by field, and the subset evaluated past b90
    n_eval: dict[str, float] = dc_field(default_factory=lambda: defaultdict(float))
    n_eval_past_b90: dict[str, float] = dc_field(default_factory=lambda: defaultdict(float))

    def covariate_share_by_field(self) -> dict[str, float]:
        out: dict[str, float] = {}
        fields = set(self.mass_full) | set(self.mass_partial)
        tot_full = tot_part = 0.0
        for f in fields:
            full = float(self.mass_full.get(f, 0.0))
            part = float(self.mass_partial.get(f, 0.0))
            tot_full += full
            tot_part += part
            denom = full + part
            out[f] = full / denom if denom > 0 else float("nan")
        denom = tot_full + tot_part
        out["__global__"] = tot_full / denom if denom > 0 else float("nan")
        return out

    def past_b90_share_by_field(self) -> dict[str, float]:
        out: dict[str, float] = {}
        tot = past = 0.0
        for f in set(self.n_eval) | set(self.n_eval_past_b90):
            n = float(self.n_eval.get(f, 0.0))
            p = float(self.n_eval_past_b90.get(f, 0.0))
            tot += n
            past += p
            out[f] = p / n if n > 0 else float("nan")
        out["__global__"] = past / tot if tot > 0 else float("nan")
        return out


class CatBoostFailureModel:
    """Trains/caches CatBoost TTF curves and hands the replay per-interval ``p_fail_fn``s.

    Parameters
    ----------
    frame : the ``build_regression_frame`` output (full population, one row per run).
    well_field : plan well code -> reporting УН (used only to attribute diagnostics).
    n_folds : cross-fit folds (well-clustered).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        well_field: dict[str, str],
        *,
        n_folds: int = 5,
        random_seed: int = 7,
        include_equipment: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.well_field = dict(well_field)
        self.n_folds = int(n_folds)
        self.random_seed = int(random_seed)
        self.include_equipment = bool(include_equipment)
        self._features: list[str] = []
        # per-variant row_id -> (b10, b50, b90)
        self._quant: dict[str, dict[object, tuple[float, float, float]]] = {"insample": {}, "xfit": {}}
        # categorical-only quantiles: per-well (borrowed categoricals) and per (field, period) grid
        self._quant_well: dict[str, tuple[float, float, float]] = {}
        self._quant_grid: dict[tuple[str, str], tuple[float, float, float]] = {}
        # join index: join_key -> sorted list of (install_ts, row_id)
        self._by_well: dict[str, list[tuple[pd.Timestamp, object]]] = {}
        # curve caches so each distinct curve is built once
        self._curve_cache: dict[object, tuple[Callable[[float], float], float]] = {}
        self._fitted = False

    # -- fitting -------------------------------------------------------------
    def fit(self) -> "CatBoostFailureModel":
        from analysis.models.ml.regression_ttf import TTFRegressor  # lazy: confine catboost

        frame = self.frame
        model = TTFRegressor(include_equipment=self.include_equipment, random_seed=self.random_seed).fit(frame)
        self._features = list(model.features_)
        self._all_data_model = model

        # (a) in-sample quantiles for every run
        pred = model.predict_ttf(frame)
        for rid, row in zip(frame["row_id"], pred.itertuples(index=False)):
            self._quant["insample"][rid] = (float(row.b10_op_d), float(row.b50_op_d), float(row.b90_op_d))

        # (b) cross-fit quantiles: well-clustered K-fold, each run scored by a model
        #     that saw neither it nor any run of its well.
        self._fit_cross(frame)

        # (c) categorical-only fallbacks (pure ML, no Weibull) for unmatched intervals
        self._fit_categorical_only(frame, model)

        # (d) join index
        ts = pd.to_datetime(frame["install_date"], errors="coerce")
        for rid, key, t in zip(frame["row_id"], frame["well_key"].astype(str).map(_join_key), ts):
            if pd.isna(t):
                continue
            self._by_well.setdefault(key, []).append((t, rid))
        for key in self._by_well:
            self._by_well[key].sort(key=lambda x: x[0])
        self._fitted = True
        return self

    @staticmethod
    def _fold_membership(join_keys: list[str], n_folds: int) -> np.ndarray:
        """Row -> test-fold id, well-clustered so no ``well_key`` ever spans folds."""
        from sklearn.model_selection import GroupKFold

        groups = np.asarray([str(k) for k in join_keys])
        k = max(2, min(int(n_folds), len(set(groups))))
        fold = np.full(len(groups), -1, dtype=int)
        gkf = GroupKFold(n_splits=k)
        for fi, (_tr, te) in enumerate(gkf.split(groups, groups=groups)):
            fold[te] = fi
        return fold

    def _fit_cross(self, frame: pd.DataFrame) -> None:
        from analysis.models.ml.regression_ttf import TTFRegressor

        join_keys = frame["well_key"].astype(str).map(_join_key).tolist()
        fold = self._fold_membership(join_keys, self.n_folds)
        for fi in sorted(set(fold.tolist())):
            te_mask = fold == fi
            tr = frame.iloc[np.where(~te_mask)[0]]
            te = frame.iloc[np.where(te_mask)[0]]
            model = TTFRegressor(include_equipment=self.include_equipment, random_seed=self.random_seed).fit(tr)
            pr = model.predict_ttf(te)
            for rid, row in zip(te["row_id"], pr.itertuples(index=False)):
                self._quant["xfit"][rid] = (float(row.b10_op_d), float(row.b50_op_d), float(row.b90_op_d))
        # any run missed by the split (shouldn't happen) falls back to in-sample
        for rid, q in self._quant["insample"].items():
            self._quant["xfit"].setdefault(rid, q)

    def _blank_numeric_row(self, template: pd.Series) -> dict:
        """Copy a template feature row but blank operational numerics (categorical-only)."""
        from analysis.models.ml.regression_ttf import EQUIPMENT_FEATURES, VERIFIED_CATS, VERIFIED_NUMERICS

        row = {c: template.get(c) for c in self._features}
        for col in VERIFIED_NUMERICS + EQUIPMENT_FEATURES:
            if col in row and col != "curvature_deg10m":
                row[col] = np.nan
        for col in list(row):
            if col.endswith("_missing"):
                row[col] = 1.0
            if col == "equipment_matched":
                row[col] = 0
            if col == "curvature_deg10m":
                row[col] = 0.0
        for col in VERIFIED_CATS:
            if col in row and (row[col] is None or (isinstance(row[col], float) and np.isnan(row[col]))):
                row[col] = "NA"
        return row

    def _fit_categorical_only(self, frame: pd.DataFrame, model) -> None:
        """Predict categorical-only curves: one per well (borrowed cats) and a field×period grid."""
        from analysis.models.ml.regression_ttf import VERIFIED_CATS

        # per-well: first run of each well, numerics blanked
        well_rows = []
        well_keys = []
        for key, grp in frame.groupby(frame["well_key"].astype(str).map(_join_key)):
            tmpl = grp.iloc[0]
            well_rows.append(self._blank_numeric_row(tmpl))
            well_keys.append(key)
        if well_rows:
            wdf = pd.DataFrame(well_rows)
            for c in VERIFIED_CATS:
                if c in wdf.columns:
                    wdf[c] = wdf[c].astype("string").fillna("NA").astype(str)
            pr = model.predict_ttf(wdf)
            for key, row in zip(well_keys, pr.itertuples(index=False)):
                self._quant_well[key] = (float(row.b10_op_d), float(row.b50_op_d), float(row.b90_op_d))

        # field × period grid (for wells absent from the frame entirely)
        tmpl = frame.iloc[0]
        grid_rows = []
        grid_keys = []
        fields = sorted(frame["field"].dropna().astype(str).unique())
        periods = ["<=2019", "2020-2022", "2023+"]
        for fld in fields:
            for per in periods:
                row = self._blank_numeric_row(tmpl)
                if "field" in row:
                    row["field"] = fld
                if "install_period" in row:
                    row["install_period"] = per
                grid_rows.append(row)
                grid_keys.append((fld, per))
        if grid_rows:
            gdf = pd.DataFrame(grid_rows)
            for c in VERIFIED_CATS:
                if c in gdf.columns:
                    gdf[c] = gdf[c].astype("string").fillna("NA").astype(str)
            pr = model.predict_ttf(gdf)
            for gkey, row in zip(grid_keys, pr.itertuples(index=False)):
                self._quant_grid[gkey] = (float(row.b10_op_d), float(row.b50_op_d), float(row.b90_op_d))

    # -- curve lookup --------------------------------------------------------
    def _curve(self, cache_key: object, quant: tuple[float, float, float]) -> tuple[Callable[[float], float], float]:
        cached = self._curve_cache.get(cache_key)
        if cached is None:
            cached = build_cb_survival(*quant)
            self._curve_cache[cache_key] = cached
        return cached

    def _resolve_curve(
        self, code: str, start: datetime, variant: str
    ) -> tuple[Callable[[float], float], float, bool]:
        """Return ``(S_cb, b90, matched)`` for one replay interval (pure ML, no Weibull)."""
        key = _join_key(code)
        rows = self._by_well.get(key)
        start_ts = pd.Timestamp(start)
        if rows:
            best_ts, best_rid = min(rows, key=lambda r: abs((r[0] - start_ts).days))
            if abs((best_ts - start_ts).days) <= _JOIN_TOL_DAYS and best_rid in self._quant[variant]:
                surv, b90 = self._curve((variant, "run", best_rid), self._quant[variant][best_rid])
                return surv, b90, True
            # same well, no install-date match -> categorical-only from borrowed cats
            if key in self._quant_well:
                surv, b90 = self._curve(("well", key), self._quant_well[key])
                return surv, b90, False
        # well absent from frame -> field×period grid
        fld = crosswalk.map_model_field_from_well(code) or ""
        per = _install_period(start)
        gkey = (str(fld), per)
        quant = self._quant_grid.get(gkey)
        if quant is None and self._quant_grid:
            # unknown field: borrow any period-matched grid entry, else the first
            quant = next((q for (f, p), q in self._quant_grid.items() if p == per), None)
            if quant is None:
                quant = next(iter(self._quant_grid.values()))
        if quant is None:  # no grid at all (degenerate) — flat curve
            quant = (1.0, 2.0, 3.0)
        surv, b90 = self._curve(("grid", gkey), quant)
        return surv, b90, False

    # -- the injected callable ----------------------------------------------
    def p_fail_fn(
        self, code: str, start: datetime, field: str, variant: str, diag: _ReplayDiag
    ) -> Callable[[dict[int, float], float], float]:
        """Build the ``p_fail_fn(age_pmf, op_days)`` the replay injects for this interval."""
        surv, b90, matched = self._resolve_curve(code, start, variant)

        def fn(age_pmf: dict[int, float], op_days: float) -> float:
            p = 0.0
            for age, weight in age_pmf.items():
                diag.n_eval[field] += float(weight)
                if float(age) > b90:
                    diag.n_eval_past_b90[field] += float(weight)
                p += float(weight) * catboost_p_fail(surv, float(age), op_days)
            p = float(np.clip(p, 0.0, 1.0))
            if matched:
                diag.mass_full[field] += p
            else:
                diag.mass_partial[field] += p
            return p

        return fn


def build_model(
    well_field: dict[str, str],
    *,
    n_folds: int = 5,
    random_seed: int = 7,
    include_equipment: bool = True,
) -> CatBoostFailureModel:
    """Convenience builder: build the regression frame and fit the CatBoost failure model."""
    from analysis.models.ml.regression_ttf import build_regression_frame

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        frame = build_regression_frame(include_equipment=include_equipment)
    return CatBoostFailureModel(
        frame,
        well_field,
        n_folds=n_folds,
        random_seed=random_seed,
        include_equipment=include_equipment,
    ).fit()


__all__ = [
    "CatBoostFailureModel",
    "build_cb_survival",
    "build_model",
    "catboost_p_fail",
    "catboost_p_fail_pmf",
    "_ReplayDiag",
]
