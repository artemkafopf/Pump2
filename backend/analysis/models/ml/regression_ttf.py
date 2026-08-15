"""CatBoost direct regression for TTF/RUL in operating days.

This module is deliberately Weibull-free.  It builds a full-population frame,
restores registry-imputed numerics back to NaN for CatBoost, fits IPCW-weighted
quantile regressors, and exposes landmark residual-life helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd

from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.data.equipment_big import load_equipment_big
from analysis.features.restart_events import build_restart_features

QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)
LANDMARKS: tuple[int, ...] = (0, 30, 90, 180, 365, 730)

VERIFIED_CATS: list[str] = [
    "field",
    "contractor",
    "h2s_class",
    "pump_gabarit",
    "install_period",
]

VERIFIED_NUMERICS: list[str] = [
    "is_sour_flagged",
    "log_glf_mean_opdays",
    "freq_above_55hz_pct_early",
    "n_freq_steps_per_100d",
    "freq_std_early",
    "frac_kpod_below_0p7",
    "kpod_freq_mean",
    "load_std_early",
    "load_mean",
    "n_restarts_early30",
    "log_motor_power_kw",
    "vg_m",
    "nominal_freq_hz",
    "pbubble_atm",
    "curvature_deg10m",
    "log_run_seq",
    "log_days_since_prev_failure",
]

EQUIPMENT_FEATURES: list[str] = [
    "pump_od_mm",
    "nkt_diam_mm",
    "any_corr_protection",
    "cable_section_mm2",
    "cable_length_m",
    "ped_max_temp_c",
]

MODE_EVENT_COLS: list[str] = [
    "mode_group",
    "event_hydraulic",
    "event_electro-thermal",
    "event_protector",
]

REQUIRED_BASE_COLS: list[str] = [
    "row_id",
    "tte",
    "event",
    "well_key",
    "install_date",
    "run_days",
    "stratum_key",
]

_RESTORE_NAN_COLS: tuple[str, ...] = (
    "glf_mean_opdays",
    "freq_above_55hz_pct_early",
    "freq_std_early",
    "load_mean",
    "load_std_early",
    "kpod_freq_mean",
    "vg_m",
    "pbubble_atm",
    "nominal_freq_hz",
    "motor_power_kw",
)


def _missing_flag_features(df: pd.DataFrame) -> list[str]:
    roots = set(_RESTORE_NAN_COLS) | {"curvature"}
    return [f"{c}_missing" for c in roots if f"{c}_missing" in df.columns]


def _feature_cols(df: pd.DataFrame, *, include_equipment: bool = False) -> list[str]:
    cols = VERIFIED_NUMERICS + VERIFIED_CATS + _missing_flag_features(df)
    if include_equipment:
        cols += EQUIPMENT_FEATURES + ["equipment_matched"]
    return list(dict.fromkeys(cols))


def _restore_nan_after_registry_imputation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _RESTORE_NAN_COLS:
        miss = f"{col}_missing"
        if col in out.columns and miss in out.columns:
            out.loc[out[miss].astype(float).eq(1.0), col] = np.nan
    if "log_glf_mean_opdays" in out.columns and "glf_mean_opdays_missing" in out.columns:
        out.loc[out["glf_mean_opdays_missing"].astype(float).eq(1.0), "log_glf_mean_opdays"] = np.nan
    if "log_motor_power_kw" in out.columns and "motor_power_kw_missing" in out.columns:
        out.loc[out["motor_power_kw_missing"].astype(float).eq(1.0), "log_motor_power_kw"] = np.nan
    return out


def _cast_categoricals(df: pd.DataFrame, cats: list[str] = VERIFIED_CATS) -> pd.DataFrame:
    out = df.copy()
    for col in cats:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("NA").astype(str)
    return out


def _assert_schema(df: pd.DataFrame, features: list[str]) -> None:
    missing = [c for c in REQUIRED_BASE_COLS + features if c not in df.columns]
    if missing:
        raise AssertionError(f"regression frame missing columns: {missing}")


def assert_no_perfect_tte_leakage(df: pd.DataFrame, features: list[str], *, tte_col: str = "tte") -> None:
    """Fail if a numeric feature is a perfect finite clone/ranking of TTE."""
    if tte_col not in df.columns:
        return
    y = pd.to_numeric(df[tte_col], errors="coerce")
    offenders = []
    for col in features:
        if col not in df.columns or col == tte_col:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 3 or x[mask].nunique() < 2:
            continue
        corr = x[mask].corr(y[mask])
        if np.isfinite(corr) and np.isclose(abs(corr), 1.0):
            offenders.append(col)
    if offenders:
        raise AssertionError(f"possible outcome leakage: {offenders}")


def build_regression_frame(
    tte_col: str = "ttf_mix",
    *,
    include_equipment: bool = True,
    equipment_match_table: Path | None = None,
) -> pd.DataFrame:
    """Build the full-population CatBoost regression frame."""
    base = build_competing_risks_df(tte_col=tte_col).copy()
    restarts = build_restart_features(window_op_days=30)[["row_id", "n_restarts_reconciled"]]
    restarts = restarts.rename(columns={"n_restarts_reconciled": "n_restarts_early30"})
    df = base.merge(restarts, on="row_id", how="left")
    df["n_restarts_early30"] = df["n_restarts_early30"].fillna(0).astype(float)
    if "curvature_filled" in df.columns:
        df["curvature_deg10m"] = pd.to_numeric(df["curvature_filled"], errors="coerce").fillna(0.0)
    elif "curvature" in df.columns:
        df["curvature_deg10m"] = pd.to_numeric(df["curvature"], errors="coerce").fillna(0.0)
    if include_equipment:
        df = join_equipment_block(df, match_table_path=equipment_match_table)
    df = _restore_nan_after_registry_imputation(df)
    df = _cast_categoricals(df)
    features = _feature_cols(df, include_equipment=include_equipment)
    keep = list(dict.fromkeys(REQUIRED_BASE_COLS + features + MODE_EVENT_COLS))
    _assert_schema(df, features)
    assert_no_perfect_tte_leakage(df, features)
    return df[keep].copy()


def join_equipment_block(
    frame: pd.DataFrame,
    *,
    tolerance_days: int = 7,
    match_table_path: Path | None = None,
    loader: Callable[[], pd.DataFrame] = load_equipment_big,
) -> pd.DataFrame:
    """Join exploratory equipment covariates by well + nearest install date."""
    out = frame.copy()
    eq = loader().copy()
    eq = eq[eq["is_esp"].fillna(False)].copy()
    eq["install_date"] = pd.to_datetime(eq["install_date"], errors="coerce")
    out["_install_ts"] = pd.to_datetime(out["install_date"], errors="coerce")
    rows = []
    eq_by_well = {w: g.sort_values("install_date") for w, g in eq.groupby("well_key")}
    for idx, r in out[["well_key", "_install_ts"]].iterrows():
        g = eq_by_well.get(r["well_key"])
        if g is None or pd.isna(r["_install_ts"]):
            rows.append({"_idx": idx, "equipment_matched": 0})
            continue
        delta = (g["install_date"] - r["_install_ts"]).abs().dt.days
        best_pos = delta.idxmin()
        if pd.isna(delta.loc[best_pos]) or float(delta.loc[best_pos]) > tolerance_days:
            rows.append({"_idx": idx, "equipment_matched": 0})
            continue
        rec = g.loc[best_pos, EQUIPMENT_FEATURES].to_dict()
        rec.update({"_idx": idx, "equipment_matched": 1, "equipment_match_delta_d": float(delta.loc[best_pos])})
        rows.append(rec)
    matched = pd.DataFrame(rows).set_index("_idx")
    for col in EQUIPMENT_FEATURES:
        raw = matched[col] if col in matched.columns else pd.Series(np.nan, index=matched.index)
        out[col] = pd.to_numeric(raw.reindex(out.index), errors="coerce")
    if "any_corr_protection" in matched.columns:
        out["any_corr_protection"] = (
            matched["any_corr_protection"].reindex(out.index).astype("boolean").astype("Int64").astype(float)
        )
    out["equipment_matched"] = matched["equipment_matched"].reindex(out.index).fillna(0).astype(int)
    out = out.drop(columns=["_install_ts"])
    rate = float(out["equipment_matched"].mean()) if len(out) else float("nan")
    rate_row = pd.DataFrame([{"n_runs": len(out), "n_matched": int(out["equipment_matched"].sum()), "match_rate": rate}])
    if match_table_path is not None:
        match_table_path.parent.mkdir(parents=True, exist_ok=True)
        rate_row.to_csv(match_table_path, index=False)
    if np.isfinite(rate) and rate < 0.8:
        warnings.warn(f"equipment match rate below 80%: {rate:.1%}", RuntimeWarning, stacklevel=2)
    return out


def _km_predict_at(kmf, t: float, eps: float = 1e-6) -> float:
    val = float(kmf.predict(float(t)))
    if not np.isfinite(val):
        return eps
    return max(val, eps)


def _fit_censoring_km(df: pd.DataFrame, *, dur: str = "tte", ev: str = "event"):
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(pd.to_numeric(df[dur], errors="coerce").to_numpy(float), 1 - df[ev].to_numpy(int))
    return kmf


def _censoring_kms_by_stratum(
    train: pd.DataFrame,
    *,
    dur: str = "tte",
    ev: str = "event",
    strata_col: str = "stratum_key",
    min_censored: int = 5,
) -> tuple[object, dict[object, object]]:
    global_km = _fit_censoring_km(train, dur=dur, ev=ev)
    by_s = {}
    for s, g in train.groupby(strata_col, observed=True):
        if int((1 - g[ev].astype(int)).sum()) >= min_censored:
            by_s[s] = _fit_censoring_km(g, dur=dur, ev=ev)
    return global_km, by_s


def _truncate_nonzero(w: pd.Series, truncate: float) -> pd.Series:
    out = w.copy()
    nz = out[out > 0]
    if len(nz) and truncate > 0:
        cap = float(nz.quantile(1.0 - truncate))
        out = out.clip(upper=cap)
    return out


def ipcw_weights(
    train: pd.DataFrame,
    *,
    dur: str = "tte",
    ev: str = "event",
    strata_col: str = "stratum_key",
    truncate: float = 0.05,
) -> pd.Series:
    """Observed-failure IPCW weights, aligned to ``train``."""
    global_km, by_s = _censoring_kms_by_stratum(train, dur=dur, ev=ev, strata_col=strata_col)
    weights = []
    for r in train[[dur, ev, strata_col]].itertuples(index=False):
        if int(getattr(r, ev)) != 1:
            weights.append(0.0)
            continue
        km = by_s.get(getattr(r, strata_col), global_km)
        weights.append(1.0 / _km_predict_at(km, float(getattr(r, dur))))
    return _truncate_nonzero(pd.Series(weights, index=train.index, dtype=float), truncate)


def conditional_ipcw_weights(
    landmark_df: pd.DataFrame,
    source_train: pd.DataFrame,
    *,
    dur: str = "tte",
    ev: str = "event",
    strata_col: str = "stratum_key",
    age_col: str = "age_op_d",
    truncate: float = 0.05,
) -> pd.Series:
    """Landmark residual-life IPCW: G(L) / G(T) for observed failures."""
    global_km, by_s = _censoring_kms_by_stratum(source_train, dur=dur, ev=ev, strata_col=strata_col)
    weights = []
    for r in landmark_df[[dur, ev, strata_col, age_col]].itertuples(index=False):
        if int(getattr(r, ev)) != 1:
            weights.append(0.0)
            continue
        km = by_s.get(getattr(r, strata_col), global_km)
        g_l = _km_predict_at(km, float(getattr(r, age_col)))
        g_t = _km_predict_at(km, float(getattr(r, dur)))
        weights.append(g_l / g_t)
    return _truncate_nonzero(pd.Series(weights, index=landmark_df.index, dtype=float), truncate)


def _latest_validation_split(df: pd.DataFrame, frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["_dt"] = pd.to_datetime(d["install_date"], errors="coerce")
    d = d.sort_values("_dt")
    n_val = max(1, int(round(len(d) * frac)))
    if len(d) <= n_val + 5:
        return d.drop(columns=["_dt"]), d.iloc[0:0].drop(columns=["_dt"])
    return d.iloc[:-n_val].drop(columns=["_dt"]), d.iloc[-n_val:].drop(columns=["_dt"])


def _catboost_fit(
    df: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    loss_function: str,
    weights: pd.Series,
    random_seed: int,
):
    from catboost import CatBoostRegressor, Pool

    fit_df, val_df = _latest_validation_split(df)
    fit_w = weights.reindex(fit_df.index).fillna(0.0)
    val_w = weights.reindex(val_df.index).fillna(0.0) if len(val_df) else None
    train_pool = Pool(fit_df[features], fit_df[target], weight=fit_w, cat_features=[c for c in VERIFIED_CATS if c in features])
    eval_set = None
    if len(val_df) and float(val_w.sum()) > 0:
        eval_set = Pool(val_df[features], val_df[target], weight=val_w, cat_features=[c for c in VERIFIED_CATS if c in features])
    model = CatBoostRegressor(
        loss_function=loss_function,
        depth=5,
        learning_rate=0.05,
        iterations=800,
        random_seed=random_seed,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=50,
    )
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)
    return model


@dataclass
class TTFRegressor:
    include_equipment: bool = True
    random_seed: int = 7
    features_: list[str] = field(default_factory=list)
    models_: dict[float, object] = field(default_factory=dict)
    mean_model_: object | None = None

    def fit(self, train_df: pd.DataFrame) -> "TTFRegressor":
        self.features_ = _feature_cols(train_df, include_equipment=self.include_equipment)
        _assert_schema(train_df, self.features_)
        w = ipcw_weights(train_df)
        fit_df = train_df.loc[w > 0].copy()
        fit_w = w.loc[fit_df.index]
        for q in QUANTILES:
            self.models_[q] = _catboost_fit(
                fit_df, features=self.features_, target="tte",
                loss_function=f"Quantile:alpha={q}", weights=fit_w,
                random_seed=self.random_seed,
            )
        self.mean_model_ = _catboost_fit(
            fit_df, features=self.features_, target="tte", loss_function="RMSE",
            weights=fit_w, random_seed=self.random_seed,
        )
        return self

    def predict_ttf(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("TTFRegressor is not fitted")
        preds = np.column_stack([self.models_[q].predict(df[self.features_]) for q in QUANTILES])
        preds = np.sort(preds, axis=1)
        out = pd.DataFrame(preds, index=df.index, columns=["b10_op_d", "b50_op_d", "b90_op_d"])
        if self.mean_model_ is not None:
            out["mean_ttf_op_d"] = self.mean_model_.predict(df[self.features_])
        return out


@dataclass
class RULRegressor:
    include_equipment: bool = True
    random_seed: int = 11
    landmarks: tuple[int, ...] = LANDMARKS
    uptime_factors_: pd.Series | None = None
    features_: list[str] = field(default_factory=list)
    models_: dict[float, object] = field(default_factory=dict)

    def make_landmark_frame(self, train_df: pd.DataFrame, landmarks: tuple[int, ...] | None = None) -> pd.DataFrame:
        landmarks = self.landmarks if landmarks is None else landmarks
        rows = []
        base_features = _feature_cols(train_df, include_equipment=self.include_equipment)
        leakage = [c for c in base_features if c in {"tte", "event", "run_days"} or c.endswith("_whole_run")]
        if leakage:
            raise AssertionError(f"leaky features in RUL frame: {leakage}")
        for lmk in landmarks:
            alive = train_df[pd.to_numeric(train_df["tte"], errors="coerce") > float(lmk)].copy()
            if alive.empty:
                continue
            alive["age_op_d"] = float(lmk)
            alive["rul_target_op_d"] = pd.to_numeric(alive["tte"], errors="coerce") - float(lmk)
            rows.append(alive)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(train_df.columns) + ["age_op_d", "rul_target_op_d"])

    def fit(self, train_df: pd.DataFrame) -> "RULRegressor":
        self.features_ = _feature_cols(train_df, include_equipment=self.include_equipment) + ["age_op_d"]
        ldf = self.make_landmark_frame(train_df)
        w = conditional_ipcw_weights(ldf, train_df)
        fit_df = ldf.loc[w > 0].copy()
        fit_w = w.loc[fit_df.index]
        for q in QUANTILES:
            self.models_[q] = _catboost_fit(
                fit_df, features=self.features_, target="rul_target_op_d",
                loss_function=f"Quantile:alpha={q}", weights=fit_w,
                random_seed=self.random_seed,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            uf = pd.to_numeric(train_df["tte"], errors="coerce") / pd.to_numeric(train_df["run_days"], errors="coerce").replace(0, np.nan)
        self.uptime_factors_ = uf.groupby(train_df["stratum_key"]).mean().clip(lower=1e-6)
        return self

    def predict_rul(self, df: pd.DataFrame, age_op_d: float) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("RULRegressor is not fitted")
        x = df.copy()
        x["age_op_d"] = float(age_op_d)
        preds = np.column_stack([self.models_[q].predict(x[self.features_]) for q in QUANTILES])
        preds = np.sort(preds, axis=1)
        return pd.DataFrame(preds, index=df.index, columns=["rul_b10_op_d", "rul_b50_op_d", "rul_b90_op_d"])

    def to_calendar(self, rul_op: pd.DataFrame, strata: pd.Series) -> pd.DataFrame:
        if self.uptime_factors_ is None:
            raise RuntimeError("uptime factors are not available; fit first or assign uptime_factors_")
        out = rul_op.copy()
        fac = strata.map(self.uptime_factors_).astype(float)
        global_fac = float(self.uptime_factors_.mean())
        fac = fac.fillna(global_fac).replace(0, global_fac)
        for col in [c for c in out.columns if c.endswith("_op_d")]:
            out[col.replace("_op_d", "_cal_d")] = out[col] / fac.to_numpy()
        return out


__all__ = [
    "EQUIPMENT_FEATURES",
    "LANDMARKS",
    "MODE_EVENT_COLS",
    "QUANTILES",
    "VERIFIED_CATS",
    "VERIFIED_FEATURES",
    "TTFRegressor",
    "RULRegressor",
    "build_regression_frame",
    "conditional_ipcw_weights",
    "ipcw_weights",
    "join_equipment_block",
    "assert_no_perfect_tte_leakage",
]

VERIFIED_FEATURES = VERIFIED_NUMERICS + VERIFIED_CATS
