"""Grey-box physical reconstruction of pump intake pressure (Рприем).

This module is as much a record of what the data *refuses* to support as of what it does.
Four physically-motivated routes were built and measured; three are rejected, and the
rejections are kept here in code because each one is a trap that looks correct on paper
and fails only against the warehouse.

The chain
---------
::

    P_res  (rpl)  --[ inflow / IPR ]-->  P_wf  (rzab)  --[ annulus ]-->  P_intake

Guarded operating-day medians are ``rpl`` 174.3, ``rzab`` 65.6, ``rpump_intake`` 51.0 atm,
so the inflow leg nominally carries ~109 atm of the drop and the annulus ~10.5.

Rejection 1 — Рзаб is *calculated*, so the annulus leg is circular
------------------------------------------------------------------
Calibrating ``P_wf − P_intake = ρ g h_eff`` against Рзаб looks superb: median absolute
error **0.7 atm**, R² 0.744, and a physically sane median ``h_eff`` of 94 m with 96 % of
wells inside 0-600 m.  It is an artefact.  The telemetry column map names the source
field **«Расчетное забойное давление»** — *calculated* bottomhole pressure — and the
warehouse behaves accordingly: Рзаб is present on **93.3 %** of days where Рприем exists
and **10.3 %** of days where it does not, with cross-sectional correlation 0.933.  It is
computed *from* Рприем, so fitting the annulus against it recovers the warehouse's own
conversion formula, not the wellbore.  :func:`fit_annulus_h` is retained for exactly that
purpose — reading the conversion — and is **not** used to impute.

Rejection 2 — Рпл carries no within-well drawdown signal
--------------------------------------------------------
Within a well, ``corr(rpl, rzab)`` has median **0.28**, lower quartile **0.00**, minimum
−0.94.  Reservoir pressure here behaves like a periodically-restated map value, not a
daily measurement of the object Рзаб tracks.  Any IPR driven off it inherits that.

Rejection 3 — Vogel over-predicts rate sensitivity here, badly
---------------------------------------------------------------
Vogel (1968) is the *right* IPR on paper: reservoir pressure (median 174 atm) sits below
the bubble point this project carries as a field label (230-260 atm on Ya), so the
reservoir is saturated and solution-gas drive is the correct regime.  It still fails.
Reaching the observed 174 → 51 atm drop requires ``q/q_max ≈ 0.87``, i.e. the well pinned
near absolute open flow, where the Vogel curve is steepest.  Halving the rate from there
swings predicted :math:`P_{wf}` by ~67 atm, while the measured within-well sd of Рприем
is **19 atm**.  The form is too stiff by roughly 3×, and the optimiser can only escape by
inflating ``q_max`` until the curve flattens — at which point it can no longer reach the
level.  Scored on held-out mid-well blocks (below) Vogel lands at **R² −0.289**, and on
unseen wells at **−1.445** — the only candidate that is worse than predicting a constant.
:func:`vogel_pwf` stays available as ``mode="vogel"`` so the rejection is reproducible.

Rejection 4 — nodal analysis through the pump curve
----------------------------------------------------
Static column at ``vg_m`` minus affinity-scaled catalog head has the real merit of
running on ``freq``, which survives on 99.9 % of the gap.  Cross-well it correlates
**+0.20** with mean Рприем and within-well |r| is 0.33.  Wellhead pressure and tubing
friction are not in the warehouse, and without them the level cannot be closed.

What survives, and how good it actually is
-------------------------------------------
Scored as deployment actually works — hide the contiguous middle 40 % of each well and
predict it from the ends (219 730 held-out days, "warm"), and separately leave whole
wells out ("cold", 558 456 days):

================================  ==============  ==============
model                             warm MAE / R²   cold MAE / R²
================================  ==============  ==============
``cb_plain`` (CatBoost, no phys)  13.61 / +0.675  15.81 / +0.597
well median (the baseline)        17.89 / +0.421  25.92 / −0.079
``phys_linear`` (this module)     18.22 / +0.403  26.27 / −0.034
``phys_vogel``                    29.35 / −0.289  45.59 / −1.445
================================  ==============  ==============

**Read that honestly: no physical form here beats predicting the well's own median.**
``phys_linear`` lands 2 % *behind* the baseline on the warm case and within noise of it on
the cold one.  The covariates carry almost no extra within-well information; a model that
claims otherwise is reporting leakage.

⚠ **And the fitted form is degenerate — say so rather than reading the constants.**
Across 757 calibrated wells the inverse productivity index ``b`` has median **0.00**, and
``h_eff`` rails to a bound in **88.6 %** of wells (25th pct −50 m, median 0 m, 75th pct
1500 m; only 11.4 % land inside the 0-600 m plausible band).  Both terms are effectively
switched off and the prediction is carried by the datum ``a`` alone — which is why the
model scores level with the well-median baseline: *it has become the well-median
baseline*.  Centring on :data:`RHO_REF` is what makes that harmless rather than
catastrophic (it moves the level into ``a``, where the data can identify it), but it does
not create signal that is not there.  **Do not quote ``b`` as a productivity index or
``h_eff`` as a wellbore geometry from this fit.**

The module is kept as the physical-realization arm of a two-realization design, and it
earns that place two ways: it is a genuinely independent estimator, so its disagreement
with the ML arm is a usable per-row uncertainty signal (median 4.94 atm), and its
prediction is the single strongest feature the ML arm has (importance 21.8, top of the
list).  It is not kept because it predicts well on its own.

The default form
----------------
.. math::
    P_{intake} = a_w - b_w\\, q_{liq} - \\frac{\\rho_{liq}(w_c)\\, g\\, h_{eff,w}}{101325}

* :math:`a_w` — datum pressure (atm): reservoir pressure less the standing column, the
  part Рпл cannot supply and the well's own history must.
* :math:`b_w` — inverse productivity index (atm per m³/d): the linearised IPR.  This is
  the same object the warehouse's ``kprod`` encodes, and linear is the right
  linearisation *because* Vogel's curvature was measured and rejected above.
* :math:`h_{eff,w}` — effective annular liquid column (m), the one term carrying genuine
  daily variation: water cut moves :math:`\\rho_{liq}`, so the predicted gradient moves
  with it.

Fitted per well by iteratively reweighted least squares (soft-L1 equivalent); the daily
telemetry carries spikes that survive the range guards and a plain OLS on three
parameters is moved materially by one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

PA_PER_ATM = 101325.0
G = 9.80665

RHO_WATER = 1010.0
RHO_OIL = 850.0
#: Reference liquid density the annulus term is centred on.  Water cut moves ρ over only
#: 850-1010 kg/m³ — an 18.8 % span — so the raw ``ρ·g·h/101325`` column is nearly a
#: constant and therefore nearly collinear with the datum term.  Fitted uncentred, ``a``
#: and ``h_eff`` trade off about 1:1, land at large offsetting values, and clipping either
#: one to its physical bound destroys the balance the other was holding: that bug scored
#: MAE **90.2 atm** on held-out blocks against 18.5 for the same form fitted centred.
#: Centring identifies ``a`` as the datum pressure at ``RHO_REF`` and leaves ``h_eff`` to
#: be identified by the *variation* in water cut alone, which is the only part of it the
#: data can actually see.
RHO_REF = 930.0

#: Physical bounds on the fitted constants.
H_EFF_BOUNDS = (-50.0, 1500.0)
QMAX_MULT_BOUNDS = (1.01, 50.0)
#: Datum pressure and inverse-PI bounds — wide enough not to bind on real wells, tight
#: enough to stop a 20-day well's fit from running away.
A_BOUNDS = (0.5, 600.0)
B_BOUNDS = (0.0, 5.0)

#: ``h_eff`` outside this band is flagged: the fit converged, but not on a wellbore with a
#: pump hanging above its perforations.
H_EFF_PLAUSIBLE = (0.0, 600.0)


def liquid_density(watercut_pct) -> np.ndarray:
    """Liquid-phase mixture density (kg/m³) from water cut in percent."""
    fw = np.asarray(watercut_pct, dtype=float) / 100.0
    return RHO_WATER * fw + RHO_OIL * (1.0 - fw)


def column_atm(rho_kgm3, h_m) -> np.ndarray:
    """Hydrostatic head of ``h_m`` metres of fluid at ``rho_kgm3``, in atm."""
    return np.asarray(rho_kgm3, dtype=float) * G * np.asarray(h_m, dtype=float) / PA_PER_ATM


def vogel_pwf(p_res, q, q_max) -> np.ndarray:
    """Flowing bottomhole pressure from Vogel's IPR, solved for :math:`P_{wf}`.

    Solving ``0.8x² + 0.2x − (1 − q/q_max) = 0`` for ``x = P_wf/P_res``::

        x = (−0.2 + sqrt(0.04 + 3.2·(1 − q/q_max))) / 1.6

    ⚠ Measured and **rejected** as the imputation form on this fleet — see module
    docstring, rejection 3.  Retained so the rejection stays reproducible and because the
    curve is still the right description of a solution-gas-drive inflow in general.
    """
    pr = np.asarray(p_res, dtype=float)
    qq = np.asarray(q, dtype=float)
    qm = np.asarray(q_max, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip(qq / np.where(qm > 0, qm, np.nan), 0.0, 1.0)
        disc = 0.04 + 3.2 * (1.0 - ratio)
        x = (-0.2 + np.sqrt(np.clip(disc, 0.0, None))) / 1.6
        return pr * np.clip(x, 0.0, 1.0)


def linear_pwf(p_res, q, pi) -> np.ndarray:
    """Straight-line-IPR bottomhole pressure, ``P_wf = P_res − q/PI`` (``pi`` = ``kprod``)."""
    pr = np.asarray(p_res, dtype=float)
    qq = np.asarray(q, dtype=float)
    j = np.asarray(pi, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pr - qq / np.where(j > 0, j, np.nan)


def fit_annulus_h(rzab, p_intake, watercut_pct) -> float:
    """Effective annular column (m) implied by the warehouse's Рзаб.

    ⚠ **This measures the warehouse's own Рприем→Рзаб conversion, not the wellbore.**
    «Расчетное забойное давление» is a calculated column (module docstring, rejection 1),
    so this number is a description of that calculation.  Useful for auditing it; not
    admissible as independent physics, and deliberately not used by
    :class:`PhysicalIntakeModel`.

    Inverts the head relation per day and takes the median — closed form in ``h``, so the
    only question is how to reduce a noisy sample.  Daily values span −2556 … 6495 m;
    the median is what makes that harmless, and it is why this is not a mean.
    """
    rho = liquid_density(watercut_pct)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = (np.asarray(rzab, float) - np.asarray(p_intake, float)) * PA_PER_ATM / (rho * G)
    h = h[np.isfinite(h)]
    if not len(h):
        return np.nan
    return float(np.clip(np.median(h), *H_EFF_BOUNDS))


def predict_intake_linear(qliq, watercut_pct, a, b, h_eff_m) -> np.ndarray:
    """The default form: datum, linearised IPR drawdown, annular column.

    The annulus term is centred on :data:`RHO_REF`, so ``a`` is the datum pressure at that
    reference density and ``h_eff_m`` scales only the water-cut *deviation* from it — see
    :data:`RHO_REF` for why the uncentred form cannot be used.
    """
    q = np.asarray(qliq, dtype=float)
    rho = liquid_density(watercut_pct)
    return a - b * q - column_atm(rho - RHO_REF, h_eff_m)


def predict_intake_vogel(p_res, qliq, watercut_pct, q_max, h_eff_m) -> np.ndarray:
    """The rejected Vogel chain, kept for the reproducibility of rejection 3."""
    return vogel_pwf(p_res, qliq, q_max) - column_atm(liquid_density(watercut_pct), h_eff_m)


def _irls(X: np.ndarray, y: np.ndarray, n_iter: int = 8) -> np.ndarray:
    """Iteratively reweighted least squares with a soft-L1 weight.

    Cheaper and more stable than handing a 3-parameter linear problem to a general
    non-linear optimiser, and the robustness is the point: one 380 atm spike in a 40 atm
    well moves an OLS fit of three parameters materially.
    """
    w = np.ones(len(y))
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        Xw = X * w[:, None]
        beta, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
        r = y - X @ beta
        s = float(np.median(np.abs(r - np.median(r)))) * 1.4826 + 1e-6
        w = 1.0 / np.sqrt(1.0 + (r / (2.0 * s)) ** 2)
    return beta


@dataclass
class WellCalibration:
    """Fitted constants for one well.

    ``a`` datum pressure (atm), ``b`` inverse productivity index (atm per m³/d),
    ``h_eff_m`` effective annular liquid column (m).  ``q_max`` is populated only in
    ``mode="vogel"``.

    ``source`` is the tier the constants came from — ``"well"``, ``"field"`` or
    ``"global"``.  Anything but ``"well"`` is an extrapolation, and the validation report
    shows it carries several times the error.  ``resid_mad`` is **in-sample**: a floor on
    the error, never an estimate of it.
    """

    well_key: str
    a: float = np.nan
    b: float = np.nan
    h_eff_m: float = np.nan
    q_max: float = np.nan
    source: str = "global"
    n_days: int = 0
    resid_mad: float = np.nan
    flag: str = ""


@dataclass
class PhysicalIntakeModel:
    """Per-well calibrated grey-box model with field-level and global fallbacks.

    Interface mirrors the ML side so the two are swappable in the validation harness::

        m = PhysicalIntakeModel().fit(train_df)
        yhat = m.predict(test_df)

    ``fit`` consumes only rows where ``is_observed`` is True.  ``mode`` is ``"linear"``
    (default, the surviving form) or ``"vogel"`` (the measured rejection).
    """

    mode: str = "linear"
    min_days: int = 20
    calib: dict[str, WellCalibration] = dc_field(default_factory=dict)
    field_pool: dict[str, dict] = dc_field(default_factory=dict)
    global_pool: dict = dc_field(default_factory=dict)

    # ── fitting ───────────────────────────────────────────────────────────────
    def _fit_linear(self, g: pd.DataFrame) -> tuple[float, float, float]:
        q = g["qliq"].to_numpy(float)
        rho = liquid_density(g["watercut"].to_numpy(float))
        y = g["rpump_intake"].to_numpy(float)
        ok = np.isfinite(q) & np.isfinite(rho) & np.isfinite(y)
        if ok.sum() < self.min_days:
            return np.nan, np.nan, np.nan
        q, rho, y = q[ok], rho[ok], y[ok]
        # Design: [1, -q, -(rho-RHO_REF)*g/PA] so coefficients read directly as
        # (a, b, h_eff) and the third column is centred (see RHO_REF).
        col_h = -(rho - RHO_REF) * G / PA_PER_ATM
        X = np.c_[np.ones(len(y)), -q, col_h]
        beta = _irls(X, y)
        a = float(np.clip(beta[0], *A_BOUNDS))
        b = float(np.clip(beta[1], *B_BOUNDS))
        h = float(np.clip(beta[2], *H_EFF_BOUNDS))
        # If any bound bit, the remaining terms no longer balance the level the fit had
        # found — so re-solve the datum with b and h_eff held.  Without this, one clipped
        # slope silently moves every prediction for the well.
        if (a != beta[0]) or (b != beta[1]) or (h != beta[2]):
            a = float(np.clip(np.median(y + b * q - col_h * h), *A_BOUNDS))
        return a, b, h

    def _fit_vogel(self, g: pd.DataFrame) -> tuple[float, float]:
        pr = g["rpl"].to_numpy(float)
        q = g["qliq"].to_numpy(float)
        rho = liquid_density(g["watercut"].to_numpy(float))
        y = g["rpump_intake"].to_numpy(float)
        ok = np.isfinite(pr) & np.isfinite(q) & np.isfinite(rho) & np.isfinite(y)
        if ok.sum() < self.min_days:
            return np.nan, np.nan
        pr, q, rho, y = pr[ok], q[ok], rho[ok], y[ok]
        qpk = float(np.max(q))
        best = None
        for m in (1.02, 1.1, 1.3, 2.0, 4.0, 10.0, 30.0):
            pw = vogel_pwf(pr, q, m * qpk)
            h = float(np.clip(np.median((pw - y) * PA_PER_ATM / (rho * G)), *H_EFF_BOUNDS))
            cost = float(np.nanmedian(np.abs(pw - column_atm(rho, h) - y)))
            if best is None or cost < best[0]:
                best = (cost, m * qpk, h)
        return best[1], best[2]

    def fit(self, df: pd.DataFrame) -> "PhysicalIntakeModel":
        obs = df[df["is_observed"]] if "is_observed" in df else df.dropna(subset=["rpump_intake"])
        obs = obs.dropna(subset=["qliq", "watercut", "rpump_intake"])

        rows = []
        for wk, g in obs.groupby("well_key", sort=False):
            if len(g) < self.min_days:
                continue
            q_max = np.nan
            if self.mode == "vogel":
                q_max, h = self._fit_vogel(g)
                a = b = np.nan
                if not np.isfinite(q_max):
                    continue
                pred = predict_intake_vogel(
                    g["rpl"].to_numpy(), g["qliq"].to_numpy(), g["watercut"].to_numpy(), q_max, h
                )
            else:
                a, b, h = self._fit_linear(g)
                if not np.isfinite(a):
                    continue
                pred = predict_intake_linear(
                    g["qliq"].to_numpy(), g["watercut"].to_numpy(), a, b, h
                )
            mad = float(np.nanmedian(np.abs(pred - g["rpump_intake"].to_numpy())))
            flag = "" if H_EFF_PLAUSIBLE[0] <= h <= H_EFF_PLAUSIBLE[1] else (
                f"h_eff={h:.0f}m outside plausible band"
            )
            self.calib[wk] = WellCalibration(
                well_key=wk, a=a, b=b, h_eff_m=h, q_max=q_max, source="well",
                n_days=len(g), resid_mad=mad, flag=flag,
            )
            rows.append({"well_key": wk, "field": g["field"].iloc[0],
                         "a": a, "b": b, "h_eff": h,
                         "q_max_per_qpk": q_max / max(float(g["qliq"].max()), 1e-9)})

        pooled = pd.DataFrame(rows)
        if len(pooled):
            for fld, sub in pooled.groupby("field"):
                self.field_pool[fld] = {
                    "a": float(sub["a"].median()), "b": float(sub["b"].median()),
                    "h_eff": float(sub["h_eff"].median()),
                    "q_max_per_qpk": float(sub["q_max_per_qpk"].median()),
                }
            self.global_pool = {
                "a": float(pooled["a"].median()), "b": float(pooled["b"].median()),
                "h_eff": float(pooled["h_eff"].median()),
                "q_max_per_qpk": float(pooled["q_max_per_qpk"].median()),
            }
        return self

    # ── prediction ────────────────────────────────────────────────────────────
    def _constants(self, well_key: str, field) -> tuple[dict, str]:
        c = self.calib.get(well_key)
        if c is not None and (np.isfinite(c.a) or np.isfinite(c.q_max)):
            return {"a": c.a, "b": c.b, "h_eff": c.h_eff_m, "q_max": c.q_max}, "well"
        if field in self.field_pool:
            return dict(self.field_pool[field]), "field"
        if self.global_pool:
            return dict(self.global_pool), "global"
        return {}, "none"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted intake pressure (atm), NaN where the form has no inputs."""
        out = np.full(len(df), np.nan)
        for wk, g in df.groupby("well_key", sort=False):
            fld = g["field"].iloc[0] if "field" in g else None
            par, src = self._constants(wk, fld)
            if not par:
                continue
            idx = df.index.get_indexer(g.index)
            if self.mode == "vogel":
                q_max = float(par.get("q_max") or np.nan)
                if not np.isfinite(q_max):
                    # Pooled tier stores q_max as a multiple of the well's own peak rate,
                    # since absolute deliverability is not transferable between wells.
                    q_max = float(par.get("q_max_per_qpk", np.nan)) * float(g["qliq"].max())
                out[idx] = predict_intake_vogel(
                    g["rpl"].to_numpy(), g["qliq"].to_numpy(), g["watercut"].to_numpy(),
                    q_max, par["h_eff"],
                )
            else:
                out[idx] = predict_intake_linear(
                    g["qliq"].to_numpy(), g["watercut"].to_numpy(),
                    par["a"], par["b"], par["h_eff"],
                )
        return np.clip(out, 0.5, 400.0)

    def predict_source(self, df: pd.DataFrame) -> np.ndarray:
        """Which calibration tier each row used — ship it alongside the prediction."""
        out = np.full(len(df), "none", dtype=object)
        for wk, g in df.groupby("well_key", sort=False):
            fld = g["field"].iloc[0] if "field" in g else None
            out[df.index.get_indexer(g.index)] = self._constants(wk, fld)[1]
        return out

    def calibration_table(self) -> pd.DataFrame:
        cols = ["well_key", "a", "b", "h_eff_m", "q_max", "source", "n_days",
                "resid_mad", "flag"]
        if not self.calib:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([vars(c) for c in self.calib.values()])[cols].sort_values("resid_mad")


class WellMedianBaseline:
    """Predict each well's own median observed Рприем; global median for unseen wells.

    Not a strawman — on held-out mid-well blocks this scores **R² +0.411**, ahead of every
    physical form tried (module docstring).  Any model that does not beat it on the warm
    case is not earning its complexity, and the validation report puts it first for that
    reason.
    """

    def __init__(self) -> None:
        self.by_well: dict[str, float] = {}
        self.by_field: dict[str, float] = {}
        self.global_med: float = np.nan

    def fit(self, df: pd.DataFrame) -> "WellMedianBaseline":
        obs = df[df["is_observed"]] if "is_observed" in df else df.dropna(subset=["rpump_intake"])
        self.by_well = obs.groupby("well_key")["rpump_intake"].median().to_dict()
        if "field" in obs:
            self.by_field = obs.groupby("field")["rpump_intake"].median().to_dict()
        self.global_med = float(obs["rpump_intake"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        w = df["well_key"].map(self.by_well)
        if "field" in df:
            w = w.fillna(df["field"].map(self.by_field))
        return w.fillna(self.global_med).to_numpy(float)


__all__ = [
    "PhysicalIntakeModel",
    "WellMedianBaseline",
    "WellCalibration",
    "vogel_pwf",
    "linear_pwf",
    "predict_intake_linear",
    "predict_intake_vogel",
    "liquid_density",
    "column_atm",
    "fit_annulus_h",
    "H_EFF_PLAUSIBLE",
    "H_EFF_BOUNDS",
]
