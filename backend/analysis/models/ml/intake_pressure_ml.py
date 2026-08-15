"""CatBoost realizations of Рприем, and the physics-informed variants of them.

Three models, one interface, deliberately different in what they are allowed to know:

``IntakeMLModel(variant="plain")``
    Gradient boosting on the gap-safe daily + equipment features.  No physics.
``IntakeMLModel(variant="physics")``
    Same, plus the physical model's intermediate quantities as features — the standing
    column, the annular head, the linearised-IPR prediction itself.  *Physics-informed
    features*: the tree is told what the engineering thinks and may overrule it.
``IntakeMLModel(variant="residual")``
    Boosting on the **residual** of :class:`~analysis.features.intake_pressure.
    PhysicalIntakeModel`, prediction added back.  *Physics-informed target*: the physical
    model owns the level and the tree only learns what it got wrong.

The two hybrid forms are not interchangeable and the difference matters when the physical
model is extrapolating.  ``physics`` can ignore a bad physical feature; ``residual``
cannot — a wrong physical baseline is added back in full.

Measured: physics-informing is an honest null, and a liability on cold wells
----------------------------------------------------------------------------
==============  ==============  ==============
variant         warm MAE / R²   cold MAE / R²
==============  ==============  ==============
``plain``       13.61 / +0.675  15.81 / +0.597
``physics``     13.73 / +0.665  19.52 / +0.503
``residual``    13.85 / +0.659  19.37 / +0.502
==============  ==============  ==============

**``plain`` wins both regimes.**  On the warm case the three are within 2 % of each other
— a null.  On the **cold** case the physics variants are **23 % worse**, and the mechanism
is the one the design anticipated: for an unseen well the physical model falls to
field-pooled constants, and both hybrids inherit that.  ``residual`` adds the bad baseline
back in full by construction; ``physics`` is nominally free to ignore the feature but does
not, because ``phys_pred_atm`` is its single most important input (importance 21.8, ahead
of ``rpl`` at 13.4) — it learned to trust a feature that is reliable in training, where
every well is calibrated, and is not reliable on a well that has never been seen.

That is a covariate-shift trap of the same family as the ``rzab`` one below, just one
level up: the *feature's meaning* changes between fit and apply, not its availability.
**Use ``plain`` for imputation.**  The physics variants are retained because they make the
null reproducible and because ``phys_pred_atm``'s importance is itself the evidence that
the physical arm carries real information — it just does not survive transfer to a new
well.

The feature rule that this project has already been bitten by
--------------------------------------------------------------
Every predictor here is checked for availability **inside the gap**, not just in training.
``rzab`` is the specific trap: present on 93.3 % of days where Рприем is observed, on
**10.3 %** of days where it is missing, and correlated 0.933 with the target because it is
*calculated from it* («Расчетное забойное давление»).  A model given Рзаб posts an
excellent cross-validation score and then meets a gap that does not have it.  It is
excluded, and :data:`FORBIDDEN` exists so it cannot be reintroduced by accident.

Well-history features and the warm/cold split
----------------------------------------------
``fit`` computes per-well aggregates of the target (median, IQR, slope) **from the
training rows only** and ``predict`` maps them on.  For a *warm* well — one with observed
history, which is 15 of the 17 structurally-incomplete wells — that hands the model the
single strongest predictor there is (the well-median baseline scores R² +0.411 alone).
For a *cold* well the map misses and the features arrive as NaN, which CatBoost routes as
its own split direction; the model then falls back on equipment and field.

This is why the same object serves both regimes, and why
:mod:`analysis.workflows.intake_pressure.validate` must score them separately: a single
pooled number would average a nearly-solved problem with a genuinely hard one and describe
neither.

⚠ Because the well aggregates are computed in ``fit``, a fold split must be by **well**
(cold) or by **contiguous block within well** (warm).  A random row split lets day *t−1*
inform day *t* through both the aggregates and the tree, and reports a number that has
nothing to do with deployment.  The validation module demonstrates the size of that lie
rather than merely warning about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from analysis.features.intake_pressure import (
    G,
    PA_PER_ATM,
    RHO_REF,
    PhysicalIntakeModel,
    liquid_density,
)

#: Columns that must never enter the feature set, with the reason.  Enforced in
#: :meth:`IntakeMLModel.feature_names`.
FORBIDDEN = {
    "rzab": "calculated FROM the target and co-missing with it (10.3 % present in the gap)",
    "rpump_intake": "the target",
    "is_observed": "the missingness mask",
    "qgas": "derived from qliq x gas_factor; adds no independent information",
}

#: Daily predictors, all measured >=90 % inside the gap.
DAILY_FEATURES = [
    "rpl", "qliq", "watercut", "gas_factor", "freq", "load", "kprod",
    "qoil", "qwater", "rho_mix_kgm3", "kpod", "op_day", "month",
]

#: Static per-run equipment.
EQUIP_FEATURES = [
    "vg_m", "pump_depth_m", "q_nom_m3d", "head_nom_m", "stages", "pump_od_mm",
    "pbubble_atm",
]

#: Physics-derived features for ``variant="physics"``.
PHYS_FEATURES = ["p_static_atm", "p_annulus_atm", "phys_pred_atm", "p_head_nodal_atm"]

#: Per-well aggregates of the target, computed on training rows only.
HIST_FEATURES = ["well_hist_median", "well_hist_iqr", "well_hist_n", "well_hist_slope"]

CAT_FEATURES = ["field"]

DEFAULT_PARAMS = dict(
    iterations=600,
    depth=6,
    learning_rate=0.06,
    loss_function="MAE",
    random_seed=0,
    verbose=False,
    allow_writing_files=False,
)


def add_physics_features(df: pd.DataFrame, phys: PhysicalIntakeModel | None) -> pd.DataFrame:
    """Attach the physical model's intermediate quantities as columns.

    ``p_static_atm``
        Standing liquid column at the pump's true vertical depth — the scale the intake
        pressure is a deviation from.
    ``p_annulus_atm``
        The fitted annular head for that well, the term that carries daily water-cut
        variation.
    ``phys_pred_atm``
        The physical model's own prediction.  For ``variant="physics"`` this is a feature
        the tree may overrule; for ``variant="residual"`` it is the baseline.
    ``p_head_nodal_atm``
        Affinity-scaled catalog pump head.  Weak on its own (rejection 4 in the physics
        module: cross-well r=+0.20) but it is the only term carrying ``freq``, which
        survives on 99.9 % of the gap, so the tree is given the chance to use it.
    """
    out = df.copy()
    rho = liquid_density(out["watercut"].to_numpy(float))
    out["p_static_atm"] = rho * G * out["vg_m"].to_numpy(float) / PA_PER_ATM

    fr = np.clip(out["freq"].to_numpy(float) / 50.0, 0.2, None)
    x = np.clip(
        out["qliq"].to_numpy(float) / (out["q_nom_m3d"].to_numpy(float) * fr), 0.0, 2.0
    )
    # Normalised ESP head curve, linear-declining over the operating range: shut-in head
    # is ~1.4x BEP head, hence phi(x) = 1.4 - 0.4x with phi(1) = 1.
    h_pump = out["head_nom_m"].to_numpy(float) * fr**2 * (1.4 - 0.4 * x)
    out["p_head_nodal_atm"] = rho * G * h_pump / PA_PER_ATM

    if phys is not None:
        out["phys_pred_atm"] = phys.predict(out)
        h_eff = out["well_key"].map(
            {k: v.h_eff_m for k, v in phys.calib.items()}
        ).to_numpy(float)
        # Centred on RHO_REF, matching how h_eff is identified in the physical fit — the
        # uncentred product would misstate the term by RHO_REF*h_eff, which varies by well.
        out["p_annulus_atm"] = (rho - RHO_REF) * G * h_eff / PA_PER_ATM
    else:
        out["phys_pred_atm"] = np.nan
        out["p_annulus_atm"] = np.nan
    return out


@dataclass
class IntakeMLModel:
    """CatBoost regressor for Рприем with optional physics coupling.

    Parameters
    ----------
    variant
        ``"plain"``, ``"physics"`` or ``"residual"`` — see module docstring.
    use_history
        Include the per-well target aggregates.  True is correct for deployment (the gap
        wells mostly *do* have history); set False to measure the cold-well case honestly
        without relying on the fold split to enforce it.
    params
        CatBoost overrides.  ``loss_function="MAE"`` is the default because the target
        carries surviving telemetry spikes and squared error would chase them; the
        validation report leads with MAE for the same reason.
    """

    variant: str = "plain"
    use_history: bool = True
    params: dict = dc_field(default_factory=lambda: dict(DEFAULT_PARAMS))
    model: object | None = None
    phys: PhysicalIntakeModel | None = None
    hist: pd.DataFrame | None = None
    _feats: list[str] = dc_field(default_factory=list)

    # ── features ──────────────────────────────────────────────────────────────
    def feature_names(self) -> list[str]:
        feats = list(DAILY_FEATURES) + list(EQUIP_FEATURES)
        if self.variant in ("physics", "residual"):
            feats += list(PHYS_FEATURES)
        if self.use_history:
            feats += list(HIST_FEATURES)
        bad = [f for f in feats if f in FORBIDDEN]
        if bad:
            raise ValueError(
                "forbidden feature(s) in the set: "
                + "; ".join(f"{b} ({FORBIDDEN[b]})" for b in bad)
            )
        return feats + list(CAT_FEATURES)

    def _well_history(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Per-well target aggregates from the training rows only."""
        def slope(g):
            if len(g) < 10:
                return np.nan
            t = g["op_day"].to_numpy(float)
            y = g["rpump_intake"].to_numpy(float)
            ok = np.isfinite(t) & np.isfinite(y)
            if ok.sum() < 10 or np.ptp(t[ok]) < 1:
                return np.nan
            return float(np.polyfit(t[ok], y[ok], 1)[0])

        g = obs.groupby("well_key")
        out = pd.DataFrame({
            "well_hist_median": g["rpump_intake"].median(),
            "well_hist_iqr": g["rpump_intake"].quantile(0.75) - g["rpump_intake"].quantile(0.25),
            "well_hist_n": g["rpump_intake"].size(),
        })
        out["well_hist_slope"] = g.apply(slope, include_groups=False)
        return out

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy up front: the padding loop below assigns missing columns, and with
        # variant="plain"/use_history=False nothing else would have copied, so the
        # caller's frame would be mutated.
        X = df.copy()
        if self.variant in ("physics", "residual"):
            X = add_physics_features(X, self.phys)
        if self.use_history:
            hist = self.hist if self.hist is not None else pd.DataFrame()
            X = X.join(hist, on="well_key") if len(hist) else X.assign(
                **{c: np.nan for c in HIST_FEATURES}
            )
            for c in HIST_FEATURES:
                if c not in X:
                    X[c] = np.nan
        feats = self.feature_names()
        for c in feats:
            if c not in X:
                X[c] = np.nan
        D = X[feats].copy()
        for c in CAT_FEATURES:
            D[c] = D[c].astype(str).fillna("NA")
        return D

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "IntakeMLModel":
        from catboost import CatBoostRegressor, Pool

        obs = df[df["is_observed"]].dropna(subset=["rpump_intake"])
        if self.variant in ("physics", "residual"):
            self.phys = PhysicalIntakeModel().fit(obs)
        if self.use_history:
            self.hist = self._well_history(obs)

        D = self._design(obs)
        y = obs["rpump_intake"].to_numpy(float)
        if self.variant == "residual":
            base = self.phys.predict(obs)
            # A NaN baseline would delete the row; fall back to the pooled level so the
            # residual stays defined and the row keeps contributing.
            base = np.where(np.isfinite(base), base, np.nanmedian(y))
            self._base_fallback = float(np.nanmedian(y))
            y = y - base

        self._feats = list(D.columns)
        pool = Pool(D, y, cat_features=CAT_FEATURES)
        self.model = CatBoostRegressor(**self.params).fit(pool)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        if self.model is None:
            raise RuntimeError("fit() first")
        D = self._design(df)[self._feats]
        pred = np.asarray(self.model.predict(Pool(D, cat_features=CAT_FEATURES)), float)
        if self.variant == "residual":
            base = self.phys.predict(df)
            base = np.where(np.isfinite(base), base, getattr(self, "_base_fallback", np.nan))
            pred = pred + base
        return np.clip(pred, 0.5, 400.0)

    def importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("fit() first")
        return (
            pd.Series(self.model.get_feature_importance(), index=self._feats)
            .sort_values(ascending=False)
        )


__all__ = [
    "IntakeMLModel",
    "add_physics_features",
    "FORBIDDEN",
    "DAILY_FEATURES",
    "EQUIP_FEATURES",
    "PHYS_FEATURES",
    "HIST_FEATURES",
]
