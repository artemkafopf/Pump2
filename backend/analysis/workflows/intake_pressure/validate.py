"""Score the Рприем realizations under the two regimes deployment actually presents.

Why one number would be a lie
-----------------------------
The missing days split into two populations that a pooled score would average into
nonsense:

**Warm** — the well has observed Рприем either side of the gap.  15 of the 17
structurally-incomplete wells are like this, and so are all 1 383 short dropout blocks.
The well's own median already scores R² +0.411 here; the question is whether anything
beats it.

**Cold** — the well has *no* observed Рприем at all (``vt_5608`` in the shipped panel,
``au_315`` too before the 30-day minimum drops it, and every future new well).  Nothing
well-specific exists, the physical model falls to field-pooled constants and the ML
history features arrive as NaN.  This is a different and much harder problem, and it is
the only one where cross-well structure has to carry the answer.  Because two wells is far
too small a sample to score on, the cold regime is measured by leaving *whole wells* out
across the whole fleet — every well takes a turn at being cold.

:func:`evaluate` scores both, plus a deliberately-wrong **random row split** whose only
purpose is to show how large the leakage is if folds are not grouped — consecutive days
are near-identical, so a random split trains on day *t−1* and tests on day *t*.

Measured (2026-08-05, 760 wells)
---------------------------------
=================  ==============  ==============  =====================
model              warm MAE / R²   cold MAE / R²   random split MAE / R²
=================  ==============  ==============  =====================
``cb_plain``       13.61 / +0.675  15.81 / +0.597  9.87 / +0.800
``cb_physics``     13.73 / +0.665  19.52 / +0.503  9.74 / +0.802
``cb_residual``    13.85 / +0.659  19.37 / +0.502  9.82 / +0.800
``well_median``    17.89 / +0.421  25.92 / −0.079  17.20 / +0.432
``phys_linear``    18.22 / +0.403  26.27 / −0.034  16.65 / +0.457
``phys_vogel``     29.35 / −0.289  45.59 / −1.445  26.63 / −0.100
=================  ==============  ==============  =====================

(Warm scores 219 730 held-out days, cold 558 456 over 5 well-grouped folds, random split
167 554.)

Three things to read off it:

* **The leakage is worth 28 % of MAE.**  The best model looks like 9.87 atm on a random
  split and is 13.61 on the honest warm one.  An ungrouped split would have invented that
  gap and no amount of care elsewhere would have caught it.
* **Cold is a different problem, not a harder version of the same one.**  ``well_median``
  and ``phys_linear`` both go to R² ≈ 0 there — with no well history, the well-level
  constant they are built on does not exist.  Only the boosted models retain skill
  (+0.597), and they get it from equipment, field and daily covariates.
* **The baseline is the bar and it is a high one.**  On the warm case only the boosted
  models clear it, and the physical arm does not.

The masking design
------------------
Warm folds hide a *contiguous* middle block of each well rather than scattered days.
Scattered holdout is the same leakage in a subtler dress: with a 1-day gap the neighbours
pin the answer, and the resulting score describes interpolation while the real gaps run
30-1442 days.  :data:`WARM_BLOCK_FRAC` sets the hidden share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.intake_pressure import PhysicalIntakeModel, WellMedianBaseline
from analysis.models.ml.intake_pressure_ml import IntakeMLModel
from analysis.workflows.intake_pressure.data import split_groups

#: Share of each well's observed span hidden as one contiguous block in the warm regime.
#: 0.4 puts the holdout on the same order as the real structural gaps (which run 30-1442
#: days against well spans of 100-2000).
WARM_BLOCK_FRAC = 0.40

#: Wells need at least this many observed days to be worth masking — below it the
#: "training" ends are too short to calibrate anything and the fold measures noise.
MIN_WELL_DAYS = 120


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return {"n": int(m.sum()), "mae": np.nan, "med_ae": np.nan, "rmse": np.nan,
                "r2": np.nan, "p90_ae": np.nan, "bias": np.nan}
    e = p[m] - y[m]
    ss = np.sum((y[m] - y[m].mean()) ** 2)
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(e))),
        "med_ae": float(np.median(np.abs(e))),
        "p90_ae": float(np.percentile(np.abs(e), 90)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "r2": float(1 - np.sum(e**2) / ss) if ss > 0 else np.nan,
    }


def _models(use_history: bool) -> dict:
    """The candidate set.  ``use_history`` is forced off for the cold regime, where a
    well-history feature is by construction unavailable."""
    return {
        "well_median": lambda: WellMedianBaseline(),
        "phys_linear": lambda: PhysicalIntakeModel(mode="linear"),
        "phys_vogel": lambda: PhysicalIntakeModel(mode="vogel"),
        "cb_plain": lambda: IntakeMLModel(variant="plain", use_history=use_history),
        "cb_physics": lambda: IntakeMLModel(variant="physics", use_history=use_history),
        "cb_residual": lambda: IntakeMLModel(variant="residual", use_history=use_history),
    }


def warm_split(df: pd.DataFrame, block_frac: float = WARM_BLOCK_FRAC) -> pd.Series:
    """Boolean mask: True where the row is hidden as part of its well's middle block."""
    hide = pd.Series(False, index=df.index)
    obs = df[df["is_observed"]]
    for _, g in obs.groupby("well_key", sort=False):
        n = len(g)
        if n < MIN_WELL_DAYS:
            continue
        g = g.sort_values("dt")
        lo = int(n * (1 - block_frac) / 2)
        hi = lo + int(n * block_frac)
        hide.loc[g.index[lo:hi]] = True
    return hide


def evaluate_warm(df: pd.DataFrame, block_frac: float = WARM_BLOCK_FRAC) -> pd.DataFrame:
    """Hide a contiguous middle block per well; train on the ends, score on the block."""
    hide = warm_split(df, block_frac)
    train = df[df["is_observed"] & ~hide].copy()
    test = df[hide].copy()
    y = test["rpump_intake"].to_numpy(float)

    rows = []
    for name, ctor in _models(use_history=True).items():
        m = ctor().fit(train)
        rows.append({"model": name, "regime": "warm", **_metrics(y, m.predict(test))})
    return pd.DataFrame(rows)


def evaluate_cold(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Leave-whole-wells-out: the test wells are unseen, so no history exists for them.

    History features are disabled outright rather than left to come back NaN — the fold
    split already guarantees they would, and turning them off makes the intent explicit
    and the run cheaper.
    """
    obs = df[df["is_observed"]].copy()
    # Balanced by observed-day count: well spans run 30-2000 days, so an unweighted random
    # assignment leaves folds of very different weight and the fold-to-fold scatter then
    # describes the split rather than the model.
    obs["_fold"] = split_groups(obs, n_folds=n_folds, seed=seed)

    preds: dict[str, list] = {k: [] for k in _models(use_history=False)}
    truth = []
    for k in range(n_folds):
        tr = obs[obs["_fold"] != k]
        te = obs[obs["_fold"] == k]
        if len(te) < 50 or len(tr) < 500:
            continue
        truth.append(te["rpump_intake"].to_numpy(float))
        for name, ctor in _models(use_history=False).items():
            preds[name].append(ctor().fit(tr).predict(te))

    y = np.concatenate(truth)
    return pd.DataFrame(
        [{"model": n, "regime": "cold", **_metrics(y, np.concatenate(p))}
         for n, p in preds.items()]
    )


def evaluate_random_leak(df: pd.DataFrame, frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """The dishonest split, reported so the size of the lie is on the record.

    Random rows to test, everything else to train.  Day *t−1* trains, day *t* tests; the
    well-history aggregates are computed over rows adjacent to the held-out ones.  Read the
    gap between this and :func:`evaluate_warm` as the amount of "accuracy" an ungrouped
    split invents.
    """
    rng = np.random.default_rng(seed)
    obs = df[df["is_observed"]].copy()
    hide = rng.random(len(obs)) < frac
    tr, te = obs[~hide], obs[hide]
    y = te["rpump_intake"].to_numpy(float)
    rows = []
    for name, ctor in _models(use_history=True).items():
        m = ctor().fit(tr)
        rows.append({"model": name, "regime": "random_LEAKY", **_metrics(y, m.predict(te))})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, include_leak_demo: bool = True) -> pd.DataFrame:
    """All regimes, one table, ordered so warm/cold sit next to each other per model."""
    parts = [evaluate_warm(df), evaluate_cold(df)]
    if include_leak_demo:
        parts.append(evaluate_random_leak(df))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["regime", "mae"]).reset_index(drop=True)


__all__ = [
    "evaluate",
    "evaluate_warm",
    "evaluate_cold",
    "evaluate_random_leak",
    "warm_split",
    "WARM_BLOCK_FRAC",
    "MIN_WELL_DAYS",
]
