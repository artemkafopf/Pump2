"""Ya -> Mc transfer test for the discrete-time hazard CatBoost line.

The question
------------
Age-standardised against the like-for-like donor ``Ya_brt``, Мирнинский runs ~23% hot
(SMR 1.23, p=0.099).  Borrowing Ya's *shape* was tested and failed (under-predicts Mc by
20%).  So: **do the strongest covariates explain Mc's excess over Ya?**

The design
----------
Three models, all fitted on ``Ya_brt`` and all asked to predict Mc's observed failures
over Mc's own exposure:

1. ``age_only``   — hazard from age alone.  This is donor shape-borrowing, reproduced
                    inside this framework; it is the thing covariates must beat.
2. ``covariates`` — age + the four transferable covariates, evaluated at *Mc's* covariate
                    values.  If the covariates carry transportable signal, this moves the
                    ratio toward 1.0.
3. ``mc_own``     — the same model fitted on Mc itself.  Not a competitor (it is in-sample
                    on 43 events) but the upper bound on what any covariate model could
                    reach.

Everything runs on **covariate-present runs only**, in both train and test.  That is not a
convenience: covariate missingness predicts high hazard in Ya (Big-only closed runs carry
unrecorded failures) and low hazard in Mc (Big-only open runs are still turning), so a
model that sees the missing rows learns an association whose sign inverts on transfer.
See ``hazard_catboost``'s module docstring.

Reproduce::

    .venv/Scripts/python.exe scripts/run/hazard_catboost_ya_to_mc.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import esp_hazard_fit, esp_population as ep
from analysis.workflows.production_risk import hazard_catboost as hc

AS_OF = "2026-07-01"
SLUG = "production_risk_hazard_catboost_ya_to_mc"


def _rate_on(model: hc.DiscreteHazardModel, pp: pd.DataFrame) -> np.ndarray:
    """Hazard rate for every bin of ``pp``, grouped by the model's own covariate values."""
    out = np.zeros(len(pp), dtype=float)
    cols = list(model.covariates)
    if not cols:
        return model.rate(pp["age_lo"].to_numpy(dtype=float), {})
    pos = {k: i for i, k in enumerate(pp.index)}
    for cov, grp in pp.groupby(cols, dropna=False, sort=False):
        cov_map = dict(zip(cols, cov if isinstance(cov, tuple) else (cov,)))
        idx = [pos[k] for k in grp.index]
        out[idx] = model.rate(grp["age_lo"].to_numpy(dtype=float), cov_map)
    return out


def expected_failures(model: hc.DiscreteHazardModel, pp: pd.DataFrame) -> float:
    """Sum of ``lambda(age, x) * exposure`` over every at-risk bin.

    This is the model's expected event count on exactly the exposure the observed count
    was accrued over, so ``expected / observed`` is a like-for-like calibration ratio.
    """
    return float(np.sum(_rate_on(model, pp) * pp["exposure"].to_numpy(dtype=float)))


def hazard_by_band(pp: pd.DataFrame, model: hc.DiscreteHazardModel | None = None) -> pd.DataFrame:
    """Empirical hazard per age bin, optionally with the model's exposure-weighted fit."""
    out = hc.empirical_hazard(pp)
    if model is None:
        return out
    expected = _rate_on(model, pp) * pp["exposure"].to_numpy(dtype=float)
    by_bin = pd.Series(expected, index=pp.index).groupby(pp["bin"]).sum()
    out["model_fail_per_month"] = [
        30.4 * float(by_bin.get(b, 0.0)) / max(float(d), 1e-6)
        for b, d in zip(out["bin"], out["pump_days"])
    ]
    out["ratio"] = out["model_fail_per_month"] / out["fail_per_month"].replace(0.0, np.nan)
    return out


def main() -> None:
    warnings.simplefilter("ignore", category=FutureWarning)
    out_dir = results_dir(SLUG)

    # Require the loading block too, so every model below is scored on ONE fixed
    # population and the ratios are comparable. It costs almost nothing: Ya 613 -> 603
    # runs, Mc 49 -> 49.
    pop = hc.attach_covariates(
        ep.build(AS_OF),
        required=[*hc.TRANSFERABLE_COVARIATES, *hc.LOADING_COVARIATES],
    )

    ya_all = ep.select(pop, "Ya", "nonsour", "brt")
    mc_all = ep.select(pop, "Mc", "nonsour", "brt")

    coverage = pd.concat(
        [hc.coverage_report(ya_all).assign(field="Ya_brt"),
         hc.coverage_report(mc_all).assign(field="Mc_brt")],
        ignore_index=True,
    )
    coverage.to_csv(out_dir / "tables" / "coverage.csv", index=False)
    print("\n== covariate availability (the transfer trap) ==")
    print(coverage[["field", "covariates_present", "n_runs", "n_events", "fail_per_month"]]
          .to_string(index=False))

    ya = ya_all[ya_all["covariates_present"]].reset_index(drop=True)
    mc = mc_all[mc_all["covariates_present"]].reset_index(drop=True)
    ya_pp = hc.person_period(ya)
    mc_pp = hc.person_period(mc)

    obs_mc = int(mc["event"].sum())
    obs_ya = int(ya["event"].sum())
    print(f"\ntrain Ya_brt: {len(ya)} runs / {obs_ya} events"
          f"   |   test Mc_brt: {len(mc)} runs / {obs_mc} events")

    support = hc.support_overlap(ya, mc)
    support.to_csv(out_dir / "tables" / "support_overlap.csv", index=False)
    print("\n== covariate support: can it transfer at all? ==")
    print(support[["covariate", "train_min", "train_max", "test_min", "test_max",
                   "share_in_train_range", "test_is_constant", "transferable"]]
          .round(3).to_string(index=False))
    usable = support.loc[support["transferable"], "covariate"].tolist()
    print(f"\nusable for transfer: {usable}")

    base = [c for c in usable if c in hc.COVARIATES]
    specs: dict[str, list[str]] = {
        # Donor shape-borrowing reproduced in this framework — the thing covariates must beat.
        "age_only": [],
        # Only covariates that survive the support guard.
        "covariates": base,
        # Deliberately ignores the guard, to show what the importance ranking alone buys you.
        "covariates_naive": list(hc.COVARIATES),
    }
    # The loading block, tested one at a time on top of the surviving base. Kpod is
    # dimensionless so it is comparable across fields by construction; qliq_early_m3d is
    # the un-normalised "Ql directly" comparison against it.
    for col in hc.LOADING_COVARIATES:
        if col in usable:
            specs[f"base+{col}"] = [*base, col]
    if all(c in usable for c in hc.LOADING_COVARIATES):
        specs["base+all_loading"] = [*base, *hc.LOADING_COVARIATES]

    models: dict[str, hc.DiscreteHazardModel] = {
        name: hc.DiscreteHazardModel(covariates=cols).fit(ya_pp) for name, cols in specs.items()
    }
    # Upper bound: in-sample on Mc's own events, not a competitor.
    models["mc_own"] = hc.DiscreteHazardModel(covariates=base).fit(mc_pp)

    rows = []
    for name, model in models.items():
        exp_mc = expected_failures(model, mc_pp)
        exp_ya = expected_failures(model, ya_pp)
        rows.append({
            "model": name,
            "trained_on": "Mc_brt" if name == "mc_own" else "Ya_brt",
            "mc_observed": obs_mc,
            "mc_expected": round(exp_mc, 1),
            "mc_ratio": round(exp_mc / obs_mc, 3),
            "ya_observed": obs_ya,
            "ya_expected": round(exp_ya, 1),
            "ya_ratio": round(exp_ya / obs_ya, 3),
            "in_sample_on_test": name == "mc_own",
        })

    # Parametric baselines fitted on Mc itself, scored the same way: expected failures over
    # Mc's own exposure. B = spike+constant (esp_hazard_fit); A's registry form is the same
    # family with w1 -> 0, so B's fitter covers both and the comparison stays one codepath.
    b = esp_hazard_fit.fit(mc["tte"], mc["event"])
    # Expected events over a bin = the cumulative hazard accrued across it,
    # Lambda(a+e) - Lambda(a) with Lambda = -log S — the same integral the CatBoost line
    # sums as lambda * exposure, so both are scored identically.
    a0 = mc_pp["age_lo"].to_numpy(dtype=float)
    a1 = a0 + mc_pp["exposure"].to_numpy(dtype=float)
    s_args = (b["w1"], b["beta1"], b["eta1"], b["eta2"])
    lam = -np.log(np.clip(esp_hazard_fit.survival(a1, *s_args), 1e-300, None)) \
          + np.log(np.clip(esp_hazard_fit.survival(a0, *s_args), 1e-300, None))
    exp_b = float(np.sum(lam))
    rows.append({
        "model": "spike_constant_B", "trained_on": "Mc_brt",
        "mc_observed": obs_mc, "mc_expected": round(exp_b, 1),
        "mc_ratio": round(exp_b / obs_mc, 3),
        "ya_observed": obs_ya, "ya_expected": np.nan, "ya_ratio": np.nan,
        "in_sample_on_test": True,
    })

    verdict = pd.DataFrame(rows)
    verdict.to_csv(out_dir / "tables" / "verdict_transfer.csv", index=False)
    print("\n== Ya -> Mc transfer: expected vs observed failures on Mc's own exposure ==")
    print(verdict.to_string(index=False))

    # ---- replication: a single target cannot separate signal from a lucky landing ----
    # Every Ya-trained model is scored against every other stratum with enough events. A
    # covariate that carries real physics must move ratios TOWARD 1.0 on all of them; one
    # that merely rescales the hazard will land near 1.0 somewhere by luck and scatter
    # everywhere else.
    rep_rows = []
    for fld in ("Mc", "Vt", "Az", "Za", "Ic", "Da"):
        for contractor in ("brt", "slb"):
            g = ep.select(pop, fld, "nonsour", contractor)
            g = g[g["covariates_present"]].reset_index(drop=True)
            if len(g) < 10 or int(g["event"].sum()) < 8:
                continue
            pp = hc.person_period(g)
            obs = int(g["event"].sum())
            row = {"target": f"{fld}_{contractor}", "n_runs": len(g), "observed": obs}
            for name, model in models.items():
                if name == "mc_own":
                    continue
                row[name] = round(expected_failures(model, pp) / obs, 3)
            rep_rows.append(row)
    replication = pd.DataFrame(rep_rows)
    replication.to_csv(out_dir / "tables" / "replication_by_target.csv", index=False)

    # Summarise each model over the targets: |log ratio| is the scale-symmetric error
    # (2x under and 2x over score the same), and the spread exposes rescaling dressed up
    # as signal.
    summary = []
    for name in models:
        if name == "mc_own" or name not in replication.columns:
            continue
        r = replication[name].astype(float)
        summary.append({
            "model": name,
            "median_ratio": round(float(r.median()), 3),
            "mean_abs_log_ratio": round(float(np.abs(np.log(r)).mean()), 3),
            "min_ratio": round(float(r.min()), 3),
            "max_ratio": round(float(r.max()), 3),
        })
    rep_summary = pd.DataFrame(summary).sort_values("mean_abs_log_ratio")
    rep_summary.to_csv(out_dir / "tables" / "replication_summary.csv", index=False)

    print("\n== replication: every Ya-trained model on every target stratum (ratio) ==")
    print(replication.to_string(index=False))
    print("\n== replication summary (lower mean_abs_log_ratio = better transfer) ==")
    print(rep_summary.to_string(index=False))

    for name, model in models.items():
        hazard_by_band(mc_pp, model).assign(model=name).to_csv(
            out_dir / "tables" / f"mc_hazard_by_band_{name}.csv", index=False
        )
    hazard_by_band(ya_pp).to_csv(out_dir / "tables" / "ya_empirical_hazard.csv", index=False)

    print("\n== Mc empirical hazard vs the Ya-trained covariate model ==")
    band = hazard_by_band(mc_pp, models["covariates"])
    print(band[band["events"] >= 3][
        ["age_lo", "age_hi", "events", "fail_per_month", "model_fail_per_month", "ratio"]
    ].round(4).to_string(index=False))

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
