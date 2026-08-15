"""Age-only hazard vs the shipped Weibull (A) and spike+constant (B), per stratum.

Why this and not the covariate line
-----------------------------------
The Ya->Mc covariate transfer is a settled null, and worse, it is *unanswerable* on this
data: covariates exist only for Свод runs, availability is confounded with hazard in
opposite directions per field, and the covariate-present subset represents neither field
(see ``hazard_catboost``'s module docstring and the Ya->Mc findings).

An **age-only** hazard model needs no covariates, so it runs on the **full corrected
population** with no selection at all — the trap is structurally absent.  And it targets
the one defect that survived the 2026-07-17 population fix: **the infant spike**.  The
shipped Weibull delivers ~0.78 of the empirical 0-30 d hazard because a 5-column
parametric schema can express a day-0 point mass *or* an infant window, not both.  A
hazard model with age as a feature has no such constraint.

What is scored
--------------
Per stratum, three fits on identical data:

* ``A``  — the shipped registry row (``esp_models.csv``), resolved as production resolves it.
* ``B``  — ``esp_hazard_fit`` spike+constant, refitted here.
* ``cb`` — age-only Poisson hazard, reported **in-sample** (representational capacity: can
  the form express the shape at all?) and **well-clustered cross-fit** (honest: does it
  generalise, or is it fitting band noise?).

Headline metric is the **0-30 d ratio** (model / empirical hazard in the infant band).
``overall`` (expected/observed failures) guards against buying the spike with a level error.

Reproduce::

    .venv/Scripts/python.exe scripts/run/hazard_age_only.py
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
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_hazard_fit, esp_population as ep
from analysis.workflows.production_risk import hazard_catboost as hc
from analysis.workflows.production_risk.survival import StrataModel

AS_OF = "2026-07-01"
SLUG = "production_risk_hazard_age_only"
MIN_EVENTS = 20
INFANT_MAX_D = 30.0
N_FOLDS = 5


def _weibull_expected(pp: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    """Expected events per bin under a registry-shaped survival curve.

    Events over [a, a+e] = Lambda(a+e) - Lambda(a) with Lambda = -log S — the same
    integral the CatBoost line sums as ``lambda * exposure``, so the two are scored
    identically rather than by two different conventions.
    """
    a0 = pp["age_lo"].to_numpy(dtype=float)
    a1 = a0 + pp["exposure"].to_numpy(dtype=float)
    s0 = np.clip(StrataModel.S(a0, params), 1e-300, None)
    s1 = np.clip(StrataModel.S(a1, params), 1e-300, None)
    return np.log(s0) - np.log(s1)


def _cb_expected(model: hc.DiscreteHazardModel, pp: pd.DataFrame) -> np.ndarray:
    return model.rate(pp["age_lo"].to_numpy(dtype=float), {}) * pp["exposure"].to_numpy(dtype=float)


def _cross_fit_expected(pop: pd.DataFrame, pp: pd.DataFrame, seed: int = 7) -> np.ndarray:
    """Well-clustered cross-fit expected events: every run scored by a model blind to it.

    Clustering on ``code`` (not on the run) matters — a well's runs share a hazard, so
    letting them span folds leaks.
    """
    from sklearn.model_selection import GroupKFold

    codes = pop["code"].astype(str).to_numpy()
    uniq = np.unique(codes)
    k = max(2, min(N_FOLDS, len(uniq)))
    out = np.zeros(len(pp), dtype=float)
    gkf = GroupKFold(n_splits=k)
    run_code = codes[pp["_run"].to_numpy()]
    for tr_idx, _te_idx in gkf.split(uniq, groups=uniq):
        tr_codes = set(uniq[tr_idx])
        tr_mask = np.isin(codes, list(tr_codes))
        te_bins = ~np.isin(run_code, list(tr_codes))
        if not te_bins.any():
            continue
        tr_pp = hc.person_period(pop[tr_mask].reset_index(drop=True))
        if int(tr_pp["y"].sum()) < 3:
            continue
        model = hc.DiscreteHazardModel(covariates=[], random_seed=seed).fit(tr_pp)
        out[te_bins] = _cb_expected(model, pp[te_bins])
    return out


def _band_ratio(pp: pd.DataFrame, expected: np.ndarray, max_age: float) -> float:
    """Model / empirical hazard over ages below ``max_age``, exposure-weighted."""
    m = pp["age_lo"].to_numpy(dtype=float) < max_age
    obs = float(pp["y"].to_numpy()[m].sum())
    if obs <= 0:
        return float("nan")
    return float(expected[m].sum() / obs)


def main() -> None:
    warnings.simplefilter("ignore", category=FutureWarning)
    out_dir = results_dir(SLUG)

    pop_all = ep.build(AS_OF)
    registry = StrataModel()

    rows, band_rows = [], []
    for (fld, h2s, ctr), g in pop_all.groupby(["field", "h2s_class", "contractor_group"], sort=True):
        g = g.reset_index(drop=True)
        if int(g["event"].sum()) < MIN_EVENTS:
            continue
        pp = hc.person_period(g)
        if pp.empty or int(pp["y"].sum()) < MIN_EVENTS:
            continue

        ctr_key = {"brt": "brt", "slb": "slb", "oth": "oth"}.get(ctr, "Pooled")
        params, resolved = registry.resolve(fld, h2s, ctr_key)

        expected = {"A": _weibull_expected(pp, params)}
        try:
            b = esp_hazard_fit.fit(g["tte"], g["event"])
            expected["B"] = _weibull_expected(pp, b)
        except Exception as exc:  # a stratum B cannot fit is reportable, not fatal
            print(f"  ! B failed on {fld}_{h2s}_{ctr}: {exc}")
            expected["B"] = np.full(len(pp), np.nan)
        cb = hc.DiscreteHazardModel(covariates=[], random_seed=7).fit(pp)
        expected["cb_insample"] = _cb_expected(cb, pp)
        expected["cb_xfit"] = _cross_fit_expected(g, pp)

        obs = float(pp["y"].sum())
        row = {
            "stratum": f"{fld}_{h2s}_{ctr}",
            "resolved_registry_row": resolved,
            "n_runs": int(len(g)),
            "n_events": int(obs),
        }
        for name, exp in expected.items():
            row[f"infant_{name}"] = round(_band_ratio(pp, exp, INFANT_MAX_D), 3)
            row[f"overall_{name}"] = round(float(np.nansum(exp) / obs), 3)
        rows.append(row)

        emp = hc.empirical_hazard(pp)
        for name, exp in expected.items():
            by_bin = pd.Series(exp, index=pp.index).groupby(pp["bin"]).sum()
            e = emp.copy()
            e["model"] = name
            e["stratum"] = row["stratum"]
            e["model_fail_per_month"] = [
                30.4 * float(by_bin.get(bi, 0.0)) / max(float(d), 1e-6)
                for bi, d in zip(e["bin"], e["pump_days"])
            ]
            e["ratio"] = e["model_fail_per_month"] / e["fail_per_month"].replace(0.0, np.nan)
            band_rows.append(e)

    verdict = pd.DataFrame(rows)
    verdict.to_csv(out_dir / "tables" / "age_only_verdict.csv", index=False)
    pd.concat(band_rows, ignore_index=True).to_csv(out_dir / "tables" / "hazard_by_band.csv", index=False)

    infant = [c for c in verdict.columns if c.startswith("infant_")]
    overall = [c for c in verdict.columns if c.startswith("overall_")]

    print("\n== 0-30 d infant band: model / empirical (1.0 = right) ==")
    print(verdict[["stratum", "n_runs", "n_events", *infant]].to_string(index=False))
    print("\n== overall: expected / observed failures (1.0 = right) ==")
    print(verdict[["stratum", "n_events", *overall]].to_string(index=False))

    summary = pd.DataFrame([
        {
            "metric": col,
            "median": round(float(verdict[col].median()), 3),
            "mean_abs_log": round(float(np.abs(np.log(verdict[col].replace(0, np.nan))).mean()), 3),
            "min": round(float(verdict[col].min()), 3),
            "max": round(float(verdict[col].max()), 3),
            "n_strata_within_20pct": int(((verdict[col] >= 0.8) & (verdict[col] <= 1.25)).sum()),
        }
        for col in [*infant, *overall]
    ])
    summary.to_csv(out_dir / "tables" / "age_only_summary.csv", index=False)
    print("\n== summary across strata ==")
    print(summary.to_string(index=False))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
