"""Fit spike + constant hazard on the Vt strata, and audit each against its own
empirical age-banded hazard.

The registry carries a 2-component schema but 19 of its 26 strata are fit as `k1`
(w1 = 0), i.e. a single Weibull — including `Global_Pooled`, `Ya_nonsour_Pooled`
(1192 events) and `Vt_sour_Pooled`.  Those are the ones with the functional defect:
`beta < 1` cannot express "infant spike, then flat", so the MLE compromises —
under-stating the spike and over-stating the plateau across the ages the fleet
actually occupies.  The `k2` strata (`Vt_nonsour_Pooled` among them) already have
the freedom to express it, so they are not expected to improve here — which makes
them the control group for this comparison.

Vt is where this can be tested properly: 667 runs / 316 events, against Мирнинский's
49.  Every fit here is scored two ways —

  * max|dS| against Kaplan-Meier — does the curve match the data;
  * fitted/empirical hazard ratio per age band — does the *shape* match,

the second being the check the registry has never had.  A curve can pass KM on
aggregate while getting the age profile wrong, which is exactly the shipped failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter, WeibullFitter  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_hazard_fit as HF  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402
from analysis.workflows.production_risk.survival import StrataModel  # noqa: E402

AS_OF = "2026-07-16"
KM_BAR = 0.05
AGE_BANDS = [(0, 30), (30, 90), (90, 180), (180, 300), (300, 450), (450, 600), (600, 900)]


def weibull_survival(t, beta: float, eta: float) -> np.ndarray:
    return np.exp(-((np.maximum(np.asarray(t, float), 0.0) / eta) ** beta))


def max_ks(g: pd.DataFrame, s_fn) -> float:
    km = KaplanMeierFitter().fit(g["tte"], g["event"])
    t = km.survival_function_.index.values
    return float(np.max(np.abs(km.survival_function_.values.ravel() - s_fn(t))))


def empirical_bands(g: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """band -> (empirical monthly hazard, pump-months of exposure)."""
    out = {}
    for a, b in AGE_BANDS:
        exposure = np.clip(np.minimum(g["tte"], b) - a, 0, None).sum() / 30.4
        events = ((g["tte"] >= a) & (g["tte"] < b) & (g["event"] == 1)).sum()
        out[f"{a}-{b}"] = (events / exposure if exposure > 2 else np.nan, exposure)
    return out


def model_band_hazard(s_fn, a: float, b: float) -> float:
    """Model's average monthly hazard over [a, b] — same estimand as the empirical
    count/exposure ratio, so the two are directly comparable."""
    grid = np.arange(a, b, 1.0)
    s = s_fn(grid)
    s_next = s_fn(grid + 1.0)
    daily_h = 1.0 - s_next / np.clip(s, 1e-12, None)
    # exposure-weight by the model's own survival, mirroring pump-days at risk
    w = s / max(s.sum(), 1e-12)
    return float(np.sum(w * daily_h) * 30.4)


def main() -> None:
    out = results_dir("production_risk_vt_hazard_fit")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pop = P.build(AS_OF)
    vt = pop[pop["field"] == "Vt"].copy()
    registry = StrataModel()

    groups: dict[str, pd.DataFrame] = {}
    for sour in ("sour", "nonsour"):
        g = vt[vt["h2s_class"] == sour]
        groups[f"Vt_{sour}_Pooled"] = g
        for ctr in ("brt", "slb", "oth"):
            gg = g[g["contractor_group"] == ctr]
            if gg["event"].sum() >= 10:
                groups[f"Vt_{sour}_{ctr}"] = gg

    fit_rows, audit_rows = [], []
    for name, g in groups.items():
        spike = HF.fit(g["tte"].values, g["event"].values)
        sp_params = HF.registry_params(spike)
        sp_s = lambda t, p=sp_params: HF.survival(t, p["w1"], p["beta1"], p["eta1"], p["eta2"])

        wf = WeibullFitter().fit(g["tte"], g["event"])
        wb_beta, wb_eta = float(wf.rho_), float(wf.lambda_)
        wb_s = lambda t, b=wb_beta, e=wb_eta: weibull_survival(t, b, e)

        _, ctr = name.split("_")[1], name.split("_")[2]
        ship, ship_key = registry.resolve("Vt", name.split("_")[1], ctr if ctr != "Pooled" else "missing")
        ship_s = lambda t, p=ship: np.asarray(registry.S(t, p), dtype=float)

        fit_rows.append({
            "stratum": name, "runs": len(g), "events": int(g["event"].sum()),
            **{k: round(v, 4) for k, v in sp_params.items()},
            "spike_b50": HF.quantile(0.5, sp_params),
            "weibull_beta": round(wb_beta, 4), "weibull_eta": round(wb_eta, 1),
            "weibull_b50": round(wb_eta * np.log(2) ** (1 / wb_beta), 1),
            "ks_spike": round(max_ks(g, sp_s), 4),
            "ks_weibull_refit": round(max_ks(g, wb_s), 4),
            "ks_shipped": round(max_ks(g, ship_s), 4),
            "shipped_key": ship_key,
            "shipped_b50": round(float(ship.get("b50", np.nan)), 1),
        })

        emp = empirical_bands(g)
        for (a, b) in AGE_BANDS:
            e_h, exposure = emp[f"{a}-{b}"]
            if not np.isfinite(e_h):
                continue
            audit_rows.append({
                "stratum": name, "band": f"{a}-{b}", "pump_months": round(exposure, 1),
                "empirical": round(e_h, 4),
                "spike": round(model_band_hazard(sp_s, a, b), 4),
                "shipped": round(model_band_hazard(ship_s, a, b), 4),
                "spike_ratio": round(model_band_hazard(sp_s, a, b) / e_h, 2),
                "shipped_ratio": round(model_band_hazard(ship_s, a, b) / e_h, 2),
            })

    fits = pd.DataFrame(fit_rows)
    audit = pd.DataFrame(audit_rows)
    fits.to_csv(tables / "vt_fit_params.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(tables / "vt_hazard_audit.csv", index=False, encoding="utf-8-sig")

    summary = (
        audit.groupby("stratum")
        .apply(lambda d: pd.Series({
            "spike_median_ratio": round(d["spike_ratio"].median(), 2),
            "spike_worst_ratio": round(max(d["spike_ratio"].max(), 1 / d["spike_ratio"].min()), 2),
            "shipped_median_ratio": round(d["shipped_ratio"].median(), 2),
            "shipped_worst_ratio": round(max(d["shipped_ratio"].max(), 1 / d["shipped_ratio"].min()), 2),
        }), include_groups=False)
        .reset_index()
    )
    summary.to_csv(tables / "vt_shape_summary.csv", index=False, encoding="utf-8-sig")

    print("=== fits (ks bar = %.2f) ===" % KM_BAR)
    print(fits[["stratum", "runs", "events", "w1", "beta1", "eta1", "eta2",
                "spike_b50", "shipped_b50", "ks_spike", "ks_weibull_refit",
                "ks_shipped"]].to_string(index=False))
    print("\n=== shape audit: model/empirical hazard ratio (1.0 = right) ===")
    print(summary.to_string(index=False))
    print("\n=== per-band detail ===")
    print(audit.to_string(index=False))
    print(f"\nwrote {tables}")


if __name__ == "__main__":
    main()
