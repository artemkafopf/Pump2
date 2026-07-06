"""Phase B B5 — Parametric planning layer (output slug: phase_b_cause_weibull).

Cause-specific Weibull per field-level stratum (K=1 default; ΔAIC vs K=2 reported
where a cause-stratum has ≥40 events), converted to **CIF-consistent** planning
quantities: the operating time to 10% / 25% cumulative incidence per mode per
stratum, with well-bootstrap CIs.

Critical modelling point (stated in the outputs): the per-mode parametric
survival S_k(t) is **not** 1−CIF_k.  The CIF combines the cause-specific hazards
through the *all-cause* survival:

    CIF_k(t) = ∫₀ᵗ S_all(u)·h_k(u) du,   S_all(u)=exp(−Σ_j H_j(u))

so a mode's incidence quantile depends on the competing modes, not just its own
Weibull.  This uses the existing ``CompetingRiskModel`` / ``competing_curve_frame``.

Reconciliation: the competing model's all-cause B50 (Σ cause hazards) vs a directly
fitted pooled all-cause Weibull — a mismatch >5% at B50 flags an error.

Outputs under ``results/phase_b_cause_weibull/<date>/``:
  tables/b5_cause_weibull_params.csv   — β,η per cause-stratum + ΔAIC(K1 vs K2)
  tables/b5_incidence_quantiles.csv    — t@10% / t@25% CIF per mode-stratum + CI
  tables/b5_reconciliation.csv         — competing vs pooled all-cause B50

Run:
    python scripts/run/phase_b_cause_weibull.py [n_boot]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.data.failure_modes import MODE_GROUPS
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.models.survival.latent_weibull_competing_risks import WeibullParameters
from analysis.models.survival.cif import parametric_competing_cif, time_to_incidence

STRATUM_MIN_FAILURES = 40      # field strata worth a parametric planning fit
CAUSE_MIN_EVENTS = 10          # a cause needs ≥10 events for a Weibull hazard
K2_MIN_EVENTS = 40             # ΔAIC K1-vs-K2 only where the cause-stratum is large
QUANTILES = (0.10, 0.25)


def _weibull_b50(beta: float, eta: float) -> float:
    return float(eta * (np.log(2.0)) ** (1.0 / beta))


def _cause_params(dur, ev) -> tuple[float, float, float]:
    """K=1 Weibull (β, η) for a cause; also ΔAIC vs K=2 where events ≥ threshold."""
    fit = fit_basic_weibull(dur, ev)
    beta, eta = float(fit["beta"]), float(fit["eta"])
    delta_aic = float("nan")
    if int(np.sum(ev)) >= K2_MIN_EVENTS:
        try:
            em = fit_latent_weibull_em(np.asarray(dur, float), np.asarray(ev, int))
            aic_k2 = float(2 * 5 + 2 * em.nll)   # 5 free params in K=2 mixture
            delta_aic = float(fit["aic"]) - aic_k2   # >0 ⇒ K=2 preferred
        except Exception:
            pass
    return beta, eta, delta_aic


def _cif_set(causes: list[WeibullParameters], max_time: float):
    """(u, S_all, cifs) for a set of cause Weibulls via the robust integrator."""
    betas = [c.beta for c in causes]
    etas = [c.eta for c in causes]
    return parametric_competing_cif(betas, etas, max_time, n_grid=4000)


def _competing_b50(u: np.ndarray, s_all: np.ndarray) -> float:
    if s_all.min() > 0.5:
        return float("nan")
    return float(np.interp(-0.5, -s_all, u))   # first time S crosses 0.5


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = results_dir("phase_b_cause_weibull")
    tbl = out / "tables"

    df = build_competing_risks_df(tte_col="ttf_mix")
    df["stratum_field"] = df["field"].astype(str) + "_" + df["h2s_class"].astype(str)

    param_rows, quant_rows, recon_rows = [], [], []

    for stratum, g in df.groupby("stratum_field", observed=True):
        if int(g["event"].sum()) < STRATUM_MIN_FAILURES:
            continue
        dur = g["tte"].to_numpy(float)
        max_time = max(float(dur.max()) * 1.5, 2000.0)

        # cause-specific Weibulls (only causes with enough events enter the model)
        causes, cause_names = [], []
        for group in MODE_GROUPS:
            ev_k = ((g["mode_group"].to_numpy() == group) & (g["event"].to_numpy() == 1)).astype(int)
            if int(ev_k.sum()) < CAUSE_MIN_EVENTS:
                continue
            beta, eta, daic = _cause_params(dur, ev_k)
            causes.append(WeibullParameters(beta=beta, eta=eta, label=group))
            cause_names.append(group)
            param_rows.append({
                "stratum": stratum, "mode_group": group, "n_events": int(ev_k.sum()),
                "beta": round(beta, 3), "eta": round(eta, 1),
                "cause_b50": round(_weibull_b50(beta, eta), 1),
                "delta_aic_k1_minus_k2": round(daic, 2) if np.isfinite(daic) else None,
                "k2_preferred": bool(np.isfinite(daic) and daic > 0),
            })
        if len(causes) < 2:
            continue

        # point incidence quantiles per cause (one CIF set, read all causes/quantiles)
        u, s_all_pt, cifs_pt = _cif_set(causes, max_time)
        point_q = {}
        for idx, group in enumerate(cause_names):
            for p in QUANTILES:
                point_q[(group, p)] = time_to_incidence(u, cifs_pt[idx], p)

        # well-bootstrap CIs on the incidence quantiles
        wells = g["well_key"].dropna().unique()
        well_rows = {w: g[g["well_key"] == w] for w in wells}
        rng = np.random.default_rng(123)
        boot = {k: [] for k in point_q}
        for _ in range(n_boot):
            drawn = rng.choice(wells, size=len(wells), replace=True)
            b = pd.concat([well_rows[w] for w in drawn], ignore_index=True)
            bdur = b["tte"].to_numpy(float)
            bcauses = []
            for group in cause_names:
                ev_k = ((b["mode_group"].to_numpy() == group) & (b["event"].to_numpy() == 1)).astype(int)
                if int(ev_k.sum()) < 5:
                    bcauses = []
                    break
                bf = fit_basic_weibull(bdur, ev_k)
                bcauses.append(WeibullParameters(beta=float(bf["beta"]), eta=float(bf["eta"]), label=group))
            if len(bcauses) != len(cause_names):
                continue
            bu, _bs, bcifs = _cif_set(bcauses, max_time)
            for idx, group in enumerate(cause_names):
                for p in QUANTILES:
                    v = time_to_incidence(bu, bcifs[idx], p)
                    if np.isfinite(v):
                        boot[(group, p)].append(v)

        for (group, p), pt in point_q.items():
            arr = np.asarray(boot[(group, p)], dtype=float)
            ok = len(arr) >= max(20, int(0.1 * n_boot))
            quant_rows.append({
                "stratum": stratum, "mode_group": group,
                "incidence_pct": int(p * 100),
                "t_days_point": round(pt, 1) if np.isfinite(pt) else None,
                "t_ci_lo": round(float(np.percentile(arr, 2.5)), 1) if ok else None,
                "t_ci_hi": round(float(np.percentile(arr, 97.5)), 1) if ok else None,
                "n_boot_ok": int(len(arr)),
                "reaches_incidence": bool(np.isfinite(pt)),
            })

        # reconciliation: competing all-cause B50 vs directly-fitted pooled B50
        pooled = fit_basic_weibull(dur, g["event"].to_numpy(int))
        pooled_b50 = _weibull_b50(float(pooled["beta"]), float(pooled["eta"]))
        comp_b50 = _competing_b50(u, s_all_pt)
        pct = abs(comp_b50 - pooled_b50) / pooled_b50 * 100 if pooled_b50 else float("nan")
        recon_rows.append({
            "stratum": stratum,
            "pooled_b50": round(pooled_b50, 1),
            "competing_b50": round(comp_b50, 1) if np.isfinite(comp_b50) else None,
            "abs_pct_diff": round(pct, 2) if np.isfinite(pct) else None,
            "flag_gt_5pct": bool(np.isfinite(pct) and pct > 5.0),
        })
        print(f"[B5] {stratum:<14} reconcile pooled_B50={pooled_b50:.0f} "
              f"competing_B50={comp_b50:.0f} Δ={pct:.1f}%")

    pd.DataFrame(param_rows).to_csv(tbl / "b5_cause_weibull_params.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quant_rows).to_csv(tbl / "b5_incidence_quantiles.csv", index=False, encoding="utf-8-sig")
    recon = pd.DataFrame(recon_rows)
    recon.to_csv(tbl / "b5_reconciliation.csv", index=False, encoding="utf-8-sig")

    n_flag = int(recon["flag_gt_5pct"].sum()) if not recon.empty else 0
    print(f"[B5] strata fitted: {len(recon)}  reconciliation flags (>5%): {n_flag}")
    print(f"[B5] outputs written to: {out}")


if __name__ == "__main__":
    main()
