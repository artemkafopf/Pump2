"""Well-cluster bootstrap CIs for K=2 mixture B10/B50/w₁ (Phase A T2).

The Phase 3 machinery resamples *runs*; but 62% of runs are repeats on the same
well, so run-resampling understates uncertainty.  This module resamples **wells**
(clusters), refits the EM on the pooled runs of the drawn wells, and returns
percentile CIs for B10, B50 and w₁ — the intervals that belong on every operational
B50 table (interval, not adjective).

Degenerate resamples (w₁ above the EM degeneracy threshold) are counted; when their
share exceeds ``max_degenerate_frac`` the stratum interval is marked unreliable
rather than reported as a fake-tight CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.models.survival.weibull_em import fit_latent_weibull_em, DEFAULT_DEGENERATE_W1
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    latent_life_quantile,
)


def bootstrap_stratum_ci(
    g: pd.DataFrame,
    point_model: TwoComponentLatentWeibullModel,
    *,
    n_boot: int = 200,
    well_col: str = "well_key",
    tte_col: str = "tte",
    event_col: str = "event",
    fix_beta1: float | None = None,
    fix_beta2: float | None = None,
    max_degenerate_frac: float = 0.30,
    num_starts: int = 1,
    max_iter: int = 120,
    seed: int = 42,
) -> dict:
    """Well-cluster bootstrap for one stratum.

    Each resample refits the EM **warm-started from the converged point model**
    (``num_starts=1``), which is far cheaper than a cold multi-start fit and adequate
    because the point estimate is already a good basin. Returns b10/b50/w1 point +
    [lo, hi] percentile CIs, the successful/degenerate resample counts, and a
    ``reliable`` flag.
    """
    wells = g[well_col].dropna().unique()
    rng = np.random.default_rng(seed)
    n_wells = len(wells)

    b10s, b50s, w1s = [], [], []
    n_degen = 0
    n_ok = 0
    well_to_rows = {w: g[g[well_col] == w] for w in wells}

    for _ in range(n_boot):
        drawn = rng.choice(wells, size=n_wells, replace=True)
        boot = pd.concat([well_to_rows[w] for w in drawn], ignore_index=True)
        t = boot[tte_col].to_numpy(dtype=float)
        e = boot[event_col].to_numpy(dtype=int)
        try:
            res = fit_latent_weibull_em(
                t, e, initial_model=point_model, num_starts=num_starts, max_iter=max_iter,
                fix_beta1=fix_beta1, fix_beta2=fix_beta2,
            )
        except Exception:
            continue
        m = res.model
        if m.weight_1 > DEFAULT_DEGENERATE_W1:
            n_degen += 1
        b10 = latent_life_quantile(0.10, m)
        b50 = latent_life_quantile(0.50, m)
        if np.isfinite(b10):
            b10s.append(float(b10))
        if np.isfinite(b50):
            b50s.append(float(b50))
        w1s.append(float(m.weight_1))
        n_ok += 1

    def _ci(a: list[float]) -> tuple[float | None, float | None]:
        if len(a) < max(10, int(0.1 * n_boot)):
            return (None, None)
        return (round(float(np.percentile(a, 2.5)), 1),
                round(float(np.percentile(a, 97.5)), 1))

    degen_frac = (n_degen / n_ok) if n_ok else 1.0
    reliable = n_ok >= max(10, int(0.5 * n_boot)) and degen_frac <= max_degenerate_frac

    b10_lo, b10_hi = _ci(b10s)
    b50_lo, b50_hi = _ci(b50s)
    w1_lo, w1_hi = _ci(w1s)
    return {
        "b10_point": round(float(latent_life_quantile(0.10, point_model)), 1),
        "b50_point": round(float(latent_life_quantile(0.50, point_model)), 1),
        "w1_point": round(float(point_model.weight_1), 3),
        "b10_lo": b10_lo, "b10_hi": b10_hi,
        "b50_lo": b50_lo, "b50_hi": b50_hi,
        "w1_lo": (round(w1_lo, 3) if w1_lo is not None else None),
        "w1_hi": (round(w1_hi, 3) if w1_hi is not None else None),
        "n_boot": n_boot,
        "n_boot_ok": n_ok,
        "n_degenerate": n_degen,
        "degenerate_frac": round(degen_frac, 3),
        "reliable": bool(reliable),
    }


__all__ = ["bootstrap_stratum_ci"]
