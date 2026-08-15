"""Free gas at the intake, computed on reconstructed Рприем — and what the reconstruction
error does to it.

Rebuilding Рприем is a means, not the end: what the survival layer consumes is β, the
free gas volume fraction at intake from :mod:`analysis.features.free_gas`.  So the
reconstruction has to be scored *in β*, not in atm.  The two are not proportional — β is
strongly non-linear in pressure below the bubble point, so the same 18 atm error is nearly
harmless at 150 atm and decisive at 30 — and a P_intake MAE is therefore not a usable
proxy for the thing that matters.  :func:`beta_error_profile` measures the map directly.

The gas-factor unit, confirmed
-------------------------------
:mod:`analysis.features.free_gas` flags a suspected "m³/t vs m³/m³ convention in
``gas_factor``, worth ~15 %".  The telemetry column map settles it: the source field is
**«Газовый фактор, м3/т»** — gas per *tonne* of oil, not per cubic metre.  Standing's
``Rs`` is m³/m³ of stock-tank oil, so the column must be multiplied by the oil density in
t/m³ before use.  :func:`gor_m3_per_m3` does that conversion from the PVT's own
``oil_sg``, and it is applied here rather than inside ``free_gas`` so the older callers of
that module keep their existing (documented, anchored) behaviour unchanged.

The warehouse also carries a properly dimensioned **«Газожидкостный фактор, м3/м3»**, but
it is gas per unit *liquid* rather than per unit oil and is 12 % populated in the raw
telemetry, so it is not a drop-in substitute.

⚠ **Imputed β is range-compressed — do not compute exposure shares on it.**
A regression to the mean is unavoidable here (a boosted tree cannot leave the convex hull
of its leaf means, and MAE loss pulls hard toward the centre), and it shows up as a
narrower reconstructed pressure: sd ratio **0.860** against the observed, minimum
0.5 → 10.5 atm, p99 199 → 184.  Because β is monotone decreasing and strongly convex in
pressure, that compression maps into a **monotone β bias gradient** — measured
out-of-sample on the held-out blocks:

===================  ===============  ==========
true Рприем band     mean β           β bias
===================  ===============  ==========
(0, 20]              0.704            **−0.091**
(20, 40]             0.655            −0.057
(40, 60]             0.584            −0.010
(60, 100]            0.404            +0.024
(100, 400]           0.147            **+0.050**
===================  ===============  ==========

So β is pulled toward its middle: **understated by ~0.09 in the gassiest wells and
overstated by ~0.05 in the least gassy**.  Rank statistics survive this (well-level
Spearman 0.980), which is why the ordering-based use β is licensed for remains valid.
What does *not* survive is anything reading β's **spread or tails** — in particular
``free_gas.free_gas_window``'s ``frac_beta_above_*`` exposure shares, which are threshold
counts in exactly the tails being shrunk.  Compute those on observed rows only, or accept
a known-direction bias.

⚠ Everything :mod:`~analysis.features.free_gas` says about β still applies here and is not
softened by a better pressure: β remains an **upper bound** with no separation credit, its
level is not comparable to published tolerance bands, and only its *ordering* is trusted.
Imputing Рприем improves coverage; it does not upgrade β from a ranking variable to an
absolute one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.free_gas import (
    DEFAULT_PVT,
    PVT,
    free_gas_fraction,
    pvt_sensitivity_grid,
)


def gor_m3_per_m3(gas_factor_m3_per_t, pvt: PVT = DEFAULT_PVT) -> np.ndarray:
    """Convert the warehouse ``gas_factor`` (m³/t of oil) to m³/m³ of stock-tank oil.

    ``m³/t × (t/m³) = m³/m³``, with the oil density taken from the PVT's own API gravity
    so the conversion moves consistently with the rest of the sensitivity sweep.  At the
    default API 32 (``oil_sg`` 0.865) this scales the reported GOR by 0.865 — the ~15 %
    the free-gas module suspected.
    """
    return np.asarray(gas_factor_m3_per_t, dtype=float) * pvt.oil_sg


def attach_beta(
    df: pd.DataFrame,
    p_intake_col: str = "rpump_intake",
    *,
    pvt: PVT = DEFAULT_PVT,
    convert_gor_units: bool = True,
    out_col: str = "beta",
) -> pd.DataFrame:
    """Attach β computed off ``p_intake_col``.

    ``p_bubble_atm`` is passed through, which puts ``Rs`` in the anchored mode the
    free-gas module establishes as the honest one on this fleet (raw Standing
    under-dissolves by 1.3-4x here).
    """
    out = df.copy()
    gor = out["gas_factor"].to_numpy(float)
    if convert_gor_units:
        gor = gor_m3_per_m3(gor, pvt)
    out[out_col] = free_gas_fraction(
        out[p_intake_col].to_numpy(float),
        gor,
        out["watercut"].to_numpy(float),
        p_bubble_atm=out["pbubble_atm"].to_numpy(float),
        pvt=pvt,
    )
    return out


def beta_error_profile(
    df: pd.DataFrame,
    p_true_col: str = "rpump_intake",
    p_hat_col: str = "p_intake_hat",
    *,
    pvt: PVT = DEFAULT_PVT,
    bins=(0, 20, 40, 60, 100, 400),
) -> pd.DataFrame:
    """β error induced by the pressure error, banded by the *true* intake pressure.

    This is the table that decides whether the imputation is fit for purpose.  A single
    pooled β-MAE would hide the structure that matters: the same pressure error maps to
    very different β errors depending on where on the curve it lands, and the low-pressure
    band is both the most sensitive and the one where free gas is actually a problem.
    """
    d = attach_beta(df, p_true_col, pvt=pvt, out_col="_b_true")
    d = attach_beta(d, p_hat_col, pvt=pvt, out_col="_b_hat")
    d = d[np.isfinite(d["_b_true"]) & np.isfinite(d["_b_hat"])]
    if not len(d):
        return pd.DataFrame()
    d = d.assign(
        band=pd.cut(d[p_true_col], bins=list(bins)),
        p_err=(d[p_hat_col] - d[p_true_col]).abs(),
        b_err=(d["_b_hat"] - d["_b_true"]).abs(),
        b_signed=d["_b_hat"] - d["_b_true"],
    )
    g = d.groupby("band", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "beta_true_mean": g["_b_true"].mean(),
        "p_mae_atm": g["p_err"].mean(),
        "beta_mae": g["b_err"].mean(),
        "beta_med_ae": g["b_err"].median(),
        "beta_bias": g["b_signed"].mean(),
    })
    # d(beta)/d(P) implied by this band -- the number that says whether pressure error
    # translates into beta error at a rate worth caring about.
    out["dbeta_per_atm"] = out["beta_mae"] / out["p_mae_atm"].replace(0, np.nan)
    return out.reset_index()


def beta_rank_stability(
    df: pd.DataFrame,
    p_true_col: str = "rpump_intake",
    p_hat_col: str = "p_intake_hat",
    *,
    by: str = "well_key",
    pvt: PVT = DEFAULT_PVT,
) -> dict:
    """Does the reconstruction preserve β's **ordering**, which is all β is trusted for?

    The free-gas module is explicit that β's level is an upper bound with an unknown
    offset and only its ranking is usable.  So the acceptance test for an imputed β is not
    "is the level right" but "does it rank wells the same way".  Spearman on the
    per-``by`` mean β answers exactly that question, and it is a far more forgiving — and
    more relevant — bar than the pointwise error.
    """
    d = attach_beta(df, p_true_col, pvt=pvt, out_col="_b_true")
    d = attach_beta(d, p_hat_col, pvt=pvt, out_col="_b_hat")
    g = d.groupby(by)[["_b_true", "_b_hat"]].mean().dropna()
    if len(g) < 3:
        return {"n_groups": len(g), "spearman": np.nan, "pearson": np.nan}
    return {
        "n_groups": int(len(g)),
        "spearman": float(g["_b_true"].corr(g["_b_hat"], method="spearman")),
        "pearson": float(g["_b_true"].corr(g["_b_hat"])),
        "mean_beta_true": float(g["_b_true"].mean()),
        "mean_beta_hat": float(g["_b_hat"].mean()),
    }


def pvt_sweep_beta(
    df: pd.DataFrame,
    p_intake_col: str = "rpump_intake",
    *,
    by: str = "well_key",
) -> pd.DataFrame:
    """The mandatory PVT sweep, restated on the reconstructed pressure.

    ``free_gas`` requires every β-based conclusion to be re-run across
    :func:`~analysis.features.free_gas.pvt_sensitivity_grid` and reported with the range.
    That obligation does not lapse because the pressure is now imputed — if anything it
    compounds, so the sweep is repeated here against the default-PVT ranking.
    """
    ref = None
    rows = []
    for pvt in pvt_sensitivity_grid():
        d = attach_beta(df, p_intake_col, pvt=pvt, out_col="_b")
        g = d.groupby(by)["_b"].mean().dropna()
        if ref is None:
            ref = g
        common = ref.index.intersection(g.index)
        rows.append({
            "api": pvt.api_gravity, "temp_c": pvt.temp_c, "gas_gravity": pvt.gas_gravity,
            "beta_median": float(g.median()),
            "spearman_vs_default": float(
                ref.loc[common].corr(g.loc[common], method="spearman")
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "attach_beta",
    "gor_m3_per_m3",
    "beta_error_profile",
    "beta_rank_stability",
    "pvt_sweep_beta",
]
