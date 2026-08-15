"""Free gas volume fraction at the pump intake — the causal mediator for pump sizing.

Why this module exists
----------------------
``P_intake / P_bubble`` looked like the natural gas proxy and it does not work: measured
over 976 landmark-eligible runs the ratio has median **0.246** and **99.0 %** of runs sit
below bubble point.  "Is there free gas?" is therefore a constant on this fleet, so any
threshold indicator (``frac_pzab_below_1`` included) carries almost no cross-run
information.  What varies is *how much* free gas, and β is strongly non-linear in P below
P_b, so the ratio cannot stand in for it.

This module computes the quantity ESP design actually uses:

.. math::
    \\beta = \\frac{V_{gas,free}}{V_{gas,free} + V_{liquid}}  \\quad\\text{at intake conditions}

Published tolerance bands (:data:`BETA_BANDS`) put a conventional pump at ~10-15 % free
gas before head degradation, ~25-40 % with a separator/gas handler, so β lands on a
physical scale rather than an arbitrary index.

The recipe, per day
-------------------
1. ``Rs(P_intake)`` — solution GOR still dissolved at intake (Standing).  Capped at the
   producing GOR: a well cannot have more gas in solution than it produces in total.
2. Gas broken out of solution: ``q_oil · (GOR − Rs(P_intake))``, floored at 0 — above
   bubble point nothing is liberated and β is exactly 0.
3. That gas expanded to intake conditions via ``Bg(P, T)`` (real-gas law).
4. Liquid volume at intake: oil shrunk/swollen by ``Bo``, water by ``Bw ≈ 1``.

Standing's correlations are field-unit fits; inputs and outputs here are **metric**
(atm, °C, m³/m³) and the conversion happens inside.

What is assumed, and how to challenge it
----------------------------------------
There is no PVT table in the warehouse, so oil API, gas gravity and reservoir
temperature are **assumptions** carried in :class:`PVT`, not measurements.  Defaults are
generic light-oil values.  Two things make this defensible rather than hand-waving:

* The assumption set is falsifiable against data we do have.  At ``P = P_bubble`` Standing
  must return ``Rs ≈ GOR``; :func:`validate_pvt_against_bubble_point` runs exactly that
  check per field and reports the ratio.  On this fleet the warehouse's own
  ``pbubble_atm`` (230-260 atm on Ya) and ``gas_factor`` (median 222 m³/m³) are mutually
  consistent under the default PVT — which is independent evidence that neither column is
  mis-scaled, and that these really are saturated, gas-rich reservoirs.
* Any conclusion drawn from β must be shown to survive :func:`pvt_sensitivity_grid`.
  β is a *ranking* variable here; the PVT constants mostly shift its level, and a finding
  that flips when API moves 28 → 36 is a finding about the assumption, not the fleet.

⚠ β is the gas fraction arriving *at the intake*.  Natural separation at the intake and
any gas separator/handler reduce what actually enters the first stage.  ``separation``
in :func:`free_gas_fraction` exposes that; it defaults to 0 (no credit) because the
warehouse records no separator equipment, so the computed β is an **upper bound** on what
the pump ingests.  Comparisons across runs stay valid as long as separator fitting is not
itself correlated with pump size — which is untested.

**Read β as a ranking variable, not as an absolute.**  Two measured facts fix how it may
be used:

* *Its level is not comparable to published thresholds.*  Median β on this fleet is
  0.586 and 66 % of daily rows exceed 0.40 — a pump truly ingesting 59 % free gas would
  gas-lock continuously, and these wells plainly run.  The gap is the separation credit
  the warehouse cannot supply (natural separation alone is commonly 50-90 % here).  So β
  is an upper bound with an unknown, probably large, offset.  Do not report "β = 0.59
  exceeds the 0.15 tolerance" — that comparison is not supported.
* *Its ordering is solid.*  Across the full :func:`pvt_sensitivity_grid` (API 28-36,
  T 60-100 °C, γg 0.70-0.90) the anchored β has Spearman correlation **1.0000** with the
  default-PVT β, and its median moves only 4.1 % (vs 10.7 % unanchored).  Every PVT
  assumption in this module is therefore irrelevant to any rank-based or
  monotone-transform conclusion, which is what the survival layer consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Unit conversions (Standing is a field-unit correlation) ──────────────────
ATM_TO_PSIA = 14.695949
#: 1 scf/STB = 0.1781076 m³/m³.
SCF_STB_TO_M3_M3 = 0.1781076
#: Russian standard conditions (ГОСТ 2939-63): 20 °C, 101.325 kPa.
T_STD_K = 293.15
P_STD_ATM = 1.0
KELVIN_0C = 273.15

#: Physical guards on raw telemetry.  These columns carry impossible values in the
#: warehouse (watercut min −58.7 / max 172 858; gas_factor max 4.9e6; rpump_intake max
#: 8990 atm) and every one of them would silently poison a mean.  Out-of-range → NaN.
WATERCUT_RANGE = (0.0, 100.0)      # percent
GOR_RANGE = (0.0, 3000.0)          # m³/m³ of oil
P_INTAKE_RANGE = (0.5, 400.0)      # atm
P_RES_RANGE = (1.0, 600.0)         # atm

#: Free-gas exposure bands (volume fraction at intake).  The first two are the published
#: equipment thresholds (conventional pumps degrade from ~0.10, gas handlers reach
#: ~0.25-0.40); the upper two exist because **this fleet sits far above the published
#: bands** — measured on 594 861 daily rows, β has median 0.586 and 86 % of days exceed
#: 0.10, so a 0.10 indicator is nearly as saturated as the P/P_b ratio it replaced.
#: What rescues β is spread, not level: IQR 0.27-0.78, p5-p95 0.014-0.920.  Use the
#: upper bands for exposure shares on this fleet and read the lower two as context.
BETA_BANDS = (0.10, 0.25, 0.50, 0.75)


@dataclass(frozen=True)
class PVT:
    """Black-oil PVT assumptions.  **Not measured** — see the module docstring.

    Parameters
    ----------
    api_gravity
        Stock-tank oil API gravity (°API).  Light West-Siberian/Yakutian crude ~30-36.
    gas_gravity
        Separator gas specific gravity (air = 1).
    temp_c
        Reservoir/flowing temperature at the intake (°C).
    z_factor
        Gas compressibility at intake conditions.  Held constant: at the 30-100 atm
        intake pressures seen here Z varies ~0.85-0.92, a ±4 % effect on Bg that is far
        below the API/temperature uncertainty.
    """

    api_gravity: float = 32.0
    gas_gravity: float = 0.80
    temp_c: float = 80.0
    z_factor: float = 0.88

    @property
    def temp_f(self) -> float:
        return self.temp_c * 9.0 / 5.0 + 32.0

    @property
    def temp_k(self) -> float:
        return self.temp_c + KELVIN_0C

    @property
    def oil_sg(self) -> float:
        """Stock-tank oil specific gravity (water = 1)."""
        return 141.5 / (131.5 + self.api_gravity)


DEFAULT_PVT = PVT()


def _clean(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Values outside ``[lo, hi]`` → NaN (mirrors the landmark/run-covariate guards)."""
    out = np.asarray(a, dtype=float).copy()
    with np.errstate(invalid="ignore"):
        out[(out < lo) | (out > hi)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Standing black-oil correlations
# ---------------------------------------------------------------------------

def solution_gor(p_atm: np.ndarray | float, pvt: PVT = DEFAULT_PVT) -> np.ndarray:
    """Solution GOR ``Rs`` (m³/m³ of stock-tank oil) at pressure ``p_atm``.

    Standing (1947):

    .. math::
        R_s = \\gamma_g \\left[ \\left(\\frac{P}{18.2} + 1.4\\right)
              10^{0.0125\\,API - 0.00091\\,T_F} \\right]^{1.2048}

    with ``P`` in psia and ``R_s`` in scf/STB; converted to metric on the way out.
    Monotone increasing in ``P``.  Non-positive / NaN pressure → NaN.
    """
    p = np.asarray(p_atm, dtype=float)
    with np.errstate(invalid="ignore"):
        p_psia = np.where(p > 0, p * ATM_TO_PSIA, np.nan)
        y = 10.0 ** (0.0125 * pvt.api_gravity - 0.00091 * pvt.temp_f)
        rs_scf = pvt.gas_gravity * np.power((p_psia / 18.2 + 1.4) * y, 1.2048)
    return rs_scf * SCF_STB_TO_M3_M3


def solution_gor_anchored(
    p_atm: np.ndarray | float,
    p_bubble_atm: np.ndarray | float,
    gor_m3m3: np.ndarray | float,
    pvt: PVT = DEFAULT_PVT,
) -> np.ndarray:
    """``Rs(P)`` with Standing's *shape* but anchored so that ``Rs(P_b) = GOR`` exactly.

    .. math::
        R_s^{anch}(P) = GOR \\cdot
            \\frac{R_s^{Std}\\!\\left(\\min(P, P_b)\\right)}{R_s^{Std}(P_b)}

    **Why this replaces raw Standing as the default.**  Run against the warehouse, raw
    Standing fails :func:`validate_pvt_against_bubble_point`: median ``Rs(P_b)/GOR`` is
    **0.55** (Za 0.24, Mc 0.28, Az 0.30, Vt 0.43, Ya 0.76), and no PVT set in
    :func:`pvt_sensitivity_grid` gets past 0.78.  The reported producing GOR is
    consistently 1.3-4× what the fluid can hold in solution at its own reported bubble
    point.  That is not a bug to be tuned away — it is the expected signature of gas-cap
    or coning contribution (and possibly a m³/t vs m³/m³ convention in ``gas_factor``,
    worth ~15 %).  Either way the *level* of raw Standing is not transferable here.

    Anchoring keeps only what Standing is actually good for — the **shape** of the
    Rs-vs-pressure curve — and pins the endpoint to the two columns the warehouse does
    carry (``pbubble_atm``, ``gas_factor``).  This is ordinary correlation tuning to a
    measured ``(P_b, R_{s,b})`` pair, and it makes β nearly independent of the assumed
    API / temperature / gas gravity, because those largely cancel in the ratio.  The
    residual PVT dependence runs through ``Bg`` and ``Bo`` only, and
    :func:`pvt_sensitivity_grid` still has to be swept to show it.

    ⚠ Anchoring makes the *excess* gas behave as though it were dissolved at ``P_b`` and
    breaks out along Standing's shape.  For gas-cap gas that is not literally true — such
    gas is free at all pressures — so this is the **conservative** choice: it assigns
    less free gas at intake than a gas-cap model would.  β stays a lower bound on the
    gas-cap component and remains valid for ranking runs.
    """
    p = np.asarray(p_atm, dtype=float)
    pb = np.asarray(p_bubble_atm, dtype=float)
    gor = np.asarray(gor_m3m3, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs_b = solution_gor(pb, pvt)
        shape = solution_gor(np.minimum(p, pb), pvt) / np.where(rs_b > 0, rs_b, np.nan)
        return gor * shape


def gas_fvf(p_atm: np.ndarray | float, pvt: PVT = DEFAULT_PVT) -> np.ndarray:
    """Gas formation volume factor ``Bg`` (m³ at intake per m³ at standard conditions).

    Real-gas law referenced to standard conditions: ``Bg = Z·(P_std/P)·(T/T_std)``.
    """
    p = np.asarray(p_atm, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = pvt.z_factor * (P_STD_ATM / np.where(p > 0, p, np.nan)) * (pvt.temp_k / T_STD_K)
    return bg


def oil_fvf(rs_m3m3: np.ndarray | float, pvt: PVT = DEFAULT_PVT) -> np.ndarray:
    """Oil formation volume factor ``Bo`` (m³ at intake per m³ stock tank), Standing.

    .. math::
        B_o = 0.9759 + 0.00012 \\left[ R_s \\sqrt{\\gamma_g/\\gamma_o}
              + 1.25\\,T_F \\right]^{1.2}

    ``R_s`` arrives metric and is converted to scf/STB internally.
    """
    rs = np.asarray(rs_m3m3, dtype=float) / SCF_STB_TO_M3_M3
    with np.errstate(invalid="ignore"):
        bo = 0.9759 + 0.00012 * np.power(
            rs * np.sqrt(pvt.gas_gravity / pvt.oil_sg) + 1.25 * pvt.temp_f, 1.2
        )
    return bo


# ---------------------------------------------------------------------------
# The feature
# ---------------------------------------------------------------------------

def free_gas_fraction(
    p_intake_atm: np.ndarray | float,
    gor_m3m3: np.ndarray | float,
    watercut_pct: np.ndarray | float,
    *,
    p_bubble_atm: np.ndarray | float | None = None,
    pvt: PVT = DEFAULT_PVT,
    separation: float = 0.0,
    apply_guards: bool = True,
) -> np.ndarray:
    """Free gas volume fraction ``β`` at the pump intake (0-1).

    Parameters
    ----------
    p_intake_atm
        Pump intake pressure (atm) — warehouse ``rpump_intake``.
    gor_m3m3
        Producing gas-oil ratio (m³ gas per m³ stock-tank **oil**) — warehouse
        ``gas_factor``.  Verified against ``qgas/qoil`` to a 17 % median discrepancy.
    watercut_pct
        Water cut in **percent** (0-100) — warehouse ``watercut``.
    p_bubble_atm
        Bubble-point pressure (atm) — warehouse ``pbubble_atm``.  **Strongly preferred.**
        When supplied, ``Rs`` is anchored to ``(P_b, GOR)`` via
        :func:`solution_gor_anchored`, which is the honest mode on this fleet; raw
        Standing under-dissolves by 1.3-4× here (see that function).  When ``None``,
        falls back to unanchored Standing and β is correspondingly over-stated.
    separation
        Fraction of free gas diverted before the first stage (natural separation +
        separator).  Default 0 → β is the upper bound actually arriving at the intake.
    apply_guards
        Drop physically impossible telemetry to NaN before computing.  Leave on for
        warehouse data; tests pass False to exercise exact synthetic values.

    Returns
    -------
    ``β`` as an array, NaN wherever any input is missing/invalid.  β is exactly 0 when
    ``Rs(P_intake) ≥ GOR`` — i.e. at or above bubble point, no gas has broken out.

    Notes
    -----
    Water cut enters as a **protective** term: gas is carried by the oil phase, so at
    fixed GOR a high-watercut well liberates less gas per unit of liquid *and* has more
    liquid volume to dilute it.  Both effects push β down.  This is why raw watercut
    correlates positively with Kpod in the fleet data and why GOR alone tests weak —
    neither is a mistake, both are this mechanism.
    """
    p = np.asarray(p_intake_atm, dtype=float)
    gor = np.asarray(gor_m3m3, dtype=float)
    wct = np.asarray(watercut_pct, dtype=float)
    pb = None if p_bubble_atm is None else np.asarray(p_bubble_atm, dtype=float)
    if apply_guards:
        p = _clean(p, *P_INTAKE_RANGE)
        gor = _clean(gor, *GOR_RANGE)
        wct = _clean(wct, *WATERCUT_RANGE)
        if pb is not None:
            pb = _clean(pb, *P_RES_RANGE)

    if pb is None:
        p, gor, wct = np.broadcast_arrays(p, gor, wct)
        rs = solution_gor(p, pvt)
    else:
        p, gor, wct, pb = np.broadcast_arrays(p, gor, wct, pb)
        rs = solution_gor_anchored(p, pb, gor, pvt)
    fw = wct / 100.0
    fo = 1.0 - fw

    # Cannot have more gas dissolved than the well produces in total.
    rs_eff = np.minimum(rs, gor)
    with np.errstate(invalid="ignore"):
        liberated = np.clip(gor - rs_eff, 0.0, None)          # m³ std gas / m³ oil
        v_gas = fo * liberated * gas_fvf(p, pvt)              # m³ at intake / m³ liquid
        v_gas = v_gas * (1.0 - float(separation))
        v_liq = fo * oil_fvf(rs_eff, pvt) + fw                # Bw ≈ 1.0
        beta = v_gas / (v_gas + v_liq)
    return np.asarray(beta, dtype=float)


def free_gas_window(
    p_intake_atm: np.ndarray,
    gor_m3m3: np.ndarray,
    watercut_pct: np.ndarray,
    *,
    p_bubble_atm: np.ndarray | float | None = None,
    pvt: PVT = DEFAULT_PVT,
    bands: tuple[float, ...] = BETA_BANDS,
) -> dict:
    """Summarise β over one daily window: mean, p90, and exposure share above each band.

    Mean is the chronic dose, ``frac_beta_above_*`` the excursion dose — the same
    level-vs-share pairing ``kpod_features`` uses, kept because the two can disagree and
    the disagreement is informative.
    """
    beta = free_gas_fraction(p_intake_atm, gor_m3m3, watercut_pct,
                             p_bubble_atm=p_bubble_atm, pvt=pvt)
    v = beta[np.isfinite(beta)]
    out: dict = {
        "beta_mean": float(np.mean(v)) if len(v) else np.nan,
        "beta_p90": float(np.percentile(v, 90)) if len(v) else np.nan,
        "beta_max": float(np.max(v)) if len(v) else np.nan,
        "beta_n_days": int(len(v)),
    }
    for b in bands:
        key = f"frac_beta_above_{str(b).replace('.', 'p')}"
        out[key] = float(np.mean(v > b)) if len(v) else np.nan
    return out


# ---------------------------------------------------------------------------
# Honesty gates on the PVT assumption
# ---------------------------------------------------------------------------

def validate_pvt_against_bubble_point(
    p_bubble_atm: np.ndarray,
    gor_m3m3: np.ndarray,
    pvt: PVT = DEFAULT_PVT,
) -> dict:
    """Check the PVT set against the warehouse's own bubble point.

    At ``P = P_b`` all produced gas is (by definition) in solution, so Standing must
    return ``Rs(P_b) ≈ GOR``.  The returned ``ratio_median`` is ``Rs(P_b)/GOR``: ≈1
    means the assumed API/temperature/gas-gravity are consistent with the two columns,
    ≫1 means Standing over-dissolves (API or temperature assumption too generous), ≪1
    means the reported GOR exceeds what the fluid can hold at its own bubble point —
    which would indicate one of the two columns is mis-scaled.

    This does not prove the PVT is right; it proves it is not grossly wrong, using data
    the β computation itself never touches.
    """
    pb = _clean(np.asarray(p_bubble_atm, dtype=float), *P_RES_RANGE)
    gor = _clean(np.asarray(gor_m3m3, dtype=float), *GOR_RANGE)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = solution_gor(pb, pvt) / np.where(gor > 0, gor, np.nan)
    v = ratio[np.isfinite(ratio)]
    return {
        "n": int(len(v)),
        "ratio_median": float(np.median(v)) if len(v) else np.nan,
        "ratio_p10": float(np.percentile(v, 10)) if len(v) else np.nan,
        "ratio_p90": float(np.percentile(v, 90)) if len(v) else np.nan,
        "share_within_2x": float(np.mean((v > 0.5) & (v < 2.0))) if len(v) else np.nan,
    }


def pvt_sensitivity_grid(
    api: tuple[float, ...] = (28.0, 32.0, 36.0),
    temp_c: tuple[float, ...] = (60.0, 80.0, 100.0),
    gas_gravity: tuple[float, ...] = (0.70, 0.80, 0.90),
) -> list[PVT]:
    """PVT sets spanning the plausible assumption space, for the mandatory sweep.

    Any β-based conclusion must be re-run across this grid and reported with the range,
    not just the default-PVT point estimate.
    """
    return [PVT(api_gravity=a, temp_c=t, gas_gravity=g)
            for a in api for t in temp_c for g in gas_gravity]


__all__ = [
    "PVT",
    "DEFAULT_PVT",
    "solution_gor",
    "solution_gor_anchored",
    "gas_fvf",
    "oil_fvf",
    "free_gas_fraction",
    "free_gas_window",
    "validate_pvt_against_bubble_point",
    "pvt_sensitivity_grid",
    "BETA_BANDS",
    "WATERCUT_RANGE",
    "GOR_RANGE",
    "P_INTAKE_RANGE",
    "P_RES_RANGE",
]
