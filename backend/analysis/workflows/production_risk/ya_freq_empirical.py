"""Empirical Ya frequency layer — the curves the Vt transfer is argued from, made reproducible.

The 2026-07-24 transfer study that produced these existed only as a figure: its manifest records
``"script": null``, so the all-pulls / failures-only curves and their bootstrap band could not be
re-derived, re-binned, or re-pinned.  This module rebuilds them from the warehouse.

**Recipe** (as described in ``project_ya_to_vt_freq_transfer``).  Take Ya runs, remove the Ya v2
Ql layer as an AFT time-scale offset so what is left varies with frequency alone::

    t_adj = t · θ_Ql(Ql) ** (1 / β)          β = the k1 baseline shape (0.7875)

then bin by frequency, take the median adjusted life per bin, and convert the life ratio back to
a hazard multiplier against the pinned reference::

    θ(f) = ( life(pin) / life(f) ) ** β

Two variants, and the difference between them is the whole argument:

* ``all_pulls`` — every run's observed time, censoring-correct in the sense that nothing is
  dropped;
* ``failures_only`` — ``event == 1`` rows only.  This is the variant the project has repeatedly
  caught inflating effects (the failure-only brt/slb reversal, the 51 Hz peak); here it inflates
  the **over-speed** arm specifically — θ(60) ≈ 1.43 against ≈ 1.00 for all-pulls — while the two
  agree closely below 45 Hz.

**The pin is a real choice, not a detail.**  ``PIN_MODE = "min"`` lets the valley land where the
data puts it (≈51.5 Hz, matching the 50–54 Hz band the transfer note describes and the original
study's symmetric-about-51.5 curve); ``"ref"`` forces θ(50) = 1, which drags the whole curve and
was the main discrepancy against the original reconstruction.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import results_dir

SLUG = "production_risk_ya_freq_empirical"

#: Frequency bin edges.  Narrow through the populated middle, wide in the sparse tails.
#: Chosen so the centres land on the original study's reported frequencies (42…60) and extend
#: a little past them at both ends.
BIN_EDGES = (28.0, 32.0, 36.0, 40.5, 43.5, 46.0, 48.5, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0,
             66.0, 70.0)
#: Reference frequency (θ = 1 here when ``PIN_MODE == "ref"``).
FREQ_REF = 50.0
#: ``"ref"`` — force θ(FREQ_REF) = 1 (what the 2026-07-24 study did); ``"min"`` — pin at the
#: observed floor.  ``"ref"`` is the default because it is the only setting that reproduces the
#: original curve: pinning at the floor forces every bin ≥ 1 and so erases the 52–54 Hz sweet
#: spot, which is the whole point of the original.
PIN_MODE = "ref"
#: Per-bin summary of adjusted life.  ``"mean"`` reproduces the original study (mean |Δmult|
#: 0.067 against its table, vs 0.142 for the median).
BIN_STAT = "mean"
#: Bins with fewer runs than this are reported but excluded from fits.
MIN_BIN_N = 8
#: Bootstrap replicates for the band.
N_BOOT = 400
BOOT_SEED = 20260727


#: Grid the fitted Ya polyline layer is tabulated on, so the explorer never refits Ya.
POLYLINE_GRID = np.arange(33.0, 70.5, 0.5)

#: Kpod = Ql / Qnom.  Same recipe as frequency: bin, median adjusted life, back to a hazard
#: multiplier.  Edges are narrow through the loaded band and wide in the sparse tails.
KPOD_BIN_EDGES = (0.10, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.30, 1.80)
KPOD_REF = 0.8
KPOD_POLYLINE_GRID = np.arange(0.15, 1.62, 0.02)
#: Anchors the controllable Kpod form stretches to.
KPOD_ANCHOR_LEFT, KPOD_ANCHOR_RIGHT = 0.2, 1.2


@dataclass
class EmpiricalCurves:
    all_pulls: pd.DataFrame
    failures_only: pd.DataFrame
    band: pd.DataFrame          # percentile band per (variant, bin)
    beta: float
    n_runs: int
    polyline: pd.DataFrame | None = None   # the Ya v2 fitted freq tent, tabulated


def _centres() -> np.ndarray:
    e = np.asarray(BIN_EDGES, float)
    return (e[:-1] + e[1:]) / 2.0


def _bin_curve(freq: np.ndarray, t_adj: np.ndarray, beta: float,
               pin_mode: str = None, edges=None, ref=None) -> pd.DataFrame:
    """Median adjusted life per bin → θ, pinned per ``pin_mode``.  Covariate-agnostic."""
    pin_mode = PIN_MODE if pin_mode is None else pin_mode
    edges = BIN_EDGES if edges is None else edges
    ref = FREQ_REF if ref is None else ref
    idx = np.digitize(freq, np.asarray(edges, float)) - 1
    e = np.asarray(edges, float); ctr = (e[:-1] + e[1:]) / 2.0
    rows = []
    for i in range(len(ctr)):
        m = idx == i
        if not m.any():
            continue
        life = float(np.mean(t_adj[m]) if BIN_STAT == "mean" else np.median(t_adj[m]))
        rows.append({"f": ctr[i], "n": int(m.sum()), "life": life})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    good = out[out["n"] >= MIN_BIN_N]
    if pin_mode == "min" and len(good):
        ref_life = float(good["life"].max())          # longest life = the valley floor
    else:
        ref_life = float(np.interp(ref, out["f"], out["life"]))
    out["theta"] = (ref_life / out["life"]) ** beta
    return out


def _ya_inputs(cached: pd.DataFrame | None = None, as_of=None):
    """Ya frame + β + the v2 Ql layer removed as an AFT offset."""
    from analysis.workflows.production_risk import ya_k1k2_hybrid as YA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = YA.run(rate="Ql", as_of=as_of, cached=cached, write=False)
    m = run.model
    beta = float(m.baseline.beta1)
    ya = run.ya.copy()
    for c in ("ql", "freq_run", "t_cal"):
        ya[c] = pd.to_numeric(ya[c], errors="coerce")
    ya = ya.dropna(subset=["ql", "freq_run", "t_cal"])
    ya = ya[ya["t_cal"] > 0].copy()
    theta_ql = np.asarray(m.theta_at("Ql", ya["ql"].to_numpy(float)), float)
    ya["t_adj"] = ya["t_cal"].to_numpy(float) * theta_ql ** (1.0 / beta)
    poly = pd.DataFrame({"f": POLYLINE_GRID,
                         "theta": np.asarray(m.theta_freq_hz(POLYLINE_GRID), float)})
    return ya, beta, poly


def derive(*, cached: pd.DataFrame | None = None, as_of=None,
           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> EmpiricalCurves:
    """Both empirical curves plus a percentile bootstrap band (runs resampled, θ_Ql fixed)."""
    ya, beta, poly = _ya_inputs(cached=cached, as_of=as_of)
    f = ya["freq_run"].to_numpy(float)
    t = ya["t_adj"].to_numpy(float)
    ev = ya["event"].to_numpy(float) == 1

    curves = {"all_pulls": _bin_curve(f, t, beta),
              "failures_only": _bin_curve(f[ev], t[ev], beta)}

    rng = np.random.default_rng(seed)
    draws = {k: {c: [] for c in _centres()} for k in curves}
    for _ in range(n_boot):
        i = rng.integers(0, len(f), len(f))
        fb, tb, eb = f[i], t[i], ev[i]
        for name, (ff, tt) in (("all_pulls", (fb, tb)),
                               ("failures_only", (fb[eb], tb[eb]))):
            c = _bin_curve(ff, tt, beta)
            if c.empty:
                continue
            for _, r in c.iterrows():
                draws[name][r["f"]].append(r["theta"])
    rows = []
    for name, per in draws.items():
        for fc, vals in per.items():
            if len(vals) < 20:
                continue
            v = np.asarray(vals, float)
            rows.append({"variant": name, "f": fc,
                         "lo": float(np.percentile(v, 2.5)),
                         "hi": float(np.percentile(v, 97.5)),
                         "p25": float(np.percentile(v, 25.0)),
                         "p75": float(np.percentile(v, 75.0)),
                         "n_boot": len(v)})
    return EmpiricalCurves(all_pulls=curves["all_pulls"],
                           failures_only=curves["failures_only"],
                           band=pd.DataFrame(rows), beta=beta, n_runs=len(ya),
                           polyline=poly)


def run(*, cached: pd.DataFrame | None = None, as_of=None, write: bool = True,
        n_boot: int = N_BOOT) -> EmpiricalCurves:
    """Derive and cache — the Streamlit explorer reads these tables rather than refitting Ya."""
    c = derive(cached=cached, as_of=as_of, n_boot=n_boot)
    if write:
        out = results_dir(SLUG) / "tables"
        out.mkdir(parents=True, exist_ok=True)
        c.all_pulls.assign(variant="all_pulls").to_csv(
            out / "empirical_all_pulls.csv", index=False, encoding="utf-8-sig")
        c.failures_only.assign(variant="failures_only").to_csv(
            out / "empirical_failures_only.csv", index=False, encoding="utf-8-sig")
        c.band.to_csv(out / "bootstrap_band.csv", index=False, encoding="utf-8-sig")
        if c.polyline is not None:
            c.polyline.to_csv(out / "ya_polyline.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"beta": c.beta, "n_runs": c.n_runs, "pin_mode": PIN_MODE,
                       "freq_ref": FREQ_REF, "n_boot": n_boot,
                       "bin_edges": "|".join(str(x) for x in BIN_EDGES)}]).to_csv(
            out / "meta.csv", index=False, encoding="utf-8-sig")
    return c


def load_cached(date: str | None = None) -> EmpiricalCurves | None:
    """Most recent cached derivation, or None."""
    from analysis.paths import RESULTS_ROOT

    base = Path(RESULTS_ROOT) / SLUG
    if not base.exists():
        return None
    days = sorted(p for p in base.iterdir() if p.is_dir())
    if not days:
        return None
    d = base / date if date else days[-1]
    t = d / "tables"
    try:
        meta = pd.read_csv(t / "meta.csv").iloc[0]
        return EmpiricalCurves(
            all_pulls=pd.read_csv(t / "empirical_all_pulls.csv"),
            failures_only=pd.read_csv(t / "empirical_failures_only.csv"),
            band=pd.read_csv(t / "bootstrap_band.csv"),
            beta=float(meta["beta"]), n_runs=int(meta["n_runs"]),
            polyline=(pd.read_csv(t / "ya_polyline.csv")
                      if (t / "ya_polyline.csv").exists() else None))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The controllable deployment form (mirrors what the explorer edits)
# ---------------------------------------------------------------------------
#: Saturation term.  ``b2 > 0`` ⇒ the denominator has no real root, so unlike the shipped Ya
#: rational (pole at 31.7 Hz, inside Vt's operating range) this form can never blow up.
B2_DEFAULT = 0.0016
FREQ_CLAMP = (35.0, 70.0)


def theta_controlled(f, m60: float, s: float, b2: float = B2_DEFAULT,
                     clamp: tuple = FREQ_CLAMP, floor: float | None = None,
                     shift: float = 0.0, level: float = 1.0) -> np.ndarray:
    """θ(f) = level · base(f − shift), base(u) = (1 + a₁u + a₂u²)/(1 + b₂u²), u = f − 50.

    Four controls:

    * ``m60``   — θ at 60 Hz of the *unshifted, unlevelled* base (exact);
    * ``s``     — under-speed share, base(40) = 1 + s·(m60 − 1); ``s > 1`` ⇒ under-speed
      costs more than over-speed;
    * ``shift`` — moves the whole curve along the frequency axis, so the optimum sits at
      50 + shift rather than 50.  The m60 anchor travels with it (it lands at 60 + shift);
    * ``level`` — moves the whole curve up/down; θ at the optimum becomes ``level``, not 1.

    ``b2 > 0`` keeps the denominator rootless, so unlike the shipped Ya rational (pole at
    31.7 Hz, inside Vt's operating range) this can never blow up.
    """
    m40 = 1.0 + s * (m60 - 1.0)
    D = 1.0 + 100.0 * b2
    a1 = (m60 - m40) * D / 20.0
    a2 = ((m60 + m40) * D - 2.0) / 200.0
    u = np.clip(np.asarray(f, float), clamp[0], clamp[1]) - 50.0 - shift
    th = level * (1.0 + a1 * u + a2 * u ** 2) / (1.0 + b2 * u ** 2)
    return np.maximum(th, floor) if floor is not None else th


# ---------------------------------------------------------------------------
# Kpod: same derivation, plus a "stretch to anchors" controllable form
# ---------------------------------------------------------------------------
def derive_kpod(*, cached: pd.DataFrame | None = None, as_of=None,
                n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> EmpiricalCurves:
    """Empirical Kpod layer + bootstrap band, by the same recipe as :func:`derive`."""
    ya, beta, _ = _ya_inputs(cached=cached, as_of=as_of)
    ya["kpod_run"] = pd.to_numeric(ya["kpod_run"], errors="coerce")
    ya = ya.dropna(subset=["kpod_run"])
    k = ya["kpod_run"].to_numpy(float)
    t = ya["t_adj"].to_numpy(float)
    ev = ya["event"].to_numpy(float) == 1
    kw = dict(edges=KPOD_BIN_EDGES, ref=KPOD_REF)
    curves = {"all_pulls": _bin_curve(k, t, beta, **kw),
              "failures_only": _bin_curve(k[ev], t[ev], beta, **kw)}

    e = np.asarray(KPOD_BIN_EDGES, float); ctr = (e[:-1] + e[1:]) / 2.0
    rng = np.random.default_rng(seed)
    draws = {n: {c: [] for c in ctr} for n in curves}
    for _ in range(n_boot):
        i = rng.integers(0, len(k), len(k))
        kb, tb, eb = k[i], t[i], ev[i]
        for name, (kk, tt) in (("all_pulls", (kb, tb)), ("failures_only", (kb[eb], tb[eb]))):
            c = _bin_curve(kk, tt, beta, **kw)
            if c.empty:
                continue
            for _, r in c.iterrows():
                draws[name][r["f"]].append(r["theta"])
    rows = []
    for name, per in draws.items():
        for fc, vals in per.items():
            if len(vals) < 20:
                continue
            v = np.asarray(vals, float)
            rows.append({"variant": name, "f": fc, "lo": float(np.percentile(v, 2.5)),
                         "hi": float(np.percentile(v, 97.5)),
                         "p25": float(np.percentile(v, 25.0)),
                         "p75": float(np.percentile(v, 75.0)), "n_boot": len(v)})
    from analysis.workflows.production_risk import vt_composed_model as CM
    poly = pd.DataFrame({"f": KPOD_POLYLINE_GRID,
                         "theta": np.asarray(CM.theta_kpod(KPOD_POLYLINE_GRID), float)})
    return EmpiricalCurves(all_pulls=curves["all_pulls"], failures_only=curves["failures_only"],
                           band=pd.DataFrame(rows), beta=beta, n_runs=len(ya), polyline=poly)


def run_kpod(*, cached: pd.DataFrame | None = None, as_of=None, write: bool = True,
             n_boot: int = N_BOOT) -> EmpiricalCurves:
    c = derive_kpod(cached=cached, as_of=as_of, n_boot=n_boot)
    if write:
        out = results_dir(SLUG) / "tables"
        out.mkdir(parents=True, exist_ok=True)
        c.all_pulls.to_csv(out / "kpod_all_pulls.csv", index=False, encoding="utf-8-sig")
        c.failures_only.to_csv(out / "kpod_failures_only.csv", index=False, encoding="utf-8-sig")
        c.band.to_csv(out / "kpod_bootstrap_band.csv", index=False, encoding="utf-8-sig")
        c.polyline.to_csv(out / "kpod_shipped.csv", index=False, encoding="utf-8-sig")
    return c


def load_cached_kpod(date: str | None = None) -> EmpiricalCurves | None:
    from analysis.paths import RESULTS_ROOT

    base = Path(RESULTS_ROOT) / SLUG
    if not base.exists():
        return None
    days = sorted(p for p in base.iterdir() if p.is_dir())
    if not days:
        return None
    t = (base / date if date else days[-1]) / "tables"
    try:
        return EmpiricalCurves(
            all_pulls=pd.read_csv(t / "kpod_all_pulls.csv"),
            failures_only=pd.read_csv(t / "kpod_failures_only.csv"),
            band=pd.read_csv(t / "kpod_bootstrap_band.csv"),
            beta=float(pd.read_csv(t / "meta.csv").iloc[0]["beta"]), n_runs=0,
            polyline=pd.read_csv(t / "kpod_shipped.csv"))
    except Exception:
        return None


def theta_kpod_controlled(k, left: float, right: float,
                          clamp: tuple = (0.2, 1.5)) -> np.ndarray:
    """The shipped Kpod bathtub, STRETCHED so θ(0.2) = ``left`` and θ(1.2) = ``right``.

    Shape is preserved exactly; only the depth of each arm is rescaled about the bathtub
    floor, so the optimum stays where it is (k ≈ 0.875) and the curve stays smooth::

        θ(k) = 1 + (θ_base(k) − 1) · (target − 1) / (θ_base(anchor) − 1)

    applied with the left anchor below the floor and the right anchor above it.
    """
    from analysis.workflows.production_risk import vt_composed_model as CM

    kk = np.clip(np.asarray(k, float), clamp[0], clamp[1])
    base = np.asarray(CM.theta_kpod(kk), float)
    grid = np.linspace(clamp[0], clamp[1], 2000)
    kmin = float(grid[int(np.argmin(np.asarray(CM.theta_kpod(grid), float)))])
    bl = float(CM.theta_kpod(KPOD_ANCHOR_LEFT)) - 1.0
    br = float(CM.theta_kpod(KPOD_ANCHOR_RIGHT)) - 1.0
    sl = (left - 1.0) / bl if abs(bl) > 1e-9 else 0.0
    sr = (right - 1.0) / br if abs(br) > 1e-9 else 0.0
    scale = np.where(kk <= kmin, sl, sr)
    return 1.0 + (base - 1.0) * scale


#: Default plateau for the arm-wise frequency form: the empirical valley sits at 51.5–54 Hz,
#: and 50 is the model reference, so the band opens at 50.
FREQ_PLATEAU = (50.0, 53.0)
#: Anchors the two arms are pinned to.
FREQ_ANCHOR_LO, FREQ_ANCHOR_HI = 40.0, 60.0


def theta_freq_arms(f, left40: float, right60: float,
                    plateau: tuple = FREQ_PLATEAU, b2: float = B2_DEFAULT,
                    clamp: tuple = FREQ_CLAMP,
                    shape_lo: float = 2.0, shape_hi: float = 2.0) -> np.ndarray:
    """Frequency layer with INDEPENDENT arms, mirroring the Kpod form.

        f < p_lo :  u = p_lo − f,  θ = 1 + c_L·u^pL/(1 + b₂u^pL),  c_L from θ(40) = ``left40``
        p_lo…p_hi:  θ = 1
        f > p_hi :  u = f − p_hi,  θ = 1 + c_R·u^pR/(1 + b₂u^pR),  c_R from θ(60) = ``right60``

    Four independent controls per curve: the two **anchors** set how bad each extreme is, the
    **plateau** sets the safe band, and ``shape_lo`` / ``shape_hi`` set the **curvature of each
    arm** — which is what ``b2`` does *not* do.  ``b2`` only saturates the far tail, leaving the
    approach out of the plateau fixed as a parabola; the exponent changes where the cost is
    incurred:

    * ``shape = 1``   — linear: cost accrues immediately outside the band (corner at the join)
    * ``shape = 2``   — parabolic (default): slow start, smooth join
    * ``shape = 3-4`` — flat-bottomed: almost free near the band, then a steep wall

    Anchors are hit exactly for any shape.  ``shape > 1`` keeps zero slope at the join, so the
    curve stays smooth; ``shape = 1`` deliberately does not.
    """
    ff = np.clip(np.asarray(f, float), clamp[0], clamp[1])
    p_lo, p_hi = float(min(plateau)), float(max(plateau))
    pl, ph = max(float(shape_lo), 1.0), max(float(shape_hi), 1.0)
    u_lo, u_hi = abs(FREQ_ANCHOR_LO - p_lo), abs(FREQ_ANCHOR_HI - p_hi)
    c_l = (left40 - 1.0) * (1.0 + b2 * u_lo ** pl) / u_lo ** pl if u_lo > 1e-9 else 0.0
    c_r = (right60 - 1.0) * (1.0 + b2 * u_hi ** ph) / u_hi ** ph if u_hi > 1e-9 else 0.0
    out = np.ones_like(ff)
    lo_m, hi_m = ff < p_lo, ff > p_hi
    ul = np.where(lo_m, p_lo - ff, 0.0)
    ur = np.where(hi_m, ff - p_hi, 0.0)
    out = np.where(lo_m, 1.0 + c_l * ul ** pl / (1.0 + b2 * ul ** pl), out)
    out = np.where(hi_m, 1.0 + c_r * ur ** ph / (1.0 + b2 * ur ** ph), out)
    return out


# ---------------------------------------------------------------------------
# Three-anchor quadratic — the deployable frequency form
# ---------------------------------------------------------------------------
#: The three frequencies the layer is specified at.  40 / 50 / 60 Hz.
FREQ_QUAD_ANCHORS = (40.0, 50.0, 60.0)


def freq_poly_coeffs(t40: float, t50: float, t60: float, cubic: float = 0.0,
                     anchors: tuple = FREQ_QUAD_ANCHORS) -> dict:
    """Coefficients of ``ln θ = a + b·u + c·u² + d·u³``, ``u = f − 50``.

    The three anchors are **exact constraints**, so with ``h = 10``::

        a = ln t50
        c = (ln t40 + ln t60 − 2·ln t50) / (2h²)      ← independent of d
        b = (ln t60 − ln t40) / (2h) − d·h²
        d = cubic                                     ← the one free parameter

    ``a`` and ``c`` do not depend on ``d``; ``b`` absorbs it exactly, which is what keeps the
    three anchors pinned for any ``d``.

    ⚠ That does **not** mean ``d`` only touches the tails: ``u³`` is non-zero at ``u = ±5`` too,
    so the curve moves between the anchors as well (at ``d = 1.5e-3``, θ(45) goes 1.12 → 1.97).
    And it is violently non-linear outside them — the same ``d`` sends θ(70) to ~2·10⁴.  Set
    this parameter through :func:`cubic_from_fourth_anchor`, never by hand.
    """
    lo, hi = float(min(anchors)), float(max(anchors))
    half = 0.5 * (hi - lo)
    d = float(cubic)
    la, lb, lc = np.log(float(t40)), np.log(float(t50)), np.log(float(t60))
    return {"a": lb,
            "b": (lc - la) / (2.0 * half) - d * half ** 2,
            "c": (la + lc - 2.0 * lb) / (2.0 * half ** 2),
            "d": d, "u0": float(sorted(anchors)[1])}


#: Where the optional fourth anchor sits.  70 Hz is the upper clamp edge — the place the
#: cubic actually needs pinning, since that is where an unconstrained ``d`` runs away.
FREQ_ANCHOR4 = 70.0


def cubic_from_fourth_anchor(t40: float, t50: float, t60: float, t4: float,
                             f4: float = FREQ_ANCHOR4,
                             anchors: tuple = FREQ_QUAD_ANCHORS) -> float:
    """The ``cubic`` that makes the curve pass through ``θ(f4) = t4`` as well.

    This is the sane way to spend the n = 4 degree of freedom.  The raw coefficient is a bad
    control — it is dimensionally tiny, wildly non-linear in its effect, and has no reading in
    θ; a fourth **anchor** is one number in the same units as the other three, and four anchors
    determine a cubic uniquely.  ``t4`` equal to the parabola's own value at ``f4`` returns
    ``d = 0``, i.e. n = 3 exactly.

    Closed form.  Writing ``θ_quad`` for the n = 3 curve and ``u₄ = f4 − 50``::

        d = (ln t4 − ln θ_quad(f4)) / (u₄³ − h²·u₄)

    ``f4`` must not be one of the three anchors (the denominator vanishes at u₄ = 0, ±h).
    """
    lo, hi = float(min(anchors)), float(max(anchors))
    half = 0.5 * (hi - lo)
    u4 = float(f4) - float(sorted(anchors)[1])
    den = u4 ** 3 - half ** 2 * u4
    if abs(den) < 1e-9:
        raise ValueError(f"f4={f4} coincides with an anchor — the cubic is not identified there")
    quad = float(theta_freq_poly(f4, t40, t50, t60, 0.0, clamp=(-1e9, 1e9), anchors=anchors)[0])
    return float((np.log(float(t4)) - np.log(quad)) / den)


def theta_freq_poly(f, t40: float, t50: float, t60: float, cubic: float = 0.0,
                    clamp: tuple = FREQ_CLAMP,
                    anchors: tuple = FREQ_QUAD_ANCHORS) -> np.ndarray:
    """Frequency layer as a polynomial **in log θ** through θ(40), θ(50), θ(60).

    ``cubic = 0`` (default) is the **n = 3** form: three anchors determine a parabola uniquely,
    so exactly one curve fits and there is no residual shape knob — unlike
    :func:`theta_freq_arms`, where the anchors and the shape exponents interact and several
    parameter sets give the same three anchor values.

    ``cubic ≠ 0`` is the **n = 4** form: same three anchors, one free parameter controlling the
    tails outside 40–60 Hz (see :func:`freq_poly_coeffs` for why it cannot touch anything
    inside).  n = 3 is nested at ``cubic = 0``, so the two are directly comparable.

    Working in log θ rather than in θ matters at the clamp edges: a plain polynomial in θ
    fitted to a valley can cross zero before 35 Hz, which would be a negative hazard
    multiplier.  The exponential form is positive everywhere by construction and, being linear
    in the log, composes with the other θ layers the way they compose with each other.
    ``c > 0`` is a valley (life best near the middle), ``c < 0`` a ridge.

    ⚠ Setting ``t50 ≠ 1`` shifts the layer's **level**, not just its depth: the reference stops
    being neutral and the baseline η₀ must absorb the difference.

    ⚠ A polynomial is unbounded, and the cubic term more so than the quadratic.  The clamp is
    doing real work here, not tidying an edge case — :func:`freq_poly_edges` reports what it
    is holding back.
    """
    k = freq_poly_coeffs(t40, t50, t60, cubic, anchors)
    # atleast_1d: a scalar argument would otherwise come back as a 0-d array, which raises on
    # the ``[0]`` every caller writes — the app and the Excel port both evaluate one run at a time.
    u = np.clip(np.atleast_1d(np.asarray(f, float)), clamp[0], clamp[1]) - k["u0"]
    return np.exp(k["a"] + k["b"] * u + k["c"] * u * u + k["d"] * u ** 3)


def theta_freq_quadratic(f, t40: float, t50: float, t60: float,
                         clamp: tuple = FREQ_CLAMP,
                         anchors: tuple = FREQ_QUAD_ANCHORS) -> np.ndarray:
    """n = 3 form — :func:`theta_freq_poly` with the cubic term switched off."""
    return theta_freq_poly(f, t40, t50, t60, 0.0, clamp, anchors)


def freq_poly_edges(t40: float, t50: float, t60: float, cubic: float = 0.0,
                    clamp: tuple = FREQ_CLAMP) -> dict:
    """θ at the clamp edges, plus where the curve actually bottoms out inside the clamp.

    The optimum is found on a grid rather than from the vertex formula: a cubic has two
    turning points, and the useful question is where θ is smallest *within the clamp*, which
    may be a clamp edge rather than a stationary point.  It is also **not** guaranteed to be
    50 Hz just because θ(50) was set to 1.
    """
    grid = np.linspace(clamp[0], clamp[1], 1401)
    th = theta_freq_poly(grid, t40, t50, t60, cubic, clamp)
    k = freq_poly_coeffs(t40, t50, t60, cubic)
    i_lo, i_hi = int(np.argmin(th)), int(np.argmax(th))
    return {"theta_lo_edge": float(th[0]), "theta_hi_edge": float(th[-1]),
            "opt_hz": float(grid[i_lo]), "theta_opt": float(th[i_lo]),
            "worst_hz": float(grid[i_hi]), "theta_worst": float(th[i_hi]),
            "monotone": bool(np.all(np.diff(th) >= -1e-12) or np.all(np.diff(th) <= 1e-12)),
            "kind": "valley" if k["c"] > 0 else ("ridge" if k["c"] < 0 else "log-linear")}


def fit_freq_poly(f, theta, weights=None, clamp: tuple = FREQ_CLAMP,
                  degree: int = 2) -> dict:
    """Least-squares θ(40), θ(50), θ(60) [+ cubic] reproducing a target curve, in log θ.

    Translates an existing layer — a fitted free polyline, an empirical bin curve — into the
    numbers the deployable form is specified by.  ``degree=2`` gives n = 3 (cubic ≡ 0),
    ``degree=3`` gives n = 4.  Both are reported with the same weighted RMSE so the extra
    parameter can be judged rather than assumed.
    """
    f = np.asarray(f, float)
    y = np.log(np.asarray(theta, float))
    m = np.isfinite(f) & np.isfinite(y) & (f >= clamp[0]) & (f <= clamp[1])
    f, y = f[m], y[m]
    w = np.ones_like(f) if weights is None else np.asarray(weights, float)[m]
    u = f - FREQ_QUAD_ANCHORS[1]
    cols = [np.ones_like(u), u, u * u] + ([u ** 3] if degree >= 3 else [])
    X = np.column_stack(cols)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    a, b, c = coef[0], coef[1], coef[2]
    d = float(coef[3]) if degree >= 3 else 0.0
    half = 0.5 * (FREQ_QUAD_ANCHORS[2] - FREQ_QUAD_ANCHORS[0])
    pred = X @ coef
    return {"t40": float(np.exp(a - b * half + c * half ** 2 - d * half ** 3)),
            "t50": float(np.exp(a)),
            "t60": float(np.exp(a + b * half + c * half ** 2 + d * half ** 3)),
            "cubic": d, "n_par": int(len(coef)),
            "rmse_log": float(np.sqrt(np.average((y - pred) ** 2, weights=w))),
            "n": int(len(f))}


def fit_freq_quadratic(f, theta, weights=None, clamp: tuple = FREQ_CLAMP) -> dict:
    """n = 3 least-squares fit — :func:`fit_freq_poly` at ``degree=2``."""
    return fit_freq_poly(f, theta, weights, clamp, degree=2)
