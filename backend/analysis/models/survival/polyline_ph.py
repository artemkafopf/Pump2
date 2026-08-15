"""Shape-constrained **polyline proportional-hazard layers** on a Weibull baseline.

The θ-layer form the repo deploys: a covariate enters the hazard as a piecewise-linear
function **in log-θ**, pinned to θ=1 at an interpretable reference and *clamped* (flat)
outside the knot range.  Two shapes are supported, both enforced by construction rather
than by penalty, so the constraint can never be violated by the optimiser:

``"mono"``
    Cumulative **non-negative increments** left→right, then re-centred at the pin.  θ is
    non-decreasing in x (hazard rises with the covariate), θ=1 at the pin, and θ<1 below it.
    Used for a covariate with a known direction and no interior optimum (e.g. Ql).

``"tent"``
    0 at the pin, cumulative non-negative increments outward on **each** arm.  ⇒ θ≥1
    everywhere, θ=1 at the pin, and θ monotone on each arm ⇔ **life is maximal exactly at
    the reference and falls monotonically as you move away in either direction**.  Used for
    covariates with a design point (Kpod at 0.8, running frequency at nominal).

``"free"``
    Increments of **either sign**, cumulated left→right and re-centred at the pin.  Imposes no
    shape at all beyond piecewise-linearity in log-θ: the arm may rise, fall, or turn over
    wherever the data says.  This is the shape to use when you want to *ask* what the curve
    looks like rather than assert it — notably for Kpod and running frequency, where the
    ``tent``'s design-point assumption (optimum pinned at 0.8 / at nominal, θ≥1 on both arms)
    is structurally unable to represent a monotone effect and returns a spurious θ≡1 flat.
    Unlike the one-sided shapes, ``free`` does **not** rectify noise upward, so a wobble here
    really is noise; lean on the ridge and the bootstrap to read it.

Fit is a direct censored **Weibull-PH MLE** (L-BFGS-B with the increments bounded ≥ 0), with
free linear terms alongside (contractor dummies, etc.)::

    h(t | x) = (β0/η0)(t/η0)^(β0−1) · exp(Σ γ_j z_j) · Π_arms θ_arm(x_arm)

A per-arm **ridge on the increments** shrinks an arm toward flat θ≡1, which is how a *minor*
layer is kept minor while a primary arm carries signal.

At the ≥0 boundary the MLE has no usable asymptotic CI — an arm resting on the constraint has
a degenerate information matrix — so intervals must come from :func:`bootstrap_theta`
(cluster bootstrap over wells), never from the Hessian.

**Read a constrained arm carefully.**  Because the increments can only be non-negative, a
one-sided arm **rectifies sampling noise upward**: on data with no true effect the fitted θ
still lifts off 1 rather than scattering around it.  So a mildly rising tent arm is not by
itself evidence of an effect — the ridge is what holds it down and the bootstrap interval is
what tells you whether it is real.  Conversely, an arm sitting exactly at θ≡1 means the
constraint is *binding* (the data wanted to go the other way), not that the effect measured
zero.  ``tests/test_polyline_ph.py`` pins both behaviours.

Extracted from the Vt v3.2 hybrid so Ya (and later fields) reuse the machinery instead of
copying it; ``workflows.production_risk.vt_v32_hybrid`` keeps its own inlined copy because it
is already shipped into the Excel calculator and is deliberately not being re-fitted here —
``tests/test_polyline_ph.py`` asserts the two implementations agree elementwise.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MONO = "mono"
TENT = "tent"
FREE = "free"


# ---------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    """One shape-constrained covariate arm.

    Parameters
    ----------
    name : label used in outputs.
    column : source column in the modelling frame.
    knots : ascending knot locations; ``pin`` must be one of them.
    pin : reference value where θ ≡ 1.
    shape : ``"mono"``, ``"tent"`` or ``"free"`` (see the module docstring).
    ridge : L2 penalty on this arm's increments (0 = free, large = shrunk to θ≡1).  On a
        ``"free"`` arm the ridge is also what keeps successive segments from sawtoothing,
        so it is doing more work there than on a one-sided arm.
    guard : ``"mono"``/``"free"`` — hold θ flat below this x, refusing to extrapolate into a
        range where the low-x rows are too few to trust.  ``None`` disables it.
    """
    name: str
    column: str
    knots: tuple
    pin: float
    shape: str = TENT
    ridge: float = 6.0
    guard: float | None = None

    def __post_init__(self):
        if self.shape not in (MONO, TENT, FREE):
            raise ValueError(f"{self.name}: shape must be one of {MONO!r}, {TENT!r}, {FREE!r}")
        k = np.asarray(self.knots, float)
        if len(k) < 3 or np.any(np.diff(k) <= 0):
            raise ValueError(f"{self.name}: knots must be ascending and at least 3")
        if float(self.pin) not in [float(x) for x in self.knots]:
            raise ValueError(f"{self.name}: pin {self.pin} is not one of the knots")

    @property
    def pin_index(self) -> int:
        return [float(x) for x in self.knots].index(float(self.pin))

    @property
    def n_increments(self) -> int:
        # every shape spends one parameter per gap: mono/free cumulate left→right,
        # tent splits the same count into (#left) + (#right) arms outward from the pin.
        return len(self.knots) - 1

    @property
    def increment_bounds(self) -> tuple:
        """L-BFGS-B bounds for one increment — sign-constrained except on a ``free`` arm."""
        return (-4.0, 4.0) if self.shape == FREE else (0.0, 4.0)


def log_theta_from_increments(spec: ArmSpec, inc: np.ndarray) -> np.ndarray:
    """Log-θ at the arm's knots, built so the shape constraint holds by construction.

    On the one-sided shapes negative increments are clipped to 0 — the bounds already forbid
    them, but clipping keeps the function total so a caller can hand it any vector.  A
    ``free`` arm takes the increments as given, which is the whole point of it."""
    inc = np.asarray(inc, float)
    n, pin_ix = len(spec.knots), spec.pin_index
    if spec.shape == FREE:
        full = np.concatenate([[0.0], np.cumsum(inc)])
        return full - full[pin_ix]
    if spec.shape == MONO:
        full = np.concatenate([[0.0], np.cumsum(np.maximum(inc, 0.0))])
        return full - full[pin_ix]
    full = np.zeros(n)
    n_left = pin_ix
    c = 0.0
    for j, i in enumerate(range(pin_ix - 1, -1, -1)):        # outward to the left
        c += max(0.0, float(inc[j])); full[i] = c
    c = 0.0
    for j, i in enumerate(range(pin_ix + 1, n)):             # outward to the right
        c += max(0.0, float(inc[n_left + j])); full[i] = c
    return full


def eval_polyline(x, knots, values) -> np.ndarray:
    """Piecewise-linear interpolation **clamped** outside the knot range (flat extrapolation).

    Deliberately linear, never splined: a smooth interpolant would invent curvature between
    knots that the fit never estimated."""
    k = np.asarray(knots, float)
    x = np.atleast_1d(np.asarray(x, float))
    return np.interp(np.clip(x, k[0], k[-1]), k, np.asarray(values, float))


def apply_guard_and_pin(spec: ArmSpec, theta: np.ndarray) -> np.ndarray:
    """Apply the ``guard`` (flat below it) and re-pin θ=1 at the reference."""
    th = np.asarray(theta, float).copy()
    k = np.asarray(spec.knots, float)
    if spec.guard is not None:
        th[k < float(spec.guard)] = float(np.interp(float(spec.guard), k, th))
    return th / float(np.interp(spec.pin, k, th))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
@dataclass
class PolylinePHFit:
    """Result of :func:`fit_polyline_ph` — θ at the knots plus the free linear terms.

    ``beta0``/``eta0`` are the PH fit's own Weibull baseline.  When the deployable baseline is
    something else (a k1/k2 mixture, say), they are a **nuisance**: they absorb the scale so
    the θ's are estimated correctly, but the deployed curve uses the other baseline."""
    arms: tuple
    theta_knots: dict          # arm name -> θ at that arm's knots (guard/pin applied)
    linear: dict               # term name -> log-HR coefficient
    beta0: float
    eta0: float
    loglik: float              # data log-likelihood, ridge EXCLUDED
    n: int
    events: int
    concordance: float
    ridge: dict = dc_field(default_factory=dict)

    def spec(self, name: str) -> ArmSpec:
        return next(a for a in self.arms if a.name == name)

    def theta_at(self, name: str, x) -> np.ndarray:
        """θ(x) for one arm, clamped outside the knot range (an assumption, not a fit)."""
        a = self.spec(name)
        return np.exp(eval_polyline(x, a.knots, np.log(self.theta_knots[name])))

    def hr(self, term: str) -> float:
        return float(np.exp(self.linear.get(term, 0.0)))


def _slices(arms) -> list[tuple[int, int]]:
    out, o = [], 0
    for a in arms:
        out.append((o, o + a.n_increments)); o += a.n_increments
    return out


def _nll(p, arms, slices, n_lin, t, e, Z, X, off_lp=None) -> float:
    beta0, ln_eta0 = p[0], p[1]
    if beta0 <= 0:
        return 1e18
    lp = Z @ p[2:2 + n_lin] if n_lin else np.zeros(len(t))
    if off_lp is not None:
        lp = lp + off_lp
    off = 2 + n_lin
    for a, (lo, hi) in zip(arms, slices):
        lt = log_theta_from_increments(a, p[off + lo:off + hi])
        lp = lp + eval_polyline(X[a.name], a.knots, lt)
    ln_t = np.log(t)
    base = np.clip(beta0 * (ln_t - ln_eta0), -700.0, 700.0)
    ll = (e * (lp + np.log(beta0) - ln_eta0 + (beta0 - 1.0) * (ln_t - ln_eta0))
          - np.exp(np.clip(lp, -700.0, 700.0)) * np.exp(base))
    v = -float(np.sum(ll))
    return v if np.isfinite(v) else 1e18


def _obj(p, arms, slices, n_lin, t, e, Z, X, off_lp=None) -> float:
    nll = _nll(p, arms, slices, n_lin, t, e, Z, X, off_lp)
    if nll >= 1e18:
        return 1e18
    off = 2 + n_lin
    pen = 0.0
    for a, (lo, hi) in zip(arms, slices):
        inc = p[off + lo:off + hi]
        pen += float(a.ridge) * float(np.dot(inc, inc))
    return nll + pen


def fit_polyline_ph(df: pd.DataFrame, *, clock: str, event_col: str, arms,
                    linear_terms=(), eta0_start: float = 400.0,
                    beta0_bounds: tuple = (0.05, 6.0),
                    offset: str | None = None) -> PolylinePHFit:
    """Constrained Weibull-PH MLE on complete-case rows (all arm columns + linear terms present).

    Multi-start L-BFGS-B; the increments are bounded so every shape constraint holds at every
    step of the optimisation, not just at the optimum.

    ``offset`` names a column of **known log-hazard** carried into the linear predictor with
    coefficient fixed at 1 — the way a layer estimated elsewhere (contractor levels fitted on a
    rate-overlap window; a shape borrowed from a thicker stratum) is held fixed while the rest
    of the model refits around it.  Rows with a null offset are dropped like any other."""
    from lifelines.utils import concordance_index

    arms = tuple(arms)
    linear_terms = tuple(linear_terms)
    need = [clock, event_col, *linear_terms] + [a.column for a in arms]
    if offset is not None:
        need.append(offset)
    d = df.dropna(subset=need).copy()
    if len(d) == 0:
        raise ValueError(
            f"no complete-case rows for {need} — an all-null column (commonly a mis-built "
            f"offset) silently produces a meaningless fit rather than an error")
    t = d[clock].to_numpy(float)
    e = d[event_col].to_numpy(float)
    Z = d[list(linear_terms)].to_numpy(float) if linear_terms else np.zeros((len(d), 0))
    X = {a.name: d[a.column].to_numpy(float) for a in arms}
    off_lp = d[offset].to_numpy(float) if offset is not None else None
    slices = _slices(arms)
    n_lin, n_inc = len(linear_terms), sum(a.n_increments for a in arms)

    inc_bounds = [b for a in arms for b in [a.increment_bounds] * a.n_increments]
    bounds = ([beta0_bounds, (np.log(5.0), np.log(5e4))]
              + [(-4.0, 4.0)] * n_lin + inc_bounds)
    # a free arm has no ≥0 wall to slide along, so seed it from both signs as well as flat
    inc_signed = [s for a in arms for s in
                  [(-0.05 if a.shape == FREE else 0.10)] * a.n_increments]
    starts = [
        np.array([0.9, np.log(eta0_start)] + [0.0] * n_lin + [0.0] * n_inc),
        np.array([1.1, np.log(eta0_start * 0.75)] + [0.2] * n_lin + [0.05] * n_inc),
        np.array([0.7, np.log(eta0_start * 1.25)] + [-0.1] * n_lin + [0.10] * n_inc),
        np.array([0.9, np.log(eta0_start)] + [0.0] * n_lin + inc_signed),
    ]
    args = (arms, slices, n_lin, t, e, Z, X, off_lp)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            r = minimize(_obj, x0, args=args, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
    p = best.x

    off = 2 + n_lin
    theta, lp = {}, (Z @ p[2:2 + n_lin] if n_lin else np.zeros(len(t)))
    if off_lp is not None:
        lp = lp + off_lp
    for a, (lo, hi) in zip(arms, slices):
        lt = log_theta_from_increments(a, p[off + lo:off + hi])
        lp = lp + eval_polyline(X[a.name], a.knots, lt)
        theta[a.name] = apply_guard_and_pin(a, np.exp(lt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            conc = float(concordance_index(t, -lp, e))
        except ZeroDivisionError:       # no admissable pairs (e.g. a fully-tied predictor)
            conc = float("nan")
    return PolylinePHFit(
        arms=arms, theta_knots=theta,
        linear={c: float(v) for c, v in zip(linear_terms, p[2:2 + n_lin])},
        beta0=float(p[0]), eta0=float(np.exp(p[1])),
        loglik=-_nll(p, *args), n=len(d), events=int(e.sum()), concordance=round(conc, 4),
        ridge={a.name: float(a.ridge) for a in arms},
    )


# ---------------------------------------------------------------------------
# Cluster bootstrap (the only honest interval at the ≥0 boundary)
# ---------------------------------------------------------------------------
def bootstrap_theta(df: pd.DataFrame, *, clock: str, event_col: str, arms,
                    linear_terms=(), cluster: str, n_boot: int = 200, seed: int = 7,
                    min_events: int = 30, offset: str | None = None) -> dict:
    """Cluster bootstrap over ``cluster`` (wells — runs of one well are dependent).

    Returns ``{key: (lo, hi)}`` 2.5/97.5 percentiles for every knot θ (``"<arm>@<knot>"``),
    every linear HR (``"hr_<term>"``) and the PH baseline.  Keys with too few successful
    resamples are omitted rather than reported on thin evidence."""
    arms = tuple(arms)
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    acc: dict[str, list[float]] = {}
    for _ in range(n_boot):
        draw = rng.choice(keys, size=len(keys), replace=True)
        b = pd.concat([groups[k] for k in draw], ignore_index=True)
        if int(b[event_col].sum()) < min_events:
            continue
        try:
            f = fit_polyline_ph(b, clock=clock, event_col=event_col, arms=arms,
                                linear_terms=linear_terms, offset=offset)
        except Exception:
            continue
        for a in arms:
            for kn, th in zip(a.knots, f.theta_knots[a.name]):
                acc.setdefault(f"{a.name}@{kn:g}", []).append(float(th))
        for term in linear_terms:
            acc.setdefault(f"hr_{term}", []).append(f.hr(term))
        acc.setdefault("beta0", []).append(f.beta0)
        acc.setdefault("eta0", []).append(f.eta0)
    out = {}
    for k, v in acc.items():
        vv = np.array([x for x in v if np.isfinite(x)])
        if len(vv) >= max(20, n_boot // 5):
            out[k] = (round(float(np.percentile(vv, 2.5)), 4),
                      round(float(np.percentile(vv, 97.5)), 4))
    return out
