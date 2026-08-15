"""Covariate hazard layers on the failure Weibull (proportional-hazards θ).

Each pump run draws a *rate* (Ql) and a *running frequency* bin, and the two
carry a PH multiplier ``θ`` on the failure hazard.  A PH multiplier rescales the
Weibull scale only::

    h(t) = θ · h₀(t)      ⇔      η_eff = η₀ · θ^(−1/β₀)

so the simulation just draws that run's failure time from ``Weibull(β₀, η_eff)``.
The shape β is untouched, which is what makes the layers *proportional*.

Three shapes, all anchored on the bin the layer is referenced to:

``ql``    monotone increasing — ``ratio`` × more hazard in the top bin than the
          bottom one.  ``log`` is the power law ``(Ql/Ql_lo)^c`` (the form the
          Vt/Ya θ_Ql layers actually take); ``linear`` is linear in Ql.
``freq``  U-shaped around 50 Hz — ``θ = 1 + k·(1 − f/50)²`` with ``k`` set so the
          worst endpoint (40 or 60 Hz) reaches ``edge``× the 50 Hz hazard.
``well``  **individual-well frailty**: a geometric θ ramp across the fleet whose
          ends span ``ratio`` = θ_max / θ_min.  Its covariate axis is not a
          measurement but the well's own percentile in that spread — it stands
          for everything about a well that the model does not carry.

Bins are equal-width on the covariate range and, for now, equally likely
(``uniform``); the per-bin population mix is a separate knob for later.

Per-run vs per-well
-------------------
``ql`` and ``freq`` are redrawn **per run** — a slot gets a new rate/frequency
every time its pump is replaced.  ``well`` is drawn **once per well slot** and
persists across every replacement in it (``per_well=True``), which is what makes
it frailty rather than noise: a bad well stays bad, so its runs are positively
correlated and the pooled fit sees a heterogeneous, non-Weibull population.

Centering
---------
With ``center=True`` (the default) θ is divided by its population mean
``Σ pᵢ θᵢ``, so switching a layer on *redistributes* hazard between bins without
moving the fleet average.  Without it, a layer that runs 1 → 2 also makes the
whole fleet worse, and "layer on" would be confounded with "shorter η".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Layer keys.
LAYER_QL = "ql"
LAYER_FREQ = "freq"
LAYER_WELL = "well"

# Monotone shapes for the Ql layer.
SHAPE_LOG = "log"
SHAPE_LINEAR = "linear"
QL_SHAPES = (SHAPE_LOG, SHAPE_LINEAR)

# Per-bin population mix (only uniform for now; kept as a named constant so the
# "which wells sit in which bin" question can be added without a schema change).
MIX_UNIFORM = "uniform"
BIN_MIXES = (MIX_UNIFORM,)

# Vertex of the frequency U-shape: the grid/synchronous speed the fleet is
# designed around.  θ(50 Hz) = 1 by construction.
FREQ_REF_HZ = 50.0

# The well-frailty layer has no physical covariate — its axis is the well's own
# rank in the θ spread, in percent.
WELL_AXIS = (0.0, 100.0)


@dataclass(frozen=True)
class LayerSpec:
    """One resolved hazard layer: bin geometry + the PH multiplier per bin."""

    key: str
    label: str            # axis label, e.g. "Ql (m³/d)"
    edges: np.ndarray     # n_bins + 1 bin edges on the covariate
    centers: np.ndarray   # n_bins bin centers (the covariate value a run gets)
    theta: np.ndarray     # n_bins PH multipliers (already centered if requested)
    probs: np.ndarray     # n_bins assignment probabilities, sums to 1
    per_well: bool = False  # drawn once per well slot, not once per run

    @property
    def n_bins(self) -> int:
        return int(self.centers.size)

    def bin_labels(self, fmt: str = "{:.0f}") -> list[str]:
        return [f"{fmt.format(a)}–{fmt.format(b)}" for a, b in zip(self.edges[:-1], self.edges[1:])]


def bin_geometry(lo: float, hi: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width bin ``(edges, centers)`` on ``[lo, hi]``."""
    if hi <= lo:
        raise ValueError("hazard layer needs hi > lo")
    if n_bins < 1:
        raise ValueError("hazard layer needs at least one bin")
    edges = np.linspace(float(lo), float(hi), int(n_bins) + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def ql_theta(ql: np.ndarray, lo: float, hi: float, ratio: float, shape: str = SHAPE_LOG) -> np.ndarray:
    """Monotone θ from 1 at ``lo`` to ``ratio`` at ``hi``.

    ``log``    → power law ``(ql/lo)^c``, ``c = ln(ratio)/ln(hi/lo)``.
    ``linear`` → ``1 + (ratio − 1)·(ql − lo)/(hi − lo)``.

    ``lo``/``hi`` are the *anchors* the ratio is measured between, not
    necessarily the range endpoints — :func:`make_ql_layer` anchors on the first
    and last bin **centers**, so the ratio holds between the bins a run can
    actually be assigned to.
    """
    ql = np.asarray(ql, dtype=float)
    if hi <= lo:
        return np.ones_like(ql)
    if shape == SHAPE_LINEAR:
        return 1.0 + (ratio - 1.0) * (ql - lo) / (hi - lo)
    if shape != SHAPE_LOG:
        raise ValueError(f"ql shape must be one of {QL_SHAPES}")
    if lo <= 0:
        raise ValueError("log-shaped Ql layer needs lo > 0")
    return np.power(ql / lo, np.log(ratio) / np.log(hi / lo))


def freq_theta(freq: np.ndarray, lo: float, hi: float, edge: float,
               ref: float = FREQ_REF_HZ) -> np.ndarray:
    """U-shaped θ = ``1 + k·(1 − f/ref)²``, worst endpoint reaching ``edge``×.

    ``k`` is calibrated on whichever of ``lo``/``hi`` sits further from ``ref``,
    so a symmetric range (40–60 around 50) puts both ends at exactly ``edge``.
    """
    freq = np.asarray(freq, dtype=float)
    u = np.array([1.0 - lo / ref, 1.0 - hi / ref]) ** 2
    span = float(np.max(u))
    if span <= 0:
        return np.ones_like(freq)
    k = (edge - 1.0) / span
    return 1.0 + k * (1.0 - freq / ref) ** 2


def well_theta(pct: np.ndarray, lo: float, hi: float, ratio: float) -> np.ndarray:
    """Geometric θ ramp across the well population: ``ratio`` = θ(hi) / θ(lo).

    Geometric (equal steps in log θ) rather than linear because a frailty is a
    *multiplier*: "twice the hazard" has to mean the same thing at both ends of
    the fleet, and it is the spread of **log θ** that the marginal shape reacts
    to.  ``pct`` is the well's rank in the spread, in percent — a bookkeeping
    axis, not a measurement.
    """
    pct = np.asarray(pct, dtype=float)
    if hi <= lo or ratio <= 0:
        return np.ones_like(pct)
    return np.power(float(ratio), (pct - lo) / (hi - lo))


def _mix_probs(n_bins: int, mix: str = MIX_UNIFORM) -> np.ndarray:
    if mix != MIX_UNIFORM:
        raise ValueError(f"bin mix must be one of {BIN_MIXES}")
    return np.full(int(n_bins), 1.0 / int(n_bins))


def _center(theta: np.ndarray, probs: np.ndarray, center: bool) -> np.ndarray:
    if not center:
        return theta
    mean = float(np.sum(probs * theta))
    return theta / mean if mean > 0 else theta


def make_ql_layer(lo: float, hi: float, n_bins: int, ratio: float, shape: str = SHAPE_LOG,
                  mix: str = MIX_UNIFORM, center: bool = True) -> LayerSpec:
    """Monotone Ql layer with ``theta[-1] / theta[0] == ratio`` between the bins.

    The ratio is anchored on the first and last bin *centers* (the Ql values a
    run is actually assigned), so "2× hazard from the low bin to the high bin"
    is exactly what the simulation gets.
    """
    edges, centers = bin_geometry(lo, hi, n_bins)
    probs = _mix_probs(n_bins, mix)
    theta = _center(ql_theta(centers, float(centers[0]), float(centers[-1]), ratio, shape),
                    probs, center)
    return LayerSpec(LAYER_QL, "Ql (m³/d)", edges, centers, theta, probs)


def make_freq_layer(lo: float, hi: float, n_bins: int, edge: float, mix: str = MIX_UNIFORM,
                    center: bool = True, ref: float = FREQ_REF_HZ) -> LayerSpec:
    """U-shaped frequency layer, ``edge``× hazard at the ``lo`` / ``hi`` Hz ends.

    Unlike the Ql layer, ``edge`` is anchored on the *frequency range* endpoints
    (40 and 60 Hz), because that is where the physical U-shape is quoted.  Bin
    centers sit inside that range, so the realised θ spread across bins is
    milder than ``edge`` — with 5 bins on 40–60 the extreme bins are at 42/58 Hz
    and reach ≈1.64× rather than 2×.
    """
    edges, centers = bin_geometry(lo, hi, n_bins)
    probs = _mix_probs(n_bins, mix)
    theta = _center(freq_theta(centers, lo, hi, edge, ref), probs, center)
    return LayerSpec(LAYER_FREQ, "Frequency (Hz)", edges, centers, theta, probs)


def make_well_layer(n_bins: int, ratio: float, mix: str = MIX_UNIFORM,
                    center: bool = True) -> LayerSpec:
    """Individual-well frailty layer with ``theta[-1] / theta[0] == ratio``.

    Each well slot draws one of ``n_bins`` frailty groups **once, for life** —
    every pump ever installed in that slot carries the same θ.  The ratio is
    anchored on the first and last bin centers, so "θ spread ×4" is exactly the
    max/min a well can be assigned.
    """
    edges, centers = bin_geometry(WELL_AXIS[0], WELL_AXIS[1], n_bins)
    probs = _mix_probs(n_bins, mix)
    theta = _center(well_theta(centers, float(centers[0]), float(centers[-1]), ratio),
                    probs, center)
    return LayerSpec(LAYER_WELL, "well frailty rank (%)", edges, centers, theta, probs,
                     per_well=True)


def theta_design(cfg, spec: LayerSpec, values: np.ndarray) -> np.ndarray:
    """The layer's continuous (uncentered) θ shape evaluated at ``values``.

    The bins in :class:`LayerSpec` are this curve sampled at the bin centers and
    then divided by one centering constant, so a caller that wants to draw the
    smooth design shape next to the bin markers only has to recover that
    constant.
    """
    if spec.key == LAYER_QL:
        return ql_theta(values, float(spec.centers[0]), float(spec.centers[-1]),
                        cfg.ql_theta_ratio, cfg.ql_shape)
    if spec.key == LAYER_FREQ:
        return freq_theta(values, cfg.freq_lo, cfg.freq_hi, cfg.freq_theta_edge)
    if spec.key == LAYER_WELL:
        return well_theta(values, float(spec.centers[0]), float(spec.centers[-1]),
                          cfg.well_theta_ratio)
    raise ValueError(f"unknown layer key {spec.key!r}")


def layers_from_config(cfg) -> list[LayerSpec]:
    """The enabled hazard layers of a :class:`SimConfig`, in a stable order."""
    out: list[LayerSpec] = []
    if getattr(cfg, "well_layer_on", False):
        out.append(make_well_layer(cfg.well_bins, cfg.well_theta_ratio,
                                   cfg.bin_mix, cfg.hazard_center))
    if getattr(cfg, "ql_layer_on", False):
        out.append(make_ql_layer(cfg.ql_lo, cfg.ql_hi, cfg.ql_bins, cfg.ql_theta_ratio,
                                 cfg.ql_shape, cfg.bin_mix, cfg.hazard_center))
    if getattr(cfg, "freq_layer_on", False):
        out.append(make_freq_layer(cfg.freq_lo, cfg.freq_hi, cfg.freq_bins, cfg.freq_theta_edge,
                                   cfg.bin_mix, cfg.hazard_center))
    return out


def draw_bins(layers: list[LayerSpec], rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    """Independent per-run bin index for each layer."""
    return {
        spec.key: rng.choice(spec.n_bins, size=size, p=spec.probs)
        for spec in layers
    }


def theta_from_bins(layers: list[LayerSpec], bins: dict[str, np.ndarray]) -> np.ndarray:
    """Total PH multiplier = product of the per-layer θ (layers are independent)."""
    if not layers:
        first = next(iter(bins.values()), None)
        return np.ones(0 if first is None else first.size)
    total = np.ones(bins[layers[0].key].size)
    for spec in layers:
        total = total * spec.theta[bins[spec.key]]
    return total


def eta_effective(eta0: float, beta: float, theta: np.ndarray | float) -> np.ndarray:
    """PH multiplier θ folded into the Weibull scale: ``η₀ · θ^(−1/β)``."""
    return float(eta0) * np.power(np.asarray(theta, dtype=float), -1.0 / float(beta))


def theta_mixture(layers: list[LayerSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Joint ``(probs, theta)`` over the full cross-product of layer bins.

    With no layers this is the degenerate single component ``([1.0], [1.0])``,
    so callers can always integrate over the mixture unconditionally.
    """
    probs = np.array([1.0])
    theta = np.array([1.0])
    for spec in layers:
        probs = np.outer(probs, spec.probs).ravel()
        theta = np.outer(theta, spec.theta).ravel()
    return probs, theta


def split_layers(layers: list[LayerSpec]) -> tuple[list[LayerSpec], list[LayerSpec]]:
    """``(per_well, per_run)`` — the two draw schedules the simulator needs."""
    return ([s for s in layers if s.per_well], [s for s in layers if not s.per_well])


def theta_spread(layers: list[LayerSpec]) -> tuple[float, float]:
    """``(max/min, sd(log θ))`` of the population θ implied by ``layers``.

    The ratio is the readable headline; ``sd(log θ)`` is the quantity the
    marginal Weibull shape actually reacts to, and the axis the β-shape study
    reports heterogeneity on.
    """
    if not layers:
        return 1.0, 0.0
    probs, theta = theta_mixture(layers)
    log_t = np.log(theta)
    mean = float(np.sum(probs * log_t))
    sd = float(np.sqrt(max(np.sum(probs * (log_t - mean) ** 2), 0.0)))
    return float(theta.max() / theta.min()), sd


def describe(layers: list[LayerSpec]) -> str:
    """One-line summary for labels / captions."""
    if not layers:
        return "no hazard layers"
    parts = [f"{s.key}×{s.n_bins} (θ {s.theta.min():.2f}–{s.theta.max():.2f})" for s in layers]
    return " · ".join(parts)
