"""Configuration for the field-simulation experiments.

A single :class:`SimConfig` fully specifies one experiment: the field growth,
the failure Weibull, and (optionally) a competing workover process.  Every field
here is meant to be editable from the Streamlit app.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .hazard import BIN_MIXES, MIX_UNIFORM, QL_SHAPES, SHAPE_LOG

# One "year" of pump life.  eta = 1 year -> eta_fail_days = 365.
DAYS_PER_YEAR: float = 365.0

# ---------------------------------------------------------------------------
# Pump-age axes: never draw past five years, never step coarser than a month.
#
# Both rules come out of the same artefact.  Panels used to auto-scale the age axis from
# the LARGEST eta (2.5x it) and then lay a fixed number of points across the result, so a
# long-lived mode -- eta 20 000 d is easy to reach with beta < 1 -- stretched the axis to
# ~137 years and pushed the grid step to 125 days.  Everything below the first grid point
# was then one straight segment, which reads as a kink at roughly three months: an artefact
# of the grid, not of the model.  A pump running five years is already past the fleet's
# edge, so the axis is capped and the grid is dense where the curvature actually is.
AGE_MAX_DAYS: float = 1825.0     # five years -- the hard cap for any pump-age axis
AGE_STEP_DAYS: float = 30.0      # monthly, the coarsest step allowed
AGE_HEAD_DAYS: float = 90.0      # daily up to here: a beta < 1 mode moves fastest here


def age_grid(tmax: float = AGE_MAX_DAYS):
    """Age grid: daily over the first quarter, monthly after, never beyond five years."""
    import numpy as np

    tmax = float(min(max(tmax, AGE_STEP_DAYS), AGE_MAX_DAYS))
    head = np.arange(0.0, min(AGE_HEAD_DAYS, tmax) + 1.0, 1.0)
    tail = np.arange(AGE_HEAD_DAYS, tmax + AGE_STEP_DAYS, AGE_STEP_DAYS)
    # tmax is pinned as the last point: a monthly tail would otherwise stop up to 29 days
    # short of it and leave the curve hanging before the edge of the axis
    return np.unique(np.concatenate([head, tail[tail <= tmax], [tmax]]))

# Workover model identifiers.
WORKOVER_NONE = "none"
WORKOVER_DETERMINISTIC = "deterministic"   # planned pull at a fixed pump age
WORKOVER_STATISTICAL = "statistical"       # competing Weibull risk
WORKOVER_MODES = (WORKOVER_NONE, WORKOVER_DETERMINISTIC, WORKOVER_STATISTICAL)

# Basis for the "% of TTF" fractions used by the workover models.  Both workover models
# quote their age as a fraction, so this is what the fraction is a fraction OF.
TTF_REF_MEAN = "mean"   # mean of the failure Weibull = eta * Gamma(1 + 1/beta) = MRL(0)
TTF_REF_RMST = "rmst"   # RMST(0, tau): the mean restricted to the reporting horizon
TTF_REF_ETA = "eta"     # the characteristic life eta itself
TTF_REFERENCES = (TTF_REF_MEAN, TTF_REF_RMST, TTF_REF_ETA)

# Default competing failure modes when the multi-mode switch is turned on: an
# early-dominated electrical mode and a wear-out mechanical one, which is the
# pairing that makes the pooled shape misleading.
DEFAULT_FAIL_MODES: list[dict] = [
    {"name": "pump", "beta": 1.6, "eta_days": 520.0},
    {"name": "cable", "beta": 0.8, "eta_days": 900.0},
]


@dataclass
class SimConfig:
    """All parameters of one field-simulation experiment.

    Times are stored in whatever unit is natural for the field (days for pump
    ages, years for the horizon / snapshots) and converted internally.
    """

    # ── field growth ────────────────────────────────────────────────────────
    total_years: float = 20.0        # simulation horizon (10 yr growth + 10 yr plateau)
    ramp_years: float = 10.0         # years over which the fleet grows linearly
    plateau_wells: int = 100         # number of well-slots at plateau

    # ── failure Weibull (the "truth") ───────────────────────────────────────
    beta_fail: float = 1.0           # shape (1.0 = memoryless / exponential)
    eta_fail_days: float = 365.0     # scale = characteristic life in days

    # ── competing failure modes (equipment nodes) ───────────────────────────
    # With modes_on the single Weibull above is replaced by several independent
    # ones racing for the pump: T_fail = min over modes, and the mode that wins
    # is recorded.  See modes().
    modes_on: bool = False
    fail_modes: list[dict] = field(
        default_factory=lambda: [dict(m) for m in DEFAULT_FAIL_MODES]
    )

    # ── workover (competing) process ────────────────────────────────────────
    workover_mode: str = WORKOVER_NONE
    ttf_reference: str = TTF_REF_MEAN
    pm_fraction: float = 0.8         # deterministic: pull at pm_fraction * TTF_ref
    beta_wo: float = 1.3             # statistical: workover Weibull shape (>1)
    wo_eta_fraction: float = 0.8     # statistical: eta_wo = fraction * TTF_ref

    # ── covariate hazard layers (PH multipliers on the failure Weibull) ─────
    # Each run draws one bin per enabled layer; theta rescales only the scale:
    # eta_eff = eta_fail_days * theta ** (-1 / beta_fail).  See hazard.py.
    hazard_center: bool = True       # divide theta by its population mean
    bin_mix: str = MIX_UNIFORM       # per-bin population weights

    # individual-well frailty: drawn once per well slot and kept for every pump
    # ever installed in it (the other two layers redraw on each replacement)
    well_layer_on: bool = False
    well_bins: int = 5
    well_theta_ratio: float = 4.0    # theta(best well) / theta(worst well)

    ql_layer_on: bool = False
    ql_lo: float = 50.0              # m3/d
    ql_hi: float = 250.0
    ql_bins: int = 5
    ql_shape: str = SHAPE_LOG        # monotone: power law in Ql, or linear
    ql_theta_ratio: float = 2.0      # theta(top bin) / theta(bottom bin)

    freq_layer_on: bool = False
    freq_lo: float = 40.0            # Hz
    freq_hi: float = 60.0
    freq_bins: int = 5
    freq_theta_edge: float = 2.0     # theta at 40 / 60 Hz relative to 50 Hz

    # ── replacement / renewal (downtime = well not producing) ───────────────
    downtime_fail_days: float = 7.0       # well down after an unplanned failure
    downtime_workover_days: float = 3.0   # well down after a planned workover

    # ── snapshots & fitting ─────────────────────────────────────────────────
    rmst_tau_days: float = 730.0     # horizon for RMST(0, tau) life summary
    km_snapshot_years: list[float] = field(
        default_factory=lambda: [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    )
    n_fit_snapshots: int = 24        # points on the beta/eta-vs-time trend grid
    n_seeds: int = 30               # independent field realizations
    base_seed: int = 12345

    # ── derived quantities ──────────────────────────────────────────────────
    def modes(self) -> list[tuple[str, float, float]]:
        """``(name, beta, eta_days)`` per competing failure mode.

        With ``modes_on`` off this is the single baseline Weibull, so every
        caller downstream can loop over modes unconditionally and the one-mode
        path stays bit-for-bit what it was.
        """
        if not self.modes_on or not self.fail_modes:
            return [("failure", float(self.beta_fail), float(self.eta_fail_days))]
        return [
            (str(m.get("name") or f"mode {i + 1}"), float(m["beta"]), float(m["eta_days"]))
            for i, m in enumerate(self.fail_modes)
        ]

    def n_modes(self) -> int:
        return len(self.modes())

    def ttf_ref_days(self) -> float:
        """Reference TTF the workover fractions are applied to.

        Four names, three numbers.  ``mean`` is E[T] — which **is** MRL(0), since the
        mean residual life of a pump of age 0 is its whole expected life; the app offers
        them as one choice under both names rather than two that would silently agree.
        ``rmst`` is RMST(0, τ) = ∫₀^τ S — the same average truncated at the reporting
        horizon, so it is always below the mean (far below it when β < 1 drags a long
        tail past τ) and a fraction of it can never exceed τ.  ``eta`` is the scale.

        All three describe the **baseline** law at θ = 1: the workover programme is a
        fleet-wide policy, so it is quoted against the fleet's law, not against whatever
        θ an individual run happens to draw.

        With competing modes there is no single η, so the ``eta`` reference is
        generalised to its defining property — the age at which the *combined*
        failure law has S(t) = 1/e.  For one mode that is exactly η again.
        """
        modes = self.modes()
        if self.ttf_reference == TTF_REF_ETA:
            if len(modes) == 1:
                return float(modes[0][2])
            from .metrics import eta_equivalent_days

            return float(eta_equivalent_days(self))
        if self.ttf_reference == TTF_REF_RMST:
            from .metrics import life_given_theta

            return float(life_given_theta(self, 1.0)["rmst"])
        return self.true_mean_ttf_days()

    def pm_age_days(self) -> float:
        """Deterministic planned-pull age (days)."""
        return float(self.pm_fraction * self.ttf_ref_days())

    def wo_eta_days(self) -> float:
        """Statistical workover Weibull scale (days)."""
        return float(self.wo_eta_fraction * self.ttf_ref_days())

    def hazard_layers(self) -> list:
        """Resolved :class:`~.hazard.LayerSpec` list for the enabled layers."""
        from .hazard import layers_from_config

        return layers_from_config(self)

    def downtime_days(self, cause: str) -> float:
        return float(self.downtime_workover_days if cause == "workover" else self.downtime_fail_days)

    def horizon_days(self) -> float:
        return float(self.total_years * DAYS_PER_YEAR)

    def true_mean_ttf_days(self) -> float:
        """E[T_fail] of the baseline law (θ ≡ 1), across all competing modes."""
        modes = self.modes()
        if len(modes) == 1:
            _, beta, eta = modes[0]
            return float(eta * math.gamma(1.0 + 1.0 / beta))
        from .metrics import life_given_theta

        return float(life_given_theta(self, 1.0)["mean"])

    # ── (de)serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_years": self.total_years,
            "ramp_years": self.ramp_years,
            "plateau_wells": self.plateau_wells,
            "beta_fail": self.beta_fail,
            "eta_fail_days": self.eta_fail_days,
            "modes_on": self.modes_on,
            "fail_modes": [dict(m) for m in self.fail_modes],
            "workover_mode": self.workover_mode,
            "ttf_reference": self.ttf_reference,
            "pm_fraction": self.pm_fraction,
            "beta_wo": self.beta_wo,
            "wo_eta_fraction": self.wo_eta_fraction,
            "hazard_center": self.hazard_center,
            "bin_mix": self.bin_mix,
            "well_layer_on": self.well_layer_on,
            "well_bins": self.well_bins,
            "well_theta_ratio": self.well_theta_ratio,
            "ql_layer_on": self.ql_layer_on,
            "ql_lo": self.ql_lo,
            "ql_hi": self.ql_hi,
            "ql_bins": self.ql_bins,
            "ql_shape": self.ql_shape,
            "ql_theta_ratio": self.ql_theta_ratio,
            "freq_layer_on": self.freq_layer_on,
            "freq_lo": self.freq_lo,
            "freq_hi": self.freq_hi,
            "freq_bins": self.freq_bins,
            "freq_theta_edge": self.freq_theta_edge,
            "downtime_fail_days": self.downtime_fail_days,
            "downtime_workover_days": self.downtime_workover_days,
            "rmst_tau_days": self.rmst_tau_days,
            "km_snapshot_years": list(self.km_snapshot_years),
            "n_fit_snapshots": self.n_fit_snapshots,
            "n_seeds": self.n_seeds,
            "base_seed": self.base_seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimConfig":
        known = {f for f in cls().to_dict()}
        clean = {k: v for k, v in payload.items() if k in known}
        cfg = cls(**clean)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.workover_mode not in WORKOVER_MODES:
            raise ValueError(f"workover_mode must be one of {WORKOVER_MODES}")
        if self.ttf_reference not in TTF_REFERENCES:
            raise ValueError(f"ttf_reference must be one of {TTF_REFERENCES}")
        if self.beta_fail <= 0 or self.eta_fail_days <= 0:
            raise ValueError("beta_fail and eta_fail_days must be positive")
        if self.modes_on:
            if not self.fail_modes:
                raise ValueError("competing modes are on but no failure mode is defined")
            for i, (_, beta, eta) in enumerate(self.modes()):
                if beta <= 0 or eta <= 0:
                    raise ValueError(f"failure mode {i + 1} needs beta > 0 and eta_days > 0")
        if self.beta_wo <= 0:
            raise ValueError("beta_wo must be positive")
        if self.plateau_wells < 1:
            raise ValueError("plateau_wells must be >= 1")
        if self.total_years <= 0 or self.ramp_years <= 0:
            raise ValueError("total_years and ramp_years must be positive")
        if self.bin_mix not in BIN_MIXES:
            raise ValueError(f"bin_mix must be one of {BIN_MIXES}")
        if self.ql_shape not in QL_SHAPES:
            raise ValueError(f"ql_shape must be one of {QL_SHAPES}")
        if self.well_layer_on:
            if self.well_bins < 2:
                raise ValueError("the well frailty layer needs at least 2 bins")
            if self.well_theta_ratio <= 0:
                raise ValueError("well_theta_ratio must be positive")
        if self.ql_layer_on:
            if self.ql_lo <= 0 or self.ql_hi <= self.ql_lo:
                raise ValueError("Ql layer needs 0 < ql_lo < ql_hi")
            if self.ql_bins < 1 or self.ql_theta_ratio <= 0:
                raise ValueError("ql_bins must be >= 1 and ql_theta_ratio > 0")
        if self.freq_layer_on:
            if self.freq_hi <= self.freq_lo:
                raise ValueError("frequency layer needs freq_lo < freq_hi")
            if self.freq_bins < 1 or self.freq_theta_edge <= 0:
                raise ValueError("freq_bins must be >= 1 and freq_theta_edge > 0")

    def label(self) -> str:
        """Short human-readable description of the experiment."""
        modes = self.modes()
        law = (f"β{self.beta_fail:g}/η{self.eta_fail_days:g}d" if len(modes) == 1
               else f"{len(modes)} modes")
        base = f"{law}, {self.plateau_wells}w/{self.ramp_years:g}yr"
        layers = "".join(f" +{k}" for k, on in
                         (("well", self.well_layer_on), ("Ql", self.ql_layer_on),
                          ("freq", self.freq_layer_on)) if on)
        if self.workover_mode == WORKOVER_NONE:
            return f"failure-only{layers} · {base}"
        if self.workover_mode == WORKOVER_DETERMINISTIC:
            return f"PM@{self.pm_fraction:g}×TTF{layers} · {base}"
        return f"stat-WO β{self.beta_wo:g}/η{self.wo_eta_fraction:g}×TTF{layers} · {base}"
