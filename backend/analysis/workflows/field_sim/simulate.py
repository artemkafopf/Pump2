"""Renewal / competing-risks simulation of a growing ESP field.

One call to :func:`simulate_runs` produces the full life history (all pump runs)
of one field realization out to the horizon.  Snapshots are taken afterwards by
:func:`observe_at`, which observes that fixed realization at a given calendar
time — later runs are hidden, the current run is right-censored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DAYS_PER_YEAR,
    SimConfig,
    WORKOVER_DETERMINISTIC,
    WORKOVER_NONE,
    WORKOVER_STATISTICAL,
)
from .hazard import (
    LAYER_FREQ,
    LAYER_QL,
    LAYER_WELL,
    LayerSpec,
    draw_bins,
    eta_effective,
    split_layers,
    theta_from_bins,
)

# Per-layer bin columns, in the order the app and the bin table expect them.
BIN_COLUMNS = [f"{k}_bin" for k in (LAYER_WELL, LAYER_QL, LAYER_FREQ)]
RUN_COLUMNS = ["slot", "run_idx", "start", "end", "run_len", "cause", "downtime_after",
               "theta", "fail_mode", *BIN_COLUMNS]

# Bin index recorded for a run when that hazard layer is switched off.
NO_BIN = -1


def commission_days(cfg: SimConfig) -> np.ndarray:
    """Day each well-slot is commissioned (linear ramp to plateau).

    Slot ``i`` is commissioned so that the count of live slots grows linearly
    from 1 at day 0 to ``plateau_wells`` at ``ramp_years``; after the ramp no
    new slots appear.  Because every slot always holds a running pump, the
    running-well population equals the number of commissioned slots.
    """
    n = int(cfg.plateau_wells)
    if n <= 1:
        return np.array([0.0])
    ramp_days = cfg.ramp_years * DAYS_PER_YEAR
    return np.round(np.arange(n) * ramp_days / (n - 1)).astype(float)


def _weibull_days(rng: np.random.Generator, beta: float, eta: float, size: int = 1) -> np.ndarray:
    """Weibull(shape=beta, scale=eta) samples in days."""
    return eta * rng.weibull(beta, size=size)


class _CovariatePool:
    """Bulk-drawn per-run hazard-layer bins, handed out one run at a time.

    The run loop is inherently scalar (each renewal depends on the previous
    one), but drawing bins one at a time from the generator is far more
    expensive than drawing them in blocks — so pre-draw a block and refill.

    Per-well layers are *not* handled here: they are drawn once per slot before
    the loop starts and never redrawn.
    """

    CHUNK = 8192

    def __init__(self, layers: list[LayerSpec], rng: np.random.Generator) -> None:
        self.layers = layers
        self.rng = rng
        self._i = self.CHUNK
        self._bins: dict[str, np.ndarray] = {}
        self._theta = np.empty(0)

    def _refill(self) -> None:
        self._bins = draw_bins(self.layers, self.rng, self.CHUNK)
        self._theta = theta_from_bins(self.layers, self._bins)
        self._i = 0

    def next(self) -> tuple[float, int, int]:
        """``(theta, ql_bin, freq_bin)`` for the next run."""
        if self._i >= self.CHUNK:
            self._refill()
        i, self._i = self._i, self._i + 1
        ql = self._bins.get(LAYER_QL)
        fq = self._bins.get(LAYER_FREQ)
        return (
            float(self._theta[i]),
            int(ql[i]) if ql is not None else NO_BIN,
            int(fq[i]) if fq is not None else NO_BIN,
        )


def _well_frailty(cfg: SimConfig, layers: list[LayerSpec], n_slots: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(theta_per_slot, well_bin_per_slot)`` for the per-well layers.

    Drawn once, before any run exists — that permanence is the whole point of a
    frailty: every pump installed in a slot inherits the same multiplier, so the
    runs within a well are correlated instead of independent draws.
    """
    if not layers:
        return np.ones(n_slots), np.full(n_slots, NO_BIN, dtype=int)
    bins = draw_bins(layers, rng, n_slots)
    theta = theta_from_bins(layers, bins)
    return theta, bins[LAYER_WELL].astype(int)


def simulate_runs(cfg: SimConfig, seed: int) -> pd.DataFrame:
    """Full run history of one field realization.

    Each row is a pump run: ``slot``, ``run_idx`` (renewal index within the
    slot), ``start``/``end``/``run_len`` (days), ``cause`` in
    {``fail``, ``workover``}, the failure mode that won the race (``fail_mode``,
    an index into :meth:`SimConfig.modes`), and the run's hazard-layer draw
    (``theta``, ``well_bin``, ``ql_bin``, ``freq_bin``; bin ``-1`` when that
    layer is off).  Runs whose ``end`` exceeds the horizon are kept (they become
    the censored current run at snapshots up to the horizon).

    Hazard layers act on the **failure** law only.  The Ql / frequency bins are
    redrawn on every replacement; the well-frailty bin is drawn once per slot
    and kept.  Their product θ enters every failure mode as an effective scale
    ``η_m·θ^(−1/β_m)`` — i.e. θ multiplies the *total* failure hazard, which is
    what makes it a fleet-level frailty rather than a per-node effect.  The
    workover process is a planned/independent pull and is deliberately left
    unmodified by θ.  With no layers enabled nothing is drawn from the
    generator, so those runs are bit-identical to before.

    ``fail_mode`` is recorded for every run, including the ones that end in a
    workover: there it is the latent cause the pull pre-empted, so it must only
    be read where ``cause == "fail"``.
    """
    rng = np.random.default_rng(seed)
    horizon = cfg.horizon_days()
    cdays = commission_days(cfg)
    mode = cfg.workover_mode
    pm_age = cfg.pm_age_days()
    beta_w, eta_w = cfg.beta_wo, cfg.wo_eta_days()
    fail_modes = [(b, e) for _, b, e in cfg.modes()]
    well_layers, run_layers = split_layers(cfg.hazard_layers())
    well_theta, well_bins = _well_frailty(cfg, well_layers, cdays.size, rng)
    pool = _CovariatePool(run_layers, rng) if run_layers else None

    rows: list[tuple] = []
    for slot, c in enumerate(cdays):
        start = float(c)
        run_idx = 0
        theta_w, well_bin = float(well_theta[slot]), int(well_bins[slot])
        while start < horizon:
            if pool is None:
                theta, ql_bin, freq_bin = theta_w, NO_BIN, NO_BIN
            else:
                theta_run, ql_bin, freq_bin = pool.next()
                theta = theta_w * theta_run
            # competing modes race for the pump; the earliest one wins
            t_fail, fail_mode = np.inf, 0
            for i, (beta_m, eta_m) in enumerate(fail_modes):
                eta_run = float(eta_effective(eta_m, beta_m, theta))
                t_m = float(_weibull_days(rng, beta_m, eta_run)[0])
                if t_m < t_fail:
                    t_fail, fail_mode = t_m, i
            if mode == WORKOVER_NONE:
                t_wo = np.inf
            elif mode == WORKOVER_DETERMINISTIC:
                t_wo = pm_age
            elif mode == WORKOVER_STATISTICAL:
                t_wo = float(_weibull_days(rng, beta_w, eta_w)[0])
            else:  # pragma: no cover - guarded by config.validate()
                raise ValueError(f"unknown workover_mode {mode!r}")

            if t_wo <= t_fail:
                run_len, cause = t_wo, "workover"
            else:
                run_len, cause = t_fail, "fail"

            end = start + run_len
            downtime = cfg.downtime_days(cause)  # well not producing until next run
            rows.append((slot, run_idx, start, end, run_len, cause, downtime,
                         theta, fail_mode, well_bin, ql_bin, freq_bin))
            start = end + downtime
            run_idx += 1

    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def observe_at(runs: pd.DataFrame, snap_days: float) -> pd.DataFrame:
    """Observe the fixed realization at calendar time ``snap_days``.

    Returns one row per run that has *started* by the snapshot, with:
      - ``duration``: run length if it ended by the snapshot, else age-so-far.
      - ``event``: 1 iff the run failed by the snapshot; workovers and the
        still-running current run are 0 (right-censored).
      - ``status`` in {``fail``, ``workover``, ``running``} for bookkeeping.
      - ``fail_mode``: which competing mode ended it — only meaningful where
        ``status == "fail"``.
    """
    started = runs.loc[runs["start"] <= snap_days].copy()
    ended = started["end"].to_numpy() <= snap_days
    cause = started["cause"].to_numpy()

    duration = np.where(ended, started["run_len"].to_numpy(), snap_days - started["start"].to_numpy())
    event = np.where(ended & (cause == "fail"), 1, 0)
    status = np.where(~ended, "running", cause)

    return pd.DataFrame(
        {
            "slot": started["slot"].to_numpy(),
            "duration": duration.astype(float),
            "event": event.astype(int),
            "status": status,
            "fail_mode": started["fail_mode"].to_numpy(dtype=int),
        }
    )


def population_at(cfg: SimConfig, snap_days: float) -> int:
    """Number of running wells (commissioned slots) at the snapshot."""
    return int(np.sum(commission_days(cfg) <= snap_days))


def runs_to_arrays(runs: pd.DataFrame) -> dict:
    """Extract the run columns as numpy arrays once (hot-loop fast path)."""
    out = {
        "start": runs["start"].to_numpy(dtype=float),
        "end": runs["end"].to_numpy(dtype=float),
        "run_len": runs["run_len"].to_numpy(dtype=float),
        "is_fail": runs["cause"].to_numpy() == "fail",
        "downtime_after": runs["downtime_after"].to_numpy(dtype=float),
        "theta": runs["theta"].to_numpy(dtype=float),
        "fail_mode": runs["fail_mode"].to_numpy(dtype=int),
    }
    out.update({c: runs[c].to_numpy(dtype=int) for c in BIN_COLUMNS})
    return out


def observe_arrays(arr: dict, snap_days: float):
    """Numpy core of :func:`observe_at`: returns ``(duration, ended, is_fail)``.

    ``is_fail`` carries each started run's cause (used only where ``ended``).
    """
    started = started_mask(arr, snap_days)
    s_start = arr["start"][started]
    s_end = arr["end"][started]
    ended = s_end <= snap_days
    dur = np.where(ended, arr["run_len"][started], snap_days - s_start)
    return dur, ended, arr["is_fail"][started]


def started_mask(arr: dict, snap_days: float) -> np.ndarray:
    """Boolean mask of the runs that have started by ``snap_days``.

    Same selection :func:`observe_arrays` applies, exposed so callers can slice
    per-run attributes (e.g. hazard-layer bins) onto the observed rows.
    """
    return arr["start"] <= snap_days
