"""Preventive-workover schedule optimiser (age-replacement on the failure law).

Developing a ГТМ schedule is a *prescriptive* problem: given the failure hazard
and the cost of a planned pull vs an unplanned failure, choose the preventive-
replacement age that minimises long-run loss.  The renewal-reward
:func:`expected_rates` already gives the exact steady-state loss for any
deterministic preventive age, so the optimiser is just a sweep over that age.

Objective (default ``oil_loss_pct``) = share of well-days not producing =
``E[downtime] / E[cycle]``; minimising it maximises production.  A finite optimum
exists only when the failure hazard is increasing (β>1) and the planned pull is
cheaper than an unplanned failure — otherwise run-to-failure wins.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR, WORKOVER_DETERMINISTIC, WORKOVER_NONE, SimConfig
from .metrics import expected_rates

OBJECTIVES = ("oil_loss_pct", "workdays_lost_per_wellyr", "fail_per_well_year", "pull_per_well_year")


def _metrics_row(cfg: SimConfig, pm_fraction: float | None) -> dict:
    """Steady-state metrics for one policy (``pm_fraction=None`` = run-to-failure)."""
    if pm_fraction is None:
        e = expected_rates(dataclasses.replace(cfg, workover_mode=WORKOVER_NONE))
        pm_fraction, pm_age = np.inf, np.inf
    else:
        e = expected_rates(dataclasses.replace(cfg, workover_mode=WORKOVER_DETERMINISTIC, pm_fraction=float(pm_fraction)))
        pm_age = float(pm_fraction) * cfg.ttf_ref_days()
    return {
        "pm_fraction": float(pm_fraction),
        "pm_age_days": float(pm_age),
        "oil_loss_pct": e["oil_loss_pct"],
        "workdays_lost_per_wellyr": e["oil_loss_pct"] / 100.0 * DAYS_PER_YEAR,
        "fail_per_well_year": e["fail_per_well_year"],
        "workover_per_well_year": e["workover_per_well_year"],
        "pull_per_well_year": e["pull_per_well_year"],
    }


def policy_curve(cfg: SimConfig, pm_fractions: np.ndarray | None = None) -> pd.DataFrame:
    """Steady-state loss for each preventive-pull age + the run-to-failure row."""
    if pm_fractions is None:
        pm_fractions = np.round(np.arange(0.20, 3.001, 0.05), 3)
    rows = [_metrics_row(cfg, f) for f in pm_fractions]
    rows.append(_metrics_row(cfg, None))  # run-to-failure
    return pd.DataFrame(rows)


def optimize_policy(cfg: SimConfig, objective: str = "oil_loss_pct",
                    pm_fractions: np.ndarray | None = None) -> dict:
    """Find the loss-minimising preventive-pull age vs the run-to-failure baseline."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    curve = policy_curve(cfg, pm_fractions)
    rtf = curve.loc[curve["pm_fraction"] == np.inf].iloc[0].to_dict()
    grid = curve.loc[curve["pm_fraction"] != np.inf]
    best_grid = grid.loc[grid[objective].idxmin()].to_dict()

    # run-to-failure wins unless a finite age strictly beats it
    if best_grid[objective] < rtf[objective] - 1e-9:
        best, policy = best_grid, f"PM @ {best_grid['pm_fraction']:.2f}×TTF"
    else:
        best, policy = rtf, "run-to-failure"

    saving_abs = rtf[objective] - best[objective]
    saving_rel = saving_abs / rtf[objective] if rtf[objective] > 0 else 0.0
    return {
        "policy": policy,
        "objective": objective,
        "tau_star_days": best["pm_age_days"],
        "tau_star_over_meanTTF": best["pm_age_days"] / cfg.true_mean_ttf_days(),
        "best": best,
        "run_to_failure": rtf,
        "saving_abs": saving_abs,   # oil-loss: percentage points; days: days/well·yr
        "saving_rel": saving_rel,
        "curve": curve,
    }


def _current_metrics(cfg: SimConfig) -> dict:
    """Steady-state metrics for the policy the config already encodes."""
    e = expected_rates(cfg)
    label = "run-to-failure" if cfg.workover_mode == WORKOVER_NONE else cfg.label()
    return {
        "policy": label,
        "pm_age_days": cfg.pm_age_days() if cfg.workover_mode == WORKOVER_DETERMINISTIC else np.nan,
        "oil_loss_pct": e["oil_loss_pct"],
        "workdays_lost_per_wellyr": e["oil_loss_pct"] / 100.0 * DAYS_PER_YEAR,
        "fail_per_well_year": e["fail_per_well_year"],
        "workover_per_well_year": e["workover_per_well_year"],
        "pull_per_well_year": e["pull_per_well_year"],
    }


def evaluate_policy(cfg: SimConfig, objective: str = "oil_loss_pct",
                    pm_fractions: np.ndarray | None = None) -> dict:
    """Score the config's *current* workover programme vs run-to-failure & optimum.

    ``capture_rel`` = fraction of the achievable improvement (run-to-failure →
    optimum) that the current programme actually captured: 1.0 = optimal, 0 = no
    better than doing nothing, **negative = the programme backfired** (worse than
    run-to-failure).  ``left_on_table_abs`` = remaining gap to the optimum.
    """
    current = _current_metrics(cfg)
    opt = optimize_policy(cfg, objective=objective, pm_fractions=pm_fractions)
    rtf = opt["run_to_failure"]
    o = objective
    achievable = rtf[o] - opt["best"][o]  # RTF loss - optimum loss (>= 0)
    captured = rtf[o] - current[o]        # RTF loss - current loss
    return {
        "objective": o,
        "current": current,
        "run_to_failure": rtf,
        "optimum": opt["best"],
        "optimum_policy": opt["policy"],
        "tau_star_days": opt["tau_star_days"],
        "vs_run_to_failure_abs": captured,          # +ve current beats doing nothing
        "left_on_table_abs": current[o] - opt["best"][o],
        "left_on_table_rel": (current[o] - opt["best"][o]) / current[o] if current[o] > 0 else 0.0,
        "capture_rel": captured / achievable if achievable > 1e-12 else float("nan"),
        "curve": opt["curve"],
    }


def sensitivity_grid(cfg: SimConfig, betas, cost_ratios, objective: str = "oil_loss_pct") -> pd.DataFrame:
    """Optimum vs failure-shape β and cost ratio C_p/C_f (planned/unplanned downtime)."""
    rows = []
    for beta in betas:
        for cr in cost_ratios:
            c = dataclasses.replace(
                cfg, beta_fail=float(beta),
                downtime_workover_days=float(cr) * cfg.downtime_fail_days,
            )
            o = optimize_policy(c, objective=objective)
            rows.append({
                "beta": float(beta),
                "cost_ratio": float(cr),
                "policy": o["policy"],
                "tau_star_over_meanTTF": o["tau_star_over_meanTTF"],
                "loss_rtf": o["run_to_failure"][objective],
                "loss_opt": o["best"][objective],
                "saving_abs": o["saving_abs"],
                "saving_rel": o["saving_rel"],
            })
    return pd.DataFrame(rows)
