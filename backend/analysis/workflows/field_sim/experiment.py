"""Run a full field-simulation experiment across seeds and snapshots."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR, WORKOVER_NONE, SimConfig
from .fit import bin_life_arrays, fit_snapshot_arrays, km_curve_arrays, km_curve_failures_only
from .metrics import snapshot_metrics_arrays
from .simulate import commission_days, observe_arrays, runs_to_arrays, simulate_runs, started_mask


@dataclass
class ExperimentResult:
    config: SimConfig
    fit_table: pd.DataFrame           # one row per (seed, snapshot)
    km_curves: dict                   # {snap_year: pooled KM dict}
    label: str
    experiment_id: str
    created: str
    meta: dict = field(default_factory=dict)
    fit_table_nowo: pd.DataFrame | None = None  # no-workover counterfactual twin
    bin_table: pd.DataFrame | None = None       # per (seed, KM snapshot, layer, bin)


def _fit_grid_years(cfg: SimConfig) -> list[float]:
    """Snapshot years for the beta/eta-vs-time trend (fine grid + KM years)."""
    lo = cfg.total_years / cfg.n_fit_snapshots
    grid = list(np.linspace(lo, cfg.total_years, cfg.n_fit_snapshots))
    grid += [y for y in cfg.km_snapshot_years if 0 < y <= cfg.total_years]
    return sorted({round(float(y), 4) for y in grid})


def _make_experiment_id(label: str, created: datetime) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")
    slug = "-".join(filter(None, slug.split("-")))[:40]
    return f"{created.strftime('%Y%m%d-%H%M%S')}_{slug or 'exp'}"


def _simulate_and_fit(cfg: SimConfig, fit_years, km_years, progress=None):
    """Core loop: simulate every seed, fit + measure every snapshot.

    Returns ``(fit_table, km_curves, bin_table)``.  Shared by the main run and
    the no-workover counterfactual twin.  ``bin_table`` is ``None`` when no
    hazard layer is enabled; otherwise it carries per-covariate-bin life
    summaries at the KM snapshot years (a cross-section, not the fine trend
    grid — one KM per bin per snapshot per seed is enough to pay for).
    """
    rows: list[dict] = []
    bin_rows: list[dict] = []
    pooled: dict[float, list[tuple]] = {y: [] for y in km_years}
    cdays = commission_days(cfg)  # constant across seeds & snapshots
    snap_specs = [(y, y * DAYS_PER_YEAR) for y in fit_years]
    km_year_set = set(km_years)
    layers = cfg.hazard_layers()
    n_modes = cfg.n_modes()

    for s in range(cfg.n_seeds):
        seed = cfg.base_seed + s
        arr = runs_to_arrays(simulate_runs(cfg, seed))
        for year, snap_days in snap_specs:
            dur, ended, is_fail = observe_arrays(arr, snap_days)
            modes = arr["fail_mode"][started_mask(arr, snap_days)] if n_modes > 1 else None
            summary = fit_snapshot_arrays(dur, ended, is_fail, rmst_tau=cfg.rmst_tau_days,
                                          fail_mode=modes, n_modes=n_modes)
            summary.update(snapshot_metrics_arrays(arr, cdays, snap_days))
            summary.update(
                {
                    "seed": seed,
                    "snap_year": year,
                    "snap_days": snap_days,
                    "running_pop": int(np.sum(cdays <= snap_days)),
                }
            )
            rows.append(summary)
            if year in km_year_set:
                pooled[year].append((dur, (ended & is_fail).astype(int)))
                if layers:
                    started = started_mask(arr, snap_days)
                    for spec in layers:
                        idx = arr[f"{spec.key}_bin"][started]
                        for row in bin_life_arrays(dur, ended, is_fail, idx, spec.n_bins,
                                                   rmst_tau=cfg.rmst_tau_days):
                            b = row["bin"]
                            row.update({
                                "seed": seed, "snap_year": year, "layer": spec.key,
                                "center": float(spec.centers[b]),
                                "theta": float(spec.theta[b]),
                            })
                            bin_rows.append(row)
        if progress is not None:
            progress((s + 1) / cfg.n_seeds)

    fit_table = pd.DataFrame(rows)
    km_curves: dict[str, dict] = {}
    for year, parts in pooled.items():
        dur_cat = np.concatenate([p[0] for p in parts])
        ev_cat = np.concatenate([p[1] for p in parts])
        curve = km_curve_arrays(dur_cat, ev_cat)
        # the same snapshot seen the naive way, so the two can be drawn together
        fo = km_curve_failures_only(dur_cat, ev_cat)
        curve.update({f"{k}_fo": v for k, v in fo.items() if k in ("time", "surv", "n_runs")})
        curve["snap_year"] = year
        curve["running_pop"] = int(np.sum(cdays <= year * DAYS_PER_YEAR))
        km_curves[f"{year:g}"] = curve
    bin_table = pd.DataFrame(bin_rows) if bin_rows else None
    return fit_table, km_curves, bin_table


def run_experiment(
    cfg: SimConfig,
    label: str | None = None,
    created: datetime | None = None,
    progress=None,
    counterfactual: bool = False,
) -> ExperimentResult:
    """Simulate ``cfg.n_seeds`` realizations and fit + measure every snapshot.

    ``progress`` (optional) is called with a float in [0, 1] for UIs.  If
    ``counterfactual`` and the config has a workover process, an identical twin
    with the workover programme switched off is also run (same seeds), so the
    programme's effect on failure rate and oil loss can be read directly.
    """
    cfg.validate()
    created = created or datetime.now()
    label = label or cfg.label()

    fit_years = _fit_grid_years(cfg)
    km_years = sorted({round(float(y), 4) for y in cfg.km_snapshot_years if 0 < y <= cfg.total_years})

    run_cf = counterfactual and cfg.workover_mode != WORKOVER_NONE

    def _main_progress(f):
        if progress is not None:
            progress(f * (0.5 if run_cf else 1.0))

    fit_table, km_curves, bin_table = _simulate_and_fit(cfg, fit_years, km_years,
                                                        progress=_main_progress)

    fit_table_nowo = None
    if run_cf:
        cfg_nowo = dataclasses.replace(cfg, workover_mode=WORKOVER_NONE)
        def _cf_progress(f):
            if progress is not None:
                progress(0.5 + f * 0.5)
        fit_table_nowo, _, _ = _simulate_and_fit(cfg_nowo, fit_years, km_years, progress=_cf_progress)

    experiment_id = _make_experiment_id(label, created)
    return ExperimentResult(
        config=cfg,
        fit_table=fit_table,
        km_curves=km_curves,
        label=label,
        experiment_id=experiment_id,
        created=created.isoformat(timespec="seconds"),
        meta={
            "true_beta": cfg.beta_fail,
            "true_eta_days": cfg.eta_fail_days,
            "true_mean_ttf_days": cfg.true_mean_ttf_days(),
            "fit_years": fit_years,
            "km_years": km_years,
            "has_counterfactual": fit_table_nowo is not None,
            "hazard_layers": [s.key for s in cfg.hazard_layers()],
            "fail_modes": [{"name": n, "beta": b, "eta_days": e} for n, b, e in cfg.modes()],
        },
        fit_table_nowo=fit_table_nowo,
        bin_table=bin_table,
    )


def _nan_stat(func):
    def inner(x):
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        return func(arr) if arr.size else np.nan
    return inner


# Per-snapshot series that always exist; the per-mode ones (beta_m0, eta_m0, …)
# are discovered from the table, since how many there are is a config choice.
SNAPSHOT_PARAMS = (
    "beta_cens", "eta_cens", "beta_fo", "eta_fo", "beta_all", "eta_all",
    "fail_rate_well_yr", "pull_rate_well_yr", "oil_loss_pct",
    "obs_ttf_fail", "obs_ttf_all", "life_rmst", "life_mrl0", "life_median",
)


def mode_columns(fit_table: pd.DataFrame) -> list[int]:
    """Indices of the competing failure modes a fit table carries, in order."""
    return sorted(int(c[len("beta_m"):]) for c in fit_table.columns
                  if c.startswith("beta_m") and c[len("beta_m"):].isdigit())


def aggregate_by_snapshot(fit_table: pd.DataFrame) -> pd.DataFrame:
    """Median / mean / p10 / p90 / valid-count of fitted params, per snapshot.

    The median line is the robust headline (a single near-degenerate early seed
    would drag the mean); the p10-p90 band carries the sampling spread.
    """
    stats = {
        "median": _nan_stat(np.median),
        "mean": _nan_stat(np.mean),
        "p10": _nan_stat(lambda a: np.percentile(a, 10)),
        "p90": _nan_stat(lambda a: np.percentile(a, 90)),
        "n": _nan_stat(lambda a: a.size),
    }
    named = {
        "running_pop": ("running_pop", "mean"),
        "n_runs": ("n_runs", "mean"),
        "n_fail": ("n_fail", "mean"),
        "n_workover": ("n_workover", "mean"),
    }
    modes = mode_columns(fit_table)
    for m in modes:
        named[f"n_fail_m{m}"] = (f"n_fail_m{m}", "mean")
    all_params = SNAPSHOT_PARAMS + tuple(
        f"{p}_m{m}" for m in modes for p in ("beta", "eta"))
    present = [p for p in all_params if p in fit_table.columns]
    for param in present:
        for stat_name, fn in stats.items():
            named[f"{param}_{stat_name}"] = (param, fn)
    agg = fit_table.groupby("snap_year").agg(**named).reset_index()
    # Back-fill columns for params absent from an older saved fit_table so that
    # downstream plots/tables always find them (rendered as empty).
    for param in all_params:
        if param not in present:
            for stat_name in stats:
                agg[f"{param}_{stat_name}"] = np.nan
    return agg


def aggregate_bins(bin_table: pd.DataFrame, layer: str, snap_year: float,
                   seed: int | None = None) -> pd.DataFrame:
    """Across-seed summary of one hazard layer at one snapshot, one row per bin.

    Counts are summed over seeds (the pooled bin population); the life
    summaries are medians of the per-seed values with a p10-p90 band, matching
    how :func:`aggregate_by_snapshot` reports everything else.

    Pass ``seed`` to look at a single field realization instead of the pool.
    The same aggregation then runs over one row per bin, so median/p10/p90 all
    collapse onto that seed's value and the band closes to nothing — what you
    see is one field's raw sampling noise, not a smoothed central line.
    """
    sub = bin_table[(bin_table["layer"] == layer) & (bin_table["snap_year"] == snap_year)]
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    if sub.empty:
        return sub
    stats = {
        "median": _nan_stat(np.median),
        "p10": _nan_stat(lambda a: np.percentile(a, 10)),
        "p90": _nan_stat(lambda a: np.percentile(a, 90)),
    }
    named: dict = {
        "center": ("center", "first"),
        "theta": ("theta", "first"),
        "n_runs": ("n_runs", "sum"),
        "n_fail": ("n_fail", "sum"),
        "n_seeds": ("seed", "nunique"),
    }
    all_params = ("obs_ttf_fail", "obs_ttf_median", "km_rmst", "km_median")
    present = [p for p in all_params if p in sub.columns]
    for param in present:
        for stat_name, fn in stats.items():
            named[f"{param}_{stat_name}"] = (param, fn)
    agg = sub.groupby("bin").agg(**named).reset_index().sort_values("center")
    # Back-fill any column a bin_table saved by an older version does not carry,
    # so callers can always look it up (it renders as empty).
    for param in all_params:
        if param not in present:
            for stat_name in stats:
                agg[f"{param}_{stat_name}"] = np.nan
    return agg
