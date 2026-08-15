"""Matplotlib figures for a field-simulation experiment (CLI / reports).

The Streamlit app renders its own interactive Plotly versions; these are the
static PNGs saved alongside each experiment.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import AGE_MAX_DAYS, age_grid
from .experiment import ExperimentResult, aggregate_by_snapshot
from .metrics import expected_life, expected_rates, mixture_survival, truth_curve_label

INK = "#24292f"
GRID = "#d9dde3"
CENS = "#1f5c99"     # censoring-aware, cause-specific (the correct one)
FO = "#e5484d"       # failures-only (drops every censored run)
ALL = "#2f9e44"      # all-pulls (workover counted as failure)
TRUE = "#6b7280"     # ground truth


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK, length=0, labelsize=8)


def plot_km_snapshots(result: ExperimentResult):
    """Pooled KM curves at each snapshot year vs the true Weibull."""
    cfg = result.config
    fig, ax = plt.subplots(figsize=(7.5, 5))
    years = sorted(result.km_curves, key=lambda k: float(k))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(years)))
    for color, key in zip(cmap, years):
        c = result.km_curves[key]
        ax.step(c["time"], c["surv"], where="post", color=color, lw=1.8,
                label=f"yr {key} · pop {c.get('running_pop','?')} · {c['n_fail']} fails")
    tmax = max((max(result.km_curves[k]["time"]) for k in years), default=cfg.eta_fail_days * 3)
    tgrid = age_grid(tmax)
    truth_label = truth_curve_label(cfg)
    ax.plot(tgrid, mixture_survival(tgrid, cfg), color=TRUE, lw=2, ls="--", label=truth_label)
    ax.set_xlabel("pump age (days)", color=INK)
    ax.set_ylabel("S(t)  (failure = event; workover/running = censored)", color=INK)
    ax.set_xlim(0, AGE_MAX_DAYS)      # never past five years — see config.age_grid
    ax.set_ylim(0, 1.02)
    ax.set_title(f"KM survival snapshots — {result.label}", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_param_trends(result: ExperimentResult):
    """beta & eta (censored vs failures-only) vs time and vs pump-starts."""
    cfg = result.config
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x_time = agg["snap_year"].to_numpy()
    x_starts = agg["n_runs"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))

    # with competing modes the pooled fit has no single truth to sit on — the
    # per-mode cause-specific fits do, but they are not this figure's subject
    single_mode = cfg.n_modes() == 1

    def _panel(ax, x, xlabel, param, truth, ylabel, title, logy=False):
        ax.fill_between(x, agg[f"{param}_cens_p10"], agg[f"{param}_cens_p90"], color=CENS, alpha=0.15, zorder=1)
        ax.plot(x, agg[f"{param}_cens_median"], color=CENS, lw=2, marker="o", ms=3.5, label="censored (cause-specific)", zorder=3)
        ax.fill_between(x, agg[f"{param}_fo_p10"], agg[f"{param}_fo_p90"], color=FO, alpha=0.13, zorder=1)
        ax.plot(x, agg[f"{param}_fo_median"], color=FO, lw=2, marker="s", ms=3.5, label="failures-only", zorder=3)
        ax.fill_between(x, agg[f"{param}_all_p10"], agg[f"{param}_all_p90"], color=ALL, alpha=0.13, zorder=1)
        ax.plot(x, agg[f"{param}_all_median"], color=ALL, lw=2, marker="D", ms=3.2, label="all-pulls (WO = failure)", zorder=3)
        if single_mode:
            ax.axhline(truth, color=TRUE, ls="--", lw=1.6, label=f"true = {truth:g}", zorder=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        _style(ax)

    _panel(axes[0, 0], x_time, "field age (years)", "beta", cfg.beta_fail, "β̂ (shape)", "Shape β vs time")
    _panel(axes[0, 1], x_time, "field age (years)", "eta", cfg.eta_fail_days, "η̂ (scale, days)", "Scale η vs time", logy=True)
    _panel(axes[1, 0], x_starts, "cumulative pump starts", "beta", cfg.beta_fail, "β̂ (shape)", "Shape β vs pump starts")
    _panel(axes[1, 1], x_starts, "cumulative pump starts", "eta", cfg.eta_fail_days, "η̂ (scale, days)", "Scale η vs pump starts", logy=True)

    # observed mean TTF (naive Σt/N) overlaid on the eta / life panels
    for ax, xx in [(axes[0, 1], x_time), (axes[1, 1], x_starts)]:
        ax.plot(xx, agg["obs_ttf_fail_median"], color=FO, lw=1.4, ls="--", label="obs. mean TTF (failures)")
        ax.plot(xx, agg["obs_ttf_all_median"], color=ALL, lw=1.4, ls="--", label="obs. mean TTF (fail+WO)")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="lower right")

    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Fitted Weibull vs truth — {result.label}   (band = p10–p90 across {cfg.n_seeds} seeds)",
                 fontsize=12, color=INK, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


RATE_FAIL = "#1f5c99"
RATE_PULL = "#e8590c"
LOSS = "#0b7285"
CF = "#868e96"


def plot_rates(result: ExperimentResult):
    """Failure/pull rate and oil loss vs time, no-workover twin overlaid."""
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x = agg["snap_year"].to_numpy()
    nowo = None
    if result.fit_table_nowo is not None:
        nowo = aggregate_by_snapshot(result.fit_table_nowo).sort_values("snap_year")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    axL.fill_between(x, agg["fail_rate_well_yr_p10"], agg["fail_rate_well_yr_p90"], color=RATE_FAIL, alpha=0.13)
    axL.plot(x, agg["fail_rate_well_yr_median"], color=RATE_FAIL, lw=2.2, marker="o", ms=3.5, label="failures / well-yr")
    axL.plot(x, agg["pull_rate_well_yr_median"], color=RATE_PULL, lw=2, ls=":", marker="^", ms=3.5, label="pulls (fail+WO) / well-yr")
    axR.fill_between(x, agg["oil_loss_pct_p10"], agg["oil_loss_pct_p90"], color=LOSS, alpha=0.13)
    axR.plot(x, agg["oil_loss_pct_median"], color=LOSS, lw=2.2, marker="o", ms=3.5, label="oil loss % (with programme)")
    if nowo is not None:
        xn = nowo["snap_year"].to_numpy()
        axL.plot(xn, nowo["fail_rate_well_yr_median"], color=CF, lw=2, ls="--", label="failures / well-yr (no workover)")
        axR.plot(xn, nowo["oil_loss_pct_median"], color=CF, lw=2, ls="--", label="oil loss % (no workover)")

    exp = expected_rates(result.config)
    axL.axhline(exp["fail_per_well_year"], color=INK, ls=":", lw=1, label=f"expected fail {exp['fail_per_well_year']:.2f}")
    if exp["workover_per_well_year"] > 1e-6:
        axL.axhline(exp["pull_per_well_year"], color=INK, ls=(0, (1, 3)), lw=1, label=f"expected pull {exp['pull_per_well_year']:.2f}")
    axR.axhline(exp["oil_loss_pct"], color=INK, ls=":", lw=1, label=f"expected {exp['oil_loss_pct']:.2f}%")

    axL.set_ylabel("events per well-year", color=INK)
    axL.set_title("Failure & pull rate", color=INK, fontsize=11, loc="left")
    axR.set_ylabel("% of well-days lost", color=INK)
    axR.set_title("Oil loss (share of non-producing well-days)", color=INK, fontsize=11, loc="left")
    for ax in (axL, axR):
        ax.set_xlabel("field age (years)", color=INK)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5, frameon=False, loc="best")
        _style(ax)
    fig.suptitle(f"Fleet rates & oil loss — {result.label}", fontsize=12, color=INK, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


LIFE_SERIES = [
    ("life_rmst", "#1f5c99", "RMST(0, τ)"),
    ("life_mrl0", "#2f9e44", "MRL(0) = KM mean"),
    ("life_median", "#8452c9", "KM median"),
    ("obs_ttf_fail", "#e5484d", "mean (naive Σt/N)"),
]


def plot_life_summaries(result: ExperimentResult):
    """RMST(0,τ), MRL(0), KM median and naive mean of the failure distribution."""
    cfg = result.config
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x = agg["snap_year"].to_numpy()
    tl = expected_life(cfg)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color, name in LIFE_SERIES:
        ax.fill_between(x, agg[f"{col}_p10"], agg[f"{col}_p90"], color=color, alpha=0.12)
        ax.plot(x, agg[f"{col}_median"], color=color, lw=2, marker="o", ms=3, label=name)
    for y, txt in [(tl["rmst"], f"true RMST {tl['rmst']:.0f}"),
                   (tl["mean"], f"true mean/MRL {tl['mean']:.0f}"),
                   (tl["median"], f"true median {tl['median']:.0f}")]:
        ax.axhline(y, color=INK, ls=":", lw=1)
        ax.annotate(txt, (x[-1], y), fontsize=7.5, color=INK, va="bottom", ha="right")
    ax.set_xlabel("field age (years)", color=INK)
    ax.set_ylabel(f"days   (τ = {cfg.rmst_tau_days:g}d)", color=INK)
    ax.set_title(f"Life summaries (failure distribution) — {result.label}", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="best")
    _style(ax)
    fig.tight_layout()
    return fig


def save_experiment_figures(result: ExperimentResult, out_dir) -> list:
    """Write the standard PNGs into ``out_dir`` and return their paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (("km_snapshots", plot_km_snapshots), ("param_trends", plot_param_trends),
                     ("life_summaries", plot_life_summaries), ("rates_oil_loss", plot_rates)):
        fig = fn(result)
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    return paths
