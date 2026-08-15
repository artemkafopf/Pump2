"""CLI for the growing-ESP-field simulation experiments.

Simulates a field that grows to a plateau of running wells, replacing each pump
on failure/workover, and tracks how KM survival and fitted Weibull parameters
(censoring-aware vs failures-only) evolve with time and pump starts.

Examples
--------
    # run the built-in preset suite (Exp 1 + Exp 2 workover variants)
    python scripts/run/field_sim.py --suite

    # one custom run
    python scripts/run/field_sim.py --workover statistical --beta-wo 1.3 \
        --wo-eta-fraction 0.8 --seeds 40 --label "stat WO 0.8"

    # with both covariate hazard layers on (2x theta low->high Ql, U-shape freq)
    python scripts/run/field_sim.py --ql-layer --freq-layer --label "theta layers"

    # two competing equipment failure modes + individual-well frailty spread 6x
    python scripts/run/field_sim.py --mode pump:1.6:520 --mode cable:0.8:900 \
        --well-layer --well-theta-ratio 6

    # list saved experiments
    python scripts/run/field_sim.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.field_sim import SimConfig, run_experiment, save_experiment
from analysis.workflows.field_sim.config import TTF_REFERENCES
from analysis.workflows.field_sim.plots import save_experiment_figures
from analysis.workflows.field_sim.store import FIELD_SIM_ROOT, list_experiments


def preset_suite(seeds: int) -> list[tuple[str, SimConfig]]:
    """The experiment matrix from the study brief (20-yr horizon)."""
    common = dict(n_seeds=seeds)
    return [
        ("Exp1 failure-only", SimConfig(workover_mode="none", **common)),
        ("Exp2 PM@0.8xTTF", SimConfig(workover_mode="deterministic", pm_fraction=0.8, **common)),
        ("Exp2 PM@1.2xTTF", SimConfig(workover_mode="deterministic", pm_fraction=1.2, **common)),
        ("Exp2 statWO 0.8", SimConfig(workover_mode="statistical", beta_wo=1.3, wo_eta_fraction=0.8, **common)),
        ("Exp2 statWO 1.2", SimConfig(workover_mode="statistical", beta_wo=1.3, wo_eta_fraction=1.2, **common)),
    ]


def _parse_mode(spec: str) -> dict:
    """``NAME:BETA:ETA_DAYS`` → one competing failure mode."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--mode expects NAME:BETA:ETA_DAYS, got {spec!r}")
    name, beta, eta = parts
    return {"name": name, "beta": float(beta), "eta_days": float(eta)}


def _run_one(cfg: SimConfig, label: str, make_figs: bool, counterfactual: bool = True) -> Path:
    print(f"  running: {label}  ({cfg.n_seeds} seeds, {cfg.plateau_wells} wells) ...", flush=True)
    result = run_experiment(cfg, label=label, counterfactual=counterfactual)
    d = save_experiment(result)
    if make_figs:
        save_experiment_figures(result, d / "figures")
    final = result.fit_table[result.fit_table.snap_year == result.fit_table.snap_year.max()]
    print(
        f"    saved -> {d.name}  |  @{cfg.total_years:g}yr  "
        f"beta_cens={final.beta_cens.mean():.2f} eta_cens={final.eta_cens.mean():.0f}  "
        f"beta_fo={final.beta_fo.mean():.2f} eta_fo={final.eta_fo.mean():.0f}",
        flush=True,
    )
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", action="store_true", help="run the built-in preset suite")
    p.add_argument("--list", action="store_true", help="list saved experiments and exit")
    p.add_argument("--no-figures", action="store_true", help="skip PNG figure generation")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--label", type=str, default=None)
    # single-run parameters (all editable)
    p.add_argument("--total-years", type=float, default=20.0)
    p.add_argument("--ramp-years", type=float, default=10.0)
    p.add_argument("--wells", type=int, default=100)
    p.add_argument("--beta-fail", type=float, default=1.0)
    p.add_argument("--eta-fail-days", type=float, default=365.0)
    p.add_argument("--workover", choices=("none", "deterministic", "statistical"), default="none")
    p.add_argument("--ttf-reference", choices=TTF_REFERENCES, default="mean",
                   # ASCII only: --help is written straight to a cp1251 console here,
                   # and a stray tau/eta aborts the whole help screen
                   help="what the workover fractions are a fraction of: the mean life "
                        "(= MRL(0)), RMST(0, tau), or the characteristic life eta")
    p.add_argument("--pm-fraction", type=float, default=0.8)
    p.add_argument("--beta-wo", type=float, default=1.3)
    p.add_argument("--wo-eta-fraction", type=float, default=0.8)
    p.add_argument("--downtime-fail-days", type=float, default=7.0)
    p.add_argument("--downtime-workover-days", type=float, default=3.0)
    # competing failure modes (equipment nodes racing for the pump)
    p.add_argument("--mode", action="append", metavar="NAME:BETA:ETA_DAYS", default=None,
                   help="add a competing failure mode, e.g. --mode cable:0.8:900 "
                        "(repeat; the single --beta-fail/--eta-fail-days law is then unused)")
    # covariate hazard layers (PH theta on the failure Weibull)
    p.add_argument("--well-layer", action="store_true",
                   help="enable the individual-well frailty layer (theta drawn once per well)")
    p.add_argument("--well-bins", type=int, default=5)
    p.add_argument("--well-theta-ratio", type=float, default=4.0,
                   help="theta(worst well)/theta(best well)")
    p.add_argument("--ql-layer", action="store_true", help="enable the monotone Ql hazard layer")
    p.add_argument("--ql-lo", type=float, default=50.0)
    p.add_argument("--ql-hi", type=float, default=250.0)
    p.add_argument("--ql-bins", type=int, default=5)
    p.add_argument("--ql-shape", choices=("log", "linear"), default="log")
    p.add_argument("--ql-theta-ratio", type=float, default=2.0, help="theta(top bin)/theta(bottom bin)")
    p.add_argument("--freq-layer", action="store_true", help="enable the U-shaped frequency hazard layer")
    p.add_argument("--freq-lo", type=float, default=40.0)
    p.add_argument("--freq-hi", type=float, default=60.0)
    p.add_argument("--freq-bins", type=int, default=5)
    p.add_argument("--freq-theta-edge", type=float, default=2.0, help="theta at 40/60 Hz vs 50 Hz")
    p.add_argument("--no-hazard-center", dest="hazard_center", action="store_false", default=True,
                   help="do not renormalise theta to a fleet mean of 1")
    p.add_argument("--counterfactual", dest="counterfactual", action="store_true", default=True,
                   help="also run a no-workover twin (default on)")
    p.add_argument("--no-counterfactual", dest="counterfactual", action="store_false")
    p.add_argument("--fit-snapshots", type=int, default=24)
    p.add_argument("--base-seed", type=int, default=12345)
    args = p.parse_args(argv)

    if args.list:
        rows = list_experiments()
        if not rows:
            print("(no saved experiments)")
            return 0
        print(f"{'experiment_id':<40} {'created':<20} label")
        for r in rows:
            print(f"{r['experiment_id']:<40} {r['created']:<20} {r['label']}")
        return 0

    make_figs = not args.no_figures
    if args.suite:
        print("Running preset suite:")
        for label, cfg in preset_suite(args.seeds):
            _run_one(cfg, label, make_figs, counterfactual=args.counterfactual)
        print(f"\nAll experiments saved under: {FIELD_SIM_ROOT}")
        return 0

    fail_modes = [_parse_mode(spec) for spec in (args.mode or [])]
    cfg = SimConfig(
        total_years=args.total_years,
        ramp_years=args.ramp_years,
        plateau_wells=args.wells,
        beta_fail=args.beta_fail,
        eta_fail_days=args.eta_fail_days,
        modes_on=bool(fail_modes),
        fail_modes=fail_modes or SimConfig().fail_modes,
        workover_mode=args.workover,
        ttf_reference=args.ttf_reference,
        pm_fraction=args.pm_fraction,
        beta_wo=args.beta_wo,
        wo_eta_fraction=args.wo_eta_fraction,
        downtime_fail_days=args.downtime_fail_days,
        downtime_workover_days=args.downtime_workover_days,
        hazard_center=args.hazard_center,
        well_layer_on=args.well_layer,
        well_bins=args.well_bins,
        well_theta_ratio=args.well_theta_ratio,
        ql_layer_on=args.ql_layer,
        ql_lo=args.ql_lo,
        ql_hi=args.ql_hi,
        ql_bins=args.ql_bins,
        ql_shape=args.ql_shape,
        ql_theta_ratio=args.ql_theta_ratio,
        freq_layer_on=args.freq_layer,
        freq_lo=args.freq_lo,
        freq_hi=args.freq_hi,
        freq_bins=args.freq_bins,
        freq_theta_edge=args.freq_theta_edge,
        n_fit_snapshots=args.fit_snapshots,
        n_seeds=args.seeds,
        base_seed=args.base_seed,
    )
    _run_one(cfg, args.label or cfg.label(), make_figs, counterfactual=args.counterfactual)
    print(f"\nSaved under: {FIELD_SIM_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
