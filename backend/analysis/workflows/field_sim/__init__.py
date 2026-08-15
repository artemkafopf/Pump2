"""Field-simulation experiments: growing ESP fleet, KM & Weibull under censoring.

See ``scripts/run/field_sim.py`` for a CLI and ``streamlit_apps/field_sim_app.py``
for the interactive explorer.
"""
from __future__ import annotations

from .config import (
    AGE_MAX_DAYS,
    AGE_STEP_DAYS,
    DAYS_PER_YEAR,
    SimConfig,
    WORKOVER_DETERMINISTIC,
    WORKOVER_NONE,
    WORKOVER_STATISTICAL,
    age_grid,
)
from .experiment import (
    ExperimentResult,
    aggregate_bins,
    aggregate_by_snapshot,
    mode_columns,
    run_experiment,
)
from .hazard import (
    LAYER_FREQ,
    LAYER_QL,
    LAYER_WELL,
    LayerSpec,
    layers_from_config,
    make_freq_layer,
    make_ql_layer,
    make_well_layer,
    split_layers,
    theta_design,
    theta_spread,
)
from .metrics import (
    LIFE_KINDS,
    eta_equivalent_days,
    eta_from_life,
    expected_life,
    expected_rates,
    life_given_theta,
    mixture_survival,
    snapshot_metrics,
    truth_curve_label,
)
from .simulate import commission_days, observe_at, population_at, simulate_runs
from .store import list_experiments, load_experiment, save_experiment

__all__ = [
    "AGE_MAX_DAYS",
    "AGE_STEP_DAYS",
    "age_grid",
    "DAYS_PER_YEAR",
    "SimConfig",
    "WORKOVER_NONE",
    "WORKOVER_DETERMINISTIC",
    "WORKOVER_STATISTICAL",
    "ExperimentResult",
    "run_experiment",
    "aggregate_by_snapshot",
    "aggregate_bins",
    "mode_columns",
    "snapshot_metrics",
    "expected_rates",
    "expected_life",
    "life_given_theta",
    "eta_from_life",
    "eta_equivalent_days",
    "mixture_survival",
    "truth_curve_label",
    "LIFE_KINDS",
    "LAYER_QL",
    "LAYER_FREQ",
    "LAYER_WELL",
    "LayerSpec",
    "layers_from_config",
    "make_ql_layer",
    "make_freq_layer",
    "make_well_layer",
    "split_layers",
    "theta_design",
    "theta_spread",
    "commission_days",
    "simulate_runs",
    "observe_at",
    "population_at",
    "save_experiment",
    "load_experiment",
    "list_experiments",
]
