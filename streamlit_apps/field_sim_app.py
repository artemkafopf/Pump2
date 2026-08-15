"""Interactive explorer for the growing-ESP-field simulation experiments.

Run new simulations (every parameter editable) or load a previously saved one,
then inspect KM survival snapshots and how the fitted Weibull parameters
(censoring-aware vs failures-only) evolve with time and pump starts.

Every pump-life quantity — η, τ, RMST, MRL(0), median, mean TTF — is in **days**
on both input and output.  The field clock (horizon, ramp, snapshots) stays in
years, since that is the field's calendar and not a pump age.

    streamlit run streamlit_apps/field_sim_app.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.field_sim import (  # noqa: E402
    AGE_MAX_DAYS,
    SimConfig,
    age_grid,
    aggregate_bins,
    aggregate_by_snapshot,
    commission_days,
    expected_life,
    expected_rates,
    mode_columns,
    run_experiment,
    save_experiment,
)
from analysis.workflows.field_sim.config import (  # noqa: E402
    DEFAULT_FAIL_MODES,
    TTF_REF_ETA,
    TTF_REF_MEAN,
    TTF_REF_RMST,
    WORKOVER_DETERMINISTIC,
    WORKOVER_NONE,
    WORKOVER_STATISTICAL,
)
from analysis.workflows.field_sim.hazard import (  # noqa: E402
    LAYER_FREQ,
    LAYER_QL,
    LAYER_WELL,
    QL_SHAPES,
    layers_from_config,
    theta_design,
    theta_spread,
)
from analysis.workflows.field_sim.metrics import (  # noqa: E402
    LIFE_MEAN,
    LIFE_MEDIAN,
    LIFE_RMST,
    eta_equivalent_days,
    eta_from_life,
    life_given_theta,
    mixture_survival,
    truth_curve_label,
    weibull_life,
)
from analysis.workflows.field_sim.store import (  # noqa: E402
    APP_DEFAULTS_PATH,
    list_experiments,
    load_app_defaults,
    load_experiment,
    save_app_defaults,
)

st.set_page_config(page_title="Field Simulation", page_icon=":material/experiment:", layout="wide")

CENS = "#1f5c99"   # censoring-aware, cause-specific (only failures = event)
FO = "#e5484d"     # failures-only (drop every censored run)
ALL = "#2f9e44"    # all-pulls (failure OR workover = event)
TRUE = "#6b7280"   # ground truth
KM_SEQ = ["#440154", "#3b528b", "#21918c", "#5ec962", "#addc30", "#fde725"]

RATE_FAIL = "#1f5c99"   # failure rate, with programme
RATE_PULL = "#e8590c"   # total pull rate (fail + workover)
LOSS = "#0b7285"        # oil loss, with programme
CF = "#868e96"          # no-workover counterfactual

# Pump-age axes come from the package (AGE_MAX_DAYS / age_grid) so the cap and the grid
# are the same here, in the matplotlib figures and in any future panel — see config.py
# for why they exist at all.

# X-axis choices for the β/η evolution plot: label -> (aggregate column, axis title, short name).
# All four are cumulative-by-snapshot and monotone; n_pulls = planned pulls (workovers).
X_CHOICES = {
    "time": ("snap_year", "field age (years)", "time"),
    "n_runs": ("n_runs", "cumulative pump starts (runs)", "n_runs"),
    "n_failures": ("n_fail", "cumulative failures", "n_failures"),
    "n_pulls": ("n_workover", "cumulative workovers (planned pulls)", "n_pulls"),
}

# Life summaries the η calculator can invert, in display order.
CALC_TARGETS = {
    "RMST(0, τ)": LIFE_RMST,
    "MRL(0) = mean": LIFE_MEAN,
    "median": LIFE_MEDIAN,
}

# What the workover "% of TTF" fractions are a percentage of.  MRL(0) and the mean are one
# option under both names, not two: MRL(0) = E[T] identically — the mean residual life at
# age 0 is the whole expected life — and two controls that always agree would read as a
# lever that does nothing.  RMST(0, τ) is the one that genuinely differs.
TTF_REFS = {
    "mean = MRL(0)": TTF_REF_MEAN,
    "RMST(0, τ)": TTF_REF_RMST,
    "η (characteristic life)": TTF_REF_ETA,
}
TTF_REF_LABEL = {v: k for k, v in TTF_REFS.items()}

OBS = "#e5484d"     # naive observed mean TTF (Σt/N over failures)
OBS_MED = "#f08c00"  # naive observed median TTF (same censoring blindness, different bias)
KMR = "#1f5c99"     # censoring-corrected KM RMST
THETA = "#8452c9"   # the θ layer itself
RMULT = "#0b7285"   # what θ is worth in RMST — the two do NOT move together
COUNT = "#adb5bd"   # per-bin failure counts (evidence behind each estimate)

# Life series drawn per covariate bin: (column, colour, marker, label, truth key, truth label).
# Each solid observed line is paired with the dotted truth it should be recovering.
LIFE_PER_BIN = [
    ("obs_ttf_fail", OBS, "circle", "observed mean TTF (naive)", "mean", "true mean"),
    ("obs_ttf_median", OBS_MED, "diamond", "observed median TTF (naive)", "median", "true median"),
    ("km_rmst", KMR, "square", "KM RMST(0, {tau}d)", "rmst", "true RMST(0, {tau}d)"),
]

# Colour per competing failure mode, in declaration order.
MODE_SEQ = ["#1f5c99", "#e8590c", "#2f9e44", "#8452c9", "#c2255c", "#0b7285"]

# How many competing failure modes the UI will let you define.
MAX_MODES = 4


# ─────────────────────────────────────────────────────────────────────────────
# input registry: one place that declares every sidebar control, so the sidebar
# and the "Defaults" pane are two renderings of the same list rather than two
# lists that drift apart
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Inp:
    """One editable input: its factory default and the widget that shows it."""

    key: str
    label: str
    default: Any
    group: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple = ()
    help: str | None = None

    @property
    def kind(self) -> str:
        if self.choices:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, str):
            return "text"
        return "int" if isinstance(self.default, int) else "num"


def _mode_inputs() -> list[Inp]:
    """Name / β / η for each of the ``MAX_MODES`` competing failure modes."""
    out: list[Inp] = []
    for i in range(MAX_MODES):
        d = DEFAULT_FAIL_MODES[i] if i < len(DEFAULT_FAIL_MODES) else {
            "name": f"mode {i + 1}", "beta": 1.0, "eta_days": 700.0}
        out += [
            Inp(f"mode{i}_name", f"Mode {i + 1} name", str(d["name"]), "Failure modes"),
            Inp(f"mode{i}_beta", f"Mode {i + 1} β", float(d["beta"]), "Failure modes", 0.1, 10.0, 0.05),
            Inp(f"mode{i}_eta", f"Mode {i + 1} η (days)", float(d["eta_days"]), "Failure modes",
                1.0, 20000.0, 10.0),
        ]
    return out


# Ceilings sized for a shared 1-vCPU box: the simulation is a scalar loop, so one
# oversized run pegs the core and every other viewer waits it out.  These bound
# the *field*; they do not bound the run count on their own — a short eta renews
# pumps faster and blows it up from inside the caps — which is what run_cost() is
# for.  Raise them freely if the app moves somewhere with cores to spare.
MAX_WELLS, MAX_YEARS, MAX_SEEDS, MAX_SNAPSHOTS = 500, 20.0, 25, 48

INPUTS: list[Inp] = [
    # ── field growth ────────────────────────────────────────────────────────
    Inp("total_years", "Horizon (years)", 20.0, "Field growth", 1.0, MAX_YEARS, 1.0),
    Inp("ramp_years", "Growth / ramp (years)", 10.0, "Field growth", 0.5, MAX_YEARS, 0.5),
    Inp("plateau_wells", "Plateau wells", 100, "Field growth", 1, MAX_WELLS, 10),
    # ── failure law ─────────────────────────────────────────────────────────
    Inp("beta_fail", "β (shape)", 1.0, "Failure Weibull", 0.1, 10.0, 0.05),
    Inp("eta_fail_days", "η (characteristic life, days)", 365.0, "Failure Weibull",
        1.0, 20000.0, 5.0,
        help="Weibull scale in days: S(η) = 1/e ≈ 0.368. Not the mean — use the "
             "calculator below to go from RMST / MRL / median to η."),
    Inp("modes_on", "Competing failure modes (equipment)", False, "Failure modes"),
    Inp("n_modes", "How many modes", 2, "Failure modes", 2, MAX_MODES, 1),
    *_mode_inputs(),
    # ── workover ────────────────────────────────────────────────────────────
    Inp("wo_label", "Model", "none", "Workover",
        choices=("none", "deterministic (planned age)", "statistical (Weibull)")),
    Inp("ttf_reference", "Fraction is % of", TTF_REF_LABEL[TTF_REF_MEAN], "Workover",
        choices=tuple(TTF_REFS),
        help="What the two fractions below are fractions OF — both the planned pull age "
             "and the workover η are this number times the fraction. All three are the "
             "baseline (θ = 1) failure law: mean = MRL(0) = E[T]; RMST(0, τ) is that mean "
             "truncated at the reporting horizon τ, so it is always shorter (much shorter "
             "when β < 1 puts a long tail past τ); η is the scale, S(η) = 1/e."),
    Inp("pm_fraction", "Pull at fraction of TTF", 0.8, "Workover", 0.1, 3.0, 0.05),
    Inp("beta_wo", "Workover β (>1 = age-driven)", 1.3, "Workover", 0.5, 6.0, 0.1),
    Inp("wo_eta_fraction", "Workover η as fraction of TTF", 0.8, "Workover", 0.1, 3.0, 0.05),
    Inp("run_counterfactual", "Also run no-workover twin", True, "Workover"),
    # ── hazard layers ───────────────────────────────────────────────────────
    Inp("well_on", "Individual-well θ variation", False, "Hazard layers"),
    Inp("well_theta_ratio", "θ(worst well) / θ(best well)", 4.0, "Hazard layers", 1.0, 40.0, 0.5),
    Inp("well_bins", "Well groups", 5, "Hazard layers", 2, 20, 1),
    Inp("ql_on", "Ql layer — monotone", False, "Hazard layers"),
    Inp("ql_lo", "Ql low (m³/d)", 50.0, "Hazard layers", 1.0, 5000.0, 10.0),
    Inp("ql_hi", "Ql high (m³/d)", 250.0, "Hazard layers", 2.0, 10000.0, 10.0),
    Inp("ql_bins", "Ql bins", 5, "Hazard layers", 2, 20, 1),
    Inp("ql_shape", "Ql shape", QL_SHAPES[0], "Hazard layers", choices=tuple(QL_SHAPES)),
    Inp("ql_theta_ratio", "θ(high Ql bin) / θ(low Ql bin)", 2.0, "Hazard layers", 1.0, 20.0, 0.1),
    Inp("freq_on", "Frequency layer — U-shape", False, "Hazard layers"),
    Inp("freq_lo", "f low (Hz)", 40.0, "Hazard layers", 20.0, 49.0, 1.0),
    Inp("freq_hi", "f high (Hz)", 60.0, "Hazard layers", 51.0, 90.0, 1.0),
    Inp("freq_bins", "Frequency bins", 5, "Hazard layers", 2, 20, 1),
    Inp("freq_theta_edge", "θ at the ends", 2.0, "Hazard layers", 1.0, 20.0, 0.1),
    Inp("hazard_center", "Center θ (fleet mean θ = 1)", True, "Hazard layers",
        help="Divide θ by its population mean so switching a layer on redistributes "
             "hazard between bins instead of shifting the whole fleet."),
    # ── downtime ────────────────────────────────────────────────────────────
    Inp("downtime_fail_days", "After failure (days)", 7.0, "Downtime", 0.0, 365.0, 0.5),
    Inp("downtime_workover_days", "After workover (days)", 3.0, "Downtime", 0.0, 365.0, 0.5),
    # ── renewal & fitting ───────────────────────────────────────────────────
    Inp("rmst_tau_days", "RMST horizon τ (days)", 730.0, "Renewal & fitting", 30.0, 20000.0, 10.0),
    Inp("n_seeds", "Seeds (field realizations)", 25, "Renewal & fitting", 1, MAX_SEEDS, 5),
    Inp("n_fit_snapshots", "Trend snapshots", 24, "Renewal & fitting", 4, MAX_SNAPSHOTS, 2),
    Inp("base_seed", "Base seed", 12345, "Renewal & fitting", 0, 10_000_000, 1),
    Inp("km_years_txt", "KM snapshot years (comma-sep)", "1, 3, 5, 10, 15, 20", "Renewal & fitting"),
]

INPUT_BY_KEY = {i.key: i for i in INPUTS}
INPUT_GROUPS = list(dict.fromkeys(i.group for i in INPUTS))


# ─────────────────────────────────────────────────────────────────────────────
# defaults: the factory value of every input, overridable from the Defaults pane
# ─────────────────────────────────────────────────────────────────────────────
def defaults() -> dict:
    """The user's saved input defaults, read from disk once per session."""
    if "defaults" not in st.session_state:
        st.session_state["defaults"] = sanitize_defaults(load_app_defaults())
    return st.session_state["defaults"]


def sanitize_defaults(raw: dict) -> dict:
    """Keep only values the registry still recognises, coerced to its types.

    The file outlives the code that wrote it, so an input that was renamed,
    retyped or had its range narrowed must be dropped rather than handed to a
    widget that will reject it — losing one stale default beats failing to start.
    """
    out: dict = {}
    for key, value in (raw or {}).items():
        inp = INPUT_BY_KEY.get(key)
        if inp is None:
            continue
        try:
            if inp.kind == "bool":
                out[key] = bool(value)
            elif inp.kind == "text":
                out[key] = str(value)
            elif inp.kind == "choice":
                if value in inp.choices:
                    out[key] = value
            elif inp.kind == "int":
                out[key] = int(np.clip(int(value), inp.lo, inp.hi))
            else:
                out[key] = float(np.clip(float(value), inp.lo, inp.hi))
        except (TypeError, ValueError):
            continue
    return out


def D(key: str):
    """The value an input starts at — the user's default if they set one."""
    return defaults().get(key, INPUT_BY_KEY[key].default)


def generation() -> int:
    """Bumped whenever the defaults change; part of every widget's key.

    A widget keeps whatever the browser last put in it: deleting its
    ``session_state`` entry is not enough, because the frontend re-sends the old
    value on the next run and Streamlit takes that over the ``value=`` argument.
    Changing the *key* is what makes it a different widget with no history, so a
    new default actually shows up.
    """
    return st.session_state.setdefault("input_generation", 0)


def sidebar_key(key: str) -> str:
    return f"w{generation()}_{key}"


def widget(key: str, container=None, **kw):
    """Render one registered input, seeded from its current default."""
    return _render(INPUT_BY_KEY[key], sidebar_key(key), D(key), container or st, **kw)


def _render(inp: Inp, state_key: str, value, c, **kw):
    label = kw.pop("label", inp.label)
    kw.setdefault("help", inp.help)
    if inp.kind == "bool":
        return c.checkbox(label, value=bool(value), key=state_key, **kw)
    if inp.kind == "text":
        return c.text_input(label, value=str(value), key=state_key, **kw)
    if inp.kind == "choice":
        opts = list(inp.choices)
        idx = opts.index(value) if value in opts else 0
        return c.selectbox(label, opts, index=idx, key=state_key, **kw)
    if inp.kind == "int":
        lo, hi = int(inp.lo), int(inp.hi)
        return c.number_input(label, lo, hi, int(np.clip(value, lo, hi)), int(inp.step or 1),
                              key=state_key, **kw)
    lo, hi = float(inp.lo), float(inp.hi)
    return c.number_input(label, lo, hi, float(np.clip(value, lo, hi)), float(inp.step or 1.0),
                          key=state_key, **kw)


@st.dialog("Defaults for every input", width="large")
def defaults_dialog() -> None:
    """Edit the value each sidebar control starts at.

    Deliberately a flat dump of the whole registry rather than a curated form —
    the point is to be able to reach any input without hunting for the expander
    it happens to live in.
    """
    st.caption("These are the values the sidebar controls open at. Applying them resets the "
               "sidebar to the new defaults and **saves them to disk**, so they survive "
               "restarting the app; the saved experiments are untouched.  \n"
               f"Stored in `{APP_DEFAULTS_PATH}`.")
    gen, cur = generation(), defaults()
    staged: dict = {}
    for group in INPUT_GROUPS:
        st.markdown(f"**{group}**")
        items = [i for i in INPUTS if i.group == group]
        cols = st.columns(3)
        # fill column-major, so a column is a contiguous run of the registry and
        # things that belong together (a mode's name/β/η) stay adjacent
        per_col = -(-len(items) // 3)
        for n, inp in enumerate(items):
            staged[inp.key] = _render(inp, f"d{gen}_{inp.key}", cur.get(inp.key, inp.default),
                                      cols[min(n // per_col, 2)])
        st.divider()

    c1, c2 = st.columns(2)
    if c1.button("Apply defaults", type="primary", use_container_width=True):
        _commit_defaults({k: v for k, v in staged.items() if v != INPUT_BY_KEY[k].default})
    if c2.button("Reset to factory", use_container_width=True):
        _commit_defaults({})


def _commit_defaults(new: dict) -> None:
    """Save the new defaults to disk and re-run the app on fresh widget keys."""
    st.session_state["defaults"] = new
    save_app_defaults(new)
    st.session_state["input_generation"] = generation() + 1
    for k in [k for k in st.session_state if _is_input_key(k)]:
        del st.session_state[k]
    # the dialog is a fragment, so an unscoped rerun would only redraw the
    # dialog and leave the sidebar on its old values
    st.rerun(scope="app")


def _is_input_key(key: str) -> bool:
    """True for a registry-generated widget key (``w<gen>_x`` / ``d<gen>_x``)."""
    head, _, name = key.partition("_")
    return bool(head[:1] in "wd" and head[1:].isdigit() and name in INPUT_BY_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# sidebar: choose to run a new experiment or load a saved one
# ─────────────────────────────────────────────────────────────────────────────
def eta_calculator(beta: float, tau: float) -> None:
    """Popover that inverts a life summary (RMST / MRL / median) back to η.

    "Apply" stages the answer in ``eta_pending``; the staged value is copied
    onto the η widget's key at the top of the next run, which is the only order
    Streamlit allows a widget's value to be set from elsewhere.
    """
    with st.popover(":material/calculate: η from RMST / MRL / median", use_container_width=True):
        st.caption("Which η gives the life I want? Inverts a **single baseline** Weibull, before "
                   "any hazard layer redistributes it and ignoring competing modes. β and τ start "
                   "from the sidebar and are then yours to vary here without touching the run.")
        c1, c2 = st.columns(2)
        b = c1.number_input("β", 0.1, 6.0, float(beta), 0.05, key="calc_beta")
        t = c2.number_input("τ (days)", 30.0, 20000.0, float(tau), 10.0, key="calc_tau")
        kind_label = st.radio("Target", list(CALC_TARGETS), horizontal=True, key="calc_kind")
        value = st.number_input("Target value (days)", 1.0, 20000.0, 400.0, 10.0, key="calc_value")

        try:
            eta = eta_from_life(value, CALC_TARGETS[kind_label], b, t)
        except ValueError as exc:
            st.warning(str(exc))
            return
        life = weibull_life(b, eta, t)
        st.metric("η (days)", f"{eta:,.1f}")
        st.caption(
            f"That η implies RMST(0,{t:g}) = **{life['rmst']:.1f} d**, "
            f"MRL(0) = **{life['mean']:.1f} d**, median = **{life['median']:.1f} d**."
        )
        if st.button("Apply as η", use_container_width=True, key="calc_apply"):
            st.session_state["eta_pending"] = float(eta)
            st.rerun()


def hazard_inputs() -> dict:
    """Hazard-layer controls; returns the SimConfig kwargs they set."""
    with st.sidebar.expander("Hazard layers (θ)", expanded=False):
        well_on = widget("well_on")
        well = dict(well_bins=int(D("well_bins")), well_theta_ratio=float(D("well_theta_ratio")))
        if well_on:
            c1, c2 = st.columns(2)
            well["well_theta_ratio"] = widget("well_theta_ratio", c1, label="θ max / min")
            well["well_bins"] = int(widget("well_bins", c2, label="Groups"))
            st.caption("Drawn **once per well and kept for life** — every pump ever installed in "
                       "that slot inherits it. That permanence is what makes it a frailty: the "
                       "runs inside a well are correlated, the fleet is a mixture, and the pooled "
                       "Weibull it produces is not the law any single pump follows.")

        ql_on = widget("ql_on")
        ql = {k: D(k) for k in ("ql_lo", "ql_hi", "ql_bins", "ql_shape", "ql_theta_ratio")}
        if ql_on:
            c1, c2 = st.columns(2)
            ql["ql_lo"] = widget("ql_lo", c1, label="Ql low (m³/d)")
            ql["ql_hi"] = widget("ql_hi", c2, label="Ql high (m³/d)")
            c3, c4 = st.columns(2)
            ql["ql_bins"] = int(widget("ql_bins", c3, label="Bins"))
            ql["ql_shape"] = widget("ql_shape", c4, label="Shape")
            ql["ql_theta_ratio"] = widget("ql_theta_ratio", label="θ(high bin) / θ(low bin)")

        freq_on = widget("freq_on")
        fq = {k: D(k) for k in ("freq_lo", "freq_hi", "freq_bins", "freq_theta_edge")}
        if freq_on:
            c1, c2 = st.columns(2)
            fq["freq_lo"] = widget("freq_lo", c1)
            fq["freq_hi"] = widget("freq_hi", c2)
            c3, c4 = st.columns(2)
            fq["freq_bins"] = int(widget("freq_bins", c3, label="Bins"))
            fq["freq_theta_edge"] = widget("freq_theta_edge", c4)
            st.caption("θ = 1 + k·(1 − f/50)², k set so the end frequencies reach that θ.")

        hazard_center = widget("hazard_center")
        st.caption("Bins are equal-width and **equally likely**. θ enters as η·θ^(−1/β) on the "
                   "failure law only — on *every* competing mode, so it is a whole-pump "
                   "multiplier. Ql and frequency redraw on each replacement; the well layer "
                   "does not.")

    kwargs = dict(well_layer_on=well_on, ql_layer_on=ql_on, freq_layer_on=freq_on,
                  hazard_center=hazard_center, **well, **ql, **fq)
    try:
        layers = layers_from_config(SimConfig(**kwargs))
    except ValueError as exc:
        st.sidebar.error(f"Hazard layer: {exc}")
        return dict(kwargs, well_layer_on=False, ql_layer_on=False, freq_layer_on=False)
    if layers:
        st.sidebar.caption(" · ".join(
            f"**{s.key}**: θ {s.theta.min():.2f}–{s.theta.max():.2f} over {s.n_bins} bins"
            for s in layers))
        span, sd = theta_spread(layers)
        st.sidebar.caption(f"Population θ spread **×{span:.1f}**, sd(log θ) = **{sd:.2f}** — "
                           "the axis the marginal β reacts to.")
    return kwargs


def mode_inputs() -> dict:
    """Competing-failure-mode controls; returns the SimConfig kwargs they set."""
    modes_on = widget("modes_on", help="Race several independent Weibulls for the pump — one per "
                                       "piece of equipment. The run ends at the first of them.")
    if not modes_on:
        return dict(modes_on=False, fail_modes=[dict(m) for m in DEFAULT_FAIL_MODES])

    n_modes = int(widget("n_modes"))
    fail_modes = []
    for i in range(n_modes):
        c1, c2, c3 = st.columns([3, 2, 3])
        fail_modes.append({
            "name": widget(f"mode{i}_name", c1, label=f"Mode {i + 1}"),
            "beta": float(widget(f"mode{i}_beta", c2, label="β")),
            "eta_days": float(widget(f"mode{i}_eta", c3, label="η (d)")),
        })
    st.caption("T_fail = **min** over the modes, and the winner is recorded — so each mode is "
               "censored by the others. The single β/η above is **not used** while this is on.")
    return dict(modes_on=True, fail_modes=fail_modes)


WO_MODES = {
    "none": WORKOVER_NONE,
    "deterministic (planned age)": WORKOVER_DETERMINISTIC,
    "statistical (Weibull)": WORKOVER_STATISTICAL,
}

# Above this the run stops being interactive; on a one-core host it also parks
# every other viewer behind it, so it is worth saying before the click, not after.
SLOW_RUN_SECONDS = 25.0


# (wells, seeds, snapshots) of the three calibration runs.  Sized near real usage
# on purpose: per-run cost falls as the field grows (the vectorised parts amortise),
# so calibrating on toy fields over-states the price of a real one by ~2x.
CALIBRATION = ((80, 3, 12), (500, 8, 12), (500, 8, 24))


@st.cache_resource(show_spinner=False)
def cost_model() -> tuple[float, float, float]:
    """``(fixed, per run, per run·snapshot)`` seconds, measured on *this* machine.

    Constants baked in here would be wrong everywhere but the laptop they were
    measured on — a small VPS core runs several times slower — so the model is
    fitted at import from three throwaway experiments costing ~0.1 s in total.

    Three, because the work has two independent factors and one point cannot see
    either: simulating is ``O(runs)`` once, but *fitting* re-reads the whole
    sample at every trend snapshot, so it is ``O(runs × snapshots)``. Two points
    varying only the field size miss the second term and under-estimate a
    24-snapshot run by half.

    Measured against real runs the result lands roughly −15 % … +50 %, biased
    high — the safe direction for a warning, and the reason the UI says "~".
    """
    pts = []
    for wells, seeds, snaps in CALIBRATION:
        cfg = SimConfig(plateau_wells=wells, total_years=10, n_seeds=seeds,
                        n_fit_snapshots=snaps, km_snapshot_years=[10])
        t0 = time.perf_counter()
        res = run_experiment(cfg)
        elapsed = time.perf_counter() - t0
        final = res.fit_table["snap_year"] == res.fit_table["snap_year"].max()
        pts.append((float(res.fit_table.loc[final, "n_runs"].mean()) * seeds, snaps, elapsed))

    (n1, s1, t1), (n2, _, t2), (n3, s3, t3) = pts
    # same field, more snapshots -> isolates the per-(run × snapshot) term
    per_run_snap = max((t3 - t2) / max((s3 - s1) * n3, 1.0), 0.0)
    # same snapshots, bigger field -> the rest of the slope is the per-run term
    per_run = max((t2 - t1) / max(n2 - n1, 1.0) - s1 * per_run_snap, 0.0)
    fixed = max(t2 - (per_run + s1 * per_run_snap) * n2, 0.0)
    return fixed, per_run, per_run_snap


def run_cost(cfg: SimConfig, twin: bool) -> tuple[float, float]:
    """``(pump runs, seconds)`` this configuration will cost, without running it.

    Renewal theory gives the answer directly: a slot turns over once per expected
    cycle, so the run count is well-days ÷ mean cycle. That catches the blow-ups
    the field-size caps cannot — a 5-day η renews pumps 70× faster than a 365-day
    one on exactly the same field.
    """
    well_days = float(np.sum(np.clip(cfg.horizon_days() - commission_days(cfg), 0.0, None)))
    cycle = float(expected_rates(cfg)["mean_cycle_days"])
    runs = well_days / max(cycle, 1e-9) * cfg.n_seeds * (2.0 if twin else 1.0)
    fixed, per_run, per_run_snap = cost_model()
    return runs, fixed + runs * (per_run + per_run_snap * cfg.n_fit_snapshots)


def run_cost_notice(cfg: SimConfig, twin: bool) -> None:
    """Show what the configured run will cost, next to the button that starts it."""
    try:
        runs, secs = run_cost(cfg, twin)
    except Exception:                      # a half-edited config is not worth a crash
        return
    took = "under a second" if secs < 1.0 else f"~**{secs:.0f} s**"
    text = f"≈ **{runs:,.0f}** pump runs · {took}" + (" (incl. twin)" if twin else "")
    if secs > SLOW_RUN_SECONDS:
        st.sidebar.warning(
            text + "  \nLong enough to block anyone else using this server. Fewer seeds or "
                   "wells, a shorter horizon, or a longer η all cut it.")
    else:
        st.sidebar.caption(text)


def sidebar_config() -> tuple[str, SimConfig | None, str | None, str, bool]:
    # staged η from the calculator — must land before the widget is created
    if "eta_pending" in st.session_state:
        st.session_state[sidebar_key("eta_fail_days")] = st.session_state.pop("eta_pending")

    st.sidebar.title(":material/experiment: Field simulation")
    mode = st.sidebar.radio("Source", ["Run new", "Load saved"], horizontal=True)

    if mode == "Load saved":
        rows = list_experiments()
        if not rows:
            st.sidebar.info("No saved experiments yet — run one first.")
            return "load", None, None, "", False
        labels = {f"{r['label']}  ·  {r['created']}": r["experiment_id"] for r in rows}
        pick = st.sidebar.selectbox("Experiment", list(labels))
        return "load", None, labels[pick], "", False

    if st.sidebar.button(":material/tune: Defaults…", use_container_width=True,
                         help="Change the value every control below opens at. Saved to disk, "
                              "so they persist across restarts."):
        defaults_dialog()
    if defaults():
        st.sidebar.caption(f":material/save: {len(defaults())} input(s) on a saved custom default.")

    with st.sidebar.expander("Field growth", expanded=False):
        total_years = widget("total_years")
        ramp_years = widget("ramp_years")
        plateau_wells = widget("plateau_wells")

    with st.sidebar.expander("Failure Weibull (truth)", expanded=True):
        mode_kwargs = mode_inputs()
        on = mode_kwargs["modes_on"]
        beta_fail = widget("beta_fail", disabled=on)
        eta_fail_days = widget("eta_fail_days", disabled=on)
        if not on:
            st.caption(f"= {eta_fail_days / 365.0:.2f} yr")

    with st.sidebar.expander("Workover (competing risk)", expanded=False):
        workover_mode = WO_MODES[widget("wo_label")]
        ttf_reference = TTF_REFS[widget("ttf_reference")]
        pm_fraction, beta_wo, wo_eta_fraction = D("pm_fraction"), D("beta_wo"), D("wo_eta_fraction")
        if workover_mode == WORKOVER_DETERMINISTIC:
            pm_fraction = widget("pm_fraction")
        elif workover_mode == WORKOVER_STATISTICAL:
            beta_wo = widget("beta_wo")
            wo_eta_fraction = widget("wo_eta_fraction")

        run_counterfactual = False
        if workover_mode != WORKOVER_NONE:
            run_counterfactual = widget(
                "run_counterfactual",
                help="Run an identical field with the workover programme off (same seeds) to "
                     "isolate its effect on failure rate and oil loss.",
            )

    hazard_kwargs = hazard_inputs()

    with st.sidebar.expander("Downtime (well not producing)", expanded=False):
        downtime_fail_days = widget("downtime_fail_days")
        downtime_workover_days = widget("downtime_workover_days")

    with st.sidebar.expander("Renewal & fitting", expanded=False):
        rmst_tau_days = widget("rmst_tau_days")
        n_seeds = widget("n_seeds")
        n_fit_snapshots = widget("n_fit_snapshots")
        base_seed = widget("base_seed")
        km_years_txt = widget("km_years_txt")
        try:
            km_years = [float(x) for x in km_years_txt.split(",") if x.strip()]
        except ValueError:
            km_years = [1, 3, 5, 10, 15, 20]
            st.warning("Could not parse KM years; using defaults.")

    with st.sidebar.container():
        eta_calculator(beta_fail, rmst_tau_days)
        label = st.text_input("Label (optional)", "", key="w_label")

    cfg = SimConfig(
        total_years=total_years,
        ramp_years=ramp_years,
        plateau_wells=int(plateau_wells),
        beta_fail=beta_fail,
        eta_fail_days=eta_fail_days,
        workover_mode=workover_mode,
        ttf_reference=ttf_reference,
        pm_fraction=pm_fraction,
        beta_wo=beta_wo,
        wo_eta_fraction=wo_eta_fraction,
        downtime_fail_days=downtime_fail_days,
        downtime_workover_days=downtime_workover_days,
        rmst_tau_days=rmst_tau_days,
        km_snapshot_years=km_years,
        n_fit_snapshots=int(n_fit_snapshots),
        n_seeds=int(n_seeds),
        base_seed=int(base_seed),
        **mode_kwargs,
        **hazard_kwargs,
    )
    run_cost_notice(cfg, run_counterfactual)
    run_clicked = st.sidebar.button(":material/play_arrow: Run simulation", type="primary", use_container_width=True)
    return ("run" if run_clicked else "idle"), cfg, None, label, run_counterfactual


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────
def has_failures_only_curves(result) -> bool:
    """Whether this experiment stored the naive failures-only KM alongside the KM."""
    return any("time_fo" in c for c in result.km_curves.values())


def failures_only_table(result) -> pd.DataFrame:
    """The β/η each snapshot's failures-only fit landed on, next to the honest one.

    One row per KM snapshot, because the whole point is that the failures-only
    numbers *move*: they are a function of how much of the field is still alive,
    not of the failure law, so a single headline value would hide the story.
    """
    agg = aggregate_by_snapshot(result.fit_table)
    rows = []
    for key in sorted(result.km_curves, key=float):
        match = agg.loc[np.isclose(agg["snap_year"], float(key))]
        if match.empty:
            continue
        r = match.iloc[0]
        bf, ef = float(r["beta_fo_median"]), float(r["eta_fo_median"])
        bc, ec = float(r["beta_cens_median"]), float(r["eta_cens_median"])
        c = result.km_curves[key]
        rows.append({
            "snapshot, yr": float(key),
            "failures": int(c["n_fail"]),
            "censored": int(c["n_runs"]) - int(c["n_fail"]),
            "β failures-only": bf,
            "η failures-only, d": ef,
            "β censoring-aware": bc,
            "η censoring-aware, d": ec,
            "η shortfall": 100.0 * (ef / ec - 1.0) if np.isfinite(ec) and ec else np.nan,
        })
    return pd.DataFrame(rows)


def km_figure(result, fo_km: bool = False, fo_fit: bool = False) -> go.Figure:
    cfg = result.config
    fig = go.Figure()
    years = sorted(result.km_curves, key=lambda k: float(k))
    agg = aggregate_by_snapshot(result.fit_table) if fo_fit else None
    tmax = max((max(result.km_curves[k]["time"]) for k in years), default=cfg.eta_fail_days * 3)
    tgrid = age_grid(tmax)

    for i, key in enumerate(years):
        c = result.km_curves[key]
        color = KM_SEQ[i % len(KM_SEQ)]
        group = f"yr{key}"
        fig.add_trace(
            go.Scatter(
                x=c["time"], y=c["surv"], mode="lines", line=dict(color=color, width=2, shape="hv"),
                name=f"yr {key} · pop {c.get('running_pop','?')} · {c['n_fail']} fails (pooled)",
                legendgroup=group,
            )
        )
        # the same snapshot, censored runs thrown away — dotted, same colour, and
        # in the same legend group so a year's pair toggles together
        if fo_km and "time_fo" in c:
            fig.add_trace(go.Scatter(
                x=c["time_fo"], y=c["surv_fo"], mode="lines",
                line=dict(color=color, width=1.6, shape="hv", dash="dot"),
                name=f"yr {key} · failures-only", legendgroup=group, showlegend=False,
                hovertemplate="failures-only · S=%{y:.3f} at %{x:.0f}d<extra></extra>"))
        if fo_fit and agg is not None:
            row = agg.loc[np.isclose(agg["snap_year"], float(key))]
            if not row.empty:
                b, e = float(row["beta_fo_median"].iloc[0]), float(row["eta_fo_median"].iloc[0])
                if np.isfinite(b) and np.isfinite(e):
                    fig.add_trace(go.Scatter(
                        x=tgrid, y=np.exp(-((tgrid / e) ** b)), mode="lines",
                        line=dict(color=color, width=1.6, dash="dashdot"),
                        # the fitted parameters ARE the result here, so they go in
                        # the legend rather than hiding behind a hover
                        name=f"yr {key} · failures-only fit  β={b:.2f}, η={e:.0f}d",
                        legendgroup=group,
                        hovertemplate=f"failures-only fit β={b:.2f}, η={e:.0f}d"
                                      "<br>S=%{y:.3f} at %{x:.0f}d<extra></extra>"))

    fig.add_trace(
        go.Scatter(
            x=tgrid, y=mixture_survival(tgrid, cfg),
            mode="lines", line=dict(color=TRUE, width=2.5, dash="dash"),
            name=truth_curve_label(cfg),
        )
    )
    # the KM overlay carries no parameters, so one style key stands in for all
    # its per-year copies; the fitted curves label themselves instead
    if fo_km:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#6b7280", width=1.6, dash="dot"),
            name="failures-only KM (censored dropped)"))
    fig.update_layout(
        # tall: this panel is read vertically (how far apart the curves sit at a
        # given age), and with the overlays on it carries up to three per year
        height=760, template="plotly_white",
        xaxis_title="pump age (days)",
        xaxis_range=[0, AGE_MAX_DAYS],
        yaxis_title="S(t)   ·   failure = event, workover/running = censored",
        yaxis_range=[0, 1.02],
        legend=dict(orientation="v", x=1.0, y=1.0, xanchor="right", font=dict(size=11)),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def _band(fig, x, agg, param, arm, color, name, row, col, symbol="circle"):
    lo, hi, mid = agg[f"{param}_{arm}_p10"], agg[f"{param}_{arm}_p90"], agg[f"{param}_{arm}_median"]
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=color.replace(")", ", 0.13)").replace("rgb", "rgba"),
                             showlegend=False, hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=mid, mode="lines+markers", line=dict(color=color, width=2.4),
                             marker=dict(size=5, symbol=symbol), name=name, legendgroup=name,
                             showlegend=(row == 1 and col == 1)), row=row, col=col)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return f"rgb({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)})"


def trends_figure(result, x_key: str = "time") -> go.Figure:
    cfg = result.config
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    col_name, x_title, short = X_CHOICES.get(x_key, X_CHOICES["time"])
    x = agg[col_name].to_numpy()
    cens_rgb, fo_rgb, all_rgb = _hex_to_rgb(CENS), _hex_to_rgb(FO), _hex_to_rgb(ALL)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Shape β vs {short}", f"Scale η vs {short}"),
        horizontal_spacing=0.09,
    )
    # with competing modes the pooled fit has no single truth to sit on — the
    # per-mode panel is where the ground truth lives
    single_mode = cfg.n_modes() == 1
    for (col, param) in [(1, "beta"), (2, "eta")]:
        _band(fig, x, agg, param, "cens", cens_rgb, "censored (cause-specific)", 1, col, symbol="circle")
        _band(fig, x, agg, param, "fo", fo_rgb, "failures-only", 1, col, symbol="square")
        _band(fig, x, agg, param, "all", all_rgb, "all-pulls (WO = failure)", 1, col, symbol="diamond")
        if single_mode:
            truth = cfg.beta_fail if param == "beta" else cfg.eta_fail_days
            fig.add_hline(y=truth, line=dict(color=TRUE, dash="dash", width=1.6), row=1, col=col)

    # observed mean TTF (naive Σt/N) overlaid on the eta / life panel
    _metric_trace(fig, x, agg, "obs_ttf_fail", FO, "obs. mean TTF (failures)", 1, 2, dash="dash", band=False)
    _metric_trace(fig, x, agg, "obs_ttf_all", ALL, "obs. mean TTF (fail+WO)", 1, 2, dash="dash", band=False)

    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_xaxes(title_text=x_title, row=1, col=2)
    fig.update_yaxes(title_text="β̂", row=1, col=1)
    fig.update_yaxes(title_text="η̂ (days)", type="log", row=1, col=2)
    fig.update_layout(
        height=480, template="plotly_white",
        legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=70, b=50),
        title=dict(text=f"Fitted Weibull vs truth (median · p10–p90 band · {cfg.n_seeds} seeds)",
                   x=0.5, xanchor="center", font=dict(size=14)),
    )
    return fig


def modes_figure(result) -> go.Figure:
    """Per-mode cause-specific β and η vs field age, against each mode's truth.

    Each mode is fitted with the *other* modes treated as censoring, which is
    the estimator that gives a node back its own law; the dotted line is the
    β/η that node was actually simulated from.
    """
    cfg = result.config
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x = agg["snap_year"].to_numpy()
    modes = cfg.modes()

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Shape β per failure mode", "Scale η per failure mode"))
    for m, (name, beta, eta) in enumerate(modes):
        color = MODE_SEQ[m % len(MODE_SEQ)]
        for col, param, truth in [(1, "beta", beta), (2, "eta", eta)]:
            _metric_trace(fig, x, agg, f"{param}_m{m}", color, name, 1, col,
                          band=True, show=(col == 1))
            # right-hand annotations: the truths differ, so they stack instead
            # of piling up on the noisy young-field end of the curve
            fig.add_hline(y=truth, line=dict(color=color, dash="dot", width=1.4), row=1, col=col,
                          annotation_text=f"true {name} {truth:g}",
                          annotation_position="top right",
                          annotation_font_size=9, annotation_font_color=color)
    # the pooled fit, for contrast: it lands on no mode's truth
    _metric_trace(fig, x, agg, "beta_cens", TRUE, "pooled (all modes as one)", 1, 1,
                  dash="dash", band=False)
    _metric_trace(fig, x, agg, "eta_cens", TRUE, "pooled (all modes as one)", 1, 2,
                  dash="dash", band=False, show=False)

    fig.update_xaxes(title_text="field age (years)", row=1, col=1)
    fig.update_xaxes(title_text="field age (years)", row=1, col=2)
    fig.update_yaxes(title_text="β̂", row=1, col=1)
    fig.update_yaxes(title_text="η̂ (days)", type="log", row=1, col=2)
    fig.update_layout(
        height=500, template="plotly_white",
        legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center", font=dict(size=10.5)),
        margin=dict(l=60, r=20, t=115, b=50),
        title=dict(text=f"Cause-specific fit per mode (median · p10–p90 · {cfg.n_seeds} seeds)",
                   x=0.5, xanchor="center", y=0.975, yanchor="top", font=dict(size=14)),
    )
    return fig


def mode_survival_figure(result) -> go.Figure:
    """Each mode's own S(t), the combined failure law, and the mode CIFs."""
    cfg = result.config
    modes = cfg.modes()
    t = age_grid()
    combined = mixture_survival(t, cfg)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Survival: each mode alone vs combined",
                                        "Share of failures each mode wins, by age"))
    for m, (name, beta, eta) in enumerate(modes):
        color = MODE_SEQ[m % len(MODE_SEQ)]
        fig.add_trace(go.Scatter(x=t, y=np.exp(-((t / eta) ** beta)), mode="lines",
                                 line=dict(color=color, width=2, dash="dot"),
                                 name=f"{name} alone", legendgroup=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=combined, mode="lines", line=dict(color=TRUE, width=2.8),
                             name="combined (what the field shows)"), row=1, col=1)
    # the 1/e line and where each curve crosses it: that crossing IS eta, so the
    # gap between the markers is the whole "why is the pooled eta below both"
    fig.add_hline(y=float(np.exp(-1.0)), line=dict(color="#adb5bd", width=1, dash="dot"),
                  row=1, col=1, annotation_text="1/e", annotation_position="right",
                  annotation_font_size=9)
    eq = eta_equivalent_days(cfg)
    xs = [e for _, _, e in modes] + [eq]
    labels = [n for n, _, _ in modes] + ["combined"]
    cols = [MODE_SEQ[m % len(MODE_SEQ)] for m in range(len(modes))] + [TRUE]
    # with the axis capped at five years, an eta beyond it has no marker to draw — say so
    # in words rather than let the number disappear silently
    on = [i for i, x in enumerate(xs) if np.isfinite(x) and x <= AGE_MAX_DAYS]
    off = [i for i, x in enumerate(xs) if np.isfinite(x) and x > AGE_MAX_DAYS]
    fig.add_trace(go.Scatter(
        x=[xs[i] for i in on], y=[np.exp(-1.0)] * len(on),
        # "x-thin" is drawn entirely by the marker outline, so the colour has to
        # go on marker.line — marker.color paints a fill this symbol has none of
        mode="markers", marker=dict(size=11, symbol="x-thin",
                                    line=dict(width=3, color=[cols[i] for i in on])),
        name="η (S = 1/e)", customdata=[labels[i] for i in on],
        hovertemplate="%{customdata}: η = %{x:.0f} d<extra></extra>"), row=1, col=1)
    if off:
        fig.add_annotation(
            row=1, col=1, xref="x domain", yref="y domain", x=0.98, y=0.98,
            xanchor="right", yanchor="top", showarrow=False, font=dict(size=9, color=TRUE),
            text="η beyond the 5-year axis: "
                 + ", ".join(f"{labels[i]} {xs[i]:.0f} d" for i in off))

    # sub-density share: mode m's hazard as a fraction of the total hazard
    haz = np.stack([(beta / eta) * np.power(np.maximum(t, 1e-9) / eta, beta - 1.0)
                    for _, beta, eta in modes])
    share = haz / np.clip(haz.sum(axis=0), 1e-12, None)
    for m, (name, _, _) in enumerate(modes):
        fig.add_trace(go.Scatter(x=t, y=100.0 * share[m], mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=MODE_SEQ[m % len(MODE_SEQ)]),
                                 name=name, legendgroup=name, showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="pump age (days)", row=1, col=1, range=[0, AGE_MAX_DAYS])
    fig.update_xaxes(title_text="pump age (days)", row=1, col=2, range=[0, AGE_MAX_DAYS])
    fig.update_yaxes(title_text="S(t)", row=1, col=1, range=[0, 1.02])
    fig.update_yaxes(title_text="% of failures at that age", row=1, col=2, range=[0, 100])
    fig.update_layout(height=420, template="plotly_white",
                      legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center",
                                  font=dict(size=10.5)),
                      margin=dict(l=60, r=20, t=80, b=50))
    return fig


def _metric_trace(fig, x, agg, param, color, name, row, col, dash=None, band=True, show=True):
    if band:
        rgba = _hex_to_rgb(color).replace("rgb", "rgba").replace(")", ", 0.13)")
        fig.add_trace(go.Scatter(x=x, y=agg[f"{param}_p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=agg[f"{param}_p10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba, showlegend=False,
                                 hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=agg[f"{param}_median"], mode="lines+markers",
                             line=dict(color=color, width=2.4, dash=dash), marker=dict(size=4),
                             name=name, showlegend=show), row=row, col=col)


def rates_figure(result) -> go.Figure:
    """Failure/pull rate and oil loss vs time, with the no-workover twin overlaid."""
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x = agg["snap_year"].to_numpy()
    nowo = None
    if result.fit_table_nowo is not None:
        nowo = aggregate_by_snapshot(result.fit_table_nowo).sort_values("snap_year")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Failure &amp; pull rate (per well-year)", "Oil loss (% of well-days)"),
        horizontal_spacing=0.10,
    )
    _metric_trace(fig, x, agg, "fail_rate_well_yr", RATE_FAIL, "failures / well-yr", 1, 1, band=True)
    _metric_trace(fig, x, agg, "pull_rate_well_yr", RATE_PULL, "pulls (fail+WO) / well-yr", 1, 1, dash="dot", band=False)
    _metric_trace(fig, x, agg, "oil_loss_pct", LOSS, "oil loss % (with programme)", 1, 2, band=True)
    if nowo is not None:
        xn = nowo["snap_year"].to_numpy()
        _metric_trace(fig, xn, nowo, "fail_rate_well_yr", CF, "failures / well-yr (no workover)", 1, 1, dash="dash", band=False)
        _metric_trace(fig, xn, nowo, "oil_loss_pct", CF, "oil loss % (no workover)", 1, 2, dash="dash", band=False)

    # analytic steady-state estimate from the parameters (dotted reference lines)
    exp = expected_rates(result.config)
    fig.add_hline(y=exp["fail_per_well_year"], line=dict(color="#212529", dash="dot", width=1),
                  row=1, col=1, annotation_text=f"exp. fail {exp['fail_per_well_year']:.2f}",
                  annotation_position="top left", annotation_font_size=9)
    if exp["workover_per_well_year"] > 1e-6:
        fig.add_hline(y=exp["pull_per_well_year"], line=dict(color="#212529", dash="dot", width=1),
                      row=1, col=1, annotation_text=f"exp. pull {exp['pull_per_well_year']:.2f}",
                      annotation_position="bottom right", annotation_font_size=9)
    fig.add_hline(y=exp["oil_loss_pct"], line=dict(color="#212529", dash="dot", width=1),
                  row=1, col=2, annotation_text=f"exp. {exp['oil_loss_pct']:.2f}%",
                  annotation_position="top left", annotation_font_size=9)

    fig.update_xaxes(title_text="field age (years)", row=1, col=1)
    fig.update_xaxes(title_text="field age (years)", row=1, col=2)
    fig.update_yaxes(title_text="events per well-year", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="% of well-days lost", row=1, col=2, rangemode="tozero")
    fig.update_layout(
        height=460, template="plotly_white",
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center", font=dict(size=10.5)),
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


LIFE_SERIES = [
    ("life_rmst", "#1f5c99", "RMST(0, τ)"),
    ("life_mrl0", "#2f9e44", "MRL(0) = KM mean"),
    ("life_median", "#8452c9", "KM median"),
    ("obs_ttf_fail", "#e5484d", "mean (naive Σt/N)"),
]


def life_figure(result) -> go.Figure:
    """RMST(0,τ), MRL(0), KM median, and naive mean of the failure distribution."""
    cfg = result.config
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    x = agg["snap_year"].to_numpy()
    fig = go.Figure()
    for col, color, name in LIFE_SERIES:
        rgba = _hex_to_rgb(color).replace("rgb", "rgba").replace(")", ", 0.12)")
        fig.add_trace(go.Scatter(x=x, y=agg[f"{col}_p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=agg[f"{col}_p10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=agg[f"{col}_median"], mode="lines+markers",
                                 line=dict(color=color, width=2.2), marker=dict(size=4), name=name))
    tl = expected_life(cfg)
    for y, txt, pos in [
        (tl["rmst"], f"true RMST {tl['rmst']:.0f}", "top left"),
        (tl["mean"], f"true mean/MRL {tl['mean']:.0f}", "bottom left"),
        (tl["median"], f"true median {tl['median']:.0f}", "top right"),
    ]:
        fig.add_hline(y=y, line=dict(color="#212529", dash="dot", width=1),
                      annotation_text=txt, annotation_position=pos, annotation_font_size=9)
    fig.update_layout(
        height=480, template="plotly_white", xaxis_title="field age (years)",
        yaxis_title=f"days   (τ = {cfg.rmst_tau_days:g}d)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def _unit_centred_range(values: np.ndarray, pad: float = 1.15) -> list[float]:
    """Axis range symmetric about 1.0, so two multiplier axes share that gridline."""
    d = float(np.nanmax(np.abs(np.asarray(values, dtype=float) - 1.0)))
    d = max(d * pad, 0.05)
    return [1.0 - d, 1.0 + d]


def _theta_curve(cfg, spec, grid: np.ndarray) -> np.ndarray:
    """Continuous design θ(covariate), rescaled onto the layer's binned θ.

    Centering divides every bin's θ by one constant, so recovering that constant
    from the bins puts the smooth shape and the markers on the same axis.
    """
    scale = float(np.mean(spec.theta / theta_design(cfg, spec, spec.centers)))
    return theta_design(cfg, spec, grid) * scale


def _bin_life(cfg, theta: np.ndarray, key: str) -> np.ndarray:
    """One life summary of the true failure law at each bin's θ (mode-aware)."""
    return np.array([life_given_theta(cfg, float(t), cfg.rmst_tau_days)[key] for t in theta])


def solo_means(cfg) -> list[float]:
    """E[T] of each failure mode **as if it were the only risk**.

    The number to compare a mode's η against: on its own a mode reaches 1/e at
    its η, but in the field it is racing the others, so the life the fleet sees
    is always shorter than any of these.
    """
    return [dataclasses.replace(cfg, modes_on=False, beta_fail=b,
                                eta_fail_days=e).true_mean_ttf_days()
            for _, b, e in cfg.modes()]


def bins_figure(result, layer_key: str, snap_year: float, seed: int | None = None) -> go.Figure:
    """Per-bin observed life vs the covariate, with the θ layer that produced it.

    Left panel: naive observed mean TTF (Σt/N over failures) and the
    censoring-corrected KM RMST(0,τ) per bin, against the truth each bin was
    simulated from, over grey bars carrying how many failures each bin is built
    on at this snapshot.  Right panel: the θ curve, design shape and bin values.

    ``seed`` picks one field realization instead of the pool; the p10-p90 band
    then closes, since a single seed has nothing to spread over.
    """
    cfg = result.config
    spec = {s.key: s for s in layers_from_config(cfg)}[layer_key]
    agg = aggregate_bins(result.bin_table, layer_key, snap_year, seed)
    x = agg["center"].to_numpy()
    theta = agg["theta"].to_numpy()
    tau = cfg.rmst_tau_days
    n_fail = agg["n_fail"].to_numpy()
    n_runs = agg["n_runs"].to_numpy()
    n_seeds = int(agg["n_seeds"].max())
    pop = f"seed {seed}" if seed is not None else f"{n_seeds} seeds pooled"

    # truth per bin: theta folded into the failure law, then its life summaries
    life_bin = [life_given_theta(cfg, float(t), tau) for t in theta]
    true = {k: np.array([b[k] for b in life_bin]) for k in ("mean", "median", "rmst")}
    true_rmst = true["rmst"]

    # what that theta is worth in life: RMST(0,tau) relative to the theta = 1 baseline.
    # theta multiplies; the RMST multiplier does NOT — it is compressed by the tau
    # restriction and by beta, which is exactly what the second axis is here to show.
    rmst0 = life_given_theta(cfg, 1.0, tau)["rmst"]
    rmst_mult = true_rmst / rmst0

    fig = make_subplots(
        # wide gutter: both inner axes carry a title and they collide at 0.10
        rows=1, cols=2, horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=(
            f"Observed life per bin at year {snap_year:g} — "
            f"{n_fail.sum():,} failures / {n_runs.sum():,} runs ({pop})",
            f"Hazard layer θ  →  RMST(0, {tau:g}d) multiplier",
        ),
    )
    # how much evidence each bin rests on — first, so it sits behind the curves
    per_seed = ("" if seed is not None else
                "<br>%{customdata[1]:,.0f} failures per seed")
    fig.add_trace(
        go.Bar(x=x, y=n_fail, name="failures in bin", marker_color=COUNT, opacity=0.5,
               width=float(np.diff(spec.edges).mean()) * 0.72,
               text=[f"{int(v):,}" for v in n_fail], textposition="outside",
               textfont=dict(size=10, color="#6b7280"),
               customdata=np.stack([n_runs, n_fail / max(n_seeds, 1)], axis=-1),
               hovertemplate="%{y:,} failures of %{customdata[0]:,} runs"
                             + per_seed + "<extra></extra>"),
        row=1, col=1, secondary_y=True,
    )
    for param, color, symbol, name, tkey, tname in LIFE_PER_BIN:
        series = agg[f"{param}_median"]
        if series.isna().all():   # a bin_table saved before this series existed
            continue
        rgba = _hex_to_rgb(color).replace("rgb", "rgba").replace(")", ", 0.13)")
        fig.add_trace(go.Scatter(x=x, y=agg[f"{param}_p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=agg[f"{param}_p10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba, showlegend=False,
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=series, mode="lines+markers",
                                 line=dict(color=color, width=2.6),
                                 marker=dict(size=9, symbol=symbol),
                                 name=name.format(tau=f"{tau:g}"), customdata=n_fail,
                                 hovertemplate="%{y:.0f} d  ·  %{customdata:,} failures"
                                               "<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=true[tkey], mode="lines+markers",
                                 line=dict(color=color, width=1.4, dash="dot"),
                                 marker=dict(size=5, symbol="x"),
                                 name=tname.format(tau=f"{tau:g}")), row=1, col=1)

    # θ: the continuous design shape plus the value each bin actually carries
    grid = np.linspace(spec.edges[0], spec.edges[-1], 240)
    theta_grid = _theta_curve(cfg, spec, grid)
    # the RMST response is smooth in θ, so evaluate it on a coarse θ ladder and
    # interpolate — a per-point life summary would be a quadrature per pixel
    theta_lo, theta_hi = float(theta_grid.min()), float(theta_grid.max())
    ladder = np.linspace(theta_lo, theta_hi, 24)
    mult_ladder = _bin_life(cfg, ladder, "rmst") / rmst0
    mult_grid = np.interp(theta_grid, ladder, mult_ladder)
    fig.add_trace(go.Scatter(x=grid, y=theta_grid, mode="lines",
                             line=dict(color=THETA, width=1.6, dash="dash"),
                             name="θ (design shape)",
                             hovertemplate="θ %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=theta, mode="markers", marker=dict(size=11, color=THETA),
                             name="θ (bin value)",
                             hovertemplate="θ %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=grid, y=mult_grid, mode="lines",
                             line=dict(color=RMULT, width=1.6),
                             name=f"RMST(0, {tau:g}d) multiplier",
                             hovertemplate="RMST ×%{y:.3f}<extra></extra>"),
                  row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=rmst_mult, mode="markers",
                             marker=dict(size=9, color=RMULT, symbol="square"),
                             name="RMST multiplier (bin)", customdata=true_rmst,
                             hovertemplate="RMST ×%{y:.3f}  =  %{customdata:.0f} d<extra></extra>"),
                  row=1, col=2, secondary_y=True)
    fig.add_hline(y=1.0, line=dict(color="#adb5bd", width=1, dash="dot"), row=1, col=2)

    for col in (1, 2):
        fig.update_xaxes(title_text=spec.label, row=1, col=col)
    fig.update_yaxes(title_text="days", row=1, col=1, rangemode="tozero", secondary_y=False)
    # leave headroom so the bar labels never collide with the life curves
    fig.update_yaxes(title_text="failures in bin", row=1, col=1, secondary_y=True,
                     rangemode="tozero", range=[0, float(n_fail.max()) * 3.2 or 1.0],
                     showgrid=False, title_font=dict(color="#6b7280"),
                     tickfont=dict(color="#6b7280"))
    # centre BOTH right-panel axes on 1.0 so the shared reference line reads honestly
    # on either scale, and the compression of the RMST response stays visible.
    fig.update_yaxes(title_text="θ (hazard multiplier)", row=1, col=2, secondary_y=False,
                     range=_unit_centred_range(np.r_[theta_grid, theta]),
                     title_font=dict(color=THETA), tickfont=dict(color=THETA))
    fig.update_yaxes(title_text=f"RMST(0, {tau:g}d) multiplier", row=1, col=2, secondary_y=True,
                     range=_unit_centred_range(np.r_[mult_grid, rmst_mult]), showgrid=False,
                     title_font=dict(color=RMULT), tickfont=dict(color=RMULT))
    fig.update_layout(
        height=500, template="plotly_white", bargap=0.25,
        # the legend runs to seven entries and wraps on a narrow window, so it
        # needs room above the subplot titles rather than on top of them
        legend=dict(orientation="h", y=1.26, x=0.5, xanchor="center", font=dict(size=10.5)),
        margin=dict(l=60, r=70, t=130, b=50),
    )
    return fig


def config_summary(result) -> None:
    cfg = result.config
    tl = expected_life(cfg)
    modes = cfg.modes()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Plateau wells", cfg.plateau_wells)
    if len(modes) == 1:
        c2.metric("True β / η", f"{cfg.beta_fail:g} / {cfg.eta_fail_days:g}d")
    else:
        c2.metric("Competing modes", f"{len(modes)}",
                  help=" · ".join(f"{n}: β={b:g}, η={e:g}d" for n, b, e in modes))
    c3.metric(f"True RMST(0,{cfg.rmst_tau_days:g}d)", f"{tl['rmst']:.0f}d")
    c4.metric("True MRL(0) / median", f"{tl['mean']:.0f} / {tl['median']:.0f}d")
    c5.metric("Seeds × horizon", f"{cfg.n_seeds} × {cfg.total_years:g}yr")
    if len(modes) > 1:
        etas = " / ".join(f"{e:g}" for _, _, e in modes)
        means = " / ".join(f"{m:.0f}" for m in solo_means(cfg))
        st.caption(
            "Competing failure modes — " + " · ".join(
                f"**{n}** β={b:g}, η={e:g}d" for n, b, e in modes)
            + ". The run ends at the first of them, so their **hazards add** and the combined "
              "law is shorter-lived than *every* mode on its own — before any hazard layer, "
              f"S(t) = 1/e at **{eta_equivalent_days(cfg):.0f}d** against {etas}d, and the mean "
              f"is **{life_given_theta(cfg, 1.0)['mean']:.0f}d** against {means}d. A pooled fit "
              "landing below both η is the arithmetic working, not a broken fit."
        )
    # Both workover models quote an age as a fraction, and the fraction is meaningless
    # without the number it multiplies — so the reference is named and priced here.
    ref = (f"«Fraction is % of» = {TTF_REF_LABEL[cfg.ttf_reference]} = "
           f"**{cfg.ttf_ref_days():.0f}d**")
    if cfg.workover_mode == WORKOVER_DETERMINISTIC:
        st.caption(f"Planned pull at {cfg.pm_fraction:g} × TTF = **{cfg.pm_age_days():.0f} days** "
                   f"({ref}).")
    elif cfg.workover_mode == WORKOVER_STATISTICAL:
        st.caption(f"Workover Weibull: β={cfg.beta_wo:g}, η={cfg.wo_eta_fraction:g} × TTF = "
                   f"**{cfg.wo_eta_days():.0f} days** ({ref} — the same reference the "
                   f"deterministic pull age uses).")
    layers = layers_from_config(cfg)
    if layers:
        span, sd = theta_spread(layers)
        st.caption("Hazard layers on — " + " · ".join(
            f"**{s.key}** {s.n_bins} bins, θ {s.theta.min():.2f}–{s.theta.max():.2f}"
            + (" (per **well**, kept for life)" if s.per_well else " (redrawn per run)")
            for s in layers
        ) + f". Population spread **×{span:.1f}**, sd(log θ) = **{sd:.2f}**. "
            "The truth lines are the resulting **mixture**, not a single Weibull.")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_cached(experiment_id: str):
    return load_experiment(experiment_id)


def render_failures_only_table(result) -> None:
    """The failures-only β/η per snapshot, with what they are wrong against."""
    cfg = result.config
    table = failures_only_table(result)
    if table.empty:
        st.info("No snapshot has enough failures to fit yet.")
        return

    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "snapshot, yr": st.column_config.NumberColumn(format="%.4g"),
            "failures": st.column_config.NumberColumn(format="%d"),
            "censored": st.column_config.NumberColumn(
                format="%d", help="Runs dropped by the failures-only fit — workovers and "
                                  "pumps still turning at that snapshot."),
            "β failures-only": st.column_config.NumberColumn(format="%.3f"),
            "η failures-only, d": st.column_config.NumberColumn(format="%.0f"),
            "β censoring-aware": st.column_config.NumberColumn(format="%.3f"),
            "η censoring-aware, d": st.column_config.NumberColumn(format="%.0f"),
            "η shortfall": st.column_config.NumberColumn(
                format="%.1f%%", help="How far the failures-only η sits below the "
                                      "censoring-aware one at the same snapshot."),
        },
    )
    # the last snapshot with a fit on both arms — a tiny run can leave the final
    # one unfitted, and "nan% below" is worse than saying nothing
    fitted = table.dropna(subset=["η shortfall"])
    if cfg.n_modes() == 1:
        truth = (f"true **β = {cfg.beta_fail:g}, η = {cfg.eta_fail_days:g}d**"
                 if not layers_from_config(cfg) else
                 f"baseline **β = {cfg.beta_fail:g}, η = {cfg.eta_fail_days:g}d** before the "
                 "hazard layers redistribute it")
    else:
        truth = ("no single truth — competing modes are on, so see the **Failure modes** tab; "
                 f"the combined law reaches 1/e at **{eta_equivalent_days(cfg):.0f}d**")
    note = ""
    if not fitted.empty:
        last = fitted.iloc[-1]
        note = (f"The failures-only η is **{abs(last['η shortfall']):.0f}% below** the "
                f"censoring-aware one even at year {last['snapshot, yr']:g}, with "
                f"{int(last['censored']):,} runs still censored and therefore missing from it. ")
    st.caption(
        f"Median across the {cfg.n_seeds} seeds, at each KM snapshot. Against {truth}.  \n"
        + note +
        "Read down the column, not across one row: the failures-only numbers **move with the "
        "snapshot** because they track how much of the field is still alive, not the failure "
        "law — which is exactly why they cannot be quoted."
    )


def render_modes_tab(result) -> None:
    """Per-mode cause-specific recovery vs the pooled fit that mixes them."""
    cfg = result.config
    modes = cfg.modes()
    agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
    if not mode_columns(result.fit_table):
        st.info("This experiment was saved before per-mode fits existed — re-run it to get them.")
        return
    last = agg.iloc[-1]

    def num(col: str) -> float:
        """One aggregate value as a float, with a missing column reading as NaN."""
        return float(np.nan_to_num(last.get(col, np.nan), nan=np.nan))

    eq, solo = eta_equivalent_days(cfg), solo_means(cfg)
    cols = st.columns(len(modes) + 2)
    counts = np.array([num(f"n_fail_m{m}") for m in range(len(modes))])
    total = float(np.nansum(counts))
    for m, (name, beta, eta) in enumerate(modes):
        share = counts[m] / total if total > 0 else float("nan")
        b, e = num(f"beta_m{m}_median"), num(f"eta_m{m}_median")
        cols[m].metric(
            f"{name} — β / η", f"{b:.2f} / {e:,.0f}d" if np.isfinite(b) else "—",
            delta=f"{share:.0%} of failures" if np.isfinite(share) else "no failures yet",
            delta_color="off",
            help=f"Simulated from β = {beta:g}, η = {eta:g}d — that is this mode **alone**, "
                 f"a mean of {solo[m]:,.0f}d if nothing else could end the run. Fitted "
                 f"cause-specifically: this mode's failures are events, every other mode's "
                 f"failure is a censoring.",
        )
    cols[-2].metric("Combined truth — η-equiv", f"{eq:,.0f}d",
                    delta=f"below every mode's η", delta_color="off",
                    help="Where the combined law reaches S(t) = 1/e — the same thing η means "
                         "for a single Weibull. Hazards add, so this is shorter than the "
                         "shortest single mode: the pump only has to lose once.")
    cols[-1].metric("Pooled fit — β / η",
                    f"{last['beta_cens_median']:.2f} / {last['eta_cens_median']:,.0f}d",
                    help="The same runs fitted as one failure law. It tracks the combined "
                         "η-equivalent next to it, not either mode — it is the law of the "
                         "*minimum*, which is nobody's.")

    st.plotly_chart(modes_figure(result), use_container_width=True)
    st.caption(
        "Each mode is fitted **cause-specifically** — its own failures are events, the other "
        "modes' failures are just more censoring — and lands on the dotted β/η it was simulated "
        "from. The grey dashed line is the same runs pooled into one law: it sits on **no mode's "
        "truth**, because the minimum of several Weibulls is not Weibull. That is the whole "
        "argument for splitting a fleet's failures by node before fitting a shape to them."
    )

    st.plotly_chart(mode_survival_figure(result), use_container_width=True)
    st.caption(
        "Left: each mode's S(t) if it were the only risk, against the combined curve the field "
        "actually shows — always below every single mode, since the pump only has to lose once. "
        "**This is why the pooled η sits under both modes' η** and is not a bug: cumulative "
        "hazards add, `S = Π exp(−(t/η_m)^β_m)`, so the combined curve crosses 1/e at "
        f"**{eq:,.0f}d** while the individual modes only get there at "
        f"{' / '.join(f'{e:,.0f}' for _, _, e in modes)}d. For pure exponentials it is the "
        "parallel-resistor rule, `1/λ = 1/(λ₁+λ₂)`.  \n"
        "Right: which mode wins as a function of age. A low-β mode dominates the early failures "
        "and a high-β one takes over later, so **the mix of causes shifts with age** — and the "
        "pooled shape you fit depends on how much of each age band your data covers."
    )

    show_cols = ["snap_year", "n_fail"] + [
        c for m in range(len(modes)) for c in
        (f"n_fail_m{m}", f"beta_m{m}_median", f"eta_m{m}_median") if c in agg.columns]
    rename = {f"n_fail_m{m}": f"{n} fails" for m, (n, _, _) in enumerate(modes)}
    rename |= {f"beta_m{m}_median": f"{n} β" for m, (n, _, _) in enumerate(modes)}
    rename |= {f"eta_m{m}_median": f"{n} η" for m, (n, _, _) in enumerate(modes)}
    st.dataframe(agg[show_cols].rename(columns=rename).round(2),
                 use_container_width=True, hide_index=True)


def render_bins_tab(result) -> None:
    """Observed failures per covariate bin: mean TTF vs Ql / frequency."""
    cfg = result.config
    layers = layers_from_config(cfg)
    if result.bin_table is None or not layers:
        st.info(
            "No hazard layer in this experiment. Enable the **individual-well θ variation**, "
            "**Ql** and/or **frequency** layer under *Hazard layers (θ)* in the sidebar and "
            "re-run — each pump (or well) then draws a bin, and this tab shows the life the "
            "bins actually produced."
        )
        return

    names = {LAYER_WELL: "Well frailty", LAYER_QL: "Ql (rate)", LAYER_FREQ: "Frequency"}
    c1, c2, c3 = st.columns([2, 3, 2])
    keys = [s.key for s in layers]
    layer_key = c1.radio("Layer", keys, format_func=lambda k: names.get(k, k),
                         horizontal=True, key="bins_layer")
    years = sorted(result.bin_table["snap_year"].unique())
    snap_year = c2.select_slider("Snapshot (field age, years)", years, value=years[-1],
                                 key="bins_year")
    all_seeds = sorted(int(s) for s in result.bin_table["seed"].unique())
    seed_pick = c3.selectbox(
        "Population", [None, *all_seeds], key="bins_seed",
        format_func=lambda s: f"all {len(all_seeds)} seeds (pooled)" if s is None else f"seed {s}",
        help="Pooled sums the counts and takes the median of the per-seed life summaries. "
             "A single seed is one field realization — the band closes and you see the "
             "sampling noise a real field of this size actually has.",
    )

    agg = aggregate_bins(result.bin_table, layer_key, float(snap_year), seed_pick)
    if agg.empty:
        st.warning("No bin data at that snapshot.")
        return

    spec = {s.key: s for s in layers}[layer_key]
    obs = agg["obs_ttf_fail_median"]
    # a single seed at a young snapshot can leave bins with no failures at all
    spread_txt = (f"{obs.max() / obs.min():.2f}×" if obs.notna().all() and obs.min() > 0
                  else "not yet measurable")
    n_fail, n_runs = agg["n_fail"].to_numpy(), agg["n_runs"].to_numpy()
    n_seeds = int(agg["n_seeds"].max())
    censored = 1.0 - n_fail.sum() / n_runs.sum()
    pooled = seed_pick is None
    pop = f"{n_seeds} seeds pooled" if pooled else f"seed {seed_pick}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Failures by year {snap_year:g}", f"{int(n_fail.sum()):,}",
              help=(f"Pooled over {n_seeds} seeds — {n_fail.sum() / n_seeds:,.0f} per field "
                    f"realization. Grows with the snapshot." if pooled else
                    f"One field realization (seed {seed_pick}). Grows with the snapshot."))
    m2.metric("Failures per bin", f"{int(n_fail.min()):,}–{int(n_fail.max()):,}")
    m3.metric("Runs started", f"{int(n_runs.sum()):,}", help=f"{censored:.0%} still censored "
              f"(running or pulled for a workover) at this snapshot.")
    layer_spread = float(spec.theta.max() / spec.theta.min())
    rmst0 = life_given_theta(cfg, 1.0, cfg.rmst_tau_days)["rmst"]
    bin_rmst = _bin_life(cfg, spec.theta, "rmst")
    rmst_spread = float(bin_rmst.max() / bin_rmst.min())
    m4.metric("θ spread → RMST spread", f"{layer_spread:.2f}× → {rmst_spread:.2f}×",
              help=f"θ multiplies the hazard; the RMST(0, {cfg.rmst_tau_days:g}d) it buys does "
                   f"not multiply with it. The τ restriction and β compress the response, so "
                   f"the life you can actually book is the smaller number.")

    st.plotly_chart(bins_figure(result, layer_key, float(snap_year), seed_pick),
                    use_container_width=True)
    drawn = ("Each **well** draws its bin uniformly and keeps it for life, so the bins hold "
             "equal numbers of wells but the bad ones burn through more pumps"
             if spec.per_well else
             "Each pump draws its bin uniformly, so bin populations are equal by construction")
    st.caption(
        f"Grey bars = **failures each bin rests on at year {snap_year:g}** ({pop}, right axis); "
        f"slide the snapshot to watch the evidence accumulate and the estimates settle. {drawn} "
        f"— any difference in life is the θ layer, not the mix. Solid = what the field shows; dotted × = the truth "
        f"that bin was simulated from. The **naive mean TTF sits below its truth in every bin** "
        f"(censoring), so read the *shape*, not the level — observed life spreads "
        f"{spread_txt} across the bins against a θ spread of {layer_spread:.2f}×.  \n"
        f"Right panel, second axis: what that θ is **worth in life**. A {layer_spread:.2f}× hazard "
        f"spread buys only a {rmst_spread:.2f}× spread in RMST(0, {cfg.rmst_tau_days:g}d) — "
        f"**θ multiplies, RMST multipliers do not**. Both axes are centred on 1.0 so the dotted "
        f"line reads correctly on either scale; the flatter curve is the RMST response, "
        f"compressed by the τ restriction and by β."
        + ("" if pooled else
           "  \n**Single seed:** the p10–p90 band is gone because there is nothing to spread "
           "over — every wiggle here is one field's sampling noise, and it is the honest picture "
           "of what a field this size can actually resolve.")
    )

    show = agg.rename(columns={"center": spec.label, "theta": "θ"})[
        [spec.label, "θ", "n_runs", "n_fail", "obs_ttf_fail_median", "obs_ttf_median_median",
         "km_rmst_median", "km_median_median"]
    ].rename(columns={"obs_ttf_fail_median": "obs. mean TTF (d)",
                      "obs_ttf_median_median": "obs. median TTF (d)",
                      "km_rmst_median": f"KM RMST(0,{cfg.rmst_tau_days:g}d)",
                      "km_median_median": "KM median (d)"})
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download per-seed bin table (CSV)",
        result.bin_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{result.experiment_id}_bin_table.csv",
        mime="text/csv",
    )


def render(result) -> None:
    st.subheader(result.label)
    st.caption(f"`{result.experiment_id}` · created {result.created}")
    config_summary(result)

    names = [":material/show_chart: KM snapshots", ":material/trending_up: Parameter trends",
             ":material/straighten: Life summaries", ":material/water_drop: Rates &amp; oil loss",
             ":material/bar_chart: Observed failures", ":material/table: Data"]
    has_modes = result.config.n_modes() > 1
    if has_modes:
        names.insert(2, ":material/cable: Failure modes")
    tabs = st.tabs(names)
    tab_km, tab_trend = tabs[0], tabs[1]
    tab_modes = tabs[2] if has_modes else None
    tab_life, tab_rates, tab_bins, tab_table = tabs[-4:]
    with tab_km:
        c1, c2 = st.columns(2)
        stored = has_failures_only_curves(result)
        fo_km = c1.checkbox(
            "Failures-only KM", value=False, key="km_fo_curve", disabled=not stored,
            help="Same snapshots with every censored run dropped. Saved before this "
                 "existed — re-run to get it." if not stored else
                 "Drop every censored run and plot the observed failures as if they were "
                 "the whole sample. Dotted, same colour as its snapshot.",
        )
        fo_fit = c2.checkbox(
            "Failures-only Weibull fit", value=False, key="km_fo_fit",
            help="The Weibull those failures-only durations fit to (median β/η across "
                 "seeds) — the smooth law you would conclude from the biased sample.",
        )
        st.plotly_chart(km_figure(result, fo_km=fo_km and stored, fo_fit=fo_fit),
                        use_container_width=True)
        if fo_fit:
            render_failures_only_table(result)
        base = ("KM pooled across all seeds at each snapshot. Young field → the curve only "
                "spans the pump ages that exist yet (limited support), but is unbiased where "
                "defined.")
        if fo_km or fo_fit:
            base += (
                "  \nThe **failures-only** overlays are the same runs with every censored "
                "record thrown away. They fall away from the solid curve because the runs "
                "still alive — the long-lived ones — never enter the sample, so the fleet "
                "reads as shorter-lived than it is. The gap is widest on a young field and "
                "narrows only as the censored runs eventually fail. Each year's overlay "
                "shares its snapshot's legend entry, so clicking a year toggles the pair."
            )
        else:
            base += ("  \nTick **failures-only** above to overlay what the same snapshots "
                     "look like once censored runs are dropped.")
        st.caption(base)
    with tab_trend:
        x_key = st.radio(
            "X-axis", list(X_CHOICES), index=0, horizontal=True, key="trend_x",
            help="How to index the evolution: field age, or cumulative pump starts / failures / "
                 "workovers (planned pulls). All are cumulative up to each snapshot.",
        )
        col_name = X_CHOICES[x_key][0]
        if aggregate_by_snapshot(result.fit_table)[col_name].nunique() < 2:
            st.info(f"`{x_key}` is constant for this experiment (e.g. no workovers) — pick another x-axis.")
        st.plotly_chart(trends_figure(result, x_key), use_container_width=True)
        st.caption(
            "**Censored (cause-specific)** → truth. **Failures-only** drops all censored runs and "
            "underestimates η. **All-pulls** counts every workover as a failure — the all-cause "
            "pull-time (МРП) Weibull, always shorter-lived than the failure law (η below truth, "
            "apparent wear-out β>1). Both naive fits understate failure life; only the cause-specific "
            "fit recovers it."
            + ("" if not has_modes else
               "  \n**Competing modes are on, so there is no single truth line here** — the pooled "
               "β/η describe the *minimum* over the modes, a law no single piece of equipment "
               "follows. The **Failure modes** tab is where the ground truth lives.")
        )

    if tab_modes is not None:
        with tab_modes:
            render_modes_tab(result)

    with tab_life:
        st.plotly_chart(life_figure(result), use_container_width=True)
        st.caption(
            "Life summaries of the **failure** distribution (cause-specific KM: workovers & running "
            "censored). RMST(0,τ), MRL(0) and the KM median are censoring-corrected and converge to "
            "their true values; the **naive mean Σt/N (red)** stays biased low. Note KM summaries need "
            "tail support — under heavy planned-pull censoring (deterministic PM) the tail is unobserved, "
            "so MRL(0)/RMST degrade while the parametric fit still extrapolates."
        )

    with tab_rates:
        exp = expected_rates(result.config)
        cols = st.columns(4)
        cols[0].metric("Exp. failures / well·yr", f"{exp['fail_per_well_year']:.3f}")
        cols[1].metric("Exp. workovers / well·yr", f"{exp['workover_per_well_year']:.3f}")
        cols[2].metric("Exp. pulls / well·yr", f"{exp['pull_per_well_year']:.3f}")
        cols[3].metric("Exp. oil loss %", f"{exp['oil_loss_pct']:.2f}%")
        if result.config.workover_mode != WORKOVER_NONE:
            base = expected_rates(dataclasses.replace(result.config, workover_mode=WORKOVER_NONE))
            st.caption(
                f"Steady-state estimate from parameters (renewal-reward). **Programme effect vs "
                f"no-workover:** failures/well·yr {exp['fail_per_well_year'] - base['fail_per_well_year']:+.3f}, "
                f"oil loss {exp['oil_loss_pct'] - base['oil_loss_pct']:+.2f} pp."
            )
        else:
            st.caption("Steady-state estimate from parameters (renewal-reward theory). Dotted lines on the charts.")
        st.plotly_chart(rates_figure(result), use_container_width=True)
        if result.fit_table_nowo is not None:
            st.caption(
                "**Grey dashed = same field with the workover programme off** (same seeds). "
                "With β=1 (random failures) the programme cannot lower the failure rate — it only "
                "adds pulls and downtime, so oil loss rises. Set β>1 (wear-out) to see workovers "
                "actually prevent failures."
            )
        else:
            cfg = result.config
            note = "No workover programme in this run." if cfg.workover_mode == "none" else \
                "Re-run with the *no-workover twin* checked to overlay the counterfactual."
            st.caption(
                f"Oil loss = share of well-days a well is not producing (downtime: "
                f"{cfg.downtime_fail_days:g}d after failure, {cfg.downtime_workover_days:g}d after workover). {note}"
            )

    with tab_bins:
        render_bins_tab(result)

    with tab_table:
        agg = aggregate_by_snapshot(result.fit_table).sort_values("snap_year")
        show_cols = ["snap_year", "running_pop", "n_runs", "n_fail", "n_workover",
                     "beta_cens_median", "eta_cens_median", "beta_fo_median", "eta_fo_median",
                     "beta_all_median", "eta_all_median",
                     "fail_rate_well_yr_median", "pull_rate_well_yr_median", "oil_loss_pct_median"]
        st.dataframe(agg[show_cols].round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "Download per-seed fit table (CSV)",
            result.fit_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{result.experiment_id}_fit_table.csv",
            mime="text/csv",
        )


def main() -> None:
    action, cfg, load_id, label, run_cf = sidebar_config()

    if action == "run" and cfg is not None:
        prog = st.progress(0.0, text="Simulating field realizations…")
        try:
            result = run_experiment(
                cfg, label=label or None, counterfactual=run_cf,
                progress=lambda f: prog.progress(min(f, 1.0)),
            )
            save_experiment(result)
            _load_cached.clear()
        finally:
            prog.empty()
        st.session_state["active_id"] = result.experiment_id
        st.success(f"Saved as `{result.experiment_id}`.")
        render(result)
        return

    if action == "load" and load_id is not None:
        render(_load_cached(load_id))
        return

    if "active_id" in st.session_state:
        try:
            render(_load_cached(st.session_state["active_id"]))
            return
        except Exception:
            pass

    st.title("Growing ESP field — survival under censoring")
    st.markdown(
        "Set the field, failure law, (optional) workover process and (optional) hazard layers in "
        "the sidebar, then **Run simulation** — or switch the source to **Load saved** to reopen "
        "a past run. **Defaults…** at the top of the sidebar changes the value every control "
        "opens at.\n\n"
        "The experiment shows how KM survival and the fitted Weibull (β, η) behave as the field "
        "grows and pumps accumulate, comparing the **censoring-aware** fit against the "
        "**failures-only** fit. Two switches make the fitted law deliberately misleading, each "
        "in its own way:\n\n"
        "- **Competing failure modes** (in *Failure Weibull*) races one Weibull per piece of "
        "equipment — pump, cable, … — and ends the run at the first of them. Fitting the pooled "
        "failures returns the shape of the *minimum*, which belongs to no single node; the "
        "cause-specific fit per mode recovers each node's own β/η.\n"
        "- **Individual-well θ variation** (in *Hazard layers*) gives every well a permanent "
        "frailty multiplier spanning a max/min ratio you set. Every pump can wear out (β > 1) "
        "and the pooled fit will still hand back β < 1 once the spread is wide enough — the "
        "heterogeneity, not the physics, bending the shape.\n\n"
        "Every pump-life quantity — η, τ, RMST, MRL(0), median, mean TTF — is in **days**; the "
        "field clock (horizon, ramp, snapshots) stays in years. The **η calculator** in the "
        "sidebar goes the other way: give it a target RMST / MRL / median and it returns the η "
        "that produces it."
    )


if __name__ == "__main__":
    main()
