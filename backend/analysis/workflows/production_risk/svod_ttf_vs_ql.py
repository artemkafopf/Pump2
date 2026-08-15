"""Свод TTF (ННО) vs liquid rate Ql — Ya (clean) vs Vt (dirty), failures vs all pulls.

The companion to :mod:`svod_nno_decomposition`, which covers the *setpoint* axes
(Kpod, Delta, freq) and controls **for** Ql.  Here Ql is the x-axis, so it cannot
also be the control; the panels are:

  * **MARGINAL** — mean ННО per Ql decile, the raw rate→life relation.
  * **RESIDUAL vs Qnom** — the same after removing ``log Qnom`` (pump size) inside
    the group.  This is the sharp question left open by the Ya v2.1 result that
    *nameplate Qnom subsumes Ql*: if Ql is only a proxy for how big a pump the
    field engineer chose, this panel flattens.

Four groups, deliberately: **Ya** is the clean signal (thick, single H₂S class,
where the Ql layer was estimated) and **Vt** is the dirty one — pooled plus the
v3.2 well-level sour relabel (``Vt_nonsour`` / ``Vt_sour``), because Vt's Ql
response is known to be ~2× stronger on nonsour than on the saturating sour arm.

Two row denominators, both emitted:

  * ``failures``   — genuine failures only (ГТМ/ППР and no-signal pulls dropped).
  * ``all_closed`` — every closed run, i.e. **all pulls**.

The comparison between them is the point, not an implementation detail: ННО exists
only for pulled pumps, so *both* variants condition on a pull; ``failures`` narrows
that to a collider (see the freq-peak artifact note).  A shape that appears only in
``failures`` is a selection effect, not physics.

Each figure is emitted twice — once as binned means alone, and once with **every
run scattered underneath**, coloured by outcome (failure vs workover/other pull).
The scatter is what shows the spread the decile means hide: these clouds are wide,
and a 40-day difference between bins sits inside a 400-day scatter.

Ql is plotted on a **linear axis** (user decision 2026-07-27) but still *fitted* in
``log Ql``: the Ql layer everywhere else in this workflow is log-rational / per-e-fold,
so the reported slope stays the same object as an HR per e-fold there.  The fitted
curve therefore renders as a curve rather than a straight line — that is the log model
seen on a linear axis, not a quadratic in Ql.

Every binned point is labelled with its **TTF multiplier** — the point's value over the
fitted life at the group's own median Ql, so ``1.00×`` is "a median-rate well of this
group" and ``0.60×`` is "40% shorter than that".

Two rate definitions, each getting the full set of figures (:data:`RATES`):

  * ``ql_svod``  — the «Дебит жидк.» cell, one representative rate per run.
  * ``ql_start`` — mean ``qliq`` over the run's **first operating month**, taken from
    the daily warehouse.  This is the causally clean one: it is fixed before the
    outcome is known, whereas any run-length average is partly a read of the failure
    itself (a pump that died on day 20 has its rate averaged over 20 days whose last
    few are already degradation).  The daily table starts 2018-01-01, so runs
    installed before it get **no** ``ql_start`` rather than a mid-life rate wearing a
    start-of-run label — see :data:`MAX_START_GAP_DAYS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import t0_covariates  # noqa: E402
from analysis.workflows.production_risk.svod_nno_decomposition import (  # noqa: E402
    _bin_mean,
    load_panel,
)

SLUG = "production_risk_svod_ttf_vs_ql"

#: The Ql the composed/v3.2 models are referenced at — drawn as the vertical guide.
QL_REF = 250.0

#: (column, axis label, short title) per rate definition.
#:
#: ``ql_svod`` is the «Дебит жидк.» cell — one representative rate per run, and it is
#: not clear at which point of the run it was read.  ``ql_start`` is the mean over the
#: run's first operating month from the daily warehouse, which is *causally clean*: it
#: is fixed before anything about the outcome is known.  A run mean is not — a pump
#: that failed on day 20 has its "rate" averaged over 20 days whose last few are
#: already degradation (see :mod:`t0_covariates`), so a rate→life slope fitted on it
#: partly reads the failure backwards.
RATES: dict[str, tuple[str, str, str]] = {
    "ql_svod": ("ql", "Ql, м³/сут (Свод snapshot)", "Ql (Свод)"),
    "ql_start": ("ql_start", "Ql_start, м³/сут (first operating month)",
                 "Ql_start (1st month)"),
}

#: Ql below this is not a rate, it is a well that barely flowed; the Свод sheet
#: carries a handful and they anchor the first decile on their own.  Dropped rows
#: are counted into ``group_coverage.csv``, never silently.
QL_BOUNDS = (5.0, 2000.0)

#: (label, predicate) — evaluated against the loaded panel.  Ya first: it is the
#: reference the Vt rows are read against.
GROUPS: tuple[tuple[str, str], ...] = (
    ("Ya", "field == 'Ya'"),
    ("Vt (all)", "field == 'Vt'"),
    ("Vt_nonsour", "field == 'Vt' and h2s_class == 'nonsour'"),
    ("Vt_sour", "field == 'Vt' and h2s_class == 'sour'"),
)

VARIANT_ROWS = {"failures": "genuine failures only", "all_closed": "all pulls"}

#: Scatter colours by outcome.  Only ``all_closed`` shows both.
EVENT_STYLE = {
    1: dict(color="#C0392B", marker="o", label="failure"),
    0: dict(color="#7F8C8D", marker="^", label="workover / other pull"),
}


#: The daily warehouse (``proc__daily_merged``) begins 2018-01-01.  For a run installed
#: before that, "the first 30 daily rows after install" are rows from the middle of the
#: run — a mid-life rate wearing a start-of-run label.  A window whose first daily row
#: is more than this many days after install is therefore refused, not used.
MAX_START_GAP_DAYS = 45

#: Operating days averaged into ``ql_start``.  Matches ``t0_covariates``' default.
QL_START_WINDOW_OP_DAYS = 30


def attach_ql_start(
    panel: pd.DataFrame,
    window_op_days: int = QL_START_WINDOW_OP_DAYS,
    max_start_gap_days: int = MAX_START_GAP_DAYS,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``ql_start`` — mean ``qliq`` over the run's first ``window_op_days`` op-days.

    An operating day is one the pump actually lifted fluid (``qliq > 0``), the same
    definition :mod:`t0_covariates` uses, so a month of standing idle does not count
    as the first month.  The window is bounded by the run's own ``[install, end]``:
    wells are re-completed, and without the bound a later run's dailies would leak
    into an earlier run's start.

    Returns ``(panel_with_ql_start, coverage)``.  Runs without a usable window keep
    ``NaN`` — reported in the coverage frame, never dropped here, because the runs
    that lose their window are systematically the oldest ones.
    """
    out = panel.copy()
    out["well_key"] = out["code"].astype(str).str.casefold()
    dailies = t0_covariates.load_dailies(
        sorted(out["well_key"].unique().tolist()), db_path=db_path)
    dailies = dailies[dailies["qliq"].notna() & (dailies["qliq"] > 0)]
    by_well = {k: g.sort_values("dt") for k, g in dailies.groupby("well_key", sort=False)}

    ql_start, n_days, gaps, reason = [], [], [], []

    def _skip(why: str, gap: float = np.nan) -> None:
        ql_start.append(np.nan); n_days.append(0); gaps.append(gap); reason.append(why)

    for _, run in out.iterrows():
        g = by_well.get(run["well_key"])
        if g is None:
            _skip("no daily rows for well")
            continue
        if pd.isna(run["install"]):
            _skip("no install date")
            continue
        install = pd.Timestamp(run["install"])
        stop = pd.Timestamp(run["end"]) if pd.notna(run["end"]) else pd.Timestamp.max
        win = g[(g["dt"] >= install) & (g["dt"] <= stop)]
        if win.empty:
            _skip("no dailies inside run")
            continue
        # The gap is kept as a NUMBER on its own column, never folded into the status
        # string: a per-run gap in the label turns the coverage rollup into one row
        # per distinct gap and hides the counts it exists to show.
        gap = float((win["dt"].iloc[0] - install).days)
        if gap > max_start_gap_days:
            _skip(f"window starts >{max_start_gap_days}d after install", gap)
            continue
        win = win.head(window_op_days)
        ql_start.append(float(win["qliq"].mean()))
        n_days.append(int(len(win)))
        gaps.append(gap)
        reason.append("full window" if len(win) >= window_op_days else "partial window")

    out["ql_start"] = ql_start
    out["ql_start_n_days"] = n_days
    out["ql_start_gap_days"] = gaps
    out["ql_start_status"] = reason
    coverage = (out.groupby(["field", "ql_start_status"])
                .agg(n=("ql_start_status", "size"),
                     median_gap_days=("ql_start_gap_days", "median"),
                     max_gap_days=("ql_start_gap_days", "max"))
                .reset_index()
                .sort_values(["field", "n"], ascending=[True, False]))
    return out, coverage


@dataclass(frozen=True)
class QlFit:
    """OLS ``y ~ 1 + u + u²`` with ``u = log Ql − mean(log Ql)``.

    ``beta1`` is the slope **per e-fold of Ql** in the response's own unit (days),
    read at the sample's geometric-mean Ql because u is centred.  ``beta2`` is
    curvature; it is reported so a reader can see whether the straight line is a
    fair summary, not because a peak in Ql is expected.
    """

    n: int
    beta0: float
    beta1: float
    se1: float
    p1: float
    beta2: float
    se2: float
    p2: float
    ql_gm: float
    grid_x: np.ndarray
    grid_y: np.ndarray
    #: What the x-axis actually is, so the legend cannot claim "per e-fold Ql" on a
    #: panel whose x is nameplate Qnom (see :mod:`svod_ttf_by_design`).
    x_name: str = "Ql"

    @property
    def label(self) -> str:
        return (f"{self.beta1:+.0f} d per e-fold {self.x_name} (p={self.p1:.3f})"
                f" | curv p={self.p2:.2f}, n={self.n}")

    @property
    def p_headline(self) -> float:
        """The p-value a plot should judge this fit by — here the slope's.

        Counterpart of ``QuadFit.p_headline``, so shared plotting can decide
        resolved-vs-tentative without knowing which fit class it was handed.
        """
        return self.p1

    def predict(self, rate: float) -> float:
        """Fitted response at ``rate``.  ``beta0`` is the value at the geometric mean."""
        u = np.log(rate) - np.log(self.ql_gm)
        return float(self.beta0 + self.beta1 * u + self.beta2 * u**2)


def _fit(ql: pd.Series, y: pd.Series, x_name: str = "Ql") -> QlFit | None:
    from scipy import stats

    x = np.log(np.asarray(ql, dtype=float))
    yv = np.asarray(y, dtype=float)
    n = len(x)
    if n < 30:
        return None
    u = x - x.mean()
    design = np.column_stack([np.ones(n), u, u**2])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    sigma2 = resid @ resid / (n - 3)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    p = 2 * stats.t.sf(np.abs(beta[1:] / se[1:]), n - 3)

    grid = np.linspace(x.min(), x.max(), 200)
    gu = grid - x.mean()
    return QlFit(
        n=n,
        beta0=float(beta[0]),
        beta1=float(beta[1]), se1=float(se[1]), p1=float(p[0]),
        beta2=float(beta[2]), se2=float(se[2]), p2=float(p[1]),
        ql_gm=float(np.exp(x.mean())),
        grid_x=np.exp(grid),
        grid_y=beta[0] + beta[1] * gu + beta[2] * gu**2,
        x_name=x_name,
    )


# ── the fields' own fitted Ql layers ──────────────────────────────────────────
#
# Removing ``log Ql`` by OLS asks "is the relation log-linear".  That is not the
# question the workflow actually has open: Ya and Vt each already ship a *fitted* Ql
# layer, and the sharper question is whether THOSE explain this axis.  So the third
# panel offsets each run by its own field model:
#
#   Ya  → ``ya_k1k2_hybrid`` **v2** (rate arm = Ql), θ_Ql monotone polyline.
#   Vt  → ``vt_composed_model`` (the deployed v3.2 closed form), θ_Ql a power law
#         **per sour stratum** — nonsour exponent 0.358 vs sour 0.160, the ~2× gap.
#
# Both are hazard multipliers, and RMST is a saturating functional of the hazard, so a
# θ is converted to a LIFE multiplier through each model's own composition rule — never
# by multiplying RMST multipliers (see the models' docstrings).

#: Number of grid points used to interpolate θ → life multiplier.  The conversion runs
#: a survival integral per distinct θ, which is far too slow per-run; the map is smooth
#: and monotone, so a grid plus linear interpolation is exact to well under a day.
LIFE_MULT_GRID = 160


def _life_mult_interp(thetas: np.ndarray, mult_of_theta) -> np.ndarray:
    """Evaluate a costly θ→life-multiplier map on a grid and interpolate onto ``thetas``."""
    finite = thetas[np.isfinite(thetas)]
    if not finite.size:
        return np.full(thetas.shape, np.nan)
    lo, hi = float(finite.min()), float(finite.max())
    if np.isclose(lo, hi):
        return np.where(np.isfinite(thetas), mult_of_theta(lo), np.nan)
    grid = np.linspace(lo, hi, LIFE_MULT_GRID)
    return np.interp(thetas, grid, np.array([mult_of_theta(t) for t in grid]))


@lru_cache(maxsize=1)
def _ya_model():
    """Ya v2, fitted once per process.

    ``n_boot=0`` skips the bootstrap — this panel needs the fitted θ curve, not its
    confidence band — and ``write=False`` keeps a plotting call out of the model's own
    results directory.
    """
    from analysis.workflows.production_risk import ya_k1k2_hybrid as YA
    return YA.run(rate="Ql", n_boot=0, write=False).model


def ya_life_mult(ql: np.ndarray) -> np.ndarray:
    """Ya v2 RMST(0,730) multiplier vs its own reference, from θ_Ql alone."""
    m = _ya_model()
    theta = np.asarray(m.theta_at("Ql", np.asarray(ql, float)), dtype=float)
    return _life_mult_interp(theta, lambda t: m.life_mult(float(t)))


def vt_life_mult(ql: np.ndarray, stratum: str) -> np.ndarray:
    """Vt v3.2 (composed closed form) RMST(0,730) multiplier from θ_Ql alone."""
    from analysis.workflows.production_risk import vt_composed_model as VC
    b = VC.BASELINE[stratum]
    beta0, eta0 = b["beta0"], b["eta0"]
    r0 = VC.rmst(eta0, beta0)
    theta = np.asarray(VC.theta_ql(stratum, np.asarray(ql, float)), dtype=float)
    return _life_mult_interp(
        theta, lambda t: VC.rmst(eta0 * t ** (-1.0 / beta0), beta0) / r0)


#: Field → (human label, life-multiplier callable taking (ql, stratum)).
FIELD_MODELS = {
    "Ya": ("Ya v2 θ_Ql", lambda ql, stratum: ya_life_mult(ql)),
    "Vt": ("Vt v3.2 θ_Ql", vt_life_mult),
}


def model_life_multiplier(g: pd.DataFrame, rate_col: str) -> pd.Series:
    """Each run's fitted life multiplier under its own field model, by sour stratum."""
    field = str(g["field"].iloc[0])
    _, fn = FIELD_MODELS[field]
    out = pd.Series(np.nan, index=g.index, dtype=float)
    # Vt's layer is stratum-specific, so a mixed group is evaluated stratum by stratum
    # rather than pooled — pooling would apply the nonsour exponent to sour runs.
    for stratum, part in g.groupby("h2s_class"):
        out.loc[part.index] = fn(part[rate_col].to_numpy(float), str(stratum))
    return out


def _model_residual(g: pd.DataFrame, rate_col: str) -> pd.Series:
    """ННО minus the field model's own Ql shape, with only the LEVEL taken from here.

    The model supplies the **shape** of the rate→life relation; the single scale ``L0``
    is least-squares fitted on this panel.  That is deliberate: these rows are Свод
    closed runs measured on ННО, which is not the population or the clock either model
    was estimated on, so forcing the model's absolute life level would mix a scale
    mismatch into what is meant to be a test of shape.  A flat residual here means the
    field's fitted Ql layer already accounts for everything this axis shows.
    """
    y = g["nno"].to_numpy(float)
    m = model_life_multiplier(g, rate_col).to_numpy(float)
    ok = np.isfinite(m) & np.isfinite(y)
    if not ok.any():
        return pd.Series(np.nan, index=g.index, dtype=float)
    level = float((y[ok] * m[ok]).sum() / (m[ok] ** 2).sum())
    return pd.Series(y - level * m, index=g.index, name="model_resid")


def _qnom_residual(g: pd.DataFrame) -> pd.Series:
    """ННО minus its ``log Qnom`` (pump size) prediction, fitted **inside** the group.

    Within-group is the right level here: the groups are the strata the Vt hybrid
    already separates, and a pooled size fit would smuggle the Ya/Vt life gap back
    in through their different pump mixes.
    """
    y = g["nno"].to_numpy(float)
    design = np.column_stack([np.ones(len(g)), g["log_qnom"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ beta, index=g.index, name="resid")


@dataclass(frozen=True)
class GroupPanel:
    """One group's two binned curves, their fits, and the rows behind them."""

    name: str
    rate: str
    rows: pd.DataFrame
    marginal: pd.DataFrame
    residual: pd.DataFrame
    model_residual: pd.DataFrame
    marginal_fit: QlFit | None
    residual_fit: QlFit | None
    model_residual_fit: QlFit | None
    #: Human label of the field model behind ``model_residual``.
    model_label: str
    #: The group's median rate, and the fitted ННО there.  Together they define the
    #: **TTF multiplier**: every panel's secondary y-axis is the same days already
    #: plotted, divided by ``ref_ttf`` — so 1.0 is "what a median-rate well of this
    #: group achieves" and 0.5 is "half that".  Per group, because a multiplier
    #: against a fleet-wide constant would just re-state the Ya/Vt life gap.
    rate_median: float
    ref_ttf: float


def build(panel: pd.DataFrame, bins: int = 10, min_n: int = 40, rate: str = "ql_svod"
          ) -> tuple[dict[str, GroupPanel], pd.DataFrame]:
    """Per-group marginal and Qnom-residual curves vs the chosen rate, plus coverage.

    ``rate`` picks the x-axis from :data:`RATES` — the Свод snapshot or the
    first-operating-month mean.  Rows with no rate are dropped **per rate**, so the
    ``ql_start`` panels sit on a smaller and younger population than the ``ql_svod``
    ones; the coverage frame carries both counts side by side rather than leaving the
    reader to assume the two figures cover the same runs.
    """
    col, _, _ = RATES[rate]
    out: dict[str, GroupPanel] = {}
    coverage = []
    for name, expr in GROUPS:
        present = panel.query(expr)
        has_rate = present[present[col].notna()]
        g = has_rate[has_rate[col].between(*QL_BOUNDS)]
        coverage.append({
            "group": name,
            "rate": rate,
            "n_present": len(present),
            "n_with_rate": len(has_rate),
            "n_in_bounds": len(g),
            "n_dropped_missing_rate": len(present) - len(has_rate),
            "n_dropped_out_of_bounds": len(has_rate) - len(g),
            "n_failures": int((g["event"] == 1).sum()),
            "n_other_pulls": int((g["event"] != 1).sum()),
            "rate_min": g[col].min() if len(g) else np.nan,
            "rate_max": g[col].max() if len(g) else np.nan,
            "mean_nno": g["nno"].mean() if len(g) else np.nan,
            "median_nno": g["nno"].median() if len(g) else np.nan,
        })
        if len(g) < max(min_n, bins * 3):
            continue
        resid = _qnom_residual(g)
        model_resid = _model_residual(g, col)
        # Bin on the rate itself so the decile edges are readable numbers; the fit runs
        # on log-rate on the unbinned rows, so the slope does not depend on them.
        b = int(np.clip(len(g) // 30, 4, bins))
        marginal_fit = _fit(g[col], g["nno"])
        rate_median = float(g[col].median())
        # The multiplier's denominator is the FITTED life at the median rate, not the
        # observed mean of the median decile: the fit uses every run, so the reference
        # does not jump with wherever a decile edge happened to fall.
        ref_ttf = marginal_fit.predict(rate_median) if marginal_fit else float("nan")
        out[name] = GroupPanel(
            name=name,
            rate=rate,
            rows=g.assign(resid=resid, model_resid=model_resid,
                          model_life_mult=model_life_multiplier(g, col)),
            marginal=_bin_mean(g[col], g["nno"], b),
            residual=_bin_mean(g[col], resid, b),
            model_residual=_bin_mean(g[col], model_resid, b),
            marginal_fit=marginal_fit,
            residual_fit=_fit(g[col], resid),
            model_residual_fit=_fit(g[col], model_resid),
            model_label=FIELD_MODELS[str(g["field"].iloc[0])][0],
            rate_median=rate_median,
            ref_ttf=ref_ttf,
        )
    return out, pd.DataFrame(coverage)


def fit_table(groups: dict[str, GroupPanel]) -> pd.DataFrame:
    rows = []
    for name, gp in groups.items():
        for panel_kind, f in (("marginal", gp.marginal_fit),
                              ("qnom_residual", gp.residual_fit),
                              ("field_model_residual", gp.model_residual_fit)):
            if f is None:
                continue
            rows.append({
                "group": name, "rate": gp.rate, "panel": panel_kind,
                "field_model": gp.model_label, "n": f.n,
                "days_per_efold": f.beta1, "se": f.se1, "p": f.p1,
                "curvature": f.beta2, "curvature_se": f.se2, "curvature_p": f.p2,
                "rate_geomean": f.ql_gm,
                # The multiplier axis' denominator, so a reader can convert any
                # plotted day-value by hand.
                "rate_median": gp.rate_median,
                "ref_ttf_at_rate_median": gp.ref_ttf,
            })
    return pd.DataFrame(rows)


# ── figure ────────────────────────────────────────────────────────────────────

RESOLVED_P = 0.05

#: How the individual runs are drawn under the binned curve.
OVERLAYS = ("none", "scatter", "heatmap")

#: Single-hue light→dark ramp for the density heatmap.  Deliberately NOT the panel's
#: own orange/green and deliberately not a rainbow: the density is the ground, the
#: binned curve is the figure, and a multi-hue ramp would both fight the curve for
#: attention and imply category boundaries where there is only "more" and "less".
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "svod_density", ["#EEF2F7", "#9DB4CE", "#4A6E96", "#1F3A5F"])

#: A column holding fewer runs than this is left blank.  Column-normalising makes a
#: 3-run column as saturated as a 90-run one, which is the failure mode of every
#: conditional-density plot; blanking is honest where re-scaling would be a lie.
MIN_COLUMN_N = 8

#: Y resolution has to serve the SHORT-lived groups too.  The panels share one y-scale
#: so Ya and Vt_sour stay comparable, which means Vt_sour (median ННО ~75 d) occupies
#: the bottom fifth of an axis reaching past 2000 d; at 26 rows its whole distribution
#: would fall in two cells.
HEATMAP_XBINS, HEATMAP_YBINS = 22, 40

#: The shared colour scale is topped at this percentile of all cells rather than at the
#: max, so one freak cell cannot wash out every panel.  Cells above it clip, and the
#: colorbar is drawn with ``extend="max"`` to say so.
DENSITY_VMAX_PCT = 98.0


def _density(rows: pd.DataFrame, rate_col: str, value_col: str,
             ylim: tuple[float, float], *, normalize: bool = True
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x_edges, y_edges, z)`` for the 2-D run density; ``z`` is NaN where blank.

    ``normalize`` divides each rate column by its own count, so a column reads as the
    **conditional distribution of TTF at that rate** rather than as "how many wells
    happen to sit here".  That is the question these panels ask, and it is the part a
    raw-count heatmap answers badly: raw counts are dominated by the modal Ql, and the
    thin tails go invisible exactly where the trend is steepest.

    Runs outside ``ylim`` (the 1st–99th percentile band) fall out of the histogram, so
    each column sums to 1 *over the plotted window* — stated on the colorbar rather
    than left for the reader to assume.
    """
    x = rows[rate_col].to_numpy(float)
    y = rows[value_col].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xe = np.linspace(x.min(), x.max(), HEATMAP_XBINS + 1)
    ye = np.linspace(ylim[0], ylim[1], HEATMAP_YBINS + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[xe, ye])

    column_n = counts.sum(axis=1)
    z = counts.copy()
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(column_n[:, None] > 0, counts / column_n[:, None], np.nan)
    z[column_n < MIN_COLUMN_N, :] = np.nan
    z[counts == 0] = np.nan          # empty cells stay surface-coloured, not ramp-zero
    return xe, ye, z


def _heatmap(ax, rows: pd.DataFrame, rate_col: str, value_col: str,
             ylim: tuple[float, float], *, normalize: bool = True,
             vmax: float | None = None):
    """Draw :func:`_density` on ``ax``.  ``vmax`` is shared across a figure's panels."""
    xe, ye, z = _density(rows, rate_col, value_col, ylim, normalize=normalize)
    return ax.pcolormesh(xe, ye, np.ma.masked_invalid(z.T),
                         cmap=DENSITY_CMAP, shading="flat", zorder=1, linewidth=0,
                         vmin=0.0, vmax=vmax, rasterized=True)


def _panel(ax, frame: pd.DataFrame, fit: QlFit | None, *, color: str, ylabel: str,
           title: str, zero_line: bool, rows: pd.DataFrame | None,
           value_col: str, rate_col: str, xlabel: str, overlay: str,
           ylim: tuple[float, float], vmax: float | None = None,
           ):
    mesh = None
    if rows is not None and overlay == "scatter":
        for ev, style in EVENT_STYLE.items():
            s = rows[rows["event"] == ev] if ev == 1 else rows[rows["event"] != 1]
            if s.empty:
                continue
            ax.scatter(s[rate_col], s[value_col], s=9, alpha=0.28, linewidths=0,
                       zorder=1, **style)
    if rows is not None and overlay == "heatmap":
        # One shared vmax and ONE colorbar for the whole figure (added by ``plot``):
        # per-panel scales would let a pale Ya cell and a dark Vt_sour cell stand for
        # the same share, which is exactly the comparison these panels exist to make.
        mesh = _heatmap(ax, rows, rate_col, value_col, ylim, vmax=vmax)
        # The curve must stay legible on the darkest cells.
        ax.plot(frame["xmid"], frame["ymean"], color="white", linewidth=4.0, zorder=3)
    ax.errorbar(frame["xmid"], frame["ymean"], yerr=frame["yse"],
                marker="o", linestyle="-", color=color, capsize=3, linewidth=2.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    if fit is not None:
        resolved = fit.p_headline < RESOLVED_P
        span = (fit.grid_x >= frame["xmid"].min()) & (fit.grid_x <= frame["xmid"].max())
        ax.plot(fit.grid_x[span], fit.grid_y[span],
                color="#333333" if resolved else "#999999",
                linestyle="-" if resolved else "--", linewidth=1.4, zorder=3,
                label=fit.label)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(QL_REF, color="grey", linestyle=":", linewidth=1.2, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    # Gridlines over a filled mesh read as cell borders; the heatmap's own cells
    # already give the eye a reference lattice.
    ax.grid(False) if overlay == "heatmap" else ax.grid(alpha=0.25, which="both")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7.5, loc="best", framealpha=0.85)
    return mesh


def multiplier(days, ref_ttf: float, *, zero_is_reference: bool):
    """The plotted days re-expressed as a multiple of the median-rate life.

    On the MARGINAL panel the days *are* the life, so this is ``days / ref``.  On the
    RESIDUAL panel the days are a deviation from the pump-size prediction, so it is
    ``1 + resid / ref`` — putting 1.0 exactly on the dashed zero line.  Either way the
    number reads "x× what a median-Ql well of this group achieves".
    """
    offset = 1.0 if zero_is_reference else 0.0
    return offset + np.asarray(days, dtype=float) / ref_ttf


#: Vertical offsets (points) a multiplier label may be placed at, tried in order.
#: On a linear Ql axis the low deciles bunch into the left tenth of the panel, so a
#: single row of labels overprints itself there; stacking gives each one clear air.
LABEL_LEVELS = (10.0, 24.0, 38.0)

#: Breathing room (display px) added around each label's box before testing overlap.
LABEL_PAD_PX = 1.5


def _annotate_multipliers(ax, frame: pd.DataFrame, ref_ttf: float, *,
                          zero_is_reference: bool, renderer) -> tuple[int, int]:
    """Print the multiplier beside every binned point; returns ``(drawn, skipped)``.

    Labelling the points themselves rather than adding a second y-axis keeps one scale
    on the chart: the reader gets the ratio exactly where the question is asked,
    instead of converting by eye against an opposite axis.

    Placement tries each of :data:`LABEL_LEVELS` and keeps the first whose **rendered
    box** clears every label already placed.  Testing real boxes rather than x-spacing
    is what makes this work on these panels: two labels at different levels still
    collide when their points sit at different heights, which an x-only rule cannot
    see.  If no level is clear the label is **skipped** rather than overprinted — an
    unreadable pile of digits conveys less than a gap, and the number is in
    ``*__fits.csv`` either way.

    Requires the axes' final limits to already be set; a later ``set_ylim`` would move
    the points out from under their labels.
    """
    mults = multiplier(frame["ymean"], ref_ttf, zero_is_reference=zero_is_reference)
    x0, x1 = ax.get_xlim()
    placed: list = []
    drawn = skipped = 0
    for xi, yi, m in zip(frame["xmid"].to_numpy(float),
                         frame["ymean"].to_numpy(float), mults):
        if not np.isfinite(m):
            continue
        # Keep the outermost labels inside the frame instead of letting them run off it.
        xf = (xi - x0) / ((x1 - x0) or 1.0)
        ha = "left" if xf < 0.04 else "right" if xf > 0.96 else "center"
        for level in LABEL_LEVELS:
            cand = ax.annotate(
                f"{m:.2f}×", (xi, yi), textcoords="offset points",
                xytext=(0, level), ha=ha, fontsize=7.5, color="#333333", zorder=6,
                # The label sits over the density mesh, so it needs its own ground.
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))
            box = cand.get_window_extent(renderer).expanded(1.0, 1.0).padded(LABEL_PAD_PX)
            if not any(box.overlaps(p) for p in placed):
                placed.append(box)
                drawn += 1
                break
            cand.remove()
        else:
            skipped += 1
    return drawn, skipped


#: Y-axis span per field, in days (user decision 2026-07-27).  One scale for every Vt
#: group keeps the sour/nonsour/pooled rows directly comparable, while Ya — which lives
#: roughly twice as long — gets its own.  Pooling the two would spend most of the Vt
#: panels' height on empty axis, which is what a single fleet-wide scale did.
Y_CAP_BY_FIELD = {"Ya": 1100.0, "Vt": 750.0}

#: Day-tick spacing, held fixed across fields so a given tick gap is the same number
#: of days in every panel even though the spans differ.
Y_TICK_DAYS = 250.0


def _group_field(gp: GroupPanel) -> str:
    return str(gp.rows["field"].iloc[0])


def _ylims(gp: GroupPanel, overlay: str
           ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """``(marginal, qnom-residual, model-residual)`` y-limits for one group's row.

    A capped field gets ``(0, cap)`` for the marginal panel and a symmetric ``±cap/2``
    for both residual ones — the same total height in days, so a vertical distance
    means the same thing across a row and the two residual panels are directly
    comparable, which is the whole point of putting them side by side.  Fields with no
    cap fall back to the 1st–99th percentile of the points (or to the binned curve's
    own extent when nothing is drawn under it), which keeps a stray 2000-day run from
    squashing every decile mean to the axis floor.
    """
    cap = Y_CAP_BY_FIELD.get(_group_field(gp))
    if cap is not None:
        half = (-cap / 2.0, cap / 2.0)
        return (0.0, cap), half, half

    def span(value_col: str, frame: pd.DataFrame) -> tuple[float, float]:
        if overlay != "none":
            lo, hi = np.nanpercentile(gp.rows[value_col].to_numpy(float), [1, 99])
        else:
            lo = float((frame["ymean"] - frame["yse"]).min())
            hi = float((frame["ymean"] + frame["yse"]).max())
        pad = 0.08 * (hi - lo) or 1.0
        return lo - pad, hi + pad

    return (span("nno", gp.marginal), span("resid", gp.residual),
            span("model_resid", gp.model_residual))


def _clipped_share(rows: pd.DataFrame, value_col: str,
                   ylim: tuple[float, float]) -> float:
    """Share of runs outside the plotted window — annotated, never left silent."""
    v = rows[value_col].to_numpy(float)
    return float(np.mean((v < ylim[0]) | (v > ylim[1]))) if len(v) else 0.0


def _shared_vmax(groups: dict[str, GroupPanel], rate_col: str,
                 limits: dict[str, tuple[tuple[float, float], tuple[float, float]]]
                 ) -> float:
    """One colour scale for every panel in a figure.

    Per-panel autoscaling is the quiet failure of a grid of heatmaps: a pale Ya cell
    and a dark Vt_sour cell would stand for the same share, and the cross-group
    comparison the figure exists to support silently stops working.
    """
    cells = []
    for name, gp in groups.items():
        for value_col, ylim in zip(("nno", "resid", "model_resid"), limits[name]):
            z = _density(gp.rows, rate_col, value_col, ylim)[2]
            cells.append(z[np.isfinite(z)])
    allc = np.concatenate(cells)
    return float(np.percentile(allc, DENSITY_VMAX_PCT)) if allc.size else 1.0


def plot(groups: dict[str, GroupPanel], out_path: Path, *, suptitle: str,
         overlay: str = "none") -> Path:
    """One row per group: MARGINAL | Qnom-RESIDUAL.

    Y-limits are per FIELD (:data:`Y_CAP_BY_FIELD`), so the three Vt rows stay
    directly comparable with each other while Ya — roughly twice the life — keeps its
    own span.  Day ticks are the same 250-day step everywhere, so the two spans are
    still read off in one unit.  ``overlay`` picks how the individual runs are drawn
    under the binned curve (:data:`OVERLAYS`).
    """
    if overlay not in OVERLAYS:
        raise ValueError(f"overlay must be one of {OVERLAYS}, got {overlay!r}")
    rate = next(iter(groups.values())).rate
    col, xlabel, short = RATES[rate]
    limits = {name: _ylims(gp, overlay) for name, gp in groups.items()}
    xlo = min(float(gp.rows[col].min()) for gp in groups.values())
    xhi = max(float(gp.rows[col].max()) for gp in groups.values())
    xpad = 0.03 * (xhi - xlo)
    xlim = (max(0.0, xlo - xpad), xhi + xpad)

    vmax = _shared_vmax(groups, col, limits) if overlay == "heatmap" else None

    fig, axes = plt.subplots(len(groups), 3, figsize=(18.0, 4.3 * len(groups)),
                             squeeze=False)
    mesh = None
    for r, (name, gp) in enumerate(groups.items()):
        pts = gp.rows if overlay != "none" else None
        mlim, rlim, klim = limits[name]
        specs = (
            (gp.marginal, gp.marginal_fit, "#E8762C", "ННО (days)",
             f"{name}: MARGINAL TTF vs {short}", False, "nno", mlim),
            (gp.residual, gp.residual_fit, "#2CA05A", "residual ННО (days)",
             f"{name}: RESIDUAL after log Qnom (pump size)", True, "resid", rlim),
            (gp.model_residual, gp.model_residual_fit, "#5B4B9E",
             "residual ННО (days)",
             f"{name}: RESIDUAL after {gp.model_label}", True, "model_resid", klim),
        )
        for c, (frame, fit, colour, ylab, title, zero, value_col, lim) in enumerate(specs):
            got = _panel(axes[r][c], frame, fit, color=colour, ylabel=ylab, title=title,
                         zero_line=zero, rows=pts, value_col=value_col,
                         rate_col=col, xlabel=xlabel, overlay=overlay, ylim=lim,
                         vmax=vmax)
            mesh = got or mesh
        for ax, lim, value_col in zip(axes[r], (mlim, rlim, klim),
                                      ("nno", "resid", "model_resid")):
            ax.set_ylim(*lim)
            ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_DAYS))
            ax.set_xlim(*xlim)
            # A capped axis hides runs.  Say how many rather than let the panel
            # imply the cloud simply ends where the frame does.
            share = _clipped_share(gp.rows, value_col, lim)
            if share > 0.005:
                ax.annotate(f"{share:.0%} of runs outside axis", (0.99, 0.985),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=7.5, color="#666666")

    # Multiplier labels go on LAST, once every axis has its final limits: they are
    # placed by rendered bounding box, and a later set_ylim would slide the points out
    # from under them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for r, gp in enumerate(groups.values()):
        if not (np.isfinite(gp.ref_ttf) and gp.ref_ttf > 0):
            continue
        _annotate_multipliers(axes[r][0], gp.marginal, gp.ref_ttf,
                              zero_is_reference=False, renderer=renderer)
        for c, frame in ((1, gp.residual), (2, gp.model_residual)):
            _annotate_multipliers(axes[r][c], frame, gp.ref_ttf,
                                  zero_is_reference=True, renderer=renderer)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    if mesh is not None:
        # A single figure-level bar: eight identical per-panel bars would be eight
        # copies of one legend, and they crowd the plots they explain.
        cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), extend="max",
                          fraction=0.02, pad=0.015, aspect=60)
        cb.set_label("share of the rate column's runs (each column sums to 1)",
                     fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── entry point ───────────────────────────────────────────────────────────────

def rate_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    """Is the ``ql_svod`` → ``ql_start`` change the DEFINITION or the POPULATION?

    ``ql_start`` exists for ~80% of runs, and the ones it loses are systematically
    the oldest, so the two rate figures do not sit on the same rows.  Comparing them
    directly would confound "the rate definition changed" with "the sample changed".
    This holds the rows fixed — every column is computed on the ``ql_start``-supported
    subset — so the only thing varying between the last two is which rate is on the
    x-axis.  ``svod_full`` is carried alongside purely to show what the restriction
    itself costs.
    """
    sub = panel[panel["ql_start"].notna()]
    rows = []
    for name, expr in GROUPS:
        rec: dict = {"group": name}
        for label, g, col in (("svod_full", panel.query(expr), "ql"),
                              ("svod_subset", sub.query(expr), "ql"),
                              ("start_subset", sub.query(expr), "ql_start")):
            g = g[g[col].notna() & g[col].between(*QL_BOUNDS)]
            f = _fit(g[col], _qnom_residual(g)) if len(g) >= 30 else None
            rec[f"{label}_n"] = len(g)
            rec[f"{label}_days_per_efold"] = f.beta1 if f else np.nan
            rec[f"{label}_p"] = f.p1 if f else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


#: Columns dumped per group so a reader can re-derive any panel point by hand.
RUN_COLUMNS = ["code", "field", "h2s_class", "install", "end", "ql", "ql_start",
               "ql_start_n_days", "ql_start_gap_days", "ql_start_status",
               "qnom", "kpod", "freq", "nno", "event", "resid",
               "model_life_mult", "model_resid"]


#: Appended to the figure title so the file says what it is drawing.
OVERLAY_SUBTITLE = {
    "none": "",
    "scatter": " · every run scattered, coloured by outcome",
    "heatmap": " · density of runs, each rate column normalised to its own total",
}


def run(
    prediction_workbook_path: Path | None = None,
    variants: tuple[str, ...] = ("failures", "all_closed"),
    rates: tuple[str, ...] = ("ql_svod", "ql_start"),
    overlays: tuple[str, ...] = OVERLAYS,
    bins: int = 10,
    mc_cohort: bool = True,
    db_path: Path | None = None,
    run_date: str | None = None,
) -> Path:
    out = results_dir(SLUG, run_date)
    tables, figures = out / "tables", out / "figures"

    summary, coverages = [], []
    for variant in variants:
        panel = load_panel(prediction_workbook_path, variant=variant, mc_cohort=mc_cohort)
        if "ql_start" in rates:
            panel, start_cov = attach_ql_start(panel, db_path=db_path)
            start_cov.insert(0, "variant", variant)
            start_cov.to_csv(tables / f"{variant}__ql_start_coverage.csv", index=False)
            if "ql_svod" in rates:
                rate_comparison(panel).to_csv(
                    tables / f"{variant}__rate_definition_vs_population.csv", index=False)
        rows = VARIANT_ROWS[variant]

        for rate in rates:
            groups, coverage = build(panel, bins=bins, rate=rate)
            coverage.insert(0, "variant", variant)
            coverages.append(coverage)
            if not groups:
                continue
            stem = f"{variant}__{rate}"
            fit_table(groups).to_csv(tables / f"{stem}__fits.csv", index=False)
            for name, gp in groups.items():
                slug = name.replace(" ", "_").replace("(", "").replace(")", "")
                gp.marginal.to_csv(tables / f"{stem}__marginal_{slug}.csv", index=False)
                gp.residual.to_csv(tables / f"{stem}__qnom_residual_{slug}.csv", index=False)
                gp.model_residual.to_csv(
                    tables / f"{stem}__field_model_residual_{slug}.csv", index=False)
                cols = [c for c in RUN_COLUMNS if c in gp.rows.columns]
                gp.rows[cols].to_csv(tables / f"{stem}__runs_{slug}.csv", index=False)

            short = RATES[rate][2]
            for overlay in overlays:
                plot(
                    groups,
                    figures / f"ttf_vs__{stem}"
                    f"{'' if overlay == 'none' else '_' + overlay}.png",
                    suptitle=(
                        f"Свод TTF (ННО) vs {short} — Ya vs Vt "
                        f"(v3.2 sour relabel), {rows}" + OVERLAY_SUBTITLE[overlay]
                    ),
                    overlay=overlay,
                )
            summary.append({
                "variant": variant, "rate": rate, "rows": rows, "mc_cohort": mc_cohort,
                "n_panel": len(panel), "n_groups": len(groups),
                "n_plotted": int(sum(len(gp.rows) for gp in groups.values())),
            })
    pd.concat(coverages, ignore_index=True).to_csv(
        tables / "group_coverage.csv", index=False)
    pd.DataFrame(summary).to_csv(tables / "panel_summary.csv", index=False)
    return out
