"""Свод TTF (ННО) vs the two DESIGN choices: nominal pump size and contractor.

The companion to :mod:`svod_ttf_vs_ql`, which puts the *measured rate* on the x-axis.
Here the x-axis is what was decided before the well ever ran:

* **Qnom** («Ном. Произв. м₃/сут») — the pump's nameplate delivery.  Continuous, so it
  gets exactly the panel grammar of the Ql figures: decile means, a log-Qnom fit, TTF
  multipliers on every point, and the same scatter / column-normalised heatmap views.
* **contractor** — brt / slb / oth.  Categorical, so decile bins and a per-e-fold slope
  are meaningless; it gets its own form (see :func:`run_contractor`).

Both are worth their own figure because of the Ya v2.1 result that nameplate **Qnom
subsumes Ql**, and because contractor is the single biggest lever in the deployed Vt
model (nonsour oth HR 3.017).  Both models carry a fitted factor for each, so both
figures can be checked against what is actually shipped.

**The confound to keep in mind on every panel here.** Neither Qnom nor contractor is
assigned at random: big pumps go in high-rate wells, and contractors do not get the same
mix of wells.  So the raw panel is the association and the adjusted panel is the part
that survives holding the operating point fixed — they answer different questions and
the gap between them is the confound, not noise.  The contractor figure additionally
inherits the standing warning that a **failures-only** contractor comparison is a
collider (brt/slb reverses); that is why both denominators are always emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk.svod_nno_decomposition import (  # noqa: E402
    _bin_mean,
    load_panel,
)
from analysis.workflows.production_risk.svod_ttf_vs_ql import (  # noqa: E402
    GROUPS,
    OVERLAY_SUBTITLE,
    OVERLAYS,
    VARIANT_ROWS,
    Y_CAP_BY_FIELD,
    Y_TICK_DAYS,
    _annotate_multipliers,
    _clipped_share,
    _density,
    _fit,
    _panel,
    QL_BOUNDS,
    QlFit,
    attach_ql_start,
    model_life_multiplier,
)
from analysis.workflows.production_risk.svod_ttf_vs_ql import (  # noqa: E402
    DENSITY_VMAX_PCT,
    FIELD_MODELS,
    _life_mult_interp,
    _ya_model,
)

SLUG = "production_risk_svod_ttf_by_design"

#: Nameplate delivery the panels are referenced at — the same 250 m³/сут the Ql figures
#: use, so the two sets of multipliers are read against comparable pump duty.
QNOM_REF = 250.0

#: «Ном. Произв. м₃/сут» runs to 9000 on a handful of rows, which no ESP in this fleet
#: delivers; those are transcription noise.  Dropped rows are counted, never silently.
QNOM_BOUNDS = (20.0, 2000.0)

CONTRACTORS = ("brt", "slb", "oth")
CONTRACTOR_REF = "brt"
CONTRACTOR_LABEL = {"brt": "Борец (brt)", "slb": "Шлюмберже (slb)", "oth": "прочие (oth)"}

#: Contractor point colours — a fixed order, never cycled, so a contractor keeps its
#: colour when a thin group drops out of a panel.
CONTRACTOR_COLOR = {"brt": "#2C6E9A", "slb": "#E8762C", "oth": "#8E44AD"}

#: Rows below this in a contractor cell are not plotted: a mean life on 5 runs is not a
#: contractor effect.  The count is reported either way.
MIN_CONTRACTOR_N = 15


# ── Qnom: continuous, same grammar as the Ql figures ──────────────────────────

@dataclass(frozen=True)
class QnomPanel:
    """One group's three curves vs nameplate Qnom, and the rows behind them."""

    name: str
    rows: pd.DataFrame
    marginal: pd.DataFrame
    residual: pd.DataFrame
    model_residual: pd.DataFrame
    marginal_fit: QlFit | None
    residual_fit: QlFit | None
    model_residual_fit: QlFit | None
    model_label: str
    qnom_median: float
    ref_ttf: float


def _ql_residual(g: pd.DataFrame) -> pd.Series:
    """ННО minus its ``log Ql`` prediction — the mirror of the Ql figure's Qnom control.

    Holding the measured rate fixed and varying nameplate size asks the converse of that
    figure: two wells producing the same liquid, one through a bigger pump — does the
    pump choice still move the life?
    """
    y = g["nno"].to_numpy(float)
    design = np.column_stack([np.ones(len(g)), np.log(g["ql"].to_numpy(float))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ beta, index=g.index, name="resid")


def _model_rate_residual(g: pd.DataFrame) -> pd.Series:
    """ННО minus the field model's own **Ql** layer, with only the level fitted here.

    The layer removed is the rate layer, not a size layer, because that is what is
    actually shipped: Vt v3.2 has no Qnom arm at all, and Ya's Qnom arm belongs to v2.1
    rather than to v2.  So this panel asks the deployable question — **after the model's
    rate layer has done its work, does nameplate size still carry structure?**  A slope
    here is capacity the shipped models do not price.
    """
    y = g["nno"].to_numpy(float)
    m = model_life_multiplier(g, "ql").to_numpy(float)
    ok = np.isfinite(m) & np.isfinite(y)
    if not ok.any():
        return pd.Series(np.nan, index=g.index, dtype=float)
    level = float((y[ok] * m[ok]).sum() / (m[ok] ** 2).sum())
    return pd.Series(y - level * m, index=g.index, name="model_resid")


def build_qnom(panel: pd.DataFrame, bins: int = 10, min_n: int = 40
               ) -> tuple[dict[str, QnomPanel], pd.DataFrame]:
    out: dict[str, QnomPanel] = {}
    coverage = []
    for name, expr in GROUPS:
        present = panel.query(expr)
        g = present[present["qnom"].notna() & present["qnom"].between(*QNOM_BOUNDS)
                    & present["ql"].notna() & (present["ql"] > 0)]
        coverage.append({
            "group": name, "n_present": len(present), "n_in_bounds": len(g),
            "n_dropped": len(present) - len(g),
            "n_failures": int((g["event"] == 1).sum()),
            "n_other_pulls": int((g["event"] != 1).sum()),
            "qnom_min": g["qnom"].min() if len(g) else np.nan,
            "qnom_max": g["qnom"].max() if len(g) else np.nan,
            "n_distinct_qnom": int(g["qnom"].nunique()) if len(g) else 0,
            "median_nno": g["nno"].median() if len(g) else np.nan,
        })
        if len(g) < max(min_n, bins * 3):
            continue
        resid = _ql_residual(g)
        model_resid = _model_rate_residual(g)
        b = int(np.clip(len(g) // 30, 4, bins))
        marginal_fit = _fit(g["qnom"], g["nno"], x_name="Qnom")
        qnom_median = float(g["qnom"].median())
        ref_ttf = marginal_fit.predict(qnom_median) if marginal_fit else float("nan")
        out[name] = QnomPanel(
            name=name,
            rows=g.assign(resid=resid, model_resid=model_resid),
            marginal=_bin_mean(g["qnom"], g["nno"], b),
            residual=_bin_mean(g["qnom"], resid, b),
            model_residual=_bin_mean(g["qnom"], model_resid, b),
            marginal_fit=marginal_fit,
            residual_fit=_fit(g["qnom"], resid, x_name="Qnom"),
            model_residual_fit=_fit(g["qnom"], model_resid, x_name="Qnom"),
            model_label=FIELD_MODELS[str(g["field"].iloc[0])][0],
            qnom_median=qnom_median,
            ref_ttf=ref_ttf,
        )
    return out, pd.DataFrame(coverage)


def qnom_fit_table(groups: dict[str, QnomPanel]) -> pd.DataFrame:
    rows = []
    for name, gp in groups.items():
        for kind, f in (("marginal", gp.marginal_fit),
                        ("log_ql_residual", gp.residual_fit),
                        ("field_model_rate_residual", gp.model_residual_fit)):
            if f is None:
                continue
            rows.append({
                "group": name, "panel": kind, "field_model": gp.model_label, "n": f.n,
                "days_per_efold_qnom": f.beta1, "se": f.se1, "p": f.p1,
                "curvature": f.beta2, "curvature_p": f.p2,
                "qnom_median": gp.qnom_median, "ref_ttf_at_qnom_median": gp.ref_ttf,
            })
    return pd.DataFrame(rows)


def _qnom_ylims(gp: QnomPanel, overlay: str):
    cap = Y_CAP_BY_FIELD.get(str(gp.rows["field"].iloc[0]))
    if cap is not None:
        half = (-cap / 2.0, cap / 2.0)
        return (0.0, cap), half, half

    def span(value_col: str, frame: pd.DataFrame):
        if overlay != "none":
            lo, hi = np.nanpercentile(gp.rows[value_col].to_numpy(float), [1, 99])
        else:
            lo = float((frame["ymean"] - frame["yse"]).min())
            hi = float((frame["ymean"] + frame["yse"]).max())
        pad = 0.08 * (hi - lo) or 1.0
        return lo - pad, hi + pad

    return (span("nno", gp.marginal), span("resid", gp.residual),
            span("model_resid", gp.model_residual))


def plot_qnom(groups: dict[str, QnomPanel], out_path: Path, *, suptitle: str,
              overlay: str = "none") -> Path:
    """Same three-panel grammar as the Ql figures, with Qnom on the x-axis."""
    if overlay not in OVERLAYS:
        raise ValueError(f"overlay must be one of {OVERLAYS}, got {overlay!r}")
    limits = {k: _qnom_ylims(gp, overlay) for k, gp in groups.items()}
    xlo = min(float(gp.rows["qnom"].min()) for gp in groups.values())
    xhi = max(float(gp.rows["qnom"].max()) for gp in groups.values())
    xpad = 0.03 * (xhi - xlo)
    xlim = (max(0.0, xlo - xpad), xhi + xpad)

    vmax = None
    if overlay == "heatmap":
        cells = []
        for k, gp in groups.items():
            for value_col, ylim in zip(("nno", "resid", "model_resid"), limits[k]):
                z = _density(gp.rows, "qnom", value_col, ylim)[2]
                cells.append(z[np.isfinite(z)])
        allc = np.concatenate(cells)
        vmax = float(np.percentile(allc, DENSITY_VMAX_PCT)) if allc.size else 1.0

    fig, axes = plt.subplots(len(groups), 3, figsize=(18.0, 4.3 * len(groups)),
                             squeeze=False)
    mesh = None
    for r, (name, gp) in enumerate(groups.items()):
        pts = gp.rows if overlay != "none" else None
        mlim, rlim, klim = limits[name]
        specs = (
            (gp.marginal, gp.marginal_fit, "#E8762C", "ННО (days)",
             f"{name}: MARGINAL TTF vs Qnom", False, "nno", mlim),
            (gp.residual, gp.residual_fit, "#2CA05A", "residual ННО (days)",
             f"{name}: RESIDUAL after log Ql (measured rate)", True, "resid", rlim),
            (gp.model_residual, gp.model_residual_fit, "#5B4B9E", "residual ННО (days)",
             f"{name}: RESIDUAL after {gp.model_label}", True, "model_resid", klim),
        )
        for c, (frame, fit, colour, ylab, title, zero, value_col, lim) in enumerate(specs):
            got = _panel(axes[r][c], frame, fit, color=colour, ylabel=ylab, title=title,
                         zero_line=zero, rows=pts, value_col=value_col,
                         rate_col="qnom",
                         xlabel="Qnom, м³/сут (nameplate delivery)",
                         overlay=overlay, ylim=lim, vmax=vmax)
            mesh = got or mesh
        for ax, lim, value_col in zip(axes[r], (mlim, rlim, klim),
                                      ("nno", "resid", "model_resid")):
            ax.set_ylim(*lim)
            ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_DAYS))
            ax.set_xlim(*xlim)
            # The Ql panels' reference line is Ql=250; the same duty in nameplate terms.
            ax.axvline(QNOM_REF, color="grey", linestyle=":", linewidth=1.2, zorder=2)
            share = _clipped_share(gp.rows, value_col, lim)
            if share > 0.005:
                ax.annotate(f"{share:.0%} of runs outside axis", (0.99, 0.985),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=7.5, color="#666666")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), extend="max",
                          fraction=0.02, pad=0.015, aspect=60)
        cb.set_label("share of the Qnom column's runs (each column sums to 1)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)

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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def qnom_vs_ql_start(panel: pd.DataFrame) -> pd.DataFrame:
    """The decisive size-vs-rate test, run against the *clean* rate.

    The Qnom figure controls for the Свод snapshot Ql, and on that axis size and rate
    each leave the other a similar residual — they are collinear proxies and neither
    wins.  But the snapshot is contaminated (it can be read near the failure), so that
    tie is not informative.  This runs the same contest on ``ql_start``, the
    first-operating-month rate, which is fixed before the outcome is known:

    * ``qnom_given_ql_start`` — does nameplate size survive the clean rate?
    * ``ql_start_given_qnom`` — does the clean rate survive nameplate size?

    Both on the same rows, so the comparison is symmetric.  This is the Свод-ННО
    counterpart of the Ya v2.1 within-covariate test.
    """
    sub = panel[panel["ql_start"].notna() & panel["qnom"].notna()
                & panel["ql_start"].between(*QL_BOUNDS)
                & panel["qnom"].between(*QNOM_BOUNDS)]
    rows = []
    for name, expr in GROUPS:
        g = sub.query(expr)
        if len(g) < 40:
            rows.append({"group": name, "n": len(g)})
            continue
        y = g["nno"].to_numpy(float)
        lq = np.log(g["ql_start"].to_numpy(float))
        ln = np.log(g["qnom"].to_numpy(float))
        rec = {"group": name, "n": len(g),
               "corr_log_qnom_log_ql_start": float(np.corrcoef(ln, lq)[0, 1])}
        for label, keep, drop in (("qnom_given_ql_start", ln, lq),
                                  ("ql_start_given_qnom", lq, ln)):
            design = np.column_stack([np.ones(len(g)), drop])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            f = _fit(pd.Series(np.exp(keep)), pd.Series(y - design @ beta))
            rec[f"{label}_days_per_efold"] = f.beta1 if f else np.nan
            rec[f"{label}_p"] = f.p1 if f else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ── contractor: categorical ───────────────────────────────────────────────────

def _vt_contractor_mult(g: pd.DataFrame) -> pd.Series:
    """Vt v3.2 contractor life multiplier, evaluated **per sour stratum**.

    v3.2 carries a different contractor HR in each stratum (nonsour oth 3.017 vs sour
    1.507).  Keeping it per run means the group-level average afterwards reflects that
    contractor's actual sour mix rather than a stratum the group may not even have.
    """
    from analysis.workflows.production_risk import vt_composed_model as VC
    out = pd.Series(np.nan, index=g.index, dtype=float)
    for stratum, part in g.groupby("h2s_class"):
        b = VC.BASELINE[str(stratum)]
        beta0, eta0 = b["beta0"], b["eta0"]
        r0 = VC.rmst(eta0, beta0)
        hr = part["contractor"].map(VC.CONTRACTOR[str(stratum)]).to_numpy(float)
        out.loc[part.index] = _life_mult_interp(
            hr, lambda t: VC.rmst(eta0 * t ** (-1.0 / beta0), beta0) / r0)
    return out


def _ya_contractor_mult(g: pd.DataFrame) -> pd.Series:
    m = _ya_model()
    hr = g["contractor"].map(
        lambda c: m.contractor_hr(c) if isinstance(c, str) else np.nan).to_numpy(float)
    return pd.Series(_life_mult_interp(hr, lambda t: m.life_mult(float(t))),
                     index=g.index, dtype=float)


#: Field → contractor life-multiplier callable.  A named seam, like ``FIELD_MODELS``:
#: the Ya entry fits ``ya_k1k2_hybrid`` off the warehouse, so tests must be able to
#: substitute it rather than pay ~130 s per session.
CONTRACTOR_MODELS = {"Ya": _ya_contractor_mult, "Vt": _vt_contractor_mult}


def contractor_model_multiplier(g: pd.DataFrame) -> pd.Series:
    """Each run's LIFE multiplier from its field model's contractor factor alone."""
    return CONTRACTOR_MODELS[str(g["field"].iloc[0])](g)


def _rate_size_residual(g: pd.DataFrame) -> pd.Series:
    """ННО minus its ``log Ql + log Qnom`` prediction — the operating point held fixed.

    Contractors are not handed the same wells.  Without this the contractor panel reads
    "who was given the hard wells" as much as "who builds the better pump"; with it, the
    remaining gap is the part that is not explained by rate and pump size.
    """
    y = g["nno"].to_numpy(float)
    design = np.column_stack([np.ones(len(g)),
                              np.log(g["ql"].to_numpy(float)),
                              np.log(g["qnom"].to_numpy(float))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ beta, index=g.index, name="adj")


def _boot_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    draws = values[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


@dataclass(frozen=True)
class ContractorPanel:
    name: str
    rows: pd.DataFrame
    table: pd.DataFrame
    ref_ttf: float


def build_contractor(panel: pd.DataFrame, n_boot: int = 1000, seed: int = 0
                     ) -> tuple[dict[str, ContractorPanel], pd.DataFrame]:
    """Per group, observed vs model-predicted life multiplier for each contractor.

    Multipliers are taken against **brt**, the reference both shipped models use, so the
    observed and predicted columns are on one scale and directly comparable.
    """
    out: dict[str, ContractorPanel] = {}
    coverage = []
    for name, expr in GROUPS:
        present = panel.query(expr)
        g = present[present["contractor"].isin(CONTRACTORS)
                    & present["ql"].notna() & (present["ql"] > 0)
                    & present["qnom"].notna() & (present["qnom"] > 0)].copy()
        if g.empty or CONTRACTOR_REF not in set(g["contractor"]):
            coverage.append({"group": name, "n_present": len(present), "n_used": len(g),
                             "note": "no brt reference in group"})
            continue
        g["adj"] = _rate_size_residual(g)
        g["model_mult"] = contractor_model_multiplier(g)
        level = float(g["nno"].mean())

        rows = []
        ref_raw = float(g.loc[g["contractor"] == CONTRACTOR_REF, "nno"].mean())
        ref_adj = level + float(g.loc[g["contractor"] == CONTRACTOR_REF, "adj"].mean())
        ref_model = float(g.loc[g["contractor"] == CONTRACTOR_REF, "model_mult"].mean())
        for c in CONTRACTORS:
            part = g[g["contractor"] == c]
            rec = {"group": name, "contractor": c, "n": len(part),
                   "n_failures": int((part["event"] == 1).sum()),
                   "plotted": len(part) >= MIN_CONTRACTOR_N}
            if len(part):
                raw = float(part["nno"].mean())
                adj = level + float(part["adj"].mean())
                lo, hi = _boot_ci(part["nno"].to_numpy(float), n_boot, seed)
                alo, ahi = _boot_ci(part["adj"].to_numpy(float), n_boot, seed)
                rec.update({
                    "mean_nno": raw, "median_nno": float(part["nno"].median()),
                    "raw_mult": raw / ref_raw if ref_raw else np.nan,
                    "raw_lo": lo / ref_raw if ref_raw else np.nan,
                    "raw_hi": hi / ref_raw if ref_raw else np.nan,
                    "adj_mult": adj / ref_adj if ref_adj else np.nan,
                    "adj_lo": (level + alo) / ref_adj if ref_adj else np.nan,
                    "adj_hi": (level + ahi) / ref_adj if ref_adj else np.nan,
                    "model_mult": float(part["model_mult"].mean()) / ref_model
                    if ref_model else np.nan,
                })
            rows.append(rec)
        table = pd.DataFrame(rows)
        coverage.append({"group": name, "n_present": len(present), "n_used": len(g),
                         "note": ""})
        out[name] = ContractorPanel(name=name, rows=g, table=table, ref_ttf=level)
    return out, pd.DataFrame(coverage)


def plot_contractor(groups: dict[str, ContractorPanel], out_path: Path, *,
                    suptitle: str) -> Path:
    """Observed vs shipped-model life multiplier per contractor, raw and adjusted.

    A categorical x gets a categorical form: one point and bootstrap interval per
    contractor, with the model's own prediction as a hollow diamond beside it.  Decile
    bins and a per-e-fold slope — the Qnom grammar — would be meaningless here.
    """
    fig, axes = plt.subplots(len(groups), 2, figsize=(12.0, 3.9 * len(groups)),
                             squeeze=False)
    spec = (("raw_mult", "raw_lo", "raw_hi", "OBSERVED (raw)"),
            ("adj_mult", "adj_lo", "adj_hi", "ADJUSTED for log Ql + log Qnom"))
    lo_all, hi_all = [], []
    for gp in groups.values():
        for m, lo, hi, _ in spec:
            t = gp.table[gp.table["plotted"]]
            lo_all += [t[lo].min(), t[m].min(), t["model_mult"].min()]
            hi_all += [t[hi].max(), t[m].max(), t["model_mult"].max()]
    ylim = (min(0.0, np.nanmin(lo_all) - 0.1), np.nanmax(hi_all) + 0.15)

    for r, (name, gp) in enumerate(groups.items()):
        for c, (mcol, locol, hicol, title) in enumerate(spec):
            ax = axes[r][c]
            for i, contractor in enumerate(CONTRACTORS):
                row = gp.table[gp.table["contractor"] == contractor]
                if row.empty:
                    continue
                row = row.iloc[0]
                if not row["plotted"]:
                    ax.annotate(f"n={int(row['n'])}\n(too few)", (i, ylim[0] + 0.08),
                                ha="center", fontsize=7.5, color="#999999")
                    continue
                colour = CONTRACTOR_COLOR[contractor]
                ax.errorbar(i, row[mcol],
                            yerr=[[row[mcol] - row[locol]], [row[hicol] - row[mcol]]],
                            marker="o", markersize=8, color=colour, capsize=4,
                            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.9,
                            zorder=3, label="observed" if i == 0 else None)
                ax.scatter([i + 0.22], [row["model_mult"]], marker="D", s=52,
                           facecolors="none", edgecolors="#333333", linewidths=1.4,
                           zorder=4, label="shipped model" if i == 0 else None)
                ax.annotate(f"{row[mcol]:.2f}×", (i, row[hicol]),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=8, color="#333333")
                ax.annotate(f"n={int(row['n'])}", (i, ylim[0] + 0.04), ha="center",
                            fontsize=7.5, color="#666666")
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, zorder=2)
            ax.set_xticks(range(len(CONTRACTORS)))
            ax.set_xticklabels([CONTRACTOR_LABEL[c] for c in CONTRACTORS], fontsize=9)
            ax.set_xlim(-0.5, len(CONTRACTORS) - 0.3)
            ax.set_ylim(*ylim)
            ax.set_ylabel("TTF multiplier vs brt")
            ax.set_title(f"{name}: {title}", fontsize=11)
            ax.grid(alpha=0.25, axis="y")
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def contractor_table(groups: dict[str, ContractorPanel]) -> pd.DataFrame:
    return pd.concat([gp.table for gp in groups.values()], ignore_index=True) \
        if groups else pd.DataFrame()


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    prediction_workbook_path: Path | None = None,
    variants: tuple[str, ...] = ("failures", "all_closed"),
    axes: tuple[str, ...] = ("qnom", "contractor"),
    overlays: tuple[str, ...] = OVERLAYS,
    bins: int = 10,
    mc_cohort: bool = True,
    run_date: str | None = None,
) -> Path:
    out = results_dir(SLUG, run_date)
    tables, figures = out / "tables", out / "figures"

    summary = []
    for variant in variants:
        panel = load_panel(prediction_workbook_path, variant=variant, mc_cohort=mc_cohort)
        rows = VARIANT_ROWS[variant]

        if "qnom" in axes:
            # The Qnom-vs-clean-rate contest needs the daily-warehouse rate attached.
            with_start, _ = attach_ql_start(panel)
            qnom_vs_ql_start(with_start).to_csv(
                tables / f"{variant}__qnom_vs_ql_start.csv", index=False)
            groups, coverage = build_qnom(panel, bins=bins)
            coverage.insert(0, "variant", variant)
            coverage.to_csv(tables / f"{variant}__qnom_coverage.csv", index=False)
            if groups:
                qnom_fit_table(groups).to_csv(
                    tables / f"{variant}__qnom_fits.csv", index=False)
                for name, gp in groups.items():
                    slug = name.replace(" ", "_").replace("(", "").replace(")", "")
                    gp.marginal.to_csv(
                        tables / f"{variant}__qnom_marginal_{slug}.csv", index=False)
                    gp.residual.to_csv(
                        tables / f"{variant}__qnom_ql_residual_{slug}.csv", index=False)
                    gp.model_residual.to_csv(
                        tables / f"{variant}__qnom_model_residual_{slug}.csv", index=False)
                for overlay in overlays:
                    plot_qnom(
                        groups,
                        figures / f"ttf_vs_qnom__{variant}"
                        f"{'' if overlay == 'none' else '_' + overlay}.png",
                        suptitle=(f"Свод TTF (ННО) vs nameplate Qnom — Ya vs Vt "
                                  f"(v3.2 sour relabel), {rows}"
                                  + OVERLAY_SUBTITLE[overlay]),
                        overlay=overlay,
                    )
                summary.append({"variant": variant, "axis": "qnom", "rows": rows,
                                "n_groups": len(groups)})

        if "contractor" in axes:
            cgroups, ccoverage = build_contractor(panel)
            ccoverage.insert(0, "variant", variant)
            ccoverage.to_csv(tables / f"{variant}__contractor_coverage.csv", index=False)
            if cgroups:
                contractor_table(cgroups).to_csv(
                    tables / f"{variant}__contractor_multipliers.csv", index=False)
                plot_contractor(
                    cgroups,
                    figures / f"ttf_by_contractor__{variant}.png",
                    suptitle=(f"Свод TTF (ННО) by contractor — observed vs shipped model "
                              f"(Ya v2 / Vt v3.2), {rows}"),
                )
                summary.append({"variant": variant, "axis": "contractor", "rows": rows,
                                "n_groups": len(cgroups)})
    pd.DataFrame(summary).to_csv(tables / "panel_summary.csv", index=False)
    return out
