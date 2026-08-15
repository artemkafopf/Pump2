"""Свод TTF (ННО) vs frequency — before and after the field's fitted Ql layer is removed.

The third panel set in this family, after :mod:`svod_ttf_vs_ql` (measured rate) and
:mod:`svod_ttf_by_design` (nameplate size, contractor).  Here the x-axis is running
frequency and the question is the one the Ya→Vt transfer work turns on: **frequency and
rate are tied by the affinity law, so any raw freq curve is partly the rate curve.**
Strip the rate out and see what frequency has left.

The extraction is the recipe the repo already uses for this (``ya_freq_empirical``,
``project_ya_to_vt_freq_transfer``) — the Ql layer removed as an **AFT time-scale
offset**, not as a subtracted residual::

    t_adj = t · θ_Ql(Ql) ** (1 / β)

A run held at high Ql carries θ_Ql > 1 (more hazard), so its observed life is scaled
*up* to what it would have been at the model's reference rate.  ``t_adj`` is therefore
still a life in days on the same axis as the raw panel, which is what makes the
before/after pair directly comparable — a subtracted residual would not be.

Each field is offset by **its own shipped layer**: Ya by **Ya v2** (`ya_k1k2_hybrid`,
rate arm Ql, β = the k1 baseline shape) and Vt by **v3.2** (`vt_composed_model.theta_ql`,
β₀ per sour stratum).  Ya v2 is the layer the frequency transfer was argued from.

Three panels per group:

1. **BEFORE** — mean ННО per frequency decile, raw.
2. **AFTER** — the same, on ``t_adj``: the Ql layer removed.
3. **control** — residual after a plain ``log Ql`` OLS, so the model-based extraction can
   be read against the model-free one.  Where 2 and 3 agree, the shape is not an artifact
   of which Ql correction was used.

Standing warnings that apply to every panel here, both already earned on this axis:
the ~51 Hz inverted-U in Свод ННО is a **failures-only collider** that halves and dies
when censored runs enter, and Свод ННО is failures-biased *by construction* (it does not
exist for running pumps).  Both denominators are therefore always emitted, and a shape
that appears only in ``failures`` should be treated as selection, not physics.
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
from matplotlib.ticker import MultipleLocator  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk.svod_nno_decomposition import (  # noqa: E402
    _bin_mean,
    load_panel,
)
from analysis.workflows.production_risk.svod_ttf_vs_ql import (  # noqa: E402
    DENSITY_VMAX_PCT,
    GROUPS,
    OVERLAY_SUBTITLE,
    OVERLAYS,
    QL_BOUNDS,
    VARIANT_ROWS,
    Y_CAP_BY_FIELD,
    Y_TICK_DAYS,
    QlFit,
    _annotate_multipliers,
    _clipped_share,
    _density,
    _fit,
    _panel,
    _ya_model,
)

SLUG = "production_risk_svod_ttf_vs_freq"

FREQ_REF = 50.0

#: «Частота» in the Свод sheet runs from 2.0 to 236 Hz, which no ESP drive does; those
#: rows are transcription noise.  Same window ``svod_nno_decomposition`` uses.
FREQ_BOUNDS = (30.0, 70.0)

#: Frequency ticks — the axis spans ~30 Hz, so 5 Hz steps stay readable.
FREQ_TICK_HZ = 5.0


# ── the Ql layer, as an AFT offset ────────────────────────────────────────────

def _ya_ql_offset(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(θ_Ql, β)`` per run under **Ya v2**.

    Ya's baseline is a two-component mixture, so there is no single shape parameter; the
    transfer work uses the **k1 component's** β (≈0.7875) and this follows it, so the two
    reconstructions stay comparable.
    """
    m = _ya_model()
    theta = np.asarray(m.theta_at("Ql", g["ql"].to_numpy(float)), dtype=float)
    return theta, np.full(len(g), float(m.baseline.beta1))


def _vt_ql_offset(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(θ_Ql, β₀)`` per run under **Vt v3.2**, both taken per sour stratum."""
    from analysis.workflows.production_risk import vt_composed_model as VC
    theta = np.full(len(g), np.nan)
    beta = np.full(len(g), np.nan)
    pos = {ix: i for i, ix in enumerate(g.index)}
    for stratum, part in g.groupby("h2s_class"):
        loc = [pos[ix] for ix in part.index]
        theta[loc] = VC.theta_ql(str(stratum), part["ql"].to_numpy(float))
        beta[loc] = VC.BASELINE[str(stratum)]["beta0"]
    return theta, beta


@lru_cache(maxsize=1)
def _vt_v4_model():
    """Vt v4, fitted once per process (``write=False`` keeps this out of v4's results)."""
    from analysis.workflows.production_risk import vt_v4 as V4
    return V4.run(write=False).model


@lru_cache(maxsize=1)
def _ya_v21_model():
    """Ya v2.1 — the nameplate-rate arm, Ya's counterpart to Vt v4."""
    from analysis.workflows.production_risk import ya_k1k2_hybrid as YA
    return YA.run(rate="Qnom", n_boot=0, write=False).model


def _ya_qnom_offset(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(θ_Qnom, β)`` per run under **Ya v2.1**."""
    m = _ya_v21_model()
    theta = np.asarray(m.theta_at("Qnom", g["qnom"].to_numpy(float)), dtype=float)
    return theta, np.full(len(g), float(m.baseline.beta1))


def _vt_qnom_offset(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(θ_Qnom, β₀)`` per run under **Vt v4**, both per sour stratum."""
    m = _vt_v4_model()
    theta = np.full(len(g), np.nan)
    beta = np.full(len(g), np.nan)
    pos = {ix: i for i, ix in enumerate(g.index)}
    for stratum, part in g.groupby("h2s_class"):
        loc = [pos[ix] for ix in part.index]
        theta[loc] = m.theta_qnom(str(stratum), part["qnom"].to_numpy(float))
        beta[loc] = m.baseline(str(stratum))[0]
    return theta, beta


#: Extraction mode → per-field ``(label, offset callable, rate column)``.  Two shipped
#: rate parameterisations, and the pair is the point:
#:
#: * ``ql``   — the realised-rate models: Ya **v2** and Vt **v3.2**.
#: * ``qnom`` — the nameplate-rate models: Ya **v2.1** and Vt **v4**.  These are the
#:   causally cleaner layer (the pump is chosen at install, so its nameplate cannot be
#:   contaminated by how the run ended) and both fields' own model selection prefers
#:   them, so a frequency shape that survives *this* extraction is the stronger claim.
#:
#: A patchable seam: every entry fits a model off the warehouse (Ya ~130 s each), so
#: tests substitute it rather than pay for it.
LAYERS = {
    "ql": {"Ya": ("Ya v2 θ_Ql", _ya_ql_offset, "ql"),
           "Vt": ("Vt v3.2 θ_Ql", _vt_ql_offset, "ql")},
    "qnom": {"Ya": ("Ya v2.1 θ_Qnom", _ya_qnom_offset, "qnom"),
             "Vt": ("Vt v4 θ_Qnom", _vt_qnom_offset, "qnom")},
}

#: Backwards-compatible alias for the original single-mode seam.
QL_LAYERS = LAYERS["ql"]


def extract_layer(g: pd.DataFrame, mode: str = "ql") -> pd.Series:
    """``t_adj = ННО · θ ** (1/β)`` — the run's life rescaled to the layer's reference.

    Monotone in ННО and strictly positive, so the adjusted panel is still a life in days
    and shares the raw panel's axis — which is what makes the before/after pair directly
    comparable.  A subtracted residual would not be.
    """
    theta, beta = LAYERS[mode][str(g["field"].iloc[0])][1](g)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = g["nno"].to_numpy(float) * theta ** (1.0 / beta)
    return pd.Series(adj, index=g.index, name="nno_adj")


def extract_ql_layer(g: pd.DataFrame) -> pd.Series:
    """Backwards-compatible alias for ``extract_layer(g, "ql")``."""
    return extract_layer(g, "ql")


def _log_rate_residual(g: pd.DataFrame, rate_col: str = "ql") -> pd.Series:
    """The model-free comparison: ННО minus its ``log rate`` OLS prediction."""
    y = g["nno"].to_numpy(float)
    design = np.column_stack([np.ones(len(g)), np.log(g[rate_col].to_numpy(float))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ beta, index=g.index, name="resid")


# ── build ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FreqPanel:
    name: str
    rows: pd.DataFrame
    before: pd.DataFrame
    after: pd.DataFrame
    residual: pd.DataFrame
    before_fit: QlFit | None
    after_fit: QlFit | None
    residual_fit: QlFit | None
    layer_label: str
    rate_col: str
    freq_median: float
    ref_before: float
    ref_after: float


def _quad_freq(freq: pd.Series, y: pd.Series):
    """``y ~ 1 + f + f²`` on raw Hz — frequency is not a log-scale covariate.

    The Ql fits in this family run in ``log Ql`` because every Ql layer in the workflow
    is per-e-fold.  Frequency is not: its layers are tents and polylines in Hz, and the
    open question on this axis has always been **curvature** (is there a valley), which a
    log transform would distort.  So this reuses the shared quadratic on raw Hz.
    """
    from analysis.workflows.production_risk.svod_nno_decomposition import quad_fit
    return quad_fit(freq, y)


def build(panel: pd.DataFrame, bins: int = 10, min_n: int = 40, mode: str = "ql"
          ) -> tuple[dict[str, FreqPanel], pd.DataFrame]:
    """Per-group freq curves; ``mode`` picks which shipped rate layer is extracted."""
    rate_col = next(iter(LAYERS[mode].values()))[2]
    out: dict[str, FreqPanel] = {}
    coverage = []
    for name, expr in GROUPS:
        present = panel.query(expr)
        g = present[present["freq"].notna() & present["freq"].between(*FREQ_BOUNDS)
                    & present[rate_col].notna()
                    & present[rate_col].between(*QL_BOUNDS)].copy()
        coverage.append({
            "group": name, "n_present": len(present), "n_in_bounds": len(g),
            "n_dropped": len(present) - len(g),
            "n_failures": int((g["event"] == 1).sum()),
            "n_other_pulls": int((g["event"] != 1).sum()),
            "freq_min": g["freq"].min() if len(g) else np.nan,
            "freq_max": g["freq"].max() if len(g) else np.nan,
            "median_nno": g["nno"].median() if len(g) else np.nan,
        })
        if len(g) < max(min_n, bins * 3):
            continue
        g["nno_adj"] = extract_layer(g, mode)
        g["resid"] = _log_rate_residual(g, rate_col)
        b = int(np.clip(len(g) // 30, 4, bins))
        before_fit = _quad_freq(g["freq"], g["nno"])
        after_fit = _quad_freq(g["freq"], g["nno_adj"])
        fmed = float(g["freq"].median())
        out[name] = FreqPanel(
            name=name, rows=g,
            before=_bin_mean(g["freq"], g["nno"], b),
            after=_bin_mean(g["freq"], g["nno_adj"], b),
            residual=_bin_mean(g["freq"], g["resid"], b),
            before_fit=before_fit, after_fit=after_fit,
            residual_fit=_quad_freq(g["freq"], g["resid"]),
            layer_label=LAYERS[mode][str(g["field"].iloc[0])][0],
            rate_col=rate_col,
            freq_median=fmed,
            # Each panel is referenced to ITS OWN level at the median frequency, so the
            # before/after multipliers both read as pure shape — the extraction changes
            # the level as well as the shape, and a shared reference would mix the two.
            ref_before=float(g.loc[g["freq"].between(fmed - 2, fmed + 2), "nno"].mean()),
            ref_after=float(g.loc[g["freq"].between(fmed - 2, fmed + 2), "nno_adj"].mean()),
        )
    return out, pd.DataFrame(coverage)


def fit_table(groups: dict[str, FreqPanel]) -> pd.DataFrame:
    rows = []
    for name, gp in groups.items():
        for kind, f in (("before", gp.before_fit), ("after_ql_extracted", gp.after_fit),
                        ("log_rate_residual", gp.residual_fit)):
            if f is None:
                continue
            rows.append({
                "group": name, "panel": kind, "layer": gp.layer_label,
                "rate_col": gp.rate_col, "n": f.n,
                "beta2": f.beta2, "se_beta2": f.se_beta2, "p_beta2": f.p_beta2,
                "vertex_hz": f.vertex, "p_inverted_u": f.p_inverted_u,
                "vertex_lo": f.vertex_lo, "vertex_hi": f.vertex_hi,
                "freq_median": gp.freq_median,
                "ref_before": gp.ref_before, "ref_after": gp.ref_after,
            })
    return pd.DataFrame(rows)


def _ylims(gp: FreqPanel, overlay: str):
    """Days panels share the field cap; the OLS residual gets ±half of it."""
    cap = Y_CAP_BY_FIELD.get(str(gp.rows["field"].iloc[0]))
    if cap is not None:
        return (0.0, cap), (0.0, cap), (-cap / 2.0, cap / 2.0)

    def span(value_col: str, frame: pd.DataFrame):
        if overlay != "none":
            lo, hi = np.nanpercentile(gp.rows[value_col].to_numpy(float), [1, 99])
        else:
            lo = float((frame["ymean"] - frame["yse"]).min())
            hi = float((frame["ymean"] + frame["yse"]).max())
        pad = 0.08 * (hi - lo) or 1.0
        return lo - pad, hi + pad

    return (span("nno", gp.before), span("nno_adj", gp.after),
            span("resid", gp.residual))


def plot(groups: dict[str, FreqPanel], out_path: Path, *, suptitle: str,
         overlay: str = "none") -> Path:
    if overlay not in OVERLAYS:
        raise ValueError(f"overlay must be one of {OVERLAYS}, got {overlay!r}")
    limits = {k: _ylims(gp, overlay) for k, gp in groups.items()}
    xlo = min(float(gp.rows["freq"].min()) for gp in groups.values())
    xhi = max(float(gp.rows["freq"].max()) for gp in groups.values())
    xpad = 0.03 * (xhi - xlo)
    xlim = (xlo - xpad, xhi + xpad)

    vmax = None
    if overlay == "heatmap":
        cells = []
        for k, gp in groups.items():
            for value_col, ylim in zip(("nno", "nno_adj", "resid"), limits[k]):
                z = _density(gp.rows, "freq", value_col, ylim)[2]
                cells.append(z[np.isfinite(z)])
        allc = np.concatenate(cells)
        vmax = float(np.percentile(allc, DENSITY_VMAX_PCT)) if allc.size else 1.0

    fig, axes = plt.subplots(len(groups), 3, figsize=(18.0, 4.3 * len(groups)),
                             squeeze=False)
    mesh = None
    for r, (name, gp) in enumerate(groups.items()):
        pts = gp.rows if overlay != "none" else None
        blim, alim, rlim = limits[name]
        specs = (
            (gp.before, gp.before_fit, "#E8762C", "ННО (days)",
             f"{name}: BEFORE — raw TTF vs freq", False, "nno", blim),
            (gp.after, gp.after_fit, "#1F6F45", "ННО adjusted (days)",
             f"{name}: AFTER — {gp.layer_label} removed (t·θ^(1/β))", False,
             "nno_adj", alim),
            (gp.residual, gp.residual_fit, "#5B4B9E", "residual ННО (days)",
             f"{name}: control — residual after log {gp.rate_col.capitalize()} (OLS)",
             True, "resid", rlim),
        )
        for c, (frame, fit, colour, ylab, title, zero, value_col, lim) in enumerate(specs):
            got = _panel(axes[r][c], frame, fit, color=colour, ylabel=ylab, title=title,
                         zero_line=zero, rows=pts, value_col=value_col,
                         rate_col="freq", xlabel="running frequency, Гц",
                         overlay=overlay, ylim=lim, vmax=vmax)
            mesh = got or mesh
        for ax, lim, value_col in zip(axes[r], (blim, alim, rlim),
                                      ("nno", "nno_adj", "resid")):
            ax.set_ylim(*lim)
            ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_DAYS))
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(MultipleLocator(FREQ_TICK_HZ))
            ax.axvline(FREQ_REF, color="grey", linestyle=":", linewidth=1.2, zorder=2)
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
        cb.set_label("share of the frequency column's runs (each column sums to 1)",
                     fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for r, gp in enumerate(groups.values()):
        for c, (frame, ref) in enumerate(((gp.before, gp.ref_before),
                                          (gp.after, gp.ref_after),
                                          (gp.residual, gp.ref_before))):
            if not (np.isfinite(ref) and ref > 0):
                continue
            _annotate_multipliers(axes[r][c], frame, ref,
                                  zero_is_reference=(c == 2), renderer=renderer)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    prediction_workbook_path: Path | None = None,
    variants: tuple[str, ...] = ("failures", "all_closed"),
    modes: tuple[str, ...] = ("ql", "qnom"),
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
        for mode in modes:
            groups, coverage = build(panel, bins=bins, mode=mode)
            coverage.insert(0, "variant", variant)
            coverage.insert(1, "mode", mode)
            stem = f"{variant}__{mode}"
            coverage.to_csv(tables / f"{stem}__freq_coverage.csv", index=False)
            if not groups:
                continue
            fit_table(groups).to_csv(tables / f"{stem}__freq_fits.csv", index=False)
            for name, gp in groups.items():
                slug = name.replace(" ", "_").replace("(", "").replace(")", "")
                gp.before.to_csv(tables / f"{stem}__before_{slug}.csv", index=False)
                gp.after.to_csv(tables / f"{stem}__after_{slug}.csv", index=False)
                gp.residual.to_csv(
                    tables / f"{stem}__log_rate_residual_{slug}.csv", index=False)
            labels = " / ".join(v[0] for v in LAYERS[mode].values())
            for overlay in overlays:
                plot(
                    groups,
                    figures / f"ttf_vs_freq__{stem}"
                    f"{'' if overlay == 'none' else '_' + overlay}.png",
                    suptitle=(f"Свод TTF (ННО) vs frequency, before and after the "
                              f"fitted rate layer is removed — {labels}, {rows}"
                              + OVERLAY_SUBTITLE[overlay]),
                    overlay=overlay,
                )
            summary.append({"variant": variant, "mode": mode, "rows": rows,
                            "n_groups": len(groups)})
    pd.DataFrame(summary).to_csv(tables / "panel_summary.csv", index=False)
    return out
