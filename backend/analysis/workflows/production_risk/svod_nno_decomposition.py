"""Свод ННО vs operating setpoint: marginal shape vs what survives field + Ql.

Reproduces (and fixes) the 2026-07-24 ``production_risk_svod_nno_decomposition``
figures.  Three candidate x-axes, all read straight off the Свод sheet:

    Kpod  = «Дебит жидк.» / «Ном. Произв. м₃/сут»   (Ql / Qnom)
    Delta = «Дельта Дебита Ж и номинала, м3/сут»    (Ql − Qnom)
    freq  = «Частота»                               (Hz)

Each gets two panels:

  * **marginal** — mean ННО (days) per decile of x.  This is the shape the slides
    show, and it is confounded: fields differ in both setpoint and life, and Ql
    drives life on its own (HR 0.314/e-fold, p=1e-11 — see the Cox notes).
  * **residual** — mean ННО (days) per decile of x *after* regressing ННО on field
    dummies + log Ql.  What is left is the part of the setpoint→life relation that
    the field label and the rate do not already explain.

Two deliberate choices, both differing from the ad-hoc 2026-07-24 run:

1. **Residuals are in days, not log-days** (user request 2026-07-24).  The response
   is fitted on its natural scale, so a residual of −40 reads "40 days shorter than
   this field at this rate would predict".  The Ql control stays ``log Ql`` — only
   the response scale changed, so the two runs isolate one difference.  The
   log-response variant is still available via ``log_response=True``.
2. **Marginals are built from genuine failures only** by default (user request).
   ``esp_population.classify`` decides: ГТМ/ППР pulls are workovers, not failures,
   and a closed run with no failure signal is not one either.  ННО on a workover
   pull measures when the crew arrived, not when the pump died, so pooling them
   flattens every curve.  Pass ``variant="all_closed"`` for the old denominator.

The Мирнинский installs-2024+ cohort rule applies (CLAUDE.md §5) — Mc rows before
``config.MC_INSTALL_COHORT_START`` are dropped unless ``mc_cohort=False``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import config as C  # noqa: E402
from analysis.workflows.production_risk import crosswalk, esp_population  # noqa: E402

SLUG = "production_risk_svod_nno_decomposition"

COL_NNO = "Наработка (сут)"
COL_QL = "Дебит жидк."
COL_QNOM = "Ном. Произв. м₃/сут"
COL_DELTA = "Дельта Дебита Ж и номинала, м3/сут"
COL_FREQ = "Частота"

#: x-axis, column, reference line, axis label, physical bounds (None = keep all).
#: «Частота» is dirty — it runs from 2.0 to 236 Hz, which no ESP drive does; those
#: rows are transcription noise, not measurements, and one 236 Hz row drags a whole
#: decile.  Dropped rows are counted into ``x_coverage.csv``, never silently.
VARIABLES = (
    ("kpod", "kpod", 0.8, "Kpod = Ql / Qnom", None),
    ("delta", "delta", 0.0, "Delta = Ql − Qnom (m3/d)", None),
    ("freq", "freq", 50.0, "freq (Hz)", (30.0, 70.0)),
)


@dataclass(frozen=True)
class QuadFit:
    """OLS ``y ~ 1 + x + x²`` on the *unbinned* rows behind a panel.

    ``beta2 < 0`` is an inverted-U (a peak); ``vertex`` is where that peak sits.
    The vertex is only meaningful when ``beta2`` is both negative and resolved —
    read ``p_beta2`` and ``p_inverted_u`` (the bootstrap share of draws that curved
    down) before quoting it.
    """

    n: int
    beta2: float
    se_beta2: float
    p_beta2: float
    vertex: float
    p_inverted_u: float
    vertex_lo: float
    vertex_hi: float
    grid_x: np.ndarray
    grid_y: np.ndarray

    @property
    def label(self) -> str:
        peak = f"peak {self.vertex:.1f}" if self.beta2 < 0 else "no peak"
        return f"β₂={self.beta2:+.2f} (p={self.p_beta2:.3f}), {peak}, n={self.n}"

    @property
    def p_headline(self) -> float:
        """The p-value a plot should judge this fit by — here the curvature's.

        Shared plotting (``svod_ttf_vs_ql._panel``) draws a fit solid when resolved and
        dashed-grey when not.  It must not reach for a class-specific field name: a
        ``QuadFit`` has no ``p1``, and on this fit the headline claim is the curvature.
        """
        return self.p_beta2


@dataclass(frozen=True)
class Decomposition:
    """One x-variable: its binned marginal, its binned residual, the fit summary."""

    name: str
    marginal: pd.DataFrame
    residual: pd.DataFrame
    ref: float
    xlabel: str
    fit: QuadFit | None = None


# ── panel ─────────────────────────────────────────────────────────────────────

#: ``field`` for a well code no prefix map claims.  Such rows are in the sheet but in no
#: field, so they need a name of their own in the selection funnel — dropping them
#: silently is what makes a panel count look like it came out of nowhere.
NO_FIELD = "—"


def _selection_funnel(out: pd.DataFrame, stages: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    """Rows surviving each stage, cumulative, split by ``(field, h2s_class)``.

    Split rather than totalled because the caller's population may be a field, a Vt
    sour/nonsour half or the whole fleet, and only the caller knows which — summing the
    wrong axis is exactly the mistake this table exists to prevent.  ``stages`` is the
    real mask sequence :func:`load_panel` applies, so the funnel cannot drift from the
    filter: the same masks produce both.
    """
    field = out["field"].fillna(NO_FIELD)
    keep = pd.Series(True, index=out.index)
    parts = []
    for step, (label, mask) in enumerate(stages):
        keep = keep & mask.reindex(out.index).fillna(False).astype(bool)
        g = (pd.DataFrame({"field": field[keep], "h2s_class": out.loc[keep, "h2s_class"]})
             .groupby(["field", "h2s_class"], dropna=False).size().rename("kept")
             .reset_index())
        parts.append(g.assign(step=step, stage=label))
    return pd.concat(parts, ignore_index=True)[["step", "stage", "field", "h2s_class", "kept"]]


def selection_funnel(
    prediction_workbook_path: Path | None = None,
    variant: str = "failures",
    mc_cohort: bool = True,
) -> pd.DataFrame:
    """The per-stage survivor counts behind :func:`load_panel`, tidy by field and H₂S class."""
    panel = load_panel(prediction_workbook_path, variant=variant, mc_cohort=mc_cohort)
    return panel.attrs["funnel"]


def load_panel(
    prediction_workbook_path: Path | None = None,
    variant: str = "failures",
    mc_cohort: bool = True,
) -> pd.DataFrame:
    """One row per closed Свод run with a reported ННО and a usable setpoint.

    ``variant`` is ``"failures"`` (genuine failures only — ГТМ/ППР and no-signal
    pulls dropped) or ``"all_closed"`` (every closed run, the old denominator).

    Four filters stand between the sheet and this frame, and on Ya they together drop more
    than half of it, so the panel carries its own accounting: ``attrs["funnel"]`` (also
    reachable as :func:`selection_funnel`) counts what each stage removed, per field.
    """
    if variant not in {"failures", "all_closed"}:
        raise ValueError(f"variant must be 'failures' or 'all_closed', got {variant!r}")

    src = Path(prediction_workbook_path or crosswalk.resolve_prediction_workbook_path())
    sv = pd.read_excel(src, sheet_name="Свод")

    num = lambda col: pd.to_numeric(sv[col], errors="coerce")  # noqa: E731
    out = pd.DataFrame(index=sv.index)
    out["code"] = sv["Скв."].map(crosswalk.norm_well)
    out["install"] = pd.to_datetime(sv["Дата монтажа"], errors="coerce")
    # Run end, for callers that need to bound a per-run daily window (``svod_ttf_vs_ql``).
    # «Дата остановки» is when the pump stopped; «Дата демонтажа» (the pull) can trail it
    # by weeks of standing idle, so the stop date is the right bound and the pull date
    # only fills in where it is missing.
    out["end"] = pd.to_datetime(sv["Дата остановки"], errors="coerce").fillna(
        pd.to_datetime(sv["Дата демонтажа"], errors="coerce"))
    out["field"] = out["code"].map(esp_population._model_field)
    # brt / slb / oth — the same grouping the Ya and Vt models carry a hazard ratio for.
    out["contractor"] = sv["Принадлежность"].map(esp_population.contractor_group)
    out["nno"] = num(COL_NNO)
    out["ql"] = num(COL_QL)
    out["qnom"] = num(COL_QNOM)
    out["delta"] = num(COL_DELTA)
    out["freq"] = num(COL_FREQ)

    # Same event definition as every other fit in this workflow.
    has_unit = ~sv["Отказавший узел"].map(esp_population._norm).str.casefold().isin({"", "нет"})
    out["event"] = [
        esp_population.classify(r, u, gtm_is_failure=False)
        for r, u in zip(sv["Причина остановки"], has_unit)
    ]
    # «Флаг отказа» = -1 is an unresolved outcome — neither event nor censoring.
    flag_ok = (pd.to_numeric(sv["Флаг отказа"], errors="coerce") != -1
               if "Флаг отказа" in sv.columns else pd.Series(True, index=sv.index))

    # v3.2 sour class: the flag is written only from the failure/workover DB, so it
    # is rolled up to WELL level and applied to every run of that well — the
    # ``SOUR_WELL_LEVEL_ALL_RUNS`` relabel the Vt hybrid deploys on.  Vt only; the
    # other fields carry no «Кислый/Некислый» meaning.  The roll-up sees only rows that
    # cleared the flag filter, as it always has: an unresolved row is not evidence.
    run_sour = sv["Кислый/Некислый"].map(crosswalk.sour_group)
    well_sour = (
        pd.DataFrame({"code": out["code"], "s": run_sour})[flag_ok]
        .groupby("code")["s"].agg(lambda s: "sour" if (s == "sour").any() else "nonsour")
    )
    out["h2s_class"] = np.where(
        out["field"] == "Vt", out["code"].map(well_sour).fillna("nonsour"), "nonsour")

    # Every filter as a named mask over the WHOLE sheet, applied in one step at the end.
    # The funnel is built from this very list, so what the app reports as dropped and what
    # is actually dropped are the same computation.
    stages: list[tuple[str, pd.Series]] = [
        ("строк на листе «Свод»", pd.Series(True, index=sv.index)),
        ("«Флаг отказа» ≠ −1 (исход разобран)", flag_ok),
        ("месторождение распознано по коду скважины", out["field"].notna()),
        # разнесено на два шага: работающий насос ННО ещё не имеет, а поднятый может не
        # иметь ставки — это разные потери, и сливать их в одну строку значит прятать,
        # что панель заодно выбрасывает часть уже отказавших пусков
        ("подъём состоялся, ННО заполнено (> 0)", out["nno"] > 0),
        ("«Дебит жидк.» и «Ном. Произв.» заполнены и > 0",
         (out["ql"] > 0) & (out["qnom"] > 0)),
    ]
    if mc_cohort:
        stages.append((f"Мирнинский: монтажи с "
                       f"{pd.Timestamp(C.MC_INSTALL_COHORT_START):%d.%m.%Y} "
                       f"(на других месторождениях не режет)",
                       esp_population.mc_cohort_mask(out)))
    if variant == "failures":
        stages.append(("настоящий отказ: не ГТМ/ППР и не подъём без признака отказа",
                       out["event"] == 1))
    funnel = _selection_funnel(out, stages)

    keep = pd.Series(True, index=sv.index)
    for _, mask in stages:
        keep &= mask.reindex(sv.index).fillna(False).astype(bool)
    out = out[keep].reset_index(drop=True)

    out["kpod"] = out["ql"] / out["qnom"]
    out["log_ql"] = np.log(out["ql"])
    out["log_qnom"] = np.log(out["qnom"])   # pump size, the alternative rate control
    out.attrs["funnel"] = funnel
    return out


# ── fit ───────────────────────────────────────────────────────────────────────

#: Label for the two control sets, used in titles and output filenames.
CONTROL_LABEL = {False: "field + log Ql", True: "field + log Ql + freq (linear+quad)"}

#: Continuous controls whose linear effect is regressed out before the residual.
#: The default holds the RATE fixed (log Ql); the ``qnom`` set holds PUMP SIZE fixed
#: instead — the recent read is that the "rate hazard" tracks Qnom, not Ql, so
#: cleaning on Qnom asks whether freq/Kpod carry anything once size is out.
CONTROL_SETS = {
    "ql": ("log_ql",),
    "qnom": ("log_qnom",),
}
CONTROL_SET_LABEL = {"ql": "field + log Ql", "qnom": "field + log Qnom (pump size)"}


def freq_supported(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows whose frequency is present and physically plausible.

    Required before ``with_freq`` residuals: a control you cannot measure on a row
    cannot be removed from it, so those rows leave the panel rather than silently
    entering with freq imputed to the mean.
    """
    bounds = next(v[4] for v in VARIABLES if v[0] == "freq")
    return panel[panel["freq"].notna() & panel["freq"].between(*bounds)]


def field_residuals(panel: pd.DataFrame, controls: tuple[str, ...] = ("log_ql",),
                    log_response: bool = False, with_freq: bool = False) -> pd.Series:
    """ННО residuals after field dummies + ``controls`` (+ optionally freq).

    Days by default: ``resid = nno − E[nno | field, controls]``.  ``controls`` names
    the continuous columns held fixed — ``("log_ql",)`` removes the rate,
    ``("log_qnom",)`` removes pump size instead (see :data:`CONTROL_SETS`).

    ``with_freq`` additionally removes frequency as **linear + quadratic** (the
    curvature found on the freq axis, not just a trend); requires
    :func:`freq_supported` rows only.
    """
    nno = panel["nno"].to_numpy(float)
    if log_response and not (nno > 0).all():
        # ``load_panel`` filters these out; a caller passing its own frame would
        # otherwise get a silent all-NaN residual out of np.log.
        raise ValueError("log_response needs strictly positive ННО")
    y = np.log(nno) if log_response else nno
    dummies = pd.get_dummies(panel["field"], prefix="f", drop_first=True, dtype=float)
    cols = [np.ones(len(panel))] + [panel[c].to_numpy(float) for c in controls]
    if with_freq:
        f = panel["freq"].to_numpy(float)
        if not np.isfinite(f).all():
            raise ValueError("with_freq needs a freq on every row — use freq_supported()")
        fc = f - f.mean()
        cols += [fc, fc**2]
    design = np.column_stack([*cols, dummies.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ beta, index=panel.index, name="resid")


def field_ql_residuals(panel: pd.DataFrame, log_response: bool = False,
                       with_freq: bool = False) -> pd.Series:
    """Backwards-compatible alias: residuals controlling field + log Ql."""
    return field_residuals(panel, controls=("log_ql",),
                           log_response=log_response, with_freq=with_freq)


def quad_fit(x: pd.Series, y: pd.Series, n_boot: int = 2000, seed: int = 0) -> QuadFit | None:
    """Fit ``y ~ 1 + x + x²`` (x centred) and bootstrap the peak location.

    Centring keeps the design conditioned; it does not change β₂ or the vertex.
    ``p_inverted_u`` is the share of bootstrap draws with β₂ < 0 — the honest way
    to say "is there a peak at all", since the vertex of an upward-curving fit is
    a minimum, not a maximum, and averaging the two is meaningless.
    """
    from scipy import stats

    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    n = len(xv)
    if n < 30:
        return None
    xc = xv - xv.mean()
    design = np.column_stack([np.ones(n), xc, xc**2])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    sigma2 = resid @ resid / (n - 3)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    p2 = float(2 * stats.t.sf(abs(beta[2] / se[2]), n - 3))

    rng = np.random.default_rng(seed)
    peaks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = xv[idx], yv[idx]
        xbc = xb - xb.mean()
        bb, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), xbc, xbc**2]), yb, rcond=None)
        if bb[2] < 0:
            peaks.append(xb.mean() - bb[1] / (2 * bb[2]))
    peaks_arr = np.array(peaks) if peaks else np.array([np.nan])

    grid = np.linspace(xv.min(), xv.max(), 200)
    gc = grid - xv.mean()
    return QuadFit(
        n=n,
        beta2=float(beta[2]),
        se_beta2=float(se[2]),
        p_beta2=p2,
        vertex=float(xv.mean() - beta[1] / (2 * beta[2])) if beta[2] != 0 else float("nan"),
        p_inverted_u=len(peaks) / n_boot,
        vertex_lo=float(np.nanpercentile(peaks_arr, 2.5)),
        vertex_hi=float(np.nanpercentile(peaks_arr, 97.5)),
        grid_x=grid,
        grid_y=beta[0] + beta[1] * gc + beta[2] * gc**2,
    )


def _bin_mean(x: pd.Series, y: pd.Series, bins: int) -> pd.DataFrame:
    b = pd.qcut(x, bins, duplicates="drop")
    g = pd.DataFrame({"b": b, "x": x, "y": y}).groupby("b", observed=True)
    out = g.agg(xmid=("x", "mean"), ymean=("y", "mean"), ysd=("y", "std"), n=("y", "size"))
    out["yse"] = out["ysd"] / np.sqrt(out["n"])
    return out.reset_index()


def decompose(
    panel: pd.DataFrame,
    bins: int = 10,
    log_response: bool = False,
    with_freq: bool = False,
    controls: tuple[str, ...] = ("log_ql",),
) -> tuple[dict[str, Decomposition], pd.DataFrame]:
    """Binned marginal + residual per x-variable, and the per-x coverage table.

    The residual fit is estimated **once, on the whole panel** — dropping an
    out-of-range frequency should not move the Kpod residual.
    """
    resid = field_residuals(panel, controls=controls, log_response=log_response,
                            with_freq=with_freq)
    out: dict[str, Decomposition] = {}
    coverage = []
    for name, col, ref, xlabel, bounds in VARIABLES:
        present = panel[panel[col].notna()]
        sub = present
        if bounds is not None:
            lo, hi = bounds
            sub = present[present[col].between(lo, hi)]
        coverage.append({
            "x": name,
            "n_panel": len(panel),
            "n_present": len(present),
            "n_in_bounds": len(sub),
            "n_dropped_missing": len(panel) - len(present),
            "n_dropped_out_of_bounds": len(present) - len(sub),
            "bounds": "" if bounds is None else f"[{bounds[0]}, {bounds[1]}]",
        })
        if len(sub) < bins * 3:
            continue
        out[name] = Decomposition(
            name=name,
            marginal=_bin_mean(sub[col], sub["nno"], bins),
            residual=_bin_mean(sub[col], resid.loc[sub.index], bins),
            ref=ref,
            xlabel=xlabel,
            # Fitted on the unbinned residuals — the bins are for the eye, the
            # curvature test must not depend on where the decile edges fell.
            fit=quad_fit(sub[col], resid.loc[sub.index]),
        )
    return out, pd.DataFrame(coverage)


def by_group(
    panel: pd.DataFrame,
    var: str = "freq",
    group_col: str = "field",
    min_n: int = 80,
    log_response: bool = False,
    label_prefix: str = "",
    with_freq: bool = False,
    controls: tuple[str, ...] = ("log_ql",),
) -> dict[str, Decomposition]:
    """The same decomposition split by ``group_col`` — does the pooled shape replicate?

    Residuals come from the **pooled** field+controls fit, so each group's panel is a
    literal slice of the pooled figure rather than a separate model.  Groups with
    fewer than ``min_n`` rows are dropped: a quadratic on 40 points will always
    find some curvature.  Returned in descending sample size.
    """
    _, col, ref, xlabel, bounds = next(v for v in VARIABLES if v[0] == var)
    resid = field_residuals(panel, controls=controls, log_response=log_response,
                            with_freq=with_freq)
    sub = panel[panel[col].notna()]
    if bounds is not None:
        sub = sub[sub[col].between(*bounds)]

    out: dict[str, Decomposition] = {}
    for key, g in sub.groupby(group_col):
        if len(g) < min_n:
            continue
        label = f"{label_prefix}{key}"
        bins = int(np.clip(len(g) // 30, 4, 10))
        out[label] = Decomposition(
            name=label,
            marginal=_bin_mean(g[col], g["nno"], bins),
            residual=_bin_mean(g[col], resid.loc[g.index], bins),
            ref=ref,
            xlabel=xlabel,
            fit=quad_fit(g[col], resid.loc[g.index]),
        )
    return dict(sorted(out.items(), key=lambda kv: -(kv[1].fit.n if kv[1].fit else 0)))


def by_field(panel: pd.DataFrame, var: str = "freq", min_n: int = 80,
             log_response: bool = False, with_freq: bool = False,
             controls: tuple[str, ...] = ("log_ql",)) -> dict[str, Decomposition]:
    return by_group(panel, var=var, group_col="field", min_n=min_n,
                    log_response=log_response, with_freq=with_freq, controls=controls)


def by_vt_stratum(panel: pd.DataFrame, var: str = "freq", min_n: int = 60,
                  log_response: bool = False) -> dict[str, Decomposition]:
    """Vt split sour/nonsour on the **v3.2 well-level relabel**.

    Caveat worth carrying into any reading of these panels: v3.2's headline gain was
    recovering 30 *running* pumps into Vt_sour, and a ННО panel cannot contain a
    running pump at all.  What the relabel does here is narrower — it rolls the flag
    onto closed runs of a sour well that never got stamped individually.
    """
    return by_group(panel[panel["field"] == "Vt"], var=var, group_col="h2s_class",
                    min_n=min_n, log_response=log_response, label_prefix="Vt_")


def fit_table(decomp: dict[str, Decomposition]) -> pd.DataFrame:
    rows = []
    for key, d in decomp.items():
        f = d.fit
        if f is None:
            continue
        rows.append({
            "key": key, "n": f.n, "beta2": f.beta2, "se_beta2": f.se_beta2,
            "p_beta2": f.p_beta2, "vertex": f.vertex,
            "p_inverted_u": f.p_inverted_u,
            "vertex_lo": f.vertex_lo, "vertex_hi": f.vertex_hi,
        })
    return pd.DataFrame(rows)


# ── figure ────────────────────────────────────────────────────────────────────

#: A fit whose curvature is this weak is drawn dashed and grey — the overlay must
#: not lend a shape more authority than its own p-value.
RESOLVED_P = 0.05


def _panel(ax, frame: pd.DataFrame, *, color: str, ref: float, xlabel: str,
           ylabel: str, title: str, zero_line: bool, annotate: bool,
           fit: QuadFit | None = None) -> None:
    ax.errorbar(frame["xmid"], frame["ymean"], yerr=frame["yse"],
                marker="o", linestyle="-", color=color, capsize=3, linewidth=1.6,
                zorder=3)
    if fit is not None:
        resolved = fit.p_beta2 < RESOLVED_P
        # Clip the curve to the plotted bin range.  The fit spans the full data
        # range, which reaches past the outer bin midpoints — drawn unclipped it
        # both widens the x-axis away from the marginal panel and drags the y-axis
        # down to fitted values nobody is looking at.
        span = (fit.grid_x >= frame["xmid"].min()) & (fit.grid_x <= frame["xmid"].max())
        ax.plot(fit.grid_x[span], fit.grid_y[span],
                color="#333333" if resolved else "#999999",
                linestyle="-" if resolved else "--", linewidth=1.4, zorder=2,
                label=fit.label)
        if resolved and fit.beta2 < 0:
            ax.axvspan(fit.vertex_lo, fit.vertex_hi, color="#333333", alpha=0.07, zorder=1)
            ax.axvline(fit.vertex, color="#333333", linewidth=1.0, alpha=0.6, zorder=2)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.85)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(ref, color="grey", linestyle=":", linewidth=1.2)
    if annotate:
        for _, r in frame.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["xmid"], r["ymean"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)


def plot(
    decomp: dict[str, Decomposition],
    out_path: Path,
    *,
    suptitle: str,
    resid_ylabel: str,
    control_label: str = CONTROL_LABEL[False],
) -> Path:
    names = list(decomp)
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 4.2 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        d = decomp[name]
        _panel(axes[row][0], d.marginal, color="#E8762C", ref=d.ref, xlabel=d.xlabel,
               ylabel="mean ННО (days)", title=f"{name}: MARGINAL mean ННО",
               zero_line=False, annotate=True)
        _panel(axes[row][1], d.residual, color="#2CA05A", ref=d.ref, xlabel=d.xlabel,
               ylabel=resid_ylabel, title=f"{name}: RESIDUAL (after {control_label})",
               zero_line=True, annotate=False, fit=d.fit)
        # Marginal and residual share the same bins, so clipping the overlay is
        # enough to align them; this only guards against float drift.
        axes[row][1].set_xlim(axes[row][0].get_xlim())
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


#: A page taller than this many group-rows is split across multiple files — each
#: row is MARGINAL|RESIDUAL at ~4.3in, and past 3 rows the whole image scales down
#: to fit a screen and every panel gets cramped.
MAX_ROWS_PER_FIGURE = 3


def plot_groups(
    per_field: dict[str, Decomposition],
    pooled: Decomposition | None,
    out_path: Path,
    *,
    suptitle: str,
    resid_ylabel: str,
) -> list[Path]:
    """Pooled group first, then MARGINAL|RESIDUAL for each group on shared axes.

    Shared limits are the point: a per-group curve that looks dramatic on its own
    axis usually turns out to be small next to the pooled one.  Marginals share one
    y-scale and residuals another — they are different quantities in the same unit.

    A page over :data:`MAX_ROWS_PER_FIGURE` group-rows is divided into the fewest
    equal-sized ``…__pXofN.png`` parts that bring each part within the cap.  POOLED
    is the reference row and leads part 1; axis limits are computed over ALL groups
    so the parts stay directly comparable.
    """
    all_items = ([("POOLED", pooled)] if pooled is not None else []) + list(per_field.items())

    def _limits(attr):
        lo = min(float((getattr(d, attr)["ymean"] - getattr(d, attr)["yse"]).min())
                 for _, d in all_items)
        hi = max(float((getattr(d, attr)["ymean"] + getattr(d, attr)["yse"]).max())
                 for _, d in all_items)
        pad = 0.08 * (hi - lo) or 1.0
        return lo - pad, hi + pad

    mlim, rlim = _limits("marginal"), _limits("residual")
    xlo = min(float(d.residual["xmid"].min()) for _, d in all_items)
    xhi = max(float(d.residual["xmid"].max()) for _, d in all_items)
    xpad = 0.05 * (xhi - xlo)
    xlim = (xlo - xpad, xhi + xpad)

    n_parts = int(np.ceil(len(all_items) / MAX_ROWS_PER_FIGURE))
    size = int(np.ceil(len(all_items) / n_parts))
    chunks = [all_items[i:i + size] for i in range(0, len(all_items), size)]

    written: list[Path] = []
    for part, items in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(items), 2, figsize=(11.6, 4.3 * len(items)),
                                 squeeze=False)
        for r, (key, d) in enumerate(items):
            am, ar = axes[r][0], axes[r][1]
            _panel(am, d.marginal, color="#E8762C" if key != "POOLED" else "#B9541A",
                   ref=d.ref, xlabel=d.xlabel, ylabel="mean ННО (days)",
                   title=f"{key}: MARGINAL mean ННО", zero_line=False, annotate=True)
            _panel(ar, d.residual, color="#2CA05A" if key != "POOLED" else "#1F6F45",
                   ref=d.ref, xlabel=d.xlabel, ylabel=resid_ylabel,
                   title=f"{key}: RESIDUAL", zero_line=True, annotate=False, fit=d.fit)
            am.set_ylim(*mlim); ar.set_ylim(*rlim)
            am.set_xlim(*xlim); ar.set_xlim(*xlim)
        title = suptitle if n_parts == 1 else f"{suptitle}  [{part}/{n_parts}]"
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        dest = out_path if n_parts == 1 else out_path.with_name(
            f"{out_path.stem}__p{part}of{n_parts}{out_path.suffix}")
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(dest)
    return written


def run_qnom_controlled(
    prediction_workbook_path: Path | None = None,
    variants: tuple[str, ...] = ("failures", "all_closed"),
    x_vars: tuple[str, ...] = ("freq", "kpod"),
    show_fields: tuple[str, ...] = ("Ya", "Vt"),
    bins: int = 10,
    log_response: bool = False,
    mc_cohort: bool = True,
    run_date: str | None = None,
) -> Path:
    """TTF residuals cleaned of **field + Qnom (pump size)**, plotted vs freq & Kpod.

    The recent read (2026-07-24) is that the rate hazard tracks **Qnom, not Ql** —
    Kpod's apparent effect is the pump-size direction in disguise.  So this holds
    pump size (and field) fixed and asks the sharper question: with size removed,
    does frequency or Kpod carry any residual TTF signal?  Residuals come from the
    **pooled** field+Qnom fit; Ya and Vt are shown as slices of it (plus POOLED).

    ``variants`` = ``"failures"`` (genuine failures) and/or ``"all_closed"`` (every
    pull).  One figure per (variant × x-var), each with POOLED + the ``show_fields``.
    """
    out = results_dir(SLUG, run_date)
    tables, figures = out / "tables", out / "figures"
    scale = "log-days" if log_response else "days"
    resid_ylabel = "residual log ННО" if log_response else "residual ННО (days)"
    controls = CONTROL_SETS["qnom"]
    csuffix = "log" if log_response else "days"

    for variant in variants:
        panel = load_panel(prediction_workbook_path, variant=variant, mc_cohort=mc_cohort)
        rows = "genuine failures" if variant == "failures" else "all pulls"
        tag = f"{variant}_{csuffix}_qnomctl"
        pooled, _ = decompose(panel, bins=bins, log_response=log_response, controls=controls)
        for var in x_vars:
            per_field = by_field(panel, var=var, log_response=log_response, controls=controls)
            keep = {k: per_field[k] for k in show_fields if k in per_field}
            if not keep:
                continue
            fit_table({**({var + "_POOLED": pooled[var]} if var in pooled else {}), **keep}
                      ).to_csv(tables / f"{tag}__{var}_fits.csv", index=False)
            for key, d in keep.items():
                d.marginal.to_csv(tables / f"{tag}__{var}_marginal_{key}.csv", index=False)
                d.residual.to_csv(tables / f"{tag}__{var}_residual_{key}.csv", index=False)
            plot_groups(
                keep,
                pooled.get(var),
                figures / f"{var}_qnom_controlled__{tag}.png",
                suptitle=(
                    f"TTF (ННО) vs {var}, cleaned of {CONTROL_SET_LABEL['qnom']} "
                    f"— {', '.join(show_fields)} ({rows}, residual in {scale})"
                ),
                resid_ylabel=resid_ylabel,
            )
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    prediction_workbook_path: Path | None = None,
    variants: tuple[str, ...] = ("failures", "all_closed"),
    bins: int = 10,
    log_response: bool = False,
    mc_cohort: bool = True,
    by_field_vars: tuple[str, ...] = ("freq", "kpod", "delta"),
    vt_stratum_vars: tuple[str, ...] = ("freq", "kpod", "delta"),
    freq_adjusted_vars: tuple[str, ...] = ("kpod", "delta"),
    run_date: str | None = None,
) -> Path:
    out = results_dir(SLUG, run_date)
    tables, figures = out / "tables", out / "figures"
    scale = "log-days" if log_response else "days"
    resid_ylabel = "residual log ННО" if log_response else "residual ННО (days)"

    summary = []
    for variant in variants:
        panel = load_panel(prediction_workbook_path, variant=variant, mc_cohort=mc_cohort)
        decomp, coverage = decompose(panel, bins=bins, log_response=log_response)
        tag = f"{variant}_{'log' if log_response else 'days'}"
        coverage.to_csv(tables / f"{tag}__x_coverage.csv", index=False)
        for name, d in decomp.items():
            d.marginal.to_csv(tables / f"{tag}__{name}_marginal.csv", index=False)
            d.residual.to_csv(tables / f"{tag}__{name}_residual.csv", index=False)
        fit_table(decomp).to_csv(tables / f"{tag}__quadratic_fits.csv", index=False)

        rows = "genuine failures only" if variant == "failures" else "all closed runs"

        # Per-field replication of whichever x-axes are worth splitting.
        for var in by_field_vars:
            per_field = by_field(panel, var=var, log_response=log_response)
            if not per_field:
                continue
            fit_table(per_field).to_csv(
                tables / f"{tag}__{var}_by_field_fits.csv", index=False)
            for field, d in per_field.items():
                d.residual.to_csv(
                    tables / f"{tag}__{var}_residual_{field}.csv", index=False)
            plot_groups(
                per_field,
                decomp.get(var),
                figures / f"{var}_by_field__{tag}.png",
                suptitle=(
                    f"Свод ННО residual vs {var}, per field ({rows}) — "
                    f"does the pooled shape replicate?"
                ),
                resid_ylabel=resid_ylabel,
            )

        # Vt sour vs nonsour on the v3.2 well-level relabel, one figure per x-axis.
        for var in vt_stratum_vars:
            strata = by_vt_stratum(panel, var=var, log_response=log_response)
            if not strata:
                continue
            fit_table(strata).to_csv(
                tables / f"{tag}__{var}_vt_stratum_fits.csv", index=False)
            for key, d in strata.items():
                d.marginal.to_csv(tables / f"{tag}__{var}_marginal_{key}.csv", index=False)
                d.residual.to_csv(tables / f"{tag}__{var}_residual_{key}.csv", index=False)
            vt_pooled = by_group(
                panel[panel["field"] == "Vt"].assign(_all="Vt all"),
                var=var, group_col="_all", min_n=40, log_response=log_response,
            ).get("Vt all")
            plot_groups(
                strata,
                vt_pooled,
                figures / f"{var}_vt_sour_nonsour__{tag}.png",
                suptitle=(
                    f"Vt ННО residual vs {var}, sour vs nonsour "
                    f"(v3.2 well-level relabel, {rows})"
                ),
                resid_ylabel=resid_ylabel,
            )

        # ---- freq-extracted pass -------------------------------------------
        # Kpod and Delta rebuilt with frequency ALSO removed (linear + quadratic).
        # Kpod = Ql/Qnom and freq are tied by the affinity law, so this separates
        # "the setpoint matters" from "the setpoint is how you read off rate+speed".
        for_ = freq_supported(panel)
        if freq_adjusted_vars and len(for_) > bins * 3:
            fdec, fcov = decompose(for_, bins=bins, log_response=log_response,
                                   with_freq=True)
            fdec = {k: v for k, v in fdec.items() if k in freq_adjusted_vars}
            ftag = f"{tag}_freqadj"
            fcov.to_csv(tables / f"{ftag}__x_coverage.csv", index=False)
            fit_table(fdec).to_csv(tables / f"{ftag}__quadratic_fits.csv", index=False)
            for name, d in fdec.items():
                d.marginal.to_csv(tables / f"{ftag}__{name}_marginal.csv", index=False)
                d.residual.to_csv(tables / f"{ftag}__{name}_residual.csv", index=False)
            plot(
                fdec,
                figures / f"nno_decomposition__{ftag}.png",
                suptitle=(
                    f"Свод ННО ({rows}, n={len(for_)} with usable freq): Kpod and Delta "
                    f"after removing field + Ql + FREQUENCY — residual in {scale}"
                ),
                resid_ylabel=resid_ylabel,
                control_label=CONTROL_LABEL[True],
            )
            for var in freq_adjusted_vars:
                pf = by_field(for_, var=var, log_response=log_response, with_freq=True)
                if not pf:
                    continue
                fit_table(pf).to_csv(tables / f"{ftag}__{var}_by_field_fits.csv", index=False)
                for fld, d in pf.items():
                    d.residual.to_csv(
                        tables / f"{ftag}__{var}_residual_{fld}.csv", index=False)
                plot_groups(
                    pf,
                    fdec.get(var),
                    figures / f"{var}_by_field__{ftag}.png",
                    suptitle=(
                        f"Свод ННО vs {var}, per field, FREQUENCY REMOVED "
                        f"(control: {CONTROL_LABEL[True]}; {rows})"
                    ),
                    resid_ylabel=resid_ylabel,
                )
            summary.append({
                "variant": variant, "control": CONTROL_LABEL[True],
                "residual_scale": scale, "mc_cohort": mc_cohort, "n_runs": len(for_),
                "n_fields": for_["field"].nunique(),
                "mean_nno": for_["nno"].mean(), "median_nno": for_["nno"].median(),
            })

        plot(
            decomp,
            figures / f"nno_decomposition__{tag}.png",
            suptitle=(
                f"Свод ННО ({'genuine failures only' if variant == 'failures' else 'all closed runs'}, "
                f"n={len(panel)}): marginal vs what survives field + Ql — residual in {scale}"
            ),
            resid_ylabel=resid_ylabel,
        )
        summary.append({
            "variant": variant,
            "control": CONTROL_LABEL[False],
            "residual_scale": scale,
            "mc_cohort": mc_cohort,
            "n_runs": len(panel),
            "n_fields": panel["field"].nunique(),
            "mean_nno": panel["nno"].mean(),
            "median_nno": panel["nno"].median(),
        })
    pd.DataFrame(summary).to_csv(tables / "panel_summary.csv", index=False)
    return out
