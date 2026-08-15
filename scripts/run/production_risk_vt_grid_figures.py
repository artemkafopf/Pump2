"""Visual assessment figures for the Vt Weibull grid — one PNG per stratum.

Two panels per figure (calendar clock | op-days clock). Each panel overlays on
the right-censored Kaplan-Meier (95% CI band + censor ticks):

  * the shipped registry model (esp_models.csv row that serves this stratum);
  * the best c0 fit on this clock from the grid (directly servable in the
    5-column schema);
  * the best eligible fit of any cut on this clock, when it differs — drawn as
    S(t | t > c) anchored at KM(c), which is the only honest way to put a
    left-truncated fit on the full-population axes.

Each panel marks **tau = the 95th percentile of observed follow-up** (events and
censorings alike). Past tau the KM rests on the last ~5% of the risk set, so its
steps are large, its CI blows out, and a model-vs-KM gap there is noise, not
misfit — the region is shaded to keep the eye from scoring the fit on it. It is
also the horizon a restricted mean should be reported to.

Eligible = k1, or k2 with LR p<0.05 and no honesty flag. Reads the grid CSV
written by ``production_risk_vt_weibull_grid.py`` (same results date-dir).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_optime as O  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402
from analysis.workflows.production_risk import weibull_grid as G  # noqa: E402
from analysis.workflows.production_risk.survival import StrataModel  # noqa: E402

AS_OF = "2026-07-16"
CLOCKS = (("cal", "t_cal"), ("op", "t_mix"))
CLOCK_TITLE = {
    "cal": ("cal — t_cal (календарь: монтаж → подъём)",
            "elapsed days — nothing reported, nothing imputed"),
    "op": ("op — t_mix (наработка по телеметрии)",
           "measured op-days; t_cal × Кэкспл where telemetry is blank"),
}
STRATA = [(h2s, ctr) for h2s in ("sour", "nonsour")
          for ctr in ("Pooled", "brt", "slb", "oth")]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
CI_FILL = "#e1e0d9"
S1_BLUE, S2_GREEN, S3_MAGENTA = "#2a78d6", "#008300", "#e87ba4"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": "#52514e",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10,
})


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    flags = df["flags"].fillna("")
    return df[(df["kind"] == "k1") | ((df["p_lr"] < 0.05) & (flags == ""))]


def fit_label(row: pd.Series) -> str:
    kind = row["kind"] if row["kind"] == "k1" else f"k2 {row['regime'][:5]}"
    return f"{kind} c{int(row['cut'])}"


def ks_upto(x, y, params: dict, tau: float, cut: float = 0.0, anchor: float = 1.0) -> float:
    """max|dS| against the KM, restricted to t <= tau — the honest scoring window.

    Past tau the KM rides the last 5% of the risk set, so a gap there says more
    about one late run than about the model.
    """
    m = (x <= tau) & (x >= cut)
    if not m.any():
        return float("nan")
    s = surv_conditional(x[m], params, cut) * anchor
    return float(np.max(np.abs(y[m] - s)))


def surv_conditional(t, params: dict, cut: float) -> np.ndarray:
    s = G.surv_params(t, params)
    if cut <= 0:
        return s
    return s / float(G.surv_params(np.array([cut]), params)[0])


def draw_panel(ax, frame: pd.DataFrame, clock: str, col: str,
               reg_params: dict, reg_ks: float, grid_df: pd.DataFrame) -> None:
    t = frame[col].to_numpy(float)
    ev = frame["event"].to_numpy(int)

    km = KaplanMeierFitter().fit(t, ev, label="KM")
    x = km.survival_function_.index.to_numpy(float)
    y = km.survival_function_.iloc[:, 0].to_numpy(float)
    ci = km.confidence_interval_
    x_max = float(t[ev == 1].max()) * 1.05  # beyond the last event KM is flat censoring
    tau = float(np.percentile(t, 95))       # 95th pct of follow-up: KM is thin past here
    at_risk_tau = int((t >= tau).sum())

    ax.fill_between(ci.index.to_numpy(float), ci.iloc[:, 0], ci.iloc[:, 1],
                    step="post", color=CI_FILL, alpha=0.75, linewidth=0,
                    label="KM 95% CI")
    ax.step(x, y, where="post", color=INK, linewidth=1.6, label="KM (censored)")
    tc = t[(ev == 0) & (t <= x_max)]
    ax.plot(tc, np.interp(tc, x, y), linestyle="none", marker="|",
            markersize=4, color=MUTED, alpha=0.5, label="censored")

    tt = np.linspace(0.0, x_max, 600)
    reg_tau = ks_upto(x, y, reg_params, tau)
    ax.plot(tt, G.surv_params(tt, reg_params), color=S1_BLUE, linewidth=2.0,
            linestyle="--", label=f"registry (KS {reg_ks:.3f} | ≤τ {reg_tau:.3f})")

    sub = eligible(grid_df[grid_df["clock"] == clock]).sort_values("ks")
    c0 = sub[sub["cut"] == 0.0]
    if not c0.empty:
        r = c0.iloc[0]
        kt = ks_upto(x, y, r, tau)
        ax.plot(tt, G.surv_params(tt, r), color=S2_GREEN, linewidth=2.0,
                label=f"best c0: {fit_label(r)} (KS {r['ks']:.3f} | ≤τ {kt:.3f})")
    if not sub.empty:
        r = sub.iloc[0]
        if not (not c0.empty and r.name == c0.iloc[0].name):
            cut = float(r["cut"])
            tt2 = np.linspace(cut, x_max, 600)
            anchor = float(np.interp(cut, x, y))
            kt = ks_upto(x, y, r, tau, cut=cut, anchor=anchor)
            ax.plot(tt2, surv_conditional(tt2, r, cut) * anchor, color=S3_MAGENTA,
                    linewidth=2.0,
                    label=f"best any-cut: {fit_label(r)} anchored KM({int(cut)}) "
                          f"(KS {r['ks']:.3f} | ≤τ {kt:.3f})")

    if tau < x_max:
        ax.axvspan(tau, x_max, color=MUTED, alpha=0.09, linewidth=0, zorder=0)
    ax.axvline(tau, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1,
               label=f"τ (95-й перцентиль) = {tau:.0f} d, n at risk {at_risk_tau}")
    # τ letter sits at the baseline, clear of the legend box in the upper right
    ax.annotate("τ", xy=(tau, 0.0), xytext=(4, 6), textcoords="offset points",
                color=MUTED, fontsize=10, fontweight="bold")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    title, sub = CLOCK_TITLE[clock]
    ax.set_xlabel(sub)
    ax.set_title(title, fontsize=10, color="#52514e", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    out = results_dir("production_risk_vt_weibull_grid")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(out / "tables" / "vt_weibull_grid.csv", encoding="utf-8-sig")
    verif = pd.read_csv(out / "tables" / "vt_registry_vs_km.csv", encoding="utf-8-sig")

    pop = P.build(AS_OF)
    vt = pop[pop["field"] == "Vt"].copy()
    vt = P.add_time_scales(vt, AS_OF)
    vt = O.measure(vt, as_of=pd.Timestamp(AS_OF))
    reg = StrataModel()

    for h2s, ctr in STRATA:
        name = f"Vt_{h2s}_{ctr}"
        frame = vt[vt["h2s_class"] == h2s]
        if ctr != "Pooled":
            frame = frame[frame["contractor_group"] == ctr]
        gdf = grid[grid["stratum"] == name]
        if gdf.empty:
            print(f"skip {name} (not in grid)")
            continue
        params, matched = reg.resolve("Vt", h2s, ctr)
        vrow = verif[verif["stratum"] == name].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
        for ax, (clock, col) in zip(axes, CLOCKS):
            draw_panel(ax, frame, clock, col, params,
                       float(vrow[f"ks_{clock}"]), gdf)
        axes[0].set_ylabel("S(t)")
        fig.suptitle(
            f"{name} — {len(frame)} runs / {int(frame['event'].sum())} failures   "
            f"(registry row: {matched}, {vrow['verdict']})",
            fontsize=12, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
