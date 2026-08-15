"""k1 (shipped) vs k2, judged on what the prognosis actually uses: the ORDERING.

A survival curve can look right and still rank the fleet wrong. The forecast does not
consume S(t) — it consumes the 30-day failure probability at each pump's current age,

    q30(a) = 1 - S(a+30)/S(a),

and turns that into a queue. So two fits that sit inside the same KM band can still
disagree about which pump to pull first. That disagreement is the deliverable here.

Three panels per stratum, all on the `cal` clock at c0 so the pair is comparable:

  1. **S(t)** against the censored KM — the usual goodness check.
  2. **q30 by age** — the ordering driver. A k1 with beta<1 decays (older = safer),
     beta~1 is flat (age carries NO information), k2 spikes early then flattens.
     The rug marks the ages of the stratum's actual RUNNING pumps, so the comparison
     is read over the age range the fleet really occupies, not over all of [0, inf).
  3. **rank-vs-rank** of the live pumps under the two models, with Spearman rho.
     rho = +1 means the models disagree about *magnitude* but produce an identical
     work queue — in which case the k1/k2 choice does not change the prognosis.
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
from scipy.stats import spearmanr  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_optime as O  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402
from analysis.workflows.production_risk import weibull_grid as G  # noqa: E402

AS_OF = "2026-07-16"
CLOCK, CLOCK_COL, CUT = "cal", "t_cal", 0.0
HORIZON = 30.0
STRATA = [(h, c) for h in ("sour", "nonsour") for c in ("Pooled", "brt", "slb", "oth")]

INK, MUTED, GRID_C = "#0b0b0b", "#898781", "#e1e0d9"
BASELINE, SURFACE, CI_FILL = "#c3c2b7", "#fcfcfb", "#e1e0d9"
K1_C, K2_C = "#2a78d6", "#eb6834"   # blue = k1 (shipped shape), orange = k2

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8, "font.size": 10,
})


def q30(ages: np.ndarray, p: dict, horizon: float = HORIZON) -> np.ndarray:
    """P(fail within `horizon` days | alive at age) — what the forecast ranks on."""
    s_a = G.surv_params(ages, p)
    s_b = G.surv_params(ages + horizon, p)
    return 1.0 - np.divide(s_b, s_a, out=np.ones_like(s_a), where=s_a > 0)


def label(r: pd.Series) -> str:
    """Every fitted parameter, so the shape can be judged from the figure alone."""
    if r["kind"] == "k1":
        return f"k1:  β={r['beta1']:.3f}  η={r['eta1']:.1f}"
    return (f"k2 {r['regime']}:  w1={r['w1']:.3f}  β1={r['beta1']:.3f} η1={r['eta1']:.1f}"
            f"  β2={r['beta2']:.3f} η2={r['eta2']:.1f}")


# Bounds the optimiser actually ran against (weibull_grid): a fit sitting on one is a
# corner solution, not a measurement.
BETA_HI, ETA_HI, RATIO_FLOOR = 8.0, 50_000.0, 2.0


def sanity(r: pd.Series, t_max: float) -> tuple[str, str]:
    """(verdict, why) — is this parameter set physically readable?"""
    bad, warn = [], []
    if r["kind"] == "k2":
        w1 = float(r["w1"])
        if w1 < 0.02 or w1 > 0.98:
            bad.append(f"w1={w1:.3f} degenerate (one component carries everything)")
        if float(r["beta1"]) >= BETA_HI - 0.01 or float(r["beta2"]) >= BETA_HI - 0.01:
            bad.append("β at the 8.0 bound = a point mass in disguise")
        if float(r["eta1"]) < 1.0:
            bad.append(f"η1={float(r['eta1']):.2f}d < 1 day — a day-0 spike, not a component")
        if float(r["eta2"]) / max(float(r["eta1"]), 1e-9) <= RATIO_FLOOR * 1.01:
            bad.append("η2/η1 pinned at the 2.0 floor — components not separated")
        if float(r["eta2"]) > 5 * t_max:
            warn.append(f"η2={float(r['eta2']):.0f}d >> longest run {t_max:.0f}d — extrapolation")
        if abs(float(r["beta2"]) - 1.0) > 0.5:
            warn.append(f"β2={float(r['beta2']):.2f} far from 1 (plateau not memoryless)")
        p = r["p_lr"]
        if pd.notna(p) and float(p) >= 0.05:
            bad.append(f"p_LR={float(p):.3f} — 3 extra parameters NOT earned")
    else:
        b = float(r["beta1"])
        if b >= BETA_HI - 0.01:
            bad.append("β at bound")
        if abs(b - 1.0) <= 0.06:
            warn.append(f"β={b:.3f} ≈ 1 — memoryless, age carries no ordering signal")
        if float(r["eta1"]) > 5 * t_max:
            warn.append("η >> longest observed run — extrapolation")
    if bad:
        return "REJECT", "; ".join(bad)
    return ("OK" if not warn else "OK*"), "; ".join(warn)


def main() -> None:
    out = results_dir("production_risk_vt_weibull_grid")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(out / "tables" / "vt_weibull_grid.csv", encoding="utf-8-sig")
    grid["flg"] = grid["flags"].astype(str).replace("nan", "")

    pop = P.build(AS_OF)
    vt = pop[pop["field"] == "Vt"].copy()
    vt = P.add_time_scales(vt, AS_OF)
    vt = O.measure(vt, as_of=pd.Timestamp(AS_OF))
    t_as_of = pd.Timestamp(AS_OF)

    summary = []
    for h2s, ctr in STRATA:
        name = f"Vt_{h2s}_{ctr}"
        g = grid[(grid.stratum == name) & (grid.clock == CLOCK) & (grid.cut == CUT)]
        if g.empty:
            continue
        k1 = g[g["kind"] == "k1"].iloc[0]
        k2s = g[g["kind"] == "k2"]
        if k2s.empty:
            continue
        k2 = k2s.loc[k2s.ks.idxmin()]

        frame = vt[vt["h2s_class"] == h2s]
        if ctr != "Pooled":
            frame = frame[frame["contractor_group"] == ctr]
        t = frame[CLOCK_COL].to_numpy(float)
        ev = frame["event"].to_numpy(int)
        live = frame[~(frame["end"].notna() & (frame["end"] <= t_as_of))]
        live_ages = live[CLOCK_COL].to_numpy(float)

        fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.7))

        # ── 1. survival vs KM ──
        km = KaplanMeierFitter().fit(t, ev)
        x = km.survival_function_.index.to_numpy(float)
        y = km.survival_function_.iloc[:, 0].to_numpy(float)
        ci = km.confidence_interval_
        x_max = float(t[ev == 1].max()) * 1.05
        tau = float(np.percentile(t, 95))
        tt = np.linspace(0, x_max, 600)
        ax[0].fill_between(ci.index.to_numpy(float), ci.iloc[:, 0], ci.iloc[:, 1],
                           step="post", color=CI_FILL, alpha=0.75, linewidth=0, label="KM 95% CI")
        ax[0].step(x, y, where="post", color=INK, linewidth=1.6, label="KM (censored)")
        ax[0].plot(tt, G.surv_params(tt, k1), color=K1_C, linewidth=2.0,
                   label=f"k1 (KS {k1['ks']:.3f})")
        ax[0].plot(tt, G.surv_params(tt, k2), color=K2_C, linewidth=2.0, linestyle="--",
                   label=f"k2 (KS {k2['ks']:.3f}, p_LR {k2['p_lr']:.3f})")
        if tau < x_max:
            ax[0].axvspan(tau, x_max, color=MUTED, alpha=0.09, linewidth=0, zorder=0)
        ax[0].axvline(tau, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax[0].set_xlim(0, x_max); ax[0].set_ylim(0, 1)
        ax[0].set_ylabel("S(t)"); ax[0].set_xlabel("t_cal, сут")
        ax[0].set_title("1. кривая выживания", fontsize=10, color="#52514e", loc="left")
        ax[0].legend(loc="upper right", frameon=False, fontsize=8)

        # ── 2. the ordering driver ──
        a_max = max(float(live_ages.max()) if live_ages.size else 0.0, tau) * 1.05
        aa = np.linspace(0.5, a_max, 500)
        ax[1].plot(aa, 100 * q30(aa, k1), color=K1_C, linewidth=2.0, label=label(k1))
        ax[1].plot(aa, 100 * q30(aa, k2), color=K2_C, linewidth=2.0, linestyle="--", label=label(k2))
        if live_ages.size:
            ax[1].plot(live_ages, np.full(live_ages.size, 0.0), marker="|", linestyle="none",
                       markersize=8, color=MUTED, alpha=0.6,
                       label=f"возраст живых насосов (n={live_ages.size})")
        ax[1].set_xlim(0, a_max); ax[1].set_ylim(bottom=0)
        ax[1].set_xlabel("возраст, сут"); ax[1].set_ylabel("P(отказ за 30 сут), %")
        ax[1].set_title("2. чем упорядочивается очередь", fontsize=10, color="#52514e", loc="left")
        ax[1].legend(loc="upper right", frameon=False, fontsize=8)

        # ── 3. does the queue actually change? ──
        rho = np.nan
        if live_ages.size >= 3:
            r1, r2 = q30(live_ages, k1), q30(live_ages, k2)
            rho = float(spearmanr(r1, r2).statistic)
            ax[2].scatter(pd.Series(r1).rank(), pd.Series(r2).rank(), s=26,
                          color=INK, alpha=0.55, linewidth=0)
            lim = live_ages.size + 1
            ax[2].plot([0, lim], [0, lim], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim)
            verdict = ("очередь ИДЕНТИЧНА" if rho > 0.999 else
                       "очередь ПЕРЕВЁРНУТА" if rho < -0.999 else "очередь МЕНЯЕТСЯ")
            ax[2].set_title(f"3. ранг риска: ρ = {rho:+.3f} — {verdict}",
                            fontsize=10, color="#52514e", loc="left")
        else:
            ax[2].set_title("3. ранг риска — мало живых насосов",
                            fontsize=10, color="#52514e", loc="left")
        ax[2].set_xlabel("ранг по k1"); ax[2].set_ylabel("ранг по k2")

        for a in ax:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        fig.suptitle(f"{name} — {len(frame)} runs / {int(frame['event'].sum())} failures   "
                     f"(clock t_cal, c0)", fontsize=12, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = figures / f"k1_vs_k2_{name}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)

        t_max = float(t.max())
        for r in (k1, k2):
            verdict, why = sanity(r, t_max)
            summary.append({
                "stratum": name, "kind": r["kind"], "regime": r["regime"],
                "n": int(r["n"]), "events": int(r["events"]),
                "w1": round(float(r["w1"]), 4), "beta1": round(float(r["beta1"]), 4),
                "eta1": round(float(r["eta1"]), 2), "beta2": round(float(r["beta2"]), 4),
                "eta2": round(float(r["eta2"]), 2),
                "eta2_over_eta1": round(float(r["eta2"]) / max(float(r["eta1"]), 1e-9), 1),
                "ks": r["ks"], "p_lr": r["p_lr"], "loglik": r["loglik"],
                "b50": r["b50"], "rmst": r["rmst"], "rmst_km": r["rmst_km"],
                "t_max_obs": round(t_max, 0),
                "q30_at_100d_pct": round(100 * float(q30(np.array([100.0]), r)[0]), 2),
                "rank_rho_k1_vs_k2": round(rho, 4) if r["kind"] == "k2" else np.nan,
                "verdict": verdict, "why": why})
        print("wrote", path)

    df = pd.DataFrame(summary)
    df.to_csv(out / "tables" / "vt_k1_vs_k2_params.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print("\n== parameters ==")
        print(df[["stratum", "kind", "regime", "w1", "beta1", "eta1", "beta2", "eta2",
                  "eta2_over_eta1", "ks", "p_lr", "rmst", "rmst_km", "t_max_obs",
                  "verdict"]].to_string(index=False))
        print("\n== why a fit is rejected / flagged ==")
        for _, r in df[df["why"] != ""].iterrows():
            print(f"  {r['stratum']:20s} {r['kind']:2s} {r['verdict']:6s} {r['why']}")
        print("\n== ordering ==")
        o = df[df["kind"] == "k2"][["stratum", "rank_rho_k1_vs_k2"]]
        print(o.to_string(index=False))


if __name__ == "__main__":
    main()
