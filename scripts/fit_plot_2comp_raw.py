"""
Fit 2-comp Weibull mixture on Vt0_base RAW (no КИП filter) and compare with:
  - KM empirical (raw)
  - 2-comp kekspl (КИП>0.5, existing fit)
  - KM empirical (kekspl)

Question: is the bimodal early-cluster structure present in raw data,
          or is it an artifact of the КИП filter?
"""
import math, os, sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lifelines import KaplanMeierFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = r"d:\GitHub\Pump2\data\warehouse\pump2.db"
OUT_DIR = r"d:\GitHub\Pump2\docs\uvch_claude"

# ── Known КИП-filtered 2-comp parameters ─────────────────────────────────────
KIP_2C = dict(pi_e=0.240, beta_e=8.26, eta_e=105.0,
              pi_n=0.760, beta_n=1.333, eta_n=286.1)

# ── Load cohorts ──────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

BASE_FILTER = """
    field = 'Vt'
    AND freq_w_mean >= 50 AND freq_w_mean < 55
    AND run_days > 0
"""

df_raw = pd.read_sql_query(f"""
    SELECT ttf_true_best_days AS duration, event, run_days,
           ttf_true_best_days / run_days AS kip
    FROM mart__vt_freq55
    WHERE {BASE_FILTER}
""", conn)

df_kip = df_raw[df_raw["kip"] > 0.5].copy()
conn.close()

print(f"Raw cohort :  n={len(df_raw)}, events={int(df_raw.event.sum())}")
print(f"КИП>0.5   :  n={len(df_kip)}, events={int(df_kip.event.sum())}")
print(f"Removed by КИП filter: {len(df_raw)-len(df_kip)} wells")
print()
removed = df_raw[df_raw["kip"] <= 0.5][["duration","event","kip"]]
print("Removed wells:")
print(removed.to_string(index=False))

# ── 2-comp Weibull MLE ────────────────────────────────────────────────────────
def weibull_sf(t, eta, beta):
    return np.exp(-(t / eta) ** beta)

def neg_log_lik_2comp(params, t, d):
    """Negative log-likelihood for 2-comp mixture."""
    logit_pi, log_be, log_ee, log_bn, log_en = params
    pi_e  = 1.0 / (1.0 + np.exp(-logit_pi))   # mixing weight early component
    beta_e = np.exp(log_be)
    eta_e  = np.exp(log_ee)
    beta_n = np.exp(log_bn)
    eta_n  = np.exp(log_en)
    pi_n   = 1.0 - pi_e

    se = weibull_sf(t, eta_e, beta_e)
    sn = weibull_sf(t, eta_n, beta_n)
    s_mix = pi_e * se + pi_n * sn

    # hazard density at event times
    fe = (beta_e / eta_e) * (t / eta_e) ** (beta_e - 1) * se
    fn = (beta_n / eta_n) * (t / eta_n) ** (beta_n - 1) * sn
    f_mix = pi_e * fe + pi_n * fn

    s_mix_safe  = np.clip(s_mix,  1e-300, None)
    f_mix_safe  = np.clip(f_mix,  1e-300, None)

    ll = np.sum(d * np.log(f_mix_safe) + (1 - d) * np.log(s_mix_safe))
    return -ll

def fit_2comp(df, label, init_params=None):
    t = df["duration"].values.astype(float)
    d = df["event"].values.astype(float)
    n, n_ev = len(t), int(d.sum())

    # Multiple starts to avoid local minima
    best_nll, best_res = np.inf, None
    starts = [
        # logit(pi_e), log(beta_e), log(eta_e), log(beta_n), log(eta_n)
        [np.log(0.24/0.76), np.log(8.0), np.log(100.0), np.log(1.3), np.log(280.0)],
        [np.log(0.30/0.70), np.log(5.0), np.log(90.0),  np.log(1.5), np.log(250.0)],
        [np.log(0.20/0.80), np.log(10.0), np.log(110.0), np.log(1.2), np.log(300.0)],
        [np.log(0.15/0.85), np.log(6.0), np.log(80.0),  np.log(1.4), np.log(260.0)],
        [np.log(0.35/0.65), np.log(4.0), np.log(120.0), np.log(1.0), np.log(230.0)],
    ]
    for s in starts:
        try:
            res = minimize(neg_log_lik_2comp, s, args=(t, d),
                           method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        except Exception:
            pass

    p = best_res.x
    pi_e   = 1.0 / (1.0 + np.exp(-p[0]))
    beta_e = np.exp(p[1])
    eta_e  = np.exp(p[2])
    beta_n = np.exp(p[3])
    eta_n  = np.exp(p[4])

    # 1-comp NLL for ΔAIC
    from scipy.optimize import minimize as mini2
    def nll_1comp(q):
        b, e = np.exp(q[0]), np.exp(q[1])
        sf = weibull_sf(t, e, b)
        ff = (b/e)*(t/e)**(b-1)*sf
        return -np.sum(d*np.log(np.clip(ff,1e-300,None)) + (1-d)*np.log(np.clip(sf,1e-300,None)))
    r1 = mini2(nll_1comp, [np.log(1.3), np.log(230)], method="Nelder-Mead")
    nll_1c = r1.fun
    aic_1c = 2*2 + 2*nll_1c
    aic_2c = 2*5 + 2*best_nll
    daic   = aic_2c - aic_1c

    print(f"\n{label}  (n={n}, events={n_ev})")
    print(f"  1-comp: b={np.exp(r1.x[0]):.3f}  eta={np.exp(r1.x[1]):.1f}d  AIC={aic_1c:.1f}")
    print(f"  2-comp: pi_e={pi_e:.3f}  b_e={beta_e:.2f}  eta_e={eta_e:.1f}d  "
          f"|  pi_n={1-pi_e:.3f}  b_n={beta_n:.3f}  eta_n={eta_n:.1f}d  AIC={aic_2c:.1f}  dAIC={daic:.1f}")

    return dict(pi_e=pi_e, beta_e=beta_e, eta_e=eta_e,
                pi_n=1-pi_e, beta_n=beta_n, eta_n=eta_n,
                nll=best_nll, aic_2c=aic_2c, aic_1c=aic_1c, daic=daic)

params_raw = fit_2comp(df_raw, "M0_base RAW (no КИП)")
params_kip = KIP_2C   # use the known-good fit (from existing analysis)

# ── Survival function helpers ─────────────────────────────────────────────────
def S_mix(t, p):
    se = np.exp(-(t / p["eta_e"]) ** p["beta_e"])
    sn = np.exp(-(t / p["eta_n"]) ** p["beta_n"])
    return p["pi_e"] * se + p["pi_n"] * sn

def mix_median(p):
    lo, hi = 0.0, 3000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if S_mix(mid, p) > 0.5: lo = mid
        else: hi = mid
    return (lo + hi) / 2

# ── Plot ──────────────────────────────────────────────────────────────────────
t = np.linspace(0, 700, 1400)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Vt0_base: 2-comp Weibull — Raw vs КИП-filtered\n"
    "Does the early-cluster structure survive the КИП filter removal?",
    fontsize=12, fontweight="bold"
)

COLORS = {
    "km_raw":  "#FF6F00",
    "km_kip":  "#2196F3",
    "2c_raw":  "#E53935",
    "2c_kip":  "#1565C0",
    "comp2_raw": "#FF7043",
    "comp2_kip": "#42A5F5",
}

for ax_idx, (ax, zoom) in enumerate(zip(axes, [False, True])):
    t_plot = np.linspace(0, 300 if zoom else 700, 1000)
    xlim = (0, 300) if zoom else (0, 700)

    # KM — raw
    kmf_raw = KaplanMeierFitter()
    kmf_raw.fit(df_raw["duration"], df_raw["event"])
    kmf_raw.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=0.12,
        color=COLORS["km_raw"], linewidth=1.8,
        label=f"KM raw (n={len(df_raw)}, {int(df_raw.event.sum())} events)"
    )

    # KM — КИП filtered
    kmf_kip = KaplanMeierFitter()
    kmf_kip.fit(df_kip["duration"], df_kip["event"])
    kmf_kip.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=0.12,
        color=COLORS["km_kip"], linewidth=1.8,
        label=f"KM КИП>0.5 (n={len(df_kip)}, {int(df_kip.event.sum())} events)"
    )

    # 2-comp — raw (newly fitted)
    ax.plot(t_plot, S_mix(t_plot, params_raw),
            color=COLORS["2c_raw"], linewidth=2.2, linestyle="-",
            label=(f"2-comp RAW fit\n"
                   f"  π_e={params_raw['pi_e']:.2f} β_e={params_raw['beta_e']:.1f} η_e={params_raw['eta_e']:.0f}d\n"
                   f"  π_n={params_raw['pi_n']:.2f} β_n={params_raw['beta_n']:.2f} η_n={params_raw['eta_n']:.0f}d\n"
                   f"  ΔAIC={params_raw['daic']:.1f}"))

    # 2-comp — КИП (known fit)
    ax.plot(t_plot, S_mix(t_plot, params_kip),
            color=COLORS["2c_kip"], linewidth=2.2, linestyle="--",
            label=(f"2-comp КИП fit (known)\n"
                   f"  π_e={params_kip['pi_e']:.2f} β_e={params_kip['beta_e']:.1f} η_e={params_kip['eta_e']:.0f}d\n"
                   f"  π_n={params_kip['pi_n']:.2f} β_n={params_kip['beta_n']:.2f} η_n={params_kip['eta_n']:.0f}d\n"
                   f"  ΔAIC=−7.9"))

    if zoom:
        # Draw component curves for raw fit
        t_z = t_plot
        se_raw = params_raw["pi_e"] * np.exp(-(t_z / params_raw["eta_e"]) ** params_raw["beta_e"])
        sn_raw = params_raw["pi_n"] * np.exp(-(t_z / params_raw["eta_n"]) ** params_raw["beta_n"])
        ax.plot(t_z, se_raw, color=COLORS["2c_raw"], linewidth=1.0, linestyle=":",
                label=f"Early × {params_raw['pi_e']:.2f} (raw)")
        ax.plot(t_z, sn_raw, color=COLORS["2c_raw"], linewidth=1.0, linestyle="-.",
                label=f"Normal × {params_raw['pi_n']:.2f} (raw)")

        se_kip = params_kip["pi_e"] * np.exp(-(t_z / params_kip["eta_e"]) ** params_kip["beta_e"])
        sn_kip = params_kip["pi_n"] * np.exp(-(t_z / params_kip["eta_n"]) ** params_kip["beta_n"])
        ax.plot(t_z, se_kip, color=COLORS["2c_kip"], linewidth=1.0, linestyle=":",
                label=f"Early × {params_kip['pi_e']:.2f} (КИП)")
        ax.plot(t_z, sn_kip, color=COLORS["2c_kip"], linewidth=1.0, linestyle="-.",
                label=f"Normal × {params_kip['pi_n']:.2f} (КИП)")

        # Early cluster peak annotation
        t_peak_raw = params_raw["eta_e"] * ((params_raw["beta_e"]-1)/params_raw["beta_e"])**(1/params_raw["beta_e"])
        ax.axvline(t_peak_raw, color=COLORS["2c_raw"], linewidth=0.8, linestyle=":", alpha=0.7)
        ax.text(t_peak_raw+2, 0.90, f"peak\n{t_peak_raw:.0f}d", fontsize=7, color=COLORS["2c_raw"])

    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days since installation", fontsize=11)
    ax.set_ylabel("Survival probability S(t)", fontsize=11)
    ax.set_title("Full range (0–700d)" if not zoom else "Early cluster region (0–300d)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

# ── Parameter comparison table ────────────────────────────────────────────────
med_raw = mix_median(params_raw)
med_kip = mix_median(params_kip)

table_text = (
    f"Parameter comparison\n"
    f"{'':12s} {'RAW':>10} {'КИП>0.5':>10}\n"
    f"{'n':12s} {len(df_raw):>10d} {len(df_kip):>10d}\n"
    f"{'events':12s} {int(df_raw.event.sum()):>10d} {int(df_kip.event.sum()):>10d}\n"
    f"{'π_e':12s} {params_raw['pi_e']:>10.3f} {params_kip['pi_e']:>10.3f}\n"
    f"{'β_e':12s} {params_raw['beta_e']:>10.2f} {params_kip['beta_e']:>10.2f}\n"
    f"{'η_e':12s} {params_raw['eta_e']:>9.1f}d {params_kip['eta_e']:>9.1f}d\n"
    f"{'π_n':12s} {params_raw['pi_n']:>10.3f} {params_kip['pi_n']:>10.3f}\n"
    f"{'β_n':12s} {params_raw['beta_n']:>10.3f} {params_kip['beta_n']:>10.3f}\n"
    f"{'η_n':12s} {params_raw['eta_n']:>9.1f}d {params_kip['eta_n']:>9.1f}d\n"
    f"{'median':12s} {med_raw:>9.0f}d {med_kip:>9.0f}d\n"
    f"{'ΔAIC':12s} {params_raw['daic']:>10.1f} {'−7.9':>10s}"
)
fig.text(0.01, 0.01, table_text, fontsize=8.5, family="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

plt.tight_layout(rect=[0, 0.20, 1, 1])
out_path = os.path.join(OUT_DIR, "weibull_2comp_raw_vs_kip.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {out_path}")
print(f"\nMedian from 2-comp:  raw={med_raw:.0f}d  kip={med_kip:.0f}d")
