"""Figures for the consolidated model report (results/model_report/2026-07-07)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "backend")):
    sys.path.insert(0, p)

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.data.failure_modes import MODE_GROUPS
from analysis.models.survival.cif import aalen_johansen_cif, event_code_series

OUT = REPO / "results" / "model_report" / "2026-07-07" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette: categorical pair (identity) + sequential blues (ordered tertiles)
C_A, C_B = "#0072B2", "#D55E00"          # nonsour / sour etc.
SEQ = ["#9ECAE1", "#4292C6", "#08519C"]  # low / mid / high tertile
INK, MUTED = "#1a1a1a", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white"})


def style(ax, xlab="Operating days (ttf_mix)", ylab=None):
    ax.grid(True, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlab, fontsize=8.5, color=MUTED)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5, color=MUTED)


def km_step(ax, t, e, label, color, ls="-"):
    km = KaplanMeierFitter()
    km.fit(t, e)
    sf = km.survival_function_
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=color, lw=1.8, ls=ls, label=label)
    return km


def mix_surv(t, w1, b1, e1, b2, e2):
    return w1 * np.exp(-((t / e1) ** b1)) + (1 - w1) * np.exp(-((t / e2) ** b2))


df = build_competing_risks_df(tte_col="ttf_mix")
df = df[df["tte"] > 0].copy()
params = pd.read_csv(REPO / "results/esp_survival_ttf_mix_phase2/2026-07-03/figures/phase2_mixture_params.csv")

# need run_days for the clock figure
if "run_days" not in df.columns:
    import sqlite3
    con = sqlite3.connect(REPO / "data/warehouse/pump2.db")
    rd = pd.read_sql("SELECT row_id, run_days FROM mart__weibull_input", con)
    con.close()
    df = df.merge(rd, on="row_id", how="left")

field_stratum = df["field"] + "_" + df["h2s_class"]

# ── F1: baseline K=2 mixture vs KM ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, strat in zip(axes, ["Ya_nonsour", "Vt_sour"]):
    sub = df[field_stratum == strat]
    km_step(ax, sub["tte"], sub["event"], "Kaplan–Meier", INK)
    r = params[params["stratum"] == strat].iloc[0]
    tg = np.linspace(0.1, float(sub["tte"].max()), 500)
    ax.plot(tg, mix_surv(tg, r.w1, r.beta1, r.eta1_days, r.beta2, r.eta2_days),
            color=C_A, lw=2, label="K=2 Weibull mixture")
    ax.plot(tg, r.w1 * np.exp(-((tg / r.eta1_days) ** r.beta1)), color=C_A, lw=1, ls=":",
            label=f"early comp. (w={r.w1:.2f}, β={r.beta1:.2f}, η={r.eta1_days:.0f}d)")
    ax.plot(tg, (1 - r.w1) * np.exp(-((tg / r.eta2_days) ** r.beta2)), color=C_B, lw=1, ls=":",
            label=f"wear-out comp. (β={r.beta2:.2f}, η={r.eta2_days:.0f}d)")
    ax.set_title(f"{strat} — n={len(sub)}, failures={int(sub['event'].sum())}, B50={r.B50_days:.0f}d",
                 color=INK)
    ax.set_ylim(0, 1)
    style(ax, ylab="S(t)")
    ax.legend(fontsize=7, frameon=False)
fig.suptitle("The baseline model: per-stratum K=2 latent Weibull mixture on operating time", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "f1_baseline_mixture.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── F2: the clock correction ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, strat in zip(axes, ["Ya_nonsour", "Vt_sour"]):
    sub = df[field_stratum == strat]
    k1 = km_step(ax, sub["run_days"], sub["event"], "calendar clock (run_days)", MUTED, ls="--")
    k2 = km_step(ax, sub["tte"], sub["event"], "operating clock (ttf_mix)", C_A)
    m1, m2 = k1.median_survival_time_, k2.median_survival_time_
    ax.set_title(f"{strat} — KM50: {m1:.0f}d calendar → {m2:.0f}d operating", color=INK)
    ax.set_ylim(0, 1)
    style(ax, xlab="Days", ylab="S(t)")
    ax.legend(fontsize=8, frameon=False)
fig.suptitle("Clock correction: idle time inflates calendar lifetimes", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "f2_clock_effect.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── F3: H2S in Vt — KM + time-varying HR ─────────────────────────────────────
vt = df[df["field"] == "Vt"].copy()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axes[0]
g0 = vt[vt["h2s_class"] != "sour"]; g1 = vt[vt["h2s_class"] == "sour"]
km_step(ax, g0["tte"], g0["event"], f"non-sour (n={len(g0)})", C_A)
km_step(ax, g1["tte"], g1["event"], f"sour (n={len(g1)})", C_B, ls="--")
ax.set_title("Vt: sour vs non-sour, KM on operating time", color=INK)
ax.set_ylim(0, 1); ax.set_xlim(0, 700)
style(ax, ylab="S(t)"); ax.legend(fontsize=8, frameon=False)
ax = axes[1]
tg = np.linspace(5, 500, 300)
hr_t = np.exp(0.137 + 0.196 * np.log(tg))
ax.plot(tg, hr_t, color=C_B, lw=2, label="HR(t) = exp(0.137 + 0.196·ln t)")
ax.axhline(2.29, color=MUTED, lw=1.2, ls="--", label="flat Cox HR = 2.29 (M3)")
ax.axhline(1.0, color=INK, lw=0.8)
ax.set_title("Sour hazard ratio vs run age — γ = +0.196 [0.000, 0.415]", color=INK)
ax.set_ylim(0, 5)
style(ax, ylab="hazard ratio")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "f3_h2s_vt.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── F4: Vt per-mode CIF sour vs nonsour ──────────────────────────────────────
vt["event_code"] = event_code_series(vt["mode_group"], vt["event"], MODE_GROUPS)
code = {g: i for i, g in enumerate(MODE_GROUPS, start=1)}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, mode in zip(axes, ["hydraulic", "electro-thermal"]):
    for sub, lab, col, ls in ((g0, "non-sour", C_A, "-"), (g1, "sour", C_B, "--")):
        subc = vt.loc[sub.index]
        c = aalen_johansen_cif(subc["tte"].to_numpy(float), subc["event_code"].to_numpy(int), code[mode])
        tg = np.linspace(0, 500, 400)
        ax.step(c.times, c.cif, where="post", color=col, lw=1.8, ls=ls,
                label=f"{lab} (CIF@90d={c.at(90.0):.2f})")
    ax.set_title(f"{mode} failures — Aalen–Johansen CIF", color=INK)
    ax.set_xlim(0, 500); ax.set_ylim(0, 0.65)
    style(ax, ylab="cumulative incidence")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("What H₂S attacks, and when: pump damage front-loaded; cable/motor compounds with time", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "f4_h2s_modes_cif.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── F5: chemistry disappears under stratification ────────────────────────────
CHEM = "calcium_mg_l"
dfx = df[df[CHEM].notna()].copy()
dfx["tert"] = pd.qcut(dfx[CHEM], 3, labels=["low", "mid", "high"])
print("\n=== F5 chemistry demo:", CHEM, "===")
print("field means (mg/l):")
print(dfx.groupby("field")[CHEM].mean().round(0).sort_values().to_string())

lr_pool = multivariate_logrank_test(dfx["tte"], dfx["tert"], dfx["event"])
print(f"pooled log-rank across tertiles: p={lr_pool.p_value:.2e}")

def cox_p(sub, strata):
    cols = ["tte", "event", f"log_{CHEM}", "well_key"] + (["stratum_key"] if strata else [])
    cph = CoxPHFitter()
    cph.fit(sub[cols].dropna(), duration_col="tte", event_col="event",
            strata=["stratum_key"] if strata else None, cluster_col="well_key", robust=True)
    r = cph.summary.iloc[0]
    return float(np.exp(r["coef"])), float(r["p"])

hr_u, p_u = cox_p(dfx, strata=False)
hr_s, p_s = cox_p(dfx, strata=True)
print(f"unstratified Cox: HR={hr_u:.3f} p={p_u:.2e}   stratified: HR={hr_s:.3f} p={p_s:.3f}")

big = ["Ya_nonsour_brt", "Ya_nonsour_slb", "Vt_nonsour_slb"]
fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), sharey=True)
ax = axes[0]
for tert, col in zip(["low", "mid", "high"], SEQ):
    s = dfx[dfx["tert"] == tert]
    km_step(ax, s["tte"], s["event"], f"{tert} (n={len(s)})", col)
ax.set_title(f"ALL runs pooled — log-rank p={lr_pool.p_value:.1e}\n'chemistry' separates", color=INK)
ax.set_xlim(0, 1200); ax.set_ylim(0, 1)
style(ax, ylab="S(t)"); ax.legend(fontsize=7.5, frameon=False)
for ax, strat in zip(axes[1:], big):
    s = dfx[dfx["stratum_key"] == strat]
    lr = multivariate_logrank_test(s["tte"], s["tert"], s["event"])
    for tert, col in zip(["low", "mid", "high"], SEQ):
        ss = s[s["tert"] == tert]
        if len(ss) > 5:
            km_step(ax, ss["tte"], ss["event"], f"{tert} (n={len(ss)})", col)
    ax.set_title(f"{strat}\nwithin-stratum log-rank p={lr.p_value:.2f}", color=INK)
    ax.set_xlim(0, 1200)
    style(ax)
    ax.legend(fontsize=7.5, frameon=False)
fig.suptitle(f"Calcium tertiles (global bins): the pooled 'effect' is field identity in disguise — "
             f"pooled log-rank p={lr_pool.p_value:.0e} → within-stratum p ≥ 0.2 everywhere", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "f5_chemistry_disappears.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── F6: operational hazards, mode-resolved CIFs ──────────────────────────────
df["event_code"] = event_code_series(df["mode_group"], df["event"], MODE_GROUPS)

def cif_panel(ax, sub, bins_col, labels, colors, mode, title):
    for lab, col in zip(labels, colors):
        s = sub[sub[bins_col] == lab]
        if len(s) < 30:
            continue
        c = aalen_johansen_cif(s["tte"].to_numpy(float), s["event_code"].to_numpy(int), code[mode])
        ax.step(c.times, c.cif, where="post", color=col, lw=1.8,
                label=f"{lab} (n={len(s)}, CIF@365d={c.at(365.0):.2f})")
    ax.set_xlim(0, 800)
    ax.set_title(title, color=INK)
    style(ax, ylab="cumulative incidence")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

# (a) frequency instability -> electro-thermal (measured = enough early freq days)
m = df[df["n_freq_steps_per_100d"].notna() & (df["n_freq_early"] >= 14)].copy()
pos_med = m.loc[m["n_freq_steps_per_100d"] > 0, "n_freq_steps_per_100d"].median()
m["bin"] = np.select(
    [m["n_freq_steps_per_100d"] <= 0, m["n_freq_steps_per_100d"] <= pos_med],
    ["stable (0 steps)", "some steps"], default="unstable (top half)")
cif_panel(axes[0, 0], m, "bin", ["stable (0 steps)", "some steps", "unstable (top half)"], SEQ,
          "electro-thermal", "(a) Frequency instability (steps/100d, early) → ELECTRO-THERMAL")

# (b) chronic underload -> electro-thermal
m2 = df[(df["kpod_mean_missing"] == 0) & df["frac_kpod_below_0p7"].notna()].copy()
m2["bin"] = np.where(m2["frac_kpod_below_0p7"] > 0.5, "underloaded >50% days", "normal loading")
cif_panel(axes[0, 1], m2, "bin", ["normal loading", "underloaded >50% days"], [SEQ[0], SEQ[2]],
          "electro-thermal", "(b) Chronic underload (kpod<0.7, early) → ELECTRO-THERMAL")

# (c) GLF -> hydraulic (protective) and (d) GLF -> electro-thermal (null)
m3 = df[df["glf_mean_opdays_missing"] == 0].copy()
m3["bin"] = pd.qcut(m3["glf_mean_opdays"], 3, labels=["low gas", "mid", "high gas"])
cif_panel(axes[1, 0], m3, "bin", ["low gas", "mid", "high gas"], SEQ, "hydraulic",
          "(c) Gas (GLF, early) → HYDRAULIC: protective ordering")
cif_panel(axes[1, 1], m3, "bin", ["low gas", "mid", "high gas"], SEQ, "electro-thermal",
          "(d) Same GLF bins → ELECTRO-THERMAL: no ordering (mode-corroboration)")

fig.suptitle("Operational hazards land on their physical failure mode (Aalen–Johansen CIF, telemetry runs, early-window exposure)",
             color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "f6_operational_modes.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nFigures written to", OUT)
