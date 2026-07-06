"""
Plot v3 2-comp Weibull mixture vs Kaplan-Meier empirical survival.
Cohort: Vt0_base_kip — Vt wells, freq_w_mean in [50, 55) Hz, КИП > 0.5.
"""
import math, os, sqlite3
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DB_PATH = r"d:\GitHub\Pump2\data\warehouse\pump2.db"
OUT_DIR = r"d:\GitHub\Pump2\docs\uvch_claude"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model parameters ──────────────────────────────────────────────────────────
# v2: 1-comp Weibull
ETA_1, BETA_1 = 240.3, 1.316

# v3: 2-comp Weibull mixture
PI_E, BETA_E, ETA_E = 0.24, 8.26, 105.0
PI_N, BETA_N, ETA_N = 0.76, 1.33, 286.0

def S_1comp(t):
    return np.exp(-(t / ETA_1) ** BETA_1)

def S_2comp(t):
    return PI_E * np.exp(-(t / ETA_E) ** BETA_E) + PI_N * np.exp(-(t / ETA_N) ** BETA_N)

def S_early(t):
    return np.exp(-(t / ETA_E) ** BETA_E)

def S_normal(t):
    return np.exp(-(t / ETA_N) ** BETA_N)

# ── Load Vt0_base_kip cohort ──────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT ttf_true_best_days AS duration, event, freq_w_mean, contractor
    FROM mart__vt_freq55
    WHERE field = 'Vt'
      AND freq_w_mean >= 50 AND freq_w_mean < 55
      AND ttf_true_best_days / run_days > 0.5
      AND run_days > 0
""", conn)
conn.close()

n_total = len(df)
n_events = int(df["event"].sum())
print(f"Cohort: n={n_total}, events={n_events}, censored={n_total-n_events}")
print(f"Duration: min={df['duration'].min():.0f}d, median={df['duration'].median():.0f}d, "
      f"max={df['duration'].max():.0f}d")

# ── Kaplan-Meier fit ──────────────────────────────────────────────────────────
kmf = KaplanMeierFitter()
kmf.fit(df["duration"], df["event"])
km_label = f"Kaplan-Meier (n={n_total}, {n_events} events)"

# ── Plot ──────────────────────────────────────────────────────────────────────
t_max = 700
t = np.linspace(0, t_max, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Vt0_base_kip: 2-comp Weibull mixture vs Kaplan-Meier\n"
    f"Vt wells, freq_w_mean 50–55 Hz, КИП > 0.5  (n={n_total}, {n_events} events, {n_total-n_events} censored)",
    fontsize=12, fontweight="bold"
)

# ── Left panel: full survival curves ─────────────────────────────────────────
ax = axes[0]

# KM with CI
kmf.plot_survival_function(
    ax=ax, ci_show=True, ci_alpha=0.15,
    color="#2196F3", linewidth=2.0,
    label=km_label
)

# Mark censored events on KM curve
# (lifelines already marks them; keep the default)

# 2-comp mixture
ax.plot(t, S_2comp(t), color="#E53935", linewidth=2.5, linestyle="-",
        label=f"v3: 2-comp mixture\n  Early: π={PI_E}, β={BETA_E}, η={ETA_E}d\n  Normal: π={PI_N}, β={BETA_N}, η={ETA_N}d")

# Component curves (scaled by mixing weight)
ax.plot(t, PI_E * S_early(t), color="#FF7043", linewidth=1.2, linestyle="--",
        label=f"Early component × {PI_E} (π={PI_E}, β={BETA_E}, η={ETA_E}d)")
ax.plot(t, PI_N * S_normal(t), color="#E53935", linewidth=1.2, linestyle=":",
        label=f"Normal component × {PI_N} (π={PI_N}, β={BETA_N}, η={ETA_N}d)")

# 1-comp Weibull v2
ax.plot(t, S_1comp(t), color="#43A047", linewidth=2.0, linestyle="--",
        label=f"v2: 1-comp Weibull\n  β={BETA_1}, η={ETA_1}d")

ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
ax.axvline(kmf.median_survival_time_, color="#2196F3", linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(kmf.median_survival_time_ + 5, 0.52,
        f"KM median={kmf.median_survival_time_:.0f}d", fontsize=9, color="#2196F3")

ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Days since installation", fontsize=11)
ax.set_ylabel("Survival probability S(t)", fontsize=11)
ax.set_title("Full survival curves", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

# ── Right panel: zoom 0-300d + residuals ─────────────────────────────────────
ax2 = axes[1]
t_zoom = np.linspace(0, 300, 600)

# KM in zoom region
km_sf = kmf.survival_function_
km_t = km_sf.index.values
km_s = km_sf.iloc[:, 0].values   # first (only) column, name varies with label

# only plot where KM is defined up to 300d
mask = km_t <= 300
ax2.step(km_t[mask], km_s[mask], where="post",
         color="#2196F3", linewidth=2.0, label="Kaplan-Meier")

# KM CI
km_lower = kmf.confidence_interval_.iloc[:, 0].values
km_upper = kmf.confidence_interval_.iloc[:, 1].values
ax2.fill_between(km_t[mask], km_lower[mask], km_upper[mask],
                 color="#2196F3", alpha=0.12, step="post", label="95% CI")

# Add censored tick marks
censored = df[df["event"] == 0]["duration"]
censored_zoom = censored[censored <= 300]
if len(censored_zoom) > 0:
    # Find S at censoring time from KM
    for ct in censored_zoom:
        # nearest KM value at ct
        idx = np.searchsorted(km_t, ct, side="right") - 1
        if 0 <= idx < len(km_s):
            ax2.plot(ct, km_s[idx], "|", color="#2196F3", markersize=8, alpha=0.6, markeredgewidth=1.5)

# 2-comp mixture zoomed
ax2.plot(t_zoom, S_2comp(t_zoom), color="#E53935", linewidth=2.5,
         label="v3: 2-comp mixture")
ax2.plot(t_zoom, PI_E * S_early(t_zoom), color="#FF7043", linewidth=1.2, linestyle="--",
         label=f"Early × {PI_E}")
ax2.plot(t_zoom, PI_N * S_normal(t_zoom), color="#E53935", linewidth=1.2, linestyle=":",
         label=f"Normal × {PI_N}")

# 1-comp
ax2.plot(t_zoom, S_1comp(t_zoom), color="#43A047", linewidth=2.0, linestyle="--",
         label="v2: 1-comp Weibull")

# Normal component alone (the "did everything right" curve)
ax2.plot(t_zoom, S_normal(t_zoom), color="#9C27B0", linewidth=1.5, linestyle="-.",
         label=f"Component 2 only (β={BETA_N}, η={ETA_N}d)\n'if done right' baseline")

# Annotate early cluster peak hazard
t_peak = ETA_E * ((BETA_E - 1) / BETA_E) ** (1 / BETA_E)
ax2.axvline(t_peak, color="#FF7043", linewidth=0.8, linestyle=":", alpha=0.7)
ax2.text(t_peak + 3, 0.85, f"Early peak\n≈{t_peak:.0f}d", fontsize=8, color="#FF7043")

ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
ax2.text(5, 0.47, "S=0.50", fontsize=8, color="gray")

ax2.set_xlim(0, 300)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("Days since installation", fontsize=11)
ax2.set_ylabel("Survival probability S(t)", fontsize=11)
ax2.set_title("Zoom 0–300d  (early cluster region)", fontsize=11)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(alpha=0.25)

# ── Key metrics annotation ────────────────────────────────────────────────────
km_median = kmf.median_survival_time_

def weibull_median(eta, beta):
    return eta * math.log(2) ** (1 / beta)

def mix_median(mx=2000):
    lo, hi = 0.0, float(mx)
    for _ in range(200):
        mid = (lo + hi) / 2
        if S_2comp(mid) > 0.5: lo = mid
        else: hi = mid
    return (lo + hi) / 2

mix_med = mix_median()
v2_med  = weibull_median(ETA_1, BETA_1)
v2n_med = weibull_median(ETA_N, BETA_N)

textstr = (
    f"Medians:\n"
    f"  KM empirical:    {km_median:.0f}d\n"
    f"  v3 2-comp mix:   {mix_med:.0f}d\n"
    f"  v2 1-comp:       {v2_med:.0f}d\n"
    f"  Component 2 only:{v2n_med:.0f}d"
)
fig.text(0.01, 0.02, textstr, fontsize=9, family="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 1])
out_path = os.path.join(OUT_DIR, "weibull_vs_km.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {out_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
print(f"\nMedians:")
print(f"  KM empirical:     {km_median:.0f}d")
print(f"  v3 2-comp:        {mix_med:.0f}d")
print(f"  v2 1-comp:        {v2_med:.0f}d")
print(f"  Component 2 only: {v2n_med:.0f}d")

# Goodness of fit: compare S_2comp and S_1comp to KM at selected time points
print(f"\nGoodness of fit (model S(t) vs KM S(t)):")
print(f"{'t':>6}  {'KM':>7}  {'2-comp':>8}  {'err_2c':>8}  {'1-comp':>8}  {'err_1c':>8}")
check_t = [50, 80, 100, 120, 150, 200, 250, 300]
for tc in check_t:
    idx = np.searchsorted(km_t, tc, side="right") - 1
    if idx < 0 or idx >= len(km_s): continue
    km_val = km_s[idx]
    s2 = S_2comp(tc)
    s1 = S_1comp(tc)
    print(f"{tc:>6}d  {km_val:>7.4f}  {s2:>8.4f}  {s2-km_val:>+8.4f}  {s1:>8.4f}  {s1-km_val:>+8.4f}")
