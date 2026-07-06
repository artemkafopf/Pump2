"""
Fit Cox Proportional Hazards model for UVCh chemical + contractor risk factors.

Model: Vt0_cox_kip_cl_h2s_ca_so4_nt  (v5)
Data:  mart__vt_freq55, Vt wells, КИП > 0.5
Covariates:
  cl_log   : log1p(cum_chloride_load_kg / run_days)  — continuous (collinear w/ H2S, likely NS)
  h2s_flag : 1 if h2s_proxy_mg_l >= 100 mg/L        — binary threshold
  ca_log   : log1p(cum_calcium_load_kg / run_days)   — continuous, scale proxy
  so4_log  : log1p(cum_sulfate_load_kg / run_days)   — continuous, scale/souring proxy
  is_nt    : 1 if contractor = НТ (Новые технологии) — binary, selection-bias caveat

Outputs:
  docs/uvch_claude/cox_params.json   — coefficients, HRs, CIs, concordance
  docs/uvch_claude/cox_survival.png  — baseline survival + covariate effect curves
"""
import math, json, os, sqlite3
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH  = r"d:\GitHub\Pump2\data\warehouse\pump2.db"
OUT_DIR  = r"d:\GitHub\Pump2\docs\uvch_claude"
os.makedirs(OUT_DIR, exist_ok=True)

NT_KEYWORDS = ("нов", "новые", "new tech", "нт")

# ── Load data ─────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT
        well, well_key,
        contractor,
        h2s_proxy_mg_l,
        cum_chloride_load_kg,
        cum_calcium_load_kg,
        cum_sulfate_load_kg,
        run_days,
        ttf_true_best_days,
        event,
        freq_w_mean
    FROM mart__vt_freq55
    WHERE field = 'Vt'
      AND run_days > 0
""", conn)
conn.close()

print(f"Loaded {len(df)} Vt runs from mart")

# ── КИП filter (Кэкспл > 0.5) ─────────────────────────────────────────────────
df = df[df["ttf_true_best_days"] / df["run_days"] > 0.5].copy()
print(f"After КИП > 0.5 filter: {len(df)} rows, {df['event'].sum():.0f} events")

# ── Feature engineering ───────────────────────────────────────────────────────
df["cl_daily"]  = df["cum_chloride_load_kg"].fillna(0) / df["run_days"]
df["ca_daily"]  = df["cum_calcium_load_kg"].fillna(0) / df["run_days"]
df["so4_daily"] = df["cum_sulfate_load_kg"].fillna(0) / df["run_days"]

df["cl_log"]   = np.log1p(df["cl_daily"])
df["ca_log"]   = np.log1p(df["ca_daily"])
df["so4_log"]  = np.log1p(df["so4_daily"])
df["h2s_flag"] = (df["h2s_proxy_mg_l"].fillna(0) >= 100).astype(int)
df["is_nt"]    = df["contractor"].fillna("").str.lower().apply(
    lambda x: 1 if any(kw in x for kw in NT_KEYWORDS) else 0
)

# Use ttf_true_best_days as duration
df["duration"] = df["ttf_true_best_days"]

print(f"\nCovariate breakdown:")
print(f"  H2S >= 100 mg/L : {df['h2s_flag'].sum()} rows ({df['h2s_flag'].mean()*100:.1f}%)")
print(f"  НТ contractor   : {df['is_nt'].sum()} rows ({df['is_nt'].mean()*100:.1f}%)")
print(f"  cl_log  range   : {df['cl_log'].min():.2f} – {df['cl_log'].max():.2f} (median {df['cl_log'].median():.2f})")
print(f"  ca_log  range   : {df['ca_log'].min():.2f} – {df['ca_log'].max():.2f} (median {df['ca_log'].median():.2f})")
print(f"  so4_log range   : {df['so4_log'].min():.2f} – {df['so4_log'].max():.2f} (median {df['so4_log'].median():.2f})")

# ── Well-level hold-out split (80/20) ─────────────────────────────────────────
wells = df["well_key"].unique()
rng = np.random.default_rng(42)
rng.shuffle(wells)
n_train = int(len(wells) * 0.80)
train_wells = set(wells[:n_train])

df_train = df[df["well_key"].isin(train_wells)].copy()
df_test  = df[~df["well_key"].isin(train_wells)].copy()
print(f"\nTrain: {len(df_train)} rows ({len(train_wells)} wells, {df_train['event'].sum():.0f} events)")
print(f"Test:  {len(df_test)} rows ({len(wells)-n_train} wells, {df_test['event'].sum():.0f} events)")

# ── Fit Cox model ─────────────────────────────────────────────────────────────
cox_cols = ["duration", "event", "cl_log", "h2s_flag", "ca_log", "so4_log", "is_nt"]
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df_train[cox_cols], duration_col="duration", event_col="event")

print("\nCox PH model summary:")
cph.print_summary()

# ── Proportional hazard test (Schoenfeld residuals) ──────────────────────────
ph_test = proportional_hazard_test(cph, df_train[cox_cols], time_transform="rank")
print("\nProportional Hazard test (Schoenfeld):")
print(ph_test.summary)

# ── Hold-out concordance ──────────────────────────────────────────────────────
c_train = cph.concordance_index_
c_test  = cph.score(df_test[cox_cols], scoring_method="concordance_index")
print(f"\nC-index  train: {c_train:.4f}  hold-out: {c_test:.4f}")

# ── Extract parameters ────────────────────────────────────────────────────────
params = cph.params_.to_dict()
hazard_ratios = cph.hazard_ratios_.to_dict()
ci = cph.confidence_intervals_

cox_result = {
    "model": "Vt0_cox_kip_cl_h2s_ca_so4_nt",
    "n_train_rows": int(len(df_train)),
    "n_train_wells": int(len(train_wells)),
    "n_train_events": int(df_train["event"].sum()),
    "c_index_train": round(float(c_train), 4),
    "c_index_test": round(float(c_test), 4),
    "coefficients": {k: round(float(v), 5) for k, v in params.items()},
    "hazard_ratios": {k: round(float(v), 4) for k, v in hazard_ratios.items()},
    "ci_lower_95": {k: round(float(ci.loc[k, "95% lower-bound"]), 4) for k in params},
    "ci_upper_95": {k: round(float(ci.loc[k, "95% upper-bound"]), 4) for k in params},
    "ph_test_pvalues": {
        row["test_statistic"]: round(float(row["p"]), 4)
        for _, row in ph_test.summary.iterrows()
    } if hasattr(ph_test, "summary") else {},
}

out_json = os.path.join(OUT_DIR, "cox_params.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(cox_result, f, indent=2, ensure_ascii=False)
print(f"\nSaved parameters to {out_json}")

# ── Baseline survival plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: baseline survival
cph.baseline_survival_.plot(ax=axes[0], legend=False)
axes[0].set_title("Baseline Survival (Vt0_cox_kip)")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("S(t)")
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)

# Plot 2: covariate effects — show profiles at typical values
t = np.arange(0, 500)
med = {"cl_log": df["cl_log"].median(), "ca_log": df["ca_log"].median(),
       "so4_log": df["so4_log"].median()}

def make_row(**overrides):
    d = dict(cl_log=med["cl_log"], h2s_flag=0, ca_log=med["ca_log"],
             so4_log=med["so4_log"], is_nt=0)
    d.update(overrides)
    return pd.DataFrame([d])

baseline_row = make_row()
h2s_row      = make_row(h2s_flag=1)
nt_row       = make_row(is_nt=1)
ca_row       = make_row(ca_log=df["ca_log"].quantile(0.90))
so4_row      = make_row(so4_log=df["so4_log"].quantile(0.90))

for row, label, color in [
    (baseline_row, "Baseline (медиана, нет H2S/НТ)", "blue"),
    (h2s_row,      "H2S >= 100 mg/L", "red"),
    (nt_row,       "НТ contractor", "purple"),
    (ca_row,       "High Ca (P90)", "orange"),
    (so4_row,      "High SO4 (P90)", "brown"),
]:
    sf = cph.predict_survival_function(row)
    axes[1].plot(sf.index, sf.values, label=label, color=color)

axes[1].set_title("Cox PH: Covariate Effects on Survival")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("S(t)")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "cox_survival.png")
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"Plot saved to {fig_path}")

# ── Human-readable summary ────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Vt0_cox_kip_cl_h2s_ca_so4_nt  —  HR summary (v5)")
print(f"{'='*55}")
for cov in ["cl_log", "h2s_flag", "ca_log", "so4_log", "is_nt"]:
    hr  = hazard_ratios.get(cov, float("nan"))
    lo  = cox_result["ci_lower_95"].get(cov, float("nan"))
    hi  = cox_result["ci_upper_95"].get(cov, float("nan"))
    b   = cox_result["coefficients"].get(cov, float("nan"))
    print(f"  {cov:12s}  beta={b:+.4f}  HR={hr:.3f}  95%CI [{lo:.3f}–{hi:.3f}]")
print(f"  C-index (hold-out): {c_test:.4f}")
