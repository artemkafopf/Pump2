# Vt TTF ~ ESP operating covariates — findings & model

**Date:** 2026-07-22 · **Field:** Ватьёганское (Vt), sour + nonsour
**Estimand:** cause-specific genuine ESP failure (ГТМ/ППР pulls and running pumps censored)
**Clock:** `t_cal` (calendar) primary, `t_mix` (op-time) / `t_vol` (cumulative volume) as
mechanism clocks · **Reporting:** RMST(0) + MRL(0) / median, never B50
**Code:** `backend/analysis/workflows/production_risk/{kpod_features,vt_ttf_covariates}.py`,
CLI `scripts/run/vt_ttf_covariates.py` · **Outputs (canonical run):** `results/production_risk_vt_ttf_covariates/2026-07-23/`
**Russian version:** `vt_ttf_operating_covariates_findings_ru.md`

---

## 1. Headline

For Vt ESP reliability, the operating parameter that matters is **throughput (Ql)** —
higher delivered rate → shorter life — and it acts by **pacing failure via cumulative
volume pumped** (not calendar exposure, and not accelerating wear-out to a threshold).
Everything else is downstream of that:

- **Frequency and Kpod** are the same throughput signal (one affinity-law chain: speed →
  flow → Kpod = Ql/Qnom); they are collinear with Ql and not deployable on the plan side.
- **There is no Kpod U-shape and no "safe valley"**; low Kpod is *not* harmful.
- **Sour vs nonsour differ fundamentally**: operating parameters act only in **nonsour**;
  in **sour**, H₂S corrosion dominates and swamps the mechanical effect.
- The apparent **brt-vs-slb "contractor" gap is largely rate** (slb runs harder); a small,
  unexplained residual remains and is kept in the model empirically.

---

## 2. Population and per-stratum baselines

**Vt: 626 runs, 313 events, 6 strata** (h2s_class × contractor). Baselines are a Weibull
fitted per stratum on `t_cal` (`per_stratum_survival_ttf.csv`):

| stratum | runs | events | Weibull β | η (d) | RMST(0) | mean TTF | median TTF |
|---|---|---|---|---|---|---|---|
| Vt_nonsour_brt | 231 | 81 | 0.82 | 897 | 964 | 1001 | **573** |
| Vt_nonsour_slb | 189 | 99 | 0.81 | 482 | 531 | 542 | **306** |
| Vt_nonsour_oth | 70 | 23 | 0.87 | 429 | 457 | 460 | **282** |
| Vt_sour_brt | 40 | 28 | 1.07 | 160 | 155 | 155 | **114** |
| Vt_sour_slb | 41 | 36 | 1.04 | 143 | 141 | 141 | **101** |
| Vt_sour_oth | 55 | 46 | 1.17 | 92 | 87 | 87 | **67** |

Two structural facts read straight off this table:

- **Sour lives ~3–4× shorter than nonsour** (median 67–114 vs 282–573 d), and has a
  **different hazard shape**: nonsour β ≈ 0.8 (< 1, decreasing hazard / early-failure
  dominated); sour β ≈ 1.0–1.2 (≈ constant / mild aging). This is the H₂S-corrosion vs
  mechanical-wear signature. (These are **marginal** per-stratum β, no covariates; the
  **conditional** v3.1 β with Ql+contractor held are higher — nonsour ≈1.15, sour ≈1.37,
  see §3/§4.1: the marginal nonsour β<1 is frailty, not an aging shape.)
- Coverage: telemetry (Kpod/Ql) covers ~145/141 of the thick nonsour strata and ~30–47 of
  the others; КВЧ is thin (≤ 26/stratum). h2s-continuous exists only as well/pad medians.

---

## 3. The model (v3.1)

`TTF(well) = M0(h2s) · (clip(Ql, lo, hi) / Ql_ref)^γ · contractor_mult`
(`revised_operating_model_v2.csv`, top of `TTF_ADJUSTMENT_EQUATION.md`).

The Ql shape is **log with limits**: Ql winsorized at the class q05–q95 before the log.
Chosen by AIC over linear / log / log+quadratic / spline / hinge — capping wins (ΔAIC −3.1
vs plain log) because it removes the low-Ql leverage (one Ql≈1 run) and the >q95
saturation; with the caps, γ **matches the empirical binned-median slope** (−0.31 vs
−0.32), resolving the earlier "AFT too flat" miscalibration. Outside the caps the effect
is held flat (no extrapolation beyond observed support). M0 is the median at Ql_ref for
the reference contractor (brt). Fitted per h2s class (Weibull AFT, capped
`log_ql + contractor`):

| h2s | term | life multiplier | 95% CI | p | caps (m³/d) |
|---|---|---|---|---|---|
| **nonsour** | Ql (per e-fold, capped) | **0.74** | [0.62, 0.87] | 0.0003 | [47, 823] |
| nonsour | contractor : slb | 0.79 | [0.59, 1.06] | 0.12 | |
| nonsour | contractor : oth | 0.54 | [0.35, 0.84] | 0.007 | |
| **sour** | Ql (per e-fold, capped) | 1.07 | [0.83, 1.37] | 0.61 (null) | [73, 779] |
| sour | contractor : slb | 1.05 | [0.68, 1.61] | 0.83 | |
| sour | contractor : oth | 0.64 | [0.44, 0.94] | 0.023 | |

**Vt_nonsour:** `TTF = 417 · (clip(Ql,47,823)/246)^(−0.31) · {brt 1, slb 0.79, oth 0.54}` d
(median). Higher throughput shortens life (0.74× per e-fold of capped Ql). The contractor
term absorbs the residual brt/slb difference that Ql alone under-predicts.

**Vt_sour:** `TTF = 114 · {brt 1, slb 1.05, oth 0.64}` d — **Ql is null under every shape
tried** (H₂S corrosion dominates, throughput irrelevant); the contractor term carries the
real sour_oth shortfall (0.64, p = 0.02).

**Full survival function** — `S(t|x) = exp(−(t/η(x))^β)`, `η(x) = η_ref · (clip(Ql,lo,hi)
/Ql_ref)^γ · contractor_mult`, `γ = ln(life_mult)`. Conditional Weibull params: **nonsour
β = 1.15, η_ref = 574 d** (at Ql_ref 246); **sour β = 1.37, η_ref = 149 d** (at Ql_ref 346).
These conditional β (≈ constant hazard) differ from the marginal per-stratum β < 1 of §2 —
the marginal < 1 is frailty across wells, not a within-profile aging shape.

**What to report as TTF — MRL(0) and RMST(0, τ), covariate-explicit.** v3.1 is an AFT
model, so all covariates enter through one time-scale factor
`φ(x) = contractor_mult · (clip(Ql,lo,hi)/Ql_ref)^γ` (φ = 1 at brt, Ql = Ql_ref), giving
`η(x) = η_ref·φ(x)`. Every reporting quantity then follows in closed form:

```text
median(x)   = φ(x) · η_ref · (ln2)^(1/β)
MRL(0|x)    = φ(x) · η_ref · Γ(1+1/β)                       (mean; whole tail)
RMST(0,τ|x) = φ(x) · η_ref · Γ(1+1/β) · P(1/β, (τ/(η_ref·φ(x)))^β)   [P = reg. lower inc. Γ]
            = φ(x) · RMST_ref(0, τ/φ(x))
```

median and MRL(0) scale **linearly** in φ(x); RMST(0, τ) is **sub-linear** — the horizon τ
is fixed, so longer-lived wells push more life past τ and RMST saturates below φ·mean. At
Ql = Ql_ref (τ = 730 d):

| class | contractor | φ | median (d) | MRL(0)=mean (d) | RMST(0,730 d) |
|---|---|---|---|---|---|
| nonsour | brt | 1.00 | 417 | 546 | 425 |
| nonsour | slb | 0.79 | 330 | 433 | 370 |
| nonsour | oth | 0.54 | 225 | 295 | 279 |
| sour | brt | 1.00 | 114 | 136 | 136 |
| sour | slb | 1.05 | 119 | 143 | 143 |
| sour | oth | 0.65 | 74 | 88 | 88 |

**Report RMST(0, 730 d) as the headline TTF** — it never extrapolates past the data and is
always estimable; carry MRL(0)=mean beside it (full expected life, but the nonsour mean
draws ~half its days from the fitted tail beyond 1 y), and keep the median as a descriptor
only, never as *the* TTF. For a specific pump's remaining life use MRL(t) at its current
age, not the population median. (Sour's short life sits inside τ, so its RMST ≈ mean.)

**Why Ql, and why the contractor term is kept:**
- Ql (throughput) is the operating driver; frequency and Kpod are the same signal and are
  *not* available on the ТМ-06 plan side, so Ql is the deployable covariate (§4).
- The brt/slb baseline gap (2×) is **largely rate**: slb runs harder (Ql 305 vs 193 m³/d).
  But a residual slb ≈ 0.79 multiplier persists at matched rate, and it **replicates on Ya
  almost exactly (0.805, p = 0.008 on 689 events)** — so it is a real, cross-field effect,
  not Vt noise (`replicate_v31_on_ya.csv`). The competing-risks test
  (`informative_censoring_test.csv`) attributes **about a third** of it to the
  ГТМ-censoring estimand (Aalen–Johansen CIF gap 5–12 % smaller than the KM gap; all-cause
  slb multiplier 0.87 vs cause-specific 0.79); the remaining ~0.87 is unattributed
  (equipment/make/service, inseparable). Status: **replicated, partially explained**.
- One more structural check: sour runs at *higher* Ql than nonsour (median 346 vs 246)
  while living 3–4× shorter, so an h2s-unstratified Vt fit inflates the Ql coefficient
  (HR/SD 1.32 vs 1.17) — the per-class fit is required, as done here.
- **The Ql exponent replicates too**: Ya_nonsour (689 events) gives γ = −0.255
  [−0.33, −0.19], p < 1e-5 — overlapping Vt's −0.305. The throughput relation is not
  Vt-specific.

**Calibration** (`model_consistency_checks.csv`, `model_validation_v31.png`): predicted vs
observed medians on the *same Ql-covered sample* land within ±10 % for **4 of 6 strata**
(nonsour brt 0.94 / slb 0.98, sour brt 0.90 / slb 1.04); the thin oth groups (19/32
events) are flagged (1.49 / 0.83). Ql concordance is ~0.58–0.61, so this remains a
*directional* model — good at the group level, not a precise per-well predictor.

---

## 4. Operating-parameter findings

### 4.1 Throughput is the driver, and it *paces* failure (not wear-out)

Fitting each covariate on three clocks within nonsour
(`clock_mechanism_throughput_vs_exposure.csv`):

| covariate (nonsour) | calendar | op-time | cumulative volume |
|---|---|---|---|
| freq_run | 1.27 (p=0.008) | 1.26 (p=0.015) | **1.00 (p=0.98)** |
| kpod_run | 1.34 (p=0.001) | 1.29 (p=0.006) | **0.90 (p=0.28)** |
| log_ql | 1.29 (p=0.023) | 1.18 (p=0.19) | 0.39* |

The rate covariates persist on the op-time clock but go **null on the cumulative-volume
clock** — failure timing is governed by *total volume pumped*, not calendar exposure. That
nulling is the primary mechanism evidence. (\*Ql reverses on the volume clock because it is
mechanically inside the volume sum — read freq/Kpod as the clean discriminators.)

**And it is not accelerating wear-out** (no "damage budget" threshold). Two Weibull shapes
must be distinguished: the **marginal** per-stratum shape (no covariates,
`weibull_shape_by_clock.csv`) is β < 1 on every clock (nonsour 0.80 / 0.67 / 0.83) — but
that is **frailty**: a mixture of wells at different η looks like a decreasing hazard,
early-failure dominated at the population level. The **conditional** shape (v3.1, Ql +
contractor held) is β ≈ **1.15** (nonsour) / **1.37** (sour) — near-constant hazard.
Neither is strong wear-out (which needs β ≫ 1, not ~1). So the mechanism is **throughput-
paced early failure** — a pump moving more fluid reaches its failure sooner — not
accumulation to a threshold.

### 4.2 Frequency and Kpod — same chain, no Kpod U-shape, low Kpod not harmful

- **Frequency** is the tightest *single* operating signal (HR 1.02/Hz, p 0.03, replicates
  on Ya p 0.008) and subsumes Kpod when both are entered — but it needs telemetry + Qnom
  and is not on the plan side, so it is not in the deployable model.
- **No Kpod U-shape / no safe valley.** Quadratic Cox β2 = −0.29 (concave, not convex);
  binned event rate rises monotonically; spline risk-minimising Kpod = 0.16 (bottom of
  range). The physics prior (Kpod > 1 bad, valley near 0.9) is **not supported**.
- **Low Kpod is NOT harmful.** The guarded `frac_days_kpod_below_0.7` dose is robustly
  *protective* (HR 0.94 per +10 % days, p 0.002; stays protective adjusting for level,
  frequency and Ql). Checked and rejected as artifact: unguarded also protective (no
  death-tail leakage), all-cause also protective (not a ГТМ effect), Kpod computed
  canonically. It is the cross-well confound: low Kpod ↔ low-throughput well ↔ longer life
  (`kpod_underload_harm_test.csv`).
- **Freq-adjusted Kpod fails the permutation null** (plain p 0.005, freq-adjusted p 0.21):
  the signal is delivered *rate*, not true hydraulic off-design.
- All Kpod effects are **run-developed, not install-setpoints** — the leakage-proof t0
  (first-30-op-day) versions of both Kpod and frequency are null.

### 4.3 Chemistry — GLF null, H₂S a threshold not a gradient

- **GLF (gas-liquid ratio) is null on all three fields** (Vt/Ya/Mc), re-confirming
  `project_cox_glf_null` on the gassiest field.
- **H₂S** is the sour/nonsour stratum split (from the Свод «Кислый/Некислый» label). The
  data confirm the effective boundary is **~120 mg/l**, not the legacy 10 mg/l: Vt_sour is
  **122–1272 mg/l**, Vt_nonsour ≤ ~1.5 mg/l (q90) — a bimodal distribution, so the exact
  cutoff reclassifies ~2 wells (`h2s_sour_threshold.csv`). **~30 % of Vt wells-with-data
  exceed 120 mg/l** (~17 % of all Vt wells); Vt is the only field with substantial high
  H₂S (`h2s_well_prevalence_*.csv`). **Within the sour stratum, more H₂S is not detectably
  worse** (log-H₂S HR 0.96–0.97, n.s.) — being sour is a threshold effect, the concentration
  gradient above it is saturated.
- **КВЧ** (mechanical impurities): thin coverage (~22 % of Vt wells); descriptive only.

---

## 5. Contractor (brt vs slb) — the 2× gap is largely rate

The two thick nonsour strata differ ~2× (brt median 573 vs slb 306 d). Diagnostics:

- **Not cohort, not H₂S, not GLF.** The gap persists/widens in the 2024+ install cohort;
  both are genuinely nonsour (~0.1 mg/l) with identical GLF (`contractor_gap_*.csv`).
- **Not more ГТМ.** slb has *fewer* ГТМ (27 % of pulls vs 35 %) and *more* genuine failures
  (52 % vs 35 %), a 1.7× higher unadjusted failure rate — real reduced reliability, not
  intervention scheduling (`contractor_pull_composition.csv`).
- **It is rate.** slb runs its wells ~1.6× harder (Ql 305 vs 193 m³/d) — same *Kpod*
  (identical loading distributions) but higher *absolute* throughput, because slb sizes
  pumps proportionally bigger (q_nom 321 vs 250; `contractor_ql_kpod_overlap.png`). Matched
  within the overlapping Ql band **100–500 (92 events), Cox HR_slb = 1.02 (p 0.91)** — brt
  and slb are indistinguishable at matched rate (`contractor_rate_band_comparison.csv`).
- **Equipment differs but is brand-confounded** — Borets ЭЦНДИК/«ЛЧ» vs REDA MT/S/D,
  collinear with the contractor, so no single component isolates the cause
  (`contractor_equipment_exec_group.csv`).
- **Residual — now quantified and replicated:** at matched rate a slb ≈ 0.79 life
  multiplier persists (n.s. on Vt alone) and **replicates on Ya at 0.805 (p = 0.008)** —
  a real cross-field effect. The competing-risks test (Aalen–Johansen CIF with ГТМ as a
  competing event + all-cause refit) attributes **~⅓ of it to the ГТМ-censoring estimand**
  (slb mult 0.79 → 0.87 all-cause; CIF gap 5–12 % smaller than KM); the remaining ~0.87 is
  unattributed (equipment/make/service bundle). Kept in the model (§3), status upgraded
  from "unexplained" to "replicated, partially explained"
  (`informative_censoring_test.csv`, `replicate_v31_on_ya.csv`).

---

## 6. Honesty gates

- **Replication (Ya, Mc 2024+).** Every covariate keeps its Vt sign; **Ql replicates
  strongly** (Vt 1.19 p 0.075, Ya 1.30 p<1e-4, Mc 3.08 p 0.002) and **frequency replicates**
  (Ya p 0.008); **GLF null on all three** (`gate_replication_ya_mc.csv`). The **full v3.1
  design replicates on Ya** too: γ −0.255 [−0.33, −0.19] vs Vt −0.305, slb multiplier
  0.805 (p 0.008) vs 0.792 (`replicate_v31_on_ya.csv`).
- **Complete-case bias.** Kpod coverage differs by outcome (77 % of events vs 61 % of
  censored) but covered-vs-uncovered KM RMST(0) is nearly identical (274.8 vs 279.8 d) —
  the modelled subset is not an early-life-biased slice (`gate_km_kpod_coverage.csv`).
- **n_failures assertion** vs shipped esp_models: small deltas from the reworked Свод, no
  anomaly (`gate_nfailures_vs_esp_models.csv`).
- **Observational.** Wells are not randomised to rate/speed; every multiplier is a relation,
  not a causal lever (the `project_ion_load_vs_concentration` "load-was-qliq×uptime"
  precedent applies). Contractor multiplier mechanism unidentified.
- **Temporal out-of-sample holdout** (`temporal_holdout_validation.csv`): fit on installs
  <2024 (72 events), evaluated on the 2024+ cohort (84 events). **γ transfers** (train
  −0.36 vs test-refit −0.26, same sign/CI overlap) and overall calibration is fair
  (predicted 335 vs observed 389 d, ratio 0.86; S(365) 0.47 vs 0.51). But the pre-2024-fit
  model is **~20–30 % pessimistic per contractor on the 2024+ cohort** (brt 0.71, slb
  0.78) — a vintage effect (newer installs live longer, cf. the 2024+ cohort table). oth
  is not evaluable (2 pre-2024 runs). **Implication:** the *relation* (γ) transfers; the
  *baseline* (M0) drifts with vintage — if this ever feeds the forecast, refit M0 on the
  recent cohort and carry γ.
- **Sour holdout is untestable** — the Ql-covered sour population is essentially all
  2024+ (1 pre-2024 run, 0 events); stated, not skipped.
- **Bootstrap CIs on the composed deliverables** (`bootstrap_model_cis.csv`,
  cluster-by-well, n=200, 200/200 draws converged): nonsour γ −0.31 [−0.51, −0.14], M0
  427 [345, 544] d, slb mult 0.78 [0.59, 1.01]; sour γ 0.08 [−0.24, 0.44] (null
  confirmed). The composed per-contractor medians' CIs **cover the observed medians in
  5/6 strata** (brt 454 [365, 574] vs obs 479; slb 311 [269, 383] vs obs 320; sour all
  covered) — only thin nonsour_oth (obs 163 vs CI [168, 401]) sits just outside,
  consistent with its flagged miss.
- **Calibration.** §3 caveat: directional model, concordance ~0.58, in-sample brt/slb
  within ±10 % after v3.1.

---

## 7. Figures & tables

**Key figures** (`figures/`):
- `model_validation_v31.png` — **the model-validation triptych**: Cox–Snell calibration per
  h2s class, predicted-vs-observed medians per stratum (same sample), and the Ql shape
  check (spline vs log-with-limits, cap positions marked).
- `nonsour_km_by_{ql,freq,kpod}_bin.png` — KM survival by Ql / frequency / Kpod band
  (each legend gives n, events, median); `ya_km_by_{ql,freq,kpod}_bin.png` — same on Ya
  (replication field; note Kpod is non-monotone on Ya — it is a rate proxy, not the driver).
- `sour_km_by_{ql,freq,kpod}_bin.png` — same on **Vt_sour** (coarse 3-band, ~110 events):
  freq/Kpod show **no monotone ordering** and Ql orders *weakly the opposite way* to nonsour
  (low-Ql sour wells fail fastest: 65 vs 117 d) but n.s. (p≈0.6, confounded) — confirming the
  §3 finding that the operating covariates are **null in sour** (H₂S corrosion dominates).
- `nonsour_ql_bin_trend.png` — median TTF per Ql bin vs AFT fit and empirical (steeper) trend.
- `nonsour_km_ql_fit.png` — KM (pooled/brt/slb) vs v2 (Ql) and v3 (Ql+contractor) fit.
- `contractor_ql_kpod_overlap.png` — brt/slb Ql (shifted) and Kpod (identical) distributions.
- `km_weibull_per_stratum.png` — per-stratum KM + fitted Weibull.
- `kpod_spline_logHR.png`, `kpod_binned_km.png`, `covariate_corr_heatmap.png`, support hists.

**Key tables** (`tables/`): `revised_operating_model_v2.csv` (the model),
`per_stratum_survival_ttf.csv` (baselines), `clock_mechanism_throughput_vs_exposure.csv`,
`weibull_shape_by_clock.csv`, `contractor_rate_band_comparison.csv`,
`contractor_pull_composition.csv`, `kpod_underload_harm_test.csv`, `h2s_*`,
`gate_replication_ya_mc.csv`. Deliverable: `TTF_ADJUSTMENT_EQUATION.md`.

## 8. Reproduction

```bash
python scripts/run/vt_ttf_covariates.py            # full run → results/production_risk_vt_ttf_covariates/<date>/
cd backend && python -m pytest tests/test_vt_ttf_covariates.py -q
```
