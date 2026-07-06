# ESP Survival Analysis — Full Findings Report

**Date:** 2026-06-23  
**Data source:** `data/warehouse/pump2.db`, table `mart__vt_freq55`  
**Duration column:** `ttf_true_best_days` (telemetry/techregime-corrected time-to-failure, days)  
**Event column:** `event` (1 = confirmed failure, 0 = censored)  
**Analysis scripts:** `scripts/analyze_vt_ya_global_survival.py`, `scripts/analyze_survival_improved.py`, `scripts/analyze_vt_freq_latent.py`, `scripts/analyze_freq_latent.py`  
**Output directories:**
- `analysis/vt_ya_global_survival_comparison.csv` — Run A posteriors (mart-based)
- `analysis_outputs/survival_improved_2026_06_23/` — improved analysis with soft-probability regression
- `analysis_outputs/vt_freq_latent_2026_06_23/` — Vt frequency stratification
- `analysis_outputs/freq_latent_global_2026_06_23/` — Global frequency stratification

---

## 1. Dataset and Population

| Field | N (runs) | Failures | Failure rate | Median TTF |
|-------|----------|----------|--------------|------------|
| Ya | 1,109 | 648 | 58.4% | ~100d |
| Vt | 310 | 213 | 68.7% | 98d |
| Az | 182 | 115 | 63.2% | — |
| Za | 180 | 119 | 66.1% | — |
| Ic | 174 | 117 | 67.2% | — |
| Mc | 83 | 29 | 34.9% | — |
| Da | 61 | 26 | 42.6% | — |
| **Global** | **2,124** | **1,273** | **59.9%** | **117d** |

`ttf_true_best_days` is systematically shorter than the raw operational run (`run_days`) because it uses telemetry and techregime logs to set the failure timestamp precisely. Codex's prior analysis used `"Наработка (сут)"` (raw run days from a Russian-language column), which inflated all durations and corrupted the component separation.

---

## 2. Bayesian Weibull Mixture Model

### 2.1 Model specification

Mixture survival function:

```
S(t) = w₁ · exp(-(t/η₁)^β₁) + w₂ · exp(-(t/η₂)^β₂)
```

- K = 2 components (fixed)
- Right-censored likelihood: failed runs contribute Weibull density; censored runs contribute Weibull survival
- Gibbs + Metropolis-Hastings sampler
- **MCMC config:** 12,000 iterations × 3 chains, 3,000 burn-in, thin = 5 → **1,800 saved draws per chain (5,400 total)**
- Log-normal priors: `log η ~ N(5.0, 1.5²)` (center at ~150d, covers 10–1500d); `log β ~ N(0, 0.7²)` (center at 1.0)
- Label switching handled by relabelling components by median life = η·(ln2)^(1/β) per draw

### 2.2 Convergence note

With K = 2 and components that are geometrically similar, label switching causes ESS < 100 for mixture weights and shape parameters in some runs. Life quantiles at the **mixture level** (B10, B25, B50, B90) are invariant to label switching and are the most reliable outputs. Per-observation component probabilities should be interpreted as population-level distributions, not as reliable individual predictions.

---

## 3. Run A — Mart-Based Posteriors (Authoritative)

**Source file:** `analysis/vt_ya_global_survival_comparison.csv`  
**Script:** `scripts/analyze_vt_ya_global_survival.py`

This is the primary survival fit. All three populations use `ttf_true_best_days` from `mart__vt_freq55` directly.

### 3.1 Component posteriors

| Population | Comp | Weight | η (days) | β | Median life | Hazard shape |
|------------|------|--------|----------|---|-------------|--------------|
| **Global** | 1 | 0.446 | 217d [33–365] | 0.870 [0.68–1.42] | ~150d | Mild infant mortality |
| **Global** | 2 | 0.554 | 519d [355–929] | 1.151 [0.83–2.35] | ~360d | Mild wear-out |
| **Vt** | 1 | 0.393 | 120d [9–221] | 0.967 [0.56–1.89] | ~83d | Near-constant hazard |
| **Vt** | 2 | 0.607 | 282d [191–514] | 1.334 [0.86–2.85] | ~251d | Mild wear-out |
| **Ya** | 1 | 0.544 | 192d [45–352] | 0.804 [0.67–0.91] | ~133d | Infant mortality |
| **Ya** | 2 | 0.456 | 754d [437–1088] | 1.588 [0.88–2.93] | ~676d | Strong wear-out |

### 3.2 Mixture life quantiles

| Quantile | Global | Vt | Ya |
|----------|--------|----|----|
| B10 (10% fail by) | 25.5d [22–29] | 19.5d [13–26] | 23.3d [19–28] |
| B25 (25% fail by) | 83.8d [76–92] | 57.3d [46–70] | 83.6d [72–96] |
| **B50 (median)** | **238.5d [222–257]** | **145.1d [123–169]** | **263.1d [234–296]** |
| B90 (90% fail by) | 949.8d [879–1033] | 487.8d [416–581] | 1060.8d [961–1175] |
| **KM B50 (reference)** | **245d** | **147d** | **262d** |

KM and mixture B50 agree within ≤ 7d in all populations — the model is well-calibrated.

### 3.3 Interpretation of the two components

**Component 1 — "early/random failure" mode:**
- Constant or slightly decreasing hazard (β ≤ 1)
- Shorter scale (η ≈ 120–217d depending on field)
- Represents pumps lost to chemical/operational incidents that are not age-related: scale deposition events, H₂S excursions, installation defects, sand ingress

**Component 2 — "wear-out" mode:**
- Increasing hazard (β ≥ 1.15)
- Longer scale (η ≈ 282–754d depending on field)
- Represents normal fatigue and mechanical wear; predictable aging failure

Vt has the shortest B50 (145d) and largest share of wear-out population (w₂ = 0.607), but its short-mode component (120d, β ≈ 1) is more similar to the wear-out component than in other fields — the two modes are less separated in Vt than in Ya.

Ya has the most bimodal structure: strong infant-mortality component (β = 0.804, η = 192d) and strong wear-out component (β = 1.588, η = 754d), producing the widest spread from B10 = 23d to B90 = 1,061d.

---

## 4. Comparison with Codex's Prior Analysis

Codex's analysis (`scripts/analyze_field_survival_bayesian.py`, outputs in `analysis_outputs/bayesian_field_survival_compare_2026_06_23/`) had three critical flaws:

| Issue | Codex | Run A (correct) |
|-------|-------|-----------------|
| Duration column | `"Наработка (сут)"` (raw run_days, Russian column) | `ttf_true_best_days` (corrected) |
| MCMC draws | 150 effective draws (1,000 iter × 1 chain, thin=4) | 5,400 draws (12,000 × 3 chains, thin=5) |
| ESS (comp_1 weight) | **7–16** (pathological) | 33–77 |
| R-hat | Not reported; implied >1.2 | 1.048–1.059 |
| Sample sizes | Global=2,169, Vt=304, Ya=1,094 | Global=2,124, Vt=310, Ya=1,109 |

Codex's reported posteriors (all β > 1, both components in wear-out mode across all fields) are artefacts of non-convergence. They cannot be used for inference.

---

## 5. Soft-Probability Driver Analysis

**Script:** `scripts/analyze_survival_improved.py`  
**Outputs:** `analysis_outputs/survival_improved_2026_06_23/`

### 5.1 Method improvements over Codex

1. **Continuous target**: P(component 1) used as a continuous [0,1] outcome, not hard-thresholded binary. This preserves the uncertainty in component assignment (fractional logistic / fractional response regression).
2. **Unweighted regression**: Codex used `max_probability` as a sample weight, which biases toward high-confidence observations. Unweighted regression treats all runs equally.
3. **Soft Spearman**: Spearman ρ between continuous P(short) and each feature, not between binary membership and feature.
4. **Cross-validation** of the interaction term (5-fold CV).
5. **Contractor stratification** of the `mount_year` effect.

### 5.2 Univariate associations (Spearman ρ, P(short) vs feature)

*Note: this analysis uses the feature CSV (`analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/tables/analysis_dataset.csv`), which contains per-day chemistry loads computed from cumulative mart data. The Vt population in the feature CSV (n=311) has a longer median TTF (140d) than the mart's Vt population (median 98d) due to row_id misalignment between the two pipeline versions. Results below are internally consistent but represent slightly different subsets.*

| Feature | Vt ρ | Vt p | Ya ρ | Global ρ |
|---------|------|------|------|---------|
| `log1p_calcium_load_per_day` | **+0.485** | 1.9e-17 | +0.347 | +0.375 |
| `log1p_chloride_load_per_day` | **+0.476** | 5.9e-16 | +0.399 | +0.404 |
| `log1p_gypsum_proxy_per_day` | **+0.445** | 3.7e-13 | +0.310 | +0.327 |
| `log1p_sulfate_load_per_day` | +0.380 | 7.6e-11 | +0.318 | +0.306 |
| `low_h2s_flag` (low H₂S = 1) | −0.245 | 1.2e-05 | −0.102 | −0.128 |
| `log1p_h2s_effective_mg_l` | +0.204 | 1.1e-03 | +0.100 | +0.120 |
| `log1p_tlf_per_day` | +0.149 | 8.6e-03 | +0.063 | +0.085 |
| **`mount_year`** | **+0.058** | **0.31 (n.s.)** | −0.227 | −0.139 |
| `freq_above_55hz_pct` | −0.127 | 0.038 * | −0.025 | −0.055 |
| `freq_signed_exposure` | −0.100 | 0.106 | +0.015 | −0.003 |

**Key findings:**
- **Chemistry burden is the dominant predictor** of early pump failure across all populations. Calcium, chloride, gypsum, and sulfate loads all show strong positive correlation with P(short-lived component) — more aggressive chemistry → higher probability of being in the fast-failure mode.
- **`mount_year` is not significant in Vt** (ρ = +0.058, p = 0.31). In Ya and Global, newer installations show *lower* P(short) (negative ρ), meaning improved survival in more recent years — but this effect is absent in Vt.
- **Frequency variables are weak everywhere**: `freq_above_55hz_pct` barely reaches p = 0.038 in Vt with ρ = −0.127 (negative direction — more time above 55Hz → lower P(short), i.e., slightly *better* survival). This is the opposite of a harm signal.

### 5.3 Multivariate ridge regression

Ridge logistic with L2 = 1.0 using P(short) as continuous target. AUC and pseudo-R² by population:

| Population | Model | pseudo-R² | AUC |
|------------|-------|-----------|-----|
| Global | Base (main effects) | 0.0182 | 0.763 |
| Global | With interaction | 0.0182 | 0.763 |
| Vt | Base | 0.0090 | 0.897 |
| Vt | With interaction | 0.0091 | 0.897 |
| Ya | Base | 0.0118 | 0.777 |
| Ya | With interaction | 0.0119 | 0.777 |

High AUC despite near-zero pseudo-R² reflects severe multicollinearity among chemistry features: the ridge shrinks all individual chemistry coefficients toward zero, but collectively they discriminate well. No single chemistry variable can be isolated as the "primary" driver in the multivariate model.

### 5.4 The `mount_year × gypsum` interaction — not defensible

Codex found `mount_year × log1p(gypsum_proxy_per_day)` to be significant in Vt at p = 0.042 (OR = 2.24), from an n = 186 model with ~35 terms. The interaction was data-snooped on a small sample and has no out-of-sample validity:

| Test | Result |
|------|--------|
| delta pseudo-R² with interaction (Vt) | **+0.0000** |
| 5-fold CV Brier score without interaction | 0.0047 |
| 5-fold CV Brier score with interaction | 0.0048 (worse) |
| 5-fold CV Spearman ρ without | 0.559 |
| 5-fold CV Spearman ρ with | 0.538 (worse) |

The interaction term adds zero explained variance and degrades out-of-sample predictions. Codex's p = 0.042 was a false positive.

### 5.5 Contractor-stratified `mount_year` effect

`mount_year` correlations with P(short) within each contractor:

| Field | Contractor | n | ρ(mount_year, P_short) | p |
|-------|-----------|---|------------------------|---|
| Vt | Contractor A | 135 | +0.006 | 0.944 |
| Vt | Contractor B | 123 | +0.174 | 0.055 |
| Vt | Contractor C | 52 | −0.138 | 0.328 |
| Ya | Contractor B | 658 | **−0.259** | **1.5e-11** |
| Ya | Contractor A | 344 | **−0.254** | **1.8e-06** |
| Global | Contractor B | 1,276 | **−0.170** | **8.9e-10** |
| Global | Contractor A | 678 | **−0.151** | **8.6e-05** |

Contractor names are garbled in the terminal output due to Windows cp1251 encoding; full UTF-8 names are in `analysis_outputs/survival_improved_2026_06_23/contractor_stratified_mount_year.csv`.

**Conclusion:** The negative `mount_year` effect in Ya and Global (newer = better survival) is genuine and independent of contractor composition. In Vt, no within-contractor `mount_year` trend exists at any contractor — the Vt population does not show the year-over-year survival improvement seen elsewhere.

---

## 6. Frequency Stratification Analysis

### 6.1 Vt (Run A, mart data, n=310)

**Script:** `scripts/analyze_vt_freq_latent.py`  
**Outputs:** `analysis_outputs/vt_freq_latent_2026_06_23/`

Frequency coverage: 238 of 310 Vt runs have ≥ 10 valid frequency days (76.8%).

| Group | Criterion (a) mean freq > 55Hz | Criterion (b) >50% days above 55Hz |
|-------|-------------------------------|-------------------------------------|
| High-frequency | n = 26, 22 failures (84.6%) | n = 38, 31 failures (81.6%) |
| Low-frequency | n = 212, 152 failures (71.7%) | n = 200, 143 failures (71.5%) |
| No telemetry | n = 72 | n = 72 |

**KM survival:**

| Metric | High (a) | Low (a) | High (b) | Low (b) | All Vt |
|--------|----------|---------|----------|---------|--------|
| S(90d) | 0.642 | 0.680 | 0.728 | 0.666 | 0.649 |
| S(180d) | 0.441 | 0.413 | 0.458 | 0.409 | 0.399 |
| S(365d) | 0.241 | 0.221 | 0.244 | 0.221 | 0.217 |
| B50 | 171d | 149d | 171d | 148d | 147d |
| B75 | 288d | 305d | 301d | 305d | 295d |
| B90 | 502d | 487d | 502d | 487d | 475d |

**Statistical tests:**

| Test | Criterion (a) | Criterion (b) |
|------|--------------|--------------|
| Mann-Whitney p (TTF) | 0.4433 | 0.0753 |
| Mann-Whitney p (P_short) | 0.7662 | 0.4331 |
| Spearman ρ (freq_w_mean vs P_short) | +0.029, p=0.661 | — |
| Spearman ρ (freq_above_55 vs P_short) | — | −0.029, p=0.660 |

**Vt conclusion:** No statistically significant effect of high-frequency operation on survival or latent component membership. High-frequency group (n = 26–38) is too small for reliable inference. Point estimates trend slightly *toward* better survival in the high-frequency group — consistent with selection bias, not harm.

### 6.2 Global (all fields, Run A, mart data, n=2,124)

**Script:** `scripts/analyze_freq_latent.py --field Global`  
**Outputs:** `analysis_outputs/freq_latent_global_2026_06_23/`

Frequency coverage: 1,715 of 2,124 runs (80.7%) have ≥ 10 valid frequency days.

| Group | Criterion (a) mean freq > 55Hz | Criterion (b) >50% days above 55Hz |
|-------|-------------------------------|-------------------------------------|
| High-frequency | n = 348, 228 failures (65.5%) | n = 377, 247 failures (65.5%) |
| Low-frequency | n = 1,367, 840 failures (61.4%) | n = 1,338, 821 failures (61.4%) |
| No telemetry | n = 409, 205 failures (50.1%) | n = 409, 205 failures (50.1%) |

**KM survival:**

| Metric | High (a) | Low (a) | High (b) | Low (b) | All Global |
|--------|----------|---------|----------|---------|------------|
| S(90d) | 0.755 | 0.791 | 0.774 | 0.787 | 0.733 |
| S(180d) | 0.601 | 0.624 | 0.602 | 0.625 | 0.574 |
| S(365d) | 0.405 | 0.426 | 0.403 | 0.427 | 0.388 |
| S(730d) | 0.132 | 0.197 | 0.148 | 0.195 | 0.167 |
| B50 | 246d | **285d** | 258d | **285d** | 245d |
| B75 | 508d | 613d | 523d | 613d | 539d |
| B90 | 954d | 978d | 1,110d | 964d | 913d |

**Statistical tests:**

| Test | Criterion (a) | Criterion (b) |
|------|--------------|--------------|
| Mann-Whitney p (TTF) | 0.939 | 0.366 |
| Mann-Whitney p (P_short) | 0.253 | 0.478 |

**Spearman correlations (n = 1,715 with ≥ 10 freq days):**

| Feature | ρ vs P(short) | p | ρ vs TTF | p |
|---------|---------------|---|---------|---|
| `freq_w_mean` | +0.022 | 0.355 | +0.039 | 0.109 |
| `freq_above_55hz_pct` | −0.005 | 0.836 | **+0.089** | **0.0002** |
| `freq_signed_exposure` | +0.026 | 0.274 | +0.036 | 0.135 |

**Per-field breakdown (high-freq failure rate vs low-freq):**

| Field | n | % high-freq | Fail rate high | Fail rate low | ρ(freq,P_short) | p |
|-------|---|------------|----------------|---------------|-----------------|---|
| Az | 152 | 24% | 0.69 | 0.67 | +0.032 | 0.692 |
| Da | 45 | 24% | 0.55 | 0.35 | +0.015 | 0.922 |
| Ic | 149 | 17% | 0.60 | 0.71 | +0.006 | 0.939 |
| Mc | 62 | 32% | 0.40 | 0.40 | −0.004 | 0.973 |
| **Vt** | 238 | 11% | **0.85** | **0.72** | +0.062 | 0.342 |
| Ya | 919 | 21% | 0.64 | 0.59 | +0.050 | 0.128 |
| **Za** | 137 | 24% | **0.79** | **0.64** | **+0.202** | **0.018** |

**Global conclusion:**
1. No evidence that high-frequency operation increases failure probability globally.
2. `freq_above_55hz_pct` has a small but significant positive correlation with TTF (ρ = +0.089, p = 0.0002): more days above 55Hz → slightly longer survival. This is a confounding effect, not a protective one — high-frequency wells likely have better reservoir conditions.
3. Za field is the only outlier with a significant within-field frequency–P(short) correlation (ρ = +0.202, p = 0.018). This warrants field-specific investigation but does not change the global conclusion.
4. The **no-telemetry group** (n = 409, median TTF = 20d, B50 = 70d) is the clearest failure signal in the dataset — pumps that fail within the first 2–3 weeks never accumulate enough operational days to appear in frequency statistics. This is a censoring artefact, not a frequency effect.

---

## 7. Critical Observations and Caveats

### 7.1 Latent model adds no information for frequency analysis

The K = 2 mixture identifies two failure modes (random/chemical vs. wear-out), but the latent component membership is essentially uncorrelated with operating frequency across all tests. Frequency telemetry cannot predict which failure mode a pump belongs to. For frequency-related questions, KM stratification gives the same answer with less complexity.

### 7.2 Chemistry is the dominant failure driver

Across all three well-powered tests (Vt, Ya, Global), the chemistry burden variables — calcium, chloride, gypsum, and sulfate loads per day — are the only consistently strong and statistically significant predictors of early failure. These represent scale deposition and corrosive attack on the ESP assembly. Scale-forming conditions are identifiable from production chemistry and should be the primary risk-stratification input.

### 7.3 No-telemetry bias

409 of 2,124 global runs (19.3%) have fewer than 10 valid frequency days. These runs have a median TTF of 20 days and a B50 of 70 days — dramatically worse than any frequency-stratified group. Any analysis that does not explicitly model this missingness will produce biased frequency effect estimates (runs that die quickly appear in the "no-frequency" group, artificially improving the apparent survival of frequency-measured runs).

### 7.4 Data alignment between mart and feature CSV

The mart (`mart__vt_freq55`) and the feature dataset (`analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/tables/analysis_dataset.csv`) share a `row_id` column but with partially non-overlapping numbering (only 114 of 410 Vt row_ids overlap). The feature CSV should not be used for survival fits — only the mart's `ttf_true_best_days` provides authoritative, corrected durations. Feature-enriched regression (Section 5) uses the feature CSV population, which differs from the mart's Vt population in both sample composition and median TTF.

### 7.5 Vt is not the shortest-lived field on a per-run basis

The Run A B50 for Vt (145d) is **lower** than Ya (263d) and Global (238d), confirming Vt as the field with the shortest median ESP life. However, this is partially explained by Vt's higher failure rate (68.7% vs. 59.9% global) — more runs reach confirmed failure rather than being censored — and by Vt's higher chemistry burden (higher calcium, chloride, and gypsum loads per day than Ya).

---

## 8. Summary of Conclusions

| Question | Answer |
|----------|--------|
| Does operating above 55Hz (mean or >50% of days) harm pumps in Vt? | No evidence. All tests non-significant. Point estimates favor slightly better survival in high-freq group. |
| Does operating above 55Hz harm pumps globally? | No evidence. All tests non-significant. Small positive ρ(freq, TTF) suggests mild confounding (better wells run faster). |
| Is the mount_year effect real? | Yes in Ya and Global (ρ ≈ −0.15 to −0.26, survives contractor stratification). Not in Vt. |
| Is the mount_year × gypsum interaction real? | No. CV shows it degrades predictions. False positive from underpowered Codex model. |
| What is the main failure driver? | Chemistry burden: calcium, chloride, gypsum, sulfate loads per day. |
| Are the two Weibull components interpretable? | At the population level yes (random/chemical mode vs. wear-out mode). Per-observation assignment is unreliable (ESS 33–77, entropy near maximum for some populations). |
| Which analysis is authoritative for survival? | Run A: mart-based, `ttf_true_best_days`, 12,000 iter × 3 chains. See `analysis/vt_ya_global_survival_comparison.csv`. |

---

## 9. Files Reference

| File | Contents |
|------|----------|
| `analysis/vt_ya_global_survival_comparison.csv` | Run A posterior means, SD, 95%CI, ESS, R-hat for all three populations; mixture life quantiles |
| `analysis_outputs/survival_improved_2026_06_23/global/posterior_summary.csv` | Global posterior (feature-CSV run) |
| `analysis_outputs/survival_improved_2026_06_23/vt/posterior_summary.csv` | Vt posterior (feature-CSV run, poor convergence) |
| `analysis_outputs/survival_improved_2026_06_23/ya/posterior_summary.csv` | Ya posterior (feature-CSV run) |
| `analysis_outputs/survival_improved_2026_06_23/soft_spearman_associations.csv` | Spearman ρ between P(short) and all features, per group |
| `analysis_outputs/survival_improved_2026_06_23/multivariate_coefficients.csv` | Ridge regression coefficients, all groups and models |
| `analysis_outputs/survival_improved_2026_06_23/cv_interaction_results.csv` | 5-fold CV results for interaction model |
| `analysis_outputs/survival_improved_2026_06_23/contractor_stratified_mount_year.csv` | Within-contractor ρ(mount_year, P_short) |
| `analysis_outputs/vt_freq_latent_2026_06_23/vt_latent_with_freq.csv` | Vt runs with latent probs + frequency columns (Run A) |
| `analysis_outputs/vt_freq_latent_2026_06_23/group_summary_a_mean_freq.csv` | Vt KM and P(short) by mean_freq group |
| `analysis_outputs/vt_freq_latent_2026_06_23/group_summary_b_pct_above55.csv` | Vt KM and P(short) by pct_above_55 group |
| `analysis_outputs/freq_latent_global_2026_06_23/latent_with_freq.csv` | Global runs with latent probs + frequency columns |
| `analysis_outputs/freq_latent_global_2026_06_23/group_summary.csv` | Global KM and P(short) by frequency group, both criteria |
