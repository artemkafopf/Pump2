# VT 60 Hz Failure Analysis — Revised Plan

## Central Question

> Is operating ESPs at higher frequency — specifically exposure above 55 Hz and the 60 Hz scenario — associated with shorter pump life after accounting for field conditions and other major confounders, or is frequency mainly a marker of those other conditions?

This is the organizing hypothesis for the entire analysis. All methods below either build evidence for or against it. The analysis is complete only when this question is answered with a qualified conclusion about association strength, confounding, and remaining uncertainty.

---

## Glossary

| Abbreviation | Meaning |
| --- | --- |
| TTF | Time to failure — calendar run days |
| TTF_true | Operational days to failure (techregime or telemetry-derived) |
| RMST | Restricted mean survival time — mean TTF up to a fixed horizon τ |
| TRF | Total frequency Hz-days = Σ(freq × days) over the run |
| TLF | Total liquid m³ = Σ(qliq) over the run |
| KM | Kaplan-Meier |
| CIF | Cumulative incidence function (competing risks) |
| PH | Proportional hazards assumption in Cox regression |
| AFT | Accelerated failure time model |
| β, η | Weibull shape and scale parameters |
| GLF | Gas-liquid factor |
| Kpod | Flow ratio = qliq / nominal_flow |
| HR | Hazard ratio |
| SHAP | Shapley additive explanations (model interpretability) |

---

## Step 0 — Data Audit (Run Before Everything Else)

Do this before designing any comparison. Results determine which analyses are feasible.

**Required outputs:**

1. Count table: runs × failures by frequency band (≤50 Hz, 50–55 Hz, 55–60 Hz, >60 Hz) per field and globally. If the "true 60 Hz" subgroup (freq_w_mean > 58 Hz) has fewer than 50 failures globally, drop the 60 Hz vs 50 Hz framing entirely and use `>55 Hz exposure` as the primary high-frequency proxy throughout.

2. Telemetry coverage: share of runs with ≥7 valid frequency days, per field and per mount-year cohort. Report the fallback rate (techregime vs telemetry) for TTF_true.

3. Failure category coverage: how many runs have a classified failure category vs `<missing>`. Validate whether `<missing>` corresponds only to censored runs (event=0) or also includes join/classification gaps. If both exist, separate them explicitly.

4. H2S coverage: share of runs with H2S available, per field, under a declared source priority:
   - direct workbook H2S measurement if present
   - otherwise lab-derived proxy
   - otherwise acidic/non-acidic label only

   Report what fraction of Vt wells have each H2S source available.

5. Corrosion-resistance (`Коррозионная стойкость`) fill rate per field. If Vt coverage is below 60%, treat it as indicative only and exclude from any model.

6. Duty-metric coverage and sanity check:
   - availability and distribution of TRF and TLF
   - count of zero or near-zero TLF runs

   If near-zero TLF is non-trivial, audit whether these represent non-operating intervals, missing telemetry, or invalid qliq before any further use.

7. Ion-proxy coverage: fill rates for calcium, chloride, sulfate, combined salt, and gypsum/precipitate proxies. Report pairwise correlations. If the combined precipitate proxy is highly correlated with only one ion in Vt, note that explicitly.

8. Mount-year × frequency band correlation: compute mean `avg_freq_hz` by mount year per field. If frequency has been trending upward over time, mount year is a direct temporal confounder and must be included in all regression steps.

9. Multiple-run wells: count how many unique wells have more than one run. If this exceeds 15% of the well population, the unit-of-analysis choice (run vs well) materially affects results and must be explicitly resolved.

**Minimum subgroup thresholds for subsequent analyses:**

| n_failures | What is permitted |
| --- | --- |
| < 20 | Descriptive statistics only — no survival model |
| 20–50 | Kaplan-Meier only — no Weibull or Cox |
| > 50 | Weibull and Cox viable |
| > 100 | Competing risks, CatBoost viable |

---

## Unit of Analysis

State the choice explicitly at the start of every analysis section.

**Run-level** is the natural unit because frequency exposure is measured per run and failure events are run-level. Use run-level throughout unless a specific question requires well-level aggregation.

**Problem:** multiple runs per well are correlated. The same well's conditions (H2S level, reservoir pressure, field) persist across runs. Using runs as independent in Cox or log-rank tests overstates effective sample size.

**Resolution:** use run-level data with cluster-robust standard errors (clustered on well_key) in all regression models. In KM plots, note that confidence bands assume independence and are therefore optimistic. If any key result is sensitive to this, re-run at well-level (first run only or worst-case run) as a robustness check.

---

## Causal Structure

Before stratifying, make the assumed causal relationships explicit. This prevents accidentally controlling for mediators.

**Assumed structure (informal DAG):**

```text
Field ──────────────────────────────────────────► TTF
  │                                                 ▲
  ├──► Reservoir conditions (H2S, GLF, pressure) ──┤
  │                                                 │
  ├──► Contractor ──► Pump family / equipment ──────┤
  │                                                 │
  └──► Operating frequency ────────────────────────►┤
            ▲                                       │
            │                                       │
       Operator choice                        Failure category
       (confounded by well severity)                │
                                                    └──► TTF (via mechanism)
```

**Key implications:**

- Field is a common cause of frequency AND TTF → control for field in all comparisons.
- Failure category is a mediator (on the path from frequency to TTF) → do NOT control for it when estimating the total frequency effect; DO control for it when decomposing the Vt-vs-global difference.
- H2S is partially downstream of field but also independently affects TTF → treat as a covariate, not a confounder to be blocked.
- Pump family / contractor may be confounders or mediators depending on whether frequency choice is made at the contractor level → treat as confounder in primary analysis, check sensitivity.

**Critical note on TRF and TLF:** TRF = freq × TTF and TLF = Σ(qliq) both accumulate mechanically with run duration. They are therefore downstream of TTF in the causal graph, not upstream. Never include raw TRF or raw TLF as predictors in any model where TTF is the outcome — this is circular by construction. Use only normalized forms (TRF/TTF as a proxy for mean frequency, or TLF/TTF as a proxy for mean qliq) and only in descriptive or secondary comparisons, not in survival models with TTF as the time axis.

---

## Execution Sequence

Run in this order. Each step informs the next.

1. Data audit (Step 0)
2. Frequency band feasibility check → confirm or replace 60 Hz framing
3. Descriptive: TTF distributions, category mix, operating environment by field (Phase 1)
4. Infant mortality rate table × threshold × field × frequency group (Phase 2)
5. Kaplan-Meier by frequency group (all fields, then per field) (Phase 3)
6. Kaplan-Meier by failure category (all fields, then per field) (Phase 3)
7. Weibull shape analysis by frequency group and by category (Phase 4)
8. Competing risks — cumulative incidence functions by category (Phase 5)
9. Cumulative duty and chemistry-proxy comparison (Phase 6)
10. Within-field and within-category stratified frequency comparisons — main confounding test (Phase 7)
11. Cox proportional hazards (if PH holds; fallback: AFT) (Phase 8)
12. Vt deep-dive — environment comparison, category mix decomposition (Phase 9)
13. CatBoost infant mortality classifier (if n ≥ 100 failures) (Phase 10)
14. Sensitivity checks (Phase 11)
15. Synthesis and conclusion

---

## Frequency Exposure Definition

Use exposure-style metrics, not a single static frequency value.

**Primary metrics (use in this priority order):**

- `signed_freq_exposure = share(days > upper_hz) − share(days < lower_hz)` — signed direction of exposure
- `freq_w_mean` — mean operating frequency over the run (valid days only)
- `freq_above_55hz_pct` — share of valid frequency days above 55 Hz

**Secondary cumulative-duty metrics (supportive, not primary):**

- `TRF / TTF` — normalized cumulative frequency duty; use as a proxy for mean operating frequency only
- `TLF / TTF` — normalized liquid throughput; treat as a production-burden variable, not a frequency variable

**Warning:** raw TRF and raw TLF must never be presented as primary explanatory variables against TTF because they grow mechanically with run duration. See the Causal Structure section.

**Grouping for non-technical output:**

- Low: `freq_w_mean ≤ 50 Hz`
- Normal: `50 < freq_w_mean ≤ 55 Hz`
- High: `freq_w_mean > 55 Hz`

If the true 60 Hz subgroup (freq_w_mean > 58 Hz) has ≥ 50 failures globally, add it as a fourth group. Otherwise note the sparsity and drop it.

**Timing of exposure matters:** post-infant frequency exposure (frequency after day 90 of the run) may be a better predictor of mature-life failure than whole-run average. Compute this separately for the mature-life analysis if sample size permits.

---

## Phase 1 — Descriptive Baseline

For all fields combined and each field separately, report:

| Metric | Frequency groups | Field groups |
| --- | --- | --- |
| Run count | ✓ | ✓ |
| Failure count and rate | ✓ | ✓ |
| Infant failure share (at each threshold: 45, 60, 90, 120 d) | ✓ | ✓ |
| Median TTF overall | ✓ | ✓ |
| Median TTF mature only (TTF ≥ 90 d) | ✓ | ✓ |
| Failure category mix (%) | ✓ | ✓ |
| Mean H2S proxy (mg/L) | ✓ | ✓ |
| Mean GLF | ✓ | ✓ |
| Mean Kpod | ✓ | ✓ |
| Normalized TRF/TTF and TLF/TTF | ✓ | ✓ |
| Ca / Cl / SO₄ load proxies (separate) | ✓ | ✓ |
| Gypsum / precipitate proxy | ✓ | ✓ |

For Vt specifically, add a side-by-side table against global on all of the above.

**For each table, include n_runs and n_failures in every cell.** Suppress cells where n_failures < 5.

**Chemistry-proxy requirement:** report separate calcium, chloride, and sulfate load proxies in addition to the combined salt and gypsum proxies. Do not collapse them by default — the ions have different physical mechanisms (brine salinity vs sulfate-scale tendency vs mixed chemistry).

---

## Phase 2 — Infant Mortality

**Definition:** event = 1 and TTF_true < threshold. Analyze separately at thresholds 45, 60, 90, 120 days. Conclusions must be stable across at least three of the four thresholds to be reported as robust.

**Denominator precision:** The infant failure rate denominator must be all runs that had sufficient follow-up to be observable as either infant or mature — specifically runs where TTF_true ≥ threshold OR event = 1. Exclude runs with TTF_true < threshold AND event = 0 (censored before the threshold), because their outcome is genuinely unknown. Report how many runs were excluded on this basis per subgroup.

**Required outputs:**

1. Infant failure rate (infant failures / eligible runs in subgroup) by field × frequency group × threshold. Present as a matrix heatmap.

2. Infant share among failures (infant failures / all failures) by field × frequency group × threshold. Present this separately as a failure-composition view, not as the primary infant-mortality rate — these are different quantities.

3. Infant failure rate by failure category × threshold (globally). This answers whether certain categories are fast-failing by nature regardless of frequency.

4. For Vt: is the elevated infant rate explained by category mix or by something within each category? Compute:
   - Vt infant rate using Vt's own category mix
   - Vt infant rate standardized to global category mix weights

   If the standardized rate is close to global, mix explains it. If still elevated, environment or equipment explains it.

5. Fisher exact test or chi-squared for infant vs mature split across frequency groups, per field. Report both the test result and the absolute rate difference — statistical significance alone is not sufficient. Adjust for multiple comparisons across fields using Bonferroni correction (note as exploratory if n_fields is small).

---

## Phase 3 — Kaplan-Meier Survival Analysis

Apply the minimum subgroup thresholds from Step 0. Do not plot KM curves for groups with fewer than 20 failures.

**Grouping scheme:**

- Primary: three frequency groups (low / normal / high) as defined above
- Secondary: failure category
- Tertiary: H2S class (Кислый / Некислый)

**Required KM outputs:**

1. All runs, by frequency group — global and per field
2. Mature failures only (TTF ≥ 90 days), by frequency group — to separate infant effect from mature-life effect
3. All runs, by failure category — global and Vt separately
4. Vt vs global survival curves on the same axes — the primary visual for the executive summary

For each curve, report: median survival, 25th percentile survival, 95% CI on the median (Greenwood formula), log-rank p-value vs reference group, and n_failures. Note that confidence bands assume run-level independence and are optimistic where multiple runs per well exist.

**Interpret carefully:** if frequency-group survival curves separate early (within 60 days) but then converge, the effect is infant-dominated. If they remain separated throughout, the effect is on mature life. State which pattern is observed.

---

## Phase 4 — Weibull Shape Analysis

Fit a two-parameter Weibull to each subgroup with ≥ 50 failures using censoring-aware maximum likelihood estimation. Do not use regression on Weibull probability paper, which handles censoring poorly.

**Report for each subgroup:**

- Shape parameter β with 95% CI (profile likelihood or bootstrap)
- Scale parameter η (characteristic life)
- Implied failure rate behavior: β < 1 = decreasing hazard (infant-dominated), β ≈ 1 = constant (random/exponential), β > 1 = increasing (wear-out)

**Primary comparison table with domain-knowledge priors:**

| Group | Prior β expectation | n_failures | β (fitted) | 95% CI | η (days) | Pattern vs prior |
| --- | --- | --- | --- | --- | --- | --- |
| Low freq — global | — | | | | | |
| Normal freq — global | — | | | | | |
| High freq — global | — | | | | | |
| Low freq — Vt | — | | | | | |
| High freq — Vt | — | | | | | |
| КЛ (R-0) — global | < 1 (external hazard, infant) | | | | | |
| ПЭД (R-0) — global | ≈ 1 (random electrical / corrosion) | | | | | |
| Слом вала — global | < 1 or ≈ 1 (mechanical shock or fatigue) | | | | | |
| Засорение РО — global | > 1 (accumulation-driven) | | | | | |
| НКТ — global | ≈ 1 (mixed: corrosion + mechanical) | | | | | |
| Износ РО — global | > 1 (wear-dominated) | | | | | |
| Износ/негермет.гидрозащиты — global | > 1 (seal wear) | | | | | |

Where fitted β differs materially from the prior, state whether this is likely a data artifact (small n, censoring pattern) or a genuine finding.

**Key interpretive question:** if high-frequency wells have a lower β than normal-frequency wells within the same field, that is mechanistic evidence — not just a median shift — that frequency changes the failure mode itself, not just the timing.

---

## Phase 5 — Competing Risks

Standard KM treats all other failure categories as independent censoring for a given category. This is incorrect: a pump that fails from ПЭД cannot later fail from КЛ. Use cumulative incidence functions (CIF) instead.

**Required outputs:**

1. CIF curves for all 7 categories simultaneously — global and Vt separately. This shows both the rate (slope) and the eventual share (asymptote) of each category.

2. Gray's test for equality of CIFs across frequency groups, per category. This is the correct statistical test when competing risks are present.

3. Fine-Gray subdistribution hazard model for the two largest categories globally. Report subdistribution hazard ratio for `signed_freq_exposure` as the primary predictor, with field and mount_year as covariates.

**Why this matters:** if high-frequency wells have more ПЭД failures and fewer Износ РО failures, the overall TTF comparison is mixing two different populations. The CIF separates rate from mix.

---

## Phase 6 — Cumulative Duty and Chemistry-Proxy Comparison

This phase checks whether normalized duty and ion chemistry explain failure timing or failure mechanism better than simple frequency-group labels alone.

**Required outputs:**

1. Normalized TRF analysis: compare failure timing by TRF/TTF across frequency groups. Report whether this metric adds information beyond `freq_w_mean`. If it does not, it is redundant with the mean frequency already captured in Phase 3.

2. Normalized TLF analysis: treat TLF/TTF as a production-burden indicator, not a frequency metric. Audit zero/near-zero TLF runs first. If usable, compare whether TLF/TTF helps differentiate wear, clogging, or corrosion-related failure categories.

3. Separate ion proxies — for each of calcium, chloride, sulfate, combined salt, gypsum: report coverage, pairwise correlations, and association with survival or failure timing after field stratification.

4. Compare which chemistry proxy best differentiates Засорение РО, Износ РО, ПЭД (R-0), and Износ/негермет.гидрозащиты. Note whether the effect is concentrated in Vt or applies globally.

**Key interpretation rule:** if separate ion proxies outperform the combined precipitate proxy, prefer the more specific explanation in the final narrative. Do not collapse chemistry into one combined proxy by default if the ions point to different mechanisms.

---

## Phase 7 — Confounding Test (Main Scientific Test)

This phase directly tests the central question using stratification.

**Step 7a — Raw descriptive correlation**
Compute Spearman rank correlation between `signed_freq_exposure` and TTF_true restricted to failure events only. Report r, 95% CI, and n. This is severely limited: it excludes all censored runs and is biased when censoring rates differ across frequency groups (which they likely do). Treat it as a directional indicator only, not as primary evidence.

**Step 7b — Within-field stratification**
Repeat within each field separately. If the sign or magnitude of the correlation changes substantially across fields, frequency is likely a field-level marker rather than an independent driver.

**Step 7c — Within-category stratification**
Repeat within each of the 7 failure categories. If the correlation disappears within categories, the frequency-TTF relationship is largely explained by category mix differences.

**Step 7d — Interpretation rule:**

- Survives both 7b and 7c: consistent with a possible real effect, but not conclusive — censoring is still excluded.
- Disappears within fields but not categories: frequency likely reflects field-level confounding.
- Disappears within categories: category mix likely explains the raw relationship.
- Sign reverses after stratification: strong evidence of confounding — do not interpret the raw correlation as directional.

**Primary evidence remains the survival models** in Phases 3, 4, 5, and 8. Phase 7 is a diagnostic layer and a cross-check, not the final basis for any claim.

---

## Phase 8 — Cox Proportional Hazards

Only run if n_failures ≥ 50 per comparison group and the PH assumption is satisfied.

**PH assumption check (required before interpreting any Cox result):**

- Scaled Schoenfeld residuals plotted vs time for each covariate
- Log-log survival plot: parallel curves across frequency groups indicate PH holds
- If PH fails for frequency, use time-stratified Cox (separate baseline hazard for early and late periods) or switch to an AFT model with log-normal or log-logistic distribution

**Model sequence:**

1. Univariate: `frequency_group` only — establishes raw hazard ratio
2. Add field fixed effects — tests whether field explains the frequency effect
3. Add `mount_year` — tests temporal confounding
4. Add H2S class and GLF bin — tests environmental confounding
5. Add separate chemistry proxies if coverage supports it (calcium, chloride, sulfate, gypsum); prefer separate terms over a combined proxy when interpretability improves and collinearity remains manageable
6. Add contractor or pump family if sample size supports it

**Report for each model:** HR per frequency group, 95% CI, concordance index (c-statistic), and whether PH holds. If HR changes materially between model 1 and model 4, describe what the adjustment explains and in which direction.

**Clustering:** cluster-robust standard errors on well_key in all models.

---

## Phase 9 — Vt Deep-Dive

**9a — Environment comparison: Vt vs global**

| Variable | Vt | Global | Test | Interpretation |
| --- | --- | --- | --- | --- |
| Mean H2S (declared source) | | | | |
| Share Кислый (%) | | | | |
| Mean GLF | | | | |
| Mean Kpod | | | | |
| Normalized TRF/TTF | | | | |
| Normalized TLF/TTF | | | | |
| Calcium load proxy | | | | |
| Chloride load proxy | | | | |
| Sulfate load proxy | | | | |
| Gypsum / precipitate proxy | | | | |
| Mean nominal frequency Hz | | | | |
| Share freq > 55 Hz (%) | | | | |
| Corrosion-resistance fill rate | | | indicative only if < 60% | |
| Contractor mix (top 3) | | | | |
| Pump family mix (top 3) | | | | |
| Telemetry coverage (%) | | | | |

**9b — Category mix decomposition**

This is the single most informative calculation for stakeholders. Use RMST (restricted mean survival time) rather than medians, because RMST has an additive decomposition property that medians do not.

**Horizon τ:** set τ to the 75th percentile of global observed failure times (i.e., where 75% of global failures have occurred). This ensures that at least 25% of runs in every major subgroup have reached τ, giving RMST estimates a stable empirical base.

Steps:

1. Compute each category's RMST(τ) within Vt and globally.
2. Compute Vt's overall RMST(τ) = Σ(Vt category weight × Vt within-category RMST).
3. Compute the **mix effect**: substitute global category weights into step 2 while keeping Vt's within-category RMST. The difference from the observed Vt RMST isolates the contribution of Vt having more fast-failing categories.
4. Compute the **within-category effect**: substitute global within-category RMST while keeping Vt's own category weights. The difference isolates Vt's worse performance inside each category.
5. The residual interaction between mix and within-category effects completes the decomposition.

This separates "Vt has more of the bad categories" from "Vt fails faster within each category." Both findings are actionable but require different interventions.

**9c — Within-Vt frequency effect**

Repeat the Phase 7 stratified diagnostics restricted to Vt only, then confirm with the applicable survival model. Does frequency still matter within Vt after accounting for H2S and GLF? If Vt-only n_failures < 50, use stratified KM or a simple tabular comparison — do not force a Cox model on a thin sample.

Also test whether the answer changes after adding normalized TRF/TTF and the separate ion proxies. If frequency weakens materially, describe whether the better explanation is cumulative duty, chemistry, or both.

**9d — H2S interaction within Vt**

Split Vt into Кислый vs Некислый. Within each group, report: infant mortality rate, KM survival curve, and category mix. Does the Кислый subgroup explain a disproportionate share of the Vt disadvantage?

### 9e — Required verdict for each Vt hypothesis

| Hypothesis | Evidence for | Evidence against | Verdict |
| --- | --- | --- | --- |
| Vt has more fast-failing categories | | | |
| Vt has harsher H2S environment | | | |
| Vt frequency effect persists after H2S/GLF control | | | |
| Vt equipment/contractor mix is different | | | |
| High-frequency × H2S interaction accelerates failure | | | |

---

## Phase 10 — CatBoost (Infant Mortality Classifier)

Use only if global infant failures ≥ 100 at the chosen threshold.

**Target:** binary — infant failure (event = 1 AND TTF_true < threshold) vs reference class. The reference class must be restricted to runs where the outcome is observable: include only runs with TTF_true ≥ threshold (mature failures or censored-but-long-running). Do not include runs censored before the threshold — their outcome is unknown and including them contaminates the negative class.

**Features (in priority order):**

- `signed_freq_exposure`, `freq_w_mean`, `freq_above_55hz_pct`
- `field` (categorical)
- `h2s_proxy_mg_l` or H2S class
- `avg_glf`, `avg_kpod`
- `mount_year`
- `contractor` (categorical)
- `nominal_freq_hz` (pump design frequency)
- separate ion proxies: calcium, chloride, sulfate, gypsum — only if coverage ≥ 60%
- `pbubble_atm` if available

**Do not include:** failure category (downstream of target), raw or normalized TRF or TLF (post-outcome: TRF = freq × TTF and TLF = Σ(qliq) are only known after the run ends and would constitute data leakage), TTF itself, or any variable that encodes the run outcome.

**Validation:** use 5-fold stratified cross-validation, stratifying on both field and infant/mature label. Report mean AUC ± std across folds and a calibration plot on held-out data.

**Interpretation protocol:**

1. Report SHAP values for the top 10 features, not raw feature importance.
2. Compare predicted infant probability by frequency group — this shows the model's estimate of the frequency effect after conditioning on all other features.
3. Treat as supportive evidence only: if the model and the stratified KM agree, the finding is stronger; if they disagree, investigate why before reporting.

**Do not interpret:** individual SHAP values as causal, or predictive accuracy as evidence of a real effect. State AUC and calibration quality before presenting any SHAP output.

---

## Phase 11 — Sensitivity Checks

After the main analysis, test whether key conclusions hold under alternative assumptions.

**Required sensitivity checks:**

1. **TTF definition:** repeat the primary KM and infant mortality comparisons using calendar TTF instead of TTF_true. If conclusions change materially, flag telemetry coverage as a limitation.

2. **Frequency band boundaries:** shift the low/normal/high thresholds by ±5 Hz. If the frequency group effect changes substantially, the conclusions are threshold-sensitive and should be reported with that caveat.

3. **Infant mortality threshold:** results must be stable across ≥ 3 of the 4 thresholds (45, 60, 90, 120 days) to be reported as robust. If they are stable, state this explicitly as a finding — it strengthens the claim.

4. **Unit of analysis:** re-run the primary frequency-group KM at well-level (first run per well) and compare to the run-level result. If the frequency effect weakens, note that within-well run correlation inflates the run-level result.

5. **RMST horizon (Phase 9b):** repeat the category mix decomposition at τ = 50th and 90th percentile of global failure times. If the mix vs within-category split changes substantially with horizon, report the range rather than a single point estimate.

Report each sensitivity check as: "Primary result X. Under alternative assumption Y, result is Z. Conclusion [stable / weakened / reversed]."

---

## Output Requirements

### Deliverable 1 — Executive Summary (≤ 2 pages)

- One paragraph answering the central question
- One visual: Vt vs global KM survival curve
- Ranked list of the top 3 findings in plain language
- One paragraph on what remains uncertain

### Deliverable 2 — Data Quality and Coverage Report

- Subgroup n tables from the data audit (Step 0)
- Coverage rates for H2S, telemetry, corrosion resistance, and ion proxies
- Statement of which planned analyses were feasible and which were dropped due to sample size, and why

### Deliverable 3 — Field-by-Field Comparison Table

All Phase 1 descriptive metrics for each field with global as the reference column. Include n_runs and n_failures per cell.

### Deliverable 4 — Survival and Competing Risks Report

- KM curves (Phase 3) with survival statistics table
- Weibull β table (Phase 4) with prior vs fitted comparison
- CIF plots (Phase 5)
- Chemistry-proxy comparison summary (Phase 6)

### Deliverable 5 — Confounding and Regression Summary

- Phase 7 stratified correlation results: raw vs within-field vs within-category
- Cox model sequence table with HR, 95% CI, c-statistic, PH test result (Phase 8)
- Plain-language statement: what each covariate adjustment changed and in which direction

### Deliverable 6 — Vt Diagnostic Section

- Environment comparison table (9a)
- Category mix decomposition with RMST numbers (9b) — the primary stakeholder output
- Within-Vt frequency effect (9c)
- H2S interaction (9d)
- Hypothesis verdict table (9e)
- Ranked list: most likely reasons Vt has shorter TTF, with confidence level per reason

### Deliverable 7 — Confidence Assessment

**What we can say with confidence** — meets all three: n ≥ 50 failures, result stable across ≥ 2 fields or ≥ 3 infant thresholds, consistent between KM and Weibull or between Phase 7 and Phase 8.

**What remains uncertain** — small subgroups, single-field patterns, conflicting signals across methods, PH assumption violated, H2S or corrosion data too sparse, sensitivity checks reversed the conclusion.

---

## Statistical Guardrails

- Use `TTF_true_best` as the primary duration variable. Fall back to calendar TTF only when TTF_true coverage is below 50% for a subgroup; state this explicitly.
- Report n_runs and n_failures for every subgroup result. Never show a hazard ratio, survival curve, or RMST estimate without these.
- Do not report p-values alone. Always report effect size alongside: median difference in days, hazard ratio, absolute rate difference, or RMST difference.
- Treat all findings as observational associations. Use language: "associated with," "consistent with," "not explained by" — not "causes," "proves," or "driven by."
- If the 60 Hz vs 50 Hz comparison is attempted and n_failures < 30 in either arm, add an explicit warning on every chart from that comparison.
- Competing risks must be acknowledged whenever category-specific survival is discussed.
- Cluster-robust standard errors on well_key in all regression models.
- Raw TRF and raw TLF are mechanically coupled to TTF and must never be presented as primary explanatory variables against TTF without duration normalization or explicit adjustment.
- Do not collapse chemistry into a single precipitate proxy by default. Show separate ion proxies first; justify any combination used in the final narrative.
- Multiple comparisons: all p-values across fields, categories, and thresholds are exploratory. Apply Bonferroni correction within structured families of tests (e.g., across fields for a single hypothesis) and state that no correction is applied across the full analysis; interpret isolated significant results with corresponding caution.
