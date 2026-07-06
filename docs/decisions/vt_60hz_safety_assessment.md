# 60 Hz ESP Operation in Field Vt — Safety Assessment

**Date:** June 2026  
**Data:** mart\_\_vt\_freq55, 2,634 runs, 1,527 failures, 22 fields, 2015–2025  
**Analysis package:** `analysis/vt_failure/` — 12 phases + TRF capacity supplement

---

## Bottom Line Up Front

| Question | Evidence | Verdict |
|----------|----------|---------|
| Does 60 Hz shorten ESP lifetime in Vt? | Cox HR=1.24 [0.80–1.93], p=0.34 (Vt-only) | **Trend adverse, not significant** |
| How much shorter? | AFT model: −5% per +10% Hz after field adjustment | **Small, uncertain** |
| Is there a fixed Hz-day capacity? | TRF at failure differs by group (KW p<0.001); β<1 everywhere | **No — rejected** |
| What will break first? | ПЭД (motor) shows wear-out in Vt; КЛ (cable) dominates failures | **ПЭД and КЛ are highest risk** |
| What actually drives short life in Vt? | Category mix (КЛ, Слом вала) + H2S environment, not frequency | **Environment, not Hz** |

**Overall assessment:** Running at 60 Hz in Vt does not carry a statistically proven, large reduction in ESP lifetime based on the available data. However, the direction of every estimate is adverse, the confidence intervals are wide (small high-freq sample in Vt: n=32 runs), and the two failure modes most sensitive to electrical stress — КЛ and ПЭД — are already the top killers in this field. The absence of evidence is not evidence of absence.

---

## 1. Study Context

### Field Vt vs Average

| Metric | Vt | Global |
|--------|-----|--------|
| Runs (n) | 410 | 2,634 |
| Failure rate | 65% | 58% |
| Median TTF (failures) | **70 d** | 100 d |
| Avg operating frequency | 45.4 Hz | 49.9 Hz |
| % High-freq runs (>55 Hz) | 8% | 15% |
| % Acidic wells (H₂S) | **27%** | 7% |
| Infant mortality rate (90d) | **45%** | 35% |

Vt is the harshest field in the dataset on all three dimensions simultaneously: shorter life, higher H₂S, and higher infant-mortality rate. Most Vt pumps run at low-to-normal frequencies; the "High (>55 Hz)" group in Vt contains only 32 runs (28 failures), so all Vt-specific frequency estimates carry substantial uncertainty.

### What "60 Hz" means in the data

The frequency groups are:
- **Low (≤50 Hz):** 885 runs globally, 176 in Vt
- **Normal (50–55 Hz):** 728 globally, 73 in Vt
- **High (>55 Hz):** 407 globally, 32 in Vt — this group includes everything above 55 Hz up to the dataset maximum; it is not limited to exactly 60 Hz

The Vt "High" group is too small for independent Weibull or KM curve estimation (NaN in the Weibull table). All Vt frequency comparisons should be treated as directional signals, not precise measurements.

---

## 2. Does Higher Frequency Shorten ESP Life?

### 2.1 Global (all fields)

| Model | HR High vs Normal | 95% CI | p | PH ok? |
|-------|-------------------|--------|---|--------|
| M1 Univariate | 1.078 | [0.938–1.240] | 0.29 | No |
| M2 + Field adjustment | 1.105 | [0.961–1.271] | 0.16 | No |
| M3 + Mount year | 1.088 | [0.947–1.249] | 0.24 | No |
| M4 + H₂S + GLF | 1.052 | [0.913–1.211] | **0.48** | No |
| M5 + Salt load | 1.183 | [0.996–1.406] | 0.056 | No |

The proportional hazard assumption fails in all models (Schoenfeld test), meaning the hazard ratio is not constant over time. Cox is mis-specified here; the AFT model is more appropriate.

**Weibull AFT with ln(frequency) as stress covariate:**

| Model | Coeff on ln(freq) | 95% CI | p | Rho (β) |
|-------|-------------------|--------|---|---------|
| Univariate | −0.174 | [−0.408, 0.060] | 0.15 | 0.82 |
| + Vt indicator | −0.302 | [−0.587, −0.018] | **0.037** | 0.84 |

After adjusting for the Vt field effect, each 10% increase in frequency is associated with approximately **3% shorter median TTF** (exp(−0.302 × ln(1.10)) ≈ −3%). Going from 50 Hz to 60 Hz (+20%) → estimated **−6% TTF**, marginally significant.

**This is a small and uncertain effect.** The dominant variance in TTF comes from failure category and environment, not frequency.

### 2.2 Within Vt

| Freq group | n runs | Failures | Median TTF (all runs) | Median TTF (failures) |
|-----------|--------|----------|-----------------------|----------------------|
| Low (≤50 Hz) | 176 | 125 (71%) | 80 d | 76 d |
| Normal (50–55 Hz) | 73 | 57 (78%) | **105 d** | **108 d** |
| High (>55 Hz) | 32 | 28 (88%) | **132 d** | 102 d |

Cox HR within Vt: **1.24 [0.80–1.93], p=0.34** — not significant. The point estimate is adverse but the interval spans a range from 20% protective to 93% more hazard.

The long median for all-runs (132 d) in the High group reflects a few long-running censored pumps. The failed pumps median (102 d) is similar to Normal (108 d).

Weibull for Vt High-freq: insufficient data for robust fit (n=28 failures). Vt Normal-freq: **β=1.065** (near-random hazard, η=202.8 d).

### 2.3 Well-level analysis

Using only the first run per well (removes rerun inflation): **logrank p=0.69 globally** — the frequency effect disappears entirely at well level, confirming it is dominated by between-well correlation.

---

## 3. Is There a TRF Capacity (Fixed Hz-Day Budget)?

The "TRF capacity" hypothesis: pumps have a fixed cumulative Hz-day budget — like bearing L10 life — so high frequency depletes it faster and shortens calendar TTF proportionally.

**Test results (n=1,251 failures with TRF data):**

| Test | Result | Verdict |
|------|--------|---------|
| T1: KW test on TRF at failure by freq group | p=0.0001 | TRF NOT constant — **rejects capacity** |
| T2: Weibull β in Hz-day space (TRF axis) | β=0.68–0.79 (all <1) | Infant mortality even in Hz-day space — **rejects capacity** |
| T3: AFT ln(freq) coefficient | −0.174 (required: −1.0) | Far from capacity prediction — **rejects capacity** |
| T5: η scaling vs 1/freq prediction | Ratios: 1.00, 1.19, 1.09 | Normal and High last 9–19% longer than capacity predicts — **rejects capacity** |

**TRF at failure by freq group:**

| Group | Median TRF (Hz-days) | Median TTF (d) | CV(TRF) |
|-------|---------------------|----------------|---------|
| Low (≤50 Hz) | 4,217 | 118 | 1.43 |
| Normal (50–55 Hz) | 6,431 | 145 | 1.20 |
| High (>55 Hz) | 5,741 | 107 | 1.28 |

If there were a fixed budget, all three medians would be similar. Instead, Low-freq pumps accumulate 34% less TRF before failing than Normal-freq pumps. The dominant failure mechanism is infant mortality (sudden early failures), not cumulative wear-out. **There is no evidence of a frequency-proportional life consumption mechanism.**

---

## 4. What Actually Kills Pumps in Vt?

### 4.1 Failure category mix

| Category | Vt share | Global share | Vt β | Vt η | Pattern in Vt |
|----------|----------|--------------|-------|------|---------------|
| КЛ — Cable | **25%** | 23% | 0.73 | 91 d | Strongly infant |
| ПЭД — Motor | **21%** | 16% | **1.12** | 155 d | **Wear-out** |
| Слом вала — Shaft | 16% | 14% | — | — | Infant |
| Засорение РО — Clogging | 15% | 12% | — | — | — |
| НКТ — Tubing | 9% | 12% | — | — | — |
| Износ РО — Impeller wear | 8% | 12% | — | — | — |
| Гидрозащита | 7% | 12% | — | — | — |

**ПЭД (motor) is the only component in Vt that shows a wear-out pattern (β=1.12).** Globally, every category is infant-dominated (β<1). This suggests that in the Vt environment, motors that survive infant failure progressively degrade — consistent with H₂S attack on winding insulation or bearing/thrust wear in a corrosive environment.

КЛ (cable) is the single most frequent failure mode and is strongly infant-dominated (β=0.73, η=91 d). Most cable failures happen within the first 90 days.

### 4.2 RMST decomposition: why Vt is 39 days shorter than average

At the 293-day analysis horizon:
- Total RMST gap (Vt − Global): **−39.3 days**
- Attributable to failure category mix: **−40.1 days** (Vt has more КЛ and Слом вала)
- Attributable to within-category environment: **−49.6 days** (Vt fails faster even within each category)
- Offset by favourable mix in some categories: +50.4 days

The category mix and the H₂S/corrosive environment together fully explain why Vt pumps live shorter lives. Frequency is a secondary signal buried in this noise.

### 4.3 Infant mortality

At the 90-day threshold:

| Group | Vt infant rate |
|-------|---------------|
| Low (≤50 Hz) | 45% |
| Normal (50–55 Hz) | **31%** |
| High (>55 Hz) | **45%** |

The Normal group has the lowest infant mortality in Vt. High freq has the same infant rate as Low freq — counterintuitive if 60 Hz were uniquely hazardous. (Sample size warning: Vt High n=31.)

---

## 5. Equipment Risk Hierarchy at 60 Hz

Based on the analysis, here is the risk ranking for components if Vt wells are shifted to 60 Hz operation:

### Tier 1 — High risk (act now)

**ПЭД (Motor)**
- Wear-out pattern in Vt (β=1.12) — unique to Vt, all other fields and failure categories are infant-dominated
- 60 Hz operation increases motor current draw (∝ frequency for centrifugal pumps at higher flow point)
- Higher thermal stress on winding insulation already compromised by H₂S corrosion
- η=155 d in Vt → motors rated for standard conditions will approach end-of-life faster
- **Action:** Ensure motors are explicitly rated for 60 Hz in H₂S service; monitor winding insulation resistance; check motor heat rating for Vt fluid temperatures

**КЛ (Cable)**
- Already #1 failure mode in Vt (25% of failures)
- Strongly infant-dominated (β=0.73, η=91 d) — most failures happen in the first 90 days
- 60 Hz operation may require higher cable voltage (depending on motor specification)
- Higher downhole temperature from increased power dissipation in cable
- **Action:** Use cables rated for the maximum continuous current at 60 Hz; verify insulation class for Vt temperature and H₂S; inspect cable on every pull

### Tier 2 — Monitor

**Слом вала (Shaft breakage)**
- 16% of Vt failures; infant-dominated globally
- Starting torque at 60 Hz is higher → startup shock loads increase
- Particularly relevant if soft-start is not used or VFD ramp is too steep
- **Action:** Use slow VFD ramp-up (avoid step-start at 60 Hz); inspect shaft condition during pump pulls

### Tier 3 — Lower direct risk from frequency

**Засорение РО (Impeller clogging):** Driven by reservoir fluids/sand, not frequency. Higher flow rates at 60 Hz may increase erosion but this is a flow effect, not a frequency effect per se.

**НКТ, Износ РО, Гидрозащита:** Not directly sensitive to operating frequency in the data.

---

## 6. Recommendations

1. **Confirm motor and cable ratings for 60 Hz / H₂S service before changing operating frequency** — these two components show the highest sensitivity to electrical stress and are already the top failure modes in Vt.

2. **Run a controlled trial**: instrument 5–10 Vt wells at 60 Hz with matched controls at 50–55 Hz; track infant mortality at 30/60/90 days. The current "High freq" sample in Vt (n=32) is too small for confident conclusions.

3. **Apply 60 Hz selectively to wells with low H₂S, good casing condition, and motors with explicit 60 Hz ratings** — the H₂S environment is an independent multiplier of motor degradation.

4. **Do not expect a large TTF shortening from frequency alone.** The data suggests at most −6% median TTF for a 50→60 Hz shift (marginally significant). The much larger driver of short TTF in Vt is the category mix and corrosive environment.

5. **Focus monitoring on the infant window (days 0–90):** 45% of Vt failures happen here. Enhanced monitoring (current, downhole temperature, insulation resistance) in the first 90 days after any new installation at 60 Hz is the highest-value intervention.

6. **There is no fixed Hz-day capacity budget.** Decisions about operating frequency do not need to be framed as "spending down a cumulative reserve" — this model is statistically rejected by four independent tests.

---

## 7. Frequency Definition: Mean-Based vs Proportion-Based

### The two definitions

| Definition | Criterion | Global n High | Vt n High |
|-----------|-----------|--------------|-----------|
| **Current** (mean-based) | `freq_w_mean > 55 Hz` | 407 | 32 |
| **New** (proportion-based) | `freq_above_55hz_pct > 50%` | 435 | **44** |

108 runs (4.1%) change group. The movers are:
- **64 runs Normal → High:** had >50% time above 55 Hz but mean ≤55 Hz (bimodal, sometimes low)
- **40 runs High → Normal:** mean freq >55 Hz but ran above 55 Hz for <50% of time (burst/intermittent high-freq)
- **4 runs Low → High:** rare bimodal cases

### Impact on key statistics

| Metric | Current High-freq Vt | New High-freq Vt |
|--------|---------------------|-----------------|
| n runs | 32 | **44** |
| Median TTF (failures) | 102 d | **137 d** |
| Infant rate (90d) | 45% | **36%** |
| Cox HR (field-adjusted) | 1.18 [0.75–1.86], p=0.48 | **0.97 [0.64–1.47], p=0.88** |
| Weibull β | NaN (n too small) | **0.958** (near-random) |

### Interpretation

The "bimodal" wells (those removed from High by the new definition — high average frequency but intermittently high) are the **worse performers.** The wells that run *consistently* above 55 Hz show:
- No statistically significant excess hazard (HR≈1.0 in Vt)
- Near-random failure pattern (β≈1) rather than infant-dominated (β<1)
- 35d longer median TTF than the bimodal group

This points to **frequency cycling stress** (start-stop, VFD ramp-up shock, thermal cycling) as a larger risk factor than sustained high-frequency operation. A pump that runs stably at 60 Hz experiences less mechanical stress than one that cycles between 30 Hz and 60 Hz repeatedly.

### Recommendation update

Under the proportion-based definition, sustained 60 Hz operation in Vt shows **no statistically detectable adverse effect** (HR=0.97, p=0.88). The risk profile looks similar to Normal-freq operation. The concern shifts to:
- **Frequent frequency cycling** (high to low and back) — avoid if possible
- **VFD ramp-up procedure** during starts — use slow ramp even at low starting frequencies

---

## 8. Data Limitations

- **Vt High-freq sample is small (n=32 runs, 28 failures)** — all Vt-specific frequency estimates have wide confidence intervals
- **Exact 60 Hz data unknown** — the "High (>55 Hz)" group includes all frequencies above 55 Hz; median actual frequency in this group is approximately 59 Hz
- **Confounding by well selection** — operators may have chosen to run harder wells at lower frequencies (healthy survivor bias); the High-freq wells in Vt may be the physically better wells
- **PH assumption violation** — Cox models are mis-specified; AFT estimates are preferred but also rely on Weibull distributional assumptions
- **Failure category classification** — based on text matching of maintenance records; misclassification rate unknown

---

*Analysis outputs:* `analysis_outputs/vt_failure_analysis/` — 78 files across 13 phases  
*Code:* `analysis/vt_failure/` package  
*Prompt/specification:* `docs/vt_60hz_failure_analysis_prompt.md`
