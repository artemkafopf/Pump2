# ESP @ 60 Hz in Field Vt — Is It Safe?
### Quantitative Survival Analysis · June 2026

Data: 2,634 ESP runs · 1,527 failures · 22 fields · 2015–2025  
Analysis: lifelines (KM, Weibull, Cox, AFT, CIF) + CatBoost + TRF capacity tests

---

## Slide 1 — The Question

> **Can we run ESP pumps at 60 Hz in field Vt?  
> Will it shorten pump life? Which component fails first?**

Current practice in Vt: most runs at **≤50 Hz** (average 45.4 Hz)  
"High-frequency" (>55 Hz) group in Vt: only **32 runs** — small sample  
Vt context: harshest field — 70-day median TTF, 27% acidic wells, 45% infant mortality

---

## Slide 2 — Field Vt at a Glance

| Metric | **Vt** | Global average |
|--------|--------|----------------|
| Median TTF (failures) | **70 d** | 100 d |
| Failure rate | 65% | 58% |
| Avg frequency | 45 Hz | 50 Hz |
| % Acidic (H₂S) | **27%** | 7% |
| Infant mortality (< 90 d) | **45%** | 35% |

**Vt is simultaneously the shortest-lived and harshest-environment field.**  
The question is whether adding frequency stress matters on top of an already difficult baseline.

---

## Slide 3 — Frequency Groups in Vt

| Group | Global (n) | Vt (n) | Vt median TTF |
|-------|-----------|--------|---------------|
| Low (≤ 50 Hz) | 885 | 176 | 76 d |
| Normal (50–55 Hz) | 728 | 73 | **108 d** |
| **High (> 55 Hz)** | 407 | **32** | 102 d |

> **Normal 50–55 Hz outperforms both Low and High in Vt.**  
> Vt High-freq sample (n=32) is too small for independent Weibull/KM estimation.

### Why is Low-freq worst in Vt?

Low-freq wells in Vt are likely the most technically difficult (low reservoir pressure),  
running at reduced speed because the well cannot support full flow — a classic survivor bias.

---

## Slide 4 — Does 60 Hz Shorten TTF? (Cox Regression)

All models use cluster-robust standard errors (clustered on well_key).

| Model | HR High vs Normal | 95% CI | p |
|-------|-------------------|--------|---|
| Univariate (global) | 1.08 | [0.94–1.24] | 0.29 |
| + Field adjustment | 1.11 | [0.96–1.27] | 0.16 |
| + H₂S, GLF | **1.05** | **[0.91–1.21]** | **0.48** |
| Vt-only | 1.24 | [0.80–1.93] | 0.34 |

**The PH assumption fails in all models** (Schoenfeld test p < 0.05).  
Cox is mis-specified here — the hazard ratio changes over time.

### AFT Model (more appropriate — allows time-varying hazard ratio)

| Covariate | Coeff on ln(freq) | 95% CI | p |
|-----------|-------------------|--------|---|
| Univariate | −0.17 | [−0.41, 0.06] | 0.15 |
| + Vt indicator | **−0.30** | **[−0.59, −0.02]** | **0.037** |

**After accounting for the Vt field effect:**  
50 Hz → 60 Hz (+20% freq) → estimated **−6% median TTF** (marginally significant, p=0.04)

> Small effect, wide uncertainty. Main conclusion: **frequency is not the dominant driver.**

---

## Slide 5 — Is There a Fixed TRF Capacity? (Four Tests)

**Hypothesis:** pumps have a fixed cumulative Hz-day budget (like L10 bearing life).  
High frequency depletes this budget faster → proportionally shorter calendar life.

| Test | Expected if capacity model | Observed | Result |
|------|---------------------------|----------|--------|
| T1: KW on TRF at failure | Same across groups | p = 0.0001 (different) | **Rejected** |
| T2: Weibull β in Hz-day space | β > 1 (wear-out) | β = 0.68–0.79 (infant!) | **Rejected** |
| T3: AFT coeff on ln(freq) | −1.0 | −0.17 (p=0.15) | **Rejected** |
| T5: η scales as 1/freq | Ratios ≈ 1 | 1.00, 1.19, 1.09 | **Rejected** |

### TRF at failure by group

| Group | Median TRF (Hz-days) | Median TTF |
|-------|---------------------|-----------|
| Low (≤ 50 Hz) | 4,217 | 118 d |
| Normal (50–55 Hz) | **6,431** | 145 d |
| High (> 55 Hz) | 5,741 | 107 d |

If a fixed budget existed, all three medians would be similar.  
Low-freq pumps accumulate 34% less TRF before failing than Normal — opposite of capacity model.

> **Conclusion: there is no frequency-proportional life consumption mechanism in the data.**

---

## Slide 6 — What Actually Kills Pumps in Vt

### Failure category mix in Vt (n = 268 failures)

| Category | Vt share | Global share | Weibull β (Vt) | Pattern |
|----------|----------|--------------|----------------|---------|
| КЛ — Cable | **25%** | 23% | 0.73 | Strongly infant (η=91 d) |
| ПЭД — Motor | **21%** | 16% | **1.12** | **Wear-out ← unique to Vt** |
| Слом вала — Shaft | 16% | 14% | — | Infant |
| Засорение РО — Clogging | 15% | 12% | — | — |
| НКТ — Tubing | 9% | 12% | — | — |
| Износ РО + Гидрозащита | 15% | 24% | — | — |

### Why Vt has 39 fewer RMST days than global average

```
Total gap:         − 39 days
  Category mix:    − 40 days  (more КЛ + Слом вала → shorter-lived categories)
  Environment:     − 50 days  (fails faster even within same category)
  Offset:          + 50 days
```

**The short life in Vt is driven by corrosive environment and category mix — not by frequency.**

---

## Slide 7 — Weibull Analysis: What the Shape Parameter Tells Us

**β < 1** → decreasing hazard → **infant mortality** (the sooner you survive, the safer you get)  
**β ≈ 1** → constant hazard → random failures  
**β > 1** → increasing hazard → **wear-out** (risk grows with age)

### Globally — all frequency groups show infant mortality

| Group | β (calendar time) | β (TRF axis / Hz-days) |
|-------|-------------------|------------------------|
| Low (≤ 50 Hz) | 0.794 | 0.684 |
| Normal (50–55 Hz) | 0.882 | 0.776 |
| High (> 55 Hz) | 0.800 | 0.737 |
| Vt field total | — | 0.794 |

**β < 1 everywhere, even when time is measured in cumulative Hz-days.**  
This means: failure is dominated by early defects and installation quality, not accumulated stress.

### In Vt — only the motor shows wear-out

| Component | β in Vt | η in Vt | Implication |
|-----------|---------|---------|-------------|
| КЛ Cable | 0.73 | 91 d | Install defects, early insulation failure |
| **ПЭД Motor** | **1.12** | **155 d** | **Aging/degradation — unique to Vt** |

The ПЭД wear-out pattern suggests motors in Vt experience progressive degradation  
(likely H₂S attack on winding insulation + bearing wear in corrosive fluid).

---

## Slide 8 — Infant Mortality (First 90 Days)

**45% of all Vt failures happen before 90 days.**  
This is the dominant risk — larger than any frequency effect.

### Infant rate by freq group in Vt (90-day threshold)

| Group | Infant rate | n (Vt) |
|-------|------------|--------|
| Low (≤ 50 Hz) | **45%** | 169 |
| Normal (50–55 Hz) | **31%** | 67 |
| **High (> 55 Hz)** | **45%** | 31 |

Normal 50–55 Hz has the lowest infant rate in Vt.  
High-freq has the same rate as Low-freq — not uniquely hazardous for infant failures.

> Warning: High-freq Vt sample is n=31. This difference is directional, not conclusive.

---

## Slide 9 — Equipment Risk Hierarchy at 60 Hz

### Risk Tier 1 — Act before switching to 60 Hz

```
┌─────────────────────────────────────────────────────────────────┐
│  ПЭД (Motor) — HIGHEST RISK                                     │
│  • Only component showing wear-out (β=1.12) in Vt              │
│  • 60 Hz → higher current → more thermal + H₂S stress          │
│  • Check: motor rated for 60 Hz in H₂S service?               │
│  Action: Verify motor rating, monitor winding insulation        │
├─────────────────────────────────────────────────────────────────┤
│  КЛ (Cable) — HIGH RISK                                         │
│  • #1 failure mode in Vt (25% of failures)                     │
│  • β=0.73 — 45% fail before 90 days                            │
│  • 60 Hz → higher voltage/current → more insulation stress     │
│  Action: Use cable rated for max continuous current at 60 Hz    │
└─────────────────────────────────────────────────────────────────┘
```

### Risk Tier 2 — Monitor closely

```
Слом вала (Shaft breakage) — 16% of Vt failures
  • Higher startup torque at 60 Hz → greater shock load
  • Action: VFD slow ramp-up, avoid step-start at 60 Hz
```

### Risk Tier 3 — Minimal direct frequency sensitivity

```
Засорение РО (clogging), НКТ (tubing), Износ РО, Гидрозащита
  → Driven by fluid/reservoir conditions, not electrical frequency
```

---

## Slide 10 — Key Conclusions

| Question | Answer |
|----------|--------|
| Does 60 Hz shorten ESP life in Vt? | **Trend adverse; −6% estimated, marginally significant** |
| Is the effect large? | **No — environment and category mix dominate** |
| Is there an Hz-day fatigue budget? | **No — rejected by 4 independent statistical tests** |
| What fails first at 60 Hz? | **ПЭД (motor) and КЛ (cable)** |
| What should we monitor? | **First 90 days (45% of failures), motor insulation** |

### Key numbers to remember

- Well-level logrank p = **0.69** — frequency effect disappears at well level
- Cox HR (High vs Normal, fully adjusted) = **1.05** [0.91–1.21], p=0.48
- AFT acceleration at 60 Hz vs 50 Hz = approximately **−6%** TTF (p=0.04)
- ПЭД β in Vt = **1.12** (only wear-out pattern in the entire dataset for this field)
- KЛ η in Vt = **91 days** (half of cable failures by day 91)
- TRF capacity: KW p = **0.0001** (definitively rejected)

### Recommendation

**Select 60 Hz operation for Vt wells with:**
1. Motor explicitly rated for 60 Hz in H₂S service
2. Cable rated for full-load current at 60 Hz operating conditions
3. VFD with programmable slow ramp-up
4. Enhanced monitoring plan for first 90 days

**Do not expect a large reduction in ESP life from frequency alone.**  
The biggest levers for extending Vt ESP life are: cable quality selection, motor H₂S rating, and reducing infant failures (installation quality, ramp-up procedure).

---

## Slide 11 — Frequency Definition Matters: Mean vs Proportion-Based

### Two ways to classify "high frequency"

| Definition | Criterion | Global n High | Vt n High |
|-----------|-----------|--------------|-----------|
| **Current** | `freq_w_mean > 55 Hz` | 407 | **32** |
| **New** | `> 50% of operating time above 55 Hz` | 435 | **44** |

108 runs (4.1%) change group. Most movers:
- 64 "Normal" → "High": ran above 55 Hz most of the time, but mean was pulled down by occasional low-speed periods
- 40 "High" → "Normal": high average but only intermittently operated above 55 Hz

### Impact on Vt high-frequency group

| Metric | Mean-based | Proportion-based |
|--------|-----------|-----------------|
| Vt n_High | 32 | **44** (+38%) |
| Median TTF (failures) | 102 d | **137 d** |
| Infant rate (90d) | 45% | **36%** |
| Cox HR (field-adjusted) | 1.18, p=0.48 | **0.97, p=0.88** |
| Weibull β | NaN (n too small) | **0.958 (≈ random)** |

### The key insight

> **Wells that run stably at high frequency show no excess hazard.**  
> The bimodal/intermittent-high wells (high mean, not majority time above 55 Hz)  
> are the **worse performers** — they drive the adverse signal in the mean-based definition.

**Physical interpretation:** The risk is not sustained 60 Hz operation — it is **frequency cycling**:  
repeated transitions between high and low speed create thermal cycling, VFD ramp-up shocks,  
and mechanical transient loads. A pump running steadily at 60 Hz is safer than one cycling 30 ↔ 65 Hz.

---

## Appendix — Sample Size Note & Data Provenance

### Current vs new high-frequency definition

The analysis uses `freq_w_mean > 55 Hz` (weighted average above 55 Hz).  
Alternative definition: `freq_above_55hz_pct > 50%` (majority of operating time above 55 Hz).

| Metric | Current (mean > 55 Hz) | New (> 50% time above 55 Hz) |
|--------|----------------------|------------------------------|
| Global n | 407 | 408 |
| Global failures | 270 | 266 |
| Vt n | **32** | **42** |
| Vt failures | 28 | 35 |
| Median TTF (failures, global) | 107 d | 131 d |
| Infant rate (90d, global) | 33% | 27% |

**83% of runs are identical under both definitions** (340 shared runs).  
The 67 runs that change from "high" to "normal" under the new definition have high mean frequency but spent <50% of time above 55 Hz — likely bimodal operation (burst-high periods).  

> The new definition adds 10 more Vt runs to the high-frequency group (32 → 42),  
> which improves statistical power for Vt-specific estimates.  
> **Full re-analysis with the new definition is being implemented.**

---

*Analysis package: `analysis/vt_failure/`  
Executive summary: `docs/vt_60hz_safety_assessment.md`  
Output files: `analysis_outputs/vt_failure_analysis/` (78 files, 13 phases)*
