# Effect of ESP operating regime on failure rate. Analysis for Vt

> Draft presentation for review. After approval → PPTX.
> Figures: `results/production_risk_vt_ttf_covariates/2026-07-23/figures/`. Data as of 2026-07-22.
>
> **Glossary.**
> **TTF** — time to failure / operating life: how long a pump runs before it fails (days).
> **Median** — the age by which half the pumps have failed.
> **Full mean life** (in statistics *MRL — Mean Residual Life*) — the mean time to failure in
> full, including the whole tail.
> **Mean life over a horizon** (in statistics *RMST — Restricted Mean Survival Time*) — how
> long a pump works on average over a fixed near-term window (e.g. 730 days), without looking
> into the far tail.
> **Ql** — liquid rate; **Kpod** — pump loading (fraction of nominal); **GLF** — gas-liquid
> factor. **Contractors:** brt / slb / other (oth).

---

## 1 — Conclusions

- **Question:** how does pump operating life (TTF) on the **Vt** field depend on the operating
  regime — liquid rate (Ql), frequency, loading (Kpod), gas (GLF) — and what formula should the
  life forecast use?
- **Data:** empirical survival curves across **6 groups** (H₂S class: sour / nonsour ×
  contractor brt / slb / other). ~530 nonsour + ~136 sour runs in total.
- **Headline:**
  1. **Liquid rate is the only operating factor** that matters for nonsour: higher rate —
     shorter life. Frequency and Kpod reflect **the same rate** (pump affinity law).
  2. **Sour wells are a separate world:** ~3–4× shorter life, and **the regime does not affect
     it** — H₂S corrosion drives everything.
  3. **Gas (GLF) — no effect**, excluded.
  4. The 2× brt/slb difference is **mostly rate** (slb runs at higher rates); a small
     unexplained residual reproduces on another field — kept as an empirical adjustment.

---

## Part I. Results: model, parameters and calculation

---

## 2 — Final life-forecast formula

**Meaning:** a group baseline life, adjusted for rate and for the contractor. The model is
**log-linear**: life is multiplied by `exp(coefficient × Δ)`, where Δ is the parameter's
deviation from its reference value (for rate — on the log scale).

```text
Life(well) = η_ref · contractor_mult · exp( γ · Δ ),   Δ = ln( clip(Ql, 47, 823) ) − ln(250)
    clip = winsorize Ql to the bounds 47–823: below 47 or above 823 the adjustment is
           held at the boundary and grows no further (no extrapolation)
```

| class | curve shape β | characteristic life η_ref, d | Ql_ref | Ql bounds | coefficient γ (per Δln Ql) | brt | slb | other |
|---|---|---|---|---|---|---|---|---|
| **nonsour** | 1.15 | 571 | 250 | 47–823 | −0.305 | ×1.00 | ×0.79 | ×0.54 |
| **sour** | 1.37 | 149 | 346 | 73–779 | ≈0 (no effect) | ×1.00 | ×1.05 | ×0.65 |

*brt is the reference contractor, its multiplier = 1.00 (the others are relative to brt).*

- **Reference value** — the point where the adjustment = 1 (`exp(0)=1`); `η_ref` is the baseline
  life at Ql = reference and contractor **brt**. `exp(γ·Δ)` is equivalent to `(Ql/ref)^γ`.
- **Curve shape β ≈ 1** means: failure risk hardly **grows with age** (not "wear-out").
- Relationships are **observational** (not an experiment). For sour the rate coefficient ≈0 → in
  practice **life = η_ref · contractor multiplier** (converted to the median).

---

## 3 — Coefficients: all three parameters (log-linear model)

Life is multiplied by `exp(γ · Δ)`, where Δ is the **parameter's deviation from its reference
value** (for rate — on the log scale). Coefficients γ (nonsour, from the data; significant in bold):

| known | γ Ql (per Δln Ql) | γ frequency (1/Hz) | γ Kpod (per 1.0) | case |
|---|---|---|---|---|
| Ql + frequency + Kpod | −0.130 | −0.0142 | **−0.729** | **all three (split)** |
| Ql + Kpod | −0.191 | — | −0.597 | two parameters |
| Ql + frequency | −0.190 | −0.0161 | — | two parameters |
| frequency + Kpod | — | −0.0171 | −0.812 | two parameters |
| Ql only | **−0.240** | — | — | one (fallback) |
| frequency only | — | **−0.0222** | — | one (fallback) |
| Kpod only | — | — | **−0.729** | one (fallback) |

**Reference values** (where all adjustments = 1): **Ql 250 m³/d, frequency 50 Hz, Kpod 0.70**;
baseline life **η_ref = 570 d** (brt). Contractor: **brt = 1.00 (reference), slb ≈0.73, other ≈0.64**.
**Applicability range** (q05–q95, do not extrapolate beyond): **Ql 47–823, frequency 31–56 Hz, Kpod 0.2–1.2**.

```text
Life = η_ref · contractor · exp( γ_Ql·(ln clip(Ql,47,823) − ln 250) + γ_freq·(freq − 50) + γ_Kpod·(Kpod − 0.70) )
```

**How to apply** — for each known parameter compute `exp(γ · Δ)`:
- frequency **55 Hz** = +5 from 50 → `exp(−0.0142·5) =` **×0.93** (−7 %);
- Kpod **0.90** = +0.20 from 0.70 → `exp(−0.729·0.20) =` **×0.86**;
- Ql **400** vs 250 → `exp(−0.130·ln(400/250)) =` **×0.94**.

Then multiply the adjustments of the known parameters × contractor × η_ref (drop missing ones,
do not impute). **For sour all γ ≈ 0** → contractor only. Median / life — slide 5.

---

## 4 — Worked example

Well **W**: nonsour, contractor **slb**, Ql **400** m³/d, frequency **52** Hz, Kpod **0.90**.
All three known → the "all three" set (β=1.28, η_ref=570; references **250 m³/d / 50 Hz / 0.70**):

```text
Δ Ql       = ln(400/250) = +0.470  → contribution = −0.130·0.470 = −0.061  (400 within 47–823)
Δ frequency = 52 − 50     = +2 Hz   → contribution     = −0.0142·2     = −0.028
Δ Kpod     = 0.90 − 0.70 = +0.20   → contribution     = −0.729·0.20   = −0.146
sum of contributions = −0.235      → exp(−0.235)                     = 0.790
contractor multiplier (slb)                                          = 0.728
characteristic life η = 570 · 0.790 · 0.728                          = 328 d
```

| quantity | what it is | value |
|---|---|---|
| Median | half the pumps have failed by this age | **247 d** |
| Full mean life (MRL) | mean time to failure in full | **304 d** |
| Mean life over 2 years (RMST, 730 d) | how long it runs on average over the next 730 d | **292 d** |

**If only Ql is known** → the baseline formula (β=1.15, η_ref=571, Ql reference 250):
multiplier `0.79 · exp(−0.305·ln(400/250)) = 0.686`, η=392 → **Median 285 / mean life 373 /
2-year life 335 d**. (Fewer parameters used — a more conservative estimate, so the life comes out longer.)

> This calculation ports 1:1 to Excel: the parameters (η, β, references, adjustments, contractor)
> on a separate sheet, functions compute the four quantities. (The Excel workbook is a separate task.)

---

## 5 — Which quantity to report as TTF

The same forecast can be expressed three ways:

- **Median** — the age by which **half** the pumps have failed. Simple, but only speaks to the
  "middle" and ignores the spread.
- **Full mean life** (MRL) — the **mean** time to failure in full. An honest mean, but it partly
  relies on the far "tail", which the data observe only weakly.
- **Mean life over a horizon** (RMST over 730 d) — how long a pump **runs on average over the
  next 2 years**. It invents nothing beyond the data and is always computable.

**Recommendation: report "mean life over 2 years" (RMST) as the primary quantity**, with
"full mean life" beside it, and the median for reference only. Values by group:

| class | contractor | multiplier | Median, d | Full mean life, d | 2-year life, d |
|---|---|---|---|---|---|
| nonsour | brt | 1.00 | 415 | 544 | 424 |
| nonsour | slb | 0.79 | 329 | 431 | 369 |
| nonsour | other | 0.54 | 224 | 294 | 278 |
| sour | brt | 1.00 | 114 | 136 | 136 |
| sour | slb | 1.05 | 119 | 143 | 143 |
| sour | other | 0.65 | 74 | 88 | 88 |

> For nonsour "full mean life" is noticeably larger than "2-year life" — because some pumps live
> longer than 2 years (long tail). For sour the lives are short, both quantities coincide.
> The remaining life of a running pump is measured from its **current age**.

---

## 6 — Limitations (what to keep in mind)

- Ql sets the **overall trend**, but it predicts the life of a **single** pump weakly — the model
  is good for planning **by group / fleet**, not for the exact date of one pump.
- The contractor multiplier is **observational**; the cause is only partly understood (about a
  third of the difference is runs cut short by **planned workovers (GTM)**, not failures).
- All relationships are **observational**, not from an experiment.
- Frequency and Kpod are **not in the TM-06 plan** — shown for understanding; the forecast is
  computed from Ql.
- For sour the regime has **no effect** — do not add regime adjustments without new H₂S data.
- **Deprecated:** the previously deployed "penalty for high Kpod loading" is **not supported** by
  this analysis (see slide 12).

---

## Part II. Analysis — model justification

---

## 7 — Groups and overall picture

Empirical survival curves (share of running pumps over time, from the data) across the 6 groups
+ the fitted model overlay:

![Survival curves + model by group](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/km_v31_per_stratum.png)

| group | runs | failures | shape β | model median, d |
|---|---|---|---|---|
| Vt_nonsour_brt | 231 | 81 | 1.15 | 436 |
| Vt_nonsour_slb | 189 | 99 | 1.15 | 320 |
| Vt_nonsour_other | 70 | 23 | 1.15 | 236 |
| Vt_sour_brt | 40 | 28 | 1.37 | 113 |
| Vt_sour_slb | 41 | 36 | 1.37 | 121 |
| Vt_sour_other | 55 | 46 | 1.37 | 72 |

> The model sits **inside the confidence band** of each group. Full parameters (characteristic
> life, reference Ql, bounds, adjustments, fallbacks) — slides 2–3.

---

## 8 — Nonsour wells: life and model

*(top row of the figure, slide 7)*

- Nonsour lives long and is **rate-driven**: model median **436 / 320 / 236 d** (brt / slb /
  other); **shape β ≈ 1.15** — failure risk hardly grows with age.
- The model (Ql + contractor) **reproduces** each contractor's empirical curve.
- The brt↔slb difference is mostly a **rate difference** (discussed on slide 14).

---

## 9 — Sour wells: life and model

*(bottom row of the figure, slide 7)*

- Life is **~3–4× shorter** than nonsour: median **113 / 121 / 72 d** (brt / slb / other).
- **Shape β ≈ 1.37** — mild wear from day one (the nonsour curve is flatter).
- The order **other < brt ≈ slb** is set by **H₂S corrosion**; the regime does not affect life,
  so for sour the model = baseline life · contractor multiplier (slides 2, 16).

---

## 10 — The main factor: liquid rate

![Survival curves by Ql band, nonsour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_km_by_ql_bin.png)

![Median life by Ql band + trend](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_ql_bin_trend.png)

- Life **steadily decreases** with rising Ql (the higher the Ql band, the lower the curve).
- The shape was chosen **from the data**: log of Ql with the **extreme 5% clipped** top and bottom
  (beyond the working bounds the adjustment is not increased).
- The Ql effect is **statistically significant** (p<0.001): life ×0.74 for a ~2.7× rise in Ql.
- **Reproduces on another field (Ya)** — the same effect, so the relationship is not by chance.

---

## 11 — Frequency: the same physics through rate

![How frequency raises rate and shortens life](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/freq_effect_mechanism.png)

![Survival curves by frequency band, nonsour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_km_by_freq_bin.png)

- **Raising frequency raises rate** (affinity law): roughly **+10 m³/d per Hz**.
- Frequency is the clearest single signal: **failure risk +1.9% per Hz** (significant), but with
  Ql held fixed the effect nearly vanishes — **frequency acts precisely through rate**.
- Net along the frequency→rate chain: **+5 Hz ≈ −4 %, +10 Hz ≈ −8 %** of life.
- **For deployment:** frequency and Kpod are not in the TM-06 plan, so the forecast **is computed
  from Ql** — frequency/Kpod are the same physics, shown for understanding.

---

## 12 — Loading (Kpod): checking the "safe zone"

![Survival curves by Kpod band, nonsour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_km_by_kpod_bin.png)

![How risk changes with loading](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/kpod_spline_logHR.png)

**Is there a "safe zone" at nominal (the previously deployed high-loading penalty)? — No.**
- The flexible risk curve (right figure) shows: **no protective valley** at nominal.
- Risk **rises monotonically with loading** across the whole working range; the minimum is at the
  low edge.
- Direct check: **low loading is not harmful** (robust to how it is counted).
- ⇒ **A penalty for exceeding nominal is not justified**; loading is the same rate, in other words.

---

## 13 — Gas (GLF): no effect — excluded

![Survival curves by GLF band, nonsour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_km_by_glf_bin.png)

- Survival curves by GLF band **cross** — no ordering (medians 459 / 287 / 343 / 438 d mixed
  up), unlike the clear ordering by Ql (slide 10).
- **No effect** on either risk or life (not significant).
- Free-gas rate is also **not significant** (a single spike in one group does not repeat).
- **Why excluded:** no signal, and gas rate is also confounded with well productivity. Vt is the
  gassiest field, so this is an **honest zero**, not a lack of data.

---

## 14 — brt vs slb, nonsour — the "lesson" explained

**Parameters by contractor** (a separate Weibull fit per stratum, no adjustments):

| contractor | runs | failures | shape β | char. life η, d | median, d | mean, d |
|---|---|---|---|---|---|---|
| **brt** | 231 | 81 | 0.82 | 897 | 573 | 1001 |
| **slb** | 189 | 99 | 0.81 | 482 | 306 | 542 |
| other | 70 | 23 | 0.87 | 429 | 282 | 460 |

*These are raw per-stratum fits; β<1 reflects the spread between wells, not wear-out. In the final
model (slide 2) the contractor is a multiplier on a common baseline, and the shape is β≈1.15.*

Empirical survival curves by contractor (the raw picture — brt clearly above slb):

![Survival by contractor, nonsour (KM)](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/nonsour_contractor_km.png)

![Ql and loading brt vs slb, nonsour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/contractor_ql_kpod_overlap.png)

- At first glance — a 2× contractor advantage (median brt 573 vs slb 306 d), **but slb runs at
  noticeably higher rates**: median **305 vs 193 m³/d** (the ranges overlap, they are not just
  different).
- **Match the rate** (range 193–305): the gap **disappears** — brt 403 d vs slb **440 d**.
  **So the 2× is mostly rate.**
- A small residual (slb ×0.79) persists and **repeats on another field (Ya)** — i.e. it is real.
  If we account for **runs cut short by planned GTM** (not failures), about **a third** of the
  residual is explained by this; the rest is unattributed (make/service).
- **Verdict:** the contractor is kept as an **observational adjustment**, the cause marked "not
  established".

---

## 15 — brt vs slb, sour

![Ql and loading brt vs slb, sour](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/sour_contractor_ql_kpod_overlap.png)

- The same rate pattern: slb runs at higher rates (median 464 vs brt 331).
- But for sour the **regime does not affect life** (slide 16), so here the contractor multiplier
  is **not a rate artefact**: it reflects a real **shortfall of the "other" group** (life ×0.65,
  significant). brt ≈ slb (difference not significant).

---

## 16 — Why the regime is excluded for sour

![sour: survival by Ql](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/sour_km_by_ql_bin.png)
![sour: survival by frequency](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/sour_km_by_freq_bin.png)
![sour: survival by Kpod](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/sour_km_by_kpod_bin.png)

**Why the regime parameters are removed from the sour formula:**
- **Ql:** no effect, and the order is even **reversed** vs nonsour (low-rate sour wells die
  faster, 65 vs 117 d) — this is **confounding** (the worst wells run quietly), not a rate law.
- **Frequency / Kpod:** **no ordering** (frequency 74/125/105 d; Kpod 112/87/106 d).
- **Cause:** H₂S corrosion is a **threshold** (~120 mg/l) that overrides everything else; once a
  well is sour, the regime barely changes its life.
- ⇒ The sour formula = **baseline life + contractor**; the regime parameters are set aside (no effect).

---

## 17 — Validation

![Model validation](../../results/production_risk_vt_ttf_covariates/2026-07-23/figures/model_validation_v31.png)

- **Agreement with fact:** predicted median within ±10 % for **4 of 6 groups** (thin "other"
  groups flagged); on the left figure the points fall on the diagonal.
- **Another field:** the rate dependence and the slb multiplier **repeat on the Ya field**.
- **Out-of-sample:** trained on installs **before 2024**, tested on **2024+** — the rate
  dependence transfers; the baseline level "drifts" ~20–30 % with install vintage (a
  recalibration knob, not a model error).
- **Robustness:** 200 recomputes on a resampled dataset — the confidence bounds cover the fact in
  **5 of 6 groups**.

---

## Appendix — files and tables

- Equation: `TTF_ADJUSTMENT_EQUATION.md` (parameters, survival curve, computation of median /
  mean life / horizon life + table).
- Full reports: `docs/notes/vt_ttf_operating_covariates_findings.md` (EN) / `_ru.md` (RU).
- Key tables: `revised_operating_model_v2.csv`, `availability_life_multipliers.csv`,
  `model_consistency_checks.csv`, `replicate_v31_on_ya.csv`, `temporal_holdout_validation.csv`,
  `bootstrap_model_cis.csv`, `informative_censoring_test.csv`,
  `contractor_rate_matched_cohorts.csv`,
  `kpod_quadratic_cox.csv` / `kpod_spline_meta.csv` / `kpod_risk_min.csv`.

---

## 18 — Agent prompt: implementing the calculation (dependencies + MRL/RMST formulas)

A copy-paste prompt: how to **write the calculation itself** of operating life from ready-made
parameters — the dependence on contractor/covariates and the four quantities (median, mean life,
horizon life, remaining life). Any language; VBA and Python examples below. The model is NOT re-fit.

```text
GOAL. Implement functions that compute ESP operating life from READY-MADE parameters (do not
re-fit). Four quantities: MEDIAN, MEAN LIFE (MRL at 0), HORIZON LIFE (RMST over tau),
REMAINING LIFE (MRL at age t). Any language (VBA/Python/…). Keep the parameters on a separate
sheet/table; the functions read them.

INPUT (sheet "Parameters", per H₂S class nonsour/sour):
  beta (curve shape), eta_ref (baseline life at the reference values, days),
  references X_ref: Ql_ref=250, freq_ref=50, Kpod_ref=0.70,
  coefficients g (log-linear): g_Ql (per Δln Ql), g_freq (1/Hz), g_Kpod (per 1.0)
     — from the row of the availability pattern (which covariates are known; DROP the missing),
  contractor multipliers (brt=1, slb, oth),
  applicability range (nonsour, q05–q95): Ql 47..823, freq 31..56, Kpod 0.2..1.2,
  horizon tau (default 730).

STEP 1 — multiplier phi via exp-log over DEVIATIONS Δ from references (known covariates only):
  clamp parameters to the applicability range (do NOT extrapolate beyond), e.g.
     q = min(max(Ql, 47), 823);
  s =  g_Ql   · (ln(q) − ln(Ql_ref))      // Ql: deviation on the LOG scale
     + g_freq · (freq − freq_ref)          // DROP the term if frequency is unknown
     + g_Kpod · (Kpod − Kpod_ref)          // DROP if Kpod is unknown
  phi = contractor_multiplier · exp(s)
  eta = eta_ref · phi

STEP 2 — quantities from (eta, beta). Survival curve S(t)=exp(−(t/eta)^beta):
  MEDIAN            = eta · ln(2)^(1/beta)
  MEAN LIFE MRL(0)  = eta · Γ(1 + 1/beta)
  HORIZON LIFE      = eta · Γ(1 + 1/beta) · P(1/beta, (tau/eta)^beta)     // = RMST(0,tau)
  REMAINING LIFE(t) = eta · Γ(1 + 1/beta) · Q(1/beta, (t/eta)^beta) / exp(−(t/eta)^beta)
      where Γ is the gamma function; P(s,x) is the regularized LOWER incomplete gamma; Q(s,x)=1−P(s,x).

SPECIAL FUNCTIONS (the only nontrivial part) — how to get them:
  Γ(z):     Python  math.gamma(z)
            VBA     Exp(WorksheetFunction.GammaLn(z))
  P(s,x):   Python  scipy.special.gammainc(s, x)          // already the regularized lower one
            VBA     WorksheetFunction.Gamma_Dist(x, s, 1, True)   // = P(s,x)
            no library: lower incomplete gamma(s,x) by series  gamma(s,x)=x^s·e^−x·Σ_{k≥0} x^k/(s·(s+1)…(s+k)),
                        then P = gamma(s,x)/Γ(s); converges fast for x≲s+1 (else via Q).
  Q(s,x) = 1 − P(s,x)

CHECK VALUES (nonsour/slb, Ql=400, freq=52, Kpod=0.90, pattern "all three":
  beta=1.28, eta_ref=570, references 250/50/0.70, g = −0.130/−0.0142/−0.729, slb=0.728):
  Δ: ln(400/250)=+0.470, freq +2, Kpod +0.20;  s=−0.235;  exp(s)=0.790;  phi=0.576;  eta=328
  → MEDIAN 247, MEAN LIFE 304, HORIZON LIFE over 730 d = 292.
  Fallback "Ql only" (beta=1.15, eta_ref=571, ref 250, g_Ql=−0.305, slb=0.792):
  phi = 0.792·exp(−0.305·ln(400/250)) = 0.686, eta=392 → 285 / 373 / 335.

NOTES. Median and mean life scale linearly with phi; horizon life scales sub-linearly (tau
fixed). Report "horizon life" as primary, the median for reference. For sour the regime
coefficients g = 0 (contractor only).
```

> This is a calculation spec (not a re-fit). It is also the basis of a separate task: an Excel
> workbook with the parameters on a sheet and functions `TTF_Median / TTF_Mean / TTF_RMST / TTF_MRL`.
