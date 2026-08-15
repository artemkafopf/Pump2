# Мирнинский / Weibull shape — findings

Дата: 2026-07-16, **существенно исправлено 2026-07-17**. Session goal: the Мирнинский
(Mc + Mr) failure model looked wrong («500 days median is very wrong»), and the
`Прогноз_ремонтов` sheet behaved oddly (`Факт ННО` > `Вероятностный прогноз`).

> ## ⚠ CORRECTION NOTICE — read before quoting anything from this file
>
> The 2026-07-16 version of this note derived its headline claims from an
> `esp_population` build that **silently omitted Big-only *closed* runs** — for Ya,
> **a third of its events** (784 present vs 1226 real). The Vt session found the gap
> (`project_vt_deposit_and_form`); it is now fixed and guarded by
> `backend/tests/test_esp_population.py`.
>
> **Re-verified on the corrected population, three headline claims DIED:**
>
> | claim (2026-07-16) | status | corrected |
> |---|---|---|
> | "Weibull over-states mid-life 1.4–1.8× **fleet-wide**" | **DEAD** | Ya median ratio **1.02**, Global **1.03** — the registry tracks them almost perfectly. The form defect is real only on **Az / Ic / Mc / Vt_sour**. |
> | "B50 unidentifiable, +59% at 37 events, converges down" | **DEAD** | Unbiased at Mc's volume (bootstrap median **494** vs truth **479**); merely **noisy** (p90 782, occasional blow-ups). The +59% was the missing-data bias. |
> | "Mc SMR 1.23 vs Ya_brt — ~23% hot" | **DEAD** | **SMR 0.82** (p=0.088) — Mc is ~18% **cooler** than comparable Борец pumps. |
>
> Ya's fitted b50 also moved: **brt 844 → 479**, **slb 476 → 368**.
>
> **The lesson, and the cheap check that would have caught it on day one:**
> **assert fit `n_failures` against `esp_models.csv` before trusting any population.**
> Everything below is post-correction unless marked otherwise.

**Nothing shipped changed.** New code is additive and default-off.

---

## Verdict

0. **TWO deliverables need TWO estimands, and one registry was asked to serve both.**

   | need | estimand | ГТМ |
   |---|---|---|
   | **Прогноз ремонтов / график бригад** — МРП | **all-cause** (any pull) | = событие |
   | **Интенсивность отказов (KPI), потери нефти** | cause-specific failure hazard | censored |

   The 2026-07-15 refit "corrected" the ГТМ labelling. That is **right for the KPI**
   and it **removes the quantity the schedule needs**. Both estimands are legitimate;
   they should **coexist, not replace each other**. That framing is the durable part.

   > ### ⚠ What the 2026-07-13 bundle actually is — corrected 2026-07-17
   >
   > An earlier version of this note claimed "07-13 IS the МРП model, and its b50 is
   > effectively calendar — hand it to planners". **Both claims were reached by the
   > wrong route and are only partly true. Do not quote them.**
   >
   > **The 07-13 file is TWO fits mixed together** (`fit_date` proves it): the
   > `Mc_nonsour_Pooled` row is 2026-07-13, **every other row is 2026-07-08**.
   >
   > **The Mc row** comes from `production_risk/mc_refit.py` — a deliberately narrow,
   > Mc-only refit:
   > * population = **Big (`is_esp`, NOT `is_esp_strict`) + Свод** — *not* V03 /
   >   `mart__weibull_input` (whose Mc side has only **114** runs, and cannot produce
   >   `n_runs=266`).
   > * event = `fail_date notna and <= cutoff` ⇒ **ГТМ pulls carrying a stamped
   >   «Дата отказа» count as failures** ⇒ effectively **all-cause** (160/266 = 60%).
   > * clock = `_duration_days()`: **`nno_days` when present, else `calendar × 0.745`**
   >   — a HYBRID, mostly ≈ Наработка (≈calendar). *This* is why Mc's b50=224 matched
   >   my calendar all-cause KM median (280), not any general property of the bundle.
   > * window = `RECENCY_START = 2024-01-01`; its docstring already records the 2024+
   >   choice and that "the residual Мирнинский under-count is an idle/workover-month
   >   failure pattern, not a fit-window artifact".
   >
   > **The other nine rows are 07-08 fits on `ttf_mix` / `ttf_true_best_days`, which is
   > GENUINELY operating time**: measured `ttf_true / run_days` median **0.667**
   > (p10 0.13, p90 0.97; only 1% exceed 0.99), sourced from `techregime_status` (2041
   > of 2634). By contrast Свод «Наработка» is **0.986** of calendar. **That is a real
   > ~33% clock gap between the two bundles** — the concern raised in-session was
   > correct, and it is larger than any modelling difference argued about all day.
   >
   > ⇒ **b50 is NOT uniformly calendar.** It is ≈calendar for the Mc row (hybrid clock)
   > and op-days for the rest. The b50↔calendar-median agreement seen on
   > Da/Za/Vt/Ic/Ya **remains unexplained** and must not be generalised.
   >
   > **`uptime_factor` is still a fudge**: Ya ships 0.47–0.62 while measured telemetry
   > uptime is **0.91**, and Da's 0.467 implies b50/uptime = 1054 d, self-evidently not
   > a pump life. `mc_refit` reads it back out of the 07-08 bundle to patch missing
   > `nno_days` — so the fudge propagates into the newer fit.

   * **All-cause wear-out does NOT replicate.** The 07-13 registry shows `beta2`
     1.28 (Mc) / 1.42 (Za) / 1.56 (Ya), but **re-fitting the same all-cause estimand
     on the corrected population gives Mc 1.05, Za 0.875, Ya 0.797** — i.e. ≈0.8, the
     same as the failure-only process. **The conditional-residual claim built on those
     shipped parameters ("Mc 225→211→172, old pumps more due") does not survive
     re-fitting and must not be used.**
   * **Trimming early failures does NOT reveal wear-out** (all-cause k2, free betas,
     left-truncated at c = 0/3/7/30/60/90): Mc `beta2` goes the WRONG way,
     **1.050 → 0.886**; Ya stays 0.797–0.818; Vt 0.802 → 0.687; Az 0.784 → 0.890.
     Only **Za (1.15–1.18 for every c≥3) and Ic (1.08–1.21)** hold `beta2 > 1` — those
     two are genuine. Fit quality *does* improve (Za max|ΔS| 0.041 → 0.019, Mc
     0.063 → 0.035) — better fit, **same shape**. `beta1` pins to its bound (10) almost
     everywhere: the early component keeps collapsing to a point mass (schema wart).

1. **Мирнинский is fine — if anything better than Ya.** Age-standardised against the
   like-for-like donor `Ya_brt` (Mc is 166/168 Борец): **SMR 0.82** (47 observed vs 57.6
   expected, p=0.088). *Match on contractor before comparing fields* — SLB fails ~1.55×
   faster, so a pooled donor confounds. Note SMR has now given **three** answers across
   two populations (0.97 → 1.23 → 0.82); trust only the corrected, contractor-matched one.
2. **The empirical hazard is infant spike → flat, with NO wear-out.** This is the one
   claim that got *stronger* under correction (see §1). It is the basis for everything else.
3. **The shipped Weibull is fine on the big strata** (Ya 1.02, Global 1.03) and poor on
   the small ones (Az 1.00 spread, Ic 1.56 spread, Mc 1.60 spread). **Do not
   blanket-replace the registry.**
4. **A per-well workover date schedule is not achievable** from this data.
5. **Best option for Mc:** its OWN spike+constant fit on the 2024+ regime — Mc is the
   worst-fitted stratum in the registry and B repairs it (§6).

---

## 1. The empirical hazard: infant spike, then flat, no wear-out  ← SURVIVES, STRONGER

`Ya_nonsour` on the **corrected** population (2166 runs / **1226** events):

| age, d | 0–30 | 30–90 | 180–300 | 300–450 | 600–800 | 1100–1500 |
|---|---|---|---|---|---|---|
| **h/month** | **0.1122** | 0.0529 | 0.0448 | 0.0303 | 0.0416 | **0.0259** |
| at risk | 2166 | 1833 | 1197 | 882 | 498 | 162 |
| failures | 218 | 173 | 181 | 114 | 108 | **38** |

**Infant spike = 2.1× the plateau.** Hazard at 1100–1500 d (0.0259) is **below** 30–90 d
(0.0529), on 38 real failures — not a small-sample artefact. `Ya_nonsour_brt` (782 events)
is identical in shape.

⇒ `beta < 1` in the registry is a **fitting artefact**, not physics — it is what produces
the unphysical "RUL grows with age". True RUL is *flat* after month one.

## 2. Fit vs its own empirical hazard — per-stratum audit (corrected)

ratio = fitted hazard at band midpoint / empirical hazard in band; bands with ≥5 events.
`median` = level bias, `spread` = shape error (see the metric note below).

| stratum | events | kind | **A median** | **A spread** | B median | B spread | verdict |
|---|---|---|---|---|---|---|---|
| Ya_nonsour_Pooled | 1226 | k1_aic | **1.02** | 0.56 | 1.05 | 1.11 | **A** |
| Global_Pooled | 2127 | k1_aic | **1.03** | 0.48 | 1.11 | 1.26 | **A** |
| Vt_nonsour_Pooled | 209 | k2_high_w1 | 1.05 | 0.64 | 0.99 | 0.73 | **A** |
| Za_nonsour_Pooled | 152 | k2 | 0.97 | **0.43** | 1.00 | 0.59 | **A** |
| Vt_sour_Pooled | 110 | k1_aic | 0.99 | 0.47 | 0.98 | **0.38** | **B** |
| Az_nonsour_Pooled | 169 | k1_aic | 0.93 | 1.00 | 0.91 | **0.81** | **B** |
| Ic_nonsour_Pooled | 159 | k1_aic | 0.88 | 1.56 | 0.92 | **0.80** | **B** |
| **Mc_nonsour_Pooled** | 51 | k1_aic | 1.12 | **1.60** | **0.96** | **0.57** | **B** |

**The registry is good where the data is thick and poor where it is thin.** Spike+constant
wins on the four badly-fitted strata and *loses* on the two best (Ya, Global). The Vt
session's hybrid idea ("B only for `k1` strata") does **not** survive either — Ya and
Global are `k1_aic` and A wins on both.

*Metric note*: median alone misleads. Ic's A-fit has median 0.88 (looks fine) but spread
1.56 — the curve is wrong everywhere and the errors merely straddle 1.0. The ≥5-event
filter also changes the band count, so low-event strata (Mc) are least trustworthy.

## 3. B50 at Mc's data volume — noisy, NOT biased  ← CORRECTED

Bootstrap from the full corrected `Ya_nonsour_brt` (truth b50 = **479**):

| subsample | b50 median | p10 | p90 | \|error\|>25% |
|---|---|---|---|---|
| n=197 runs (~106 events) | **494** | 411 | **782** | 16% |
| n=800 runs (~429 events) | 490 | 454 | 4215 | 22% |

**Unbiased, heavy right tail, occasional blow-ups.** The time-truncation experiment that
produced the old "+59%" claim cannot even reach Mc-like statistics on the corrected
population — Ya_brt already has 420 events in 2019. Its errors are now −19% → +2%.

Still true: **the forecast is insensitive to b50** — the fleet occupies ages 0–1200 d
where the curves barely differ. Target the hazard over the occupied range, not the median.

## 4. The op-day clock  ← SURVIVES

**Damage accrues while running, at a constant rate.** Landmark test, leakage-proof
(Кэкспл over days 0–90 → failures **after** day 90), corrected population:

| field | runs/events | **h_op CV** | **h_cal CV** | winner |
|---|---|---|---|---|
| Ya | 456 / 224 | **0.084** | 0.163 | op-days |
| Vt | 212 / 112 | **0.218** | 0.298 | op-days |

(The pre-correction run reported CV 0.028 — that was partly a coverage confound.)

**`Наработка (сут)` is NOT operating time.** Measured op-days / «Наработка», well-observed
runs: Ya **0.912**, Vt 0.906, Az 0.903, Mc **0.896**, Za 0.863, Ic 0.816, Da 0.793.
λ per op-day vs per «Наработка»-day: **1.37× (Mc)**, **1.28× (Ya)**.

**Кэкспл as a hazard covariate is a NULL and a TRAP.** The whole-run association is 2.6×
and in the "cycling damages pumps" direction — it is **reverse causation** (failing pumps
get cycled). The landmark dissolves it. Anyone re-running this without a landmark will
"confirm" the hypothesis and be wrong.

Plan `op_days`/`cal_days` = **0.846 (Mc)** / **0.911 (Ya)** — matches realized Кэкспл,
so **the plan's `op_days` is already correct op-day exposure**. See
`docs/notes/production_risk_clock_fix_plan.md`.

## 5. A per-well DATE is not predictable  ← SURVIVES

Constant hazard ⇒ memoryless ⇒ remaining life independent of age. How `repair_compat`
places a `0`:

1. `E_w` = expected failures for the well over the horizon.
2. Fleet total `N = round(Σ E_w)`.
3. Each well gets `floor(E_w)` — **0 for every Mc well** (all `E < 1`; mean 0.606, max 0.839).
4. The `N` remaining go to the wells with the largest fractional part → **top-N by E**.
5. **When**: inverse-CDF — event *k* lands where the well's cumulative expected-failure
   curve crosses `(k − u)·E_total/n`, with `u` a **hash of the well code**.
6. `_month_spread_date` picks the day via another stable hash.

⇒ **which wells** = top-N by E (near-arbitrary: CV 0.185), **when** = a seeded
pseudo-random quantile ⇒ ≈ uniform (median days-to-workover 244 vs horizon midpoint 274).
Reproducible, not predictive.

**Prioritise by RISK, not probability.** Probability is near-flat (CV 0.27); **oil rate
spans ~79×** (4.4 → 343 т/сут). `risk = E × oil` gives 60× spread; top-10 by risk overlaps
top-10 by probability on only **2 of 10** wells.

## 6. Option B for Mc

```
S(t) = w1·exp(-(t/eta1)^beta1) + (1-w1)·exp(-t/eta2)     beta2 pinned to 1
```

Fitted on Mc's own runs, left-truncated to the 2024+ regime:
`w1=0.0235, beta1=10, eta1=0.5, beta2=1, eta2=922.97` → **B50 618**; 2.4% infant mortality
then a flat ~3.2%/month. Uses the shipped five registry columns — `StrataModel.S` evaluates
it natively.

**Calibration (model/факт):** 2024 **1.05**, 2025 **0.83**, 2026H1 **1.23**; aggregate
**0.97**. Forecast **40** vs shipped **50**.

**Why B for Mc:** A gets the infant window backwards — P(fail)/month at age 0 vs 300 is
**0.044 / 0.044** under A (no discrimination) vs **0.055 / 0.032** under B; empirically a
new pump is ~2.1× worse. **~58 Мирнинский wells start at age 0 during the horizon.** And
A's extra 10 failures are extrapolated wear-out into ages where Mc has **zero** observed
failures (last Mc failure: 663 d).

**Weakness:** `beta1=10, eta1=0.5` sit on their bounds — the infant component collapses to
a point mass at day 0.5 rather than finding a real window. The Vt session showed this is a
**schema limit, not a fitter limit**: the 5-column form expresses a day-0 mass **or** an
infant window, not both, and the MLE genuinely prefers the mass.

## 6a. ННО / МРП — how to measure them, and how NOT to  (2026-07-17)

**`Σt / N_событий` is a RATE, not a LIFE. It is unbiased ONLY under a constant
hazard, and the hazard here is not constant.** Its error scales with the censored
fraction, so it fails worst on the young growing field:

| Mc, установки 2024+ (35% ещё работают) | | |
|---|---|---|
| наивное среднее поднятых пробегов | **175** | занижено: длинные ещё работают |
| **`Σt / подъёмы`** | **305** | **ЗАВЫШЕНО на 27%** — 58 непод­нятых насосов дают время в числитель и ноль в знаменатель |
| KM медиана (все причины) | 233 | |
| **KM RMST до 475 сут** | **241** | **честное среднее** — учитывает цензуру И форму |

Ya (8% running): `Σt/N` = 412 vs RMST **381** (+8%) — the bias tracks censoring.
The same flaw is worse for `Σt / N_отказов` ("ННО"): only ~26% of Mc runs end in a
failure, which is how it reaches **969 d against an observed failed-run mean of 191**
— the ГТМ counterfactual again. Its implied COUNT is still usable (38 700 pump-days
/ 969 ≈ 40, matching the model's 40–50) because a rate and a mean coincide under an
exponential; the *life* is not.

**`two_layer.mrp_days()` therefore uses RMST off the all-cause KM**, not `Σt/N`.
Corrected МРП (calendar; true op-days ≈ ×0.90):

| страта | МРП уст. 2024+ | вся история |
|---|---|---|
| Mc_nonsour_brt | **246** | 298 |
| Ya_nonsour_brt | 429 | 396 |
| Vt_nonsour_brt | 301 | 308 |
| Vt_sour_slb | 111 | 112 |

*(Better still: take the median from the 07-13 all-cause FIT — it is censoring-aware,
per-stratum, and ships `b20/b50/b80` + CIs. RMST here was a stopgap.)*

**Filter on INSTALL date, not exposure.** Left-truncating exposure keeps pre-window
installs' recent time and hides that recent runs are shorter: Mc ГТМ mean 295 → **182**,
отказ 191 → **166** once filtered on installs 2024+.

**Clock:** `tte` = Свод «Наработка (сут)» ≈ **calendar** (median tte/calendar = 0.986
over 1644 completed runs). Measured true op-days / Наработка: Mc **0.896**, Ya 0.912,
Vt 0.906, Ic 0.816, Da 0.793. An engineer hearing «МРП 305 суток» will assume operating
time — **label it «246 календарных / 222 оп-суток»**.

## 7. Estimand / naming  ← SURVIVES

* `Вероятностный прогноз` was an **ННО** (failure interval), not an **МРП**
  (межремонтный период, all-cause ≈ 294 d for Mc). It now carries **days to the first `0`**
  (remaining life from the forecast start) — verified against all 766 rows, 0 discrepancies.
  It is therefore **not comparable with «Факт ННО»** (an age) and cannot contradict it.
* Each `install → pull` is **one survival unit**; ages are **not** concatenated across
  pulls. Correct: `pump_serial` shows the same pump returns after ГТМ only **4.0%**
  (35/872 fleet-wide; 2/45 Mc).
* Censoring ГТМ is **right** for a failure forecast (cause-specific hazard); reading the
  resulting *survival* as real is not — it is a counterfactual.
* **`ДФ-04` contains no workovers.** Its 39 Мирнинский horizon entries are **30 ВНС** +
  **9 Перевод в ППД**; 22/39 `Нагнетательная`, 35/39 not running. **No forward source
  exists** for the ГТМ that drive real changeouts ⇒ the sheet can honestly forecast
  failures only.
* **Day-0 failures are real** (5 of 49 Mc: ПЭД / кабельная линия at Наработка = 0) and are
  silently dropped by `tte > 0` filters.

## 8. Мирнинский's "2024 regime change"

| Mc era | observed | expected (Ya_brt hazards × Mc age mix) | SMR |
|---|---|---|---|
| 2019–2023 | 2 | 11.8 *(pre-correction; recompute)* | 0.17 |
| 2024–2026 | 47 | **57.6** | **0.82** |

The **pre-2023 era** is the anomaly (2 failures on 21 runs / 11,316 pump-days — luck or
incomplete early records), not 2024+. Big holds **no hidden pre-2023 failures**: only 21
producer-ESP Mc runs installed pre-2023, 19 already in Свод, 2 Big-only (neither a
failure). The 40 pre-2023 rows with fail dates are `Нагнетательная` — injection wells,
correctly excluded by `is_esp_strict`.

## 9. Tested and FAILED — do not retry

* **ГТМ pump-age reset in the projection** — implemented, effect **−2%** (50.3 → 49.2):
  `beta≈1` ⇒ constant hazard ⇒ no age to reset. Its input (ДФ-04) has no real ГТМ anyway.
  Keep `GTM_AGE_RESET_ENABLED = False`.
* **Fixed-beta shrinkage** — **worse** (+84% vs +59% at 37 events, pre-correction).
* **Equipment covariates** on Ya_brt+slb (1446 runs / 708 events, 98% Big join):
  `H3` HR 1.14 (p=0.12), `depth_km` 1.09 (p=0.49), `install_2023plus` 1.11 (p=0.25),
  **concordance 0.52**. **H3 flips sign by contractor** (Борец 0.80 protective vs
  Шлюмберже 1.59) — confounding by indication; Mc is ~all Борец ⇒ H3 is protective there
  ⇒ cannot explain Mc. Harness validated (contractor slb HR 1.551, p≈0). Replicated
  independently by the Vt session. Consistent with Phase C θ and Phase D nulls.
* **Borrowing Ya's shape for Mc** — `Ya_brt` donor under-predicts (agg 0.80). *(Computed
  pre-correction; the donor's hazard has since risen, so re-check before reusing.)*
* **Кэкспл as a hazard covariate** — null (§4).

## 10. Code added (all additive, default-off)

| file | what |
|---|---|
| `esp_population.py` | run population on the Свод clock, rebuildable "as of" any date. Keeps day-0 failures; no within-Свод dedup; `add_entry_age()`; Big open+closed with `_within_tol` dedup; sour inherited from the well's Свод history. **Guarded by `tests/test_esp_population.py`.** |
| `esp_optime.py` | **true op-day clock** from `proc__daily_operating`: true calendar denominator, explicit `coverage`, field-median-ratio imputation, `audit()`. |
| `esp_hazard_fit.py` | spike + constant-hazard MLE with right-censoring **and left truncation**; emits the shipped 5 columns. |
| `streamlit_apps/production_risk_field_plan.py` | plan review app, УН selector (Mc/Vt/Ya), A/B toggle. |
| `scripts/run/export_mc_running_pumps.py` | Свод-style export of running Mc/Mr pumps. |
| `failure_rate.compute(..., only_fields=…)` | scopes to one УН (16s → 3.3s). Default `None` = unchanged. |
| `config.GTM_AGE_RESET_ENABLED` (**False**) | ГТМ age reset, wired through `run_projection` / `project_well`. |
| `repair_compat._days_to_first_event()` | the sheet column = days to the first `0`. |

## 11. Data traps (each cost a session)

* **`esp_population` Big-only closed-run gap** — fixed; see the correction notice.
* **`proc__daily_operating` codes "no data" as `in_operation = 0`** (43% of rows,
  `op_source='missing'`), and its outer join means well-days absent from *both* sources
  have **no row at all** (median well coverage **0.759**; Ya only **0.30** because its
  history predates the 2018 telemetry start). Counting *rows* as the denominator conflates
  idle with unobserved. `esp_optime` handles this; anything else must too.
* **`crosswalk.py:644`** reads «Нспуска» into `EspRun.run_seq`, but **«Нспуска» is the
  setting depth in metres** (2623/2816 Свод rows equal `Глубина спуска УЭЦН, по НКТ`). The
  Cox layer's `log_run_seq` uses a *different, correct* source. Downstream impact untraced.
* **Match on contractor before comparing fields** — SLB fails ~1.55× faster than Борец.

## 12. Open

1. Ship B for Mc? Forecast 50 → 40. **Per-stratum only** — A wins on Ya/Global/Vt-nonsour/Za.
2. Fix the infant-component bounds — needs a **schema** change (`p0` split), not a fitter change.
3. ННО vs МРП for the sheet — МРП needs a ГТМ hazard model (ДФ-04 cannot supply one).
4. End-to-end op-day clock fix — `docs/notes/production_risk_clock_fix_plan.md`
   (predicted effect now **~10%**, not 15%: Mc's op/Наработка is 0.896, not 0.84).
5. Vt is analysed in a parallel conversation — **do not touch**.
