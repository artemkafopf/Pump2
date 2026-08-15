# CatBoost, re-done as a hazard model — assessment and Ya→Mc verdict

Дата: 2026-07-17. Session goal: assess the original CatBoost and the way it produced
`Прогноз_ремонтов`, then build an alternative CatBoost around the strongest covariates,
replicating TTF/failure rates against the ТМ-06 plan. Focus: **Ya** as the clean data
source, **Mc** as the target.

**Nothing shipped changed.** New code is additive and default-off. Bundle, calibration
factors, VBA and EXE untouched.

> **Depends on the corrected population.** This work post-dates the 2026-07-17
> `esp_population` fix. `production_risk_mc_weibull_shape_findings.md` is **stale** where
> it rests on Ya's empirical hazard — see the CORRECTION block atop
> `production_risk_mc_plan_prompt.md`. In particular "A over-states the plateau 1.4–1.8×"
> is retracted, and spike+constant (B) does **not** out-calibrate A on Mc.

---

## Verdict (read this first)

1. **The original CatBoost has no concept of censoring**, and its published number is,
   for most pumps, not a model output at all but a random draw from a histogram (§1).
2. **The `5 strongest covariates` premise does not survive contact with the data.** The
   top two by importance are a **field label in disguise**: `pbubble_atm` has **0% support
   overlap** between Ya and Mc, and `nominal_freq_hz` is **constant within Mc**. Fitting
   them and predicting Mc over-states failures **8.6×** (§4).
3. **No covariate transports — including Kpod.** The transferable pair explains ~2% of
   Mc's excess (§5). **Kpod/Ql looked like a real result on Mc alone (0.927 vs 0.566
   age-only) and failed replication**: across 10 target strata the loading block inflates
   the hazard rather than correcting it (spread 0.27–4.52; 19.9× on Da). A single target
   cannot tell signal from a lucky landing (§5b).
4. **Covariate availability is confounded with hazard in opposite directions per field**
   (Ya missing = 2.01× hot; Mc missing = 0.51× cold). There is **no subset that is both
   trap-free and representative**, so the Ya→X transfer question is **not answerable on
   this data** whatever covariate is tried. A data limitation, not a modelling choice (§2).
5. **The age-only hazard model beats both incumbents and is the deliverable.** It
   reproduces the 0–30 d infant spike in **19/19 strata under well-clustered cross-fit**
   (median 0.999, range 0.974–1.053), against **3/19** for the shipped Weibull (A) and
   13/19 for spike+constant (B) — without a level-error trade (§7). Needing no covariates,
   it sidesteps the §2 trap entirely.
6. **`Mc_nonsour_brt` is the shipped model's worst stratum on both counts at once**:
   `infant_A` = **0.365** (a third of the real infant hazard) while `overall_A` = **1.299**
   (30% over on total). ~58 Mc wells start at age 0, so this is where Mc's forecast lives.
7. A per-well workover date schedule remains unachievable; nothing here changes that.

---

## 1. The original CatBoost (`backend/app/services/repair_forecast.py`)

This — not `models/ml/regression_ttf.py` — is what produced the shipped
`Прогноз_ремонтов` sheet. Four structural defects, worst first:

* **No censoring.** `app/services/analysis.py:595 train_forecast_model` is a plain
  `CatBoostRegressor(loss_function="RMSE")` on whatever column the user picked as target
  (the ННО column of Факт ЭПУ), with a **random shuffle split**. Running pumps either sit
  out of training or enter it with their current age treated as a completed lifetime.
  Both bias the fit **down** — this is the direct source of «500 days median is very
  wrong».
* **The tail-resampling hack is the model, most of the time.** Because the fit is
  censoring-blind, `t_pred` routinely lands **below the pump's current age** `t_fact` —
  inevitable for survivors, who *are* the long lifetimes the model never learned. When it
  does, `postprocess_runtime_prediction` (line 423) discards the CatBoost prediction and
  draws a **random sample from the right tail of the empirical ННО histogram**
  conditioned on `> t_fact`. So for every long-lived pump the «Прогноз CatBoost» number
  is a random draw. It is an ad-hoc stand-in for conditional survival `S(t | T > age)` —
  but sampled from an **unconditional, censoring-ignoring** distribution, and when a
  group is thin `_align_values_to_fact` (line 97) standardises the *global* values onto
  the group's mean/std, which has no probabilistic meaning at all.
* **Recurrent failures are a deterministic renewal at the unconditional mean.**
  `build_statuses_for_well` (line 1028) places the first failure at `t_pred − t_fact`
  days, then repeatedly adds `catboost_nno`. No hazard, no aging, no uncertainty.
  `calculate_first_failure_offset` then arbitrates between the two predictions with a
  branch cascade whose cases overlap.
* **Everything binds by fuzzy Russian substring match.** `find_best_column` scores columns
  by pattern-substring length; `("наработ",)` / `("нно",)` silently pick whichever column
  scores highest.

Minor, but worth deleting: dead mojibake. `category == "Р‘Р°Р·Р°"` (lines 1275, 1360) can
never be true (`category` is `"БАЗА"`/`"ВНС"`), so the first `build_statuses_for_well`
call is computed and thrown away.

### What `Прогноз_ремонтов` is today

`repair_compat.py` replaced the above with the survival line and **left the CatBoost
column empty** (`"catboost_nno": None`, line 246). The `Факт ННО` > `Вероятностный
прогноз` oddity is not a bug but two different quantities sharing a table:
`probabilistic_nno` is **days to the first predicted event** (remaining life from the
forecast start); `Факт ННО` is the **current pump's age**. They are not comparable and
neither can contradict the other.

## 2. The covariate-availability trap

`build_regression_frame` is built on the **warehouse** population and joins to Свод runs
only — **no Big-only run carries covariates**. Big-only runs are exactly what the
2026-07-17 population fix added, and their meaning is field-specific:

| stratum | covariates | runs | events | fail/mo |
|---|---|---|---|---|
| Ya_brt | present | 613 | 323 | **0.0288** |
| Ya_brt | **missing** | 845 | 459 | **0.0578** (2.01× hotter) |
| Mc_brt | present | 49 | 21 | **0.0480** |
| Mc_brt | **missing** | 144 | 29 | **0.0245** (0.51×, colder) |

Ya's Big-only runs are **closed** runs carrying failures Свод never recorded; Mc's are
**open**, still-turning pumps. CatBoost treats NaN as a split direction, so a model that
sees the missing rows learns *"missing ⇒ hot"* on Ya and applies it to an Mc whose missing
rows are cold — **the sign inverts**. Hence: fit and score on covariate-present runs only.

**But the restriction is not free.** Availability is itself confounded with hazard, so the
covariate-bearing subset represents neither field: on it **Mc/Ya = 1.66**, against the
population-level age-standardised **SMR 1.23**. Selecting on availability *amplifies* the
very gap the covariates are asked to explain. **There is no subset that is both trap-free
and representative.** Every number in §4–§5 inherits this.

## 3. Why a hazard model, not TTF quantiles

`models/ml/regression_ttf.py` predicts TTF quantiles and bridges them to survival. Two
structural problems follow from that choice alone:

* **IPCW deletes every censored run.** `ipcw_weights` assigns weight 0 to censored rows
  and `fit()` drops them — Mc trains on its **43 events**, not its 193 runs.
* **It predicts the wrong object.** The hazard is spike-then-flat with **no wear-out arm**,
  so TTF is near-exponential: b50/b90 are pure scale with no shape to learn, and the
  invented post-b90 exponential tail then covers **23%** of evaluations.

The alternative (`hazard_catboost.py`) models the rate directly,
`λ(age, x) = exp(f(log_age, x))` in failures per operating day, by **Poisson regression
over person-period age bins with `log(exposure)` as offset**. Consequences:

* censored runs contribute their at-risk bins natively — no IPCW, nothing discarded;
* **age is a feature**, so the infant spike is expressed non-parametrically. That is the
  one surviving form defect of the shipped Weibull (it delivers 0.78 of the empirical
  0–30 d hazard) and the one a 5-column parametric schema provably cannot hold;
* it predicts the monthly failure probability the failure-rate line already scores, so it
  plugs into the injected `p_fail_fn` with **no survival-curve bridge**;
* left truncation is free (bins simply start at `entry`).

Verified on synthetic ground truth (`tests/test_hazard_catboost.py`): recovers a constant
hazard to 30%, recovers a **10× infant spike**, recovers a covariate effect, is memoryless
under constant hazard, and its `p_fail` matches the exponential closed form.

*Implementation gotcha:* CatBoost's Poisson `predict` defaults to
`prediction_type="Exponent"` and already returns the rate — exponentiating it again
squares the hazard. `rate()` asks for `RawFormulaVal` explicitly.

## 4. The «5 strongest covariates» premise fails the support test

The importance ranking that motivates the premise (2026-07-13 mean-TTF audit:
`nominal_freq_hz` 18.5, `pbubble_atm` 17.3, `freq_std_early` 5.9, `log_glf_mean_opdays`
5.4, `field` 4.6) is measured **within the pooled fleet**, where a covariate that merely
labels the field scores highly. `support_overlap(Ya_brt, Mc_brt)`:

| covariate | train range (Ya) | test range (Mc) | share in train range | constant in test | transferable |
|---|---|---|---|---|---|
| `nominal_freq_hz` | 0 – 220 | 50 – 50 | 1.00 | **yes** | **no** |
| `pbubble_atm` | 230.0 – 259.5 | 0 – 158 | **0.00** | no | **no** |
| `freq_std_early` | 0 – 14.28 | 0 – 9.81 | 1.00 | no | yes |
| `log_glf_mean_opdays` | 1.77 – 8.94 | 2.86 – 8.02 | 1.00 | no | yes |

* **`pbubble_atm` is a field label in disguise.** Not one Mc run falls inside Ya's
  observed range. This is *why* it ranked #2 — and why `field` itself ranked only 4.6:
  pbubble had already absorbed it.
* **`nominal_freq_hz` has no within-Mc variance** (all 99 runs at 50.0 Hz), so it cannot
  discriminate anything there. Ya's range also contains non-physical values (0 and 220 Hz)
  — a data-quality flag worth raising separately.
* **`field` is unusable by construction**: single-valued in a single-field fit, an unseen
  level at transfer.

**High importance means "separates this fleet's runs", not "transports to another field".**
`support_overlap` is now a reusable guard; run it before any cross-field covariate fit.

## 5. Ya → Mc: do the covariates explain Mc's excess?

Fitted on `Ya_brt` (613 runs / 323 events, covariate-present), scored as expected failures
over **Mc's own exposure** (49 runs / 21 events observed):

| model | trained on | Mc expected | **Mc ratio** | Ya ratio (sanity) |
|---|---|---|---|---|
| `age_only` (= donor shape-borrowing) | Ya_brt | 12.0 | **0.571** | 0.997 |
| `covariates` (support-guarded) | Ya_brt | 11.8 | **0.560** | 0.991 |
| `covariates_naive` (all 4, guard ignored) | Ya_brt | 180.8 | **8.611** | 0.990 |
| `mc_own` (in-sample upper bound) | Mc_brt | 21.0 | 0.998 | 23.513 |
| `spike_constant_B` (in-sample) | Mc_brt | 21.2 | 1.009 | — |

* **The covariates explain ~2% of the gap** — 0.560 vs 0.571 age-only. Mc is 1.66× hotter
  than Ya on this subset and the transferable covariates move the needle from 0.571 to
  0.560, i.e. essentially nowhere. **Honest null**, triangulating the linear θ C5 gate,
  Phase-D dynamics, and the TTF-CatBoost verdict by a fourth, orthogonal route.
* **Ignoring the support guard costs a factor of ~15** (8.611 vs 0.560). This is the
  concrete cost of taking an importance ranking at face value across fields.
* **Sanity holds**: every Ya-trained model calibrates to 0.99 in-sample on Ya, and both
  Mc-fitted models land at ~1.00 on Mc. The `mc_own → Ya` ratio of **23.5** is the same
  extrapolation blow-up in reverse — the failure is symmetric, and it is about support,
  not about which field is "better".

## 5b. Kpod and Ql — checked, and rejected on replication

Kpod was the best remaining candidate and for a good reason: it is **dimensionless**
(`qliq / nominal_flow_m3d`), so unlike `pbubble_atm` it is comparable across fields by
construction — and indeed the whole loading block clears `support_overlap` on Ya→Mc.
All of it is early-window (`run_covariates` aggregates under `oprank <= n_days`), so
despite `kpod_mean`'s whole-run-sounding name **nothing here leaks**. Requiring the block
costs almost nothing (Ya 613→603 runs, Mc 49→49), so every model below is scored on one
fixed population.

**On Mc alone it looked like a real result:**

| model (trained Ya_brt) | Mc expected (obs. 21) | ratio |
|---|---|---|
| `age_only` | 11.9 | 0.566 |
| `base` (freq_std + log_glf) | 13.2 | 0.627 |
| `base+kpod_mean` | 30.6 | 1.456 |
| `base+qliq_early_m3d` | 23.8 | 1.133 |
| **`base+all_loading`** | 19.5 | **0.927** |

**It does not replicate.** Scored against every stratum with ≥8 events, the loading block
does not move ratios *toward* 1.0 — it inflates the hazard generally and lands near 1.0 on
Mc by luck:

| target | obs | age_only | base | base+kpod_mean | base+all_loading |
|---|---|---|---|---|---|
| Mc_brt | 21 | 0.566 | 0.627 | 1.456 | **0.927** |
| Vt_brt | 40 | 0.311 | 0.290 | 0.316 | 0.325 |
| Vt_slb | 44 | 0.239 | 0.373 | 0.323 | 0.528 |
| Az_brt | 41 | 0.394 | 0.539 | 0.472 | 0.478 |
| Az_slb | 36 | 0.358 | 0.751 | 0.742 | 1.155 |
| Za_brt | 43 | 0.245 | 0.322 | 0.274 | 0.267 |
| Za_slb | 17 | 0.408 | 0.453 | 0.458 | 0.870 |
| Ic_brt | 50 | 0.628 | 1.565 | 1.472 | 2.151 |
| Ic_slb | 38 | 0.546 | 1.347 | 1.741 | **3.818** |
| Da_brt | 21 | 1.732 | 2.485 | **19.942** | 4.517 |

Summary (`mean_abs_log_ratio` = scale-symmetric error; 0.72 ⇒ a typical miss of ~2.05×):

| model | median ratio | mean_abs_log_ratio | min | max |
|---|---|---|---|---|
| `covariates` (base) | 0.583 | **0.718** | 0.290 | 2.485 |
| `base+qliq_early_m3d` | 0.770 | 0.750 | 0.275 | 3.663 |
| `base+all_loading` | 0.899 | 0.779 | 0.267 | 4.517 |
| `age_only` | 0.401 | 0.905 | 0.239 | 1.732 |
| `base+kpod_mean` | 0.607 | 0.972 | 0.274 | **19.942** |

**No model transfers** — every one is off by ~2× typically, and the loading block buys a
better *median* (0.899) purely by rescaling while its *spread widens* (0.267–4.517).
`base+kpod_mean` blows up 19.9× on Da. A covariate carrying real physics moves every
target toward 1.0; one that rescales lands somewhere by luck and scatters elsewhere.
**A single target cannot distinguish the two — always replicate.**

Note also `age_only`'s median of **0.401**: a Ya-trained model under-predicts *every* other
stratum by ~2.5×, because Ya's covariate-present subset is its **cold half** (§2). That is
the selection bias, visible directly. **The Ya→X transfer question is not answerable on
this data at all**, independent of which covariate is tried.

## 6. Tested and failed (do not retry)

* **Feeding runs with missing covariates to CatBoost and letting NaN-handling sort it
  out.** The NaN split direction encodes a field-specific association that inverts on
  transfer (§2). Not fixable with a `_missing` flag — the flag *is* the leak.
* **Using the importance-ranked top covariates for cross-field transfer** without a
  support check (§4): 8.6× over-prediction.
* **`field` as a transfer covariate**: degenerate in train, unseen in test.
* **Kpod / Ql as transfer covariates** (§5b) — the best-motivated candidates left
  (dimensionless, early-window, full support overlap), and they fail on replication.
  Do not re-run on a single target and do not read Mc's 0.927 as a result.

## 7. Age-only hazard — the one that works

No covariates ⇒ **no covariate-availability selection** ⇒ the §2 trap is structurally
absent and this runs on the **full corrected population** (4,310 runs, 19 strata with ≥20
events). It targets the only defect that survived the population fix: the infant spike.

Three fits per stratum on identical data: **A** = the shipped registry row (resolved as
production resolves it), **B** = `esp_hazard_fit` spike+constant refitted, **cb** =
age-only Poisson hazard, reported both in-sample (can the form express the shape at all?)
and **well-clustered cross-fit** (does it generalise, or is it fitting band noise?).

### 0–30 d infant band — model / empirical hazard (1.0 = right)

| stratum | events | A | B | cb in-sample | **cb x-fit** |
|---|---|---|---|---|---|
| Ya_nonsour_brt | 781 | 0.765 | 1.001 | 0.999 | **0.999** |
| Ya_nonsour_slb | 382 | 0.594 | 0.698 | 1.000 | **0.996** |
| Mc_nonsour_brt | 50 | **0.365** | 1.084 | 1.001 | **0.989** |
| Vt_nonsour_slb | 101 | 0.610 | 0.524 | 0.998 | **0.998** |
| Ic_nonsour_brt | 90 | 0.494 | 0.547 | 0.998 | **0.998** |
| Ic_nonsour_slb | 64 | 0.494 | 0.492 | 0.999 | **1.003** |
| Da_nonsour_brt | 48 | 0.451 | 0.529 | 1.001 | **0.989** |
| Az_nonsour_brt | 91 | 0.735 | 0.983 | 1.002 | **0.986** |
| *(19 strata total — full table in `age_only_verdict.csv`)* | | | | | |

| metric | median | mean_abs_log | min | max | **strata within ±20%** |
|---|---|---|---|---|---|
| `infant_A` | 0.735 | 0.428 | 0.365 | 1.068 | **3 / 19** |
| `infant_B` | 0.906 | 0.215 | 0.492 | 1.084 | **13 / 19** |
| `infant_cb_insample` | 1.000 | 0.001 | 0.998 | 1.003 | **19 / 19** |
| **`infant_cb_xfit`** | **0.999** | **0.014** | 0.974 | 1.053 | **19 / 19** |

| metric | median | min | max | strata within ±20% |
|---|---|---|---|---|
| `overall_A` | 0.970 | 0.601 | **1.299** | 17 / 19 |
| `overall_B` | 1.000 | 0.994 | 1.001 | 19 / 19 |
| **`overall_cb_xfit`** | 1.013 | 0.972 | 1.120 | **19 / 19** |

**Verdict: the age-only hazard model reproduces the infant spike in all 19 strata, and it
holds out of sample** (cross-fit 0.974–1.053) — so it is learning the shape, not band
noise. A gets the infant band right in **3 of 19**; B in 13 of 19. Neither buys it with a
level error: cb's overall calibration is 0.972–1.120 across every stratum.

This is the first thing in this work that **beats both incumbents**, and it is exactly the
defect §2 of the Mc doc predicted a 5-column parametric schema could never fix: the schema
can hold a day-0 point mass *or* an infant window, not both. A model with age as a feature
has no such constraint, and needs no covariates to get it.

Two specifics worth flagging:

* **`Mc_nonsour_brt`: `infant_A` = 0.365** — the shipped Weibull delivers barely a third of
  Mc's infant hazard, its worst stratum. And **`overall_A` = 1.299**: it over-predicts Mc's
  total by 30% while under-delivering its spike. Both errors at once, in the УН that
  started this. ~58 Mc wells start at age 0, so the infant band is where Mc's forecast
  actually lives.
* **`Az_nonsour_oth`: `overall_A` = 0.601** — a second stratum where the shipped registry
  is materially off on level, not just shape.

## 8. Open / next

* **The age-only hazard line now has a verdict that justifies shipping it** (§7) — but it
  is not shipped. The blocker is the **5-column registry schema**: `w1/beta1/eta1/beta2/
  eta2` cannot carry a non-parametric hazard, which is the whole point of it. Wiring it in
  means either an injected `p_fail_fn` on the Python side only (the EXE/VBA keep A), or a
  schema change. **That is a decision, not a task** — worth taking deliberately.
* If a schema change is off the table, the cheap partial win is to **refit A's `w1/eta1`
  against the measured infant band per stratum** rather than letting the MLE compromise.
  B already recovers 13/19 that way; the remaining 6 are where the day-0 point mass and the
  infant window compete for the same two parameters.
* **`nominal_freq_hz ∈ {0, 220}` on Ya is not physical** (220 is a voltage). It is the
  #1-importance covariate in the shipped TTF audit — worth a data audit regardless of
  whether any covariate model ever ships.
* `overall_A` = **0.601** on `Az_nonsour_oth` — a second stratum where the shipped registry
  is materially off on *level*. Not investigated here.
* The **original CatBoost** (§1) is still live code behind `app/services/repair_forecast.py`.
  If any path still calls it, its output is not a forecast (see §1) and should be retired
  rather than repaired.

---

## Code added this session (all additive, default-off)

| file | what |
|---|---|
| `backend/analysis/workflows/production_risk/hazard_catboost.py` | Poisson discrete-time hazard over person-period bins (censoring-native, left-truncation-capable, infant-spike-capable); `build_covariate_frame`, `attach_covariates`, `coverage_report`, `support_overlap` guard, `empirical_hazard`. |
| `backend/tests/test_hazard_catboost.py` | 19 tests: synthetic constant-hazard/spike recovery, covariate recovery, memorylessness, left truncation, day-0 failures, support-guard cases. |
| `scripts/run/hazard_catboost_ya_to_mc.py` | The Ya→Mc transfer experiment + the **replication table** across all target strata; writes `coverage.csv`, `support_overlap.csv`, `verdict_transfer.csv`, `replication_by_target.csv`, `replication_summary.csv`. |
| `scripts/run/hazard_age_only.py` | **The age-only verdict** (§7): A vs B vs cb in-sample vs cb cross-fit, per stratum, on the full corrected population. Writes `age_only_verdict.csv`, `age_only_summary.csv`, `hazard_by_band.csv`. |

Reproduce:

```bash
.venv/Scripts/python.exe scripts/run/hazard_catboost_ya_to_mc.py   # covariate nulls (§4, §5, §5b)
.venv/Scripts/python.exe scripts/run/hazard_age_only.py            # the age-only verdict (§7)
```

Outputs: `results/production_risk_hazard_catboost_ya_to_mc/<date>/tables/`,
`results/production_risk_hazard_age_only/<date>/tables/`
