# Handoff Prompt — Мирнинский workover plan (option B: spike + constant hazard)

> ## ⚠️ CORRECTION 2026-07-17 — read before acting on any number below
>
> The parallel Vt session found **three population defects in `esp_population.py`**, the
> module this prompt's evidence was computed on. All are now fixed; see
> **`docs/notes/production_risk_vt_findings.md` §2 and §7**. Consequences here:
>
> * ❌ **"A over-states the plateau by 1.4–1.8×" / "A's extra failures are extrapolated
>   wear-out" — RETRACTED.** On the corrected population the shipped Weibull's plateau
>   ratio on Ya is **0.94–1.46 (median ≈ 1.0)**, i.e. calibrated. The inflation was an
>   artifact of (a) 619 missing Big-only closed Ya runs — a third of Ya's events — and
>   (b) 171 double-counted censored open runs, both of which depress the empirical
>   denominator the ratio divides by.
> * ✅ **"A gets the infant window backwards" — STANDS, and strengthens.** The shipped fit
>   delivers **0.78** of the empirical 0–30 d hazard, and the true spike is *larger* than
>   this doc knew (**0.0989**/mo for `Ya_nonsour_brt`, not 0.0666). This is now the **only
>   surviving argument for B** — and it is a real one (~58 Mc wells start at age 0).
> * ✅ **"No wear-out arm" — STANDS** (1100–1500 d hazard is still below 30–90 d).
> * 🔢 **Mc's own numbers moved:** B now forecasts **43**, not 40. B calibration
>   **1.17 / 0.92 / 1.35** vs A **1.06 / 0.86 / 1.33** — **A is not worse than B on level.**
>   The § "Why B and not A" below must be re-derived on this basis before B ships.
> * The § "Tested and FAILED" items and the § "Settled" items are unaffected **except**
>   where they rest on Ya's empirical hazard.
>
> `esp_population.build()` now takes `big_runs=` (was `open_runs=`).

You are continuing the Мирнинский (Mc + Mr) workover-plan work. A review session on
**2026-07-16** established what the model is allowed to claim and chose the fit. Your job
is to finish/ship it — **not** to re-open the questions already settled below.

Read these first, in order:

1. `docs/notes/production_risk_mc_weibull_shape_findings.md` — the full evidence for
   everything in this prompt. Numbers, tables, and the failed experiments.
2. Project memory `project_mc_weibull_shape.md` (loaded via MEMORY.md) — the compact
   version + traps.
3. `docs/notes/production_risk_joint_refit_prompt.md` — the earlier Workstream A brief
   this supersedes for Mc specifically.

Follow `CLAUDE.md` / `agents/AGENTS.md`: paths from `analysis.paths`, outputs via
`results_dir()`, reusable logic in `backend/analysis/`, scripts are thin CLI wrappers.
**Run everything with `.venv/Scripts/python.exe`** — the venv has streamlit **1.48.1**
(`width='stretch'` is NOT supported; use `use_container_width`). The system Python has
1.54 and will silently mislead you.

---

## What was built (2026-07-16)

**Review app — the deliverable to look at first:**

```
.venv/Scripts/python.exe -m streamlit run streamlit_apps/production_risk_field_plan.py
```

> **Renamed 2026-07-16** from `production_risk_mc_plan.py`: the Vt session generalised it
> to a **УН selector** (Мирнинский / Верхнетирский / Ярактинский). Mc is the default and
> its numbers are unchanged — 83 wells / 40 workovers / 2.2 per month, verified via
> `AppTest`. Option B now fits **every stratum of the selected field** (Mc 1, Vt 8, Ya 4);
> for Mc that is the same single-stratum fit as before.

`streamlit_apps/production_risk_field_plan.py` — plan review, 4 tabs:

1. **Сводка** — 83 wells, **40 workovers** in horizon, **2.2/month**, calibration by year.
2. **Интенсивность отказов** — monthly fact vs model, counts + per-well rate, forecast
   boundary marked.
3. **План ремонтов** — per-well `Дней до ремонта` (= distance to the first `0` in the
   row) + its distribution.
4. **Приоритет** — ranked by **risk = expected failures × oil rate**.

An **A/B toggle** in the sidebar switches between the shipped fit and option B, so the
choice is reviewable rather than asserted. It injects params at runtime; **`esp_models.csv`
is untouched and still carries A.**

**Supporting modules (new, additive, nothing shipped depends on them yet):**

| file | what |
|---|---|
| `backend/analysis/workflows/production_risk/esp_population.py` | ESP run population on the Свод clock, rebuildable "as of" any historical date. Keeps day-0 failures; no within-Свод dedup; `add_entry_age()` for left truncation; open runs inherit sour class from the well's Свод history. Reproduces Свод exactly (Mc: 133 runs / 49 events). |
| `backend/analysis/workflows/production_risk/esp_hazard_fit.py` | spike + constant-hazard MLE with right-censoring **and left truncation**. Emits the shipped 5 registry columns. |
| `streamlit_apps/production_risk_survival_tuner.py` | generic per-stratum tuner: edit params → KM/Weibull overlay + fact-vs-model rate. |
| `scripts/run/export_mc_running_pumps.py` | Свод-style export of running Mc/Mr pumps. |

**Changes to existing code (all additive / default-off):**

* `failure_rate.compute(..., only_fields=…)` — scopes the whole computation to one УН
  (16s → 3.3s). `well_field` is the single point everything keys off. Default `None` =
  unchanged.
* `config.GTM_AGE_RESET_ENABLED` (**default False**) + `layers.run_projection(gtm_age_reset=…)`
  + `survival.project_well(gtm_reset_months=…)` — ГТМ pump-age reset. **Do not enable**
  (see "tested and failed").
* `repair_compat._days_to_first_event()` — the per-well sheet column now carries **days to
  the first `0`** (remaining life from the forecast start), not an ННО interval. Verified
  against all 766 rows, zero discrepancies. Header `Вероятностный прогноз` and JSON keys
  kept (integration contract); `days_to_workover` added as an explicit alias and the old
  interval preserved as `mean_op_days_per_failure` (still feeds the monthly chart).
* `repair_compat._predicted_interval()` — rounds to whole days.

31 tests pass (`backend/tests/test_production_risk.py`). **Nothing shipped changed.**

---

## The model — option B (chosen)

```
S(t) = w1·exp(-(t/eta1)^beta1)  +  (1 - w1)·exp(-t/eta2)
       \___ infant component ___/    \__ constant hazard __/     beta2 pinned to 1
```

Fitted on **Mc's own runs, left-truncated to the 2024+ regime** (180 runs / 45 events):

| w1 | beta1 | eta1 | beta2 | eta2 | B50 |
|---|---|---|---|---|---|
| 0.0235 | 10.0 | 0.5 | 1.0 | 922.97 | 618 |

Reads as **2.4% infant mortality at day 0, then a flat ~3.2%/month**. Uses the shipped
five registry columns — `StrataModel.S` evaluates it natively, nothing downstream changes.

**Calibration (model/факт):** 2024 **1.05**, 2025 **0.83**, 2026H1 **1.23**; aggregate
2024–26H1 **0.97**. Forecast **40** workovers vs the shipped **50**.

### Why B and not A (the shipped single Weibull, b50=503)

The empirical hazard is **infant spike (~30 d, 2.4–3.6×) then FLAT to 1500 d — no wear-out
arm** (Ya_brt, 440 events: 0.0666/mo at 0–30, then 0.023–0.028 all the way out; at
1100–1500 it is 0.0172, *lower* than at 30–90, on 132 pumps at risk and 22 real events).
A single Weibull cannot express "spike then flat", so its MLE compromises at `beta<1` — a
curve that decays forever. Consequences:

* **A gets the infant window backwards.** At `beta=1.252` it puts P(fail)/month at
  0.044 for age 0 and 0.044 at age 300 (no discrimination); B gives **0.055 vs 0.032**.
  The empirical ratio is ~2.3× *worse* when new. **~58 Мирнинский wells start at age 0
  during the horizon** and must pass through that window.
* **A's extra 10 failures are extrapolated wear-out** into ages where Mc has **zero
  observed failures** (Mc's last failure is at 663 d).
* On Mc's own age-banded hazard, B is far better: spread **1.67 → 0.59**, infant ratio
  **0.43 → 1.07**.

Aggregate calibration cannot separate them (1.01 vs 0.97) — the difference is *shape*,
which is why it shows up in the forecast (50 vs 40) and in the priority list.

---

## Settled — do not re-derive

> **⚠ Corrected 2026-07-17.** The 2026-07-16 version of this prompt was written against an
> `esp_population` that omitted Big-only **closed** runs (Ya: 784 of 1226 events). Fixed by
> the Vt session, guarded by `tests/test_esp_population.py`. **Three claims died** — the
> corrected versions are below. **Always assert fit `n_failures` against `esp_models.csv`
> before trusting a population.**

**Мирнинский is fine — if anything better than Ya.** Age-standardised vs the
**like-for-like** donor `Ya_brt` (Mc is 166/168 Борец): **SMR 0.82** (47 observed vs 57.6
expected, p=0.088) — Mc runs ~18% **cooler**. *Always match on contractor before comparing
fields* (SLB fails ~1.55× faster). ~~SMR 1.23~~ and ~~0.97~~ were both artefacts — of the
broken population and of a pooled donor respectively. The **pre-2023 era** is the anomaly
(2 failures vs ~12 expected, on 21 runs) — luck or incomplete early records, not a regime
change. Big holds **no hidden pre-2023 failures** (its rows with fail dates are
`Нагнетательная` injection wells, correctly excluded by `is_esp_strict`).

**B50 is noisy but UNBIASED at Mc's volume — and still the wrong target.** Bootstrap from
the corrected `Ya_nonsour_brt` (truth 479): at 197 runs the median is **494** — unbiased —
with p10–p90 411–782 and 16% of draws off by >25%. ~~"Unidentifiable below 150 events,
+59% at 37 events"~~ was the **missing-data bias**, not small-sample bias; the truncation
design cannot even reach Mc-like statistics on the corrected data (Ya_brt has 420 events in
2019). Ya's truths moved: **brt 844 → 479**, **slb 476 → 368**. Still true: the forecast is
insensitive to b50 (the fleet occupies ages 0–1200 d where curves barely differ), so
**target the hazard over the occupied age range**. `beta` is stable (0.71–0.80). The
residual **0.83 on 2025 is small-sample noise**, not a clock artefact (the history replay
is already self-consistent — `uptime_factor=1.0` ⇒ calendar × calendar).

**A per-well DATE is not predictable.** Constant hazard ⇒ memoryless ⇒ remaining life is
independent of age. How `repair_compat` places a `0`:

1. `E_w` = expected failures for the well over the horizon.
2. Fleet total `N = round(Σ E_w)` (= 50 shipped / 40 under B).
3. Each well gets `floor(E_w)` — **0 for every Mc well** (all `E < 1`, mean 0.606, max 0.839).
4. The `N` remaining go to the wells with the largest fractional part → **top-N by E**.
5. **When**: inverse-CDF — event *k* lands where the well's cumulative expected-failure
   curve crosses `(k − u)·E_total/n`, with `u` a **hash of the well code**.
6. `_month_spread_date` picks the day via another stable hash.

So **which wells** = top-N by E (near-arbitrary: CV 0.27), **when** = a seeded
pseudo-random quantile ⇒ ≈ uniform over the horizon (median days-to-workover **244** vs
horizon midpoint 274). The hash makes it *reproducible*, not *predictive*.

**Prioritise by RISK, not probability.** Probability is near-flat across wells (CV 0.27);
**oil rate spans ~79×** (4.4 → 343 т/сут). `risk = E × oil` gives a 60× spread, and the
top-10 by risk overlaps the top-10 by probability on only **2 of 10** wells.

**Estimand / naming.** `Вероятностный прогноз` was an **ННО** (failure interval), not an
**МРП** (межремонтный период, all-cause ≈ 294 d for Mc). It now carries days-to-workover.
Each `install → pull` is **one survival unit** — ages are **not** concatenated across pulls
(correct: `pump_serial` shows the same pump returns after ГТМ only **4.0%**, 35/872
fleet-wide). Censoring ГТМ is **right** for a failure forecast (it yields the cause-specific
hazard); what was wrong was reading the resulting *survival* as real.

**`ДФ-04` contains no workovers.** Its 39 Мирнинский horizon entries are **30 ВНС** +
**9 Перевод в ППД**; 22 of 39 are `Нагнетательная` and 35 of 39 are not running. There is
**no forward source** for the ГТМ that drive real pump changeouts, so `Прогноз_ремонтов`
can honestly forecast **failures only**.

## Tested and FAILED — do not retry

* **ГТМ pump-age reset in the projection** — implemented, effect **−2%** (50.3 → 49.2).
  `beta≈1` means the hazard is constant, so there is no age to reset. Its input (ДФ-04)
  has no real ГТМ anyway. Keep `GTM_AGE_RESET_ENABLED = False`.
* **Fixed-beta shrinkage** (pin beta to the fleet value, fit scale only) — **worse**
  (+84% vs +59% at 37 events); the global beta (0.668) is flatter than each stratum's and
  b50 is hypersensitive to it.
* **Equipment covariates** on Ya_brt+slb (1446 runs, 708 events, 98% Big join):
  `H3` HR 1.14 (p=0.12), `depth_km` HR 1.09 (p=0.49), `install_2023plus` HR 1.11 (p=0.25),
  **concordance 0.52**. **H3 flips sign by contractor** — Борец **0.80** (protective,
  p=0.066) vs Шлюмберже **1.59** (p=0.0002) — confounding by indication; not causal, not
  transferable. Mc is ~all Борец ⇒ H3 is *protective* there ⇒ **cannot explain Mc's rise**.
  Harness validated (contractor slb HR 1.551, p≈0). Concordance never exceeds 0.56 ⇒ no
  per-well discrimination is achievable. Consistent with Phase C θ and Phase D nulls.
* **Borrowing Ya's shape for Mc** — `Ya_brt` donor **under-predicts 20%** (agg 0.80).
  Ya *pooled* scores 1.00 only because its SLB share offsets a contractor mix Mc does not
  have: right answer, wrong reason. **Mc's own 49-event fit beats both.**

## Known weaknesses of option B (state these when presenting)

* **`beta1=10, eta1=0.5` sit on their bounds** — the infant component collapses to a point
  mass at day 0.5 instead of finding a real window (Ya_brt found a genuine 27-d one). The
  spike *magnitude* is therefore partly set by the constraint, not by Mc's data. **This is
  the weakest part of B.**
* Mc `max|ΔS|` vs KM = **0.080**, still failing the 0.05 acceptance bar (A is 0.107).
* B's *level* carries the ±20% small-sample uncertainty (49 events). So does A.

---

## Open items / suggested next steps

1. **Decide whether to ship B** into a candidate bundle. Forecast changes 50 → 40 for
   Мирнинский. Do **not** blanket-replace the registry: the per-stratum audit says
   spike+constant **wins** on Ya/Ic/Mc, **ties** Vt/Az, and **LOSES** on Za
   (beta=0.962 — already effectively constant-hazard, median 1.05 / spread 0.35) and
   **Global** (spread 0.64 → 1.17; a pooled heterogeneous fleet is not one spike+plateau).
2. **Fix the infant-component bounds** so the spike is data-driven rather than constrained.
3. **Decide the sheet's estimand**: ННО (failures only, current) vs МРП (all causes) — the
   latter needs a ГТМ hazard model, since ДФ-04 cannot supply one.
4. **Separate bug, still open:** `crosswalk.py:644` reads «Нспуска» into `EspRun.run_seq`,
   but **«Нспуска» is the setting depth in metres** (2623 of 2816 Свод rows equal
   `Глубина спуска УЭЦН, по НКТ`). The Cox layer's `log_run_seq` uses a *different, correct*
   source (reference 1.245 → seq ≈ 3.5), so the Cox is fine. Downstream impact of the
   crosswalk read is untraced.
5. **Vt is being analysed in a parallel conversation — do not touch it.**

## Verification bar for any change here

Judge a fit by **median ratio** (level bias) **and spread** (shape error) against that
stratum's **own empirical age-banded hazard** — the check the registry has never had.
Median alone misleads: Ic's old fit had median 1.10 (looks unbiased) with spread 1.80
(0.50→2.30 — wrong everywhere, errors merely straddle 1.0). Note the ≥5-event band filter
changes the band count, so low-event strata (Mc, Da) are the least trustworthy.
