# Plan — End-to-end op-day clock fix (fit ↔ age ↔ replay ↔ forecast)

Дата: 2026-07-17. Scoping document — **no code written yet**. Companion to
`docs/notes/production_risk_mc_weibull_shape_findings.md` and
`docs/notes/production_risk_vt_findings.md`.

---

## 1. The problem: four clocks in one chain, and they disagree

| stage | what it measures time in | source |
|---|---|---|
| **Fit** | Свод «Наработка (сут)» ≈ **calendar** | `esp_population` / the refit builder |
| **Current age** (`age_op`) | Свод «Наработка» or ТР «Время наработки (ННО)» ≈ **calendar** | `crosswalk.py:647,564` |
| **History replay** | `op_days_month = overlap × uptime_factor` → with `uptime_factor = 1.0` fleet-wide this is **calendar** | `failure_rate.py:681` |
| **Forward forecast** | `planned_op_days` from the ПП plan = **true op-days** | `layers.project_well` |

**Measured, this session:**

* `Наработка` is **not** operating time. On well-observed runs, true op-days / `Наработка`
  = **0.84 (Mc)**, **0.91 (Ya)** at the median (p10 ≈ 0.16 for Mc — a long tail).
* λ per op-day vs per `Наработка`-day: **1.37× (Mc)**, **1.28× (Ya)**.
* Plan Кэкспл (`op_days/cal_days`) = **0.846 (Mc)** / **0.911 (Ya)** — matches realized
  telemetry Кэкспл (0.864 / 0.908) and matches the op/`Наработка` ratio. **The plan's
  `op_days` is already a correct op-day exposure.**
* `uptime_factor = 1.0` for **all 9 Pooled strata** in `2026-07-15-mc2023plus`.

## 2. The key insight: the COUNT is invariant if fit and exposure are PAIRED

Under a constant hazard (which is what the data shows — see §3):

```
failures = λ_op × op_days  ≡  λ_cal × cal_days        (identically, since λ_op·Кэкспл = λ_cal)
```

So the clock choice does **not** change the fleet count — *provided the λ and the exposure
use the same clock*. What is broken is the **pairing**:

```
current forward path:  λ_cal  ×  op_days   =  λ_cal × 0.85 × cal  =  0.85 × truth
                       ^fit on Наработка      ^plan op_days
```

⇒ **the forward forecast under-states by ~15% (Mc) / ~9% (Ya).**

The history replay is *internally consistent* (calendar fit × calendar exposure, because
`uptime_factor = 1.0`), which is why its ratios (Mc 2024 **1.05** / 2025 **0.83**) are
**not** clock artifacts — 0.83 remains the measured small-sample bias.

**So this is a forward-forecast bug, not a calibration bug.** That is a narrower and more
tractable claim than "the model is on the wrong clock".

## 3. Why op-days is nonetheless the RIGHT clock (not just a pairing choice)

Landmark test, leakage-proof (Кэкспл over days 0–90 → failures **after** day 90), Ya,
493 runs / 221 events, restricted to runs with daily coverage ≥ 0.95:

| early Кэкспл | runs | failures | **h_op** | **h_cal** |
|---|---|---|---|---|
| <0.75 | 138 | 46 | **0.0402** | 0.0231 |
| 0.75–0.90 | 200 | 97 | **0.0383** | 0.0322 |
| 0.90–0.97 | 154 | 78 | **0.0383** | 0.0352 |
| **CV** | | | **0.028** | **0.209** |

**λ_op is constant across wells (CV 0.028); λ_cal is not (CV 0.209).** Damage accrues
while running, at a constant rate. So for a fleet with *heterogeneous* utilization,
`λ_cal × cal_days` is wrong **per well** even though it is right on average — high-uptime
wells fail more per calendar day. Op-days is the correct clock for per-well work
(the priority ranking, the schedule), and the pairing fix is worth doing on those grounds
as well as the 15%.

**Кэкспл as a hazard covariate is a NULL — do not add it.** The whole-run association is
strong (2.6×) and in the "cycling damages pumps" direction, but it is **reverse causation**
(failing pumps get cycled). The landmark above dissolves it. Anyone re-running this without
a landmark will "confirm" the hypothesis and be wrong.

## 4. Prerequisites — two traps that must be closed FIRST

1. **`esp_population.build()` Big-only closed-run gap (OPEN).** It carries Свод + Big
   *open* runs only, never Big-only *closed* runs. Mc/Vt are ~unaffected (Свод is
   complete); **Ya is 1596/784 vs the registry's 2118/1192 — a third of events missing.**
   Any Ya fit off it under-predicts. Found by the Vt session. **All Ya-based *level*
   numbers in this plan and in the Mc findings are provisional until this is fixed**
   (the landmark test compares *ratios across bins* and the op/`Наработка` ratio is a
   per-run median, so §1/§3 conclusions should survive — but re-verify).
   **Guard: assert fit `n_failures` against `esp_models.csv` before trusting any field.**
2. **`proc__daily_operating` codes "no data" as `in_operation = 0`** (43% of rows,
   `op_source='missing'`), and its outer join means well-days absent from *both* telemetry
   and techregime have **no row at all** (median well coverage **0.759**). A naive
   Σ`in_operation` / row-count conflates *idle* with *unobserved* — this invalidated the
   first version of the test in §3. Any op-day builder must use a **true calendar
   denominator** and a **coverage filter or explicit imputation**.

## 5. The fix, component by component

| # | component | change | risk |
|---|---|---|---|
| 5.1 | **op-day truth source** | New `esp_optime.py`: per-run `tte_op` = Σ`in_operation` over [install, end] with **true calendar denominator**, a `coverage` column, and an explicit imputation rule for gaps (`tte_op = Наработка × field-median ratio` when coverage < threshold). Telemetry starts **2018-01-01** — pre-2018 runs get the fallback. | Imputation adds error; must report the imputed share. |
| 5.2 | **fit** | `esp_population.build()` gains `tte_op` alongside `tte`; fits switch to `tte_op`. **Fix the Big-only closed-run gap at the same time** (§4.1). | Sample shrinks to telemetry-covered runs unless 5.1's fallback works. |
| 5.3 | **current age** | `crosswalk` `age_op` → op-days: Σ`in_operation` from install to today for running pumps; fallback `× field ratio`. Affects `WellState.age_pmf`. | **Low impact**: `beta≈1` ⇒ memoryless ⇒ age barely matters. Do it for coherence, not for the number. |
| 5.4 | **history replay** | `failure_rate._append_interval_predictions`: replace `overlap × uptime_factor` with per-month telemetry op-days (the `TimeMap` already does part of this — **reconcile, do not duplicate**). | Changes the *history* ratios, which are currently the calibration evidence. Must re-baseline. |
| 5.5 | **forward** | No change — `planned_op_days` is already correct op-day exposure (§1). | — |
| 5.6 | **retire `uptime_factor`** | Drop from the registry once 5.2+5.4 are paired. It is a fudge for exactly this seam. | VBA/EXE read the registry — check `vba_bundle.py` and `ValidateRegistry`. |

## 6. Falsifiable predictions (write these down BEFORE running)

1. **Forward forecast rises ~10% for Mc, ~9% for Ya.** Mc option-B forecast **40 → ~44**.
   If it does not move by roughly the uptime factor, the pairing analysis in §2 is wrong.
   *(Revised 2026-07-17: on the corrected population Mc's measured op/«Наработка» is
   **0.896**, not the 0.84 this plan was written against — so the prize is ~10%, not 15%.)*
2. **History replay ratios stay ~unchanged** after 5.4, because the replay is already
   internally consistent. If 5.4 moves Mc's 2024 ratio far from 1.05, something else is
   wrong. **This is the sharpest test in the plan** — it separates "the pairing story is
   right" from "I have mis-modelled the replay".
3. **`uptime_factor` can go to 1.0 honestly** (it already is) and the per-stratum
   `ttf_mix` clock label becomes accurate rather than aspirational.
4. **Per-well hazard spread should widen** — high-uptime wells get proportionally more
   exposure, so the priority ranking changes. Currently per-well hazard CV = 0.185.

## 7. What this does NOT fix

* **The 0.83 on 2025** — that is small-sample bias (49 events), measured on the Ya
  truncation experiment, not a clock artifact (§2).
* **The per-well DATE** — still a memoryless draw. The clock fix improves per-well
  *exposure*, not predictability of *when*.
* **The B50 identifiability problem** — unchanged.
* **The 5-column schema's inability to hold day-0 mass AND an infant window**
  (Vt session finding; the same wart as "beta1/eta1 on their bounds").

## 8. Sequencing

1. **§4.1 Big-only closed-run gap** + the `n_failures` guard. *Blocks everything Ya.*
2. **§5.1 op-day builder** with coverage handling + imputation audit table.
3. **Re-verify §1/§3 on the corrected population** (op/`Наработка` ratios; the landmark).
4. **§5.2 fit on `tte_op`** → re-run the per-stratum audit (median ratio + spread vs each
   stratum's own empirical hazard).
5. **§5.4 replay** → check prediction 6.2 (history ratios unchanged).
6. **§5.3 age**, **§5.6 retire `uptime_factor`**.
7. Re-run the Mc plan app; compare A / B / B+clock.

## 9. Effort / verdict

Steps 1–3 are cheap and independently valuable (they fix a known-wrong population and give
an honest op-day column). Steps 4–6 touch the shipped path and need the full verification
tiers. **Recommend doing 1–3 first and re-deciding at step 4** — if the corrected Ya
population moves the numbers much, the plan's magnitudes need re-estimating before any
shipped code changes.

The prize is a **~15% under-forecast on Mc** (and ~9% on Ya) plus a per-well exposure that
is physically right, at the cost of touching the calibration evidence base. It is the only
open item measured this session with a material effect on the shipped number.
