# Time scales for ESP survival — audit and naming proposal

Дата: 2026-07-20. Companion to `production_risk_clock_fix_plan.md` (which scoped the
fit↔exposure pairing bug). This note covers a different and more basic problem: **we do
not have agreed names for the raw quantities**, and one column silently carries two
different clocks at once.

---

## 1. What Mc is actually fit on (checked, not assumed)

The production Mc model (`esp_forecast.fit_mc_model`, the "cal c0 k2" that feeds
`Прогноз_ремонтов`) fits `mc["tte"]` straight off `esp_population.build`
(`esp_forecast.py:181-182`). It is **not** `esp_optime`, and it is **not** pure calendar.

`esp_population.build` fills `tte` as:

```python
g["tte"] = np.where(ended, g["tte_full"], np.minimum(g["tte_full"].fillna(np.inf), age_at_t))
```

so the column resolves differently depending on the row:

| row type | what `tte` ends up being |
|---|---|
| closed Свод run | «Наработка (сут)» — a **reported** figure |
| closed Big-only run | `nno_days` — reported |
| **open (running) Big run** | `tte_full` is NaN ⇒ **true elapsed calendar age** |
| open Свод run | min(Наработка, calendar age) |

**Measured (as-of 2026-07-01):**

| field | n | `tte` == calendar | closed rows == cal | open rows == cal | closed Наработка/cal (med) |
|---|---|---|---|---|---|
| Mc (2024+) | 164 | 69 (42%) | 9 / 103 | **60 / 61** | 0.947 |
| Vt | 641 | 174 (27%) | 36 / 503 | **138 / 138** | 0.962 |
| Ya | 2166 | 436 (20%) | 270 / 1999 | **166 / 167** | 0.984 |

⇒ **Events are timed on Наработка; censorings are largely timed on calendar.** The two
differ by ~4-5% at the median on closed runs, and the mismatch is *systematic in the
direction that inflates survival* — censored rows get the longer clock. This is a
population-construction defect, not a modelling choice, and it applies to every field.

So the user's recollection "Mc uses calendar days" is **approximately right and worth
making exact**: Наработка sits at 0.988 (Mc) / 0.983 (Vt nonsour) / 0.951 (Vt sour) of
true calendar at the median, i.e. Наработка is calendar-like, NOT operating time
(telemetry op-days sit at 0.85-0.89 of calendar). And for 42% of Mc rows `tte` *is*
literally the calendar number.

## 2. The `ttf_mix` name means two different things

| name | where | definition |
|---|---|---|
| `ttf_mix` (mart) | `esp_survival/data_mart.py:4-6` | `ttf_true_best_days` (telemetry/ТР true op-days, capped at `run_days`), falling back to `run_days` when the source is missing. **Genuinely operating time.** |
| `ttf_mix_svod_plus_big_nno` (registry `clock` column) | `esp_models.csv`, written by `refit_weibull_big_censoring.py:188,202,275` | Свод «Наработка (сут)» / Big `nno_days`. **Reported, calendar-like.** |
| `clock = "ttf_mix"` (Mc row) | `mc_refit.py:284` | `_duration_days`: Наработка if present, ELSE `(end-start).days × uptime_factor(0.745)` — a **third** hybrid. |

`clock_fix_plan.md:116` already flags the registry label as "aspirational rather than
accurate". Three different quantities under one name is the root of every clock argument
we keep re-having.

## 3. Agreed scheme — TWO model families, `cal` and `op`

Decided 2026-07-20. There is no reason to carry more than two families of fits, so ННО
is demoted out of the clock set entirely.

| name | definition | role |
|---|---|---|
| **`t_cal`** | `pull − install` (or `as_of − install` while running) | **the `cal` family.** Two dates subtracted: always available, never reported, never imputed. |
| **`t_op`** | telemetry operating days, `NaN` where unmeasurable | the honest measurement. Audit against this; never fit on it alone. |
| **`t_mix`** | `t_op` where measured, else `t_cal × Кэкспл` (field median, then global) | **the `op` family — the servable clock.** |
| `t_mix_source` | `measured` \| `imputed_field_kexp` \| `imputed_global_kexp` | provenance, per row |
| `kexp_used` | the ratio actually applied | lets any imputed row be undone |
| `t_nno` | «Наработка (сут)» / ННО, `NaN` where unreported | **reporting only** — never a fit clock |
| `t_op_plan` | ПП «Отработанное время» | forward exposure; pairs with `t_mix` |

**Why `t_op` alone cannot be the op family.** Measured coverage is 44% (Mc), 59% (Vt),
30% (Ya). A measured-only fit drops half the population, and not at random —
poorly-instrumented wells are exactly the ones with worse records. `t_mix` keeps every
run and carries its provenance so the imputed share is always auditable.

**Why the imputation base is `t_cal`, not ННО** (the legacy `tte_op` used ННО): the
transferable quantity is Кэкспл = op-days / calendar-days, a physical utilisation share
bounded by 1. Imputing off ННО inherits ННО's mixed clock and reporting noise, and
op/ННО has no physical meaning. Measured Кэкспл medians come out at **0.844 (Mc),
0.861 (Vt), 0.887 (Ya)** — matching the plan's own Кэкспл (0.846 Mc / 0.911 Ya) and the
realized telemetry values in `clock_fix_plan.md:23-25`, which is the cross-check that
the ladder is sane.

**Why ННО is not a clock.** It exists for only 106/164 Mc, 508/641 Vt runs — every
running pump lacks one. So it can time events but never censorings, which is precisely
the defect in §1. It is also calendar-like, so it adds no information `t_cal` lacks.

Rules that follow:

1. **One clock per fit, applied to events AND censorings alike.**
2. **Fit on `t_mix`, pair with plan op-days; or fit on `t_cal`, pair with calendar.**
   Never fit on one and serve the other (`clock_fix_plan.md:28-44` — λ_cal × plan
   op-days under-states by ~9-15%). Convert for reporting at the end, via `kexp_used`.
3. **The registry's `clock` column must name the actual column** — `t_cal` or `t_mix`.
   Anything reading `ttf_mix*` is re-stamped only after verification.
4. **`uptime_factor` stays 1.0 and is deprecated** — it exists only to fake one scale
   from another; with explicit columns it has no job.
5. **Data-quality gate**: `t_nno_valid = t_nno <= t_cal`. Currently 19 Vt, 63 Ya, 0 Mc
   runs fail it. Violations are flagged, never silently kept.

Implemented in `esp_population.add_time_scales` (`t_cal`, `t_nno`, `t_nno_valid`) and
`esp_optime.add_op_scales` (`t_op`, `t_mix`, `t_mix_source`, `kexp_used`). The legacy
`tte` / `tte_op` columns are left untouched so the shipped bundle's consumers do not
move; new fits take `t_cal` or `t_mix`.

## 3a. Next step agreed — Кэкспл as a Cox covariate

`kexp_used` is now a per-run column with provenance, which makes Кэкспл available as a
hazard covariate rather than only a clock conversion. Two cautions carried from prior
work before that is attempted:

* **`kexp` is a LOWER bound** where coverage is imperfect — unobserved well-days are
  counted idle (`esp_optime` docstring, traps 1-2). Restrict any Cox fit to
  `t_mix_source == "measured"`, or the covariate is partly a coverage artifact.
* **Reverse causation.** A pump that is failing gets shut in, so low Кэкспл can be the
  *consequence* of impending failure rather than its cause. The Phase-D landmark design
  (Кэкспл over days 0-90 -> failures after day 90) is the leakage-proof form and already
  showed the per-op-day rate is flat across utilisation (CV 0.028) while the
  per-calendar-day rate is not (CV 0.209) — i.e. Кэкспл's first-order effect is already
  absorbed by choosing the op clock. The open question is whether a *residual* effect
  survives on top of `t_mix`, which is a genuinely different test.

## 4. What this changes for the Vt refit in flight

The three-clock grid (`production_risk_vt_weibull_grid.py`) already fits `narabotka`
(= `t_nno`), `cal_true` (= `t_cal`) and `op_days` (= `t_op`) separately, so the Vt
candidates can be re-read on whichever scale we standardise on. Best-fit KS by clock
shows the choice is not cosmetic — `t_cal` wins on 4 of 8 strata, `t_op` on 3, `t_nno`
on 1 — and RMST moves up to 15% between scales on the same stratum.

**Blocked pending agreement**: which scale the shipped registry should be on. Until then
no rows are swapped.
