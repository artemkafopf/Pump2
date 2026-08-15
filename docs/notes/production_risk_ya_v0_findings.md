# Ya v0 — base competing-risks model (failure + ГТМ), findings

**Date:** 2026-07-28 · **Slug:** `production_risk_ya_v0`
**Code:** `backend/analysis/workflows/production_risk/ya_v0.py`
**CLI:** `scripts/run/production_risk_ya_v0.py` · **Tests:** `backend/tests/test_ya_v0.py` (24)
**Outputs:** `results/production_risk_ya_v0/2026-07-28/` · срез `SVOD_OPEN_ASOF = 2026-06-30`
**Below:** [`project_ya_v2_model`] (v2/v2.1 θ-layers) — v0 carries **no covariate at all**.

Ya v0 answers "how long does a Ya pump last and what takes it out" with two things the
single-curve fit was papering over: a **workover is not censoring** (754 of 1971 closed runs
end in ГТМ/ППР), and **the clock and the cohort are choices**. Each cause gets its own
k1/k2 latent-Weibull baseline; the two compose as latent failure times.

---

## Population

| cohort | runs | wells | failures | ГТМ | running | ГТМ share of pulls | censored |
|---|---|---|---|---|---|---|---|
| `full` | 2145 | 478 | 1217 | 754 | 174 | **38.3 %** | 8.1 % |
| `modern` (install ≥ 2022-01-01) | 681 | 263 | 351 | 169 | 161 | 32.5 % | 23.6 % |

Cause assignment (`cause_assignment_audit.csv`): 1123 closed runs by an explicit failure
reason, 754 by a workover reason, and **94 (4.8 % of closed) by the fallback branch** — blank
«Причина остановки», cause decided by the presence of a named failed unit, all landing on
*failure*. If that inheritance is wrong the failure count is overstated by ≤94; the ГТМ count
is untouched either way.

---

## 1. Which clock is honest — measured, not argued

The disqualifying test is **availability read across outcomes**: a clock present for events
but absent for censorings times the two halves of the likelihood differently.

| cohort | clock | availability spread | measured-share spread | verdict |
|---|---|---|---|---|
| full | `t_cal` | **0.000** | 0.000 | **ok** |
| full | `t_nno` (ННО) | **0.971** | 0.971 | disqualified — structural |
| full | `t_op` | 0.465 | 0.465 | disqualified — structural |
| full | `t_mix` | 0.000 | **0.465** | disqualified — differential imputation |
| modern | `t_cal` | **0.000** | 0.000 | **ok** |
| modern | `t_mix` | 0.000 | 0.178 | **borderline — sensitivity only** |

**ННО («Наработка», the mart's `ttf_true_best`/`ttf_mix_svod_plus_big_nno`) is disqualified,
and it hides the fact.** The raw column looks 100 % complete only because
`load_svod_runs` back-fills every open run with its calendar age — **169 of 174 running Ya
pumps carry a substituted value**. Strip the substitution out (`t_nno_reported`) and coverage
is 100 % on failures and ГТМ against **2.9 % on running pumps**. That is the legacy `tte`
defect exactly ([`project_time_scales`]): ННО times the events, the calendar times the
censorings, and since ННО < calendar (median ratio 0.981–0.987) it biases survival **up**.
61 Ya runs additionally report an ННО exceeding their own calendar span.

**`ttf_mix` is not one thing on Ya — it is two clocks split by era.** Telemetry op-time is
0 % covered before 2020 and ~87 % from 2021, so measured coverage is **differential in the
worst direction**: 69.0 % of running runs against 30.6 % of failures and 22.5 % of ГТМ. On
full Ya, `t_mix` is 69 % `t_cal × 0.892` — a rescaling of the calendar clock wearing an
op-clock's name. On **Ya-modern it is 82.4 % genuinely measured** and the spread falls to
0.178, which is a real second measurement.

> **Verdict: fit `t_cal`. Use `t_mix` as a sensitivity on `modern` only. Never fit ННО.**
> Reported ННО stays in the tables because the business reads it, not as a clock.

---

## 2. The fits (clock `t_cal`)

| cohort | cause | kind | w₁ | β₁ | η₁ | β₂ | η₂ | ΔAIC | **RMST₇₃₀** [95 % CI] | MRL(0) | median | KM RMST₇₃₀ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | failure | k1 | — | 0.7875 | 704.0 | — | — | +3.92 | **427.4** [415.7, 441.4] | 806.9 | 443 | 430.8 |
| full | **ГТМ** | **k2** | 0.159 | 1.1064 | 144.7 | 1.1164 | 1483.4 | **−9.18** | **522.2** [508.2, 536.0] | 1220.4 | 827 | 521.8 |
| full | all-pulls (composed) | — | — | — | — | — | — | — | 321.4 | 431.6 | 239 | 323.0 |
| modern | failure | k1 | — | 0.7310 | 714.0 | — | — | +2.90 | 421.6 [399.7, 455.5] | 868.7 | 433 | 425.7 |
| modern | ГТМ | k1 | — | 0.7638 | 1823.2 | — | — | −1.63 | 555.8 [523.4, 588.7] | 2130.3 | 1129 | 552.9 |

CIs are a 200-replicate **well-cluster** bootstrap (200/200 replicates converged in every
cell). Model−KM agreement is −4.1 … +2.9 d everywhere on `t_cal`.

**The failure arm reproduces Ya v2's baseline exactly** (RMST₇₃₀ 427.4, β 0.7875, η 704 —
identical to [`project_ya_v2_model`]). That is by construction: `classify(..., gtm_is_failure=
False)` and `classify_cause(...) == FAILURE` select the same rows. So **v0 is a strict
decomposition of v2's baseline, not a rival fit** — it adds the arm v2 was discarding.

**The two causes have genuinely different shapes.** Failure is **β ≈ 0.79 < 1** —
infant-dominated, hazard falling with age, no wear-out. ГТМ is **β ≈ 1.11 > 1** on full
history. A workover is not a failure with a different label; the processes have opposite
age signatures, which is the mechanical reason censoring one as if it were the other distorts
both.

⚠ **The ГТМ k2 is a pure scale mixture, not a shape mixture.** β₂ = 1.1164 sits essentially
*on* the EM's `β₂ > max(1, β₁) = 1.1064` constraint, which is a **regulariser, not a
measurement** ([`project_esp_survival_em`]). Read the k2 as "two ГТМ timescales, η ≈ 145 d
(16 % of runs) and η ≈ 1483 d", not as "the long mode wears out faster".

---

## 3. The competing-risks composition validates

`S_all = S_fail · S_gtm`, `CIF_k = ∫ S_all dH_k` assumes the causes are conditionally
independent given age. Checked rather than asserted (`cif_check.csv`):

* parametric CIF vs **Aalen–Johansen**: max |gap| **0.013** across all `t_cal` cells;
* composed `S_all` vs the **all-pulls KM**: max |gap| **0.008**;
* identity `S_all + Σ CIF_k = 1`: residual **0.00000**.

The model is usable as a competing-risks object.

### What censoring ГТМ as random costs you (full, `t_cal`)

| horizon | failure CIF (AJ) | naive 1−KM | overstatement | ГТМ CIF | naive | overstatement |
|---|---|---|---|---|---|---|
| 90 d | 0.171 | 0.179 | +4.7 % | 0.100 | 0.111 | +11.7 % |
| 180 d | 0.265 | 0.289 | +9.1 % | 0.154 | 0.182 | +18.4 % |
| 365 d | 0.382 | 0.443 | **+16.1 %** | 0.234 | 0.309 | **+32.3 %** |
| 730 d | 0.502 | 0.641 | **+27.6 %** | 0.308 | 0.470 | **+52.8 %** |

Same family as the Mc (+27 % @365 d) and Vt (+17 % @365 d) results, and **larger than both** —
Ya has the heaviest ГТМ load of the three fields.

### Deliverable form — P(pull within 365 d | alive at age a), full/`t_cal`

| age | P(failure) | P(ГТМ) | P(any pull) | failure share |
|---|---|---|---|---|
| 0 | 0.385 | 0.234 | **0.619** | 62 % |
| 90 | 0.345 | 0.220 | 0.565 | 61 % |
| 365 | 0.313 | 0.186 | 0.499 | 63 % |
| 730 | 0.286 | 0.191 | 0.477 | 60 % |

Workshop load and its failure/ГТМ split now come from one object. The failure share of pulls
is **flat at ~62 %** across age — the mix does not drift with pump age.

---

## 4. Full Ya vs Ya-modern — the cohort matters for ГТМ, not for failure

Install-cutoff sweep (`cohort_scan.csv`, `t_cal`, model and KM agree within 8 d throughout):

| cut | runs | fail | ГТМ | censored | failure RMST₇₃₀ | ГТМ RMST₇₃₀ | measured op |
|---|---|---|---|---|---|---|---|
| 2019 | 1172 | 605 | 394 | 14.8 % | 452.5 | 517.3 | 0.566 |
| 2020 | 929 | 491 | 266 | 18.5 % | 441.0 | 552.3 | 0.714 |
| 2021 | 772 | 407 | 199 | 21.5 % | 424.8 | 556.6 | 0.834 |
| **2022** | **681** | **351** | **169** | **23.6 %** | **421.6** | **555.8** | **0.824** |
| 2023 | 497 | 240 | 111 | 29.4 % | 431.3 | 568.4 | 0.807 |
| 2024 | 282 | 114 | 41 | 45.0 % | 462.5 | 615.9 | 0.755 |

* **Failure life is flat across cohorts** — 422–463 d with no monotone trend, and full Ya
  (427.4) sits inside the range. **There is no failure-side reason to restrict to a modern
  cohort**; the full population's extra 866 failures are worth more than the recency.
* **ГТМ life lengthens monotonically**, 517 → 616 d. Workovers are getting rarer and later.
  This is what actually separates `modern` from `full`: ГТМ RMST 555.8 vs 522.2 (+6 %).
* **The full-Ya ГТМ k2 is an era mixture, not two populations.** k2 is selected only at the
  full-history and 2019 cutoffs; from 2020 onward both causes are k1. The short ГТМ mode
  (η ≈ 145 d, 16 % weight) is the pre-2020 workover regime, not a standing subpopulation.
* Read the 2024 row with care — 45 % censored, max observed age 908 d, 41 ГТМ events.

**Recommendation: quote `full` + `t_cal` for the failure arm and for anything ННО-facing;
quote `modern` for ГТМ rates and for anything forward-looking about workover load.**

---

## 5. `t_mix` on `modern` fails three independent diagnostics — do not ship it

Both causes on `modern`/`t_mix` select a **degenerate k2**: a spike at η = 1.6 d (failure,
w₁ = 0.093, ΔAIC −51.3) and η = 0.7 d (ГТМ, w₁ = 0.054, ΔAIC −16.5), flagged by
`k2_degenerate_spike`. Three checks agree independently:

1. the degeneracy flag fires (component η < 5 d);
2. the CIF check degrades — `S_all` vs KM gap reaches **0.050** (vs ≤0.008 on `t_cal`);
3. the window probabilities go **flat with age** (0.392 → 0.394 from day 180 to 730) because
   the surviving component has β ≈ 1.01, i.e. memoryless.

Cause: the op-clock legitimately produces near-zero durations for runs that barely operated,
and a mixture will buy likelihood with a spike on them. The op-clock's own physics, not a
fitting bug — but it means **the modern op-time fit needs a short-duration policy before it
is usable**, and until then `modern`/`t_cal` is the modern model.

---

## Reproduce

```bash
python scripts/run/production_risk_ya_v0.py --n-boot 200 --num-starts 60
```

~70 min (8 cause-specific 60-restart EM fits + 4 cluster bootstraps + a 12-fit sweep).
`--num-starts 12` for a smoke run; below that the multimodal EM is a sample, not a fit.

## Open

* **Short-duration policy for the op-clock** — blocks `t_mix` on `modern` (§5).
* **The 94 fallback-branch runs** (blank reason + a named failed unit → failure). A spot
  check against the ремонт records would either confirm them or shave ≤7.7 % off the failure
  count.
* **ГТМ informativeness is untested here.** v0 assumes conditional independence and the CIF
  check passes, but that check cannot see a *common* driver. On Vt ГТМ is informative
  ([`project_vt_competing_risks_lab_h2s`]) and on Mc it is not — Ya is unknown, and it is the
  one assumption between v0 and a trustworthy latent-no-ГТМ counterfactual.
* **v0 × v2 composition** — whether v2's θ-layers apply to the failure arm only, or whether
  ГТМ needs its own (contractor is the obvious candidate: ГТМ scheduling is an operator
  decision, so a contractor effect on the ГТМ arm would be a *policy* signal, not a
  reliability one).
