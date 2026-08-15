# CatBoost vs Weibull — ESP Failure-Rate Comparison: Verdict

**Date:** 2026-07-14 · **Branch:** `data_storage_and_processing` · **Scope:** fact window
2024-01..2026-04, per УН and fleet-global.

## TL;DR

**CatBoost does not track the actual monthly ESP failure rate better than the Weibull
survival model.** On the headline metric — cross-fitted, well-clustered CatBoost vs the two
shipped Weibull lines — Weibull-base tracks fact at least as well as CatBoost globally
(monthly-rate MAE **0.0142** vs **0.0151**) and in 8 of 13 УН. This is the **expected,
publishable honest-null outcome**: every prior out-of-sample gate (linear θ in C5, Phase-D
landmark dynamics) found the covariates do not transport, and an orthogonal gradient-boosted
method now confirms it on the failure-rate line. The one nuance worth acting on is
**Мирнинский** (below).

Weibull-base and Weibull-hazard (θ) MAE are **identical to 6 decimals in every УН** —
re-confirming the project's own finding that θ adds ~nothing, which is part of what made an
ML cross-check interesting.

## What was built (parallel line, nothing replaced)

- **The bridge** (`failure_rate_catboost.build_cb_survival`): per-run TTF quantiles
  `(b10,b50,b90)` in operating days → a monotone `S_cb(t)=1-F(t)` anchored at
  `F(b10)=.1, F(b50)=.5, F(b90)=.9` (plus `F(0)=0`), with a linear pre-b10 ramp and an
  exponential post-b90 tail whose rate is the implied hazard of the `b50→b90` segment
  (hazard continuous at the b90 joint). Degenerate/tied/≤0 anchors are guarded to a
  steep-but-finite CDF — never a step function or a division by zero.
- **`catboost_p_fail`** is byte-for-byte the conditional formula of the Weibull
  `current_pump_p_fail`: `clip(1 − S_cb(age+Δ)/S_cb(age), 0, 1)`. The *only* difference
  between the two model lines is the survival curve.
- **Parity by construction:** `failure_rate._append_interval_predictions` now takes an
  injected `p_fail_fn(age_pmf, op_days)`; its default is the current Weibull closure (so the
  Weibull path stays byte-identical), and the CatBoost line reuses the identical
  month-slicing / aging / exposure code. The shared `uptime_factor` parameterises *exposure*,
  not the survival curve — no Weibull enters the ML line.
- **Capacity fairness:** two CatBoost lines — all-data in-sample (parity with how the shipped
  Weibull bundle is fit) and a **5-fold, well-clustered cross-fit** line (each history run
  scored by a model that saw neither it nor any run of its well). The cross-fit line is the
  headline; the in-sample line is a memorisation-bounded secondary.
- **Covariate join** (§6): fleet intervals → `build_regression_frame` rows by casefolded
  `well_key` + nearest `install_date` (±7 d). Three tiers, all **100% ML** (never Weibull):
  matched run → full covariates; same well / no date match → categorical-only from the
  well's borrowed categoricals; well absent from the frame → a `field × install_period`
  categorical-only grid. Operational numerics are left NaN for the latter two (CatBoost
  handles NaN natively).

Gated behind `RunConfig.enable_catboost_compare` / CLI `--catboost-compare` (default off).
`import catboost` is lazy and confined to the comparison path — the frozen production EXE
never imports it. **VBA / launcher / EXE are untouched** (productionisation deferred).

## Step 0 — out-of-sample transport verdict (cite, not restaged)

From the C5 temporal hold-out harness (`results/catboost_regression_v2/2026-07-13/`), the OOS
*transport* anchor for everything below:

| cutoff | CatBoost C-index | stratum-KM C-index | Δ | CatBoost IPCW-MAE(b50) | stratum-KM MAE |
|---|---|---|---|---|---|
| 2023-12-31 (primary) | 0.581 | 0.565 | +0.016 | 169 op-d | 314 op-d |
| 2022-12-31 | 0.616 | 0.541 | +0.075 | 190 op-d | 202 op-d |

C-index barely above chance (0.5) with a small delta over the stratum baseline; the MAE edge
is real but the stratum-KM b50 quantile baseline is coarse. Equipment-block ablation
ΔC-index ≈ **+0.004** (negligible). Top mean-TTF importances are `nominal_freq_hz`,
`pbubble_atm`, `freq_std_early`, `log_glf_mean_opdays`, `field`. **Read: covariates carry a
little rank signal in-fold but do not transport out-of-sample** — the honest-null expectation.

## The comparison verdict (fact window, monthly-rate MAE vs fact)

Full table: `results/catboost_vs_weibull_failure_rate/2026-07-14/tables/verdict_mae.csv`.
Headline = cross-fitted CatBoost. Selected rows:

| УН | n | Weibull-base | Weibull-hazard | CatBoost in-sample | **CatBoost cross-fit** | best |
|---|---|---|---|---|---|---|
| **ГЛОБАЛЬНО** | 28 | **0.0142** | 0.0142 | 0.0173 | 0.0151 | weibull-base |
| Ярактинский УН | 28 | **0.0212** | 0.0212 | 0.0214 | 0.0227 | weibull-base |
| Даниловский УН | 28 | **0.0317** | 0.0317 | 0.0543 | 0.0548 | weibull-base |
| Верхнетирский УН | 28 | **0.0382** | 0.0382 | 0.0548 | 0.0541 | weibull-base |
| Мирнинский УН | 28 | **0.0670** | 0.0670 | 0.0704 | 0.0697 | weibull-base |
| Аянский (Западный) УН | 28 | 0.0383 | 0.0383 | **0.0327** | 0.0329 | catboost |
| Кийский УН | 12 | 0.1812 | 0.1812 | 0.1790 | **0.1790** | catboost-xfit |

- **Global:** Weibull-base (0.0142) < CatBoost cross-fit (0.0151) < CatBoost in-sample
  (0.0173). CatBoost systematically **under-predicts** the fleet rate in late history (e.g.
  2026-04 fleet 508: fact 0.0748, Weibull 0.0785, CatBoost 0.0592). Note the in-sample line is
  *worse* than cross-fit globally — there is no memorisation advantage at the fleet level, so
  the "an all-data CatBoost that wins the history MAE" failure mode the handoff warned about
  does not even arise here.
- **Per-УН:** Weibull-base wins 8 of 13 УН; CatBoost wins a handful, all small/short
  (Аянский-Западный; and the thin Большетирский n=3 / Иктехский n=1 / Кийский n=12 groups).
- **CatBoost cross-fit ≈ CatBoost in-sample** everywhere — the covariates that would let the
  in-sample model memorise per-run failure months simply do not carry that far, matching
  step 0.

### Мирнинский — the one nuance

Мирнинский is the УН that motivated the whole production-risk refit, and it is the only place
CatBoost does something structurally interesting. Pre-registered expectation: Weibull
under-counts Мирнинский because **~34 % of its failures fall in plan-non-producing months**,
and the exposure plumbing (shared by both lines) assigns those months zero exposure — so **no
model, CatBoost included, can place those failures**; the answerable question is only whether
CatBoost redistributes predicted failures across the *producing* months closer to fact.

Over the fact window, mean producing-month rate:

| | fact | Weibull-base | CatBoost cross-fit |
|---|---|---|---|
| Мирнинский mean rate | **0.115** | 0.087 (ratio ≈ 1.33) | **0.106** |

**On level/bias, CatBoost helps**: it lifts the producing-month rate from 0.087 to 0.106,
most of the way to the 0.115 fact level — i.e. it *does* redistribute mass toward fact on the
producing months, the narrow question we could answer. **On month-to-month MAE it is a wash**
(0.0697 vs 0.0670): the lift comes with slightly noisier monthly tracking, so the MAE ranking
still favours Weibull. Neither line closes the non-producing-month gap — that is structural,
not a CatBoost failure.

## Diagnostics (`tables/catboost_coverage.csv`)

- **Covariate-coverage share** (predicted-failure mass from full-covariate matched runs vs
  categorical-only): **global 84 %**. High for the majors (Ярактинский 90 %, Западно-Ярактинский
  81 %, Мирнинский 78 %), low where the CatBoost line is weakly informed and should be read as
  such (Кийский 36 %, Иктехский 0 % — a single categorical-only well/month).
- **Past-b90 tail-dependence share** (pump-month evaluations at ages beyond the predicted b90,
  where the CatBoost line rides the exponential tail choice): **global 23 %** — material, so the
  CatBoost line is partly a tail-extrapolation artefact, not purely data-driven. Highest at
  Даниловский (44 %) and Ярактинский (29 %). This is a genuine caveat on the CatBoost line's
  credibility, especially for long-lived-survivor УН.

## Fairness / parity confirmations

- Same clock (`ttf_mix` op-days), same fleet denominator, same months, same УН grouping, same
  fact-blanking, same materiality — only the survival curve changes.
- Weibull `predicted_rate` is unchanged with the flag off: the whole CatBoost block is gated,
  and the injected-callable refactor is proven byte-identical to the prior Weibull closure by
  `test_injected_default_equals_weibull_closure` (plus the unchanged 19-test
  `test_production_risk` suite).
- CatBoost columns are NaN from `forecast_start` (2026-07) on — there is **no CatBoost forecast
  line** (the Weibull forward line is a renewal simulation; a non-renewal ML overlay there
  would be apples-to-oranges, and there is no fact to judge it against).

## Limitations

- Operational features (freq/load early-window stats) exist only for runs that actually
  operated, so CatBoost is best-informed on telemetry-bearing runs and leans on
  categorical-only inputs on gap runs; 16 % of predicted-failure mass is categorical-only.
- Categorical-only fallback predictions (tiers 2–3) use the all-data model for both variants;
  they carry negligible memorisation risk (no operational features) but are, by construction,
  weakly informed.
- The 23 % past-b90 tail share means a non-trivial slice of the CatBoost line is exponential
  tail extrapolation beyond the quantile anchors.

## Reproduce

```bash
# Step 0 (OOS transport, once):
python scripts/run/catboost_regression_v2.py
# The comparison (figures + tables + verdict):
python scripts/run/catboost_vs_weibull_failure_rate.py \
  --pp-master ... --gtm-schedule ... --prediction-workbook ... \
  --techregime-workbook ... --equipment-big ...
# Or fold the CatBoost line into the standard run's full tables:
python scripts/run/production_risk.py --catboost-compare --full-tables
```

Outputs: `results/catboost_vs_weibull_failure_rate/<date>/{figures,tables}/`
(`F_failure_rate_compare_monthly.csv`, `verdict_mae.csv`, `catboost_coverage.csv`, per-УН and
global PNGs).

## Bottom line

An independent, capacity-fair, cross-fitted gradient-boosting model **does not beat the
7-parameter stratum Weibull** at tracking the actual monthly ESP failure rate — it ties or
loses globally and in most УН, and only nudges the Мирнинский producing-month *level* toward
fact without improving its monthly MAE. That is the triangulation this exercise was for: a
methodologically orthogonal method reaching the same conclusion the linear stack did — the
fleet's failure timing does not forecast from these covariates out-of-sample. **Do not ship
CatBoost into the Excel/VBA/EXE tool** on this evidence.
