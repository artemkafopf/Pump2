---
name: field-k1k2
description: Fit a k1/k2 latent-Weibull mixture survival model for one or more fields. Use whenever the user says "build a field k1/k2 model", "k1/k2 for <field>", "latent Weibull for <field>", "mixture baseline for <field>", or asks for a field-level survival baseline / RMST / MRL / hazard shape. Covers the censoring rules, the Мирнинский cohort filter, the multi-start requirement and the mandatory KM control.
---

# Field k1/k2 latent-Weibull model

"k1/k2" = a **two-component latent Weibull mixture** on time-to-event, fitted with
**all censored runs included**. k1 is the short-lived (infant) component, k2 the
long-lived one. This is a baseline fit — no covariates — unless the user asks for
θ-layers on top.

## Population — censoring is not optional

Load via `analysis.workflows.esp_survival.data.load_failures_df()`.

`event = 1` only for `failure_flag == 1`. **Everything else is censored, not dropped:**

- running pumps (still installed at snapshot) → censored at current age
- ГТМ / workover pulls, planned pulls, non-failure pull reasons → censored at pull
- pulls with an unknown reason → censored (do not promote to failure)

A failures-only fit is **wrong and biased, not merely noisy** — it is asymptotic
bias, proven on synthetic data where the censored fit recovers β₁/η₃₆₅ in every
scenario and the failures-only fit does not. Never filter to `event == 1`, never
drop young runs, never apply an `age > TTF` cutoff. If the user asks for
failures-only, fit it *alongside* the honest fit and report the gap.

## Hard filters

- **Мирнинский (Mc/Mr): installs 2024-01-01 or later only.**
  `pop[(pop.field == "Mc") & (pop.install >= C.MC_INSTALL_COHORT_START)]`
  This is a **cohort filter on the install date**, never left truncation —
  `add_entry_age` keeps pre-2024 exposure and hides that recent runs are shorter.
  Other fields use all history.
- Ya has a single H₂S class → never split it sour/non-sour.
- Vt: sour vs non-sour differ ~3× in life **and** in shape → never pool them.
  Use the well-level Свод-sour relabel if sour censoring matters.

## Clock

`tte` **mixes clocks** — events land on Наработка, censorings on calendar. State
which clock the fit used. If the answer depends on it, fit both and say so.

## Fit

`analysis.models.survival.weibull_em.fit_latent_weibull_em`, or
`mixture_baseline.fit_baseline` when a `BaselineFit` object with `compose()`-ready
parameters is wanted.

- **The likelihood is multimodal.** A single-seed EM is a sample, not a fit.
  Use ≥60 starts (`num_starts=60` is the `fit_baseline` default; Halton starts in
  the EM path). Report the best NLL *and* whether the top starts agree.
- `β₂ > max(1, β₁)` is a **regulariser, not a wear-out finding**. Never test or
  defend it by log-likelihood.
- Assert the fit's `n_failures` against `esp_models.csv` before believing any
  number — a Big-only-closed gap has silently killed claims before.

## Report — standing rule

For **every** survival fit, report **RMST(0, 730)** and **MRL(0)**, not B50.
Expect roughly a 13 % MRL-vs-RMST tail gap. Include a **KM control** on the same
population — if the fitted RMST and the KM RMST disagree by more than a few
percent, the fit is not shippable.

Also report: n_runs, n_failures, n_censored, w1, (β₁, η₁), (β₂, η₂), and the
hazard shape in words (infant spike / flat / wear-out).

## Output

`out = results_dir("esp_survival_<field>_k1k2")` → `tables/`, `figures/`.
Nothing to `analysis_outputs/` or `temp/`.

## Reference implementations

- [scripts/run/esp_survival_field.py](../../../scripts/run/esp_survival_field.py) — whole-field, no split
- [backend/analysis/workflows/esp_survival/field_mixture.py](../../../backend/analysis/workflows/esp_survival/field_mixture.py) — `run(df, out_dir)`
- [backend/analysis/models/survival/mixture_baseline.py](../../../backend/analysis/models/survival/mixture_baseline.py) — `fit_baseline`, `life_from_S`
