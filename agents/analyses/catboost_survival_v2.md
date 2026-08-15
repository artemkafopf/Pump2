# CatBoost Direct Regression (v2) — an independent, Weibull-free second opinion

**Status:** PLAN ONLY (do not implement). **Date:** 2026-07-08. **Supersedes** the earlier
survival-mimic draft of this file per user direction (2026-07-08): the goal is now a
**completely independent ML *regression* analysis**, deliberately decoupled from the
Weibull / Cox survival stack, as a cross-check — not another survival model.

---

## 0. Did the former CatBoost model include Weibull parameters? — No (you're right)

Precisely:

- The trained model is `CatBoostRegressor(loss="SurvivalAft:dist=Extreme")`. The **trees
  predict a single scalar** `μ(x)` = a log-time location. **No shape or scale parameter is
  learned by the model.**
- The Weibull appears only as **scaffolding bolted on afterward**:
  - `BETA_BASE = 1.316`, `ETA_BASE = 240.3` — a *hardcoded* Weibull baseline band model
    (`h_base`) used for the frequency-scenario RUL;
  - `beta_weibull = 1/σ̂` with `σ̂ = std(residuals)·√6/π` — the shape **estimated post-hoc
    from residuals**, not by CatBoost;
  - `conditional_rul` then wraps `η_i = exp(μ(x))` and that global `β` into a Weibull
    conditional quantile.

So "the CatBoost model did not include Weibull parameters" is correct: it emits one number
per pump; the Weibull is a post-processing wrapper that turns that number into RUL. (The AFT
*loss* assumes an extreme-value error internally for censoring, but that is a fixed
likelihood, not learned parameters.) The redesign below removes **all** of this Weibull
scaffolding.

## 1. What v2 is: an independent regression triangulation

A standalone ML regressor that predicts time-to-failure and remaining life **directly**, with
**no** survival-model machinery — no Weibull, no η/β, no mixture, no stratum baseline, no
hazard function. Its only shared DNA with the survival stack is what makes the comparison
fair:

- the **data** (`ttf_mix`, full population, 2,634 runs),
- the **verified-hazard feature set** (the model-report §2 covariates), and
- the **C5 temporal hold-out** gate.

Everything else — target construction, loss, RUL definition, validation metrics — is its
own. **Why bother:** if a methodologically orthogonal approach reproduces the stack's
conclusions (which hazards rank, the TTF ranges, and — critically — the out-of-sample
transport failure), that is strong triangulation. If it disagrees, that is a finding. A
reskinned survival model cannot provide either; a genuinely independent regressor can.

## 2. Carried over vs dropped

**Kept (the hard-won lessons that are method-agnostic):**
- Censored data must be used — full population, not failures-only (Act I).
- `ttf_mix` operating-time clock (not calendar, not raw `ttf_true>0`); drop the old
  `kip>0.5` uptime selection.
- Verified hazards only, at t0 / first-30-op-day windows (model-report §2/§4); the new
  equipment + reconciled-restart covariates included.
- Well-clustered, install-cohort temporal validation (C5).

**Dropped (all Weibull/survival scaffolding):**
- `SurvivalAft` loss, `ETA_BASE/BETA_BASE` band model, post-hoc `β` from residuals,
  `conditional_rul`, the mixture idea, any per-pump η/β.
- Rejected covariates (ion/gypsum/salt chemistry, whole-run `_m_mean` mature features,
  `idle_frac`, КВЧ) — never fed.
- Vt-only scope → full fleet.

## 3. The one real problem: censoring in a plain-regression frame

Naive regression on the observed time is biased — censored runs have `ttf_mix` shorter than
their true life, so ordinary RMSE/quantile loss under-predicts. Three clean,
distribution-free ways to fix it **without** a survival model (pick one primary, one
cross-check):

- **(a) IPCW-weighted regression (recommended primary).** Fit the Kaplan–Meier of the
  *censoring* distribution `Ĝ(t)` (censoring as the "event"); weight each **observed
  failure** by `w_i = 1/Ĝ(T_i)`, censored rows drop out. A standard CatBoost regressor
  (RMSE or pinball) on the weighted failures is then consistent for the full-population
  conditional mean/quantile. Fit `Ĝ` **per stratum** so the independent-censoring
  assumption is credible. Pure regression; censoring handled by weights only.
- **(b) Pseudo-observation regression (cross-check).** Replace the censored outcome with
  jackknife **pseudo-values** of a chosen functional — RMST `E[min(T,τ)]` or survival
  `S(t₀)=E[1{T>t₀}]` — computed from the KM estimator. Pseudo-values are defined for
  censored subjects too and satisfy `E[pseudo_i|x_i]=θ(x_i)`, so an ordinary CatBoost RMSE
  regression on them recovers the conditional functional. This turns the censored problem
  into textbook regression.
- **(c) Completed-runs-only + horizon cap (sensitivity floor).** Regress on failures only up
  to a horizon τ (treat `T>τ` as `τ`), to *show* the censoring bias the others correct — a
  deliberate wrong-answer baseline that quantifies why (a)/(b) are needed (mirrors the
  failures-only demonstration from Act I).

## 4. TTF from regression (no distribution assumed)

Target = `ttf_mix` (operating days to failure), censoring handled per §3.

- **Quantile regression (primary):** train three CatBoost models with **pinball loss** at
  τ = 0.1 / 0.5 / 0.9 (IPCW-weighted) → **B10 / B50 / B90 directly** as regression outputs.
  No survival curve, no Weibull — the quantiles are read straight off the learners.
- **Point/RMST (secondary):** an RMSE model (or RMST pseudo-value model) → expected life /
  restricted mean life to a horizon.
- Monotonicity (B10 ≤ B50 ≤ B90) enforced by post-hoc sorting of the three quantile
  predictions (standard quantile-crossing fix).

"TTF" = these predicted quantiles for a *new* pump (age 0), the same B10/B50/B90 the
survival stack reports — but obtained by regression, not from `S(t)`.

## 5. RUL from regression (landmark/conditional — the pure-regression analogue)

RUL is **not** derived from a survival curve here. Instead, **landmark conditional
regression**:

- Choose landmark operating-ages `L ∈ {0, 30, 90, 180, 365, …}`. At each `L`, keep pumps
  still running at `L`; **target = residual life `T_i − L`** (censored ⇒ residual censored ⇒
  IPCW/pseudo-value within the landmark set).
- Features = the verified hazards **+ current operating-age `L`** (and, if desired, the
  Phase-D early-window state — but Phase D showed dynamics don't transport, so default to
  t0/early only).
- Pinball-loss CatBoost → **RUL B10/B50/B90 for a running pump directly**: `RUL_p(a) =`
  model output at features(x, age=a). One model trained across landmarks (age as a feature)
  or one per landmark (sensitivity).

This is a genuinely different object from the survival RUL: it regresses remaining life
against covariates-and-age, rather than integrating a hazard. It needs no `S(t)`, no `η/β`.

**Operating → calendar:** RUL and TTF come out in operating days; convert to a wall-clock
maintenance date with the stratum uptime factor `u_s` (`RUL_cal ≈ RUL_op / u_s`), same
guardrail as VBA v2 §0.1. Every output carries an `_op_d` / `_cal_d` label.

## 6. Design specifics

- Data builder: `build_competing_risks_df(tte_col="ttf_mix")` + equipment + reconciled
  restarts; full population; `well_key`, `install_date` retained.
- Features: the §2-kept verified-hazard set; categoricals (`field`, `contractor`,
  `h2s_class`, `pump_gabarit`) native to CatBoost; keep `*_missing` flags.
- Models: CatBoost regressors, pinball (τ=.1/.5/.9) + one RMSE/RMST; IPCW weights from
  per-stratum censoring KM; modest depth (4–6), early-stopping on a temporal validation fold.
- Strata: **as categorical features only** — deliberately *not* a baseline offset (that would
  re-couple it to the survival stack; independence is the point).
- Optional per-mode: separate TTF/RUL regressions with the competing event as censoring
  (cause-specific residual life) — the regression analogue of Phase B.

## 7. Validation — the C5 gate, and the honest-null expectation

- **Gate:** temporal install-cohort hold-out (train ≤ 2023-12-31, test 2024+; also the 2022
  cutoff), well-clustered. This is the whole point — a random split leaks vintage.
- **Regression-appropriate metrics** (no survival curve needed): IPCW-weighted MAE / pinball
  loss on the test failures; concordance of predicted TTF ordering (rank agreement with
  observed times, censoring-aware); quantile calibration (does B50 cover ~50%?). Compare
  head-to-head against the deployed stratum-baseline predictions on the **same** test runs —
  the triangulation.
- **Prior expectation (state it up front):** linear θ (C5) and dynamic landmarks (Phase D)
  both failed this gate at ≈ chance. A regressor on the same verified hazards will *probably*
  do the same — and that is a legitimate, valuable result: an orthogonal method confirming
  the fleet doesn't forecast from covariates. If instead the flexible regressor **beats**
  the baseline out-of-sample, it means nonlinearity/interactions the linear stack couldn't
  see do transport — the one way this project's forecasting could still improve. Either
  outcome is reportable; in-sample fit (the old ~0.86 C-index) is diagnostic-only.

## 8. Deliverables, order, risks, non-goals

**Order:** (1) data/feature builder (kept features only, ttf_mix, full pop); (2) per-stratum
censoring-KM + IPCW weights; (3) quantile-regression TTF (B10/50/90) + calibration;
(4) landmark conditional RUL regression; (5) C5 temporal hold-out harness + baseline
comparison; (6) pseudo-observation cross-check; (7) optional per-mode; (8) report with the
triangulation verdict.

**Deliverables:** `backend/analysis/models/ml/regression_ttf.py` (IPCW/pseudo targets,
quantile fit, TTF/RUL predict), thin `scripts/run/catboost_regression_v2.py`,
`results/catboost_regression_v2/…` (hold-out metrics vs baseline, calibration, quantile
examples, feature-importance/interaction audit), tests (IPCW weight correctness on a
synthetic censored set, quantile monotonicity, landmark-target censoring correctness).

**Risks:** IPCW instability when `Ĝ(t)` is small in long tails (cap weights / truncate
horizon); quantile crossing (post-sort); tree extrapolation beyond observed age (cap RUL at
training support, flag); small 2024+ test cohort (CIs, both cutoffs).

**Non-goals:** replacing the deployed VBA baseline; any Weibull/survival machinery; feeding
rejected covariates; whole-run features; calendar clock. This model stands or falls on its
own C5 result as an independent check.

---

## 9. One-line summary

A Weibull-free CatBoost **regressor** that predicts **TTF as pinball-loss quantiles
(B10/B50/B90)** and **RUL as landmark conditional-residual-life quantiles**, uses the full
population on `ttf_mix` with censoring handled by **IPCW / pseudo-observations** (not a
survival distribution), is fed only the **verified hazards**, and is judged solely on the
**C5 temporal hold-out against the stratum baseline** — an independent triangulation whose
most likely (and still valuable) result is to confirm the covariate transport failure, with
an upside chance of finding transportable nonlinear structure the linear stack missed.
