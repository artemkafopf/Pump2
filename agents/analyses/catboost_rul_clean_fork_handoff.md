# Clean CatBoost RUL Fork — Agent Handoff

**Read this fully before writing code.** This document is self-contained: it assumes you have
NO access to the parent project's chat history or memory. It defines a *clean* rebuild of the
ESP (УЭЦН) pump failure-forecast model, deliberately dropping the contaminated parts of the
existing WOWPUMP tool.

---

## 1. Purpose

Build a **censoring-aware conditional Remaining-Useful-Life (RUL) model** for ESP pumps and
benchmark it against the incumbent survival (Weibull/KM) model. **Offline / repo-only first** —
do not wire into any launcher, EXE, or web service until the benchmark says go.

## 2. Why the current system is contaminated — do NOT carry these over

1. **Censoring-blind training.** The current model is a plain `CatBoostRegressor(loss=RMSE)`
   regressing ННО = «Наработка (сут)» with **no event indicator**. But ~**67 % of the failure
   database is censored** (running pumps + workover pulls). Treating a censored duration as a
   true failure time is simply wrong for the majority of rows.
2. **`TTF − age` as RUL.** RUL is computed as `predicted_total_TTF − current_age`. This is
   **mis-specified for this hazard shape**: ESP hazard here is *infant-spike-then-flat* (≈
   memoryless after the infant period), so true mean-residual-life is **flat or rising with
   age**. `TTF − age` forces RUL to fall 1:1 with age and wrongly flags old survivors as
   "overdue" — the safest pumps get the worst score.
3. **Probabilistic empirical-tail patch.** When `predicted_TTF ≤ current_runtime`, the value is
   discarded and replaced by a resample from the empirical runtime tail above current age
   (the «Вероятностный прогноз» column, `Использовано` flag). This is a band-aid for defects
   1+2. **The clean fork must NOT include any tail resampling.** A correct conditional
   `RUL(age) ≥ 0` by construction, so the patch is unnecessary.

## 3. Architecture — learn on Свод, calculate on ТМ-06

- **Свод** (`Отказы свод с анализом`) = failure-history database = **LEARNING substrate**. It is
  the only place with ground-truth outcomes. **Must include running pumps as right-censored
  rows** (historical ingest dropped them; putting them back is the whole point).
- **ТМ-06** (production plan, **holds 2024-current data**) = the **forward PROGNOSIS frame**.
  RUL is *realized* by applying the Свод-trained model **per well, per future date**, using the
  planned covariates + the well's **projected op-age at that date**.
- **Consequence:** accuracy is validated **only on Свод** (historical, has ground truth). ТМ-06
  is the future — no ground truth — so it gets **consistency checks only**, never accuracy.

## 4. Data

| Thing | Location / column |
|---|---|
| Свод, all runs (incl. censored) | `data/inputs/Отказы свод с анализом_БДА_V03_all.xlsx` (also `d:\Projects\Pumps\data\target\…`) |
| Свод, failures only | `data/inputs/Отказы свод с анализом_БДА_V03_failures.xlsx` |
| Duration / target | column **«Наработка (сут)»** = op-day runtime = `tte` |
| Event indicator | column **«Признак отказа» / Failure Flag** |
| ТМ-06 plan | `d:\Projects\Pumps\data\pp\ТМ-06_2026_2027_Р50_Базовый_Мастер файл.xlsx` — MAP sheet keyed on `Well_ID*`; Свод sheet = wide future-date production matrix; **no ННО/failure columns** (features must be baseline-filled) |
| Manual | `c:\Users\alexe\Downloads\WOWPUMP_User_Guide_RealScreens.docx` |

Censoring composition (V03_all, 4728 rows): **1537 failures (Flag=1)** vs **3191 censored
(Flag=0/-1) = 67 %**. Handle the `-1` flag explicitly. Мирнинский (Mc/Mr) is event-thin:
~38 failures vs ~74 censored.

Note the Excel files are cp1251/Russian; column headers often read as mojibake — decode with
`str.encode('latin1').decode('cp1251')` and index sheets/columns positionally when needed.

## 5. Two estimands — build BOTH

- **All-cause** (event = any run termination: failure + workover) → schedule / МРП question.
- **Cause-specific ESP failure** (event = genuine ESP failure; workover → censored) →
  reliability KPI.

## 6. Method (recommended)

**Landmark conditional RUL + IPCW + quantile CatBoost:**

1. For each landmark age `L`: take runs still alive at `L`, set `age_op_d = L`,
   target `rul_target_op_d = tte − L`. **Leakage guard:** never feed `tte`, `event`,
   `run_days`, or `*_whole_run` as features.
2. **IPCW** censoring weights (conditional `G(L)/G(T)`, KM of the censoring distribution,
   stratified) so censored runs contribute survival information instead of being dropped or
   mistaken for failures.
3. Fit **quantile** CatBoost at α = 0.1 / 0.5 / 0.9 → RUL *distribution* (b10/b50/b90), not a
   point.
4. Convert op-days ↔ calendar via a per-stratum uptime factor.
5. **Apply to ТМ-06:** `predict_rul(covariates_from_TM06_row, age = projected_op_age_at_date)`
   for each well × future date. ТМ-06 must carry (or baseline-fill) the model's features **plus
   a per-well current-age / as-of anchor** — the age anchor is what makes RUL conditional.

**Reference implementation to mirror** (in the parent repo — port the *design*, drop everything
tail-related): `backend/analysis/models/ml/regression_ttf.py :: RULRegressor`
(`make_landmark_frame`, `conditional_ipcw_weights`, `predict_rul`, `to_calendar`).

## 7. Evaluation — censoring-aware, built to detect a NULL honestly

- **Cross-fit, WELL-CLUSTERED folds** — a well's runs must never span train/test (use
  GroupKFold on `well_key`), else the model memorises wells.
- **Metrics:** Uno/Harrell **C-index**, **IPCW Integrated Brier Score**, and **calibration of
  predicted MRL vs KM-MRL** by stratum × age band. Never score RMSE on failures only.
- **Report MRL(age) and RMST(0) — never median / B50.**
- **Baselines to beat:** age-only MRL, stratum-only MRL, incumbent `TTF − age`, and the
  survival (Weibull/KM) model on the *same* population/folds.
- **Win condition (state up front):** the CatBoost RUL wins only if it beats the survival model
  out-of-fold on C-index/IBS **and** is calibrated. Otherwise **record the null.** Strong prior
  from prior rounds: age + stratum is hard to beat and CatBoost covariate lift has repeatedly
  died on replication — so a null is a real and acceptable outcome; design for it.

## 8. Project-specific scars — respect these

- **Мирнинский (Mc/Mr) = installs on/after 2024-01-01 ONLY.** This is a **cohort filter on the
  install date**, NOT left-truncation of exposure (left-truncation hides that recent runs are
  shorter and was tried and is wrong). Mc is event-thin → its verdict is a confidence statement,
  not a point claim.
- **ALWAYS assert fit `n_failures` per estimand against the population count** before trusting
  any result — a silent population/join gap has repeatedly inflated headline numbers here.
- **Run support-overlap checks before any cross-field covariate claim** — several "covariates"
  (bubble pressure, nominal frequency) are field labels in disguise; cross-field transfer looks
  real on one field and dies on replication. Always replicate.
- **Informative-censoring / competing-risk caveat:** for the cause-specific estimand, workover
  censoring may not be independent of failure risk. IPCW assumes conditionally-independent
  censoring — flag the sensitivity, don't hide it.

## 9. Scope of the first pass

Offline benchmark only. Deliverable = `results/<slug>/` tables + figures (MRL-vs-age curves for
CatBoost / survival / baselines; calibration plots) + a short **go/no-go note** on whether to
replace the incumbent `TTF − age` step. No production, launcher, or web-app changes in this pass.
