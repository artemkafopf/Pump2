# Task Briefing: Extended Cox HR Integration into VBA Prediction System

## Context

A fully working Excel VBA prediction system exists at:
`D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx` (saved as `.xlsm`)

The system uses **K=2 latent Weibull mixture** baseline survival models per stratum.
VBA modules live in `D:\GitHub\Pump2\vba\`.

---

## What already exists

### Baseline VBA system (working)

**Stratum key format:** `{field}_{h2s}_{ctr}` where:
- field: Ya, Za, Vt, Ic, Az, Mc, Da
- h2s: sour / nonsour
- ctr: brt (Borec), slb (Schlumberger), oth (others), Pooled

**3-priority lookup cascade:**
1. `{field}_{h2s}_{ctr}` — most specific
2. `{field}_{h2s}_Pooled` — field-level pooled
3. `Global_Pooled` — domain fallback

**Model parameters per stratum** (stored in `ESP_Models` sheet):
- `w1, beta1, eta1` — C1 early-failure component
- `beta2, eta2` — C2 wear-out component (w2 = 1 - w1)
- `degenerate` flag (w1 > 0.75)

---

## Current baseline stratum parameters

All times in days. E[T] = exact closed-form mean. B10/B50/B90 via bisection.
`DEGEN` = w1 > 0.75 (C1 collapses onto a single Weibull — model unreliable).

| Stratum | n | w1 | β1 | η1 | β2 | η2 | E[T] | B10 | B50 | B90 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ya_nonsour_brt | 397 | 0.195 | 1.072 | 63.8 | 1.203 | 657 | 509 | 34 | 355 | 1210 | Borec dominant |
| Ya_nonsour_slb | 261 | 0.127 | 1.067 | 33.2 | 1.077 | 465 | 399 | 24 | 270 | 954 | |
| Ya_nonsour_oth | 44 | 0.618 | 1.285 | 60.4 | 1.295 | 316 | 146 | 15 | 73 | 396 | Novomet — high w1, short life |
| Ya_nonsour_Pooled | 702 | 0.155 | 1.081 | 41.3 | 1.091 | 540 | 447 | 26 | 299 | 1081 | |
| Za_nonsour_brt | 87 | 0.320 | 1.285 | 35.4 | 1.295 | 310 | 205 | 15 | 126 | 512 | |
| Za_nonsour_slb | 33 | 0.487 | 0.900 | 104.5 | 1.600 | 464 | 267 | 20 | 184 | 638 | two-stage; only 33 failures |
| Za_nonsour_Pooled | 135 | 0.799 | 0.931 | 123.5 | 2.494 | 597 | 208 | 14 | 120 | 558 | **DEGEN** — contractor split resolves |
| Vt_sour_brt | 23 | 0.349 | 1.347 | 25.0 | 1.392 | 117 | 77 | 9 | 51 | 184 | two-stage; preset β from M3 |
| Vt_sour_slb | 33 | 0.135 | 1.347 | 9.0 | 1.392 | 124 | 99 | 8 | 81 | 215 | slb: longest B50 despite acute η1 |
| Vt_sour_oth | 33 | 0.366 | 1.347 | 13.0 | 1.392 | 99 | 62 | 5 | 37 | 154 | NT (Novye Tekhnologii) — worst |
| Vt_sour_Pooled | 89 | 0.258 | 1.347 | 14.4 | 1.392 | 114 | 80 | 7 | 58 | 187 | |
| Vt_nonsour_brt | 65 | 0.333 | 0.974 | 63.0 | 1.139 | 334 | 234 | 16 | 139 | 586 | |
| Vt_nonsour_slb | 84 | 0.227 | 1.250 | 18.0 | 1.260 | 254 | 186 | 10 | 131 | 448 | |
| Vt_nonsour_Pooled | 167 | 0.944 | 0.871 | 168.6 | 9.416 | 533 | 199 | 14 | 122 | 506 | **DEGEN** — contractor split resolves |
| Ic_nonsour_Pooled | 141 | 0.254 | 1.209 | 37.0 | 1.413 | 519 | 361 | 19 | 271 | 850 | no contractor split |
| Az_nonsour_Pooled | 143 | 0.528 | 0.992 | 67.0 | 1.864 | 480 | 236 | 14 | 135 | 607 | no contractor split |
| Mc_nonsour_Pooled | 38 | 0.046 | 1.145 | 18.2 | 1.403 | 254 | 222 | 36 | 186 | 454 | tiny w1; essentially single Weibull |
| Da_nonsour_Pooled | 31 | 0.153 | 1.145 | 52.5 | 1.403 | 735 | 575 | 43 | 465 | 1262 | longest-lived field |
| Global_Pooled | — | 0.200 | 1.100 | 41.0 | 1.400 | 500 | 372 | 25 | 292 | 843 | domain-knowledge fallback |

### Key observations for Cox HR modelling

- **B50 range**: 37d (Vt_sour_oth) to 465d (Da_nonsour_Pooled) — 12× spread
- **Degenerate strata**: Za_nonsour_Pooled and Vt_nonsour_Pooled. Cox covariates that correlate with contractor assignment may partially explain the degeneracy.
- **Vt_sour β1 ≈ β2 ≈ 1.35**: Both components have similar shapes; only η differs. Cox rescaling of η is sufficient to shift survival curves.
- **Ya_nonsour_oth (Novomet)**: w1 = 0.618, B50 = 73d — extreme outlier vs brt (B50=355d). Any Cox model should include contractor or at least flag this sub-population.
- **Mc, Da**: two-stage fits with only 31–38 failures. Cox HR adjustment here is dominated by prior uncertainty; treat with caution.

**Working UDFs:** `ESP_B50`, `ESP_B10`, `ESP_B90`, `ESP_RUL`, `ESP_SF`, `ESP_TTF_Mean`,
`ESP_Hazard`, `ESP_Component`, `ESP_Stratum`, `ESP_IsDegenerate`, `ESP_Predict`, `ESP_Debug`

**VBA module files:**
```
vba/mdlMath.bas           — WeibullSF, GammaApprox, SimpsonLatentSF, BisectQuantile
vba/mdlLatentWeibull.bas  — LatentSF, LatentRUL, LatentQuantile, LatentHazard, LatentC1Posterior
vba/mdlModelRegistry.bas  — LoadModelRegistry, LookupModel, ContractorGroup, H2SClass, CreateModelSheet
vba/mdlPublicFunctions.bas — all 12 worksheet UDFs
vba/mdlBatchProcess.bas   — RunPredictions, ImportModelCSV, ExportSurvivalCurve
```

**Critical VBA lessons already learned** (do not repeat these bugs):
- Use `ChrW()` not `Chr()` for Unicode code points > 255
- `mLoaded`, `mCount`, `mModels` are Private to `mdlModelRegistry` — never access them from other modules; use `EnsureRegistryLoaded()` instead
- Private functions cannot be called cross-module with `Option Explicit`
- VBA is case-insensitive: local var `h2sCls` ≠ function `H2SClass()` OK, but `h2sClass` would collide
- Never call `MsgBox` from code triggered by a UDF — causes `#VALUE!` in worksheet

### Python pipeline (baseline models)

Baseline K=2 models fitted via EM algorithm.
- Phase 1: `backend/analysis/workflows/esp_survival/phase1_weibull.py`
- Phase 2: `backend/analysis/workflows/esp_survival/phase2_mixture.py`
- Results slug: `esp_survival_phase2_mixture`
- Contractor sub-strata: `scripts/run/vt_contractor_substrata.py`, `scripts/run/field_contractor_substrata.py`

### Extended Cox HR scripts (existing — read these first)

Before writing any code, **read all existing Cox HR scripts** in the repo:
```bash
find D:/GitHub/Pump2 -name "*cox*" -o -name "*Cox*" | sort
```
Understand:
- What covariates are used (fluid chemistry, operational parameters, pump type, frequency, etc.)
- What the output is (β coefficients, per-covariate hazard ratios)
- What stratification is used (same strata as baseline, or different?)
- Whether the model is semi-parametric (Breslow baseline) or parametric (Weibull baseline)

---

## Mathematical formulation for integration

### Cox PH with K=2 latent Weibull baseline

Under proportional hazards, the individual hazard multiplier is:

```
theta(x) = exp(beta · x)   where x is the covariate vector for one pump
```

For the K=2 mixture, apply the Cox multiplier by rescaling each component's scale parameter:

```
eta1_eff(x) = eta1 / theta^(1/beta1)
eta2_eff(x) = eta2 / theta^(1/beta2)
```

The mixture survival function becomes:

```
S(t | x) = w1 * exp(-(t / eta1_eff)^beta1) + w2 * exp(-(t / eta2_eff)^beta2)
```

**Shape parameters beta1, beta2 and mixing weight w1 are unchanged.**
Only the eta values are scaled. This means **all existing VBA math functions work unchanged** —
only the eta inputs passed to them change.

### Why this formulation

Under Weibull baseline `h_k(t) = (beta_k/eta_k) * (t/eta_k)^(beta_k-1)`:

```
h_k(t | x) = h_k(t) * theta
S_k(t | x) = exp(-theta * (t/eta_k)^beta_k)
            = exp(-(t / (eta_k * theta^(-1/beta_k)))^beta_k)
            = exp(-(t / eta_k_eff)^beta_k)
```

So the Cox adjustment is a pure eta rescaling, which is exact (not approximate).

### Handling theta > 1 vs theta < 1

- `theta > 1` (higher risk than baseline): eta_eff decreases → shorter expected life
- `theta < 1` (lower risk): eta_eff increases → longer expected life
- `theta = 1` (average covariate profile): baseline model unchanged

---

## Resolved design decisions (from Phase 5e analysis, 2026-07-02)

### Covariate selection — final answer

3 covariates → **Option A (individual function arguments)**

Exclusion rationale:

- `motor_load`: Schoenfeld p=2×10⁻²⁰ (massive PH violation), VIF=31 when paired with x·log(t) term
- `water_cut`: p=0.648 in multivariate (not significant after controlling for strata)
- `glr`: Schoenfeld violation in final model
- All chemistry variables (Cl, SO₄, Ca, pH, KVCh): not significant in multivariate

### Fitted coefficients

Stratified Cox, cluster-robust SE on `well_key`. Coefficients are global (same across all strata).
Strata = Field × H2S class (9 strata).

| Covariate | β | HR | 95% CI | p |
| --- | --- | --- | --- | --- |
| `delta_bep` = Q/Q_nom − 1 | 0.3605 | 1.434 | [1.279, 1.608] | 0.000 |
| `p_bot` = bottomhole pressure, atm | 0.000466 | 1.0005 | [1.0001, 1.0008] | 0.009 |
| `n_stages_ratio` = n_stages / q_nominal | −0.05203 | 0.949 | [0.929, 0.970] | 0.000 |

### Reference profile (theta = 1 ≡ population-mean pump)

```
delta_bep_ref      = −0.342  (most pumps run below BEP — under-loaded)
p_bot_ref          = 110.3 atm
n_stages_ratio_ref =   2.249
```

### theta formula (centered — theta=1 for average pump)

```
theta = exp(
    0.360455 * (delta_bep      − (−0.342))  +
    0.000466 * (p_bot          −  110.3)    +
   −0.052032 * (n_stages_ratio −   2.249)
)
```

### UDF signatures — Option A

```vb
=ESP_Theta(delta_bep, p_bot, n_stages_ratio)
=ESP_B50_Cox(field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
=ESP_RUL_Cox(age_days, field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
```

Original `ESP_B50`, `ESP_RUL` etc. remain unchanged for backward compatibility.

Source file: `results/esp_survival_phase5_cox/<date>/tables/cox_coefficients.csv`

### Step 2 — DONE: Python already exports coefficients

`cox_coefficients.csv` written by `phase5_extended_cox.py` step 5e.
No refitting of K=2 mixture needed — Cox is fitted stratified on same strata as mixture,
so the baseline mixture η values correspond to the mean covariate profile.

### Step 3 — DONE: cox_coefficients.csv in results slug

### Step 4: VBA — new sheet `ESP_CoxCoeffs`

Add a new sheet (alongside `ESP_Models`) with columns:
```
covariate | beta
```

Create a new sub `CreateCoxSheet()` in `mdlModelRegistry.bas` that hardcodes
the fitted coefficients (like `CreateModelSheet` does for baseline params).

Add to `LoadModelRegistry` (or a separate `LoadCoxRegistry`):
- Read `ESP_CoxCoeffs` into a module-level array `mCoxCoeffs()`
- Expose `ComputeTheta(covarVector)` as a Public Function

### Step 5: VBA — `mdlCoxHR.bas` (new module)

```vb
' mdlCoxHR.bas -- Cox proportional hazards covariate adjustment.
'
' ComputeTheta(x()) -> theta = exp(sum(beta_i * x_i))
' ApplyCoxToEta(eta, beta_shape, theta) -> eta_eff = eta / theta^(1/beta_shape)
```

Key function:
```vb
Public Function ComputeTheta(ByVal pumpFreqHz As Double, _
                              ByVal h2sConc As Double, _
                              ' ... one arg per covariate
                              ) As Double
    ' Load coefficients from mCoxCoeffs array
    ' Return exp(beta . x)
End Function
```

### Step 6: VBA — update public UDFs

Two options — pick one based on number of covariates:

**Option A (few covariates, <5):** Add covariate args directly to UDFs
```vb
=ESP_B50_Cox(A2, BW2, D2, freq_hz, h2s_ppm, motor_kw)
```
Keep original `ESP_B50` unchanged for backward compatibility.

**Option B (many covariates):** Covariate range argument
```vb
=ESP_B50_Cox(A2, BW2, D2, CV2:CZ2)   ' CV:CZ = covariate columns
```
The UDF reads a row range and looks up which columns correspond to which beta.

Recommended: add a `ESP_Theta(cov_range)` UDF that returns theta for inspection,
then `ESP_B50_Cox` and `ESP_RUL_Cox` that call it internally.

### Step 7: VBA — update `RunPredictions` macro

Add Cox-adjusted columns after existing ones:
```
CI  ESP_Theta     -- individual hazard multiplier
CJ  ESP_B50_Cox   -- Cox-adjusted median TTF
CK  ESP_RUL_Cox   -- Cox-adjusted RUL
```

Read covariate values from their columns inside the row loop,
compute theta, then pass effective etas to the math functions.

---

## Column mapping in the data sheet

The sheet `Свод` (hard-coded in VBA as `ChrW(1057)&ChrW(1074)&ChrW(1086)&ChrW(1076)`)
already has 77 columns. New covariate columns need to be identified.

Before implementing, verify which columns hold each covariate:
```python
import pandas as pd
df = pd.read_excel(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx",
                   sheet_name=0, nrows=2)
print(list(enumerate(df.columns)))
```

Map each Cox covariate to its Excel column letter. Add these as constants in
`mdlBatchProcess.bas` alongside `COL_FIELD`, `COL_AGE`, etc.

---

## Implementation order

1. Read existing Cox scripts → document covariate list and beta values
2. Decide on refitting strategy (joint vs. two-stage)
3. Python: produce `cox_coefficients.csv`
4. VBA: `mdlCoxHR.bas` + `CreateCoxSheet()` + `LoadCoxRegistry()`
5. VBA: `ESP_Theta()` UDF — verify values are sensible for a few pumps
6. VBA: `ESP_B50_Cox()`, `ESP_RUL_Cox()` UDFs
7. VBA: extend `RunPredictions` with Cox columns
8. Validate: for a pump with all-average covariates, Cox UDFs should equal baseline UDFs

---

## Files to read before starting

```
D:\GitHub\Pump2\vba\mdlModelRegistry.bas       -- existing registry pattern to follow
D:\GitHub\Pump2\vba\mdlPublicFunctions.bas     -- existing UDF pattern
D:\GitHub\Pump2\vba\mdlBatchProcess.bas        -- existing batch macro pattern
D:\GitHub\Pump2\CLAUDE.md                      -- project rules (paths, structure)
C:\Users\alexe\.claude\projects\d--GitHub-Pump2\memory\feedback_vba_encoding.md  -- VBA bug list
```

Then find and read all existing Cox HR scripts before writing any new code.
