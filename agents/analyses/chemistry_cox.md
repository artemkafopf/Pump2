# Chemistry Block — Adaptive Extended Cox Analysis

## Purpose

Quantify the effect of **fluid chemistry, water cut, and GLF** on ESP failure hazard.
These are environmental / fluid parameters the operator does not control.

This is **Block 1** of three planned Cox analyses:
- **Block 1 (this file):** Chemistry + water cut + GLF — what the reservoir gives you
- **Block 2 (later):** Operational parameters — what the operator sets (frequency, load, BEP deviation, p_bot)
- **Block 3 (later):** Completion parameters — what was designed (stages, depth, nominal flow)

Blocks will be merged into a single joint θ(t) once each is validated independently.

---

## Covariates in scope

| Covariate | Source | Transform | Notes |
|---|---|---|---|
| `chloride_mg_l` | `proc__daily_lab` | log1p | Cl⁻ corrosion driver |
| `sulfate_mg_l` | `proc__daily_lab` | log1p | SO₄²⁻ scale/corrosion |
| `calcium_mg_l` | `proc__daily_lab` | log1p | Ca²⁺ gypsum precursor |
| `bicarbonate_mg_l` | `proc__daily_lab` | log1p | HCO₃⁻ buffer |
| `total_mineralization_g_l` | `proc__daily_lab` | log1p | overall salinity |
| `ph` | `proc__daily_lab` | as-is | water pH |
| `mechanical_impurities_mg_l` | `lab_samples` | log1p | КВЧ — abrasive wear |
| `h2s_proxy_mg_l` | `mart__weibull_input` | log1p | H₂S concentration (not class) |
| `ca_so4_product` | derived | log1p | Ca × SO₄ gypsum saturation index |
| `watercut_percent` | `lab_samples` | as-is (0–100) | water fraction |
| `glf_m_mean` | `mart__weibull_input` | log1p | gas-liquid fraction (mean over run) |

**NOT in this block:** delta_bep, frequency, motor load, p_bot, p_bubble, stages, setting depth.

---

## Stratum definition

```
stratum_key = "{field}_{h2s_class}_{contractor_group}"
```

Exactly as in the baseline K=2 Weibull models (see `agents/analyses/cox_hr_vba_integration.md`).

**Contractor mapping** (same as `ContractorGroup()` in `mdlModelRegistry.bas`):

| Raw value in mart | Code |
|---|---|
| Борец | `brt` |
| Шлюмберже | `slb` |
| Новые технологии, Новомет, ИНК, any other | `oth` |
| missing / unknown | `unk` → treat as `Pooled` |

**H2S class** — derive from `h2s_proxy_source` or `h2s_proxy_mg_l` threshold, consistent
with how the baseline models assigned `sour` / `nonsour` per run.

**3-priority lookup for small strata** (same cascade as baseline):
1. `{field}_{h2s}_{ctr}` — most specific
2. `{field}_{h2s}_Pooled` — field × H2S level pooled
3. `Global_Pooled` — fallback

For Cox fitting, all strata are included simultaneously as stratification variable.
β coefficients are **global** — one set across all strata. Strata absorb baseline hazard differences.
Strata with 0 events contribute no information and are dropped automatically by lifelines.

---

## Data pipeline

**All data from SQLite warehouse. Do not read the Excel workbook.**

```
D:\GitHub\Pump2\data\warehouse\pump2.db
    mart__weibull_input     — survival data + h2s_proxy_mg_l + glf_m_mean (2,634 runs)
    proc__daily_lab         — daily ion chemistry, all wells (883,959 rows)
    raw__v03_runs           — install_date, stop_date per run (for date windowing)

D:\GitHub\Pump2\data\sqlite\lab.sqlite
    lab_samples             — КВЧ, watercut_percent, sample_date (8,981 rows)
```

**New file:** `backend/analysis/data/chemistry_run_features.py`

Function signature:
```python
def build_chemistry_df() -> pd.DataFrame:
    """One row per run. Columns: all mart__weibull_input fields
    + run-averaged chemistry from proc__daily_lab
    + run-averaged КВЧ and watercut from lab_samples
    + derived features (ca_so4_product, log transforms, stratum_key).
    """
```

**Aggregation rules:**

For `proc__daily_lab` (high-resolution, all runs have rows):
```sql
SELECT
    r.well_key,
    AVG(dl.chloride_mg_l)            AS chloride_mg_l,
    AVG(dl.sulfate_mg_l)             AS sulfate_mg_l,
    AVG(dl.calcium_mg_l)             AS calcium_mg_l,
    AVG(dl.bicarbonate_mg_l)         AS bicarbonate_mg_l,
    AVG(dl.total_mineralization_g_l) AS total_mineralization_g_l,
    AVG(dl.ph)                       AS ph
FROM raw__v03_runs r
JOIN proc__daily_lab dl
  ON dl.well_key = r.well_key
 AND dl.dt BETWEEN r.install_date AND r.stop_date
GROUP BY r.well_key, r.install_date
```

For `lab_samples` (КВЧ, watercut — lower frequency):
```sql
SELECT
    r.well_key,
    AVG(ls.mechanical_impurities_mg_l) AS mechanical_impurities_mg_l,
    AVG(ls.watercut_percent)           AS watercut_percent
FROM raw__v03_runs r
JOIN lab_samples ls
  ON ls.well_key = r.well_key
 AND ls.sample_date BETWEEN r.install_date AND r.stop_date
GROUP BY r.well_key, r.install_date
```

Fallback for runs with no samples in window: use the nearest sample before `install_date`
within 180 days. If still missing → NaN (handled by imputation).

**Imputation** (within `build_chemistry_df`):
1. Well-level mean across all runs for the same `well_key`
2. Pad-level mean (all wells with same pad prefix)
3. `stratum_key`-level mean

If a covariate is still >50% missing within a stratum → exclude that covariate from
the model for that stratum; note in output.

**Coverage after join** (from warehouse audit, 2026-07-03):

| Covariate | Runs with data | % |
|---|---|---|
| Cl⁻, SO₄²⁻, Ca²⁺, HCO₃⁻, pH | 2,634 (all) | 100% |
| Water cut | ~2,181 | 83% |
| КВЧ | ~1,091 | 41% |
| H₂S proxy | ~2,188 | 83% |
| GLF | ~2,400 | 91% |

КВЧ at 41% coverage — include it but flag strata where it drops below 50%.

---

## Feature engineering

```python
import numpy as np

# Log transforms
LOG_COLS = [
    'chloride_mg_l', 'sulfate_mg_l', 'calcium_mg_l',
    'bicarbonate_mg_l', 'total_mineralization_g_l',
    'mechanical_impurities_mg_l', 'h2s_proxy_mg_l',
]
for col in LOG_COLS:
    df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

# Ca × SO₄ gypsum saturation index
df['ca_so4_product'] = df['calcium_mg_l'] * df['sulfate_mg_l']
df['log_ca_so4'] = np.log1p(df['ca_so4_product'].clip(lower=0))

# GLF: already a fraction
df['log_glf'] = np.log1p(df['glf_m_mean'].clip(lower=0))

# Water cut: keep as percent (0–100), linear
# (logit optional if model shows non-linearity)
```

Use log-transformed versions as primary covariates in Cox (right-skewed distributions).
Keep raw versions for KM tertile diagnostic plots.

---

## Adaptive Cox procedure

**New file:** `backend/analysis/workflows/chemistry_cox/phase_chem.py`

Run for ALL strata combined (global β), stratified Cox.

### Step 1 — Univariate screening

```python
from lifelines import CoxPHFitter

for col in candidate_cols:
    sub = df[['run_days', 'event', 'stratum_key', 'well_key', col]].dropna()
    if sub['event'].sum() < 15:
        continue
    cph = CoxPHFitter()
    cph.fit(sub, duration_col='run_days', event_col='event',
            strata=['stratum_key'], cluster_col='well_key',
            formula=col, robust=True)
    # Record: HR, CI, p, C-index
    # Keep if p < 0.10
```

### Step 2 — PH assumption test (Schoenfeld residuals)

```python
from lifelines.statistics import proportional_hazard_test

for col in univariate_survivors:
    result = proportional_hazard_test(fitted_cph, sub, time_transform='log')
    p_schoenfeld = result.summary['p'].iloc[0]

    if p_schoenfeld > 0.05:
        assignment = 'STANDARD'   # β·X term
    else:
        assignment = 'EXTENDED'   # β·X + γ·X·log(t) term pair
```

### Step 3 — Correlation screen

Drop one from any pair with Spearman |ρ| > 0.65.
Keep the covariate with higher univariate C-index.

`ca_so4_product` vs `calcium_mg_l` + `sulfate_mg_l` separately — include all three;
the joint model will resolve redundancy via VIF.

### Step 4 — Joint multivariate model

```python
# Build interaction terms for EXTENDED covariates
for col in extended_covs:
    df[f'{col}_x_logt'] = df[col] * np.log(df['run_days'].clip(lower=1))

formula_parts = (
    standard_covs +
    [f'{c} + {c}_x_logt' for c in extended_covs]
)
formula = ' + '.join(formula_parts)

cph_joint = CoxPHFitter()
cph_joint.fit(df, duration_col='run_days', event_col='event',
              strata=['stratum_key'], cluster_col='well_key',
              formula=formula, robust=True)
```

### Step 5 — VIF check and cleanup

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Drop one from any pair with VIF > 5 (keep higher C-index)
# Refit final model
```

### Step 6 — Final model output

For standard covariate X:
```
θ contribution = exp(β · (X − X_ref))
```

For extended covariate X:
```
θ(t) contribution = exp((β + γ · log(t)) · (X − X_ref))
```

Combined log-hazard multiplier:
```
log θ(t) = Σ_standard  β_i · (X_i − ref_i)
          + Σ_extended  (β_j + γ_j · log(t)) · (X_j − ref_j)
```

Reference values `X_ref` = population-weighted mean across all strata (same approach as
operational block, so that θ=1 for an average pump).

---

## Output

`results_dir("chemistry_cox")` → `tables/` and `figures/`

| File | Content |
|---|---|
| `chem_univariate.csv` | covariate, HR, CI\_lo, CI\_hi, p, C\_index, pass\_screen |
| `chem_ph_test.csv` | covariate, schoenfeld\_stat, p\_ph, assignment (standard/extended/excluded) |
| `chem_joint_model.csv` | term, coef (β or γ), se, z, p, exp\_coef |
| `chem_vif.csv` | term, VIF |
| `chem_final_coeffs.csv` | covariate, β, γ (0 if standard), ref\_value, type |
| `chem_km_tertile_*.png` | KM curves by tertile for each surviving covariate |
| `chem_forest_beta.png` | Forest plot: β ± 95% CI for all surviving covariates |
| `chem_forest_gamma.png` | Forest plot: γ ± 95% CI for extended covariates only |

`chem_final_coeffs.csv` is the handoff artifact for the VBA integration step.

---

## VBA implications (to be resolved after fitting)

The current `mdlCoxHR.bas` computes a **time-independent** θ:
```
θ = exp(Σ β_i · (x_i − ref_i))
```

If any chemistry covariate ends up as EXTENDED (time-varying HR), VBA needs:
```
θ(t) = exp(Σ_standard β_i·(x_i−ref_i) + Σ_extended (β_j + γ_j·log(t))·(x_j−ref_j))
```

This makes θ a function of t. The `ApplyCoxEta` call must move inside the S(t) loop.
**Do not modify VBA until fitting is complete** and it's clear which covariates are extended.
If γ is negligible for all surviving covariates, the current architecture is sufficient.

---

## Implementation order

1. `build_chemistry_df()` — join mart + proc\_\_daily\_lab + lab\_samples, impute, derive features
2. Sanity check coverage and distributions (print n, mean, std, pct\_missing per covariate)
3. KM tertile plots for all candidates (diagnostic, already partially done 2026-07-03)
4. Step 1: univariate Cox screening
5. Step 2: Schoenfeld PH test per survivor
6. Step 3: correlation screen
7. Step 4–5: joint model + VIF cleanup
8. Step 6: output tables and forest plots
9. Review `chem_final_coeffs.csv` before any VBA work

---

## Files to read before implementing

```
agents/analyses/cox_hr_vba_integration.md   — stratum table, VBA architecture, known bugs
CLAUDE.md                                   — path rules (results_dir, no hardcoded paths)
backend/analysis/data/                      — put build_chemistry_df() here
backend/analysis/workflows/chemistry_cox/   — put phase_chem.py here (create dir)
backend/analysis/workflows/esp_survival/data.py  — reference for imputation pattern
```
