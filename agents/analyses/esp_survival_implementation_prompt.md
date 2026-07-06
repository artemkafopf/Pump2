# ESP Failure Survival Analysis — Full Implementation Prompt

## Project Overview

Build a stratified latent-mixture Weibull survival model for Electric Submersible Pump (ESP) failures
across multiple oilfields, contractors, and reservoir conditions. The model must:

1. Separate early-failure and wear-out modes (latent K=2 Weibull mixture) per stratum — because
   single-Weibull fits yield β ≈ 0.8–1.0 for most strata, producing implausibly increasing RUL
   estimates as pumps age.
2. Quantify the effect of operational and chemical parameters on top of the stratum baseline via
   Extended Cox regression.
3. Produce an individual RUL estimate for any pump given its field, H2S class, contractor, and
   operating parameters.

---

## Data Source

**Primary file:** `Отказы свод с анализом_БДА_В03_failures.xlsx` — 1,573 records, 77 columns.

**Key columns:**

| Column | Description |
|---|---|
| `Месторождение` | Field (Ya=748, Vt=274, Ic=155, Az=153, Za=147, Mc=41, Da=33, smaller fields negligible) |
| `Принадлежность` | Contractor (Борец=864, Шлюмберже=569, Новые технологии=94, Новомет=43) |
| `Кислый/Некислый` | H2S classification (Некислый=1482, Кислый=91) |
| `Наработка (сут)` | Time-to-event in days |
| `Failure Flag` | 1 = failure (n=1537), −1 = censored (n=36) |
| `Скв.` | Well identifier (for cluster-robust SE) |
| `Куст` | Pad identifier (for pad-level imputation) |

---

## Stratification Design

### Key finding: all sour wells are in Vt field

**All 91 sour records come exclusively from Vt field** (91/274 Vt runs = 33%). No other field has
any H2S exposure. This simplifies the architecture:

- `Sour stratum` = `Vt sour` (91 failures, 0 censored — 100% failure rate)
- `Vt non-sour` is a separate stratum (178 failures)
- All other fields are non-sour by definition — no H2S stratification needed outside Vt

Median TTE: Vt sour = 62 days vs Vt non-sour = 110 days (1.8× faster failure).

### H2S as a mandatory stratum — NOT a Cox covariate

H2S classification changes the Weibull mixture structure at three levels simultaneously:
- Shape parameters β₁ and β₂ (early and wear modes)
- Scale parameters η₁ and η₂
- Mixing weights w₁ and w₂

From the existing M2 analysis:

| | β₁ | η₁ | w₁ | β₂ | η₂ |
|---|---|---|---|---|---|
| Non-sour | 0.683 | 79d | 0.44 | 1.276 | 350d |
| Sour (Vt) | 1.000 | 13d | 0.33 | 1.722 | 129d |

Cox PH can shift η (scale); Extended Cox with X·log(t) can shift effective β (shape); neither can
shift mixture weights w. Therefore H2S must be a stratum.

### Vt sour failure mechanism (from data)

Dominant failure nodes: ПЭД (motor) = 35 (38%), ЭЦН (pump) = 22 (24%), cable = 15 (16%).
Dominant failure character: inter-winding short circuit (Межвитковое замыкание) = 28 cases —
classic H2S chemical attack on motor winding insulation.
Confirmed H2S cause: 42/91 = 46% explicitly attributed to `Сероводород` in `Причина отказа УЭЦН`.

This confirms two physically distinct early-failure mechanisms within Vt sour:
- **H2S chemical attack** on motor windings (fast, β₁ ≈ 1.0, η₁ ≈ 13d)
- **Mechanical/abrasive wear** compounded by sour environment (slower, β₂ ≈ 1.7, η₂ ≈ 129d)

### Primary strata

```
Vt sour        91 failures   ← detailed pilot model (see Phase 2a below)
Vt non-sour   178 failures   ← full mixture model
Ya non-sour   733 failures   ← largest stratum, primary validation anchor
Az non-sour   145 failures   ← sufficient for mixture
Za non-sour   145 failures   ← sufficient for mixture
Ic non-sour   150 failures   ← sufficient for mixture
Mc non-sour    41 failures   ← single Weibull only; borrow shapes from global
Da non-sour    32 failures   ← single Weibull only; borrow shapes from global
Other fields   <10 failures  ← exclude or pool into "Other"
```

Minimum threshold for fitting a K=2 mixture independently: **≥ 40 observed failures**.
Strata with 20–40 failures: two-stage approach (borrow global shapes, fit local scales/weights).
Strata below 20 failures: single Weibull or Cox covariate only.

### Contractor

**Within Vt sour, contractor effect is substantial** (from data):

| Contractor | n | Median TTE |
|---|---|---|
| Шлюмберже | 34 | 72d |
| Борец | 23 | 48d |
| Новые технологии | 34 | 34d |

Шлюмберже median TTE is 2× that of Новые технологии in the same sour environment. Evaluate as
Cox covariate first; if Schoenfeld p < 0.05 or w₁ differs > 10 pp across contractors, promote
to stratum within Vt sour analysis.

---

## Implementation Phases

### Phase 0 — Data Audit and Stratum Viability

**Goal:** confirm stratum counts, covariate coverage, and imputation feasibility.

Stratum counts are already known (see Stratification Design above). Phase 0 focuses on:

```python
# 1. Per-stratum covariate coverage report
# For each stratum × covariate: % present, % imputable from pad, % field-fallback only

# 2. Pad-level imputation feasibility
# For each pad (Куст): how many wells? Are chemistry values consistent within pad?
# Flag pads with only 1 well (no within-pad averaging possible → fall back to field level)

# 3. Contractor audit within each field
# For each contractor × field: failure count, empirical B50
# Flag combinations with < 10 failures → too sparse for contractor-specific analysis
```

**Known covariate coverage for Vt (pilot field):**

| Variable | Vt sour (n=91) | Vt non-sour (n=183) |
|---|---|---|
| Motor load | 92% | 87% |
| Water cut | 95% | 90% |
| GOR (ГЖФ) | 95% | 90% |
| Bottomhole pressure (Рзаб) | 88% | 91% |
| pH | 77% | 37% |
| KVCh (solids) | 59% | 77% |
| H2S concentration | 57% | 32% |
| Frequency | 46% | 67% |
| Curvature | 49% | 44% |
| Cl⁻, SO₄²⁻, Ca²⁺, HCO₃⁻ | 45% | 38–43% |

Frequency (46% in Vt sour) requires pad-level imputation before Cox fitting.
Chemistry ions (45%) require pad-level imputation; verify pad consistency first.

---

### Phase 1 — Single Weibull MLE per Stratum

**Goal:** fast baseline diagnostics; identify strata where β < 0.9 (mixed-mode suspects).

For each viable stratum:
- Fit Kaplan-Meier
- Fit single Weibull by MLE: `S(t) = exp(-(t/η)^β)`
- Record: β, η, B10, B50, 95% CI on β
- Flag: β < 0.9 → latent mixture likely needed; β > 1.5 → may be pure wear-out

**Expected finding:** most strata will show β ≈ 0.8–1.1, confirming that single Weibull produces
near-constant or slightly decreasing hazard, which generates implausible RUL behavior for aged pumps.

---

### Phase 2 — Latent K=2 Weibull Mixture per Stratum

#### Phase 2a — Vt Pilot: Full Detailed Model

Vt is the pilot field because it is the **only field with H2S exposure** and has sufficient data
in both sour and non-sour strata to fit independent K=2 mixtures and compare them directly.

**Vt pilot deliverables:**
1. KM curves for Vt sour and Vt non-sour with 95% CI
2. K=2 mixture for each: recover (β₁, η₁, w₁, β₂, η₂, w₂) with bootstrap CI
3. Contractor effect within Vt sour: KM per contractor (Борец / Шлюмберже / Новые технологии)
4. Component assignment r_j1 per pump → cross-validate against `Причина отказа УЭЦН`:
   - Expect r_j1 > 0.7 for pumps where cause = "Сероводород" (42 cases)
   - Expect r_j1 < 0.3 for long-running pumps with wear causes
5. Extended Cox on Vt sour + Vt non-sour jointly:
   - Strata: sour / non-sour
   - Covariates from Tier 1 (operational) with best coverage: motor_load, water_cut, glr, Рзаб
   - Add H2S concentration as continuous covariate within sour stratum where available (57%)
   - Contractor as Cox covariate; check Schoenfeld
6. RUL curves for representative Vt sour pumps at t₀ = 30, 60, 90 days

Vt pilot validates the full pipeline before scaling to other fields.

**Goal:** separate early-failure (C1) and wear-out (C2) modes; recover physically meaningful β₂ > 1.

**Model:**
```
S(t) = w₁ · exp(-(t/η₁)^β₁) + w₂ · exp(-(t/η₂)^β₂)
w₁ + w₂ = 1
```

**Fitting procedure:**
- Fit weighted least-squares to KM curve with blended-tail weighting (upweight sparse tail)
- Hard constraints: β₁ < 1.0, β₂ > 1.0 (encode physical prior: C1 = infant mortality,
  C2 = wear-out / aging)
- Use ≥ 8 random initializations to avoid local optima
- Label-switching fix: enforce η₁ < η₂ (C1 always has shorter characteristic life)

**Two-stage approach for data-sparse strata (< 20 failures):**
- Stage 1: fit global mixture on all non-sour data → obtain global (β₁_global, β₂_global)
- Stage 2: for sparse stratum, fix shapes at global values; fit only (w₁, η₁, η₂) locally
- This is partial pooling without full hierarchical model — adequate for most sparse strata

**Validation per stratum:**
- Visual: mixture curve vs KM (should lie within KM 95% CI band)
- AIC/BIC: mixture vs single Weibull (expect ΔAIC > 4 to justify extra parameters)
- Bootstrap CI (50–100 resamples) on β₁, β₂, w₁ — check stability
- Posterior component membership r_j1 per pump (from EM E-step or WLS residuals)
  → pumps with r_j1 > 0.7 are likely C1 (early failure); use for diagnostic validation
    against `Признак отказа` (Ранний / Преждевременный vs Многосуточный)

**Expected outcome:**
- C1: β₁ ≈ 0.6–1.0, η₁ ≈ 10–80 days (depending on sour/non-sour)
- C2: β₂ ≈ 1.2–1.8, η₂ ≈ 200–400 days
- Increasing RUL correctly emerges from C2 component for surviving pumps past ~100 days
- w₁ (early-failure fraction) varies by field — itself a quality/operational signal

---

### Phase 3 — Mixture Diagnostics and Stability

For each fitted mixture:

```python
# 1. Goodness of fit
plot_km_vs_mixture(km_curve, mixture_curve, ci_band)

# 2. Information criteria
aic_mixture vs aic_single_weibull

# 3. Bootstrap stability on key parameters
bootstrap_ci(beta1, beta2, w1, n_bootstrap=100)

# 4. Component assignment validation
# Cross-tabulate r_j1 > 0.5 against 'Признак отказа' column:
# Expect: r_j1 high for 'Ранний', 'Преждевременный'; low for 'Многосуточный'

# 5. Per-field w1 comparison — w1 should be interpretable:
# fields with poor completion practices or aggressive H2S → higher w1
```

---

### Phase 4 — Contractor Effect

**Decision rule:**

```python
# Fit Cox on pooled data with contractor as covariate (stratum = field × H2S):
cph.fit(df, 'tte', 'failed', strata=['field_h2s'], formula='contractor')
# Check: HR per contractor, Schoenfeld p-value

# If Schoenfeld p < 0.05 → contractor effect is time-varying → Extended Cox term
# If w1 differs by >10pp across contractors within same field → promote to stratum
# Otherwise → keep as standard Cox covariate
```

---

### Phase 5 — Covariate Exploration for Extended Cox

For each candidate covariate (listed in section below):

**Step 5a — Binning and KM:**
```python
# Divide variable into 3–4 quantile bins
# Fit KM per bin
# Visual check:
#   - Do curves separate? → variable has signal
#   - Do curves cross? → PH violated → Extended Cox term (X·log(t)) needed
#   - Are curves parallel? → standard Cox term sufficient
#   - U-shape? → use |X - X_opt| or spline transformation
```

**Step 5b — Correlation screening:**
```python
spearman_corr = df[continuous_candidates].corr(method='spearman')
# Flag pairs with |ρ| > 0.65 for resolution:
#   Option A: keep physically primary variable, drop secondary
#   Option B: construct composite (e.g. delta_BEP = Q_actual/Q_nominal - 1)
#   Option C: orthogonalize via OLS residuals
```

**Step 5c — Univariate Cox:**
```python
for var in candidates:
    cph_uni = CoxPHFitter().fit(df, 'tte', 'failed', formula=var, strata=['field_h2s'])
    # record: HR, CI, p-value, concordance index contribution
# Keep variables with p < 0.10 in univariate for multivariate consideration
```

**Step 5d — Multivariate Extended Cox:**
```python
from lifelines import CoxPHFitter
# Build formula with:
#   - Standard Cox terms: static variables with PH holding
#   - Extended terms: X + X:log_t for variables where KM bins crossed
formula = ('contractor + well_type + curvature + setting_depth + '
           'n_stages_ratio + execution_group + '         # static equipment
           'frequency + freq_x_logt + '                 # operational, time-varying
           'motor_load + motor_load_x_logt + '
           'water_cut + glr + delta_bep + '
           'log_cl + log_kvch + log_h2s_conc + ph + '   # chemistry
           'ca_so4_product')                             # scale index

cph = CoxPHFitter().fit(df, 'tte', 'failed',
                        strata=['field_h2s'],
                        formula=formula,
                        cluster_col='well_key',          # cluster-robust SE
                        robust=True)

# Check VIF on final set — remove if VIF > 5
# Schoenfeld test per covariate — verify extended terms resolved violations
```

---

### Phase 6 — RUL Calculator

For a pump with known (field, H2S class, contractor, operational parameters X):

```python
def compute_rul(t0, field, h2s_class, X, mixture_params, cox_model):
    """
    t0: current age in days
    X: covariate vector (frequency, motor_load, water_cut, ...)
    """
    # 1. Select baseline mixture for stratum
    w1, beta1, eta1, w2, beta2, eta2 = mixture_params[(field, h2s_class)]

    # 2. Baseline mixture survival
    def S_base(t):
        c1 = w1 * np.exp(-(t/eta1)**beta1)
        c2 = w2 * np.exp(-(t/eta2)**beta2)
        return c1 + c2

    # 3. Cox adjustment — include X·log(t) terms for time-varying covariates
    def cox_adjustment(t, X):
        log_t = np.log(t + 1e-6)
        X_extended = build_extended_X(X, log_t)   # adds freq*log_t etc.
        return np.exp(cox_model.params_ @ X_extended)

    # 4. Full conditional survival
    def S_conditional(t):
        return S_base(t) * cox_adjustment(t, X) / (S_base(t0) * cox_adjustment(t0, X))

    # 5. RUL = expected remaining life
    t_grid = np.linspace(t0, t0 + 1500, 3000)
    rul = np.trapz(np.array([S_conditional(t) for t in t_grid]), t_grid)

    return rul

# Confidence interval: bootstrap mixture params + Cox CI → Monte Carlo over RUL
```

---

## Candidate Covariates for Extended Cox

### Tier 1 — Operational (expected time-varying effects, include X·log(t))

| Variable | Column | Transform | Notes |
|---|---|---|---|
| VFD frequency | `Частота` | raw Hz | 17% missing — pad-level imputation |
| Motor load | `Загр, Двиг,` | % | 17% missing; contains '-' strings |
| Q_actual / Q_nominal | derived: `Дебит жидк.` / `Ном. Произв.` | ratio; use \|ratio−1\| for U-shape | Check BEP deviation |
| Water cut | `Обводненность` | % | 7% missing |
| Gas-liquid ratio | `ГЖФ` | log | 7% missing; log-transform for skew |
| Bottomhole pressure | `Рзаб` | atm | 8% missing; low BHP → gas ingestion |
| Bubble-point pressure | `Дав. Нас` | atm | 5% missing; interact with Рзаб |

### Tier 2 — Water Chemistry (primarily static per well, standard Cox)

| Variable | Column | Transform | Missingness | Notes |
|---|---|---|---|---|
| Chloride ion Cl⁻ | `Cl⁻, мг/л` | log | 57% | Primary corrosion driver |
| Sulphate SO₄²⁻ | `SO₄²⁻, мг/л` | log | 54% | CaSO₄ scale precursor |
| Calcium Ca²⁺ | `Ca₂⁺, мг/л` | log | 55% | Scale forming |
| Bicarbonate HCO₃⁻ | `HCO₃⁻, мг/л` | log | 55% | CaCO₃ scale (separate mechanism) |
| Ca²⁺ · SO₄²⁻ | derived | log | derived | Saturation index proxy for CaSO₄ |
| pH | `pH` | raw or bins | 56% | Non-linear — consider spline or <6 / 6–7 / >7 |
| Solids KVCh | `Механические примеси (КВЧ), мг/дм³` | log | 31% | Abrasive wear |
| H2S concentration | `Массовая доля сероводорода, мг/дм³` | log | 80% | Within non-sour stratum: continuous exposure |
| Total mineralization | `Общая минерализация, г/л` | log | 54% | May be collinear with Cl⁻ |

### Tier 3 — Equipment and Completion (static, standard Cox)

| Variable | Column | Type | Missingness | Notes |
|---|---|---|---|---|
| Pump execution group | `Группа исполнения УЭЦН` | ordinal | 3% | Corrosion/abrasion protection level |
| Corrosion resistance | `Коррозионная стойкость` | categorical | 60% | Complement to execution group |
| ESP body size | `Габарит УЭЦН` | categorical | 5% | 5 / 5A / 5.5 / DN-387 etc. |
| Tubing ID / ESP size ratio | `Диаметр НКТ` / `Габарит УЭЦН` | ratio | 3–5% | Clearance → vibration transmission |
| Tubing grade | `Марка НКТ` | categorical | 27% | Chrome vs standard vs E-grade |
| Curvature at pump | `Работа в кривизне` | continuous, deg/10m | 40% | Lateral load on shaft |
| Setting depth | `Нспуска` | continuous, m | 0.1% | Proxy for T and P at pump |
| Well type | `Тип ствола скв` | categorical | 5% | Vertical / directional / horizontal |
| Number of stages (normalized) | `Кол.ступеней` / `Ном. Произв.` | ratio | 3% | Longer pump → more shaft bending |
| No-load / nominal current ratio | `Ток x.x` / `Ном. ток/ A` | ratio | 3% | Assembly quality at installation |
| Two-stage separator | `2-х этапка` | count | 0% | Gas separation count |

### Tier 4 — Treatment History (use with caution — endogenous)

| Variable | Column | Notes |
|---|---|---|
| Prior formation treatments | `ОПЗ` | Count before this run; high count → problem well |
| Acid jobs on ESP | `СКО УЭЦН` | May indicate prior scale/corrosion events |

---

## Missing Data Strategy

**Imputation hierarchy (apply per variable):**

1. **Well-level average** across all runs for that well — for variables measured periodically
2. **Pad-level (Куст) average** — primary strategy for water chemistry (lab samples taken at
   pad separator, applied to all wells on pad). This is the main source of uncertainty.
3. **Field × H2S group average** — fallback when pad has only one well
4. **Add `is_imputed` binary indicator** for each chemistry variable — allows the model to
   absorb systematic bias from imputed vs measured values

**High-missingness variables (>50%) — handling:**
- Cl⁻, SO₄²⁻, Ca²⁺, HCO₃⁻, pH, total mineralization: pad-level imputation is primary
- H2S concentration (80% missing): use `Кислый/Некислый` label as binary plus H2S concentration
  only where measured; consider as sensitivity analysis variable
- Corrosion resistance (60% missing): encode as "known_K2 / known_K1 / unknown"; do not impute
  categorical metallurgy from averages

**Acknowledged uncertainty:** pad-level imputation for water chemistry is the dominant source of
model uncertainty. This should be explicitly propagated into RUL confidence intervals via
bootstrap resampling that treats imputed values as random draws from the pad distribution.

---

## Correlation Structure — Known or Expected

Expected high-correlation pairs requiring resolution before Cox fitting:

| Pair | Expected ρ | Resolution |
|---|---|---|
| Frequency ↔ Motor load | High positive | Keep frequency (operational control variable); motor load as residual |
| Cl⁻ ↔ Total mineralization | High positive | Keep Cl⁻ (mechanistic); drop or use as robustness check |
| Ca²⁺ ↔ Ca·SO₄ product | High | Use product only |
| Q_actual ↔ Q_nominal/motor_load | Moderate | Use delta_BEP = Q/Q_nom − 1 |
| Setting depth ↔ Рзаб | Moderate negative | Keep both (different mechanisms) |

**Screening protocol:**
```python
# Spearman correlation matrix on all Tier 1+2 continuous variables
# Flag |ρ| > 0.65
# VIF check on final multivariate set — remove if VIF > 5
```

---

## File and Code Structure

Following project conventions (`CLAUDE.md`):

```
backend/analysis/
  models/survival/
    weibull_model.py              ← Phase 1 (exists)
    latent_weibull_competing_risks.py  ← Phase 2 (exists, extend to multi-stratum)
    extended_cox.py               ← Phase 5 (exists for M4/M5, extend)
  workflows/
    esp_survival/
      phase0_data_audit.py        ← stratum viability, missing data report
      phase1_single_weibull.py    ← per-stratum single Weibull
      phase2_mixture.py           ← per-stratum K=2 mixture
      phase3_diagnostics.py       ← bootstrap, GOF, component assignment
      phase4_contractor.py        ← contractor Cox/stratum decision
      phase5_extended_cox.py      ← covariate exploration + final model
      phase6_rul.py               ← RUL calculator
  features/
    imputation.py                 ← pad-level imputation logic
    covariate_transforms.py       ← log transforms, ratios, X·log(t) builder
```

All outputs via:
```python
from analysis.paths import results_dir
out = results_dir("esp_survival_<phase>_<variant>")
```

---

## Success Criteria

| Phase | Success criterion |
|---|---|
| Phase 2a (Vt pilot) | β₂ > 1.0 for both Vt sour and Vt non-sour |
| Phase 2a (Vt pilot) | r_j1 > 0.5 correlated with `Причина отказа = Сероводород` (42 known H2S cases) |
| Phase 2a (Vt pilot) | Contractor KM curves within Vt sour separate meaningfully (Шлюмберже > Борец > Новые технологии) |
| Phase 2 (all fields) | β₂ > 1.0 in all viable strata; ΔAIC > 4 vs single Weibull in ≥ 80% strata |
| Phase 2 (all fields) | Component assignment r_j1 > 0.5 correlated with 'Ранний'/'Преждевременный' failure flag |
| Phase 5 | All Schoenfeld p > 0.05 after adding extended terms for violated variables |
| Phase 5 | VIF < 5 for all covariates in final model |
| Phase 6 | RUL decreasing with age for pumps past ~100 days (wear-out regime) |
| Phase 6 | Vt sour RUL at t₀=60d meaningfully lower than Vt non-sour at same age |
| Phase 6 | Within Vt sour: Шлюмберже RUL > Борец RUL > Новые технологии RUL at same age |
