# Presentation Verification App Plan

## Prompt Count

The attached prompt does **not** contain 14 top-level numbered sections.

It contains **14 section-like requirement blocks** if counted as:

1. `## 1. Goal`
2. `## 2. Main Engineering Hypotheses to Verify`
3. `## 3. Required Input Data Structure`
4. `### H1`
5. `### H2`
6. `### H3`
7. `### H4`
8. `### H5`
9. `### H6`
10. `### 3.1 runs`
11. `### 3.2 daily_or_monthly_regime`
12. `### 3.3 design`
13. `### 3.4 telemetry`
14. `### 3.5 gas_limits`

If the expectation was **14 major task points**, the saved attachment is currently shorter than that.

## Current App vs Prompt

Current implementation is a generic Weibull stress-model UI:

- Single selected sheet upload with manual configuration.
- Manual stress-term setup.
- Weibull grouped fitting with category fallback.
- Survival, PDF, CDF, histogram, and sensitivity plots.
- Hierarchy browsing for `Field`, `Contractor`, and `Field + Contractor`.
- Local preset for:
  - `Event`
  - `Наработка (сут)`
  - `Месторождение`
  - `Принадлежность`
  - filter `Наработка (сут) > 30`

This is useful infrastructure, but it is **not yet the specialized presentation-verification app** described in the prompt.

## Coverage Matrix

| Prompt area | Status | Notes |
|---|---|---|
| Main Excel upload | Partial | Workbook upload exists, but analysis still starts from one manually chosen sheet. |
| Multiple sheets (`runs`, `daily_or_monthly_regime`, `design`, `telemetry`, `gas_limits`) | Missing | No automatic sheet detection or cross-sheet joins yet. |
| Support already-processed files | Partial | Generic app can use derived columns if already present. |
| Verify presentation conclusions statistically | Partial | Weibull/survival is present, but no presentation-specific hypothesis report. |
| Quantify design-envelope violation | Partial | Possible only through manual stress terms; not automated. |
| H1 `Kpod` | Missing | No dedicated `Kpod` feature derivation or canned analysis. |
| H2 pressure ratio (`Pintake/Pbubble` fallback chain) | Missing | No automatic ratio construction or threshold analysis. |
| H3 gas/free gas/gas limits | Missing | No gas-limit table or gas-excess derivation. |
| H4 ineffective frequency increase | Missing | No event detection from time series. |
| H5 instability before failure | Missing | No variability/shutdown/restart feature engineering from telemetry/regime. |
| H6 design-vs-actual mismatch | Missing | No joins to design sheet or mismatch feature set. |
| Survival analysis | Implemented | Present in the current app and backend. |
| Classification | Missing | No binary classifier or feature importance workflow. |
| Infant mortality separation | Missing | No dedicated early-failure segmentation/reporting. |
| Automated conclusions report | Missing | No "supported / weak / unsupported" hypothesis summary. |
| Low-click workflow | Missing | Current app is flexible but still manual. |

## Recommended Direction

Build a **new specialized Streamlit app** that reuses the existing `backend/analysis` Weibull engine where it helps, instead of trying to force the current general-purpose page into the presentation workflow.

Recommended shape:

- Keep current `streamlit_apps/streamlit_app.py` as the generic model builder.
- Add a second app, for example `streamlit_apps/presentation_app.py`.
- Reuse backend fitting/plotting primitives where possible.
- Add a domain-specific preprocessing layer that:
  - auto-detects prompt sheets,
  - derives presentation features,
  - applies defaults,
  - runs a near one-click analysis pipeline.

This will be cleaner than overloading the current UI with many special cases.

## Proposed Automated Workflow

The new app should aim for this user flow:

1. Upload workbook.
2. Auto-detect `runs`, `daily_or_monthly_regime`, `design`, `telemetry`, `gas_limits`.
3. Auto-map columns using aliases.
4. Show one compact review page:
   - detected sheets,
   - detected columns,
   - missing optional inputs,
   - default thresholds.
5. User presses one main button: `Run Presentation Verification`.
6. App produces:
   - executive hypothesis summary,
   - automated subgroup survival analysis,
   - stress-function results,
   - infant mortality split,
   - design-vs-actual mismatch summary,
   - exportable result tables.

## Backend Plan

### Phase 1. Data contract and auto-detection

Add a new backend module, for example:

- `backend/analysis/presentation_schema.py`
- `backend/analysis/presentation_loader.py`

Responsibilities:

- Sheet auto-detection by exact name plus aliases.
- Column alias mapping for English/Russian variants.
- Validation report with required, optional, missing fields.
- Default gas-limit dictionary support.

### Phase 2. Domain feature engineering

Add a new module, for example:

- `backend/analysis/presentation_features.py`

Derived outputs per `run_id`:

- `pressure_ratio_*`
- `Kpod_*`
- `GLF_*`
- `free_gas_*`
- `gas_excess_*`
- `freq_change_*`
- `freq_up_event_count`
- `delta_Q_after_freq_up`
- `ineffective_freq_increase_*`
- `motor_load_*`
- `current_*`
- `intake_pressure_*`
- `shutdown_count_30d`
- `restart_count_30d`
- `unstable_operation_score`
- design-vs-actual mismatch features
- `infant_mortality_flag`

This module should aggregate `daily_or_monthly_regime` and `telemetry` into one run-level feature table joined back to `runs`.

### Phase 3. Automated hypothesis tests

Add a module, for example:

- `backend/analysis/presentation_hypotheses.py`

For each hypothesis H1-H6:

- define required fields,
- define fallback field chain,
- compute analysis-ready datasets,
- fit default comparisons,
- produce effect-size tables,
- produce a support verdict:
  - `supported`
  - `mixed`
  - `not_supported`
  - `insufficient_data`

### Phase 4. Presentation-specific model presets

Add fixed model recipes instead of manual stress-term setup:

- `Kpod` stress recipe
- pressure-ratio stress recipe
- gas-excess stress recipe
- instability stress recipe
- design-mismatch stress recipe

Each recipe should:

- choose default transform,
- choose default threshold/reference,
- choose default plots,
- expose only a few expert overrides.

### Phase 5. Specialized Streamlit UI

Create `streamlit_apps/presentation_app.py` with 4 tabs:

1. `Upload & Auto-Detect`
2. `Executive Summary`
3. `Hypothesis Checks`
4. `Data Quality & Details`

Recommended defaults:

- auto-run once workbook is valid,
- hide advanced controls inside expanders,
- keep one primary button,
- preselect field/contractor hierarchy views,
- auto-build exports.

## MVP Scope

For the first usable version, I would implement only:

1. Auto-detect workbook sheets and columns.
2. Build a run-level merged dataset.
3. Implement H1, H2, H3, and H6 first.
4. Add infant mortality split.
5. Add executive summary output.
6. Keep H4 and H5 as partial/experimental if telemetry quality is weak.

Reason:

- H1/H2/H3/H6 are the clearest from the prompt.
- H4/H5 depend more heavily on time-series consistency and event logic.
- This gives a strong first version without overcomplicating the UI.

## Proposed First Implementation Slice

The first coding pass should deliver:

- new `presentation_app.py`
- auto-detect workbook sheets
- alias-based column mapping
- merged run-level analysis dataset
- default gas limits
- infant mortality flag
- H1/H2/H3/H6 feature derivation
- hypothesis summary tables
- survival comparisons for each hypothesis
- export of prepared features and hypothesis results

## Reuse from Current App

Keep and reuse:

- Weibull grouped fitting in `backend/analysis/weibull_model.py`
- plotting primitives in `backend/analysis/plotting.py`
- filter and group fallback utilities in `backend/analysis/data_utils.py`
- current local Streamlit launch scripts

Do **not** reuse as-is:

- current manual stress-term-heavy setup page as the primary workflow for the presentation app

## Recommendation

Build a **new specialized app** first, and only after it stabilizes decide whether to expose it as:

- a second Streamlit entrypoint, or
- a preset mode inside the existing app.

For this task, a separate entrypoint is the safer and more automated design.
