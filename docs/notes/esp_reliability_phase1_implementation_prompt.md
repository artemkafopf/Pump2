# ESP Reliability Modeling — Phase 1 Implementation Prompt

You are implementing an ESP (Electric Submersible Pump) reliability modeling system in Python.

The codebase is at:
`d:\GitHub\Pump2`

Primary project-local data sources:

- `data/inputs/Отказы свод с анализом_БДА_V03_all.xlsx`
- `data/inputs/Отказы свод с анализом_БДА_V03_failures.xlsx`
- `data/inputs/Отказность Аналитика(1).pptx`
- `data/sqlite/telemetry.sqlite`
- `data/sqlite/techregime.sqlite`

Telemetry is the primary dynamic source. Techregime is the fallback where telemetry is missing.

Existing relevant code:

- `backend/analysis/`
- `backend/analysis/weibull_model.py`
- `backend/analysis/stress_transforms.py`
- `scripts/analyze_failure_horizon.py`
- `scripts/analyze_vt_60hz_scenario.py`
- `scripts/analyze_v03_stress.py`
- `scripts/analyze_kpod_window_thresholds.py`
- `scripts/verify_presentation_claims.py`

The purpose of this prompt is implementation. Do not redesign the architecture unless a requirement is internally inconsistent. If something is ambiguous, preserve existing project conventions and make the smallest consistent extension.

Phase 1 is a `make it work well and produce results` milestone, not a `make it perfect` milestone.

Practical rule for this phase:
- favor a complete working pipeline over an unfinished theoretically cleaner one
- document known flaws and methodological debts explicitly
- do not block delivery trying to solve every secondary issue now
- if something is imperfect but still usable and auditable, implement it and mark it for Phase 2

The minimum standard for Phase 1 is:
- the code runs
- the outputs are reproducible
- the feature construction is auditable
- the main caveats are visible in code comments and output notes

## Context

We are separating three distinct analytical pipelines:

1. `Infant mortality QA pipeline`
2. `Run-level TTF / MRP pipeline`
3. `Rolling-window RUL / next-horizon failure pipeline`

These must not be mixed conceptually or in feature construction.

## Decisions Already Made — Do Not Reopen

The following are resolved constraints:

1. `Kpod_freq = Kpod × nominal_freq / freq` is the primary hydraulic operating-point variable in all models.
   - Frequency varies approximately 40–65 Hz across wells; nominal is 50 Hz for most pump types. This range makes the frequency correction material.
   - Raw `Kpod` may remain as a secondary CatBoost feature.
   - Raw `Kpod` must not be the primary Weibull stress term when frequency varies across wells.

2. Feature scope tags are mandatory in derived column names:
   - `_m_`   : mature fixed-window summary
   - `_w_`   : whole-life post-infant summary, explanatory only
   - `_30d_` : rolling 30d window feature
   - `_rtd`  : run-to-date post-infant feature

3. Any CatBoost feature matrix built with `predictive=True` must reject `_w_` columns by raising `ValueError`.

4. CatBoost TTF regression trains on `event == 1` rows only.
   - Censored runs are not valid regression targets.
   - Weibull / survival handles censoring.

5. Cross-validation splits by `run_id`, not by `well_id`.
   - All windows from one run must stay in a single fold.

6. Scenario analysis such as "force 60 Hz now" belongs to the rolling-window pipeline only.
   - Scenario output labels must use language such as `state-conditioned next-30d risk`, `state-conditioned next-90d risk`, `state-conditioned next-365d risk`.
   - Do not label these outputs as total remaining life.

7. Infant mortality is a QA / operational intelligence pipeline only in Phase 1. It is not a deployed predictive model.

8. Telemetry is primary and techregime is fallback at the daily merged-data level.

9. `GLF_THRESHOLD = 300.0` should be implemented as a configurable project constant.

## Questions Still Worth Validating

These are not open architecture questions; they are validation tasks to run alongside implementation and log as evidence:

1. Whether a meaningful infant-mortality break exists near 45 / 60 / 90 / 120 days globally and stratified by field, contractor, and pump family.
2. Whether `Kpod_freq` is statistically stronger than raw `Kpod` for mature-life wear modeling, measured by AIC difference on the Weibull explanatory fit.
3. Whether post-infant whole-life exposure features improve explanatory Weibull fit relative to mature fixed-window summaries alone (compare AIC, log-likelihood).
4. What fraction of wells have ≥ 14 daily observations in the `_m_` window — log this as a coverage report before modeling.
5. What fraction of daily rows come from telemetry vs techregime per field — log this as a source-coverage audit.

Validation outputs must be logged to `analysis_outputs/`. They must not alter implementation scope or reopen resolved decisions.

If a validation result suggests a weakness, record it as:
- a warning
- a report caveat
- or a Phase 2 TODO

Do not let validation work derail Phase 1 unless it reveals a fundamental implementation bug.

## Phase 1 Implementation Task

Implement the data foundations and feature-building utilities first. Do not start with frontend or report polish.

## Shared Constants And Feature Guards

Create a shared module:

- `backend/analysis/modeling_config.py`

Add:

- `DEFAULT_INFANT_MORTALITY_DAYS = 90`
- `DEFAULT_MATURE_WINDOW_DAYS = 90`
- `DEFAULT_MIN_MATURE_OBS = 14`
- `DEFAULT_GLF_THRESHOLD = 300.0`
- `FEATURE_SCOPES = {...}`

Implement:

- `assert_no_explanatory_features(columns: list[str], predictive: bool) -> None`

Behavior:

- If `predictive=False`, do nothing.
- If `predictive=True`, scan `columns`.
- If any column contains `_w_`, raise `ValueError` listing offending columns.

## Shared Daily Merge Utility

Create:

- `scripts/data_utils.py`

Implement:

```python
def load_daily_merged(
    wells: list[str],
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ...
```

Rules:

- Query `telemetry.sqlite`
- Query `techregime.sqlite`
- Normalize both to canonical columns: `well_id, dt, freq, load, rpl, rpump_intake, rzab, qliq, watercut, gas_factor, qgas, kprod`
- Per `(well_id, dt)`: use telemetry row if present, otherwise use techregime row
- Return canonical columns above plus a `source` column with `"telemetry"` or `"techregime"`
- Respect optional `date_from`, `date_to`
- Reuse existing local path resolvers for project-local SQLite files

Also implement:

```python
def source_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Returns per-`well_id` counts of `telemetry` vs `techregime` rows and the telemetry fraction.

If field metadata is cheaply available via run metadata or a prebuilt well-to-field mapping, include `field` as an extra column. If not, do not block Phase 1 on recovering `field` inside this function.

Write output to `analysis_outputs/.../source_coverage.csv`. This is the telemetry selection bias audit — wells with systematically different coverage rates may be a covariate shift risk and should be flagged.

## Infant Mortality QA Split

Implement in:

- `scripts/build_run_features.py`

```python
def split_infant_vs_mature_runs(
    runs: pd.DataFrame,
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS,
    mature_window_days: int = DEFAULT_MATURE_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ...
```

Rules:

- Input is full run population from `load_runs()`
- Define:
  - `infant_runs`: `duration_days < infant_mortality_days`
  - `gap_runs`: `infant_mortality_days <= duration_days < infant_mortality_days + mature_window_days` — runs that survived infancy but ended before the `_m_` window closed; these lack full `_m_` predictive features
  - `mature_runs`: `duration_days >= infant_mortality_days + mature_window_days`
- Return four values: `mature_runs` dataframe, `infant_runs` dataframe, `gap_runs` dataframe, QA summary dict

QA summary must include:

```python
{
  "total_runs": int,
  "infant_mortality_runs": int,
  "infant_mortality_rate": float,
  "gap_population_runs": int,
  "gap_population_rate": float,
  "mature_runs": int,
  "mature_rate": float,
  "infant_mortality_by_field": dict,
  "infant_mortality_by_contractor": dict,
  "infant_mortality_by_pump_family": dict,
  "gap_by_field": dict,
  "median_infant_duration_days": float | None,
  "median_gap_duration_days": float | None,
}
```

Write QA summary to `analysis_outputs/.../infant_mortality_summary.json`.

Interpretation rule for Phase 1:
- `gap_runs` are excluded from `_m_`-based predictive CatBoost datasets by default
- `gap_runs` are reported explicitly and not silently discarded
- later phases may choose to reincorporate them into explanatory-only survival analyses
- Phase 1 does not need to solve that reintegration

## Pipeline A — Run-Level Feature Builder

In:

- `scripts/build_run_features.py`

Implement:

```python
def build_run_features(
    runs: pd.DataFrame,
    glf_threshold: float = DEFAULT_GLF_THRESHOLD,
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS,
    mature_window_days: int = DEFAULT_MATURE_WINDOW_DAYS,
    min_mature_obs: int = DEFAULT_MIN_MATURE_OBS,
    include_whole_life: bool = True,
    predictive: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ...
```

Returns: `mature_features_df`, `infant_mortality_df`, `gap_population_df`, `summary_dict`.

Rules:

- Split infant / gap / mature before feature aggregation using `split_infant_vs_mature_runs()`.
- Load merged daily telemetry/techregime data using `load_daily_merged()`.
- Compute daily variables first:
  - `kpod_daily = qliq / nominal_qliq`
  - `kpod_freq_daily = kpod_daily * nominal_freq / freq`
  - `pressure_ratio_daily = rzab / pbubble`
- Mature fixed-window interval: start = `install_date + infant_mortality_days`, end = `start + mature_window_days`
- Require minimum `min_mature_obs` daily observations for `_m_` feature construction. If fewer observations are available, set `_m_` features to `NaN` and log counts — do not silently drop or invent values.
- `_w_` features aggregate from post-infant start to run end; explanatory only.
- If `predictive=True`, call `assert_no_explanatory_features(...)`.

Phase 1 simplification:
- it is acceptable if `build_run_features()` first delivers a clean, auditable feature table and exclusion summaries, even before every downstream model is fully migrated to it
- prioritize correct construction and traceability over polishing every consumer immediately

Required `_m_` examples:

- `kpod_m_mean`
- `kpod_freq_m_mean`
- `frac_kpod_below_0p7_m`
- `pzab_over_pbubble_m_mean`
- `frac_pzab_below_1_m`
- `glf_m_mean`
- `frac_glf_above_thr_m`
- `load_m_mean`
- `load_m_std`
- `freq_m_mean`
- `mean_neg_excess_kpod_m`

Required `_w_` examples:

- `kpod_freq_w_mean`
- `frac_kpod_below_0p7_w`
- `frac_pzab_below_1_w`
- `frac_glf_above_thr_w`
- `load_w_std`
- `mean_neg_excess_kpod_w`
- `integrated_glf_excess_w`

Static features should include: field, contractor, pump family, nominal qliq, nominal head, nominal frequency, motor power, stages, submergence, pbubble, curvature flag, run number at well if derivable, season of install if derivable.

## Pipeline B — Extend Rolling-Window Dataset

Modify:

- `scripts/analyze_failure_horizon.py`

Required behavior changes:

1. `build_dataset(...)` — accept `infant_mortality_days`; skip any anchor where `anchor_date < install_date + infant_mortality_days`.

2. `build_window_feature_entry(...)` — add the following new features:

   Daily-derived rolling features:

   - `kpod_freq_30d_mean`
   - `frac_kpod_below_0p7_30d`
   - `frac_pzab_below_1_30d`
   - `frac_glf_above_thr_30d`
   - `mean_neg_excess_kpod_30d`

   Trend features:

   - `kpod_30d_to_prev30_ratio`
   - `load_30d_to_prev30_ratio`
   - `pzab_30d_to_prev30_ratio`

   Run-to-date post-infant features:

   - `frac_kpod_below_0p7_rtd`
   - `frac_pzab_below_1_rtd`
   - `frac_glf_above_thr_rtd`

   Joint-state features for the 30d window:

   - `frac_high_glf_low_pzab_30d` — fraction of days where `gas_factor > GLF_THRESHOLD` AND `pressure_ratio < 1.0` simultaneously; qualitatively distinct from either condition alone
   - `frac_high_load_low_kpod_30d` — fraction of days where `load > load_p90_well` AND `kpod_freq < 0.7` simultaneously

3. Preserve existing columns already used by downstream scripts where possible.

Definition note:
- `load_p90_well` should mean the post-infant per-well 90th percentile of daily `load` computed from the merged daily series available to the pipeline.
- If this turns out awkward to implement cleanly in Phase 1, a simpler documented approximation is acceptable.
- If even that is too messy for Phase 1, omit this one joint-state feature and leave a clear TODO note rather than blocking delivery.

Note on class imbalance: with average run length ~300 days, failure windows are roughly 1 in 10 rows. When training the 30d binary classifier, use CatBoost's `scale_pos_weight` or `class_weights` parameter. Do not apply SMOTE or row-level resampling — that would break the run-level CV grouping.

Calibration guidance for Phase 1:
- preferred: add a simple held-out probability calibration step such as isotonic regression
- acceptable fallback: skip explicit calibration for now, but log calibration diagnostics and document calibration as a Phase 2 improvement item

The priority is a working grouped-CV classifier with usable outputs, not a perfect probability-estimation stack.

## Cross-Validation By Run ID

Implement in a shared utility module or inside modeling helpers:

```python
def split_by_run_id(
    df: pd.DataFrame,
    run_id_column: str = "row_id",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Rules:

- Hash on `run_id` so all rows/windows from one run stay together.
- Use the same stable deterministic splitting pattern already used elsewhere.

Replace any `well_id`-based fold assignment in the rolling-window modeling helpers with run-based splitting.

## Weibull Stress-Term Usage

For the run-level mature population, fit explanatory Weibull on mature runs only and use `_w_` features as candidate explanatory stress terms.

Preferred priority order:

1. `kpod_freq_w_mean`
2. `frac_kpod_below_0p7_w`
3. `frac_pzab_below_1_w`
4. `frac_glf_above_thr_w`
5. `load_w_std`

Use existing stress transforms and ranking logic:

- `negative_excess`
- `positive_excess`
- `relative_abs_deviation`

Validation task: for each top-ranked stress term, fit a parallel model substituting raw `Kpod` for `Kpod_freq` and log the AIC difference. This is evidence for Decision 1, not a reason to change it.

Do not modify:

- `backend/analysis/weibull_model.py`
- `backend/analysis/stress_transforms.py`

## Rolling Scenario Model

Do not redesign scenario propagation.

For the rolling scenario stack:

- keep `scripts/analyze_vt_60hz_scenario.py` as the scenario surface
- allow it to consume the expanded rolling features
- use `Kpod_freq_30d` as an input feature in CatBoost and Weibull candidate sets
- preserve existing output files
- scenario language must remain state-conditioned

Editing `scripts/analyze_vt_60hz_scenario.py` is allowed in Phase 1 for:
- updating feature lists
- switching CV grouping to `run_id`
- exposing new rolling features in reports and outputs

Do not replace the overall scenario mechanism in Phase 1.

Important: the qliq/load/pressure response models (used inside scenario propagation) and the failure model are both trained on the same rolling-window dataset. This creates a potential circularity — response-model outputs feed the failure model under a counterfactual state that neither model was trained on jointly. Do not attempt to fix this in Phase 1. Document it in a comment near the scenario propagation entry point so it is visible for Phase 2.

This is an acceptable Phase 1 limitation:
- the system can still produce useful scenario outputs
- but the caveat must be explicit

## Deliverables

Phase 1 should produce:

1. Shared config/constants module
2. Shared merged daily-loader with source-coverage audit
3. Infant-vs-mature-vs-gap split helper with QA summary
4. Run-level feature builder
5. Extended rolling-window feature builder with joint-state features
6. Run-ID-based CV split helper
7. Leakage guard for explanatory features
8. Output JSON for infant mortality QA
9. Output CSV for source coverage by field/well

## Implementation Notes

- Prefer small composable helpers over one large script.
- Reuse existing naming and column conventions where possible.
- Preserve backward compatibility for downstream scripts unless explicitly changing their contract.
- If a feature cannot be built because required inputs are absent, return `NaN` rather than silently inventing a fallback.
- Log counts of excluded rows and insufficient-data rows.
- If forced to choose between a complete documented implementation and an incomplete cleaner one, choose the complete documented implementation.
- Add short TODO notes for known methodological debts instead of trying to solve them all immediately.

## Out of Scope For Phase 1

- FastAPI / frontend redesign
- Replacing Weibull core implementation
- Fixing the response-model circularity in scenario propagation
- Large scenario methodology redesign
- Full presentation generation
- New deployment packaging
- perfect calibration / perfect causal interpretation
