## Phase 1 Plan

This note translates the agreed modeling decisions into an execution order for implementation.

### Goal

Build reliable data foundations so the project can support:
- infant mortality QA
- mature-life run-level TTF analysis
- rolling-window RUL / failure-horizon analysis

without mixing populations or leaking future information into predictive feature sets.

### Why We Need A Split

We are solving three different problems:

1. `Infant mortality QA`
   - very early failures
   - operational intelligence
   - not a deployed predictive model in Phase 1

2. `Run-level TTF`
   - one row per run
   - total run life / censoring
   - explanatory Weibull and CatBoost-on-failures

3. `Rolling-window RUL / next-horizon risk`
   - one row per `(run, anchor_date)`
   - state-conditioned risk / scenario analysis

The current confusion came from interpreting rolling-window outputs as if they were classic total-life TTF outputs.

There is also a practical Phase 1 constraint:
- we need a working system that produces results now
- some flaws may remain
- those flaws should be outlined clearly
- but they should not block the first usable implementation

### Step Order

1. Create shared modeling constants and feature-scope guards.
2. Create one shared merged daily loader using telemetry first and techregime fallback.
3. Implement infant-vs-mature run splitting and QA summary.
4. Build run-level mature-life features with scope-tagged names and report the `gap` population explicitly.
5. Extend rolling-window features with `Kpod_freq`, exposure fractions, and run-to-date summaries.
6. Change CV discipline to split by `run_id`.
7. Update mature-life explanatory Weibull feature candidates.
8. Keep scenario analysis on the rolling pipeline and consume the enriched rolling features.

### Feature Scopes

- `_m_`
  - mature fixed-window summary
  - suitable for predictive or explanatory use after the fixed mature observation window exists

- `_w_`
  - whole-life post-infant summary
  - explanatory only
  - never allowed in predictive CatBoost feature matrices

- `_30d_`
  - rolling state summary
  - scenario / RUL features

- `_rtd`
  - run-to-date exposure after the infant window
  - rolling state history feature

### Minimum Acceptance Criteria

Phase 1 is successful if:

1. We can build a merged telemetry-first daily dataset reproducibly.
2. We can split runs into infant and mature populations with an auditable summary.
3. We can build mature run-level features without leakage ambiguity.
4. We can build rolling features that exclude the infant period.
5. We can enforce run-level CV grouping.
6. We can block accidental use of explanatory `_w_` features in predictive CatBoost matrices.
7. We can produce usable outputs even if some limitations remain explicitly documented.

### Key Risks

- Incomplete post-infant daily coverage for some runs
- Multiple naming schemes if scope tags are not enforced strictly
- Reusing whole-life explanatory features in predictive models by accident
- Continuing to describe rolling-window outputs as total-life forecasts
- Over-expanding Phase 1 trying to solve every methodological issue before delivering working outputs

### Recommended Output Locations

- code:
  - `backend/analysis/modeling_config.py`
  - `scripts/data_utils.py`
  - `scripts/build_run_features.py`

- QA / artifacts:
  - `analysis_outputs/.../infant_mortality_summary.json`
  - `analysis_outputs/.../mature_feature_coverage.csv`
  - `analysis_outputs/.../rolling_feature_coverage.csv`
  - `analysis_outputs/.../source_coverage.csv`

### Known Phase 1 Acceptable Imperfections

- the infant-mortality cutoff may remain a provisional default
- some joint-state features may use simplified thresholds
- probability calibration may be basic or deferred
- scenario propagation may retain response-model circularity

These should be surfaced as caveats and TODO items, not hidden.
