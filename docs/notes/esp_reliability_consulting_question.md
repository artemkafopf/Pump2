## Claude Consulting Question

We are implementing an ESP reliability system with three separated pipelines:

1. infant mortality QA
2. run-level mature-life TTF
3. rolling-window RUL / next-horizon risk

We have already fixed these design decisions:

- `Kpod_freq = Kpod × nominal_freq / freq` is the primary hydraulic operating-point variable
- feature scope tags are mandatory:
  - `_m_`   : mature fixed-window (days [infant_cutoff] to [infant_cutoff + 90d], ≥ 14 daily obs)
  - `_w_`   : whole-life post-infant summary, explanatory only
  - `_30d_` : rolling 30d window feature
  - `_rtd`  : run-to-date post-infant exposure feature
- `_w_` features are explanatory only and forbidden in predictive CatBoost matrices
- CatBoost TTF trains on failure rows only
- CV splits by `run_id`
- scenario analysis belongs to the rolling-window pipeline only
- infant mortality is a QA pipeline in Phase 1, not a deployed model
- telemetry is primary; techregime is fallback at `(well_id, dt)` level

What we want checked is not the architecture itself, but whether the data representation and validation tasks are sensible.

Important practical constraint:
- Phase 1 is intentionally a `working system first` milestone
- we are willing to accept some documented methodological flaws temporarily
- we do not want advice that explodes scope unless the flaw would make outputs fundamentally unusable

### Data context

Available daily channels after telemetry/techregime merge:
`freq`, `load`, `rzab`, `rpump_intake`, `qliq`, `watercut`, `gas_factor`, `qgas`, `kprod`.
No vibration, temperature, or current-signature data.

Frequency range in the dataset: approximately 40–65 Hz. Nominal frequency is 50 Hz for most pump types.

### Questions

1. **Infant mortality breakpoint validation.** What is the best method to confirm or reject a meaningful infant-mortality break from this data?
   - Test candidate thresholds: 45, 60, 90, 120 days.
   - Test globally and stratified by field, contractor, pump_family.
   - Expected output: a recommended threshold or evidence that no single global threshold is justified, with per-group breakdown where the global threshold does not hold.

2. **`_m_` and `_w_` feature soundness.** For mature-life run-level TTF, are the proposed scopes statistically sound?
   - `_m_` = first 90d of post-infant life (days [cutoff] to [cutoff + 90d]); requires ≥ 14 daily observations.
   - Runs that die between day [cutoff] and day [cutoff + 90d] passed infancy but lack a full mature window — call this the gap population. Should these be excluded from TTF models, allowed with partial windows, or modeled separately?
   - `_w_` = full post-infant to end-of-run; explanatory only.

3. **Kpod_freq vs raw Kpod for Weibull calibration.** Given that frequency varies 40–65 Hz across wells (nominal 50 Hz), is prioritizing `Kpod_freq` over raw `Kpod` the correct choice for the primary Weibull stress term? What statistical comparison (e.g., log-likelihood ratio, AIC difference, concordance index) would confirm or reject this choice?

4. **Additional dynamic features for rolling RUL / next-30d failure.** Beyond the existing set:
   - `Kpod_freq_30d_mean`
   - threshold-exposure fractions (`frac_kpod_below_0p7_30d`, `frac_glf_above_thr_30d`, `frac_pzab_below_1_30d`)
   - last-vs-prev30 trends (`kpod_30d_to_prev30_ratio`, etc.)
   - post-infant run-to-date exposure (`frac_*_rtd`)

   What additional features derivable from the channels above would you prioritize? In particular:
   - Are there high-value joint-state features (e.g., simultaneous high GLF and low pressure ratio, which are qualitatively different from either alone)?
   - Are there event-count or threshold-crossing-rate features worth adding?
   - Are there lagged or autoregressive signal summaries?

5. **Hidden leakage risks.** Are there any remaining leakage risks in this three-pipeline design? Specifically, evaluate:
   - **Response-model circularity**: the qliq/load/pressure response models (used for scenario propagation) are trained on the same rolling-window dataset as the failure model. Does this create a circular dependency that inflates scenario confidence?
   - **Telemetry selection bias**: wells with telemetry coverage may differ systematically from wells covered only by techregime (newer infrastructure, selected fields, different pump types). Is this a material covariate shift risk, and how should it be diagnosed?
   - **Class imbalance**: if average run length is ~300 days, the ratio of non-failure windows to failure windows is roughly 9:1. How should this be handled in CatBoost training and in probability output calibration?
   - Any other risks not listed here.

Please respond with:

- critique of the validation logic for each question
- critique of feature construction soundness
- any overlooked risks
- any high-value additions that preserve the current architecture

For each recommendation, state: (a) whether it requires a code change, and if so, which file and function; (b) whether it can be validated with a simple data check before any modeling is done.

Also distinguish between:
- `must fix in Phase 1 to avoid invalid or misleading outputs`
- `good Phase 2 improvement once the working system exists`
