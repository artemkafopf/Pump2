# Phase C — Operational (Block 2) + Completion (Block 3) Cox, Merged θ, First Holdout

## Purpose

Blocks 2 and 3 of the Cox program, on the foundation Phases A–B fixed:

1. **Block 3 — completion/design hazards** (what was installed): stages, head-per-stage,
   motor power, pump габарит/series, curvature, setting depth. All knowable at t=0, 93–98%
   coverage — the cleanest covariates in the warehouse, never yet modeled.
2. **Block 2 — operational hazards** (what the operator controls): frequency, load,
   kpod = Ql/Qnominal, drawdown, restarts/idle. The risks we can change by changing the
   operating mode — and the decisive test of the GLF puzzle.
3. **Merged θ(t)**: survivors of Blocks 1+2+3 in one joint model → the VBA handoff candidate.
4. **The stack's first out-of-sample validation** (temporal holdout) — every number produced
   so far is in-sample; Phase C ends that.

Context (read first):

```
results/analysis_review/2026-07-06/README.md              — roadmap; status block at top
agents/analyses/phase_a_cox_foundation.md                  — covariate registry (§Part 2) + design decisions
agents/analyses/phase_a_cox_foundation_results.md          — T6 correction, КВЧ verdict, registry coverage
results/phase_b_report/2026-07-06/reports/phase_b_report.md — mode-specific results, corrected B4 headline
results/phase_a_covariate_registry/2026-07-06/             — registry.csv (measured coverage), OPEN_QUESTIONS.md
agents/analyses/chemistry_cox.md                           — the screening cascade (reused per block)
CLAUDE.md                                                  — path rules
```

---

## §0 — CONFIRM WITH USER BEFORE IMPLEMENTING

### 0.1 Block 3 covariate semantics (from Phase A OPEN_QUESTIONS — still open)

| Item | Status | If unresolved at implementation time |
|---|---|---|
| `vg_m` meaning (Excel "ВГ, м"; 0–2,740 m, mean ≈ 2,310) | pending data owner | Enter as *uninterpreted monotone depth proxy*; effect direction reportable, physical units not |
| `curvature_flag` units (Excel "Работа в кривизне"; ~0.31–0.51, 56% coverage) | pending data owner | Enter with `_missing` indicator as *unitless dogleg proxy*; no quantitative operating-envelope claims |
| Габарит → OD mm mapping (confirmed only 5/5А/6) | pending user | Use габарит as **categorical**, never numeric OD |
| Tubing (НКТ) diameter | absent from warehouse | Note in report as still-open data request |

Proceeding with proxies is acceptable **only** with the flags above stamped into the outputs;
if the user answers before implementation, use the real semantics.

### 0.2 Block 2 exposure window and causal framing

- Default window: **first 30 operating days** (`early`), pre-registered in Phase A. Confirm,
  or pick 60/90 d — then that window applies to every Block 2 covariate uniformly (no
  per-covariate window shopping).
- Operational covariates are *chosen responses* to well conditions (confounding by
  indication: troubled wells get run differently). Phase C reports **associations under
  adjustment**, never "changing X will cause Y" — that language is reserved until a
  design-based analysis (e.g., within-well changes) exists. The report must carry this
  disclaimer verbatim in its header.

### 0.3 Merged-θ handoff scope

The merged coefficient file is a **candidate** for VBA integration: which blocks feed it
(default: all three), and whether EXTENDED (time-varying) terms are acceptable to the VBA
side (`ApplyCoxEta` currently supports STANDARD only — see `agents/analyses/cox_hr_vba_integration.md`).
Confirm before C4. **No VBA changes in Phase C either way.**

---

## Design decisions (inherited — not re-litigated)

- Clock `ttf_mix`; every output stamped `clock`/`window`. `pct_mixed_clock` in inventories.
- One joint stratified Cox per model: `strata=['stratum_key']` (field × h2s × contractor),
  **global β**, `cluster_col='well_key'`, `robust=True`. §0.2(c) stands: `is_sour_flagged`
  is a covariate, not a stratum.
- Time-varying coefficients only via `analysis.models.survival.time_interaction_cox`
  (event-time episode split). Curve-regression γ is banned for inference.
- Cause-specific views via `analysis.data.competing_risks_loader` + `cause_specific_cox`
  (Phase B machinery). Mode differentiation doubles as a confounding check: a covariate
  elevated equally in all modes is suspect; one that lands on its physically expected mode
  is corroborated.
- Every operational number carries a bootstrap or profile CI. ΔAIC for nested comparisons.
- Multiple testing: state the grid size in every screening table and flag hit counts vs
  null expectation (Phase B discipline).

### Mandatory adjusters (in every Block 2/3 model — the cohort covariates never yet used)

| Covariate | Definition | Why mandatory |
|---|---|---|
| `install_period` | install_year binned: ≤2019 / 2020–2022 / 2023+ | 13-year technology + telemetry drift; confounds everything, especially contractor and freq covariates |
| `run_seq` | 1 + count of prior runs on the well_key (log1p or binned 1/2/3+) | 62% of runs are repeats; hostile wells accumulate short runs |
| `days_since_prev_failure` | previous stop → install gap (log1p; missing → first run) | rushed-workover signal |
| `has_telemetry` | `ttf_true_source != 'missing'` | Phase A T4: telemetry absence is informative (those runs differ), not random |

---

## Tasks

### C0 — Feature completion + audit refresh (output: `phase_c_features`)

Extend `analysis/data/run_covariates.py` with what the registry specified but Phase A did
not materialize (verify against `registry.csv` first — build only what's missing):

- **Block 2, early window** (window from §0.2): `freq_mean_early`, `freq_std_early`,
  `n_freq_steps_per_100d` (day-over-day |Δf| > 1 Hz), `load_mean_early`, `load_std_early`,
  `kpod_early` (daily qliq / nominal_flow_m3d), `kpod_freq_early` (Qnom scaled by f/50),
  `frac_kpod_below_0p7_early`, `pzab_over_pbubble_early` (daily rzab / pbubble_atm),
  `frac_pzab_below_1_early`, `rpump_intake_mean_early`,
  `n_restarts_per_100d` (daily qliq 0→>0 transitions), `idle_frac` (1 − ttf_true/run_days;
  whole-run by construction — label it so).
- **Cohort**: the four mandatory adjusters above.
- Every covariate: `_missing` indicator + imputation-source provenance (Phase A pattern).
- Outputs: refreshed `registry.csv` with measured coverage per new covariate,
  coverage-by-stratum heatmap, Spearman matrix extension. Unit tests for kpod and restart
  derivations (synthetic daily frames).

**Coverage reality check** (from Phase A audit — plan around it, don't rediscover it):
freq ≥30 valid days ≈ 1,377 runs; load ≈ 78% of daily rows; rpump_intake ≈ 59%;
qliq ≈ 88%. Block 2 models will run on the telemetry-covered subpopulation — `has_telemetry`
plus explicit n per model row make that visible.

### C1 — Block 3: completion Cox (output: `phase_c_block3_completion`)

Candidates (t0, from registry §Block 3): `stages`, `head_per_stage`, `log_motor_power_kw`,
`log_nominal_flow_m3d`, `pump_gabarit` (categorical), `pump_series` (categorical),
`curvature` (+missing), `vg_m`, `pbubble_atm`, `nominal_freq_hz`. Plus the four mandatory
adjusters.

1. **Collinearity triage first**: motor_power ~ nominal_flow ~ stages ~ head_per_stage ~
   габарит are one "size" axis. Spearman + VIF; keep the physically distinct residuals
   (e.g., head_per_stage *given* nominal_flow = stage aggressiveness). Document what was
   dropped and why.
2. Chemistry-cascade screening (univariate → Schoenfeld → correlation → VIF → joint),
   pooled all-cause on ttf_mix.
3. **Cause-specific pass** for joint-model survivors (hydraulic / electro-thermal /
   protector; protector = survivors only, 186 events). Expected physics to check against:
   curvature → cable damage (electro-thermal); vg_m/depth → temperature → electro-thermal;
   stage aggressiveness → hydraulic.
4. Per-field heterogeneity table for every survivor (Phase B pattern).

EPV budget: 1,527 pooled events — a 12–15 term joint model is safe; per-mode limit to
screened survivors.

### C2 — Block 2: operational Cox (output: `phase_c_block2_operational`)

Same cascade, same structure, on the C0 early-window covariates + mandatory adjusters,
**including the Block 1 survivors as adjusters** (log_h2s_proxy or is_sour_flagged per the
B3 clean-refit lesson — not both; log_glf).

Named hypotheses to answer explicitly (each gets a yes/no/inconclusive line in the report):

- **H-GLF (the decisive test).** GLF→hydraulic HR 0.877 (p=0.003) has survived every
  robustness check so far. Add `pzab_over_pbubble_early`, `kpod_early` and
  `rpump_intake_mean_early` to the model: if GLF's effect is absorbed → it was an
  operating-point proxy; if it survives → treat as real gas-interference/regime effect and
  say so. This closes the last suspect finding from the original review.
- **H-FREQ.** Fleet-wide frequency exposure vs the Vt-only 60 Hz work (which found stable
  60 Hz not harmful, ПЭД+КЛ = electro-thermal highest-risk). Test `freq_mean_early`,
  `freq_above_55hz_pct`, and *instability* (`freq_std_early`, `n_freq_steps_per_100d`)
  separately — the prior says instability, not level, is the killer; the electro-thermal
  cause-specific pass is the corroboration channel.
- **H-KPOD.** Off-BEP operation: model kpod as categories (<0.7 / 0.7–1.2 [ref] / >1.2)
  — U-shape expected, hydraulic-mode dominant. Also `frac_kpod_below_0p7_early`.
- **H-CYCLING.** `n_restarts_per_100d` and `idle_frac` → all-cause and electro-thermal
  (start-up inrush). Note idle_frac is whole-run — reverse-causation caveat applies; the
  early-window restart count is the cleaner term.
- **H-LOAD.** `load_mean_early` (test linear + quadratic) and `load_std_early` →
  electro-thermal expected.

### C3 — Mode-differentiation forest report (output: `phase_c_mode_forests`)

For every Block 2/3 joint-model survivor: one forest across the three cause groups (reuse
the B3 plotting pattern), plus the "expected mode?" verdict column. This figure set is the
confounding audit for the whole phase.

### C4 — Merged θ(t) (output: `phase_c_joint_theta`)

1. Pool the surviving covariates of Blocks 1 (from `phase_a_chem_ttfmix`), 2, 3 + mandatory
   adjusters. One stratified Cox, VIF cleanup across blocks (chemistry–operational overlaps:
   GLF vs drawdown, watercut vs kpod).
2. Schoenfeld per survivor → STANDARD / EXTENDED assignment; EXTENDED terms estimated with
   the corrected time-interaction module; if §0.3 says VBA is STANDARD-only, report the
   γ but export the STANDARD approximation with an explicit bias note.
3. Final artifact `theta_final_coeffs.csv`: covariate | β | γ | ref_value | type | block |
   window | clock — population-weighted ref values (θ=1 for the average pump), stamped
   `clock=ttf_mix`. This **supersedes** `chem_final_coeffs.csv` as the VBA handoff candidate.
4. Per-field interaction checks on the final model; sign flips flagged.

### C5 — Temporal holdout: the stack's first out-of-sample test (output: `phase_c_holdout`)

- Split by install date: **train ≤ 2023-12-31 (1,953 runs / 1,102 events), test > 2023-12-31
  (681 runs / 425 events)** — verified counts, recompute at run time.
- Refit on train only: (a) stratum-baseline-only model (the K=2/Weibull-equivalent null),
  (b) + merged θ covariates.
- Evaluate on test, censoring-aware: C-index (well-clustered), and Brier score / calibration
  at 90 / 180 / 365 d (test runs are young — horizons must respect their follow-up; report
  n at risk per horizon).
- The deliverable is the **honest delta**: how much does θ add over strata alone, out of
  sample? If the answer is "little", that is a headline finding, not a failure — report it.
  Sensitivity: cutoff 2022-12-31 (1,541/844 vs 1,093/683) to check split-choice fragility.

### C6 — Report (output: `phase_c_report`)

The story: what the completed pump brings (Block 3) → how it is run (Block 2, with the five
hypothesis verdicts) → the merged θ and what it's worth out of sample (C5) → VBA handoff
candidate + implications (deferred). §0 confirmations recorded in the header; §0.2
disclaimer verbatim; open questions updated (carry forward anything still unresolved from
§0.1).

---

## Rules

- Paths via `analysis.paths`; reusable logic in `backend/analysis/` (features → `data/`,
  models → `models/survival/`); scripts thin.
- No VBA changes, no stratum changes; §0 gates stop-and-ask if unconfirmed.
- `cd backend && python -m pytest tests/ -v` green for all new modules (kpod/restart
  derivations, holdout splitter). The 4 pre-existing collection errors
  (`analysis.plotting` refactor) are known — do not touch, do not add to them.
- Commit only when the user asks.

## Definition of done

- [ ] §0.1 semantics resolved or proxy-flags stamped; §0.2 window confirmed; §0.3 scope confirmed
- [ ] C0 features built with tests; registry + coverage heatmap refreshed
- [ ] Block 3 screened, joint-fitted, cause-specific pass + heterogeneity table published
- [ ] Block 2 ditto, with explicit verdicts on H-GLF, H-FREQ, H-KPOD, H-CYCLING, H-LOAD
- [ ] Mode-differentiation forests for every survivor
- [ ] `theta_final_coeffs.csv` (merged, stamped, superseding chem-only handoff) + EXTENDED-term policy per §0.3
- [ ] Temporal holdout: C-index + Brier/calibration deltas over baseline, both cutoffs, honestly reported
- [ ] Phase C report with §0 decisions, disclaimer, and updated open questions
