# Vt TTF ~ operating covariates — workflow plan

Date: **2026-07-22**. Status: **EXECUTED 2026-07-22** — the Свод rework landed (running/
censored runs native, brine phantoms filtered), unblocking the fits. Implementation:
`backend/analysis/workflows/production_risk/kpod_features.py` + `vt_ttf_covariates.py`,
CLI `scripts/run/vt_ttf_covariates.py`. Results + interpretation:
`results/production_risk_vt_ttf_covariates/2026-07-22/` (see `FINDINGS.md`). Memory:
`project_vt_ttf_covariates`.

> **Verdict:** **running frequency (ESP speed) is the strongest operating signal** —
> faster pumps fail sooner (HR ≈ 1.02/Hz, p=0.03, replicates on Ya p=0.008) and it
> subsumes Kpod (Kpod → null when both are in). No Kpod U-shape / no safe valley; the
> causal-anchor (t0) versions of both frequency and Kpod are null, so the effects are
> run-developed not install-setpoints, and frequency/Kpod/Ql are one collinear
> affinity-law chain (observational). Ql also replicates; GLF null everywhere. See
> `FINDINGS.md` for the full account. *(Frequency was added as a first-class covariate
> after the initial run; the plan's covariate table originally carried it only inside the
> freq-adjusted Kpod.)*

---

_Original plan below (kept as the executed spec)._

## 0. Goal

Derive **simple mathematical relations for how TTF changes with ESP operating
parameters** — not a forecast, not a prognosis layer. Target population: **Vt (sour +
nonsour)**, with separate per-stratum results (h2s_class × contractor_group) alongside
the pooled-β stratified fit. Deliverable = a formula table of life multipliers, not just
hazard ratios (§5).

## 1. Covariates, roles, sources

| covariate | role | source | notes |
|---|---|---|---|
| contractor | **stratum**, never a β | Свод «Принадлежность» → `contractor_group` | contractor is a service-quality label, not an operating parameter — stratify, don't regress |
| h2s flag (sour/nonsour) | **stratum** | `chemistry_run_features._derive_h2s_class` (threshold 10 mg/l, field fallback) | never pool sour/nonsour (3× life gap AND shape differ — `project_vt_weibull_scan`) |
| h2s continuous (mg/l) | covariate **within sour only** | `h2s_proxy_mg_l` + `h2s_proxy_source` | ⚠ circularity: the flag is derived from this same proxy — including flag-as-stratum AND continuous-h2s across strata double-counts. Use continuous h2s only inside the sour stratum, and only where `h2s_proxy_source` = measured (field-fallback rows carry no within-stratum information) |
| Kpod | main effect, **shape analysis** (§3) | telemetry: `qliq / (nominal_flow_m3d · freq/50)` | BOTH variants carried as covariates (user decision): plain `kpod_mean` and freq-adjusted `kpod_freq_mean` — their disagreement is itself informative (freq-trim vs true off-design) |
| Kpod exposure shares | covariates | telemetry, run-long with tail guard (§2) | `frac_days_kpod_below_0.7`, `frac_days_kpod_above_1.1` (thresholds refined after §3 binning) — chronic-exposure alternative to the mean; compare which correlates better |
| Ql | covariate | t0 window (`t0_covariates.build`) | known fleet-level signal: +0.314/SD, p=1e-11, survives stratification (`project_cox_glf_null`) — expect it to hold; the question is the Vt-specific magnitude |
| Qg | covariate | t0 window | log(1+Qg); fleet-level weaker than Ql |
| GLF = Qg/Ql | covariate | t0 window | fleet-level NULL after stratification (p=0.20) — re-test within Vt (gassiest field, glf 179/211) but expect null; do NOT use the techregime `gas_factor` column (gas per OIL, control only) |
| КВЧ | covariate | lab.sqlite `lab_samples` via `chemistry_run_features` (in-run, then 180d pre-install fallback) | check Vt coverage first (8,981 rows fleet-wide); mark sample-timing source per run |

**Nominal-point imputation rules (user decision):**
* `nominal_freq_hz` missing → **assume 50 Hz**;
* `nominal_flow_m3d` missing → take the best value from the ESP equipment database
  (`equipment_big` — Big «Насос» sheet), and as a last resort parse the pump type name
  (ЭЦН5-80 → 80 м³/сут);
* every run gets a `kpod_qnom_source` column ∈ {mart, big, type_parse, missing} — the
  U-shape claim must be shown robust to dropping the `type_parse` tier.

## 2. Windowing rule (user decision: run-long features, tail-guarded)

Chronic loading over the WHOLE run is the quantity of interest — a 30-day snapshot can
misrepresent a pump that ran off-design for two years. So every telemetry covariate is
computed in a **feature family**, and the analysis compares which member correlates
better rather than pre-committing to one:

* `*_t0` — first 30 op-days (the causal anchor, immune to degradation leakage);
* `*_run` — mean over the whole run **excluding the last 30 calendar days before stop**
  (the tail guard; 30 not 60 — user decision, a longer guard erases too many short-lived
  runs, which are exactly the infant failures);
* `frac_days_*` — share of op-days in a Kpod band, same tail guard.

**Why the tail guard is non-negotiable.** A failing pump's qliq collapses BEFORE the
pull, so an unguarded run mean contains the failure read backwards (`t0_covariates`
docstring). Worse, the contamination is **differential**: event runs carry a death tail,
censored runs don't — an unguarded run-long feature "wins" any correlation contest
partly mechanically. Guarded comparison is fair; unguarded run means are disallowed.
Phase D is the precedent (`project_phase_d`): leakage-proof within-run dynamics gave an
honest null where naive features looked predictive.

Two artifacts to report alongside:
* **exposure-length asymmetry** — a 20-day run's "run mean" is 20 noisy days, an
  800-day run's is stable; and after the 30-day guard, runs shorter than ~30 days have
  NO guarded run-long value at all (they fall back to `*_t0`, flagged in a source
  column). Sensitivity pass: re-score the §3.5 contest with a 60-day guard — if the
  winner flips, the "signal" was living in days 30–60 before the pull, i.e. still
  plausibly degradation, and the claim downgrades;
* **t0-vs-run divergence** — `kpod_run − kpod_t0` is itself a drift feature; if it
  carries the signal, the story is "degrading operating point", not "chronic setpoint".

КВЧ/h2s keep the chemistry in-run/pre-install rule.

**Trap (hit before, `project_cox_glf_null`):** `proc__daily_merged` starts 2018 —
requiring a full t0 window silently deletes the infant-failure layer (failures ≤3d went
68 → 1 in Phase A). Rules:
* coverage report (`T0Coverage`) is mandatory output, split by stratum AND by event/censor;
* runs with a partial window keep partial-window means + `t0_n_days`; runs with no window
  keep NaN and are reported, never dropped silently;
* sensitivity fit: full-window-only vs any-days — if signs flip, say so.

## 3. Kpod U-shape protocol

Physics prior: Kpod ≪ 1 = downthrust/low-efficiency/heating; Kpod > 1 = upthrust. The
user's question: **is Kpod > 1 statistically bad**, and where is the safe valley.

Ladder, per stratum where events allow, else stratified-pooled:

Both `kpod_mean` and `kpod_freq_mean` run through the whole ladder (user decision), in
their `_t0` and guarded `_run` versions (§2). If plain and freq-adjusted disagree on
shape, the disagreement localizes WHERE frequency trim matters — report it, don't
average it away.

1. **Binned descriptive**: bins `[<0.5, 0.5–0.7, 0.7–0.85, 0.85–1.0, 1.0–1.15, >1.15]`;
   per bin: n runs / n events / censored share / KM RMST(0) / event rate per 1000
   op-days. This is the honest picture before any smoothing.
2. **Quadratic Cox**: `β1·K + β2·K²` in the stratified model — test β2 > 0 (convexity).
3. **Penalized spline Cox** (df 3–4): plot log-HR vs Kpod with CI band per stratum —
   the shape figure that either shows the U or doesn't.
4. **Hinge decomposition** (the deliverable form): `left = (0.85 − K)₊`,
   `right = (K − 1.0)₊` as two covariates → two slopes with CIs. Directly answers "left
   penalty per 0.1 of underloading" and "right penalty per 0.1 above nominal" in one
   line each. Hinge knots {0.85, 1.0} are priors — refit at spline-suggested knots as
   sensitivity.
5. **Exposure-share vs mean contest**: `frac_days_kpod_below_0.7` and
   `frac_days_kpod_above_1.1` (guarded, §2) against the mean-based forms — same
   stratified Cox, compare partial-likelihood / concordance and coefficient stability.
   Shares and mean answer different questions (intermittent excursions vs chronic
   level); if shares win, the mechanism story is dose-of-excursion, and the formula
   table quotes "per +10% of days in the band". Band thresholds re-derived from the
   spline shape (step 3) before the contest is scored, not fixed by prior.
6. Report the risk-minimizing Kpod with bootstrap CI.

Guards: the U-shape is overfit-prone — permutation null for the spline (refit on
permuted Kpod, compare deviance gain); and Kpod = Ql/Qnom is **mechanically correlated
with Ql** — always fit Kpod-only, Ql-only, and joint models; if the joint fit kills
both, the data can't separate rate from loading (Qnom variation is the only thing that
separates them — report its spread).

## 4. Model stage

* **Estimand**: cause-specific genuine failure; ГТМ/ППР pulls and running pumps censored.
  Running (censored) runs come from the NEW Свод — that's the blocking dependency.
* **Clock**: `t_cal` (the Vt refit decision, `project_time_scales`); one sensitivity
  pass on `t_mix`. Never the legacy mixed `tte`.
* **Model**: stratified Cox, strata = `h2s_class × contractor_group` (free baseline
  hazard per stratum), global β; then per-stratum β for stability. Plain PH for β —
  **no X·log t interactions** (the collinearity fakes nulls, `project_cox_glf_null`).
* **Scaling**: log Ql, log(1+Qg), standardized; report HR per 1 SD AND per natural unit
  (per 10 м³/сут, per 0.1 Kpod) — the formula table needs natural units.
* **Diagnostics**: Schoenfeld PH test per covariate; VIF / correlation matrix (Ql–Kpod,
  Qg–GLF pairs especially); events-per-parameter ≥ ~10 per stratum or the stratum gets
  descriptive treatment only (expect Vt_sour_slb/oth to be thin).
* **Missing data**: complete-case per model, with the coverage table proving the
  complete-case subset isn't a biased slice (compare KM of covered vs uncovered runs).

## 5. From β to "simple relations" (the deliverable)

1. **PH route** (no refit needed): `S(t|X) = S0(t)^exp(βΔX)` on each stratum's shipped
   Weibull baseline → life multiplier per covariate as RMST-ratio and median-ratio at
   representative ΔX (e.g. Kpod 0.6 vs 0.9, Ql +10 м³/сут).
2. **Weibull AFT companion fit** of the same design: `η(X) = η0 · exp(γ·X)` — γ IS the
   life multiplier, the most readable "simple math relation". Report AFT vs PH-derived
   multipliers side by side; disagreement >~15% means PH is violated for that covariate
   and the AFT number is the one to quote.
3. Final formula table, one row per surviving covariate:
   `TTF multiplier ≈ exp(γ·ΔX)` with CI, natural units, stratum validity, and the
   support range it was estimated on (no extrapolation outside observed support).
4. Reporting standard: RMST(0) + MRL(0), never B50 (`feedback_report_rmst_mrl`).

## 6. Honesty gates (all mandatory before any claim ships)

* **Replication**: every Vt-surviving covariate re-checked on Ya and on Mc (2024+
  cohort). Known precedent: Kpod/Ql "looked real on Mc alone, died on replication"
  (`project_hazard_catboost`). A Vt-only effect that fails replication is reported as
  Vt-specific-unconfirmed, not dropped and not generalized.
* **Support overlap** per stratum before any pooled statement — quantile overlap and
  histograms, NOT min..max (`support_overlap`'s min..max gives false transferable=True).
* **Confounding disclosure**: wells are not randomized to Kpod/Ql — low Kpod ↔
  low-productivity wells ↔ possibly different completion/zone. The ion-load precedent
  (`project_ion_load_vs_concentration`): a "load" trend that was qliq × uptime in
  disguise. State the observational caveat on every formula row.
* `n_failures` of every fit asserted against `esp_models.csv` counts (standing
  population trap).

## 7. Outputs

`results_dir("production_risk_vt_ttf_covariates")`:
* `tables/`: coverage by stratum, Kpod binned table, hinge slopes, Cox summaries
  (per-stratum + stratified), AFT-vs-PH multiplier comparison, final formula table,
  replication verdicts;
* `figures/`: spline log-HR per covariate per stratum with CI, binned KM for Kpod,
  support histograms per stratum, covariate correlation heatmap;
* memory update (`project_*` note) once results exist.

## 8. Open items to resolve at execution time

1. КВЧ and h2s-measured coverage counts on Vt specifically (may kill those covariates).
2. Kpod telemetry coverage per Vt stratum (nonsour_brt is the thick one; sour strata thin).
3. Whether the reworked Свод carries per-run h2s / КВЧ natively or we keep the
   lab.sqlite join.
4. Кэкспл as a covariate (user intent noted earlier) — same design slots in as one more
   column, but it is exposure-like, not an operating setpoint; keep out of v1.
