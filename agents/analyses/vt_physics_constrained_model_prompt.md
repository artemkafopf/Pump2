# Vt physics-constrained TTF model — implementation prompt

> Handoff prompt for an implementation agent. Self-contained: carries the design, data
> sources, constraints, fitting recipe, deliverables and acceptance gates. Repo: `d:\GitHub\Pump2`.

## Mission

Build a **physics-constrained Weibull AFT model** of УЭЦН time-to-failure for the Vt field
that replaces the monotone-covariate v3.1 form. The customer explicitly mandates the shape
constraints even though the current data does not statistically demand them ("bring physics
in"): life must have an **inverted-U (∩) dependence with a possible plateau** on Кпод
(max somewhere in 0.7–1.0, location = fit variable) and on frequency (max at the pump's
nominal frequency). Keep the form simple — piecewise-linear (polyline) arms; do not build
anything fancier. Parameters stay fit variables.

## Standing repo rules (non-negotiable)

- All paths via `from analysis.paths import results_dir, ...`; never hardcode paths.
- Outputs → `results_dir("production_risk_vt_physics_model")` (`tables/`, `figures/`).
- Reusable logic → `backend/analysis/workflows/production_risk/vt_physics_model.py`;
  scripts are thin wrappers with the sys.path bootstrap
  (`sys.path.insert(0, str(REPO_ROOT / "backend"))`).
- **Do NOT commit anything to git.** Leave changes in the working tree.
- Reporting standard: **RMST(0, 730) is the headline TTF, plus MRL(0); median reference
  only; never B50 alone. Always show a KM control curve next to any parametric fit.**
- Windows: run Python with `PYTHONIOENCODING=utf-8` (Cyrillic output crashes cp1251).
- Tests live in `backend/tests/`; run `cd backend && python -m pytest tests/ -v`.

## Data

Frame: `analysis.workflows.production_risk.vt_ttf_covariates.build_frame()` → `(df, coverage)`;
filter `field == "Vt"`. Clock: `vt_ttf_covariates.CLOCK_PRIMARY` (= `t_cal`), keep rows with
clock > 0. Event: `vt_ttf_covariates.EVENT_COL` (genuine failures; ГТМ/pulls censored).
Relevant columns: `ql`, `log_ql`, `freq_run`, `kpod_run`, `nominal_freq_hz`,
`contractor_group` (`brt`/`slb`/`oth`), `well_key`, `pull_reason`, `install`,
`h2s_proxy_mg_l`, `h2s_proxy_source`, `h2s_class` (legacy svod label — do NOT use as the
flag, see below).

### Sour flag — WELL-level from lab data (new definition)

The legacy `h2s_class` comes from the run-level Свод «Кислый/Некислый» label and has a known
defect: open svod rows carry no label and default to nonsour, so **30 currently-running pumps
on sour wells are misfiled nonsour** (sour class = 100% completed pulls — survivors excluded,
biasing sour life low by ~30%). The new flag must be **per WELL, from lab measurements**:

1. per-well lab H₂S: median of `h2s_proxy_mg_l` where `h2s_proxy_source == "well_median"`;
2. fallback `pad_median`;
3. fallback: legacy svod label at well level (`sour` if ANY run of the well is labeled sour).

`sour = (well_lab_h2s >= SOUR_H2S_MG_L)`, config constant, default 50 mg/l (on current data
sour q05 ≈ 122 and nonsour q90 ≈ 1.5 mg/l, so any threshold in ~2–120 gives the same split —
document this). Apply the flag to **every run of the well, including open runs**. Report a
reconciliation table vs the legacy label: wells reclassified, runs moved, and confirm the
sour class now contains open (censored) runs.

### Strata — the ONLY split

Three contractor strata × 2 sour classes (6 fits max):

| stratum | definition |
|---|---|
| `Vt_slbbrt` | `contractor_group in ("slb", "brt")` — pooled deliberately: the brt/slb gap failed the informative-censoring collider test (full-KM 522/315 but failure-only 156/147), so it is not trusted as a real service-quality difference |
| `Vt_other` | `contractor_group == "oth"` |
| `Vt_pooled` | ALL runs — the fallback stratum used when the contractor is unknown |

No other stratification. `Vt_pooled` is fit independently (not derived from the other two).

## Model form (per stratum × sour class)

Weibull AFT, S(t) = exp(−(t/η)^β), one (β, η_ref) per fit, and

```
ln η_i = ln η_ref
         + γ_q · ( ln clip(Ql_i, 47, 823) − ln 250 )        # monotone log-Ql arm, as in v3.1
         − Pk(Kpod_i)                                        # ∩ penalty, Кпод
         − Pf(freq_i)                                        # ∩ penalty, frequency
```

Piecewise-linear hinge penalties (≥ 0 by construction ⇒ life multiplier exp(−P) ≤ 1 with its
maximum ON the plateau — the inverted U is guaranteed structurally, not hoped for):

```
Pk(k) = aL · max(k_lo − k, 0) + aR · max(k − k_hi, 0)
        aL, aR ≥ 0;   0.70 ≤ k_lo ≤ k_hi ≤ 1.00            # plateau [k_lo, k_hi] inside 0.7–1.0
Pf(f) = bL · max(f_nom − f, 0) + bR · max(f − f_nom, 0)
        bL, bR ≥ 0;   f_nom = per-run nominal_freq_hz, fallback 50 Hz
```

Notes:
- An arm fitting to ~0 is a legitimate outcome (plateau extends to the data edge). Expect
  this: prior analysis found NO overspeed penalty above nominal (hinge >50 Hz p=0.93,
  >55 Hz p=0.52) and a monotone, not U, Kpod trend — so `bR` and possibly `aR` will hit 0.
  Report it, don't fight it.
- Check `nominal_freq_hz` coverage/variance within Vt first; if it is effectively constant
  50, say so and use 50.
- Reference point for η_ref: Ql=250, freq on the plateau, Kpod on the plateau (penalties = 0
  there by construction, so η_ref IS the plateau life scale).
- Sour side: fit the same form; if covariate arms collapse to 0 (prior finding: operating
  covariates null on sour), the model degrades gracefully to intercept+Ql or intercept-only.

## Fitting recipe

`lifelines` cannot fit hinge knots — write a direct censored-Weibull MLE:

- log-lik: `Σ_events [ln β − ln η + (β−1)(ln t − ln η) − (t/η)^β] + Σ_censored [−(t/η)^β]`
  with η_i from the formula above.
- **Profile the knots**: outer coarse grid over (k_lo, k_hi) ∈ {0.70, 0.75, …, 1.00},
  k_lo ≤ k_hi; inner smooth problem (β, η_ref, γ_q, aL, aR, bL, bR) via
  `scipy.optimize.minimize` (L-BFGS-B), positivity via bounds. This avoids optimizing a
  non-smooth objective.
- **Multi-start the inner problem** (≥5 starts incl. the v3.1 solution as a warm start) —
  this family of likelihoods is known to be multimodal; a single-seed fit is a sample, not
  a fit.
- Complete-case fit on rows with ql+freq+kpod all present; ALSO fit a **Ql-only fallback**
  per stratum (same clip/reference) for deployment when regime data is missing. Report
  covariate coverage per stratum.
- Uncertainty: **bootstrap ≥ 200 resamples** (resample wells, not runs, to respect
  within-well correlation) for CIs on η_ref, plateau location, and the reported RMST/MRL.

## Comparisons & honesty gates (report, do not hide)

1. Log-lik/AIC table: constrained ∩ model vs (a) v3.1 monotone linear form, vs (b)
   unconstrained quadratic — state plainly where the constraint binds and what it costs.
   The customer knows the statistics do not demand the ∩ shape; the report must still show
   the price honestly.
2. Binned empirical control: KM medians / RMST by Kpod bins and freq bins overlaid on the
   fitted multiplier curve, per stratum (nonsour at least).
3. Cross-check event counts vs `esp_models.csv` n_failures (standing rule).
4. Confirm the sour reclassification moved sour RMST(0,730) up ~25–35% vs the legacy label
   (expected from the misfiled-open-runs fix); if it doesn't, investigate before shipping.

## Deliverables

1. `backend/analysis/workflows/production_risk/vt_physics_model.py` — frame prep (well-level
   lab sour flag), constrained MLE, bootstrap, tables/figures writers; thin CLI
   `scripts/run/vt_physics_model.py`.
2. Tables (`tables/`): `params.csv` (per stratum×class: β, η_ref, γ_q, aL, aR, k_lo, k_hi,
   bL, bR, f_nom policy, n, events, loglik, AIC + bootstrap CIs); `ttf_summary.csv`
   (median / MRL(0) / RMST(0,730) at reference and on a small Ql×Kpod×freq grid);
   `model_comparison.csv`; `sour_flag_reconciliation.csv`.
3. Figures (`figures/`): life-multiplier vs Кпод and vs частота (fitted polyline + binned
   empirical overlay + rug of data), KM+fit per stratum×class, sour-flag old-vs-new KM.
   Figure text in Russian.
4. Tests in `backend/tests/` (extend `test_production_risk.py` or a new file): multiplier
   ≤ 1 everywhere with max on plateau; plateau within [0.7, 1.0]; penalties zero at
   reference; a small synthetic-data recovery test of the MLE (simulate from known params,
   assert recovery within tolerance).
5. A short findings note `docs/notes/vt_physics_model_findings.md`: parameters, where
   constraints bind, comparison vs v3.1, and the new sour-flag impact. Russian.

## Known gotchas

- `WeibullAFTFitter` dies on NaN covariates — but you're writing your own MLE; still drop/flag
  NaNs explicitly.
- `freq_run`/`kpod_run`/`ql` are one affinity-law chain (collinear); with the constrained
  form this is tolerated, but expect unstable individual arms — bootstrap will show it.
- 31% of Vt rows miss `ql`; more miss `kpod_run`. Never silently drop — report coverage.
- `pbubble_atm`/`nominal_freq_hz` are field labels in disguise ACROSS fields; within Vt fine.
- Applicability bounds to carry into outputs: Ql 47–823 (clip), freq 31–56, Кпод 0.2–1.2.
- Do not left-truncate anything; do not touch Mc (this is Vt-only work).
