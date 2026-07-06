# Phase B — Competing Risks: Cause-Specific Survival by Failed Node

## Purpose

Everything fitted so far models pooled time-to-any-failure. But all 1,527 failures carry a
component label (`failed_node`), and different covariates physically attack different
components: chemistry/solids attack the pump and tubing; heat and electrical stress attack
the cable and motor. Pooling dilutes every covariate effect by mixing it with modes it cannot
influence — the leading suspect for the weak Block 1 chemistry results.

Phase B splits the hazard by failure-mode group and answers three questions:

1. **Does chemistry act on the pump?** (re-test Block 1 against the hydraulic-mode hazard only)
2. **What does H₂S attack?** (per-mode H₂S effect in Vt — the mode mix hints cable/motor is hit
   at least as hard as the pump)
3. **What is the mode mix per stratum?** (absolute cumulative incidence per mode — repair
   resource planning needs this, not just pooled B50)

Context (read first):

```
results/analysis_review/2026-07-06/README.md         — review; Phase B rationale in §Plan
agents/analyses/phase_a_cox_foundation.md             — Phase A spec (design decisions inherited)
agents/analyses/phase_a_cox_foundation_results.md     — Phase A outcomes incl. T6 correction
results/phase_a_za_sour_check/2026-07-06/             — non-Vt sour evidence (gate §0.2)
CLAUDE.md                                             — path rules
```

---

## §0 — CONFIRM WITH USER BEFORE IMPLEMENTING

### 0.1 Failure-mode group mapping (the load-bearing physical assumption)

Every cause-specific result is conditional on this table. Counts are from the warehouse
post-hygiene (2026-07-06, trailing spaces trimmed; recompute at run time — the Ic alias merge
added 7 runs / 6 failures).

| Group | failed_node values | Failures | Physical rationale |
|---|---|---:|---|
| **hydraulic** | ЭЦН, НКТ, Газосепаратор, Диспергатор, Входной модуль | ~689 | Flow path: scale, corrosion, abrasion, gas interference |
| **electro-thermal** | Кабельная линия, ПЭД, ТМС | ~603 | Electrical/thermal: insulation, overheating |
| **protector** | Гидрозащита | ~186 | Seal section: its own wear mechanism |
| **other** | Клапан сливной/обратный, Подвесной патрубок, Входной/переводник, Дополнительное оборудование, remaining small labels | ~49 | Heterogeneous small categories |

Debatable assignments the user must confirm or move:

- **ТМС** (downhole sensor) → electro-thermal? It is electronics, but its failure rarely kills
  the installation the same way; n=9, low impact either way.
- **Газосепаратор / Диспергатор** → hydraulic? They are rotating flow-path equipment (gas
  handling); alternative view: separate "gas-handling" group, but n=68 total says fold in.
- **НКТ** (tubing) → hydraulic? Tubing failures are corrosion/leak driven (chemistry-relevant),
  but tubing is not the ESP itself. Alternative: own group (n=147) or exclude from ESP analysis.
  **Default: keep in hydraulic; run the B3 chemistry model with and without НКТ as sensitivity.**

Per-stratum × group event counts (field-level, pre-alias-merge — regenerate as `b1_inventory`):

| Stratum | hydraulic | electro-thermal | protector | other |
|---|---:|---:|---:|---:|
| Ya_nonsour | 316 | 271 | 123 | 21 |
| Vt_nonsour | 83 | 79 | 15 | 3 |
| Vt_sour | 36 | 45 | 5 | 2 |
| Az_nonsour | 80 | 46 | 14 | 4 |
| Ic_nonsour | 60 | 74 | 10 | 3 |
| Za_nonsour | 75 | 49 | 9 | 10 |
| Mc_nonsour | 19 | 15 | 6 | 1 |
| Da_nonsour | 15 | 15 | 2 | 0 |

Zero unlabeled failures — coverage is complete.

### 0.2 Sour classification outside Vt (decision gate from Phase A item 4)

The Phase A check (`results/phase_a_za_sour_check/2026-07-06/`) shows flagged-sour runs
(h2s_proxy > 10 mg/l) fail materially faster **in all three candidate fields**, on operating time:

| Field | n flagged | KM50 flagged | KM50 rest | log-rank p | Flag source |
|---|---:|---:|---:|---:|---|
| Az | 66 | 112d | 304d | 0.001 | 88% well-level medians — **real measurements** |
| Ya | 39 | 96d | 258d | 0.0015 | mixed (21 pad / 18 well) |
| Za | 112 | 131d | 220d | 0.049 | half pad-median; curves converge after ~400d |

This is not imputation echo (Az especially). Options for Phase B:

- **(a) Status quo** — sour strata remain Vt-only; the flagged runs stay inside `*_nonsour`
  baselines (known blend, understates nonsour B50 for Az/Za).
- **(b) Extend sour strata** to Az/Za (new strata Az_sour n=66, Za_sour n=112) — cleaner
  baselines but thinner strata and a breaking change for every downstream table and VBA.
- **(c) Recommended for Phase B:** keep strata as-is, add `is_sour_flagged` as a **covariate**
  in all cause-specific Cox models. Captures the effect without redefining baselines;
  the stratum redesign decision moves to Phase C/E with this evidence in hand.

**User picks (a)/(b)/(c) before B3 starts.**

---

## Design decisions (inherited from Phase A — not re-litigated)

- Clock: `ttf_mix` everywhere; `pct_mixed_clock` shown in every stratum inventory.
- One joint stratified Cox per cause group: `strata=['stratum_key']`, global β,
  `cluster_col='well_key'`, `robust=True`.
- Covariate windows: `t0` / `early` per the Phase A registry (`run_covariates.py`);
  never whole-run means in a baseline Cox.
- Every operational number carries a bootstrap interval (reuse `models/survival/bootstrap_ci.py`).
- Labels: warehouse already trimmed (2026-07-06) and ingest wiring applied — but B1 must
  assert `failed_node == TRIM(failed_node)` and fail loudly if new debris appears.

## Statistical rules specific to competing risks

1. **Cause-specific hazard** (fit cause k treating other-cause failures as censored at their
   failure time) is the estimand for **etiology** — all Cox models in B3/B4 are cause-specific.
2. **Absolute risk** uses the **Aalen–Johansen** cumulative incidence function (CIF), never
   `1 − KM_cause` (which overstates incidence when competing risks exist — with ~40% of
   failures being "the other group", the overstatement here is material).
3. Group differences in CIF: **Gray's test** (or bootstrap CIF bands), not log-rank on
   cause-censored data.
4. **Fine–Gray** subdistribution model: optional sensitivity for the planning layer only
   (B5); do not mix its coefficients with cause-specific HRs in one table.
5. Cause-specific parametric fits: single Weibull per cause-stratum by default; K=2 mixture
   only where the cause-stratum has ≥40 events (per Phase A ΔAIC discipline, report
   `delta_aic` for K=1 vs K=2).

---

## Tasks

### B0 — Reusable machinery (before any fitting)

1. **`backend/analysis/data/failure_modes.py`** — the §0.1 mapping as a constants dict +
   `assign_mode_group(failed_node) -> str`, with tests. The confirmed version of the table is
   the single source of truth; VBA/report code reads its CSV export, never re-hardcodes it.
2. **`backend/analysis/models/survival/time_interaction_cox.py`** — factor the corrected
   event-time episode split + `CoxTimeVaryingFitter` wrapper out of
   `scripts/run/phase_a_extended_cox_fix.py` into a reusable, tested module.
   **Preserve the correction**: episodes cut at every distinct failure time, covariate =
   value at the shared risk-set time. Add a regression test that a synthetic dataset with a
   known flat HR recovers γ ≈ 0 (this is the test that would have caught the −0.851 bug).
3. **Loader**: `mart__weibull_input` ⋈ `raw__v03_runs.failed_node` (by `row_id`) + mode group +
   `event_k` indicators per group + Phase A covariates via `run_covariates.py`.

### B1 — Inventory (output: `phase_b_inventory`)

Mode-group × stratum × clock event counts (the §0.1 table, regenerated post-hygiene, both
field-level and field×contractor); `pct_mixed_clock` per stratum; fit-eligibility flags
(≥40 events → K=2 eligible; ≥20 → single Weibull; <20 → nonparametric only). Assert zero
unlabeled failures.

### B2 — Nonparametric layer (output: `phase_b_cif`)

Per field-level stratum: Aalen–Johansen CIF per mode group (stacked CIF plot per stratum —
the four groups partition all-cause incidence); cause-specific Nelson–Aalen hazards;
mode-mix table (share of failures by group at 90d / 365d / end-of-follow-up, from CIF not raw
counts). Gray's test: sour vs nonsour CIF per mode within Vt.

### B3 — Cause-specific Cox: does chemistry act on the pump? (output: `phase_b_cause_cox`)

For each of {hydraulic, electro-thermal, protector}:

- Refit the Block 1 survivors + near-misses (`log_h2s_proxy`, `log_glf` [early window],
  watercut, КВЧ complete-case, salt/gypsum proxies, `is_sour_flagged` if option (c)) as a
  cause-specific stratified Cox.
- Output one **forest plot per covariate across the three cause groups** — the money figure:
  chemistry β should be nonzero for hydraulic and ≈0 for electro-thermal if the physical
  hypothesis holds. A covariate significant in *all* modes equally is a red flag for residual
  confounding (field/vintage) — say so in the report.
- Sensitivity: hydraulic with/without НКТ (§0.1); heterogeneity per field for any survivor.

### B4 — What does H₂S attack in Vt? (output: `phase_b_vt_h2s_modes`)

- Cause-specific Cox HR for sour within Vt, per mode group (hydraulic n≈36+83, el-thermal
  n≈45+79 events).
- Time-interaction γ per mode via the B0.2 module (event-time split). Expect wide CIs
  (~80–124 events per mode) — report them honestly; no curve-regression p-values.
- CIF comparison sour vs nonsour per mode (Gray's test) — the operational statement is
  "by 90 days, X% of sour installs have lost the cable/motor vs Y% nonsour".

### B5 — Parametric planning layer (output: `phase_b_cause_weibull`)

- Cause-specific Weibull per stratum (K=1 default; K=2 + ΔAIC where ≥40 events).
- Convert to CIF-consistent quantities: time to 10% / 25% cumulative incidence per mode per
  stratum, with bootstrap CIs (wells resampled). Note explicitly that per-mode parametric
  S_k(t) does **not** equal 1−CIF_k; the CIF combination uses all-cause survival.
- Reconciliation check: sum of cause-specific cumulative hazards vs the pooled all-cause fit
  from the ESP survival stack — mismatch >5% at B50 means an error somewhere; investigate.
- Optional: Fine–Gray per mode as sensitivity on the incidence quantiles.

### B6 — Report (output: `phase_b_report`)

One markdown report telling the story: mode mix per stratum → what chemistry does and
doesn't touch → what H₂S attacks → planning-ready CIF tables. Every claim with an interval;
every table stamped `clock=ttf_mix`. Close with explicit VBA implications (deferred — no VBA
changes in Phase B) and updated open questions.

---

## Rules

- Paths via `analysis.paths`; heavy logic in `backend/analysis/`; scripts thin.
- No VBA changes; no stratum-definition changes beyond the user's §0.2 pick.
- `cd backend && python -m pytest tests/ -v` green for all new modules (mode mapping,
  time-interaction Cox regression test, loader).
- If any §0 confirmation is still pending at implementation time, stop and ask — do not
  default silently.

## Definition of done

- [ ] §0.1 mapping confirmed by user and frozen in `failure_modes.py` (+ CSV export)
- [ ] §0.2 sour-handling option picked and recorded in the report header
- [ ] Time-interaction Cox module extracted with the γ≈0 synthetic regression test
- [ ] B1 inventory published, zero unlabeled failures asserted
- [ ] Aalen–Johansen CIF + mode-mix tables per stratum (no 1−KM anywhere)
- [ ] Cause-specific Cox forest plots: chemistry × mode, H₂S × mode
- [ ] Vt per-mode H₂S HRs + γ with bootstrap CIs
- [ ] CIF-based incidence quantiles with CIs; reconciliation vs pooled all-cause fit passes
- [ ] Phase B report published; VBA implications listed, not implemented
