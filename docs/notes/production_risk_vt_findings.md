# Верхнетирский (Vt) — findings

Дата: 2026-07-16, **существенно исправлено 2026-07-17**. Session goal: analyse Vt in its
own right — the «Слой» Bt/Ic split, the sour/non-sour split, `Группа исполнения` (Н2/Н3),
and then fit spike+constant.

**Nothing shipped changed.** New code is additive; the bundle, calibration factors and
EXE are untouched. Companion to `production_risk_mc_weibull_shape_findings.md` (Mc) —
**see §7: two of that doc's central claims do not survive the population fix.**

---

## Verdict (read this first)

1. **«Слой» is УН_Месторождение, and месторождение does NOT drive failure** — a
   well-powered null (HR 1.10 [0.79–1.52], p=0.59). The model's existing УН-keyed
   strata are the right axis. **Do not regroup Vt by месторождение.**
2. **`esp_population.py` had THREE population defects, all now fixed.** They biased the
   fitted hazard down and manufactured most of the "findings" this session started with.
   **This is the main result of the session.**
3. **Н3 buys nothing measurable**, on Vt as on Ya — confounding by indication.
4. **On the corrected population the shipped bundle is FINE on Vt and Ya.**
   Spike+constant adds nothing: Vt 257 vs 261 workovers, Ya 137 vs 133, calibration
   identical. The earlier "B fixes sour / B under-predicts" results were both artifacts.
5. **The only real form defect that survives is the infant spike** — the shipped Weibull
   delivers 0.73–0.78 of the empirical 0–30 d hazard. The claimed 1.4–1.8× plateau
   over-statement does **not** survive (corrected: 0.96–1.29).
6. **Day-0 startup failures are a distinct third mode the 5-column schema cannot hold.**
7. **Vt does not need a mixture at all** — a K=2 is not justified over a single Weibull at
   ANY cut, including c=0 (§8). Across Mc/Vt/Ya the second component's only job is
   absorbing the day-0 mass.

---

## 1. «Слой» = УН_Месторождение — and the model keys on the wrong half

`Слой` decodes as `{УН}_{Месторождение}`: `Vt_Bt` = Верхнетирский УН + Большетирское НМ,
`Vt_Ic` = Верхнетирский УН + Ичёдинское НМ, `Zy_Ic` = Западно-Ярактинский УН +
Ичёдинское НМ, `Bt_Bt` = Большетирский УН + Большетирское НМ.

**The well-code prefix is the УН, not the месторождение.** The two are crossed, so both
effects are identifiable:

| | Большетирское НМ | Ичёдинское НМ |
|---|---|---|
| Верхнетирский УН | `VT_` (198 nonsour + 182 sour) | `VT_` (258) |
| Западно-Ярактинский УН | — | `IC_` (276) → modelled as field **`Ic`** |
| Большетирский УН | `BT_` (22) → **`Global_Pooled`** | — |

Свод's own «Месторождение» column carries **one value per prefix** (VT → 480 rows, all
«Верхнетирское»), so it cannot see this split. **The ПП plan is the only source.**
Added `crosswalk.well_deposit_map()` / `well_un_map()` for it.

### The test (`scripts/run/production_risk_vt_deposit_test.py`)

Each contrast holds the other axis fixed — the full-design Cox cannot answer cleanly
because its УН reference cell (Большетирский, 4 events) is too thin, and a likelihood-ratio
test is invalid under clustering (the partial likelihood ignores the within-well correlation
the cluster-robust SEs absorb). Cluster-robust Wald, adjusted for contractor + vintage:

| holding fixed | axis tested | HR | 95% CI | p | runs / events |
|---|---|---|---|---|---|
| УН = Верхнетирский, nonsour | **месторождение** | **1.10** | 0.79–1.52 | **0.59** | 438 / 205 |
| месторождение = Ичёдинское | **УН** | **0.62** | 0.47–0.83 | **0.0014** | 527 / 277 |

**Месторождение is a well-powered null; УН carries a real ~38% difference.** PH holds for
every covariate. Also fitted: **sour HR ~2.3**, contractor Борец **~0.52** /
Шлюмберже **~0.67** vs прочие, vintage **~0.87**/year.

*(Both contrasts re-run on the corrected population of §2 — the verdict is unchanged from
the first pass, which is the one conclusion here that the population defects did not
touch.)*

**Consequence:** `Vt` stays one field, `Ic` stays separate from `Vt`, «Слой» stays unread.
`BT_` → `Global_Pooled` remains defensible — it is its own УН, and месторождение does not
license borrowing Vt's curve (22 runs / 4 events anyway).

## 2. THREE population defects in `esp_population` — the main result

Each one biases the fitted hazard **down** or distorts a stratum, and between them they
manufactured most of what this session first "found". All fixed; 6 tests in
`backend/tests/test_esp_population.py`.

**(a) Every open run stamped `nonsour`.** Big carries no «Кислый/Некислый», so all 733
open runs arrived nonsour — misfiling **52 censored sour Vt runs**. Sour lost its
survivors (event rate **82%**, 111/135, hazard 0.2–0.35/month) while nonsour gained 52
event-free runs. Ичёдинское has **zero** sour wells, so it was untouched, and that
asymmetry **alone manufactured an apparent 2× Bt/Ic gap**. Fixed: runs inherit sour from
their well's Свод history (`well_sour_map`).

**(b) Big-only CLOSED runs were never loaded at all.** `build()` took Свод + Big *open*
runs only, so failures Big recorded and Свод never did were simply absent:

| field | Big-only open | Big-only **closed** |
|---|---|---|
| **Ya** | 166 | **619** |
| Vt | 132 | 26 |
| Mc | 56 | 5 |

Ya lost **a third of its events** (1596 runs / 784 events vs the registry's 2118 / 1192).
Mc and Vt barely noticed — **which is exactly why the defect survived the Mc session**.

**(c) Open runs were not deduped against Свод.** **171 of 733** Big open rows sat within
±7 d of a Свод install for the same well — the same physical run counted twice, as free
censored exposure. Vt was worst (**53**). This inflates the exposure denominator, which
**deflates the empirical hazard** — and since the model/empirical ratio divides by it, it
**inflates every "the model over-states the hazard" ratio.** That is the artifact behind
§4 and §7.

Fixed by `load_big_runs()`: all Big strict-ESP runs (open **and** closed) that Свод does
not already carry, deduped against Свод **and** V03 by (well, install ±7 d) — matching the
shipped refit's own population construction.

**Result — the population now reconciles with the registry:**

| stratum | before | after | registry | after/registry (events) |
|---|---|---|---|---|
| `Ya_nonsour_Pooled` | 1596 / 784 | **2166 / 1226** | 2118 / 1192 | **1.03** (was 0.66) |
| `Ya_nonsour_brt` | 988 / 440 | **1458 / 782** | 1440 / 770 | **1.02** |
| `Vt_nonsour_Pooled` | 480 / 203 | **476 / 209** | 452 / 198 | **1.06** |
| `Vt_sour_Pooled` | 187 / 111 | **165 / 110** | 164 / 111 | **0.99** |
| `Ic_nonsour_Pooled` | — | 286 / 159 | 275 / 151 | 1.05 |
| `Az_nonsour_Pooled` | — | 328 / 169 | 315 / 164 | 1.03 |

Mc reads 197/51 vs 166/42 (**1.21**) — expected, the registry's Mc row is vintage-selected
to 2023+ (`mc_window`).

**Rule going forward: cross-check any new field's run/event counts against
`esp_models.csv` before trusting a fit off this population.**

## 3. `Группа исполнения` Н2/Н3 — no evidence of benefit (replicates Ya)

`scripts/run/production_risk_vt_exec_group.py`. Big join 97.3% (649/667).

Assignment is confounded by indication (Н3 = 40% of sour vs 28% of nonsour) and, within
sour, nearly collinear with contractor:

| sour | Н2 | Н3 |
|---|---|---|
| Борец | 29 | 25 | ← the only balanced contrast |
| Шлюмберже | 7 | 47 |
| прочие | 73 | 0 | ← no contrast at all |

| model | target | HR | 95% CI | p |
|---|---|---|---|---|
| Vt all, adj contractor+vintage | h3 | 1.32 | 1.01–1.73 | 0.040 |
| **Vt all + Н3×sour interaction** | **h3_x_sour** | **1.37** | **0.78–2.40** | **0.27** |
| sour only | h3 | 1.77 | 0.99–3.17 | 0.056 |
| nonsour only | h3 | 1.20 | 0.87–1.66 | 0.26 |
| **sour & Борец** (balanced) | h3 | **1.86** | **0.86–3.98** | 0.11 |
| nonsour & Борец | h3 | 0.85 | 0.50–1.44 | 0.54 |
| nonsour & Шлюмберже | h3 | 1.52 | 0.94–2.44 | 0.085 |

**The direct test of "Н3 helps sour" is the interaction: null, p=0.27, wrong sign.**
Н3 reading *harmful* fleet-wide is the signature of protection being assigned to the
wells that need it. The **sign flip by contractor reproduces Ya almost digit for digit**
(Ya: Борец 0.80, Шлюмберже 1.59) — two independent fields, same flip ⇒ not causal, not
transferable. The one cell that could answer (sour × Борец) spans 0.86–3.98: cannot rule
out a 14% benefit or a 4× harm. **Underpowered. Do not add to the model.**

## 4. Spike+constant on Vt — adds nothing. The shipped bundle is fine.

`scripts/run/production_risk_vt_hazard_fit.py`, re-run on the corrected population:

| stratum | shipped kind | ks shipped | ks spike | shipped median ratio | spike median ratio |
|---|---|---|---|---|---|
| `Vt_sour_Pooled` | `k1_aic` (single Weibull) | **0.060** | 0.054 | **1.04** | 1.12 |
| `Vt_nonsour_Pooled` | `k2_high_w1` (mixture) | **0.047** | 0.088 | 1.17 | 0.97 |

The shipped curves **pass or nearly pass** the 0.05 KM bar and are essentially unbiased on
shape. Spike+constant trades a little KM for a little median and **wins nothing overall**.
Plan-level, the two are indistinguishable (§6).

> ⚠️ **Retracted.** The first pass reported `Vt_sour_Pooled` shipped ks **0.168** and a
> **1.84×** median over-statement, concluding "B fixes sour, B50 98 → 131". **That was
> defect (c)** — 53 double-counted censored Vt runs inflated the exposure denominator,
> deflating the empirical hazard and so inflating every model/empirical ratio. On the
> corrected population the shipped sour fit is unbiased (1.04) and B50 moves 98 → 108.
> **There is no sour form defect.**

19 of 26 registry strata are `k1` (w1=0) — that part is a fact — but being `k1` turns out
**not** to imply being wrong: `Vt_sour_Pooled` is `k1` and fits fine. The proposed
"hybrid: B only for k1 strata" is therefore **moot and was not run.**

## 5. Day-0 startup failures — a third mode the schema cannot express

**15 of 205** Vt nonsour failures occur at tte ≤ 1 day (3 of 111 for sour) — real ПЭД /
кабельная линия startup failures, kept (clipped to 0.5 d) by `esp_population`.

The 5-column form `S(t) = w1·W(beta1,eta1) + (1-w1)·exp(eta2)` can express **either** a
day-0 point mass (`eta1 → 0`) **or** an infant window — **not both**. The MLE prefers the
point mass, and this is genuine, not an optimiser failure:

| eta1 bounds | fit | log-lik |
|---|---|---|
| [0.2, 90] (shipped fitter) | w1=0.023, beta1=10, eta1=**0.50** | **−1475.80** |
| [5, 90] | w1=0.198, beta1=0.57, eta1=**90.0** | −1488.05 |
| [0.2, 400] | w1=0.023, beta1=10, eta1=**0.50** | −1475.80 |

So the early component lands on the day-0 mass and the **real 7–30 d infant window
(32 events) is absorbed into the plateau** → the 0–30 d hazard comes back **36%
under-stated** (ratio 0.64).

Splitting the startup mass out as a separate `p0` frees the component to find a genuine
**22.3-day** window (`w1=0.058, beta1=1.75, eta1=22.3`) and improves ks **0.103 → 0.076**.
This is the same wart the Mc findings flagged (`beta1=10, eta1=0.5` on their bounds) —
**it is the schema, not the fitter.** Modelling all three phases needs either a 7-param
form (breaks the registry / VBA / EXE) or `p0` handled in the projection layer.
**Decide this before any refit ships.**

## 6. Plan calibration — A and B are indistinguishable outside Mc

Historical модель/факт by year, measured through the app on the **corrected** population:

| УН | 2024 | 2025 | 2026H1 | ремонтов в горизонте |
|---|---|---|---|---|
| Мирнинский, **B** | 1.17 | 0.92 | 1.35 | **43** |
| Мирнинский, **A** | 1.06 | 0.86 | 1.33 | **50** |
| Верхнетирский, **B** | 1.09 | 0.96 | 0.94 | **257** |
| Верхнетирский, **A** | 1.09 | 0.96 | 0.95 | **261** |
| Ярактинский, **B** | 0.98 | 1.27 | 1.25 | **137** |
| Ярактинский, **A** | 0.94 | 1.18 | 1.11 | **133** |

**Vt and Ya: A ≡ B for practical purposes** (261 vs 257; 133 vs 137). Only Mc shows a
material gap (50 vs 43), and there **A is no worse calibrated than B** — the difference is
shape, not level.

> ⚠️ **Retracted.** The first pass reported B under-predicting Vt (0.95/0.80/0.80) and Ya
> (0.70/0.87/0.84) and concluded "B does not generalise beyond Mc". **Both were population
> defects (b) and (c)** — missing Big-only closed events on Ya, double-counted censored
> exposure on Vt. Neither survives. **Vt's 0.80 was NOT a genuine fit result**, contrary to
> what that pass claimed on the grounds that Vt's totals matched the registry — the totals
> matched by coincidence (+53 spurious censored, −26 missing closed).

The app still guards the underlying failure mode: if a field's fit population carries
< 90% of the registry's events, it shows a blocking error instead of a comparison. With
the fix, no field trips it.
**Aggregate calibration cannot pick between A and B; shape cannot either. Both must be
checked.**

**Suggested (untested): hybrid «B only for k1 strata».** The k1/k2 split (§4) predicts
exactly where B should help, and it is the natural rule — but it was not run.

## 7. Consequences for `production_risk_mc_weibull_shape_findings.md`

That doc's Ya evidence was computed on the same defective population, so its two headline
claims must be re-derived. Measured here on the corrected population
(`Ya_nonsour_brt`, 1458 runs / 782 events — the doc had 988 / 440):

| band | empirical (doc) | empirical (corrected) | shipped/empirical (doc) | shipped/empirical (corrected) |
|---|---|---|---|---|
| 0–30 | 0.0666 | **0.0989** | 0.82 | **0.78** |
| 30–90 | 0.0275 | 0.0503 | 1.71 | **1.06** |
| 90–180 | — | 0.0477 | 1.54 | **0.94** |
| 180–300 | 0.0233 | 0.0408 | 1.39 | **0.97** |
| 300–450 | — | 0.0245 | 1.83 | **1.46** |
| 1100–1500 | 0.0172 | 0.0213 | 1.44 | **1.29** |

* ❌ **"The Weibull over-states mid-life hazard by 1.4–1.8× — this is the fleet-wide bug"
  does NOT survive.** Corrected, the plateau ratios are **0.94–1.46** (median ≈ 1.0). The
  inflation came from the missing Big-only events and the double-counted censored
  exposure, both of which depress the empirical denominator.
* ✅ **"The Weibull under-states the infant spike" SURVIVES and strengthens** — 0.78 here
  (doc: 0.82), and the true spike is larger than the doc knew (0.0989 vs 0.0666).
* ✅ **"No wear-out arm" SURVIVES** — 1100–1500 (0.0213) is still below 30–90 (0.0503).

**Net for Mc's option B:** the *infant-window* argument stands (and is the reason B is
still the Mc default here); the *"A extrapolates wear-out / over-states the plateau"*
argument does not. Mc's own numbers also shift: **B forecasts 43, not 40**, and B's
calibration (1.17/0.92/1.35) is **not better than A's** (1.06/0.86/1.33).

---

## 8. Vt needs no mixture — the k2's only job anywhere is the day-0 mass

`scripts/run/production_risk_mc_k2_cut_sweep.py`, run per stratum on `esp_population`
(«Наработка» clock, cause-specific: failure=1, ГТМ=0), constraints **β₂ > max(β₁,1)** and
**η₂ > 2·η₁**. Each K=2 is tested by likelihood ratio (3 df) against the **nested single
Weibull** `two_layer` already fits at the same cut:

| stratum | runs / fails | day-0 mass (tte≤1) | c=0 | c=3 | c=7 | c=30 |
|---|---|---|---|---|---|---|
| **Vt_sour** | 165 / 110 | 3 | p=0.79 ✗ | 0.70 ✗ | 0.66 ✗ | 0.18 ✗ |
| **Vt_nonsour** | 476 / 209 | 18 | p=0.15 ✗ | 0.89 ✗ | 0.28 ✗ | 0.64 ✗ |
| Ya_nonsour | 2166 / 1226 | 54 | **p=0.0032 ✓** | 0.79 ✗ | 0.79 ✗ | 0.99 ✗ |
| Mc_nonsour | 164 / 44 | 5 | **p=0.00016 ✓** | 0.90 ✗ | 0.98 ✗ | 0.99 ✗ |

**The K=2 is justified ONLY at c=0, and only where the day-0 mass is large enough to
register (Ya, Mc). Past day 3 it is never justified on any field.** On **Vt it is never
justified at all** — not even at c=0.

Read together with §4 (spike+constant adds nothing on Vt) this is the same conclusion from
a second direction: **Vt wants one Weibull.** It also validates `two_layer`'s design
directly — a single Weibull plus a per-install infant probability is exactly the structure
the LR test says is there, and the mixture's extra components are fitting noise (at c>0
they wander to w₁=0.70–0.93 and η₂=180 000 while adding nothing).

Day-0 mass is a stable ~2.5–4% of runs everywhere (Mc 3.0%, Vt_nonsour 3.8%, Ya 2.5%,
Vt_sour 1.8%) — consistent with the `p_infant` the two-layer fit books independently.

## Code added / changed this session (all additive)

| file | what |
|---|---|
| `esp_population.py` | **fixes:** sour inherited from the well's Свод history; **new `load_big_runs()`** — Big open **and** closed runs, deduped vs Свод + V03 (±7 d), replacing `load_open_runs()`. `build(..., big_runs=)` replaces `open_runs=`. **new:** `attach_equipment()` joins Big per-run equipment by well + nearest install (±10 d). |
| `crosswalk.py` | **new:** `well_deposit_map()` / `well_un_map()` — well → «Месторождение» / «УН» from the ПП plan (the prefix is the УН only). |
| `backend/tests/test_esp_population.py` | **new:** 6 tests — sour inheritance, Big-only closed runs, the ±7 d dedup window. |
| `scripts/run/production_risk_vt_deposit_test.py` | месторождение vs УН crossed design. |
| `scripts/run/production_risk_vt_exec_group.py` | Н2/Н3 effect + assignment confounding. |
| `scripts/run/production_risk_vt_hazard_fit.py` | per-stratum spike+const fit + hazard-band audit vs shipped. |
| `streamlit_apps/production_risk_field_plan.py` | renamed from `production_risk_mc_plan.py`; **УН selector** (Mc / Vt / Ya), per-stratum option-B fit, per-field default from measured calibration (Mc→B, Vt→A, Ya→A), blocking guard when the fit population carries < 90% of registry events. |

## Open

* **Re-derive the Mc option-B decision** against §7 — its plateau argument is gone, its
  infant argument stands. Mc B now forecasts 43 (not 40) and does not out-calibrate A.
* Decide the `p0` question (§5) before any refit ships — it is the remaining real defect.
* The `attach_equipment` / exec-group and deposit results were re-run post-fix and are
  unchanged; the hazard-fit and calibration numbers here are all post-fix.
* Inherited, still open: `crosswalk.py:644` reads «Нспуска» (setting depth, metres) into
  `EspRun.run_seq`. Downstream impact untraced.
