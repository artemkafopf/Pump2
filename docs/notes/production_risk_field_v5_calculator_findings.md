# Calculator v5 — per-field models, two sizing fixes, and a Python twin

2026-08-04. Covers `field_v4.py`, `pikpolka_sim.py`, the `ModuleFailureV4` rewire and the
changes to `unified_v4.fit_contractor_levels`.

## Why

The ПикПолка and NPV/УВЧ workbooks both run `ModuleFailureV4` (unified v4). Four things
blocked wider use, and a fifth made every study expensive:

1. `LookupNominal4` returned **999999** when Ql ran above the top of the КРС_ЭПУ ladder, so
   `Kpod = Ql/999999` hit the 0.15 clamp (θ_Kpod ≈ 2.3) while θ_Qnom pinned at the 1600 cap —
   ~40 % of the modelled life lost for a purely cosmetic reason. The sheets carried the same
   defect in their own XLOOKUP defaults.
2. The model was Vt-only (later Vt + Ya) although both workbooks already had a field cell.
3. Contractor levels were fitted on the Ql overlap window with **no size term**, so a
   contractor running bigger pumps had part of θ_Qnom baked into its level — and θ_Qnom then
   applied it again.
4. Vt_sour's θ_Qnom **falls** above Qnom 1000 (1.3668 → 1.2034), i.e. the shipped VBA modelled
   a 1600 pump as longer-lived than a 1000 one, on 20 runs / 14 events.
5. Everything had to be answered one well at a time, by hand, in Excel.

## Fixes

**Ladder.** `LookupNominal4` falls back to the **largest** catalogue nominal. An undersized
pump then reads as Kpod > 1, which both layers can express. No observed run in any field
exceeds Ql 1500, so this only ever fires on a user-entered scenario.

**Sour clamp.** `unified_v4.deploy_theta_qnom` enforces non-decreasing θ **above** the
reference at export time; the fit itself is untouched. Applied to every field, so a thin top
knot cannot invert a new one either. It also caught **Ic** (640/1000) and **Za** (1600).

**Contractor de-biasing.** `fit_contractor_levels` keeps the window and adds `log(Qnom/250)`
and `log(Kpod/0.8)`, standardised in-window so one ridge constant means the same thing in
every field. The level is now the contractor effect *at the reference point*.

| field | slb plain → adjusted | oth plain → adjusted | window n/events | geomean Qnom brt/slb |
|---|---|---|---|---|
| Ya | 1.325 → 1.221 | 4.243 → 3.970 | 287 / 164 | 331 / 440 |
| Vt | 1.200 → 1.196 | 1.910 → 1.750 | 154 / 88 | 354 / 465 |
| Az | 1.354 → 0.977 | — (5 window runs) | 54 / 33 | 309 / 436 |
| Ic | 0.911 → 0.899 | — (2 window runs) | 52 / 29 | 348 / 409 |
| Za | 0.839 → 0.758 | 1.884 → 1.919 | 60 / 34 | 376 / 404 |
| Mc | — (no slb at all) | — | 32 / 8 | 375 / — |

⚠ **Az's correction is the least trustworthy** — 33 window events and a size term of 1.7 per
e-fold. Read it as "the plain level was mostly pump size", not as "slb is neutral on Az".

⚠ **The correction does not close the loop in the deployed calculators.** slb's size bias and
its low-Kpod bias cancel only against the *fitted* Kpod layer. The workbooks keep the operator
**bathtub** (explicit decision), which has the opposite sign at low Kpod, so a typical slb pump
(Qnom ≈ 465, Kpod ≈ 0.57) still composes to θ ≈ 1.75 against ≈ 1.31 fit-consistent — about 13 %
shorter life. That is a stated assumption, not a fit defect.

## Per-field models (`field_v4.py`)

One knot grid for every field; complexity is controlled by the ridge, never by moving knots —
so the deploy block stays a rectangle and a shrunk arm degrades toward θ ≡ 1 rather than
quietly meaning something different field to field.

| stratum | n / events (complete case) | β₀ | RMST₇₃₀ ref | all-runs control | CV kpod / freq |
|---|---|---|---|---|---|
| Ya | 1227 / 635 | 1.094 | 524.0 | 522.1 | +1.19 / +1.13 |
| Vt_nonsour | 249 / 127 | 1.248 | 403.9 | 406.6 | +8.60 / −0.60 |
| Vt_sour | 116 / 81 | 1.340 | 177.7 | 152.1 | (pooled shape) |
| Az | 199 / 108 | 1.140 | 438.3 | — | +0.88 / −0.28 |
| Ic | 188 / 112 | 1.097 | 467.1 | — | +3.14 / −0.79 |
| Za (= **Au**) | 168 / 88 | 1.223 | 371.1 | — | +0.95 / +1.93 |
| Mc (incl. **Mr**) | 90 / 27 | 1.611 | 435.9 | 437.5 | +0.32 / −0.01 |

Notes that matter more than the numbers:

* **Mc is a 2024+ install cohort** (162 runs / 42 events; 90 / 27 complete case) with **one
  contractor** — 158 brt, 4 oth, zero slb. The contractor layer is not estimable and ships as
  1.0 for all three. β₀ = 1.61 also disagrees with the older "infant spike then flat, no
  wear-out" reading of Mc; on a young, heavily censored cohort that is what a Weibull will do.
  Treat Mc as provisional.
* `Au` and `Mr` are **pads**, not fields: the population build folds `AU*` into `Za` and `MR`
  into `Mc`. Both get their own deploy row pointing at the parent fit.
* The **all-runs control** (covariate-free Weibull on every run, complete case or not) tracks
  the layered baseline within ~3 d on Ya, Vt_nonsour and Mc. Vt_sour is the exception,
  152 vs 178 d, because the layered number is quoted at the reference operating point while
  the sour population sits well away from it.
* Kpod earns its place everywhere it can be measured; **frequency does not** (negative on Vt,
  Az and Ic). It stays an operator-set prior, which is what the CtrlBlock is for.

## The trap that cost the most time

`vt_v32_hybrid.prepare_frame` only forces `config.SOUR_WELL_LEVEL_ALL_RUNS` when it builds the
frame **itself**. Handing it a plain `cached=` frame skips the v3.2 sour relabel, 30 Vt running
pumps stay in `Vt_nonsour`, and every Vt number moves — contractor slb 1.200 → 1.172, oth
1.910 → 1.992 — with nothing in the output saying why. Fixed by `unified_v4.build_cached_frame`,
which forces the flag and restores it; the relabel is a no-op for the single-H₂S-class fields,
so one frame serves them all.

## The Python twin (`pikpolka_sim.py`)

A faithful port of the sheet engine — layers, decline, water cut off the ХВ recovery curve,
nominal re-pick at each restart, the consume-and-renew failure rule, uptime, costs,
discounting — with the reference tables read out of the workbook rather than re-typed.

**Verified against both workbooks' cached values**, all columns:

| workbook | branch | ННО sheet/py | failures | NPV relative error |
|---|---|---|---|---|
| NPV_УВЧ | ТекЧастота | 102 / 102 | 40 / 40 | 1.0e-15 |
| NPV_УВЧ | Разгон | 78 / 78 | 53 / 53 | 4.2e-16 |
| ПикПолка v7 | Пик | 145 / 145 | 4 / 4 | 2.0e-15 |
| ПикПолка v7 | Полка | 284 / 284 | 3 / 3 | 2.2e-15 |

Three sheet behaviours the port had to reproduce exactly, each found by column-diffing:

1. **The decline changes form at day 181** (`C195 = C$194*…`, `C191 = C$190*…`): a curved
   phase to 180 days, then linear over 915. Extrapolating phase 1 drives Ql negative inside a
   year on a steep case.
2. **XLOOKUP `match_mode=1` means smallest-≥, not first-encountered.** The ХВ curve is not
   monotone at its tail (…0.99901, 1.00085, 1.00227, 1.0), so at a recovery share of 0.99903
   next-larger picks the final 1.0 → водность 1.0 → oil 0 → the cumulative freezes and the well
   is dead. First-encountered leaks 1 % of the rate forever and eventually runs off the end of
   the curve, where the lookup falls back to 0 % water cut and the well returns at **full rate**
   — 1729 phantom days on the NPV grid.
3. **ПикПолка's «Полка» branch does not decline** (`N16 = E$4*T$4`), and its uptime window is
   «смена, дней Полка» 12 while «Пик» uses the ускоренная 19 — although both spills hand the
   UDF the same 12. The displayed «Номинал» and the model's internal one therefore step on
   different days; the port tracks both.

Sweeps: `S.sweep(cfg, **{"branch1:freq": [50,55,60,65], "branch0:ql0": [400,800]})`, over
field/stratum, contractor, nominal, downtime, discount, decline, and every CtrlBlock anchor.

## Effect on the two live cases

Через порт, before touching Excel:

| workbook | вариант | ННО | отказов | NPV |
|---|---|---|---|---|
| NPV_УВЧ (Vt кислый, oth, 1250, 800→960 м³/сут, 50→60 Гц) | сейчас | 102 / 78 | 40 / 53 | 2.3057 / 2.2943 млрд |
| | + клапан по Qном | 98 / 75 | 41 / 53 | 2.3031 / 2.2913 |
| | + пофондовые параметры | 103 / 79 | 39 / 51 | 2.3202 / 2.3130 |
| ПикПолка v7 (Vt кислый, brt, 320/60, 55 Гц) | сейчас | 145 / 284 | 4 / 3 | 83.9 / 79.8 млн |
| | + пофондовые параметры | 142 / 278 | 4 / 3 | 83.8 / 79.8 |

The УВЧ verdict does not change sign: Δ ННО ≈ −24 сут, Δ NPV ≈ −11.5 млн. ⚠ But that verdict
rests almost entirely on **one operator cell**: θ_freq(60)/θ_freq(50) = 1.3/0.9 = 1.444, which
at the sour exponent costs ~24 % of ННО. Fitted free, frequency is flat within noise in every
field measured. Sweep `ctrl:t60` before quoting the number.

## Reproduce

```bash
python scripts/run/field_v4.py --boot 200            # fits + deploy blocks + KM control
python scripts/run/pikpolka_sim.py --workbook <xlsm> --validate
python scripts/deploy/wire_pikpolka_v5.py --workbook <xlsm> --layout npv --check
cd backend && python -m pytest tests/test_field_v4.py tests/test_pikpolka_sim.py -q
```

Outputs → `results/production_risk_field_v4/<date>/`.

---

## Addendum, same day — the pooled fallback and the sheet rebuild

### Fleet: a model for the fields that have none

The workbooks price codes the survival mart has never seen — **Ki** (Кийский), **Ma**
(Марковский), **Bt** (Большетирский) — and **Da** has 50 events with a single contractor. The
VBA used to fall back to **Vt**, i.e. it priced them on a sour-capable field with the steepest
nameplate slope on the roster. `Fleet` replaces that: every run in the mart, pooled.

| | n / events | β₀ | η₀ | RMST₇₃₀ | all-runs control | slb | oth | CV kpod / freq |
|---|---|---|---|---|---|---|---|---|
| Fleet | 2308 / 1204 | 1.0489 | 749.2 | **472.3** | 485.8 | 1.1494 | 2.5307 | **+12.77** / −0.38 |

θ_Qnom: 0.525 / 0.652 / 0.756 / 1.000 / 1.122 / 1.231 / 1.427 / 1.668 — monotone, and steeper
at the small-pump end than any single field.

* Contractor levels are taken **within field** (a fixed effect per field in the stage-1 window
  fit), so the level is not a field-mix artefact. `fit_contractor_levels` now adds a dummy per
  extra stratum generally; for Vt that is bit-for-bit the `Vt_sour` indicator it always used.
* ⚠ **θ_Qnom is fitted on the pool with no field term.** Between-field baseline differences
  that correlate with pump size therefore leak into the slope. This is a fallback, not a
  substitute for a field's own fit, and the sheet says so.
* Kpod's +12.77 is the largest gain measured anywhere, and frequency is negative **again** —
  the seventh independent field-level confirmation that it does not earn its place.

`CM4_ResolveKey` now resolves `<field>_sour` → `<field>` → **`Fleet`** → Vt, and an unknown code
lands on Fleet: verified in Excel by typing `XX` into the field cell.

### МодельОтказов rebuilt as three blocks

1. **ВВОД** (A1:B12) — УН, кислый and the θ scenario are formulas into `ПрогнозРемонтов`, so
   the sheet cannot disagree with the forecast; the resolved model key is displayed; подрядчик /
   Qном / частота / Кпод are the curve-drawing point. ННО at that point and at the reference.
2. **КРИВЫЕ** (A14:H27 + three charts) — Кпод / частота / Qном, all driven by block 1.
   This fixed a live defect: the curve tables were hardcoded to «Борец» **and to Vt**, on a
   workbook whose model had just become per-field.
3. **ПАРАМЕТРЫ** (A30:I82) — TuneBlock `A34:F47` (14 keys), QnomBlock `A50:I64`,
   CtrlBlock `E68:E77` (B базовый / C стресс / D пользовательский / E в расчёте), then the notes.

Deleted for good: the «Проверка применимости» block (still checking **v1** bounds Ql 100–823,
Кпод 0.2–1.5), `ClampNote_v1`, three stray `FailureFlag_v1` cells returning `#VALUE!`, and the
`ModuleFailureV1` + `TestFailureFlagV1` components. The VBProject is now `Module1`, `Module2`
(the owner's) and `ModuleFailureV4`. The rebuild scans every other component first and keeps
the legacy modules if anything still calls in.

ННО is unchanged by the re-layout in both workbooks (218/269 and 102/84).

### ⚠ Open: two operator tables do not know the new codes

The model now resolves every code, but the workbook plumbing does not:

| таблица | есть | НЕТ |
|---|---|---|
| `КРС_ЭПУ!G10:H16` — смена УЭЦН, сут | Au, Az, Ic, Mc, Mr, Vt, Ya | **Da, Ki, Ma, Bt** |
| `Экономика!C10:E20` — netback, ₽/т | Az, Da, Ki, Ma, Vt, Ya, Bt, Ic, Au, Mr | **Mc** |

The gaps are complementary. Consequences, both silent: a missing downtime row makes `U4` `#N/A`,
which propagates into the schedule's `DownDays` and turns the whole spill into `#VALUE!`
(ННО reads 9999); a missing netback row returns a plausible-looking wrong number through an
**approximate-match** `VLOOKUP` on an unsorted list (Δ NPV −2 147 млн for Mc). Two fixes, both
needing the operator's numbers: add the four downtime rows, and give `Mc` a netback.

**Resolved for Mc (operator decision, 2026-08-04): Mc and Mr share the survival curves and the
netback, but keep their own «смена УЭЦН».** The netback is aliased inside the lookup —
`VLOOKUP(IF(<field>="Mc","Mr",<field>), …)`, `scripts/deploy/alias_mc_to_mr_netback.py` —
rather than duplicating 19 661 ₽/т into a second row that could drift. The two `КРС_ЭПУ`
downtime rows (Mc 12 сут, Mr 11 сут) stay as they are: that one day is the whole difference
between the two codes, +111.93 vs +112.94 млн on the live NPV case. Mc's Δ NPV was −2 146.83 млн
before the alias.

**Still open: Ki / Ma / Bt / Da have no «смена УЭЦН» row** (`КРС_ЭПУ` reads its field list from
`C10:C16`, seven codes). The survival side is ready — `Fleet` resolves and gives 468.9 сут at the
reference point — so it is purely the missing downtime figure. Changing the two lookups to exact
match would turn any future gap into `#Н/Д` instead of a plausible wrong number.
