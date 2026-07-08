# Handoff — VBA Prediction System v2 (finish the Excel-side integration)

## Your job

The **v2 rebuild is code-complete and tested on the Python side**. What remains is
**getting the VBA modules to compile and validate green inside the production Excel
workbook**, iterating on compile/runtime errors the user reports (usually as
screenshots of the VBA editor). Original spec: `agents/analyses/vba_model_v2.md`.
What-changed write-up: `results/vba_model_v2/2026-07-07/README.md`. Project memory:
`[[project_vba_model_v2]]`, `[[feedback_vba_encoding]]`.

You cannot run Excel yourself. The loop is: user runs a macro → pastes the error →
you fix the `.bas` in `d:\GitHub\Pump2\vba\` → user removes+re-imports that module →
repeat. Keep fixes surgical and explain the *why* each time.

## Definition of done (only the last box is open)

- [x] Python bundle regenerated (`n_boot=200`), `backend/tests/test_vba_bundle.py` green (10).
- [x] All VBA modules written; θ shelved; clock document-only; mode-mix layer; validation harness.
- [x] Write-up + `vba/README.md` + memory updated.
- [ ] **In the workbook: all modules compile, `ImportBundle` succeeds, `ValidateRegistry`
      is all-green, and the `ESP_Validation` sheet contents are pasted into
      `results/vba_model_v2/2026-07-07/README.md` (replacing the last checkbox).**

## The bundle (single source of truth — do NOT hand-edit)

`results/esp_survival_vba_models/2026-07-07/`:
`esp_models.csv` (26 rows), `esp_mode_mix.csv` (32), `mode_group_map.csv` (9),
`bundle_manifest.txt`, `vba/mdlModelSeed.bas` (generated CreateModelSheet).
Regenerate with `python scripts/run/export_model_csv_for_vba.py 200` (from repo root;
mart is at `data/warehouse/pump2.db`, reachable). Heavy logic lives in
`backend/analysis/workflows/esp_survival/vba_bundle.py`.

## VBA module set (`d:\GitHub\Pump2\vba\`)

Import as standard modules (File → Import File), removing any old same-named module
first (right-click → Remove → No to export — stale-ghost bug otherwise):
`mdlMath`, `mdlLatentWeibull`, `mdlModelRegistry`, `mdlModeMix`, `mdlPublicFunctions`,
`mdlCoxHR`, `mdlBatchProcess`, `mdlValidation`, `mdlModelSeed`.

`ThisWorkbook_setup.bas` is **pasted** into the `ThisWorkbook` object, NOT imported
(its `Attribute VB_Name = "ThisWorkbook"` line collides on import — paste only the
`Workbook_Open` sub).

Workbook procedure to run once modules compile:
```
ImportBundle "D:\GitHub\Pump2\results\esp_survival_vba_models\2026-07-07"
ValidateRegistry
RunPredictions
```
`ImportBundle` → loads ESP_Models, ESP_ModeMix, ESP_ModeMap, hidden ESP_Version.
Status/results go to the `ESP_Status` and `ESP_Validation` sheets (never MsgBox from
UDF-reachable code). First debugging move for the user: **Debug → Compile VBAProject**
surfaces every compile error at once.

## Design decisions already locked (do not relitigate)

- §0.1 clock = **document-only**: time headers carry `_op_d`; no in-code conversion;
  `uptime_factor` exported + `ESP_Uptime` UDF for manual `age_op≈age_cal*u_s`.
- §0.2 θ = **deprecate-keep**: `ESP_*_Cox` return the baseline (θ≡1) unless an
  `ESP_CoxCoeffs` sheet has `enabled=TRUE`; Cox columns dropped from RunPredictions.
- §0.3 single-Weibull/degenerate strata stored `w1=0` + `model_kind ∈ {k2,k1_aic,k1_degenerate}`.
- §0.4 mode-mix layer + B50 CI columns included. §0.5 sour-caution outside Vt skipped.

## Bugs already found & fixed (don't reintroduce; watch for the same class)

1. **`repr(np.float64)`** adds a dtype wrapper in numpy 2.x → corrupts the generated
   `.bas`. `_vba_literal` now `.item()`s numpy scalars. (This env is numpy<2, but keep it.)
2. **"Too many line continuations"**: VBA caps a statement at 25 `_` continuations.
   The seed emits one `rows(i) = Array(...)` per line (zero continuations).
3. **utf-8-sig BOM + Cyrillic**: VBA `Line Input` reads ANSI → BOM becomes `ï»¿` and
   Cyrillic (the node→mode map) becomes mojibake. All imports read via `ReadAllLines`
   (ADODB.Stream, Charset utf-8); header matchers also BOM-strip defensively.
4. **"Clear method of Range class failed" / "Application-defined error" on cell write**:
   the production workbook is protected and/or read-only. `PrepWorkbook()` now runs
   before every writer: aborts with a MsgBox if `ThisWorkbook.ReadOnly`, else
   unprotects the workbook structure + every sheet; `FreshSheet()` unprotects→clears,
   or deletes+recreates a locked sheet. Password protection is the one case it can't
   self-heal (writes a status message telling the user to unprotect manually).
5. **`cStr` reserved-word collision**: a variable named `cStr` is the same identifier
   as VBA's `CStr()` → "Syntax error". Renamed to `cStrt`. **General rule: never name a
   variable `cStr/cInt/cLng/cDbl/cBool/cByte/cCur/cDate/cDec/cSng/cVar/cVErr` or any
   case-variant of a built-in function** (VBA is case-insensitive).

## VBA rules (each cost a session before — see `[[feedback_vba_encoding]]`)

- `ChrW()` not `Chr()` for code points >255; no Cyrillic literals in `.bas` source
  (sheet/name/contractor matching is built from `ChrW` sequences).
- Module-level `mLoaded/mCount/mModels` are Private to `mdlModelRegistry`; other
  modules use `EnsureRegistryLoaded()` / public accessors only.
- **No `MsgBox` in any code path reachable from a worksheet UDF** (→ `#VALUE!`). The
  only sanctioned MsgBox is in `PrepWorkbook` (Sub-only path) for the read-only case.
- No variable that collides case-insensitively with a function name (e.g. `h2sClass`
  vs `H2SClass()`; use `h2sCls`).
- `Option Explicit` everywhere; no cross-module calls to `Private` members.
- Public function names must be globally unique across modules.

## When the user reports a new error

1. Read the exact message + the highlighted line from the screenshot.
2. Classify: **compile** (syntax/name/type — fix the `.bas`, ask them to
   Debug→Compile) vs **runtime 1004** (usually protection/read-only/missing sheet —
   check `PrepWorkbook`/`FreshSheet` path) vs **`#VALUE!`/`#N/A` in a cell** (UDF error
   path — check for MsgBox, wrong arg types, unloaded registry).
3. Make the minimal edit in `vba/`, tell them exactly which module to remove+re-import.
4. After it compiles clean, drive them to `ImportBundle` → check `ESP_Status` for
   `... OK: 26 rows loaded (clock=ttf_mix)` → `ValidateRegistry` → read `ESP_Validation`.
5. If a `ValidateRegistry` row FAILS, the message names the stratum/quantity; fix the
   corresponding math/param path (the Python `test_vba_bundle.py` already proves the
   bundle side, so a VBA-side FAIL is a VBA bug, not a data bug).

## Acceptance to close out

`ValidateRegistry` all-green in the workbook, and paste the `ESP_Validation` sheet rows
into `results/vba_model_v2/2026-07-07/README.md`. Then update `[[project_vba_model_v2]]`
to mark the Excel-side validation complete.
