"""Phase D — D6: assemble the report.

Composes the story from the materialised Phase D results — precursor atlas → what
carries hazard (D2, mode-checked) → does it predict (D3/D4, the verdict with intervals)
→ watchlist or telemetry recommendation (D5) — into a single markdown report, with the
§0 decisions in the header, the leakage rules attested (test names cited), and the
prediction-first discipline visible (every conclusion carries its out-of-sample delta).

Outputs under ``results/phase_d_report/<date>/``:
  reports/README.md

Run (needs the other phase_d_* scripts first):
    python scripts/run/phase_d_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir, RESULTS_ROOT


def _latest(slug: str) -> Path | None:
    base = RESULTS_ROOT / slug
    if not base.exists():
        return None
    dirs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
    return dirs[0] if dirs else None


def _read(slug: str, rel: str) -> pd.DataFrame | None:
    d = _latest(slug)
    if d is None:
        return None
    p = d / rel
    return pd.read_csv(p) if p.exists() else None


def _read_text(slug: str, rel: str) -> str | None:
    d = _latest(slug)
    if d is None:
        return None
    p = d / rel
    return p.read_text(encoding="utf-8") if p.exists() else None


def main() -> None:
    out = results_dir("phase_d_report")
    L: list[str] = []

    L += ["# Phase D — Within-Run Dynamics: Time-Varying Hazard + Early-Warning Layer",
          "",
          "**Question:** can we flag, H days ahead, which running pumps will fail — "
          "materially better than *stratum + current age* alone?  Phase C ended with the "
          "program's defining negative result (static covariate θ ≈ chance out of sample); "
          "Phase D tests the one channel the static layer cannot reach — how a run is "
          "behaving *right now*, from the 1.25M-row daily table.",
          ""]

    # ── §0 decisions header ──────────────────────────────────────────────────
    L += ["## §0 decisions (user-confirmed 2026-07-07)",
          "",
          "| Parameter | Value |",
          "|---|---|",
          "| Prediction horizon H | **60 operating days** (30 / 90 secondary) |",
          "| Landmark cadence | every **30 operating days** of run age |",
          "| Actionable capacity | top **5%** and top **10%** of running fleet |",
          "| Guard gap g | **7 operating days** (sensitivity {3,7,14}); `stop_date` = pull |",
          "| Eligibility | ≥ **30 valid telemetry days** before first landmark |",
          "| Scope | all-cause primary; hydraulic & electro-thermal secondary |",
          "",
          "**Leakage rules attested** (tests in `backend/tests/test_landmark_features.py`): "
          "rule 1 past-only — `TruncationInvariance.test_features_bit_identical_after_truncation`; "
          "rule 2 guard gap — `GuardGapTargets.*`; rule 3 train-before-screening — split fixed "
          "in D3 before any univariate screen; rule 4 no outcome-clock features — cycling enters "
          "only as trailing-window counts (no run_days/ttf totals).",
          "",
          "**Prediction-first discipline:** every conclusion below cites its out-of-sample "
          "delta; in-sample hazard ratios (D2) are the diagnostic view only.",
          ""]

    # ── D0 frame ────────────────────────────────────────────────────────────
    summ = _read("phase_d_landmarks", "tables/summary.csv")
    recon = _read("phase_d_landmarks", "tables/reconciliation.csv")
    L += ["## D0 — landmark frame", ""]
    if summ is not None:
        prim = summ[(summ["H"] == 60) & (summ["g"] == 7)]
        if not prim.empty:
            r = prim.iloc[0]
            L.append(f"Landmark frame built on the operating-day clock (qliq>0). At the "
                     f"primary H=60/g=7: **{int(r['n_eligible'])} eligible landmarks**, "
                     f"{int(r['n_pos'])} positive (fail-in-window), {int(r['n_neg'])} negative, "
                     f"pos-rate {r['pos_rate']}.")
    if recon is not None and "flag" in recon.columns:
        n_flag = int(recon["flag"].sum()); n_scored = int(recon["discrepancy"].notna().sum())
        L.append("")
        L.append(f"**Data-quality flag:** {n_flag}/{n_scored} runs show >10% discrepancy "
                 "between the qliq operating-day count and `ttf_true_best_days` — the "
                 "operating-day clock and the survival clock disagree materially on a third "
                 "of runs; a ceiling on any within-run feature's fidelity.")
    L.append("")

    # ── D1 precursor atlas ───────────────────────────────────────────────────
    smd = _read("phase_d_precursors", "tables/standardized_diffs.csv")
    L += ["## D1 — precursor atlas (train only)", ""]
    if smd is not None:
        smd = smd.copy(); smd["abs"] = smd["smd"].abs()
        top = smd.sort_values("abs", ascending=False).head(8)
        L.append("Matched (stratum × age-band) standardized mean differences, pre-failure "
                 "vs survivor landmarks — the honest expectation cap. Strongest:")
        L.append("")
        L.append("| feature | SMD |")
        L.append("|---|---|")
        for _, r in top.iterrows():
            L.append(f"| `{r['feature']}` | {r['smd']:+.3f} |")
        mx = top["abs"].max()
        L.append("")
        L.append(f"Largest |SMD| ≈ {mx:.2f} — modest; the atlas already caps expectations "
                 "(no single feature cleanly separates pre-failure windows).")
    L.append("")

    # ── D2 inference view ────────────────────────────────────────────────────
    L += ["## D2 — time-varying Cox (inference view; associations, heavily confounded)", ""]
    ac = _read("phase_d_tv_cox", "tables/tvcox_all_cause.csv")
    if ac is not None and not ac.empty:
        L.append("All-cause stratified time-varying Cox, per-SD HRs (episode-split; "
                 "**model-based SEs** — lifelines has no cluster-robust option for the "
                 "time-varying fitter, so these are screening signals, not confirmatory). "
                 "Top movers:")
        L.append("")
        L.append("| dynamic | HR/SD | 95% CI | p |")
        L.append("|---|---|---|---|")
        for _, r in ac.sort_values("p").head(6).iterrows():
            L.append(f"| `{r['covariate']}` | {r['hr']:.3f} | "
                     f"[{r['hr_ci_lo']:.2f}, {r['hr_ci_hi']:.2f}] | {r['p']:.4f} |")
    pc = _read("phase_d_tv_cox", "tables/prior_check.csv")
    if pc is not None and not pc.empty:
        L += ["", "Phase B/C prior check (cycling/frequency → electro-thermal; "
              "kprod/qliq decline → hydraulic):", ""]
        L.append("| feature | expected mode | significant modes | verdict |")
        L.append("|---|---|---|---|")
        for _, r in pc.iterrows():
            L.append(f"| `{r['feature_stem']}` | {r['expected_mode']} | "
                     f"{r['significant_modes']} | {r['verdict']} |")
    L += ["", "*§0.2 disclaimer:* these are associations under adjustment; within-run "
          "dynamics are even more confounded by indication than static settings — a rising "
          "load-variability or watercut trend may be the operator responding, not the pump "
          "failing.  The inference view does not license a causal or a predictive claim; "
          "that is D4's job.", ""]

    # ── D3/D4 verdict ────────────────────────────────────────────────────────
    L += ["## D3 / D4 — prediction view and the verdict (out of sample)", ""]
    verdict = _read_text("phase_d_landmark_models", "logs/alarm.txt")
    grid = _read("phase_d_landmark_models", "tables/alarm_grid.csv")
    delta = _read("phase_d_landmark_models", "tables/delta_ci.csv")
    op = _read("phase_d_landmark_models", "tables/primary_operational.csv")

    if verdict:
        vline = [ln for ln in verdict.splitlines() if ln.startswith("[VERDICT]")]
        if vline:
            L += ["> " + vline[0].replace("[VERDICT] ", ""), ""]

    if grid is not None and not grid.empty:
        L.append("Out-of-sample AUC (install-cohort split, test = installs after "
                 "2023-12-31), dynamic (age+stratum+dynamics) vs baseline (age+stratum):")
        L.append("")
        L.append("| H | g | AUC base | AUC dyn | Δ AUC | Brier base | Brier dyn |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in grid.iterrows():
            L.append(f"| {int(r['H'])} | {int(r['g'])} | {r['auc_base']} | {r['auc_dyn']} "
                     f"| {r['auc_delta']} | {r.get('brier_base','')} | {r.get('brier_dyn','')} |")
        L.append("")
    if delta is not None and not delta.empty:
        L.append("Well-cluster bootstrap 95% CI on the AUC delta (dynamic − baseline):")
        L.append("")
        L.append("| H | g | Δ AUC | CI low | CI high |")
        L.append("|---|---|---|---|---|")
        for _, r in delta.iterrows():
            L.append(f"| {int(r['H'])} | {int(r['g'])} | {r['auc_delta']} | "
                     f"{r['ci_lo']} | {r['ci_hi']} |")
        L.append("")
    if op is not None and not op.empty:
        L.append("Operational metrics at the primary H=60/g=7 (dynamic model):")
        L.append("")
        L.append("| capacity | precision | recall (dyn) | recall (base) | "
                 "false alarms/pump-yr | median lead (d) | runs caught |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in op.iterrows():
            L.append(f"| top-{int(r['k_pct'])}% | {r['precision_dyn']} | {r['recall_dyn']} | "
                     f"{r['recall_base']} | {r['false_alarms_per_pump_year']} | "
                     f"{r['lead_time_median']} | {int(r['n_caught_runs'])} |")
        L.append("")

    # ── D5 outcome ───────────────────────────────────────────────────────────
    L += ["## D5 — watchlist or telemetry recommendation", ""]
    wl = _read("phase_d_watchlist", "tables/watchlist.csv")
    rec = _read_text("phase_d_watchlist", "reports/telemetry_recommendation.md")
    if wl is not None and not wl.empty:
        L.append(f"**Watchlist earned and emitted** — {len(wl)} currently-running pumps "
                 "scored at their latest landmark (`phase_d_watchlist/…/tables/watchlist.csv`), "
                 "ranked by predicted next-60-operating-day risk with per-pump top drivers.")
    elif rec is not None:
        L.append("**Honest null → telemetry-improvement recommendation** (watchlist not "
                 "shipped). Summary:")
        L.append("")
        body = "\n".join(rec.splitlines()[2:])  # drop the md title
        L.append(body)
    L.append("")

    # ── open questions ───────────────────────────────────────────────────────
    L += ["## Open questions carried forward", "",
          "- The op-day (qliq>0) clock and `ttf_true` disagree >10% on ~a third of runs — "
          "reconcile the operating-time definition before the dynamic layer can be trusted.",
          "- `days_since_restart` / idle structure is the most under-recorded precursor; "
          "explicit on/off event logging would test the cycling→electro-thermal prior properly.",
          "- Re-evaluate when a telemetry-complete daily feed exists or the 2024+ cohort "
          "matures (the same gate Phase C set for θ).",
          ""]

    (out / "reports" / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[D6] report -> {out / 'reports' / 'README.md'}")


if __name__ == "__main__":
    main()
