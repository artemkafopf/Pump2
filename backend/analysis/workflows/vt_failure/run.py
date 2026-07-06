"""
VT failure analysis — main orchestrator.

Usage:
    python backend/analysis/workflows/vt_failure/run.py [phases]

    phases: comma-separated subset, e.g. "0,1,2,3" or "all" (default)
            valid values: 0 1 2 3 4 5 6 7 8 9 10 11 trf freqdef
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.vt_failure.data import load_analysis_df

OUT_ROOT = results_dir("vt_failure")


def _parse_phases(arg: str) -> set:
    if arg.lower() == "all":
        return set(range(12)) | {"trf"}
    parts = arg.replace(",", " ").split()
    result: set = set()
    for p in parts:
        if p.lower() in ("trf", "freqdef", "8b", "extended_cox", "8c", "latent_cox"):
            result.add(p.lower())
        elif p.lower() in ("8d", "latent_cox_cov"):
            result.add("8d")
        else:
            try:
                result.add(int(p))
            except ValueError:
                pass
    return result


def main(phases_arg: str = "all") -> None:
    phases = _parse_phases(phases_arg)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    print("\n" + "=" * 70)
    print("VT FAILURE ANALYSIS — STARTING")
    print(f"Output directory: {OUT_ROOT}")
    print("=" * 70)

    print("\n[Data] Loading analysis dataset...")
    df = load_analysis_df()
    print(f"[Data] Loaded {len(df):,} runs  |  {int(df['event'].sum()):,} failures  |  "
          f"fields: {df['field'].nunique()}")

    df.to_csv(OUT_ROOT / "tables" / "analysis_dataset.csv", index=False, encoding="utf-8-sig")

    all_results = {}

    if 0 in phases:
        from analysis.workflows.vt_failure import audit
        print("\n[Phase 0] Data Audit...")
        all_results[0] = audit.run(df, OUT_ROOT / "tables" / "step0_audit")

    if 1 in phases:
        from analysis.workflows.vt_failure import phase1
        print("\n[Phase 1] Descriptive Baseline...")
        all_results[1] = phase1.run(df, OUT_ROOT / "figures" / "phase1_descriptive")

    if 2 in phases:
        from analysis.workflows.vt_failure import phase2
        print("\n[Phase 2] Infant Mortality...")
        all_results[2] = phase2.run(df, OUT_ROOT / "figures" / "phase2_infant")

    if 3 in phases:
        from analysis.workflows.vt_failure import phase3
        print("\n[Phase 3] Kaplan-Meier...")
        all_results[3] = phase3.run(df, OUT_ROOT / "figures" / "phase3_km")

    if 4 in phases:
        from analysis.workflows.vt_failure import phase4
        print("\n[Phase 4] Weibull Shape Analysis...")
        all_results[4] = phase4.run(df, OUT_ROOT / "figures" / "phase4_weibull")

    if 5 in phases:
        from analysis.workflows.vt_failure import phase5
        print("\n[Phase 5] Competing Risks (CIF)...")
        all_results[5] = phase5.run(df, OUT_ROOT / "figures" / "phase5_cif")

    if 6 in phases:
        from analysis.workflows.vt_failure import phase6
        print("\n[Phase 6] Duty & Chemistry Proxies...")
        all_results[6] = phase6.run(df, OUT_ROOT / "figures" / "phase6_chemistry")

    if 7 in phases:
        from analysis.workflows.vt_failure import phase7
        print("\n[Phase 7] Confounding Test...")
        all_results[7] = phase7.run(df, OUT_ROOT / "figures" / "phase7_confounding")

    if 8 in phases:
        from analysis.workflows.vt_failure import phase8
        print("\n[Phase 8] Cox Proportional Hazards...")
        all_results[8] = phase8.run(df, OUT_ROOT / "tables" / "phase8_cox")

    if 9 in phases:
        from analysis.workflows.vt_failure import phase9
        print("\n[Phase 9] Vt Deep-Dive...")
        all_results[9] = phase9.run(df, OUT_ROOT / "figures" / "phase9_vt")

    if 10 in phases:
        from analysis.workflows.vt_failure import phase10
        print("\n[Phase 10] CatBoost Infant Classifier...")
        all_results[10] = phase10.run(df, OUT_ROOT / "models" / "phase10_catboost")

    if 11 in phases:
        from analysis.workflows.vt_failure import phase11
        print("\n[Phase 11] Sensitivity Checks...")
        all_results[11] = phase11.run(df, OUT_ROOT / "figures" / "phase11_sensitivity")

    if "trf" in phases or 12 in phases:
        from analysis.workflows.vt_failure import trf_capacity
        print("\n[Phase TRF] TRF Capacity Hypothesis Test...")
        all_results["trf"] = trf_capacity.run(df, OUT_ROOT / "tables" / "phase_trf_capacity")

    if "8d" in phases or "latent_cox_cov" in phases:
        from analysis.workflows.vt_failure import phase8d
        print("\n[Phase 8d] Extended Cox on Latent Weibull — Tier 1/2/3 covariates (Vt)...")
        all_results["8d"] = phase8d.run(df, OUT_ROOT / "tables" / "phase8d_latent_cox_cov")

    if "8c" in phases or "latent_cox" in phases:
        from analysis.workflows.vt_failure import phase8c
        print("\n[Phase 8c] Latent Weibull + Extended Cox (H2S, Vt)...")
        all_results["8c"] = phase8c.run(df, OUT_ROOT / "tables" / "phase8c_latent_cox")

    if "8b" in phases or "extended_cox" in phases:
        from analysis.workflows.vt_failure import phase8b
        print("\n[Phase 8b] Extended Cox — H2S time-varying effect (Vt)...")
        all_results["8b"] = phase8b.run(df, OUT_ROOT / "tables" / "phase8b_extended_cox")

    if "freqdef" in phases or 13 in phases:
        from analysis.workflows.vt_failure import freq_definition_compare
        print("\n[Phase FREQDEF] Frequency Definition Comparison...")
        all_results["freqdef"] = freq_definition_compare.run(df, OUT_ROOT / "tables" / "phase_freqdef_compare")

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE — {elapsed:.0f}s")
    print(f"All outputs saved to: {OUT_ROOT}")
    print(f"{'=' * 70}")

    summary_rows = []
    if 3 in all_results:
        vt_lr_p = all_results[3].get("vt_vs_global_logrank_p")
        if vt_lr_p is not None:
            summary_rows.append({"metric": "vt_vs_global_logrank_p", "value": round(vt_lr_p, 4)})
    if 9 in all_results:
        decomp = all_results[9].get("rmst_decomp", {})
        for k, v in decomp.items():
            if isinstance(v, float) and k != "tau_days":
                summary_rows.append({"metric": k, "value": round(v, 2)})
    if 8 in all_results:
        cox = all_results[8].get("cox_table", pd.DataFrame())
        if not cox.empty and "freq_high_hr" in cox.columns:
            last = cox.dropna(subset=["freq_high_hr"]).iloc[-1]
            summary_rows.append({"metric": "cox_best_model_freq_high_hr", "value": last["freq_high_hr"]})
    if 10 in all_results:
        summary_rows.append({"metric": "catboost_cv_auc",
                              "value": round(all_results[10].get("cv_auc_mean", np.nan), 4)})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUT_ROOT / "tables" / "executive_summary_numbers.csv",
            index=False, encoding="utf-8-sig",
        )
        print("\nKey numbers:")
        for row in summary_rows:
            print(f"  {row['metric']}: {row['value']}")


if __name__ == "__main__":
    phases_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(phases_arg)
