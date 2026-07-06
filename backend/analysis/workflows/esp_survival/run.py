"""ESP Survival Analysis — main orchestrator.

Usage:
    python backend/analysis/workflows/esp_survival/run.py [phases]

    phases: comma-separated subset, e.g. "0,1,2,3" or "all" (default)
            valid values: 0 1 2 3 4 5 6
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.workflows.esp_survival.data import load_failures_df, stratum_summary


def _try_load_phase2_cache() -> dict | None:
    """Find the most recent Phase 2 models.json and deserialize it."""
    from analysis.workflows.esp_survival.phase2_mixture import load_models
    phase2_root = RESULTS_ROOT / "esp_survival_phase2_mixture"
    if not phase2_root.exists():
        return None
    dated_dirs = sorted(phase2_root.glob("????-??-??"), reverse=True)
    for d in dated_dirs:
        json_path = d / "models" / "phase2_models.json"
        if json_path.exists():
            try:
                models, gb1, gb2, two_stage = load_models(json_path)
                print(f"[Phase 2] Loaded {len(models)} cached models from {json_path.parent.parent.name}")
                return {"models": models, "global_beta1": gb1, "global_beta2": gb2,
                        "two_stage_strata": two_stage, "kmf_cache": {}}
            except Exception as exc:
                print(f"[Phase 2] Cache load failed ({exc}), re-running Phase 2 is required.")
    return None


def _parse_phases(arg: str) -> set[int]:
    if arg.lower() == "all":
        return set(range(7))
    parts = arg.replace(",", " ").split()
    result: set[int] = set()
    for p in parts:
        try:
            result.add(int(p))
        except ValueError:
            pass
    return result


def main(phases_arg: str = "all") -> None:
    phases = _parse_phases(phases_arg)
    t_start = time.monotonic()

    print("\n" + "=" * 70)
    print("ESP SURVIVAL ANALYSIS — STARTING")
    print(f"Phases: {sorted(phases)}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[Data] Loading ESP failures dataset...")
    df = load_failures_df()
    print(f"[Data] {len(df):,} runs  |  {int(df['event'].sum()):,} failures  "
          f"|  {df['stratum'].nunique()} strata  |  {df['field_clean'].nunique()} fields")

    strat = stratum_summary(df)
    print("\n[Data] Stratum overview:")
    print(strat[["stratum", "n_failures", "fit_mode"]].to_string(index=False))

    # Store shared state across phases
    state: dict = {"df": df, "models": {}, "kmf_cache": {}}

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    if 0 in phases:
        from analysis.workflows.esp_survival import phase0_data_audit
        out = results_dir("esp_survival_phase0_audit")
        print("\n" + "-" * 60)
        print("[Phase 0] Data audit")
        phase0_data_audit.run(df, out / "tables")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if 1 in phases:
        from analysis.workflows.esp_survival import phase1_single_weibull
        out = results_dir("esp_survival_phase1_weibull")
        print("\n" + "-" * 60)
        print("[Phase 1] Single Weibull MLE per stratum")
        r1 = phase1_single_weibull.run(df, out / "figures")
        state["kmf_cache"].update(r1.get("kmf_cache", {}))

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if 2 in phases:
        from analysis.workflows.esp_survival import phase2_mixture
        out = results_dir("esp_survival_phase2_mixture")
        print("\n" + "-" * 60)
        print("[Phase 2] K=2 latent Weibull mixture per stratum")
        r2 = phase2_mixture.run(df, out / "figures")
        state["models"].update(r2.get("models", {}))
        state["global_beta1"] = r2.get("global_beta1")
        state["global_beta2"] = r2.get("global_beta2")
        state["two_stage_strata"] = r2.get("two_stage_strata", [])
        if not state["kmf_cache"]:
            state["kmf_cache"].update(r2.get("kmf_cache", {}))

    # ── Load Phase 2 cache for downstream phases if not in state ─────────────
    needs_models = ({3, 4, 6} & phases) - ({2} & phases)
    if needs_models and not state["models"]:
        cached = _try_load_phase2_cache()
        if cached:
            state["models"].update(cached["models"])
            state.setdefault("global_beta1", cached["global_beta1"])
            state.setdefault("global_beta2", cached["global_beta2"])
            state.setdefault("two_stage_strata", cached["two_stage_strata"])

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    if 3 in phases and state["models"]:
        from analysis.workflows.esp_survival import phase3_diagnostics
        out = results_dir("esp_survival_phase3_diagnostics")
        print("\n" + "-" * 60)
        print("[Phase 3] Mixture diagnostics — bootstrap, GOF, component assignment")
        phase3_diagnostics.run(
            df, out / "tables", state["models"], state["kmf_cache"],
            global_beta1=state.get("global_beta1"),
            global_beta2=state.get("global_beta2"),
            two_stage_strata=state.get("two_stage_strata", []),
        )
    elif 3 in phases and not state["models"]:
        print("\n[Phase 3] Skipped — no mixture models available (run Phase 2 first).")

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    if 4 in phases:
        from analysis.workflows.esp_survival import phase4_contractor
        out = results_dir("esp_survival_phase4_contractor")
        print("\n" + "-" * 60)
        print("[Phase 4] Contractor effect")
        phase4_contractor.run(df, out / "tables", state["models"])

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    if 5 in phases:
        from analysis.workflows.esp_survival import phase5_extended_cox
        out = results_dir("esp_survival_phase5_cox")
        print("\n" + "-" * 60)
        print("[Phase 5] Covariate exploration + Extended Cox")
        r5 = phase5_extended_cox.run(df, out / "tables")
        state["cox_results"] = r5

    # ── Phase 6 ──────────────────────────────────────────────────────────────
    if 6 in phases and state["models"]:
        from analysis.workflows.esp_survival import phase6_rul
        out = results_dir("esp_survival_phase6_rul")
        print("\n" + "-" * 60)
        print("[Phase 6] RUL calculator")
        # Extract Cox coef from Phase 5 if available
        cox_coef: dict[str, float] | None = None
        if "cox_results" in state:
            multi_df = state["cox_results"].get("multivariate", None)
            if multi_df is not None and not multi_df.empty:
                try:
                    coef_col = "coef"
                    name_col = multi_df.columns[0]
                    cox_coef = dict(zip(multi_df[name_col], multi_df[coef_col]))
                except Exception:
                    pass
        phase6_rul.run(df, out / "figures", state["models"], cox_coef)
    elif 6 in phases and not state["models"]:
        print("\n[Phase 6] Skipped — no mixture models available (run Phase 2 first).")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    print("\n" + "=" * 70)
    print(f"ESP SURVIVAL ANALYSIS COMPLETE — {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    phases_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(phases_arg)
