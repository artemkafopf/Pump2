"""ESP survival analysis with ttf_mix as time-to-event.

Re-runs Phase 1 (single Weibull) and Phase 2 (K=2 EM mixture) using
ttf_mix = ttf_true_best_days if available, else run_days (calendar fallback).

Data source: mart__weibull_input in the SQLite warehouse.

Output slugs:
    esp_survival_ttf_mix_phase1   — single Weibull per stratum
    esp_survival_ttf_mix_phase2   — K=2 mixture per stratum
    esp_survival_ttf_mix_compare  — side-by-side params vs calendar-TTF baseline

Usage:
    python scripts/run/run_esp_survival_ttf_mix.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.workflows.esp_survival.data_mart import load_mart_df, stratum_summary


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_baseline_phase2() -> tuple[Path, str] | None:
    """Return (path, kind) for the most recent calendar-TTF Phase 2 results.

    Prefers the models.json (exact params); falls back to the figures CSV.
    Returns None if neither is found.
    """
    phase2_root = RESULTS_ROOT / "esp_survival_phase2_mixture"
    if not phase2_root.exists():
        return None
    for d in sorted(phase2_root.glob("????-??-??"), reverse=True):
        json_path = d / "models" / "phase2_models.json"
        if json_path.exists():
            return json_path, "json"
        csv_path = d / "figures" / "phase2_mixture_params.csv"
        if csv_path.exists():
            return csv_path, "csv"
    return None


def _load_models_json(path: Path) -> dict[str, dict]:
    """Load raw models dict from a phase2_models.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["models"]  # stratum → {weight_1, beta1, eta1, beta2, eta2}


def _load_models_csv(path: Path) -> dict[str, dict]:
    """Load raw models dict from the phase2_mixture_params CSV fallback."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        stratum = row["stratum"]
        result[stratum] = {
            "weight_1": float(row["w1"]),
            "beta1": float(row["beta1"]),
            "eta1": float(row["eta1_days"]),
            "beta2": float(row["beta2"]),
            "eta2": float(row["eta2_days"]),
        }
    return result


def _build_param_row(stratum: str, m: dict) -> dict:
    """Compute B10 / B50 from a raw model dict."""
    from analysis.models.survival.latent_weibull_competing_risks import (
        TwoComponentLatentWeibullModel,
        WeibullParameters,
        latent_life_quantile,
    )
    model = TwoComponentLatentWeibullModel(
        weight_1=m["weight_1"],
        component_1=WeibullParameters(beta=m["beta1"], eta=m["eta1"], label="C1"),
        component_2=WeibullParameters(beta=m["beta2"], eta=m["eta2"], label="C2"),
    )
    b10 = latent_life_quantile(0.10, model)
    b50 = latent_life_quantile(0.50, model)
    return {
        "stratum": stratum,
        "w1": round(m["weight_1"], 3),
        "beta1": round(m["beta1"], 3),
        "eta1": round(m["eta1"], 1),
        "beta2": round(m["beta2"], 3),
        "eta2": round(m["eta2"], 1),
        "B10": round(b10, 0) if np.isfinite(b10) else None,
        "B50": round(b50, 0) if np.isfinite(b50) else None,
    }


def _compare_models(
    baseline: dict[str, dict],
    new: dict[str, dict],
) -> pd.DataFrame:
    """Side-by-side comparison of baseline (calendar TTF) vs new (ttf_mix) params."""
    all_strata = sorted(set(baseline) | set(new))
    rows = []
    for stratum in all_strata:
        row: dict = {"stratum": stratum}
        if stratum in baseline:
            b = _build_param_row(stratum, baseline[stratum])
            for k, v in b.items():
                if k != "stratum":
                    row[f"base_{k}"] = v
        else:
            for k in ("w1", "beta1", "eta1", "beta2", "eta2", "B10", "B50"):
                row[f"base_{k}"] = None

        if stratum in new:
            n = _build_param_row(stratum, new[stratum])
            for k, v in n.items():
                if k != "stratum":
                    row[f"mix_{k}"] = v
        else:
            for k in ("w1", "beta1", "eta1", "beta2", "eta2", "B10", "B50"):
                row[f"mix_{k}"] = None

        if stratum in baseline and stratum in new:
            base_b50 = row.get("base_B50") or float("nan")
            mix_b50 = row.get("mix_B50") or float("nan")
            row["B50_delta_pct"] = round(100 * (mix_b50 - base_b50) / max(base_b50, 1), 1) if np.isfinite(base_b50) and np.isfinite(mix_b50) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.monotonic()

    print("\n" + "=" * 70)
    print("ESP SURVIVAL — TTF-MIX ANALYSIS")
    print("TTE source: ttf_true_best_days (if available) else run_days")
    print("Data: mart__weibull_input (SQLite warehouse)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Data] Loading mart__weibull_input...")
    df = load_mart_df()
    n_total = len(df)
    n_fail = int(df["event"].sum())
    n_has_true = int((df["ttf_true_best_days"].notna()).sum())
    n_calendar = n_total - n_has_true
    print(
        f"[Data] {n_total:,} runs  |  {n_fail:,} failures  "
        f"|  {df['stratum'].nunique()} strata"
    )
    print(
        f"[Data] ttf_mix source: {n_has_true:,} runs use true-operating-days, "
        f"{n_calendar:,} fall back to calendar days"
    )
    print(
        f"[Data] avg ttf_mix (failures): "
        f"{df.loc[df['event']==1,'tte'].mean():.1f}d  "
        f"(vs avg run_days: {df.loc[df['event']==1,'run_days'].mean():.1f}d)"
    )

    strat = stratum_summary(df)
    print("\n[Data] Stratum overview:")
    print(strat[["stratum", "n_failures", "fit_mode"]].to_string(index=False))

    strat_out = results_dir("esp_survival_ttf_mix_phase1")
    strat.to_csv(strat_out / "tables" / "stratum_summary.csv", index=False, encoding="utf-8-sig")

    # ── Phase 1 — single Weibull ─────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("[Phase 1] Single Weibull MLE per stratum")
    from analysis.workflows.esp_survival import phase1_single_weibull
    out1 = results_dir("esp_survival_ttf_mix_phase1")
    r1 = phase1_single_weibull.run(df, out1 / "figures")

    # ── Phase 2 — K=2 EM mixture ─────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("[Phase 2] K=2 latent Weibull mixture (EM) per stratum")
    from analysis.workflows.esp_survival import phase2_mixture
    out2 = results_dir("esp_survival_ttf_mix_phase2")
    r2 = phase2_mixture.run(df, out2 / "figures")
    new_models_raw: dict[str, dict] = {}
    for stratum, model in r2["models"].items():
        new_models_raw[stratum] = {
            "weight_1": model.weight_1,
            "beta1": model.component_1.beta,
            "eta1": model.component_1.eta,
            "beta2": model.component_2.beta,
            "eta2": model.component_2.eta,
        }

    # ── Comparison with calendar-TTF baseline ─────────────────────────────────
    print("\n" + "-" * 60)
    print("[Compare] Loading calendar-TTF baseline Phase 2 models...")
    baseline_found = _find_baseline_phase2()
    if baseline_found is None:
        print("[Compare] No baseline Phase 2 models found — skipping comparison.")
    else:
        baseline_path, baseline_kind = baseline_found
        print(f"[Compare] Baseline ({baseline_kind}): {baseline_path.parent.parent.name}")
        baseline_raw = (
            _load_models_json(baseline_path)
            if baseline_kind == "json"
            else _load_models_csv(baseline_path)
        )
        cmp_df = _compare_models(baseline_raw, new_models_raw)

        out_cmp = results_dir("esp_survival_ttf_mix_compare")
        cmp_path = out_cmp / "tables" / "ttf_mix_vs_calendar_params.csv"
        cmp_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print(f"[Compare] Saved comparison table: {cmp_path}")

        print("\n[Compare] B50 shift (ttf_mix vs calendar-TTF, % change):")
        shared = cmp_df.dropna(subset=["B50_delta_pct"])
        for _, row in shared.sort_values("stratum").iterrows():
            delta = float(row["B50_delta_pct"])
            sign = "+" if delta >= 0 else ""
            base_b50 = row.get("base_B50", "—")
            mix_b50 = row.get("mix_B50", "—")
            print(
                f"  {row['stratum']:30s}  "
                f"baseline B50={base_b50:>5}d  "
                f"ttf_mix B50={mix_b50:>5}d  "
                f"delta={sign}{delta:.1f}%"
            )

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 70)
    print(f"TTF-MIX ANALYSIS COMPLETE — {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
