"""Phase A T2 — bootstrap 95% CIs on every operational B50 table.

Well-cluster bootstrap (resample wells, refit EM) for each stratum in the field-level
(9) and contractor-level (17) ttf_mix Phase 2 tables.  Adds b10/b50/w1 [lo, hi]
columns and a reliability flag (degenerate-resample share > 30% ⇒ unreliable).

Produces, under ``results/phase_a_b50_ci/<date>/``:
  tables/field_b50_ci.csv        — field-level params + 95% CIs
  tables/contractor_b50_ci.csv   — contractor-level params + 95% CIs
  tables/recommended_b50.csv     — B50 [lo, hi] recommendation table (intervals, not adjectives)

Run (long; start in background):
    python scripts/run/phase_a_b50_ci.py [n_boot]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.workflows.esp_survival.data_mart import load_mart_df, MIN_FAILURES_TWO_STAGE
from analysis.workflows.esp_survival import phase2_mixture
from analysis.models.survival.bootstrap_ci import bootstrap_stratum_ci

DEFAULT_N_BOOT = 200


def _run_level(df: pd.DataFrame, stratum_col: str, out_figs: Path,
               n_boot: int) -> pd.DataFrame:
    """Fit Phase 2 for a stratification and bootstrap every fitted stratum."""
    work = df.copy()
    work["stratum"] = df[stratum_col]
    # keep strata with enough failures for a K=2 fit
    ev = work.groupby("stratum")["event"].sum()
    keep = ev[ev >= MIN_FAILURES_TWO_STAGE].index
    work = work[work["stratum"].isin(keep)].copy()

    r2 = phase2_mixture.run(work, out_figs)
    models = r2["models"]
    two_stage = set(r2.get("two_stage_strata", []))
    gb1, gb2 = r2.get("global_beta1"), r2.get("global_beta2")

    rows = []
    for stratum, model in models.items():
        g = work[work["stratum"] == stratum]
        is_two = stratum in two_stage
        t0 = time.monotonic()
        ci = bootstrap_stratum_ci(
            g, model, n_boot=n_boot,
            fix_beta1=gb1 if is_two else None,
            fix_beta2=gb2 if is_two else None,
        )
        rows.append({
            "stratum": stratum,
            "n_runs": len(g),
            "n_failures": int(g["event"].sum()),
            "fit_mode": "two_stage" if is_two else "independent",
            "beta1": round(model.component_1.beta, 3),
            "eta1": round(model.component_1.eta, 1),
            "beta2": round(model.component_2.beta, 3),
            "eta2": round(model.component_2.eta, 1),
            **ci,
        })
        print(f"  {stratum:26s} B50={ci['b50_point']:>6} "
              f"[{ci['b50_lo']}, {ci['b50_hi']}]  "
              f"degen={ci['degenerate_frac']:.0%} reliable={ci['reliable']} "
              f"({time.monotonic()-t0:.0f}s)")
    return pd.DataFrame(rows).sort_values("n_failures", ascending=False)


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_BOOT
    out = results_dir("phase_a_b50_ci")
    tbl = out / "tables"

    df = load_mart_df(tte_col="ttf_mix")
    df["stratum_field"] = df["field_clean"] + "_" + df["h2s_class"]
    df["stratum_ctr"] = (
        df["field_clean"] + "_" + df["h2s_class"] + "_" + df["contractor_group"]
    )
    print(f"[T2] {len(df)} runs; bootstrap n_boot={n_boot} (well-cluster)")

    print("\n[T2] Field-level (9 strata):")
    field = _run_level(df, "stratum_field", out / "figures_field", n_boot)
    field.to_csv(tbl / "field_b50_ci.csv", index=False, encoding="utf-8-sig")

    print("\n[T2] Contractor-level (~17 strata):")
    ctr = _run_level(df, "stratum_ctr", out / "figures_ctr", n_boot)
    ctr.to_csv(tbl / "contractor_b50_ci.csv", index=False, encoding="utf-8-sig")

    # Recommended B50 with intervals (contractor-level, reliable first)
    rec = ctr[["stratum", "n_failures", "b50_point", "b50_lo", "b50_hi",
               "reliable", "degenerate_frac"]].copy()
    rec["b50_interval"] = rec.apply(
        lambda r: (f"{r['b50_point']:.0f} [{r['b50_lo']:.0f}, {r['b50_hi']:.0f}]"
                   if r["reliable"] and pd.notna(r["b50_lo"])
                   else f"{r['b50_point']:.0f} (unreliable — wide/degenerate)"),
        axis=1,
    )
    rec.to_csv(tbl / "recommended_b50.csv", index=False, encoding="utf-8-sig")
    print("\n[T2] Recommended B50 (intervals, not adjectives):")
    print(rec[["stratum", "n_failures", "b50_interval"]].to_string(index=False))

    print(f"\n[T2] Outputs written to: {out}")


if __name__ == "__main__":
    main()
