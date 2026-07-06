"""Phase A T4 — mixed-clock sensitivity.

510 runs (~19%) have no operating-time measurement (``ttf_true_source='missing'``);
for those, ``ttf_mix`` *is* calendar time.  This re-runs field-level Phase 1 + Phase 2
on ttf_mix with and without those runs and reports the B50 shift per stratum.

Produces, under ``results/phase_a_ttf_sensitivity/<date>/``:

  tables/b50_full_vs_excl.csv     — stratum | B50_full | B50_excl | delta_pct | flag(>15%)
  tables/stratum_inventory.csv    — per-stratum n, n_failures, pct_mixed_clock

Run:
    python scripts/run/phase_a_ttf_sensitivity.py
"""
from __future__ import annotations

import sys
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
from analysis.workflows.esp_survival.data_mart import load_mart_df, stratum_summary
from analysis.workflows.esp_survival import phase2_mixture
from analysis.models.survival.latent_weibull_competing_risks import latent_life_quantile

FLAG_PCT = 15.0


def _b50_by_stratum(models: dict) -> dict[str, float]:
    out = {}
    for stratum, model in models.items():
        b50 = latent_life_quantile(0.50, model)
        out[stratum] = float(b50) if np.isfinite(b50) else float("nan")
    return out


def main() -> None:
    out = results_dir("phase_a_ttf_sensitivity")
    tbl = out / "tables"

    df = load_mart_df(tte_col="ttf_mix")
    n_missing = int((df["ttf_true_source"] == "missing").sum())
    print(f"[T4] {len(df)} runs; {n_missing} with ttf_true_source='missing' "
          f"({100*n_missing/len(df):.1f}% mixed clock)")

    # ── Per-stratum inventory with pct_mixed_clock ────────────────────────────
    inv = stratum_summary(df)
    mixed = df.groupby("stratum").apply(
        lambda g: 100.0 * (g["ttf_true_source"] == "missing").mean()
    ).rename("pct_mixed_clock").reset_index()
    inv = inv.merge(mixed, on="stratum", how="left")
    inv["pct_mixed_clock"] = inv["pct_mixed_clock"].round(1)
    inv.to_csv(tbl / "stratum_inventory.csv", index=False, encoding="utf-8-sig")
    print("\n[T4] Stratum inventory (pct_mixed_clock):")
    print(inv[["stratum", "n_failures", "pct_mixed_clock"]].to_string(index=False))

    # ── Full vs excluded Phase 2 ──────────────────────────────────────────────
    print("\n[T4] Phase 2 on FULL ttf_mix...")
    r_full = phase2_mixture.run(df, out / "figures_full")
    b50_full = _b50_by_stratum(r_full["models"])

    df_excl = df[df["ttf_true_source"] != "missing"].copy()
    print(f"\n[T4] Phase 2 EXCLUDING {n_missing} mixed-clock runs "
          f"({len(df_excl)} remain)...")
    r_excl = phase2_mixture.run(df_excl, out / "figures_excl")
    b50_excl = _b50_by_stratum(r_excl["models"])

    # ── Comparison ────────────────────────────────────────────────────────────
    rows = []
    for stratum in sorted(set(b50_full) | set(b50_excl)):
        bf = b50_full.get(stratum, float("nan"))
        be = b50_excl.get(stratum, float("nan"))
        delta = (100.0 * (be - bf) / bf) if np.isfinite(bf) and bf > 0 and np.isfinite(be) else float("nan")
        rows.append({
            "stratum": stratum,
            "B50_full": round(bf, 1) if np.isfinite(bf) else None,
            "B50_excl": round(be, 1) if np.isfinite(be) else None,
            "delta_pct": round(delta, 1) if np.isfinite(delta) else None,
            "flag_gt15pct": bool(np.isfinite(delta) and abs(delta) > FLAG_PCT),
        })
    cmp = pd.DataFrame(rows)
    cmp = cmp.merge(inv[["stratum", "pct_mixed_clock"]], on="stratum", how="left")
    cmp.to_csv(tbl / "b50_full_vs_excl.csv", index=False, encoding="utf-8-sig")
    print("\n[T4] B50 full vs excluded (mixed-clock sensitivity):")
    print(cmp.to_string(index=False))

    flagged = cmp[cmp["flag_gt15pct"]]
    if not flagged.empty:
        print(f"\n[T4] ⚠ {len(flagged)} strata move >{FLAG_PCT:.0f}% — flag in ESP survival README:")
        print(flagged[["stratum", "delta_pct", "pct_mixed_clock"]].to_string(index=False))
    else:
        print(f"\n[T4] No stratum moves >{FLAG_PCT:.0f}% — mixed-clock effect is small.")

    print(f"\n[T4] Outputs written to: {out}")


if __name__ == "__main__":
    main()
