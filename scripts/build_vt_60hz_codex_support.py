from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from analysis.paths import results_dir, RESULTS_ROOT

# Input: vt_failure analysis output (legacy path; new path: results/vt_failure/LATEST/)
SRC = REPO / "analysis_outputs" / "vt_failure_analysis"
_SLUG = "vt_60hz_codex_support"
OUT = results_dir(_SLUG)

GRPS = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]


def infant_rate(g: pd.DataFrame, threshold: int = 90) -> tuple[float, int, int]:
    eligible = g[(g["duration"] >= threshold) | (g["event"] == 1)]
    if eligible.empty:
        return np.nan, 0, 0
    n_inf = int(((eligible["event"] == 1) & (eligible["duration"] < threshold)).sum())
    return float(n_inf / len(eligible) * 100.0), len(eligible), n_inf


def logrank_hi_vs_normal(sub: pd.DataFrame) -> float:
    n = sub[sub["freq_group"] == "Normal (50–55 Hz)"].dropna(subset=["duration"])
    h = sub[sub["freq_group"] == "High (>55 Hz)"].dropna(subset=["duration"])
    if int(n["event"].sum()) < 5 or int(h["event"].sum()) < 5:
        return np.nan
    return float(logrank_test(n["duration"], h["duration"], n["event"], h["event"]).p_value)


def build_rerun_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    first = df.sort_values("mount_date").groupby("well_key", observed=True).first().reset_index()

    for level, level_df in [("all_runs", df), ("first_run", first)]:
        for scope, scope_df in [("Global", level_df), ("Vt", level_df[level_df["field"] == "Vt"])]:
            for grp in GRPS:
                g = scope_df[scope_df["freq_group"] == grp].dropna(subset=["duration"])
                if g.empty:
                    continue
                ir, n_eligible, n_inf = infant_rate(g, 90)
                rows.append(
                    {
                        "level": level,
                        "scope": scope,
                        "freq_group": grp,
                        "n_runs": len(g),
                        "n_fail": int(g["event"].sum()),
                        "median_duration_all": float(g["duration"].median()),
                        "median_duration_failures": float(g.loc[g["event"] == 1, "duration"].median()) if int(g["event"].sum()) > 0 else np.nan,
                        "infant_rate_90d_pct": ir,
                        "n_eligible_90d": n_eligible,
                        "n_infant_90d": n_inf,
                    }
                )
            rows.append(
                {
                    "level": level,
                    "scope": scope,
                    "freq_group": "High vs Normal logrank p",
                    "n_runs": np.nan,
                    "n_fail": np.nan,
                    "median_duration_all": np.nan,
                    "median_duration_failures": logrank_hi_vs_normal(scope_df),
                    "infant_rate_90d_pct": np.nan,
                    "n_eligible_90d": np.nan,
                    "n_infant_90d": np.nan,
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "rerun_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    well_counts = df.groupby("well_key", observed=True).size()
    pd.DataFrame(
        [
            {
                "n_wells": int(well_counts.size),
                "share_multi_run_wells": float((well_counts > 1).mean()),
                "mean_runs_per_well": float(well_counts.mean()),
                "median_runs_per_well": float(well_counts.median()),
            }
        ]
    ).to_csv(out_dir / "rerun_profile.csv", index=False, encoding="utf-8-sig")


def build_definition_fallback(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for scope, scope_df in [("Global", df), ("Vt", df[df["field"] == "Vt"])]:
        hi = scope_df[scope_df["freq_group_pct"] == "High (>55 Hz)"]
        rows.append(
            {
                "scope": scope,
                "n_rows": len(scope_df),
                "missing_freq_above_55hz_pct_share": float(scope_df["freq_above_55hz_pct"].isna().mean()),
                "high_pct_group_runs": int(len(hi)),
                "high_pct_group_fallback_share": float(hi["freq_above_55hz_pct"].isna().mean()) if len(hi) else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "definition_fallback_check.csv", index=False, encoding="utf-8-sig")


def build_rmst_check(out_dir: Path) -> None:
    decomp = pd.read_csv(SRC / "phase9_vt" / "p9b_rmst_decomposition.csv")
    d = decomp.iloc[0].to_dict()
    residual = d["rmst_vt_minus_global"] - d["mix_effect_days"] - d["withincategory_effect_days"]
    pd.DataFrame(
        [
            {
                **d,
                "residual_days_unexplained": residual,
            }
        ]
    ).to_csv(out_dir / "rmst_decomposition_check.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC / "analysis_dataset.csv", low_memory=False)
    build_rerun_sensitivity(df, OUT)
    build_definition_fallback(df, OUT)
    build_rmst_check(OUT)
    print(f"Saved support tables to {OUT}")


if __name__ == "__main__":
    main()
