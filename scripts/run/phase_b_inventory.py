"""Phase B B1 — Competing-risks inventory (output slug: phase_b_inventory).

Mode-group × stratum event counts regenerated from the warehouse post-hygiene,
at both field-level and field×contractor granularity, with per-stratum mixed-clock
share and fit-eligibility flags.  Asserts **zero unlabelled failures** (a failed
run with no node label would silently vanish from every cause-specific model).

Fit-eligibility rule (Phase A ΔAIC discipline, per cause-stratum):
    ≥40 events → K=2 mixture eligible
    ≥20 events → single Weibull
    <20 events → nonparametric only

Outputs under ``results/phase_b_inventory/<date>/``:
  tables/b1_inventory_field.csv        — mode_group × field-stratum counts + flags
  tables/b1_inventory_contractor.csv   — mode_group × field×contractor counts
  tables/b1_stratum_clock.csv          — pct_mixed_clock per stratum
  tables/b1_mode_group_map.csv         — frozen §0.1 mapping (single source of truth)
  figures/b1_mode_group_by_field.png   — stacked failure counts by field

Run:
    python scripts/run/phase_b_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.data.failure_modes import MODE_GROUPS, export_mode_group_csv

K2_MIN_EVENTS = 40
WEIBULL_MIN_EVENTS = 20

# Group-agnostic colours for the stacked figure (brand-neutral, colourblind-safe).
GROUP_COLORS = {
    "hydraulic": "#4C78A8",
    "electro-thermal": "#F58518",
    "protector": "#54A24B",
    "other": "#B279A2",
}


def _fit_flag(n_events: int) -> str:
    if n_events >= K2_MIN_EVENTS:
        return "K2_eligible"
    if n_events >= WEIBULL_MIN_EVENTS:
        return "single_weibull"
    return "nonparametric_only"


def _inventory(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """Long mode_group × stratum event counts + fit-eligibility flag."""
    rows = []
    for stratum, g in df.groupby(stratum_col, observed=True):
        for group in MODE_GROUPS:
            n_ev = int(g[f"event_{group}"].sum())
            rows.append({
                "stratum": stratum,
                "mode_group": group,
                "n_runs": int(len(g)),
                "n_events": n_ev,
                "fit_flag": _fit_flag(n_ev),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["stratum", "mode_group"]).reset_index(drop=True)


def main() -> None:
    out = results_dir("phase_b_inventory")
    tbl, figs = out / "tables", out / "figures"

    df = build_competing_risks_df(tte_col="ttf_mix")
    # Field-level stratum = field + h2s_class (Vt_sour / Vt_nonsour / *_nonsour),
    # pooling contractors — matches the §0.1 illustrative table and the baseline
    # survival strata.  stratum_key adds the contractor for the finer view.
    df["stratum_field"] = df["field"].astype(str) + "_" + df["h2s_class"].astype(str)

    # ── zero-unlabelled-failures assertion (loud) ────────────────────────────
    unlabelled = int(((df["event"] == 1) & (df["mode_group"] == "")).sum())
    assert unlabelled == 0, f"{unlabelled} unlabelled failures — abort"
    total_fail = int(df["event"].sum())
    cause_sum = int(sum(df[f"event_{g}"].sum() for g in MODE_GROUPS))
    assert cause_sum == total_fail, f"cause sum {cause_sum} != failures {total_fail}"
    print(f"[B1] runs={len(df)}  failures={total_fail}  "
          f"(cause indicators partition exactly: {cause_sum})  unlabelled=0")

    # ── freeze the confirmed §0.1 mapping as the single source of truth ──────
    export_mode_group_csv(tbl / "b1_mode_group_map.csv")

    # ── field-level and contractor-level inventories ─────────────────────────
    field_inv = _inventory(df, "stratum_field")
    field_inv.to_csv(tbl / "b1_inventory_field.csv", index=False, encoding="utf-8-sig")

    contractor_inv = _inventory(df, "stratum_key")   # field_h2sclass_contractor
    contractor_inv.to_csv(tbl / "b1_inventory_contractor.csv", index=False, encoding="utf-8-sig")

    # ── per-stratum mixed-clock share ────────────────────────────────────────
    clock = (
        df.groupby("stratum_key", observed=True)
        .agg(n_runs=("event", "size"),
             n_failures=("event", "sum"),
             pct_mixed_clock=("pct_mixed_clock", lambda s: round(100.0 * float(s.mean()), 1)))
        .reset_index()
        .sort_values("stratum_key")
    )
    clock.to_csv(tbl / "b1_stratum_clock.csv", index=False, encoding="utf-8-sig")

    # ── group totals (matches §0.1) ──────────────────────────────────────────
    totals = (field_inv.groupby("mode_group")["n_events"].sum()
              .reindex(MODE_GROUPS).astype(int))
    print("[B1] mode-group failure totals:")
    for g in MODE_GROUPS:
        print(f"       {g:<16} {totals[g]:>4}")

    # ── stacked bar figure: failures by field × mode group ───────────────────
    pivot = (field_inv.pivot(index="stratum", columns="mode_group", values="n_events")
             .reindex(columns=MODE_GROUPS).fillna(0))
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot))
    for group in MODE_GROUPS:
        ax.bar(pivot.index, pivot[group], bottom=bottom,
               label=group, color=GROUP_COLORS[group])
        bottom += pivot[group].to_numpy()
    ax.set_ylabel("failures")
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title("Phase B B1 — failures by field stratum × mode group (clock=ttf_mix)")
    ax.legend(title="mode group", frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "b1_mode_group_by_field.png", dpi=130)
    plt.close(fig)

    print(f"[B1] outputs written to: {out}")


if __name__ == "__main__":
    main()
