"""Phase B B2 — Nonparametric CIF layer (output slug: phase_b_cif).

Absolute per-cause incidence via the **Aalen–Johansen** CIF (never 1−KM_cause).
Per field-level stratum: a stacked CIF plot (the four mode groups partition
all-cause incidence), cause-specific Nelson–Aalen cumulative hazards, and a
mode-mix table (share of incidence by group at 90 d / 365 d / end-of-follow-up,
from the CIF, not raw counts).  Within Vt: a Gray-style bootstrap comparison of
each mode's CIF, sour vs nonsour.

Outputs under ``results/phase_b_cif/<date>/``:
  tables/b2_mode_mix.csv          — CIF-based incidence share by group & horizon
  tables/b2_cause_cif_points.csv  — CIF per cause at 90/365/end per stratum
  tables/b2_nelson_aalen.csv      — cause-specific NA cumulative hazard points
  tables/b2_vt_gray_test.csv      — sour vs nonsour CIF diff per mode (Vt), bootstrap CI+p
  figures/b2_cif_<stratum>.png    — stacked AJ CIF per big stratum

Run:
    python scripts/run/phase_b_cif.py [n_boot]
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
from analysis.data.failure_modes import MODE_GROUPS
from analysis.models.survival.cif import (
    aalen_johansen_cif,
    all_cause_km,
    cif_by_cause,
    event_code_series,
    mode_mix_at,
    bootstrap_cif_difference,
)

PLOT_MIN_FAILURES = 40      # stacked CIF plot only for well-populated strata
MIX_MIN_FAILURES = 20       # mode-mix table for moderately-populated strata
HORIZONS = (90.0, 365.0)    # + end-of-follow-up per stratum
GROUP_COLORS = {
    "hydraulic": "#4C78A8", "electro-thermal": "#F58518",
    "protector": "#54A24B", "other": "#B279A2",
}


def _nelson_aalen_cumhaz(durations, event_k, t):
    """Cause-specific NA cumulative hazard at time ``t`` (other causes censored)."""
    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter()
    naf.fit(durations, event_observed=event_k)
    return float(np.asarray(naf.cumulative_hazard_at_times([t])).ravel()[0])


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = results_dir("phase_b_cif")
    tbl, figs = out / "tables", out / "figures"

    df = build_competing_risks_df(tte_col="ttf_mix")
    df["stratum_field"] = df["field"].astype(str) + "_" + df["h2s_class"].astype(str)
    df["event_code"] = event_code_series(df["mode_group"], df["event"], MODE_GROUPS)

    mix_rows, cif_pt_rows, na_rows = [], [], []

    for stratum, g in df.groupby("stratum_field", observed=True):
        n_fail = int(g["event"].sum())
        if n_fail < MIX_MIN_FAILURES:
            continue
        dur = g["tte"].to_numpy(float)
        code = g["event_code"].to_numpy(int)
        end = float(dur.max())
        cifs = cif_by_cause(dur, code, MODE_GROUPS)
        km = all_cause_km(dur, (code > 0).astype(int))

        for horizon_name, t in list(zip(("90d", "365d", "end"), (*HORIZONS, end))):
            mix = mode_mix_at(cifs, t)
            for group in MODE_GROUPS:
                cif_val = float(cifs[group].at(t))
                mix_rows.append({
                    "stratum": stratum, "horizon": horizon_name, "time": round(t, 1),
                    "mode_group": group, "cif": round(cif_val, 4),
                    "incidence_share": round(mix[group], 4),
                    "n_failures": n_fail,
                })
            cif_pt_rows.append({
                "stratum": stratum, "horizon": horizon_name, "time": round(t, 1),
                "all_cause_incidence": round(float(km.at(t)), 4),
                **{f"cif_{group}": round(float(cifs[group].at(t)), 4) for group in MODE_GROUPS},
            })

        # Nelson–Aalen cause-specific cumulative hazard at the horizons.
        for group in MODE_GROUPS:
            ev_k = (g["mode_group"].to_numpy() == group) & (g["event"].to_numpy() == 1)
            for horizon_name, t in list(zip(("90d", "365d", "end"), (*HORIZONS, end))):
                na_rows.append({
                    "stratum": stratum, "mode_group": group,
                    "horizon": horizon_name, "time": round(t, 1),
                    "na_cumhaz": round(_nelson_aalen_cumhaz(dur, ev_k.astype(int), t), 4),
                })

        # Stacked CIF plot for well-populated strata.
        if n_fail >= PLOT_MIN_FAILURES:
            grid = np.linspace(0, end, 300)
            fig, ax = plt.subplots(figsize=(8, 5))
            bottom = np.zeros_like(grid)
            for group in MODE_GROUPS:
                vals = np.asarray(cifs[group].at(grid), dtype=float)
                ax.fill_between(grid, bottom, bottom + vals, label=group,
                                color=GROUP_COLORS[group], alpha=0.9)
                bottom += vals
            ax.plot(grid, np.asarray(km.at(grid), dtype=float), color="black",
                    lw=1.3, ls="--", label="1 − S_all (all-cause)")
            ax.set_xlabel("operating days (ttf_mix)")
            ax.set_ylabel("cumulative incidence")
            ax.set_title(f"Phase B B2 — Aalen–Johansen CIF: {stratum}  (n_fail={n_fail})")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(figs / f"b2_cif_{stratum}.png", dpi=130)
            plt.close(fig)

    pd.DataFrame(mix_rows).to_csv(tbl / "b2_mode_mix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cif_pt_rows).to_csv(tbl / "b2_cause_cif_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(na_rows).to_csv(tbl / "b2_nelson_aalen.csv", index=False, encoding="utf-8-sig")
    print(f"[B2] mode-mix rows: {len(mix_rows)}  strata plotted: "
          f"{len(list(figs.glob('b2_cif_*.png')))}")

    # ── Gray-style: sour vs nonsour CIF per mode within Vt ───────────────────
    vt = df[df["field"] == "Vt"].copy()
    vt_end = float(vt["tte"].max())
    gray_rows = []
    for i, group in enumerate(MODE_GROUPS, start=1):
        res = bootstrap_cif_difference(
            vt, group_col="h2s_class", group_a="sour", group_b="nonsour",
            duration_col="tte", event_code_col="event_code", cause_code=i,
            at_times=(90.0, 365.0, vt_end), cluster_col="well_key",
            n_boot=n_boot, seed=42,
        )
        res.insert(0, "mode_group", group)
        gray_rows.append(res)
    gray = pd.concat(gray_rows, ignore_index=True)
    gray = gray.rename(columns={"cif_sour": "cif_sour", "cif_nonsour": "cif_nonsour"})
    gray.to_csv(tbl / "b2_vt_gray_test.csv", index=False, encoding="utf-8-sig")
    print("[B2] Vt sour−nonsour CIF diff (90d), by mode:")
    for _, r in gray[gray["time"] == 90.0].iterrows():
        print(f"       {r['mode_group']:<16} Δ={r['diff_point']:+.3f} "
              f"[{r['diff_ci_lo']:+.3f},{r['diff_ci_hi']:+.3f}] p={r['boot_p_two_sided']:.3f}")

    print(f"[B2] outputs written to: {out}")


if __name__ == "__main__":
    main()
