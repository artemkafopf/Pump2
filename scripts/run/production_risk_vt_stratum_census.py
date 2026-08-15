"""Outcome census per Vt stratum — what each survival fit is actually made of.

For every stratum (sour/nonsour x Pooled/brt/slb/oth), split the runs into the
three things the likelihood treats differently:

  * **failures**       — cause-specific events (``event == 1``);
  * **ГТМ/ППР**        — preventive pulls, censored;  a high share pulled EARLY
    (median below the failure median) is informative censoring, i.e. the fit is a
    with-ГТМ hazard, not the pump's intrinsic life;
  * **прочие pulls**   — closed with no recognised failure reason and no failed
    unit stamped; censored by ``classify``'s conservative fallback;
  * **running**        — right-censored at the analysis date.

Medians/means of time-to-event are reported per group on both clocks because the
*ordering* between them is diagnostic: censored-early < failure median means the
preventive programme is biting.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_optime as O  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402

AS_OF = "2026-07-16"
WORKOVER = {"гтм", "ппр"}


def classify_status(g: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    ended = g["end"].notna() & (g["end"] <= as_of)
    reason = g["pull_reason"].fillna("").astype(str).str.strip().str.lower()
    return pd.Series(
        np.where(~ended, "running",
                 np.where(g["event"] == 1, "failure",
                          np.where(reason.isin(WORKOVER), "gtm", "other_pull"))),
        index=g.index)


def stratum_row(name: str, g: pd.DataFrame) -> dict:
    n = len(g)
    row = {"stratum": name, "runs": n}
    for key, label in (("failure", "fail"), ("gtm", "gtm"),
                       ("other_pull", "oth"), ("running", "run")):
        sub = g[g["status"] == key]
        row[f"n_{label}"] = len(sub)
        row[f"pct_{label}"] = round(100.0 * len(sub) / n, 1) if n else np.nan
        for clock, col in (("cal", "t_cal"), ("op", "t_mix")):
            row[f"med_{label}_{clock}"] = round(float(sub[col].median()), 1) if len(sub) else np.nan
            row[f"mean_{label}_{clock}"] = round(float(sub[col].mean()), 1) if len(sub) else np.nan
    row["pct_censored"] = round(100.0 * (n - row["n_fail"]) / n, 1) if n else np.nan
    # informative-censoring flag: preventive pulls happening EARLIER than failures
    row["gtm_early"] = (
        "yes" if (row["n_gtm"] >= 5 and np.isfinite(row["med_gtm_cal"])
                  and row["med_gtm_cal"] < row["med_fail_cal"]) else "no")
    return row


def main() -> None:
    out = results_dir("production_risk_vt_weibull_grid")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pop = P.build(AS_OF)
    vt = pop[pop["field"] == "Vt"].copy()
    vt = P.add_time_scales(vt, AS_OF)
    vt = O.measure(vt, as_of=pd.Timestamp(AS_OF))
    vt["status"] = classify_status(vt, pd.Timestamp(AS_OF))

    rows = []
    for h2s in ("sour", "nonsour"):
        base = vt[vt["h2s_class"] == h2s]
        rows.append(stratum_row(f"Vt_{h2s}_Pooled", base))
        for ctr in ("brt", "slb", "oth"):
            g = base[base["contractor_group"] == ctr]
            if len(g):
                rows.append(stratum_row(f"Vt_{h2s}_{ctr}", g))
    rows.append(stratum_row("Vt_ALL", vt))

    df = pd.DataFrame(rows)
    df.to_csv(tables / "vt_stratum_census.csv", index=False, encoding="utf-8-sig")

    counts = df[["stratum", "runs", "n_fail", "n_gtm", "n_oth", "n_run",
                 "pct_censored", "gtm_early"]]
    cal = df[["stratum", "med_fail_cal", "mean_fail_cal", "med_gtm_cal", "mean_gtm_cal",
              "med_run_cal", "mean_run_cal"]]
    op = df[["stratum", "med_fail_op", "mean_fail_op", "med_gtm_op", "mean_gtm_op",
             "med_run_op", "mean_run_op"]]
    with pd.option_context("display.width", 220):
        print("== counts ==\n", counts.to_string(index=False))
        print("\n== calendar days ==\n", cal.to_string(index=False))
        print("\n== operating days ==\n", op.to_string(index=False))
    print("\nwrote", tables / "vt_stratum_census.csv")


if __name__ == "__main__":
    main()
