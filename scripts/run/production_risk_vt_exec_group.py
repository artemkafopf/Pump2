"""Does «Группа исполнения» (Н2 vs Н3) buy anything on Vt — and especially in sour?

Н3 is the corrosion-resistant execution, so the expectation is that it earns its
keep on sour runs.  The obstacle is confounding by indication: Н3 is *assigned*
to the wells thought to need it, so a naive Н3 coefficient measures the wells,
not the pump.  On Vt the entanglement is with contractor —

              sour        H2 / H3
    Борец                 29 / 25     <- the only balanced contrast
    Шлюмберже              7 / 47
    прочие                73 /  0     <- no contrast at all

— and contractor is itself a large effect (HR 0.52 / 0.67 vs прочие).  So the
fleet-wide Н3 coefficient is reported for completeness only; the Борец-sour cell
is the one that can answer the question, and it is small.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402

AS_OF = "2026-07-16"
AGE_BANDS = [(0, 30), (30, 90), (90, 180), (180, 300), (300, 450), (450, 600), (600, 900)]


def build() -> pd.DataFrame:
    pop = P.build(AS_OF)
    vt = P.attach_equipment(pop[pop["field"] == "Vt"].copy())
    vt["install_year"] = pd.to_datetime(vt["install"]).dt.year
    vt["vintage"] = vt["install_year"] - 2020
    vt["h3"] = (vt["pump_exec_group"] == "H3").astype(int)
    vt["sour"] = (vt["h2s_class"] == "sour").astype(int)
    vt["h3_x_sour"] = vt["h3"] * vt["sour"]
    vt["ctr_brt"] = (vt["contractor_group"] == "brt").astype(int)
    vt["ctr_slb"] = (vt["contractor_group"] == "slb").astype(int)
    vt["lch"] = vt["pump_exec_lch"].fillna(False).astype(int)
    return vt


def fit(df: pd.DataFrame, covs: list[str]) -> CoxPHFitter | None:
    sub = df[covs + ["tte", "event", "code"]].dropna()
    # A covariate with no contrast, or a cell with too few events, cannot be fit.
    if sub["event"].sum() < 10 or any(sub[c].nunique() < 2 for c in covs):
        return None
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col="tte", event_col="event", cluster_col="code")
    except Exception:
        return None
    return cph


def row(name: str, df: pd.DataFrame, covs: list[str], target: str) -> dict:
    model = fit(df, covs)
    base = {"model": name, "runs": len(df), "events": int(df["event"].sum()), "target": target}
    if model is None or target not in model.summary.index:
        return {**base, "HR": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                "note": "not identifiable"}
    s = model.summary
    return {
        **base,
        "HR": round(float(s.loc[target, "exp(coef)"]), 3),
        "ci_low": round(float(s.loc[target, "exp(coef) lower 95%"]), 3),
        "ci_high": round(float(s.loc[target, "exp(coef) upper 95%"]), 3),
        "p": round(float(s.loc[target, "p"]), 4),
        "note": "",
    }


def hazard_bands(g: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, b in AGE_BANDS:
        exposure = np.clip(np.minimum(g["tte"], b) - a, 0, None).sum() / 30.4
        events = ((g["tte"] >= a) & (g["tte"] < b) & (g["event"] == 1)).sum()
        out[f"{a}-{b}"] = round(events / exposure, 4) if exposure > 2 else np.nan
    return out


def main() -> None:
    out = results_dir("production_risk_vt_exec_group")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    vt = build()
    coverage = {
        "vt_runs": len(vt),
        "exec_group_joined": int(vt["pump_exec_group"].notna().sum()),
        "join_rate": round(float(vt["pump_exec_group"].notna().mean()), 3),
        "h2_or_h3": int(vt["pump_exec_group"].isin(["H2", "H3"]).sum()),
        "dropped_other_vocab": int(
            vt["pump_exec_group"].notna().sum() - vt["pump_exec_group"].isin(["H2", "H3"]).sum()
        ),
    }
    pd.Series(coverage).to_csv(tables / "coverage.csv", encoding="utf-8-sig")

    df = vt[vt["pump_exec_group"].isin(["H2", "H3"])].copy()

    assign = pd.crosstab(df["pump_exec_group"], [df["h2s_class"], df["contractor_group"]])
    assign.to_csv(tables / "assignment.csv", encoding="utf-8-sig")

    sour = df[df["sour"] == 1]
    nonsour = df[df["sour"] == 0]
    rows = [
        row("Vt all, adj contractor+vintage", df, ["h3", "sour", "ctr_brt", "ctr_slb", "vintage"], "h3"),
        row("Vt all, + H3xsour interaction", df,
            ["h3", "sour", "h3_x_sour", "ctr_brt", "ctr_slb", "vintage"], "h3_x_sour"),
        row("sour only, adj contractor+vintage", sour, ["h3", "ctr_brt", "ctr_slb", "vintage"], "h3"),
        row("nonsour only, adj contractor+vintage", nonsour, ["h3", "ctr_brt", "ctr_slb", "vintage"], "h3"),
        row("sour & Борец (balanced cell)", sour[sour["ctr_brt"] == 1], ["h3", "vintage"], "h3"),
        row("sour & Шлюмберже", sour[sour["ctr_slb"] == 1], ["h3", "vintage"], "h3"),
        row("nonsour & Борец", nonsour[nonsour["ctr_brt"] == 1], ["h3", "vintage"], "h3"),
        row("nonsour & Шлюмберже", nonsour[nonsour["ctr_slb"] == 1], ["h3", "vintage"], "h3"),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(tables / "h3_effects.csv", index=False, encoding="utf-8-sig")

    band_rows = []
    for (s, grp), g in df.groupby(["h2s_class", "pump_exec_group"]):
        band_rows.append({"h2s": s, "exec": grp, "runs": len(g),
                          "events": int(g["event"].sum()), **hazard_bands(g)})
    for (s, grp), g in df[df["ctr_brt"] == 1].groupby(["h2s_class", "pump_exec_group"]):
        band_rows.append({"h2s": f"{s} (Борец)", "exec": grp, "runs": len(g),
                          "events": int(g["event"].sum()), **hazard_bands(g)})
    bands = pd.DataFrame(band_rows)
    bands.to_csv(tables / "hazard_bands.csv", index=False, encoding="utf-8-sig")

    print("=== coverage ===")
    print(pd.Series(coverage).to_string())
    print("\n=== assignment: exec x (sour, contractor) ===")
    print(assign.to_string())
    print("\n=== H3 effect (HR<1 = H3 better) ===")
    print(res.to_string(index=False))
    print("\n=== empirical hazard bands ===")
    print(bands.to_string(index=False))
    print(f"\nwrote {tables}")


if __name__ == "__main__":
    main()
