"""Does «Месторождение» or «УН» carry the Vt-area failure hazard?

The shipped model keys strata on the well-code prefix, which is the УН (operating
unit).  «Месторождение» — the deposit — is a separate axis it never reads, and the
two are crossed in the Vt area:

                        Большетирское НМ   Ичёдинское НМ
    Верхнетирский УН           VT_               VT_
    Западно-Ярактинский УН      -                IC_      (modelled as field "Ic")
    Большетирский УН           BT_                -       (routed to Global_Pooled)

so both effects are identifiable.  This fits the crossed design and reports which
axis survives adjustment for contractor, vintage and sour.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import crosswalk  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402

AS_OF = "2026-07-16"
DEPOSITS = ("Большетирское НМ", "Ичёдинское НМ")
AGE_BANDS = [(0, 30), (30, 90), (90, 180), (180, 300), (300, 450), (450, 600), (600, 900), (900, 1500)]


def build_frame() -> pd.DataFrame:
    pop = P.build(AS_OF)
    pop["deposit"] = pop["code"].map(crosswalk.well_deposit_map())
    pop["un"] = pop["code"].map(crosswalk.well_un_map())
    pop = pop[pop["deposit"].isin(DEPOSITS)].copy()
    pop["install_year"] = pd.to_datetime(pop["install"]).dt.year

    pop["is_ichedinskoe"] = (pop["deposit"] == "Ичёдинское НМ").astype(int)
    pop["is_un_verkhnetirsky"] = (pop["un"] == "Верхнетирский УН").astype(int)
    pop["is_un_zapyarakta"] = (pop["un"] == "Западно-Ярактинский УН").astype(int)
    pop["sour"] = (pop["h2s_class"] == "sour").astype(int)
    pop["ctr_brt"] = (pop["contractor_group"] == "brt").astype(int)
    pop["ctr_slb"] = (pop["contractor_group"] == "slb").astype(int)
    pop["vintage"] = pop["install_year"] - 2020
    return pop.reset_index(drop=True)


CONTROLS = ["sour", "ctr_brt", "ctr_slb", "vintage"]
FIT_KW = dict(duration_col="tte", event_col="event", cluster_col="code")


def fit(df: pd.DataFrame, covs: list[str]) -> CoxPHFitter:
    cph = CoxPHFitter()
    cph.fit(df[covs + ["tte", "event", "code"]], **FIT_KW)
    return cph


def contrast(df: pd.DataFrame, covariate: str, covs: list[str]) -> dict:
    """One cluster-robust Wald contrast inside a subset where the other axis is held fixed.

    The full-design Cox cannot answer this cleanly: its УН reference cell
    (Большетирский УН) holds 4 events, so both УН coefficients come back with
    uselessly wide CIs.  A likelihood-ratio test is not an option either — the
    partial likelihood ignores the within-well correlation the cluster-robust SEs
    exist to absorb, so LR p-values here are anti-conservative.
    """
    model = fit(df, covs)
    s = model.summary
    return {
        "contrast": covariate,
        "runs": len(df),
        "events": int(df["event"].sum()),
        "HR": round(float(s.loc[covariate, "exp(coef)"]), 3),
        "ci_low": round(float(s.loc[covariate, "exp(coef) lower 95%"]), 3),
        "ci_high": round(float(s.loc[covariate, "exp(coef) upper 95%"]), 3),
        "p": float(s.loc[covariate, "p"]),
    }


def hazard_bands(g: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, b in AGE_BANDS:
        exposure = np.clip(np.minimum(g["tte"], b) - a, 0, None).sum() / 30.4
        events = ((g["tte"] >= a) & (g["tte"] < b) & (g["event"] == 1)).sum()
        out[f"{a}-{b}"] = round(events / exposure, 4) if exposure > 2 else np.nan
    return out


def main() -> None:
    out = results_dir("production_risk_vt_deposit_test")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = build_frame()

    cells = (
        df.groupby(["deposit", "un", "h2s_class"])
        .agg(runs=("event", "size"), events=("event", "sum"), wells=("code", "nunique"))
        .reset_index()
    )
    cells.to_csv(tables / "design_cells.csv", index=False, encoding="utf-8-sig")

    m_dep = fit(df, ["is_ichedinskoe"] + CONTROLS)
    m_un = fit(df, ["is_un_verkhnetirsky", "is_un_zapyarakta"] + CONTROLS)
    m_both = fit(df, ["is_ichedinskoe", "is_un_verkhnetirsky", "is_un_zapyarakta"] + CONTROLS)

    rows = []
    for name, model in [("deposit_only", m_dep), ("un_only", m_un), ("deposit_plus_un", m_both)]:
        s = model.summary
        for cov in s.index:
            rows.append(
                {
                    "model": name,
                    "covariate": cov,
                    "HR": round(float(s.loc[cov, "exp(coef)"]), 3),
                    "ci_low": round(float(s.loc[cov, "exp(coef) lower 95%"]), 3),
                    "ci_high": round(float(s.loc[cov, "exp(coef) upper 95%"]), 3),
                    "p": float(s.loc[cov, "p"]),
                }
            )
    coeffs = pd.DataFrame(rows)
    coeffs.to_csv(tables / "cox_coefficients.csv", index=False, encoding="utf-8-sig")

    # Each contrast holds the other axis fixed, so it is read off a populated cell
    # rather than the 4-event Большетирский УН reference.
    ich = df[df["deposit"] == "Ичёдинское НМ"]
    vt_nonsour = df[(df["un"] == "Верхнетирский УН") & (df["h2s_class"] == "nonsour")]
    gates = pd.DataFrame(
        [
            {"holding": "deposit = Ичёдинское", "axis_tested": "УН",
             **contrast(ich, "is_un_zapyarakta", ["is_un_zapyarakta", "ctr_brt", "ctr_slb", "vintage"])},
            {"holding": "УН = Верхнетирский, nonsour", "axis_tested": "deposit",
             **contrast(vt_nonsour, "is_ichedinskoe", ["is_ichedinskoe", "ctr_brt", "ctr_slb", "vintage"])},
        ]
    )
    gates.to_csv(tables / "pairwise_contrasts.csv", index=False, encoding="utf-8-sig")

    ph = proportional_hazard_test(m_both, df[["is_ichedinskoe", "is_un_verkhnetirsky",
                                              "is_un_zapyarakta", *CONTROLS,
                                              "tte", "event", "code"]],
                                  time_transform="rank")
    ph.summary.reset_index().to_csv(tables / "ph_check.csv", index=False, encoding="utf-8-sig")

    band_rows = []
    for (dep, sour), g in df.groupby(["deposit", "h2s_class"]):
        for un, gg in g.groupby("un"):
            if len(gg) < 20:
                continue
            band_rows.append(
                {"deposit": dep, "h2s": sour, "un": un, "runs": len(gg),
                 "events": int(gg["event"].sum()), **hazard_bands(gg)}
            )
    pd.DataFrame(band_rows).to_csv(tables / "hazard_bands.csv", index=False, encoding="utf-8-sig")

    print(f"population: {len(df)} runs, {int(df['event'].sum())} events")
    print("\n=== design cells ===")
    print(cells.to_string(index=False))
    print("\n=== Cox: deposit + УН, adjusted ===")
    print(coeffs[coeffs.model == "deposit_plus_un"].to_string(index=False))
    print("\n=== pairwise contrasts (other axis held fixed, cluster-robust) ===")
    print(gates.to_string(index=False))
    print("\n=== PH check (p<0.05 = PH violated) ===")
    print(ph.summary[["test_statistic", "p"]].round(4).to_string())
    print("\n=== empirical hazard bands ===")
    print(pd.DataFrame(band_rows).to_string(index=False))
    print(f"\nwrote {tables}")


if __name__ == "__main__":
    main()
