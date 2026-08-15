"""Vt survival models: verify the shipped registry against the censored KM, then
sweep the full Weibull grid per stratum and shortlist replacement candidates.

Strata: Vt splits sour / nonsour (never pooled — 3x life gap AND different shape),
each further by contractor (brt / slb / oth), plus the two pooled rows. For every
stratum with enough events:

  1. resolve the shipped registry model (esp_models.csv via StrataModel.resolve)
     and score it against the right-censored KM — max|dS| plus RMST control;
  2. refit the grid  clock in {calendar, op-days} x cut in {0,3,7,30} x
     {k1, k2 constrained, k2 free}  (weibull_grid.sweep — same left-truncated
     likelihood as the Мирнинский grid, so numbers are comparable across fields);
  3. shortlist candidates: k1 always eligible, k2 only when the LR test earns its
     3 extra parameters (p<0.05) AND no honesty flag (bound-riding / degenerate w1).

Estimand: cause-specific — ГТМ/ППР pulls and running pumps are censored.
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
from analysis.workflows.production_risk import weibull_grid as G  # noqa: E402
from analysis.workflows.production_risk.survival import StrataModel  # noqa: E402

AS_OF = "2026-07-16"
CUTS = (0.0, 3.0, 7.0, 30.0)
# TWO model families, per the agreed scheme (docs/notes/production_risk_time_scales_proposal.md):
#   cal = t_cal — elapsed install→pull, always available, never imputed
#   op  = t_mix — measured op-days where telemetry supports it, else t_cal × Кэкспл
# «Наработка»/ННО is NOT a fit clock: it is absent for every running pump (so it can
# only time events, never censorings) and it is calendar-like anyway (0.95-0.99 of
# t_cal vs 0.85-0.89 for real op-time). It stays a reporting figure.
CLOCKS = (("cal", "t_cal"), ("op", "t_mix"))
KM_BAR = 0.05          # max|dS| above this = registry model out of line with KM
MIN_EVENTS = 15        # below this a 5-parameter grid is noise, stratum is skipped
CONTRACTORS = ("brt", "slb", "oth")


def strata_frames(pop: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """(h2s, ctr) -> stratum frame; ctr='Pooled' is the sour/nonsour aggregate."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    vt = pop[pop["field"] == "Vt"]
    for h2s in ("sour", "nonsour"):
        g = vt[vt["h2s_class"] == h2s]
        out[(h2s, "Pooled")] = g
        for ctr in CONTRACTORS:
            out[(h2s, ctr)] = g[g["contractor_group"] == ctr]
    return out


def verify_registry(reg: StrataModel, name: str, h2s: str, ctr: str,
                    frame: pd.DataFrame) -> dict:
    params, matched = reg.resolve("Vt", h2s, ctr)
    row = {"stratum": name, "registry_key": matched,
           "n": int(len(frame)), "events": int(frame["event"].sum()),
           **{k: round(params[k], 3) for k in ("w1", "beta1", "eta1", "beta2", "eta2")}}
    for clock, col in CLOCKS:
        t = frame[col].to_numpy(float)
        ev = frame["event"].to_numpy(int)
        row[f"ks_{clock}"] = round(G.max_ks(t, ev, params, 0.0), 3)
        row[f"rmst_km_{clock}"] = round(G.rmst_km(t, ev), 1)
    row["rmst_model"] = round(G.rmst_model(params), 1)
    # The registry was fit on ННО, which no longer exists as a fit clock. t_cal is its
    # closest honest comparator (ННО is 0.95-0.99 of t_cal); scoring it on the op clock
    # would measure the ~15% clock gap rather than the model.
    row["verdict"] = "OK" if row["ks_cal"] <= KM_BAR else "OUT_OF_LINE"
    return row


def restarts_for(events: int) -> tuple[int, int]:
    """Fewer optimizer restarts on thin strata — the surface is simpler there."""
    if events >= 100:
        return 20, 50
    if events >= 40:
        return 15, 35
    return 10, 25


def shortlist(df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    eligible = df[
        (df["kind"] == "k1")
        | ((df["p_lr"] < 0.05) & (df["flags"].fillna("") == ""))
    ].copy()
    eligible = eligible.sort_values("ks")
    return eligible.head(top)


def main() -> None:
    out = results_dir("production_risk_vt_weibull_grid")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pop = P.build(AS_OF)
    vt = pop[pop["field"] == "Vt"].copy()
    vt = P.add_time_scales(vt, AS_OF)
    vt = O.measure(vt, as_of=pd.Timestamp(AS_OF))
    print(f"Vt population as of {AS_OF}: {len(vt)} runs, {int(vt['event'].sum())} "
          f"failures (ГТМ+live censored)")
    print(vt.groupby(["h2s_class", "contractor_group"])["event"]
            .agg(runs="size", events="sum").to_string(), "\n")

    reg = StrataModel()
    frames = strata_frames(vt)

    verif_rows, grid_rows, cand_rows, skipped = [], [], [], []
    for (h2s, ctr), frame in frames.items():
        name = f"Vt_{h2s}_{ctr}"
        events = int(frame["event"].sum())
        if events < MIN_EVENTS:
            skipped.append((name, len(frame), events))
            continue

        verif_rows.append(verify_registry(reg, name, h2s, ctr, frame))

        k1_r, k2_r = restarts_for(events)
        ev = frame["event"].to_numpy(int)
        stratum_rows = []
        for clock, col in CLOCKS:
            t = frame[col].to_numpy(float)
            stratum_rows += G.sweep(t, ev, clock, cuts=CUTS,
                                    k1_restarts=k1_r, k2_restarts=k2_r)
        for r in stratum_rows:
            r["stratum"] = name
        grid_rows += stratum_rows

        sdf = G.to_frame(stratum_rows)
        best = shortlist(sdf)
        best.insert(1, "rank", range(1, len(best) + 1))
        cand_rows.append(best)
        print(f"{name}: {events} events — best {best.iloc[0]['clock']}/"
              f"c{int(best.iloc[0]['cut'])}/{best.iloc[0]['kind']}"
              f"({best.iloc[0]['regime']}) ks={best.iloc[0]['ks']:.3f}")

    verif = pd.DataFrame(verif_rows)
    grid = G.to_frame(grid_rows)
    cands = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()
    verif.to_csv(tables / "vt_registry_vs_km.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(tables / "vt_weibull_grid.csv", index=False, encoding="utf-8-sig")
    cands.to_csv(tables / "vt_candidates.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 260, "display.max_rows", 400):
        print("\n== Registry vs censored KM ==")
        print(verif.to_string(index=False))
        print("\n== Candidates (top 3 per stratum) ==")
        if not cands.empty:
            print(cands.to_string(index=False))
    for name, n, events in skipped:
        print(f"skipped {name}: {n} runs / {events} events < {MIN_EVENTS}")
    print("\nwrote", tables)


if __name__ == "__main__":
    main()
