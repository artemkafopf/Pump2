"""Phase D — D5: fleet watchlist scorer (gated on the D4 verdict).

**If** D4 showed a material, interval-backed improvement of dynamics over age+stratum
(primary AUC-delta CI strictly above 0), this scores every currently-running pump at
its latest landmark and emits a ranked watchlist CSV — the operational payoff.

**If not** (honest null, the C5 lesson repeating one layer up), it skips the watchlist
and instead writes the telemetry-improvement recommendation: which channels / coverage
would most raise the ceiling, from D1's precursor atlas and the D0 coverage tables.

Outputs under ``results/phase_d_watchlist/<date>/``:
  tables/watchlist.csv                    — (if earned) ranked running-pump scores
  reports/telemetry_recommendation.md     — (if null) what to collect to move the needle
  logs/watchlist.txt

Run (needs phase_d_landmarks.py + phase_d_landmark_models.py first):
    python scripts/run/phase_d_watchlist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.data.landmark_features import (
    load_latest_landmark_frame, HORIZONS, GUARD_GAPS, DEFAULT_GUARD, WINDOWS,
)
from analysis.models.survival.landmark_cox import fit_landmark_cox, predict_window_risk
from analysis.models.survival.screening_cox import univariate_screen, correlation_prune

PRIMARY_H, PRIMARY_G = 60, DEFAULT_GUARD
_STEMS = ["kprod_mean", "kprod_slope", "qliq_slope", "load_mean", "load_slope",
          "load_std", "load_exc", "freq_mean", "freq_std", "freq_step", "freq_days55",
          "restarts", "longest_idle", "rintake_mean", "rintake_slope", "rzab_slope",
          "drawdown", "watercut_mean", "watercut_jump", "gasfactor_mean", "gasfactor_slope"]


def _latest_table(slug: str, name: str) -> Path | None:
    base = RESULTS_ROOT / slug
    if not base.exists():
        return None
    for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        p = d / "tables" / name
        if p.exists():
            return p
    return None


def _zscore(frame, cols, train_mask):
    d = frame.copy()
    stats = {}
    for c in cols:
        mu, sd = d.loc[train_mask, c].mean(), d.loc[train_mask, c].std()
        stats[c] = (mu, sd)
        d[c] = (d[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return d, stats


def _verdict_is_material() -> tuple[bool, dict]:
    p = _latest_table("phase_d_landmark_models", "delta_ci.csv")
    if p is None:
        return False, {"note": "no D4 delta_ci found"}
    delta = pd.read_csv(p)
    row = delta[(delta["H"] == PRIMARY_H) & (delta["g"] == PRIMARY_G)]
    if row.empty:
        return False, {"note": "primary H/g missing in delta_ci"}
    r = row.iloc[0]
    material = np.isfinite(r["ci_lo"]) and r["ci_lo"] > 0
    return bool(material), {"auc_delta": r["auc_delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}


def _write_watchlist(frame, features, out, log):
    dt = pd.to_datetime(frame["install_dt"], errors="coerce")
    train_mask = dt <= pd.Timestamp("2023-12-31")
    fz, _ = _zscore(frame, [c for c in features if c != "op_age_z"], train_mask)
    fz["op_age_z"] = (frame["op_age"] - frame.loc[train_mask, "op_age"].mean()) / \
        frame.loc[train_mask, "op_age"].std()
    fz["op_age"] = frame["op_age"]

    fit = fit_landmark_cox(fz, features)
    if not fit.success:
        log.append(f"[D5] watchlist fit failed ({fit.message}); no CSV emitted")
        return
    # currently-running = censored runs (event==0), scored at their latest landmark
    running = fz[fz["event_land"] == 0].sort_values("op_age").groupby("row_id").tail(1)
    u = PRIMARY_G + PRIMARY_H
    risk = predict_window_risk(fit, running, u)
    running = running.assign(risk_next_H=risk)

    # top drivers per pump: coef × z-feature (relative to fleet mean 0)
    coefs = {t.split("[")[0].strip(): c for t, c in
             zip(fit.cph.params_.index, fit.cph.params_.values)}
    drv_feats = [f for f in fit.features if f in fz.columns]

    def _drivers(row):
        contribs = {f: coefs.get(f, 0.0) * (row[f] if np.isfinite(row[f]) else 0.0)
                    for f in drv_feats}
        top = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:3]
        return "; ".join(f"{k}({v:+.2f})" for k, v in top if abs(v) > 1e-6)

    running["top_drivers"] = running.apply(_drivers, axis=1)
    running["data_fresh"] = (running["n_valid_op_days"] >= running["op_age"] * 0.5) \
        if "n_valid_op_days" in running.columns else True

    wl = running[["row_id", "well_key", "stratum_key", "op_age", "risk_next_H",
                  "top_drivers", "data_fresh"]].copy()
    wl = wl[np.isfinite(wl["risk_next_H"])].sort_values("risk_next_H", ascending=False)
    wl["risk_next_H"] = wl["risk_next_H"].round(4)
    wl.rename(columns={"op_age": "op_age_days",
                       "risk_next_H": f"risk_next_{PRIMARY_H}d"}, inplace=True)
    wl.to_csv(out / "tables" / "watchlist.csv", index=False, encoding="utf-8-sig")
    log.append(f"[D5] watchlist emitted: {len(wl)} running pumps scored; "
               f"top risk {wl.iloc[0][f'risk_next_{PRIMARY_H}d']:.3f} "
               f"(well {wl.iloc[0]['well_key']})")


def _write_recommendation(out, log, verdict):
    lines = ["# Phase D — telemetry-improvement recommendation (D5, honest-null branch)",
             "",
             "The dynamic layer did **not** beat age+stratum out of sample at the primary "
             f"H={PRIMARY_H}d / g={PRIMARY_G} capacity "
             f"(AUC delta {verdict.get('auc_delta')} "
             f"[{verdict.get('ci_lo')}, {verdict.get('ci_hi')}] — CI includes 0). "
             "Per the phase's honest-null clause the watchlist is not shipped; instead this "
             "records what would most raise the ceiling.",
             ""]

    smd_p = _latest_table("phase_d_precursors", "standardized_diffs.csv")
    if smd_p is not None:
        smd = pd.read_csv(smd_p)
        smd["abs"] = smd["smd"].abs()
        top = smd.sort_values("abs", ascending=False).head(8)
        lines += ["## Where the little signal that exists lives (D1 atlas)",
                  "The strongest matched pre-failure separations (still modest) were:",
                  ""]
        for _, r in top.iterrows():
            lines.append(f"- `{r['feature']}` — SMD {r['smd']:+.3f}")
        lines.append("")

    cov_p = _latest_table("phase_d_landmarks", "coverage.csv")
    if cov_p is not None:
        cov = pd.read_csv(cov_p).sort_values("pct_measured")
        low = cov[cov["pct_measured"] < 90].head(12)
        lines += ["## Coverage gaps capping the dynamic layer (D0)",
                  "Trailing-window features below 90% landmark coverage — better/denser "
                  "telemetry here is the prerequisite for a working early-warning model:",
                  ""]
        for _, r in low.iterrows():
            lines.append(f"- `{r['feature']}` — {r['pct_measured']}% of landmarks measured")
        lines.append("")

    lines += ["## Recommendation",
              "1. Densify the channels behind the strongest (if weak) precursors above — "
              "load variability, watercut, intake-pressure/drawdown, restart logging.",
              "2. Restart/idle events (`days_since_restart`) are the most under-recorded "
              "signal; capture explicit on/off events rather than inferring from qliq gaps.",
              "3. Re-evaluate the dynamic layer once a telemetry-complete daily feed exists "
              "or the 2024+ cohort matures (mirrors the Phase C θ recommendation).",
              ""]
    (out / "reports" / "telemetry_recommendation.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log.append(f"[D5] honest null -> telemetry recommendation written "
               f"({len(lines)} lines)")


def main() -> None:
    out = results_dir("phase_d_watchlist")
    logs = out / "logs"
    log: list[str] = []

    material, verdict = _verdict_is_material()
    log.append(f"[D5] D4 primary verdict material={material} ({verdict})")

    if material:
        frame = load_latest_landmark_frame()
        cands = [f"{s}_w{w}" for w in WINDOWS for s in _STEMS if f"{s}_w{w}" in frame.columns]
        dt = pd.to_datetime(frame["install_dt"], errors="coerce")
        train_mask = dt <= pd.Timestamp("2023-12-31")
        fz, _ = _zscore(frame, cands, train_mask)
        train = fz[train_mask]
        uni = univariate_screen(train, cands, duration_col="tte_land", event_col="event_land")
        survivors = uni.loc[uni["pass_screen"], "covariate"].tolist()
        kept, _ = correlation_prune(train, survivors,
                                    uni.set_index("covariate")["c_index"].to_dict())
        _write_watchlist(frame, ["op_age_z"] + kept, out, log)
    else:
        _write_recommendation(out, log, verdict)

    (logs / "watchlist.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"[D5] outputs -> {out}")


if __name__ == "__main__":
    main()
