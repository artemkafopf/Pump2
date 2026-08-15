"""Write a candidate bundle with the accepted Vt survival rows.

Ships ONLY rows that beat the shipped registry on a **servable** fit — meaning a
`c0` fit on the `cal` clock (`t_cal`):

* `c > 0` fits are conditional on surviving to day c.  The 5-column registry schema
  cannot express "and the first c days are missing", so a c3/c7/c30 winner is not
  shippable even when it scores better.
* `op`-clock winners are held back: serving still ages pumps on ННО
  (`crosswalk` -> `age_op`), which is 0.95-0.99 of `t_cal` but only ~0.86 of `t_mix`.
  Shipping an op-clock row without moving the serving path introduces ~14%
  train/serve skew — exactly the pairing bug `clock_fix_plan.md` documents.

Accepted (gain = registry KS - new KS on t_cal, threshold 0.005):

    Vt_sour_Pooled   0.074 -> 0.050   (+0.024)
    Vt_sour_slb      0.134 -> 0.079   (+0.055)
    Vt_sour_oth      0.143 -> 0.080   (+0.063)

All three come out beta ~= 1.0-1.09: **sour Vt is memoryless**, so pump age carries
no risk information there and a single-parameter exponential is the honest model.

`Vt_nonsour_oth` is DELETED rather than refitted: 25 events, no fit in the grid gets
below KS 0.246, and its resolution falls back to `Vt_nonsour_Pooled` (KS 0.041, the
one row that passes as shipped).  The remaining Vt rows are left untouched — the
refit did not beat them.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.workflows.production_risk import config as C  # noqa: E402
from analysis.workflows.production_risk import weibull_grid as G  # noqa: E402

NEW_BUNDLE = "2026-07-20-vt-refit"
GRID_TABLES = REPO_ROOT / "results" / "production_risk_vt_weibull_grid" / "2026-07-20" / "tables"
MIN_GAIN = 0.005

# stratum -> (clock, cut, kind, regime) picked from vt_candidates/vt_weibull_grid
ACCEPT = {
    "Vt_sour_Pooled": ("cal", 0.0, "k1", "-"),
    "Vt_sour_slb": ("cal", 0.0, "k1", "-"),
    "Vt_sour_oth": ("cal", 0.0, "k1", "-"),
}
DELETE = ["Vt_nonsour_oth"]


def quantile(params: dict[str, float], p: float) -> float:
    """B{p} of the fitted mixture, by grid inversion (no closed form for k2)."""
    grid = np.arange(0.5, 40000, 0.5)
    s = G.surv_params(grid, params)
    hit = np.nonzero(s <= 1.0 - p)[0]
    return round(float(grid[hit[0]]), 1) if hit.size else float("nan")


def main() -> None:
    grid = pd.read_csv(GRID_TABLES / "vt_weibull_grid.csv", encoding="utf-8-sig")
    verif = pd.read_csv(GRID_TABLES / "vt_registry_vs_km.csv", encoding="utf-8-sig").set_index("stratum")

    src_dir = C.bundle_dir(C.BUNDLE_DATE)
    dst_dir = C.bundle_dir(NEW_BUNDLE)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"bundle {C.BUNDLE_DATE} -> {NEW_BUNDLE}")

    models = pd.read_csv(dst_dir / "esp_models.csv", encoding="utf-8-sig")
    changed = []

    for stratum, (clock, cut, kind, regime) in ACCEPT.items():
        m = grid[(grid.stratum == stratum) & (grid.clock == clock) & (grid.cut == cut)
                 & (grid["kind"] == kind) & (grid.regime == regime)]
        if m.empty:
            raise SystemExit(f"{stratum}: no grid row for {clock}/c{cut:.0f}/{kind}/{regime}")
        r = m.iloc[0]
        gain = float(verif.loc[stratum, "ks_cal"]) - float(r["ks"])
        if gain < MIN_GAIN:
            raise SystemExit(f"{stratum}: gain {gain:.3f} below {MIN_GAIN} — do not ship")

        params = {k: float(r[k]) for k in ("w1", "beta1", "eta1", "beta2", "eta2")}
        idx = models.index[models["stratum"] == stratum]
        if idx.empty:
            raise SystemExit(f"{stratum}: not present in esp_models.csv")
        i = idx[0]
        before = float(models.at[i, "b50"])
        for k, v in params.items():
            models.at[i, k] = v
        models.at[i, "model_kind"] = "k1_cal_refit" if kind == "k1" else f"k2_{regime}_cal_refit"
        models.at[i, "b20"] = quantile(params, 0.20)
        models.at[i, "b50"] = quantile(params, 0.50)
        models.at[i, "b80"] = quantile(params, 0.80)
        models.at[i, "n_runs"] = int(r["n"])
        models.at[i, "n_failures"] = int(r["events"])
        models.at[i, "n_censored"] = int(r["n"]) - int(r["events"])
        # The clock column must name the ACTUAL column, not an intent — the shipped
        # "ttf_mix_svod_plus_big_nno" was aspirational (clock_fix_plan.md:116).
        models.at[i, "clock"] = "t_cal"
        models.at[i, "uptime_factor"] = 1.0
        changed.append((stratum, before, float(models.at[i, "b50"]), gain))
        print(f"  {stratum:20s} KS {verif.loc[stratum,'ks_cal']:.3f} -> {r['ks']:.3f} "
              f"(+{gain:.3f})  b50 {before:.0f} -> {models.at[i,'b50']:.0f}  clock=t_cal")

    for stratum in DELETE:
        n = len(models)
        models = models[models["stratum"] != stratum]
        print(f"  {stratum:20s} DELETED ({n - len(models)} row) -> falls back to Vt_nonsour_Pooled")

    models.to_csv(dst_dir / "esp_models.csv", index=False, encoding="utf-8-sig")

    manifest_path = dst_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["derived_from"] = C.BUNDLE_DATE
        manifest["vt_refit"] = {
            "accepted": {s: f"{c}/c{int(cut)}/{k}" for s, (c, cut, k, _) in ACCEPT.items()},
            "deleted": DELETE,
            "clock": "t_cal",
            "held_back_op_rows": ["Vt_nonsour_slb", "Vt_sour_brt"],
            "note": "op-clock winners held until the serving path ages pumps on t_mix",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {dst_dir / 'esp_models.csv'}  ({len(changed)} refit, {len(DELETE)} deleted)")
    print(f"run:  .venv/Scripts/python.exe scripts/run/production_risk.py --bundle-date {NEW_BUNDLE}")


if __name__ == "__main__":
    main()
