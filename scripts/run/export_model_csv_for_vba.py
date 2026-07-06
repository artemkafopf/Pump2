"""Export fitted model parameters as a CSV for the Excel VBA model registry.

Collects the best available model for every VBA stratum key from the
Python pipeline result files, writes esp_models.csv to the results slug
`esp_survival_vba_models`.

The VBA ImportModelCSV() macro reads this file.

Stratum key format: {field}_{h2s_class}_{contractor_group}
  contractor_group: brt (Borec) | slb (Schlumberger) | oth (others) | Pooled

Priority (same as VBA LookupModel):
  1. Contractor-specific strata (Ya, Za, Vt)
  2. Field-level pooled strata  (Ic, Az, Mc, Da)
  3. Global fallback row
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from analysis.paths import results_dir, RESULTS_ROOT

# Maps Cyrillic contractor names (as they appear in pipeline stratum keys) to VBA codes.
# Keep this in sync with ContractorGroup() in mdlModelRegistry.bas.
_CTR_CODE = {
    "Борец": "brt",         # Борец
    "Шлюмберже": "slb",  # Шлюмберже
    "Other": "oth",
    "Pooled": "Pooled",
}

def _ctr_to_code(name: str) -> str:
    return _CTR_CODE.get(name, "oth")


def _latest_csv(slug: str, name: str) -> pd.DataFrame | None:
    root = RESULTS_ROOT / slug
    candidates = sorted(root.glob(f"????-??-??/**/{name}"), reverse=True)
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def main():
    rows: list[dict] = []

    def add(stratum, field, h2s, ctr_code, row: pd.Series, degenerate=None):
        degen = degenerate if degenerate is not None else bool(row.get("degenerate", False))
        rows.append({
            "stratum":          stratum,
            "field":            field,
            "h2s_class":        h2s,
            "contractor_group": ctr_code,
            "n_failures":       int(row.get("n_failures", 0)),
            "fit_mode":         row.get("fit_mode", "unknown"),
            "w1":               round(float(row["w1"]), 6),
            "beta1":            round(float(row["beta1"]), 6),
            "eta1":             round(float(row["eta1_days"]), 2),
            "beta2":            round(float(row["beta2"]), 6),
            "eta2":             round(float(row["eta2_days"]), 2),
            "degenerate":       degen,
        })

    # -- Contractor-level strata (from field_contractor_substrata run) ----------
    fc = _latest_csv("esp_survival_field_contractor", "field_contractor_phase2.csv")
    if fc is not None:
        for _, r in fc.iterrows():
            # stratum col is like "Ya_Борец", "Za_Шлюмберже", etc.
            parts = r["stratum"].split("_", 1)
            field    = parts[0]
            ctr_raw  = parts[1] if len(parts) > 1 else "Pooled"
            ctr_code = _ctr_to_code(ctr_raw)
            key      = f"{field}_nonsour_{ctr_code}"
            add(key, field, "nonsour", ctr_code, r)

    # -- Vt contractor-level strata (from vt_contractor_substrata run) ----------
    base = RESULTS_ROOT / "esp_survival_vt_contractor_substrata"
    for date_dir in sorted(base.glob("????-??-??"), reverse=True):
        for group_csv in date_dir.rglob("*phase2_params.csv"):
            df = pd.read_csv(group_csv)
            for _, r in df.iterrows():
                # stratum like "Vt_pooled_Борец", "Vt_sour_Шлюмберже", etc.
                parts = r["stratum"].split("_")
                if len(parts) < 3:
                    continue
                field    = parts[0]   # "Vt"
                h2s      = parts[1]   # "pooled" | "sour" | "nonsour"
                ctr_raw  = parts[2]   # Cyrillic or "Other"
                ctr_code = _ctr_to_code(ctr_raw)
                key      = f"{field}_{h2s}_{ctr_code}"
                add(key, field, h2s, ctr_code, r)
        break   # only most recent date

    # -- Field-level pooled strata (main phase2 run) ----------------------------
    ph2 = _latest_csv("esp_survival_phase2_mixture", "phase2_mixture_params.csv")
    if ph2 is not None:
        for _, r in ph2.iterrows():
            stratum = r["stratum"]   # e.g. "Ya_nonsour", "Vt_sour", "Mc_nonsour"
            parts   = stratum.split("_", 1)
            field   = parts[0]
            h2s     = parts[1] if len(parts) > 1 else "nonsour"
            key     = f"{stratum}_Pooled"
            add(key, field, h2s, "Pooled", r)

    # -- Global fallback --------------------------------------------------------
    rows.append({
        "stratum": "Global_Pooled", "field": "", "h2s_class": "", "contractor_group": "Pooled",
        "n_failures": 0, "fit_mode": "domain_default",
        "w1": 0.20, "beta1": 1.10, "eta1": 41.0, "beta2": 1.40, "eta2": 500.0,
        "degenerate": False,
    })

    out_dir = results_dir("esp_survival_vba_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "esp_models.csv"

    df_out = pd.DataFrame(rows).drop_duplicates(subset=["stratum"], keep="first")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Exported {len(df_out)} model rows to:\n  {out_path}")
    print(df_out[["stratum", "n_failures", "w1", "eta1", "eta2", "degenerate"]].to_string(index=False))


if __name__ == "__main__":
    main()
