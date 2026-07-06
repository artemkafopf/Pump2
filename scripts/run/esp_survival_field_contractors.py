"""Per-contractor w₁ at field level (no sour/non-sour split).

Uses field-level fitted shapes (β₁, β₂) from field_mixture.py and
estimates w₁, η₁, η₂ per contractor within each field via EM with
fixed shapes.  Mirrors Phase 4 but uses pooled field models.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.workflows.esp_survival.data import load_failures_df
from analysis.workflows.esp_survival.phase2_mixture import load_models
from analysis.models.survival.weibull_em import fit_latent_weibull_em

MIN_CONTRACTOR_FAILURES = 10


def _load_field_models():
    root = RESULTS_ROOT / "esp_survival_field_mixture"
    for d in sorted(root.glob("????-??-??"), reverse=True):
        p = d / "models" / "phase2_models.json"
        if p.exists():
            models, gb1, gb2, two_stage = load_models(p)
            print(f"[Cache] Loaded field models from {d.name}")
            return models, gb1, gb2
    raise FileNotFoundError("No field_mixture models cache found — run esp_survival_field.py first")


def _logrank_p(g: pd.DataFrame) -> float:
    durations, events, groups = [], [], []
    for contractor, sub in g.groupby("contractor"):
        if int(sub["event"].sum()) < MIN_CONTRACTOR_FAILURES:
            continue
        durations.extend(sub["tte"].tolist())
        events.extend(sub["event"].tolist())
        groups.extend([contractor] * len(sub))
    if len(set(groups)) < 2:
        return float("nan")
    return float(multivariate_logrank_test(durations, groups, events).p_value)


def run(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    models, gb1, gb2, = _load_field_models()

    # Focus on fields where contractor variation is likely
    focus_fields = [f for f in ["Ya", "Vt", "Az", "Ic", "Za"]
                    if f in df["field_clean"].unique() and f in models]

    rows = []
    for field in focus_fields:
        g = df[df["field_clean"] == field].copy()
        base_model = models[field]
        fix_b1 = float(base_model.component_1.beta)
        fix_b2 = float(base_model.component_2.beta)

        lr_p = _logrank_p(g)

        contractors = sorted(
            c for c, sub in g.groupby("contractor")
            if int(sub["event"].sum()) >= MIN_CONTRACTOR_FAILURES
        )
        if not contractors:
            continue

        w1_by_contr: dict[str, float] = {}
        median_tte: dict[str, float] = {}

        for contractor in contractors:
            sub = g[g["contractor"] == contractor]
            dur = sub["tte"].to_numpy(dtype=float)
            ev = sub["event"].to_numpy(dtype=int)
            n_fail = int(ev.sum())

            kmf = KaplanMeierFitter()
            kmf.fit(dur, ev)
            med = kmf.median_survival_time_
            median_tte[contractor] = float(med) if med not in (None, np.inf) else float("nan")

            try:
                res = fit_latent_weibull_em(
                    dur, ev,
                    initial_model=base_model,
                    num_starts=6,
                    max_iter=200,
                    fix_beta1=fix_b1,
                    fix_beta2=fix_b2,
                )
                w1_by_contr[contractor] = round(float(res.model.weight_1), 4)
            except Exception as exc:
                print(f"  {field}/{contractor}: EM failed — {exc}")

        w1_range = (max(w1_by_contr.values()) - min(w1_by_contr.values())
                    if len(w1_by_contr) > 1 else float("nan"))

        decision = "promote_to_stratum" if (not np.isnan(w1_range) and w1_range > 0.10) \
                   else "standard_cox_covariate"

        sour_frac = float(g.loc[g["event"] == 1, "h2s_class"].eq("sour").mean())

        row = {
            "field": field,
            "n_failures": int(g["event"].sum()),
            "sour_failure_frac": round(sour_frac, 3),
            "field_beta1": round(fix_b1, 4),
            "field_beta2": round(fix_b2, 4),
            "field_w1": round(float(base_model.weight_1), 4),
            "logrank_p": round(lr_p, 4) if not np.isnan(lr_p) else None,
            "w1_range": round(w1_range, 4) if not np.isnan(w1_range) else None,
            "decision": decision,
        }
        for c in contractors:
            safe = c.replace(" ", "_")
            row[f"median_tte_{safe}"] = round(median_tte.get(c, float("nan")), 1)
            row[f"w1_{safe}"] = w1_by_contr.get(c)

        rows.append(row)

        w1_str = "  ".join(f"{c}={w1_by_contr.get(c, 'n/a')}" for c in contractors)
        print(
            f"  {field}: logrank_p={lr_p:.4f}  w1_range={w1_range:.3f}  [{w1_str}]  "
            f"→ {decision}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "field_contractor_w1.csv", index=False, encoding="utf-8-sig")
    return result_df


if __name__ == "__main__":
    t0 = time.monotonic()
    print("\n" + "=" * 70)
    print("ESP SURVIVAL — FIELD-LEVEL CONTRACTOR w₁ ANALYSIS")
    print("=" * 70)

    df = load_failures_df()
    out = results_dir("esp_survival_field_mixture")
    result = run(df, out / "tables")

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {time.monotonic() - t0:.0f}s")
    print("=" * 70)
