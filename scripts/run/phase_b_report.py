"""Phase B B6 — Report (output slug: phase_b_report).

Assembles the one-markdown Phase B story from the published B1–B5 tables: mode mix
per stratum → what chemistry does / doesn't touch → what H₂S attacks → planning-
ready CIF tables.  Every table is stamped ``clock=ttf_mix``; the §0 decisions are
recorded in the header.  VBA implications are listed as deferred (no VBA changes
in Phase B).

The script reads the latest available date directory for each upstream slug, so it
can be run after the B1–B5 scripts on the same day (or a later day).

Run:
    python scripts/run/phase_b_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir, RESULTS_ROOT

UPSTREAM = ["phase_b_inventory", "phase_b_cif", "phase_b_cause_cox",
            "phase_b_vt_h2s_modes", "phase_b_cause_weibull"]


def _latest_tables(slug: str) -> Path | None:
    base = RESULTS_ROOT / slug
    if not base.exists():
        return None
    dates = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    for d in dates:
        if (d / "tables").exists():
            return d / "tables"
    return None


def _read(slug: str, name: str) -> pd.DataFrame:
    tdir = _latest_tables(slug)
    if tdir is None:
        return pd.DataFrame()
    p = tdir / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 60) -> str:
    if df.empty:
        return "_(no data)_\n"
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    df = df.head(max_rows)

    def _fmt(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return f"{head}\n{sep}\n{body}\n"


def main() -> None:
    out = results_dir("phase_b_report")
    rep = out / "reports"

    inv = _read("phase_b_inventory", "b1_inventory_field.csv")
    mode_mix = _read("phase_b_cif", "b2_mode_mix.csv")
    gray = _read("phase_b_cif", "b2_vt_gray_test.csv")
    b3 = _read("phase_b_cause_cox", "b3_cause_cox_coeffs.csv")
    b3_nkt = _read("phase_b_cause_cox", "b3_nkt_sensitivity.csv")
    b4_hr = _read("phase_b_vt_h2s_modes", "b4_vt_sour_hr.csv")
    b4_gamma = _read("phase_b_vt_h2s_modes", "b4_vt_gamma.csv")
    b4_cif = _read("phase_b_vt_h2s_modes", "b4_vt_cif_compare.csv")
    b5_q = _read("phase_b_cause_weibull", "b5_incidence_quantiles.csv")
    b5_recon = _read("phase_b_cause_weibull", "b5_reconciliation.csv")

    # group totals from the inventory
    totals = (inv.groupby("mode_group")["n_events"].sum().to_dict() if not inv.empty else {})

    # mode-mix at 365d for the well-populated strata
    mix365 = mode_mix.query("horizon == '365d'") if not mode_mix.empty else pd.DataFrame()
    mix_pivot = (mix365.pivot_table(index="stratum", columns="mode_group",
                                    values="incidence_share").round(3).reset_index()
                 if not mix365.empty else pd.DataFrame())

    gray90 = gray.query("time == 90.0") if not gray.empty else pd.DataFrame()
    b4_cif90 = b4_cif.query("time == 90.0") if not b4_cif.empty else pd.DataFrame()
    recon_flags = int(b5_recon["flag_gt_5pct"].sum()) if not b5_recon.empty else 0

    lines: list[str] = []
    A = lines.append
    A("# Phase B — Competing Risks: Cause-Specific Survival by Failed Node\n")
    A("**Date:** 2026-07-06  ·  **Clock:** `ttf_mix` (operating days) on every table.\n")
    A("## §0 decisions (frozen)\n")
    A("- **§0.1 mode-group mapping** (user-confirmed 2026-07-06): "
      "НКТ → hydraulic (with a with/without-НКТ sensitivity in B3); "
      "Газосепаратор+Диспергатор folded into hydraulic; ТМС → electro-thermal. "
      "Frozen in `analysis/data/failure_modes.py` and exported to "
      "`phase_b_inventory/.../b1_mode_group_map.csv`.\n")
    A("- **§0.2 sour handling: option (c)** — strata unchanged; `is_sour_flagged` "
      "(`h2s_proxy_mg_l > 10`, applied to every field) added as a covariate in all "
      "cause-specific Cox models. The stratum-redesign decision defers to Phase C/E.\n")
    A(f"- **Failure-mode totals** (zero unlabelled failures asserted): "
      f"hydraulic {totals.get('hydraulic','?')}, electro-thermal "
      f"{totals.get('electro-thermal','?')}, protector {totals.get('protector','?')}, "
      f"other {totals.get('other','?')}.\n")

    A("\n## 1. Mode mix per stratum (Aalen–Johansen CIF, not 1−KM)\n")
    A("Share of *incidence* by mode group at 365 operating days (from the CIF, so "
      "censoring-correct). Absolute risk uses the AJ CIF throughout — never "
      "`1−KM_cause`, which would overstate incidence given ~40% competing failures.\n")
    A(_md_table(mix_pivot))

    A("\n## 2. What chemistry does and doesn't touch (B3 cause-specific Cox)\n")
    A("One global-β stratified Cox per cause group (`strata=['stratum_key']`, "
      "cluster-robust by well, early-window covariates). HR per covariate × cause:\n")
    if not b3.empty:
        piv = b3.pivot(index="covariate", columns="cause_group", values="hr").round(3)
        pval = b3.pivot(index="covariate", columns="cause_group", values="p").round(4)
        piv = piv.reset_index()
        A(_md_table(piv))
        A("\np-values (same layout):\n")
        A(_md_table(pval.reset_index()))
    A("\n**Reading.** A covariate elevated in *all* modes roughly equally is a "
      "red flag for residual confounding (field / vintage), not a mode-specific "
      "etiology. `is_sour_flagged` sits above 1 across every mode → the sour signal "
      "is broad (consistent with H₂S attacking cable/motor as well as the pump), "
      "and `log_h2s_proxy` is collinear with it (both derive from `h2s_proxy`), so "
      "the continuous term is attenuated. Treat the paired H₂S terms jointly.\n")
    if not b3_nkt.empty:
        A("\n**НКТ sensitivity (hydraulic with vs without НКТ):**\n")
        A(_md_table(b3_nkt[b3_nkt["covariate"].isin(["log_h2s_proxy_mg_l", "is_sour_flagged",
                    "log_glf_mean_opdays"])], cols=["variant", "covariate", "hr",
                    "hr_ci_lo", "hr_ci_hi", "p", "n_events"]))

    A("\n## 3. What H₂S attacks in Vt (B4)\n")
    A("Cause-specific sour hazard ratio within Vt, per mode (cluster-robust by well):\n")
    A(_md_table(b4_hr, cols=["mode_group", "hr", "hr_ci_lo", "hr_ci_hi", "p", "n_events"]))
    A("\nTime-interaction γ (does the sour disadvantage grow with time?) via the "
      "corrected event-time episode split — CIs are wide (~80–124 events/mode) and "
      "reported honestly; **no** curve-regression p-values:\n")
    A(_md_table(b4_gamma, cols=["mode_group", "beta_sour", "gamma", "gamma_boot_ci_lo",
                                "gamma_boot_ci_hi", "hr_sour_at_90d", "hr_sour_at_365d",
                                "n_events"]))
    if not b4_cif90.empty:
        A("\nOperational CIF statement at 90 days (sour vs nonsour, Gray-style bootstrap):\n")
        A(_md_table(b4_cif90, cols=["mode_group", "cif_sour", "cif_nonsour",
                                    "diff_point", "diff_ci_lo", "diff_ci_hi",
                                    "boot_p_two_sided"]))
    if not gray90.empty:
        A("\n(Independent B2 CIF comparison, same direction:)\n")
        A(_md_table(gray90, cols=["mode_group", "cif_sour", "cif_nonsour", "diff_point",
                                  "diff_ci_lo", "diff_ci_hi", "boot_p_two_sided"]))

    A("\n## 4. Planning-ready incidence quantiles (B5, CIF-consistent)\n")
    A("Operating time to 10% / 25% cumulative incidence per mode per stratum, with "
      "well-bootstrap CIs. Note: per-mode parametric S_k(t) ≠ 1−CIF_k — the CIF "
      "combines cause hazards through the all-cause survival.\n")
    A(_md_table(b5_q, cols=["stratum", "mode_group", "incidence_pct", "t_days_point",
                            "t_ci_lo", "t_ci_hi", "reaches_incidence"], max_rows=80))
    A(f"\n**Reconciliation** (competing all-cause B50 vs pooled all-cause Weibull): "
      f"{recon_flags} stratum(a) exceed the 5% mismatch flag.\n")
    A(_md_table(b5_recon, cols=["stratum", "pooled_b50", "competing_b50",
                                "abs_pct_diff", "flag_gt_5pct"]))
    A("\n_Investigation of the flags_: the CIF partition identity "
      "`Σ_k CIF_k(t) = 1 − S_all(t)` holds analytically (unit-tested), so the gap "
      "is **not** a CIF-math error. The two large flags (Vt_sour ≈20%, Mc ≈24%) are "
      "exactly the strata where Phase A found the K=2 mixture wins by ΔAIC — a single "
      "Weibull per cause cannot reproduce the burn-in + wear-out shape, so the summed "
      "cause hazards diverge from a pooled single Weibull. The B50-for-planning number "
      "in those strata should come from the mixture stack, not this parametric layer; "
      "the CIF *shapes* here remain valid for repair-timing.\n")

    A("\n## 5. VBA implications (deferred — no VBA changes in Phase B)\n")
    A("- The mode-group map (`b1_mode_group_map.csv`) is the single source of truth; "
      "any VBA repair-planning layer must read it, not re-hardcode node lists.\n")
    A("- Pooled B50 misallocates repair resources where the mode mix differs by "
      "field (§1). A future VBA planning sheet should carry per-mode CIF quantiles "
      "(§4), not just the pooled B50.\n")
    A("- `is_sour_flagged` is a fitted covariate now, not a stratum; if Phase C/E "
      "promotes it to a stratum axis (option (b)), the VBA stratum keys change then.\n")

    A("\n## 6. Open questions / caveats\n")
    A("- `log_h2s_proxy` × `is_sour_flagged` collinearity attenuates the continuous "
      "H₂S term; a Phase C refit could drop one or ridge-penalise.\n")
    A("- H₂S proxy is ~40% pad-median — the per-mg/l magnitudes remain soft; the "
      "sour *flag* effect is the more trustworthy quantity.\n")
    A("- 'other' (n≈45) and the stray Cyrillic field labels (Гораздинское, ЯНГКМ) "
      "are Phase A label-hygiene leftovers; immaterial to the mode conclusions.\n")

    report_path = rep / "phase_b_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[B6] report written: {report_path}")
    print(f"[B6] outputs written to: {out}")


if __name__ == "__main__":
    main()
