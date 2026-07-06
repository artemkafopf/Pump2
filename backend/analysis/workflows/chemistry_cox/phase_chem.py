"""Chemistry Block Cox Analysis — Phase CHEM.

Adaptive extended Cox model for fluid chemistry, КВЧ, GLF, and water cut.
Block 1 of three planned Cox blocks (chemistry / operational / completion).

Steps:
  1. Univariate Cox screening (stratified, cluster-robust)
  2. Schoenfeld PH test → STANDARD (β·X) or EXTENDED (β·X + γ·X·log t)
  3. Spearman correlation screen (|ρ| > 0.65 → drop lower C-index covariate)
  4. Joint multivariate model with all surviving terms
  5. VIF check on standard covariates (drop VIF > 5, refit)
  6. Output tables + forest plots + chem_final_coeffs.csv (VBA handoff)

Run:
    cd backend && python -m analysis.workflows.chemistry_cox.phase_chem
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.chemistry_run_features import build_chemistry_df
from analysis.paths import results_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Convergence.*")


# ---------------------------------------------------------------------------
# Candidate covariates
# ---------------------------------------------------------------------------

# All candidates enter univariate screening. Log-transformed columns are primary.
CANDIDATES: list[str] = [
    "log_chloride_mg_l",
    "log_sulfate_mg_l",
    "log_calcium_mg_l",
    "log_bicarbonate_mg_l",
    "log_total_mineralization_g_l",
    "log_mechanical_impurities_mg_l",  # КВЧ
    "log_h2s_proxy_mg_l",
    "log_glf_mean",
    "log_ca_so4",
    "ph",
    "watercut_percent",
]

# Univariate p-value threshold for proceeding to PH test
P_UNIVARIATE = 0.10
# PH test p-value threshold (below → EXTENDED term required)
P_PH_EXTENDED = 0.05
# Spearman ρ threshold for dropping one covariate from a correlated pair
SPEARMAN_DROP_THR = 0.65
# VIF threshold for dropping a covariate from joint model
VIF_DROP_THR = 5.0
# Minimum events required per stratum to include stratum in stratified fit
MIN_STRATUM_EVENTS = 5
# Minimum total events for univariate screening
MIN_TOTAL_EVENTS = 15

DURATION_COL = "tte"
EVENT_COL = "event"


# ---------------------------------------------------------------------------
# Step 1 — Univariate Cox screening
# ---------------------------------------------------------------------------

def run_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """Fit one stratified Cox per candidate. Return summary DataFrame."""
    rows = []
    for col in CANDIDATES:
        if col not in df.columns:
            print(f"  [SKIP] {col} — not in DataFrame")
            continue

        sub = df[["well_key", DURATION_COL, EVENT_COL, "stratum_key", col]].dropna()
        n_events = int(sub[EVENT_COL].sum())
        if n_events < MIN_TOTAL_EVENTS:
            print(f"  [SKIP] {col} — only {n_events} events after dropna")
            continue

        # Drop strata with < MIN_STRATUM_EVENTS events
        event_counts = sub.groupby("stratum_key")[EVENT_COL].sum()
        valid_strata = event_counts[event_counts >= MIN_STRATUM_EVENTS].index
        sub = sub[sub["stratum_key"].isin(valid_strata)]
        n_events = int(sub[EVENT_COL].sum())
        if n_events < MIN_TOTAL_EVENTS:
            print(f"  [SKIP] {col} — only {n_events} events in valid strata")
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(
                sub,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                strata=["stratum_key"],
                cluster_col="well_key",
                formula=col,
                robust=True,
            )
            summ = cph.summary
            hr = float(np.exp(summ.loc[col, "coef"]))
            p = float(summ.loc[col, "p"])
            ci_lo = float(np.exp(summ.loc[col, "coef lower 95%"]))
            ci_hi = float(np.exp(summ.loc[col, "coef upper 95%"]))
            c_idx = float(cph.concordance_index_)
            rows.append(
                {
                    "covariate": col,
                    "n_runs": len(sub),
                    "n_events": n_events,
                    "HR": round(hr, 4),
                    "HR_lo": round(ci_lo, 4),
                    "HR_hi": round(ci_hi, 4),
                    "p": round(p, 6),
                    "c_index": round(c_idx, 4),
                    "pass_screen": p < P_UNIVARIATE,
                }
            )
            flag = "PASS" if p < P_UNIVARIATE else "fail"
            print(f"  [{flag}] {col}: HR={hr:.3f} [{ci_lo:.3f},{ci_hi:.3f}] p={p:.4f} C={c_idx:.3f}")
        except Exception as exc:
            print(f"  [ERR]  {col}: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Schoenfeld PH test
# ---------------------------------------------------------------------------

def run_ph_tests(df: pd.DataFrame, survivors: list[str]) -> pd.DataFrame:
    """Run Schoenfeld residual test for each survivor. Return assignment table."""
    rows = []
    for col in survivors:
        sub = df[["well_key", DURATION_COL, EVENT_COL, "stratum_key", col]].dropna()
        event_counts = sub.groupby("stratum_key")[EVENT_COL].sum()
        valid_strata = event_counts[event_counts >= MIN_STRATUM_EVENTS].index
        sub = sub[sub["stratum_key"].isin(valid_strata)].copy()

        try:
            cph = CoxPHFitter()
            cph.fit(
                sub,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                strata=["stratum_key"],
                cluster_col="well_key",
                formula=col,
                robust=True,
            )
            ph_result = proportional_hazard_test(cph, sub, time_transform="log")
            p_ph = float(ph_result.summary["p"].iloc[0])
            test_stat = float(ph_result.summary["test_statistic"].iloc[0])
            assignment = "EXTENDED" if p_ph <= P_PH_EXTENDED else "STANDARD"
            print(f"  {assignment:8s} {col}: schoenfeld_p={p_ph:.4f}")
        except Exception as exc:
            p_ph = float("nan")
            test_stat = float("nan")
            assignment = "STANDARD"
            print(f"  STANDARD {col}: PH test failed ({exc}) — defaulting to STANDARD")

        rows.append(
            {
                "covariate": col,
                "schoenfeld_stat": round(test_stat, 4) if not np.isnan(test_stat) else None,
                "p_ph": round(p_ph, 6) if not np.isnan(p_ph) else None,
                "assignment": assignment,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Spearman correlation screen
# ---------------------------------------------------------------------------

def correlation_screen(
    df: pd.DataFrame,
    candidates: list[str],
    univariate_df: pd.DataFrame,
) -> list[str]:
    """Drop one from each highly correlated pair (|ρ| > SPEARMAN_DROP_THR).

    Keep the covariate with higher univariate C-index.
    """
    sub = df[candidates].dropna()
    corr = sub.corr(method="spearman").abs()

    c_index = univariate_df.set_index("covariate")["c_index"].to_dict()
    drop: set[str] = set()

    for i, col_a in enumerate(candidates):
        if col_a in drop:
            continue
        for col_b in candidates[i + 1 :]:
            if col_b in drop:
                continue
            r = corr.loc[col_a, col_b] if col_a in corr.index and col_b in corr.columns else 0.0
            if r > SPEARMAN_DROP_THR:
                keep = col_a if c_index.get(col_a, 0) >= c_index.get(col_b, 0) else col_b
                dropped = col_b if keep == col_a else col_a
                print(
                    f"  DROP {dropped} (|r|={r:.2f} with {keep},"
                    f" C {c_index.get(col_a,0):.3f} vs {c_index.get(col_b,0):.3f})"
                )
                drop.add(dropped)

    kept = [c for c in candidates if c not in drop]
    return kept


# ---------------------------------------------------------------------------
# Step 4 — Joint multivariate model
# ---------------------------------------------------------------------------

def _add_logt_terms(df: pd.DataFrame, extended_cols: list[str]) -> pd.DataFrame:
    """Add {col}_x_logt interaction columns for EXTENDED covariates."""
    df = df.copy()
    log_tte = np.log(df[DURATION_COL].clip(lower=1.0))
    for col in extended_cols:
        df[f"{col}_x_logt"] = df[col] * log_tte
    return df


def run_joint_model(
    df: pd.DataFrame,
    standard_cols: list[str],
    extended_cols: list[str],
) -> CoxPHFitter | None:
    """Fit joint stratified Cox with standard + extended (X·log t) terms."""
    all_terms = standard_cols + extended_cols + [f"{c}_x_logt" for c in extended_cols]
    needed = ["well_key", DURATION_COL, EVENT_COL, "stratum_key"] + all_terms
    sub = df[needed].dropna()

    event_counts = sub.groupby("stratum_key")[EVENT_COL].sum()
    valid_strata = event_counts[event_counts >= MIN_STRATUM_EVENTS].index
    sub = sub[sub["stratum_key"].isin(valid_strata)]

    n_events = int(sub[EVENT_COL].sum())
    print(f"  Joint model: {len(sub)} runs, {n_events} events, "
          f"{len(sub.stratum_key.unique())} strata")
    print(f"  Standard: {standard_cols}")
    print(f"  Extended: {extended_cols}")

    formula_parts = standard_cols + [
        f"{c} + {c}_x_logt" for c in extended_cols
    ]
    formula = " + ".join(formula_parts)
    if not formula:
        print("  No terms — skipping joint model")
        return None

    try:
        cph = CoxPHFitter()
        cph.fit(
            sub,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            strata=["stratum_key"],
            cluster_col="well_key",
            formula=formula,
            robust=True,
        )
        print(cph.summary[["coef", "exp(coef)", "p"]].round(4))
        return cph
    except Exception as exc:
        print(f"  Joint model FAILED: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 5 — VIF check
# ---------------------------------------------------------------------------

def vif_check(
    df: pd.DataFrame,
    standard_cols: list[str],
    extended_cols: list[str],
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Compute VIF on standard covariate matrix; drop any col with VIF > VIF_DROP_THR.

    Extended terms (X·log t) are excluded from VIF — they are by design correlated
    with their parent X. The VIF screen detects cross-covariate redundancy only.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    all_cols = standard_cols + extended_cols
    sub = df[all_cols].dropna()
    if sub.shape[1] < 2:
        return standard_cols, extended_cols, pd.DataFrame()

    # statsmodels VIF uses uncentered R² — must center to get standard VIF
    X = (sub - sub.mean()).values
    vif_vals = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_df = pd.DataFrame({"term": all_cols, "VIF": [round(v, 2) for v in vif_vals]})
    print(vif_df.to_string(index=False))

    drop = set(vif_df[vif_df["VIF"] > VIF_DROP_THR]["term"])
    new_standard = [c for c in standard_cols if c not in drop]
    new_extended = [c for c in extended_cols if c not in drop]
    if drop:
        print(f"  VIF drop: {drop}")

    return new_standard, new_extended, vif_df


# ---------------------------------------------------------------------------
# Step 6 — Output: tables + plots
# ---------------------------------------------------------------------------

def _forest_plot(
    coef_df: pd.DataFrame,
    title: str,
    path: Path,
    coef_col: str = "coef",
    se_col: str = "se(coef)",
    label_col: str = "term",
) -> None:
    sub = coef_df.dropna(subset=[coef_col, se_col]).copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(sub))))
    y = np.arange(len(sub))
    ci = 1.96 * sub[se_col].values
    ax.errorbar(
        sub[coef_col].values,
        y,
        xerr=ci,
        fmt="o",
        color="steelblue",
        ecolor="steelblue",
        capsize=4,
        linewidth=1.2,
        markersize=6,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub[label_col].values, fontsize=9)
    ax.set_xlabel("Coefficient (log-hazard scale)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _km_tertile_plot(df: pd.DataFrame, col: str, out_path: Path) -> None:
    """KM curves by low/mid/high tertile of `col`."""
    from lifelines import KaplanMeierFitter

    sub = df[["tte", "event", col]].dropna()
    if sub.empty or int(sub["event"].sum()) < 10:
        return

    try:
        labels = ["Low", "Mid", "High"]
        bins = pd.qcut(sub[col], q=3, labels=labels, duplicates="drop")
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Low": "steelblue", "Mid": "goldenrod", "High": "firebrick"}
    for label in labels:
        mask = bins == label
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub.loc[mask, "tte"], sub.loc[mask, "event"], label=label)
        kmf.plot_survival_function(
            ax=ax,
            color=colors.get(label, "gray"),
            ci_show=True,
            ci_alpha=0.12,
        )
    ax.set_title(f"KM by tertile — {col}", fontsize=10)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_final_coeffs(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    standard_cols: list[str],
    extended_cols: list[str],
) -> pd.DataFrame:
    """Build chem_final_coeffs.csv — the VBA handoff artifact."""
    summ = cph.summary

    # Reference values: population-weighted mean (all 2,634 runs)
    all_cols = standard_cols + extended_cols
    ref_vals = {col: float(df[col].mean()) for col in all_cols if col in df.columns}

    rows = []
    for col in standard_cols:
        coef = float(summ.loc[col, "coef"]) if col in summ.index else float("nan")
        rows.append(
            {
                "covariate": col,
                "beta": round(coef, 6),
                "gamma": 0.0,
                "ref_value": round(ref_vals.get(col, float("nan")), 6),
                "type": "STANDARD",
            }
        )
    for col in extended_cols:
        term_x = col
        term_xt = f"{col}_x_logt"
        beta = float(summ.loc[term_x, "coef"]) if term_x in summ.index else float("nan")
        gamma = float(summ.loc[term_xt, "coef"]) if term_xt in summ.index else float("nan")
        rows.append(
            {
                "covariate": col,
                "beta": round(beta, 6),
                "gamma": round(gamma, 6),
                "ref_value": round(ref_vals.get(col, float("nan")), 6),
                "type": "EXTENDED",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> dict:
    out = results_dir("chemistry_cox")
    tbl = out / "tables"
    fig_dir = out / "figures"

    print("=" * 60)
    print("CHEMISTRY BLOCK COX — Phase CHEM")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading chemistry DataFrame...")
    df = build_chemistry_df()
    df = _add_logt_terms(df, CANDIDATES)  # pre-compute all x_logt columns
    print(f"  Loaded {len(df)} runs, {int(df[EVENT_COL].sum())} events")

    # Step 1: Univariate screening
    print("\n[2/6] Univariate Cox screening (p < 0.10)...")
    uni_df = run_univariate(df)
    uni_df.to_csv(tbl / "chem_univariate.csv", index=False, encoding="utf-8-sig")

    survivors = uni_df.loc[uni_df["pass_screen"], "covariate"].tolist()
    print(f"\n  Survivors ({len(survivors)}): {survivors}")

    if not survivors:
        print("No covariates survived univariate screening. Stopping.")
        return {"univariate": uni_df}

    # Step 2: PH test
    print("\n[3/6] Schoenfeld PH test...")
    ph_df = run_ph_tests(df, survivors)
    ph_df.to_csv(tbl / "chem_ph_test.csv", index=False, encoding="utf-8-sig")

    standard_survivors = ph_df.loc[ph_df["assignment"] == "STANDARD", "covariate"].tolist()
    extended_survivors = ph_df.loc[ph_df["assignment"] == "EXTENDED", "covariate"].tolist()
    print(f"\n  STANDARD: {standard_survivors}")
    print(f"  EXTENDED: {extended_survivors}")

    # Step 3: Correlation screen (on all survivors combined)
    print("\n[4/6] Spearman correlation screen (|r| > 0.65)...")
    all_survivors = standard_survivors + extended_survivors
    kept = correlation_screen(df, all_survivors, uni_df)
    final_standard = [c for c in kept if c in standard_survivors]
    final_extended = [c for c in kept if c in extended_survivors]
    print(f"\n  After correlation screen — STANDARD: {final_standard}")
    print(f"  After correlation screen — EXTENDED: {final_extended}")

    if not (final_standard or final_extended):
        print("No covariates survived correlation screen. Stopping.")
        return {"univariate": uni_df, "ph_test": ph_df}

    # Step 4: Joint model
    print("\n[5/6] Joint multivariate model...")
    cph = run_joint_model(df, final_standard, final_extended)

    if cph is None:
        print("Joint model failed. Stopping.")
        return {"univariate": uni_df, "ph_test": ph_df}

    summ = cph.summary.copy().reset_index().rename(columns={"covariate": "term"})
    summ.to_csv(tbl / "chem_joint_model.csv", index=False, encoding="utf-8-sig")

    # Step 5: VIF check on standard covariate matrix
    print("\n[6/6] VIF check...")
    vif_std, vif_ext, vif_df = vif_check(df, final_standard, final_extended)
    vif_df.to_csv(tbl / "chem_vif.csv", index=False, encoding="utf-8-sig")

    if set(vif_std) != set(final_standard) or set(vif_ext) != set(final_extended):
        print("  Refitting after VIF drops...")
        cph = run_joint_model(df, vif_std, vif_ext)
        if cph is None:
            print("  Refit failed — keeping pre-VIF model")
            vif_std, vif_ext = final_standard, final_extended
        else:
            summ = cph.summary.copy().reset_index().rename(columns={"covariate": "term"})
            summ.to_csv(tbl / "chem_joint_model.csv", index=False, encoding="utf-8-sig")

    # Final coefficients (VBA handoff)
    if cph is not None:
        coeffs = build_final_coeffs(cph, df, vif_std, vif_ext)
        coeffs.to_csv(tbl / "chem_final_coeffs.csv", index=False, encoding="utf-8-sig")
        print(f"\n  chem_final_coeffs.csv:")
        print(coeffs.to_string(index=False))

    # Plots
    print("\n  Generating plots...")
    if cph is not None:
        # Forest plot: β coefficients
        summ_full = cph.summary.copy().reset_index().rename(columns={"covariate": "term"})
        # β terms only (exclude x_logt)
        beta_rows = summ_full[~summ_full["term"].str.endswith("_x_logt")].copy()
        _forest_plot(
            beta_rows,
            title="Chemistry Cox — β coefficients",
            path=fig_dir / "chem_forest_beta.png",
            coef_col="coef",
            se_col="se(coef)",
            label_col="term",
        )
        # Forest plot: γ coefficients (extended only)
        gamma_rows = summ_full[summ_full["term"].str.endswith("_x_logt")].copy()
        if not gamma_rows.empty:
            _forest_plot(
                gamma_rows,
                title="Chemistry Cox — γ coefficients (time-dependent terms)",
                path=fig_dir / "chem_forest_gamma.png",
                coef_col="coef",
                se_col="se(coef)",
                label_col="term",
            )

    # KM tertile plots for all surviving covariates
    all_final = vif_std + vif_ext
    for col in all_final:
        _km_tertile_plot(df, col, fig_dir / f"chem_km_tertile_{col}.png")

    print(f"\nOutputs written to: {out}")
    return {
        "univariate": uni_df,
        "ph_test": ph_df,
        "joint_model": summ if cph is not None else None,
        "vif": vif_df,
        "final_coeffs": coeffs if cph is not None else None,
        "out_dir": out,
    }


if __name__ == "__main__":
    result = run()
