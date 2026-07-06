"""
All-fields frequency-exposure correlation analysis.

For each field separately, for runs with mount_year > YEAR0, computes:
  y_high =  days_freq_above_55 / days_total    (high-freq stress)
  y_low  = -days_freq_below_45 / days_total    (low-freq exposure, sign-flipped)

Outputs per field:
  - 2×2 scatter plot: (all runs / failures only) × (y_high / y_low)
    with Spearman ρ annotated on each panel
  - Univariate Cox PH regression: HR, 95% CI, p-value for each metric
  - Summary CSV with all correlation + Cox results across fields
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, norm as _norm

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.input_paths import resolve_v03_all_path
from analysis.paths import results_dir
from scripts.analyze_failure_horizon import load_runs
from scripts.data_utils import load_daily_merged

YEAR0 = 2022
FREQ_HIGH_HZ = 55.0
FREQ_LOW_HZ = 45.0
MIN_FREQ_DAYS = 7
MIN_COX_EVENTS = 5      # minimum failures for Cox to be meaningful

_SLUG = "all_fields_freq_exposure"
OUTPUT_DIR = results_dir(_SLUG)

# ---------------------------------------------------------------------------
# Frequency metric computation
# ---------------------------------------------------------------------------

def compute_freq_metrics(runs: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    working = daily.copy()
    working["well_key"] = working["well_id"].astype("string").str.strip().str.casefold()

    rows: list[dict] = []
    for _, run in runs.iterrows():
        install = pd.Timestamp(run["Дата монтажа"])
        stop    = pd.Timestamp(run["Дата остановки"])
        well_key = str(run["Скв."]).strip().casefold()

        run_daily = working.loc[
            (working["well_key"] == well_key)
            & (working["dt"] >= install)
            & (working["dt"] <= stop)
        ]
        freq = pd.to_numeric(run_daily["freq"], errors="coerce")
        n_valid = int(freq.notna().sum())

        if n_valid >= MIN_FREQ_DAYS:
            y_high =  float((freq > FREQ_HIGH_HZ).sum()) / n_valid
            y_low  = -float((freq < FREQ_LOW_HZ).sum())  / n_valid
        else:
            y_high = float("nan")
            y_low  = float("nan")

        rows.append({
            "row_id":      run["row_id"],
            "well_id":     str(run["Скв."]).strip(),
            "field":       str(run.get("Месторождение", "")).strip(),
            "mount_year":  int(install.year),
            "ttf_days":    float(run["run_days"]),
            "event":       int(run["event"]),
            "n_freq_days": n_valid,
            "y_high":      y_high,
            "y_low":       y_low,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Cox PH (univariate, scalar covariate)  — partial likelihood via BFGS
# ---------------------------------------------------------------------------

def _cox_neg_log_partial_likelihood(beta: np.ndarray, t: np.ndarray, e: np.ndarray, x: np.ndarray) -> float:
    """Breslow tie-handling partial log-likelihood (negated for minimisation)."""
    order = np.argsort(t)
    t, e, x = t[order], e[order], x[order]
    xb = x * beta[0]
    exp_xb = np.exp(xb - xb.max())       # shift for numerical stability

    log_pl = 0.0
    n = len(t)
    for i in range(n):
        if e[i] == 0:
            continue
        risk_mask = t >= t[i]
        log_pl += xb[i] - xb.max() - np.log(exp_xb[risk_mask].sum())
    return -log_pl

def _cox_hessian_diag(beta: np.ndarray, t: np.ndarray, e: np.ndarray, x: np.ndarray) -> float:
    """Diagonal of the observed information matrix (scalar covariate)."""
    order = np.argsort(t)
    t, e, x = t[order], e[order], x[order]
    xb = x * beta[0]
    exp_xb = np.exp(xb - xb.max())

    info = 0.0
    for i in range(len(t)):
        if e[i] == 0:
            continue
        risk_mask = t >= t[i]
        w = exp_xb[risk_mask]
        wx  = w * x[risk_mask]
        wx2 = w * x[risk_mask] ** 2
        denom = w.sum()
        info += wx2.sum() / denom - (wx.sum() / denom) ** 2
    return float(info)

def cox_univariate(
    t: np.ndarray,
    e: np.ndarray,
    x: np.ndarray,
) -> dict:
    """
    Fit univariate Cox PH.  Returns dict with:
      beta, se, hr, hr_lo95, hr_hi95, z, pval, n, n_events
    or NaN-filled dict on failure.
    """
    mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
    t, e, x = t[mask], e[mask].astype(float), x[mask]
    n = len(t)
    n_events = int(e.sum())

    nan_result = dict(beta=np.nan, se=np.nan, hr=np.nan,
                      hr_lo95=np.nan, hr_hi95=np.nan, z=np.nan,
                      pval=np.nan, n=n, n_events=n_events)
    if n_events < MIN_COX_EVENTS:
        return nan_result

    x_std = x.std()
    if x_std == 0:
        return nan_result
    x_scaled = x / x_std        # scale once for numerical stability

    res = minimize(
        _cox_neg_log_partial_likelihood,
        x0=np.array([0.0]),
        args=(t, e, x_scaled),
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 500},
    )
    if not res.success:
        return nan_result

    beta_scaled = float(res.x[0])
    info = _cox_hessian_diag(res.x, t, e, x_scaled)
    if info <= 0:
        return nan_result

    se_scaled = 1.0 / np.sqrt(info)
    # convert back to original scale
    beta = beta_scaled / x_std
    se   = se_scaled   / x_std
    z    = beta / se
    pval = float(2 * _norm.sf(abs(z)))
    hr   = float(np.exp(beta))
    return dict(
        beta=round(beta, 4),
        se=round(se, 4),
        hr=round(hr, 3),
        hr_lo95=round(float(np.exp(beta - 1.96 * se)), 3),
        hr_hi95=round(float(np.exp(beta + 1.96 * se)), 3),
        z=round(z, 3),
        pval=round(pval, 4),
        n=n,
        n_events=n_events,
    )

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

C_CENSORED = "#4878CF"
C_FAILURE  = "#C44E52"

def _spearman_str(x: pd.Series, y: pd.Series) -> str:
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return "ρ = n/a"
    rho, pval = spearmanr(x[mask], y[mask])
    star = "*" if pval < 0.05 else ""
    return f"ρ = {rho:+.2f}{star}  p={pval:.3f}"

def _panel(ax: plt.Axes, df: pd.DataFrame, y_col: str, title: str, ylabel: str) -> None:
    df = df.loc[df[y_col].notna()].copy()
    cens = df.loc[df["event"] == 0]
    fail = df.loc[df["event"] == 1]
    ax.scatter(cens["ttf_days"], cens[y_col], s=18, alpha=0.40,
               color=C_CENSORED, marker="o", zorder=2,
               label=f"Censored (n={len(cens)})")
    ax.scatter(fail["ttf_days"], fail[y_col], s=55, alpha=0.85,
               color=C_FAILURE, marker="X", edgecolors="black", linewidths=0.35,
               zorder=4, label=f"Failure (n={len(fail)})")

    rho_str = _spearman_str(df["ttf_days"], df[y_col])
    ax.text(0.98, 0.97, rho_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))

    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("TTF (days)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="upper right")

def _cox_annotation(ax: plt.Axes, cox: dict, metric_label: str) -> None:
    if np.isnan(cox["hr"]):
        text = f"Cox {metric_label}: insufficient data"
    else:
        star = "*" if cox["pval"] < 0.05 else ""
        text = (
            f"Cox {metric_label}: HR={cox['hr']:.2f} "
            f"[{cox['hr_lo95']:.2f}–{cox['hr_hi95']:.2f}]{star}  "
            f"p={cox['pval']:.3f}"
        )
    ax.text(0.02, 0.03, text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBE6", alpha=0.85, ec="none"))

def build_field_plot(df: pd.DataFrame, field: str,
                     cox_high: dict, cox_low: dict,
                     output_path: Path) -> None:
    usable = df.loc[df["y_high"].notna()].copy()
    failures_only = usable.loc[usable["event"] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Field: {field}  |  mount year > {YEAR0}  |  "
        f"runs={len(df)}, with_freq={len(usable)}, failures={len(failures_only)}",
        fontsize=11,
    )

    # Row 0: y_high
    _panel(axes[0, 0], usable,       "y_high",
           "y_high – all runs", f"y_high = days>{FREQ_HIGH_HZ:.0f}Hz / total")
    _panel(axes[0, 1], failures_only, "y_high",
           "y_high – failures only", f"y_high = days>{FREQ_HIGH_HZ:.0f}Hz / total")
    _cox_annotation(axes[0, 0], cox_high, f"y_high (>{FREQ_HIGH_HZ:.0f}Hz)")
    _cox_annotation(axes[0, 1], cox_high, f"y_high (>{FREQ_HIGH_HZ:.0f}Hz)")

    # Row 1: y_low
    _panel(axes[1, 0], usable,       "y_low",
           "y_low – all runs", f"y_low = −days<{FREQ_LOW_HZ:.0f}Hz / total")
    _panel(axes[1, 1], failures_only, "y_low",
           "y_low – failures only", f"y_low = −days<{FREQ_LOW_HZ:.0f}Hz / total")
    _cox_annotation(axes[1, 0], cox_low, f"y_low (<{FREQ_LOW_HZ:.0f}Hz)")
    _cox_annotation(axes[1, 1], cox_low, f"y_low (<{FREQ_LOW_HZ:.0f}Hz)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    runs = load_runs(resolve_v03_all_path())
    runs = runs.loc[runs["Дата монтажа"].dt.year > YEAR0].copy()
    if runs.empty:
        print(f"No runs with mount_year > {YEAR0}.")
        return

    fields = sorted(str(f).strip() for f in runs["Месторождение"].dropna().unique() if str(f).strip())
    print(f"Fields: {fields}")
    print(f"Total runs (mount_year > {YEAR0}): {len(runs)}")

    wells = runs["Скв."].dropna().astype(str).unique().tolist()
    print(f"Loading daily data for {len(wells)} wells …")
    daily = load_daily_merged(wells)
    print(f"Daily rows: {len(daily)}\n")

    metrics = compute_freq_metrics(runs, daily)
    metrics.to_csv(OUTPUT_DIR / "all_fields_freq_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []

    for field in fields:
        field_df = metrics.loc[metrics["field"] == field].copy()
        usable = field_df.loc[field_df["y_high"].notna()].copy()
        if usable.empty:
            print(f"  {field}: no frequency data — skip")
            continue

        t = usable["ttf_days"].to_numpy(float)
        e = usable["event"].to_numpy(float)

        cox_results: dict[str, dict] = {}
        for y_col in ("y_high", "y_low"):
            x = usable[y_col].to_numpy(float)
            cox_results[y_col] = cox_univariate(t, e, x)

        slug = field.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plot_path = OUTPUT_DIR / f"field_{slug}_freq_exposure.png"
        build_field_plot(field_df, field,
                         cox_results["y_high"], cox_results["y_low"],
                         plot_path)

        # --- summary rows ---
        for y_col, label in [("y_high", f">55Hz"), ("y_low", f"<45Hz(neg)")]:
            # Spearman — all usable
            x_series = usable[y_col]
            mask_all = x_series.notna()
            if mask_all.sum() >= 5:
                rho_all, p_all = spearmanr(usable.loc[mask_all, "ttf_days"], x_series[mask_all])
            else:
                rho_all, p_all = np.nan, np.nan

            # Spearman — failures only
            fail_df = usable.loc[usable["event"] == 1]
            mask_f = fail_df[y_col].notna()
            if mask_f.sum() >= 5:
                rho_f, p_f = spearmanr(fail_df.loc[mask_f, "ttf_days"], fail_df.loc[mask_f, y_col])
            else:
                rho_f, p_f = np.nan, np.nan

            cox = cox_results[y_col]
            summary_rows.append({
                "field":         field,
                "metric":        y_col,
                "metric_label":  label,
                "n_total":       len(field_df),
                "n_with_freq":   len(usable),
                "n_failures":    int(usable["event"].eq(1).sum()),
                # Spearman
                "spearman_rho_all":       round(rho_all, 3) if np.isfinite(rho_all) else np.nan,
                "spearman_p_all":         round(p_all, 4)   if np.isfinite(p_all)   else np.nan,
                "spearman_rho_fail_only": round(rho_f, 3)   if np.isfinite(rho_f)   else np.nan,
                "spearman_p_fail_only":   round(p_f, 4)     if np.isfinite(p_f)     else np.nan,
                # Cox
                "cox_hr":       cox["hr"],
                "cox_hr_lo95":  cox["hr_lo95"],
                "cox_hr_hi95":  cox["hr_hi95"],
                "cox_pval":     cox["pval"],
                "cox_n":        cox["n"],
                "cox_n_events": cox["n_events"],
            })

        print(
            f"  {field}: runs={len(field_df)}, with_freq={len(usable)}, "
            f"failures={int(usable['event'].eq(1).sum())}"
        )
        for y_col in ("y_high", "y_low"):
            c = cox_results[y_col]
            star = "*" if (not np.isnan(c["pval"]) and c["pval"] < 0.05) else ""
            print(
                f"    {y_col}: Cox HR={c['hr']}, 95%CI [{c['hr_lo95']}, {c['hr_hi95']}]{star}  "
                f"p={c['pval']}  (n={c['n']}, events={c['n_events']})"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "field_freq_correlation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # --- console table for large-enough fields ---
    print("\n── Spearman ρ (failures only) + Cox HR ──────────────────────────────────")
    display = summary.loc[summary["n_failures"] >= MIN_COX_EVENTS].copy()
    if not display.empty:
        cols = ["field", "metric_label", "n_with_freq", "n_failures",
                "spearman_rho_fail_only", "spearman_p_fail_only",
                "cox_hr", "cox_hr_lo95", "cox_hr_hi95", "cox_pval"]
        print(display[cols].to_string(index=False))

    print(f"\nSummary CSV : {summary_path}")
    print(f"Plots in    : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
