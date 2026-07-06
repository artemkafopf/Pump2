"""Compare mean-based vs proportion-based high-frequency definition.

Definition A (current):  freq_group      — based on freq_w_mean
Definition B (new):      freq_group_pct  — based on freq_above_55hz_pct > 50%

Outputs key statistics and survival estimates for both, globally and Vt-only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import kruskal

from .config import FREQ_GROUP_COLORS, MAJOR_FIELDS, MIN_FAILURES_COX


_GRPS = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
_COLORS = FREQ_GROUP_COLORS


def _survival_stats(df: pd.DataFrame, grp_col: str) -> pd.DataFrame:
    rows = []
    for scope, mask in [("GLOBAL", df.index), ("Vt", df[df["field"] == "Vt"].index)]:
        sub = df.loc[mask]
        for grp in _GRPS:
            g = sub[sub[grp_col] == grp]
            if len(g) == 0:
                continue
            n_fail = int(g["event"].sum())
            fail_dur = g.loc[g["event"] == 1, "duration"]
            eligible = g[(g["duration"] >= 90) | (g["event"] == 1)]
            n_inf = int(((eligible["event"] == 1) & (eligible["duration"] < 90)).sum())
            rows.append({
                "scope": scope,
                "freq_group": grp,
                "definition": grp_col,
                "n_runs": len(g),
                "n_fail": n_fail,
                "fail_rate_pct": round(n_fail / len(g) * 100, 1),
                "median_TTF_fail": round(fail_dur.median(), 1) if n_fail > 0 else np.nan,
                "median_TTF_all": round(g["duration"].median(), 1),
                "infant_rate_90d": round(n_inf / max(len(eligible), 1) * 100, 1),
            })
    return pd.DataFrame(rows)


def _logrank_p(df: pd.DataFrame, grp_col: str, scope: str) -> dict:
    sub = df if scope == "GLOBAL" else df[df["field"] == "Vt"]
    sub = sub[sub[grp_col].isin(_GRPS)].dropna(subset=["duration", "event"])
    result = {"scope": scope, "definition": grp_col}
    if sub[grp_col].nunique() < 2:
        result["logrank_p"] = np.nan
        return result
    try:
        res = multivariate_logrank_test(sub["duration"], sub[grp_col], sub["event"])
        result["logrank_p"] = round(float(res.p_value), 4)
    except Exception:
        result["logrank_p"] = np.nan
    return result


def _weibull(df: pd.DataFrame, grp_col: str) -> pd.DataFrame:
    rows = []
    for scope, mask in [("GLOBAL", df.index), ("Vt", df[df["field"] == "Vt"].index)]:
        sub = df.loc[mask]
        for grp in _GRPS:
            g = sub[sub[grp_col] == grp]
            if g["event"].sum() < 30:
                continue
            wf = WeibullFitter()
            try:
                wf.fit(g["duration"], event_observed=g["event"])
                rows.append({
                    "scope": scope,
                    "freq_group": grp,
                    "definition": grp_col,
                    "n_fail": int(g["event"].sum()),
                    "beta": round(float(wf.rho_), 3),
                    "eta_days": round(float(wf.lambda_), 1),
                    "beta_lo": round(float(wf.summary.loc["rho_", "coef lower 95%"]), 3),
                    "beta_hi": round(float(wf.summary.loc["rho_", "coef upper 95%"]), 3),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


def _cox_hr(df: pd.DataFrame, grp_col: str, scope: str) -> dict:
    sub = df if scope == "GLOBAL" else df[df["field"] == "Vt"]
    sub = sub[sub[grp_col].isin(_GRPS)].copy()
    sub["freq_high"] = (sub[grp_col] == "High (>55 Hz)").astype(float)
    sub["freq_low"]  = (sub[grp_col] == "Low (≤50 Hz)").astype(float)

    # Field dummies for global
    covariates = ["freq_high", "freq_low"]
    if scope == "GLOBAL":
        for f in MAJOR_FIELDS:
            if f != "Ya" and (sub["field"] == f).sum() > 0:
                sub[f"field_{f}"] = (sub["field"] == f).astype(float)
                covariates.append(f"field_{f}")

    data = sub[covariates + ["duration", "event", "well_key"]].dropna()
    if data["event"].sum() < MIN_FAILURES_COX:
        return {"scope": scope, "definition": grp_col, "hr_high": np.nan, "hr_ci_lo": np.nan, "hr_ci_hi": np.nan, "p_high": np.nan}

    try:
        cph = CoxPHFitter()
        cph.fit(data, duration_col="duration", event_col="event",
                cluster_col="well_key", robust=True)
        s = cph.summary
        return {
            "scope": scope,
            "definition": grp_col,
            "n_fail": int(data["event"].sum()),
            "hr_high": round(float(s.loc["freq_high", "exp(coef)"]), 3),
            "hr_ci_lo": round(float(s.loc["freq_high", "exp(coef) lower 95%"]), 3),
            "hr_ci_hi": round(float(s.loc["freq_high", "exp(coef) upper 95%"]), 3),
            "p_high": round(float(s.loc["freq_high", "p"]), 4),
        }
    except Exception as e:
        return {"scope": scope, "definition": grp_col, "error": str(e)}


def _km_figure(df: pd.DataFrame, scope: str, out: Path) -> None:
    sub = df if scope == "GLOBAL" else df[df["field"] == "Vt"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (grp_col, label) in zip(axes, [
        ("freq_group", "Mean-based (freq_w_mean)"),
        ("freq_group_pct", "Proportion-based (>50% time @ >55 Hz)"),
    ]):
        for grp in _GRPS:
            g = sub[sub[grp_col] == grp]
            if g["event"].sum() < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(g["duration"], event_observed=g["event"], label=f"{grp} (n={len(g)})")
            kmf.plot_survival_function(ax=ax, ci_show=True, color=_COLORS.get(grp, "#888"), linewidth=2)
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"{scope} — {label}")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1)
    fig.suptitle(f"KM curves: mean-based vs proportion-based definition ({scope})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"freq_def_compare_km_{scope.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _crossover_table(df: pd.DataFrame) -> pd.DataFrame:
    """Which runs change classification between definitions?"""
    ct = pd.crosstab(
        df["freq_group"].rename("mean_based"),
        df["freq_group_pct"].rename("pct_based"),
        margins=True,
    )
    return ct


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ---------------------------------------------------------------
    # Cross-tabulation: which runs change definition?
    # ---------------------------------------------------------------
    ct = _crossover_table(df)
    ct.to_csv(out / "freq_def_crossover.csv", encoding="utf-8-sig")
    results["crossover"] = ct

    # Runs that CHANGE group
    changed = df[df["freq_group"] != df["freq_group_pct"]].copy()
    print(f"\nRuns that change freq group: {len(changed)} ({len(changed)/len(df)*100:.1f}%)")
    print(changed.groupby(["freq_group", "freq_group_pct"]).size().rename("n").reset_index().to_string(index=False))

    # ---------------------------------------------------------------
    # Survival stats under both definitions
    # ---------------------------------------------------------------
    stats_a = _survival_stats(df, "freq_group")
    stats_b = _survival_stats(df, "freq_group_pct")
    all_stats = pd.concat([stats_a, stats_b], ignore_index=True)
    all_stats.to_csv(out / "freq_def_survival_stats.csv", index=False, encoding="utf-8-sig")
    results["survival_stats"] = all_stats

    # ---------------------------------------------------------------
    # Log-rank tests
    # ---------------------------------------------------------------
    lr_rows = []
    for grp_col in ["freq_group", "freq_group_pct"]:
        for scope in ["GLOBAL", "Vt"]:
            lr_rows.append(_logrank_p(df, grp_col, scope))
    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv(out / "freq_def_logrank.csv", index=False, encoding="utf-8-sig")
    results["logrank"] = lr_df

    # ---------------------------------------------------------------
    # Weibull shape parameters
    # ---------------------------------------------------------------
    wb_a = _weibull(df, "freq_group")
    wb_b = _weibull(df, "freq_group_pct")
    wb_all = pd.concat([wb_a, wb_b], ignore_index=True)
    wb_all.to_csv(out / "freq_def_weibull.csv", index=False, encoding="utf-8-sig")
    results["weibull"] = wb_all

    # ---------------------------------------------------------------
    # Cox HR
    # ---------------------------------------------------------------
    cox_rows = []
    for grp_col in ["freq_group", "freq_group_pct"]:
        for scope in ["GLOBAL", "Vt"]:
            cox_rows.append(_cox_hr(df, grp_col, scope))
    cox_df = pd.DataFrame(cox_rows)
    cox_df.to_csv(out / "freq_def_cox.csv", index=False, encoding="utf-8-sig")
    results["cox"] = cox_df

    # ---------------------------------------------------------------
    # KM figures
    # ---------------------------------------------------------------
    _km_figure(df, "GLOBAL", out)
    _km_figure(df, "Vt", out)

    # ---------------------------------------------------------------
    # Side-by-side bar: infant rate by definition
    # ---------------------------------------------------------------
    high_stats = all_stats[all_stats["freq_group"] == "High (>55 Hz)"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ylabel in zip(axes,
        ["median_TTF_fail", "infant_rate_90d"],
        ["Median TTF (failures, days)", "Infant rate 90d (%)"]):
        for i, (scope, grp_col, label, color) in enumerate([
            ("GLOBAL", "freq_group", "Mean-based", "#4878CF"),
            ("GLOBAL", "freq_group_pct", "Pct-based", "#D65F5F"),
            ("Vt", "freq_group", "Mean-based Vt", "#4878CF"),
            ("Vt", "freq_group_pct", "Pct-based Vt", "#D65F5F"),
        ]):
            row = all_stats[(all_stats["scope"] == scope) & (all_stats["definition"] == grp_col) & (all_stats["freq_group"] == "High (>55 Hz)")]
            if row.empty:
                continue
            val = float(row[col].values[0])
            ax.bar(i, val, color=color, alpha=0.8 if "Vt" not in label else 0.5,
                   label=label if i < 2 else None,
                   hatch="" if "Vt" not in label else "//")
            ax.text(i, val + 0.5, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Mean\nGlobal", "Pct\nGlobal", "Mean\nVt", "Pct\nVt"], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"High-freq group: {ylabel}")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Effect of freq definition change on 'High (>55 Hz)' group statistics", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "freq_def_compare_highfreq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FREQUENCY DEFINITION COMPARISON")
    print("=" * 65)

    print("\n--- Cross-tabulation (mean-based rows × pct-based cols) ---")
    print(ct)

    print("\n--- Survival stats for High (>55 Hz) group ---")
    high = all_stats[all_stats["freq_group"] == "High (>55 Hz)"]
    print(high[["scope", "definition", "n_runs", "n_fail", "fail_rate_pct",
                "median_TTF_fail", "infant_rate_90d"]].to_string(index=False))

    print("\n--- Log-rank p (High vs Normal vs Low) ---")
    print(lr_df.to_string(index=False))

    print("\n--- Cox HR for High vs Normal ---")
    show = [c for c in ["scope", "definition", "n_fail", "hr_high", "hr_ci_lo", "hr_ci_hi", "p_high"] if c in cox_df.columns]
    print(cox_df[show].to_string(index=False))

    print("\n--- Weibull β for High-freq group ---")
    wb_high = wb_all[wb_all["freq_group"] == "High (>55 Hz)"]
    print(wb_high[["scope", "definition", "n_fail", "beta", "beta_lo", "beta_hi", "eta_days"]].to_string(index=False))

    return results
