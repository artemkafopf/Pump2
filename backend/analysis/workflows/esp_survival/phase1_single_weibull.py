"""Phase 1 — Single Weibull MLE per stratum.

Fits:
  1. Kaplan-Meier curve
  2. Single Weibull by MLE (β, η, B10, B50, 95% CI)

Flags strata where β < 0.9 as latent-mixture suspects.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats

from analysis.models.survival.weibull_model import fit_basic_weibull
from .data import MIN_FAILURES_TWO_STAGE, stratum_summary


def _km_frame(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Return KM estimate as a DataFrame with columns: time, survival, n_risk."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label="KM")
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]
    et = kmf.event_table.reset_index()
    et = et[["event_at", "at_risk"]].rename(columns={"event_at": "time", "at_risk": "n_risk"})
    merged = sf.merge(et, on="time", how="left")
    merged["n_risk"] = merged["n_risk"].ffill().fillna(0)
    merged = merged[merged["time"] > 0].reset_index(drop=True)
    return merged


def _weibull_quantile(p: float, beta: float, eta: float) -> float:
    """Weibull quantile: t_p = η · (-ln(1-p))^(1/β)."""
    return float(eta * (-np.log(1.0 - p)) ** (1.0 / beta))


def _beta_ci_bootstrap(durations: np.ndarray, events: np.ndarray,
                        n_boot: int = 200, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI on Weibull β."""
    betas = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(durations), size=len(durations))
        try:
            res = fit_basic_weibull(durations[idx], events[idx])
            betas.append(res["beta"])
        except Exception:
            pass
    if len(betas) < 10:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(betas, 100 * alpha / 2))
    hi = float(np.percentile(betas, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _plot_km_weibull(
    km: pd.DataFrame,
    kmf: "KaplanMeierFitter",
    beta: float,
    eta: float,
    stratum: str,
    ax: plt.Axes,
) -> None:
    t_max = float(km["time"].max()) * 1.1
    t_grid = np.linspace(1e-3, t_max, 300)
    weibull_s = np.exp(-(t_grid / eta) ** beta)

    # KM with CI
    ci = kmf.confidence_interval_
    ci.columns = ["lo", "hi"]
    ci = ci.reset_index()
    ci.columns = ["time", "lo", "hi"]
    ci = ci[ci["time"] > 0]

    ax.fill_between(ci["time"], ci["lo"], ci["hi"], alpha=0.2, color="steelblue", label="KM 95% CI")
    ax.step(km["time"], km["survival"], where="post", color="steelblue", lw=2, label="Kaplan-Meier")
    ax.plot(t_grid, weibull_s, color="crimson", lw=1.5, ls="--",
            label=f"Weibull β={beta:.3f} η={eta:.0f}d")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival")
    ax.set_title(stratum, fontsize=9)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)


def run(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = stratum_summary(df)
    viable = summary[summary["n_failures"] >= MIN_FAILURES_TWO_STAGE]["stratum"].tolist()

    results = []
    kmf_cache: dict[str, KaplanMeierFitter] = {}

    for stratum in viable:
        g = df[df["stratum"] == stratum]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)

        # KM
        kmf = KaplanMeierFitter()
        kmf.fit(durations, events, label=stratum)
        kmf_cache[stratum] = kmf

        km = _km_frame(durations, events)

        # Single Weibull MLE
        try:
            w = fit_basic_weibull(durations, events)
            beta, eta = w["beta"], w["eta"]
            b10 = _weibull_quantile(0.10, beta, eta)
            b50 = _weibull_quantile(0.50, beta, eta)
            beta_lo, beta_hi = _beta_ci_bootstrap(durations, events, n_boot=100)
            mix_suspect = bool(beta < 0.9)
        except Exception as exc:
            print(f"  [Phase 1] {stratum}: Weibull fit failed — {exc}")
            beta, eta, b10, b50 = np.nan, np.nan, np.nan, np.nan
            beta_lo, beta_hi = np.nan, np.nan
            mix_suspect = False

        n_fail = int(events.sum())
        results.append({
            "stratum": stratum,
            "n_total": len(g),
            "n_failures": n_fail,
            "beta": round(beta, 4) if not np.isnan(beta) else None,
            "beta_lo95": round(beta_lo, 4) if not np.isnan(beta_lo) else None,
            "beta_hi95": round(beta_hi, 4) if not np.isnan(beta_hi) else None,
            "eta_days": round(eta, 1) if not np.isnan(eta) else None,
            "B10_days": round(b10, 1) if not np.isnan(b10) else None,
            "B50_days": round(b50, 1) if not np.isnan(b50) else None,
            "KM_median_days": round(float(kmf.median_survival_time_), 1)
                              if kmf.median_survival_time_ not in (None, np.inf) else None,
            "mixture_suspect": mix_suspect,
        })

        print(f"  [Phase 1] {stratum}: β={beta:.3f} [{beta_lo:.3f}–{beta_hi:.3f}]  "
              f"η={eta:.0f}d  B50={b50:.0f}d"
              + (" *** MIX SUSPECT" if mix_suspect else ""))

    # ── Summary table ────────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "phase1_single_weibull.csv", index=False, encoding="utf-8-sig")

    # ── Combined plot: one panel per stratum ──────────────────────────────────
    n_panels = len(viable)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, stratum in enumerate(viable):
        g = df[df["stratum"] == stratum]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)
        km = _km_frame(durations, events)
        row = next((r for r in results if r["stratum"] == stratum), None)
        if row is None or row["beta"] is None:
            axes_flat[idx].set_visible(False)
            continue
        _plot_km_weibull(km, kmf_cache[stratum], row["beta"], row["eta_days"],
                         stratum, axes_flat[idx])

    for idx in range(len(viable), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Phase 1 — KM + Single Weibull per Stratum", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "phase1_km_weibull_grid.png", dpi=150)
    plt.close(fig)

    n_suspect = sum(1 for r in results if r.get("mixture_suspect"))
    print(f"\n[Phase 1] Complete. {n_suspect}/{len(viable)} strata flagged β<0.9 (mixture suspect).")
    return {"results": result_df, "kmf_cache": kmf_cache}
