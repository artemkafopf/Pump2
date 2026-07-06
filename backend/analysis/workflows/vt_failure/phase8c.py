"""Phase 8c — Latent Weibull mixture model + conventional and extended Cox for H2S (Vt).

Population : Vt field, all runs with known H2S classification (raw baseline).
Covariate  : is_acidic  (X=0 Некислый, X=1 Кислый)

Approach
--------
1. Fit K=2 Weibull mixture to each H2S group's KM curve via weighted least-squares
   (uses the existing latent_weibull_competing_risks infrastructure).

2. Conventional Cox PH  — CoxPHFitter(is_acidic), cluster-robust on well_key.

3. Extended Cox on MIXTURE curves  — the mixture survival S_mix(t) is smooth and
   analytically defined; the log-log regression y(t) = β* + γ·log(t) runs on a fine
   time grid instead of on jagged KM steps.  This removes the discretisation noise
   that limits the KM-based regression in phase 8b.

   Because the mixture baseline H₀(t) = −log S₀_mix(t) is not a Weibull power law,
   the analytical Weibull correction β = β* − log[β₀/(β₀+γ)] does not apply.
   We report β* directly as the "log-scale offset" at t=1 day plus γ·0 = β*.

4. Cross-comparison table: KM-based Extended Cox (8b) vs mixture Extended Cox (8c)
   vs conventional Cox.

Outputs (tables/)
-----------------
    p8c_mixture_params.csv          — fitted (β₁, η₁, w₁, β₂, η₂) per H2S group
    p8c_summary.csv                 — 3-model comparison table
    p8c_regression_points.csv       — (time, y_obs, y_fit) from mixture regression

Outputs (figures/)
------------------
    p8c_mixture_fit.png             — KM + mixture overlay per group (4-panel)
    p8c_component_bars.png          — component parameter comparison (bar chart)
    p8c_loglog_mixture.png          — smooth log(−log S_mix) vs log(t)
    p8c_y_regression_mixture.png    — y(t) regression on mixture curves
    p8c_tv_hr.png                   — time-varying HR from mixture Extended Cox +
                                      conventional Cox reference line
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    fit_latent_weibull_to_km_frame,
    latent_survival,
    latent_hazard,
    build_survival_curve_weights,
)
from analysis.models.survival.extended_cox import fit_extended_cox_log_log


# ── palette ───────────────────────────────────────────────────────────────────
_C = {"acidic": "#d62728", "neutral": "#2ca02c"}
_LABEL_AC = "Кислый (H₂S)"
_LABEL_NE = "Некислый"

# Initial guess for mixture fit — from Bayesian Vt posteriors in survival_analysis_report.md
_INIT_MODEL = TwoComponentLatentWeibullModel(
    weight_1=0.40,
    component_1=WeibullParameters(beta=0.85, eta=100.0, label="infant"),
    component_2=WeibullParameters(beta=1.30, eta=280.0, label="wear-out"),
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare_vt_h2s(df: pd.DataFrame) -> pd.DataFrame:
    vt = df[df["is_vt"]].copy()
    known = vt[vt["h2s_label"].isin(["Кислый", "Некислый"])].copy()
    known["is_acidic"] = (known["h2s_label"] == "Кислый").astype(float)
    return known[known["duration"] > 0].reset_index(drop=True)


def _fit_km(sub: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(sub["duration"], sub["event"])
    return kmf


def _kmf_to_frame(kmf: KaplanMeierFitter) -> pd.DataFrame:
    """Convert lifelines KMF to the km_frame format expected by latent_weibull fitters."""
    sf = kmf.survival_function_.copy()
    sf.columns = ["survival"]
    sf.index.name = "time"
    sf = sf.reset_index()
    et = kmf.event_table
    sf = sf.merge(
        et[["at_risk"]].rename(columns={"at_risk": "n_risk"}).reset_index().rename(columns={"event_at": "time"}),
        on="time",
        how="left",
    )
    sf["n_risk"] = sf["n_risk"].ffill().fillna(1.0)
    return sf


def _fit_mixture(kmf: KaplanMeierFitter, label: str) -> TwoComponentLatentWeibullModel:
    """Fit K=2 latent Weibull to a KM curve with multi-start."""
    km_frame = _kmf_to_frame(kmf)
    result = fit_latent_weibull_to_km_frame(
        km_frame,
        initial_model=_INIT_MODEL,
        min_n_risk=5,
        num_starts=8,
        max_iter=600,
    )
    m = result.model
    print(
        f"  [{label}] β₁={m.component_1.beta:.3f}  η₁={m.component_1.eta:.1f}d  "
        f"w₁={m.weight_1:.3f}  |  "
        f"β₂={m.component_2.beta:.3f}  η₂={m.component_2.eta:.1f}d  "
        f"w₂={m.weight_2:.3f}  |  RMSE={result.rmse:.4f}"
    )
    return m


def _mixture_sf_df(model: TwoComponentLatentWeibullModel, times: np.ndarray) -> pd.DataFrame:
    """Return a pseudo-KM survival DataFrame for the mixture at given times."""
    surv = np.asarray(latent_survival(times, model), dtype=float)
    return pd.DataFrame({"KM_estimate": surv}, index=times)


def _mixture_params_row(label: str, model: TwoComponentLatentWeibullModel) -> dict:
    return {
        "group": label,
        "w1": round(model.weight_1, 4),
        "beta1": round(model.component_1.beta, 4),
        "eta1_days": round(model.component_1.eta, 1),
        "median1_days": round(model.component_1.eta * (np.log(2) ** (1 / model.component_1.beta)), 1),
        "w2": round(model.weight_2, 4),
        "beta2": round(model.component_2.beta, 4),
        "eta2_days": round(model.component_2.eta, 1),
        "median2_days": round(model.component_2.eta * (np.log(2) ** (1 / model.component_2.beta)), 1),
    }


# ── conventional Cox (same as 8b) ────────────────────────────────────────────

def _fit_conventional_cox(sub: pd.DataFrame) -> dict:
    fit_df = sub[["duration", "event", "is_acidic", "well_key"]].dropna().copy()
    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col="duration", event_col="event",
                cluster_col="well_key", robust=True)
    except Exception as exc:
        return {"error": str(exc)}
    row = cph.summary.loc["is_acidic"]
    ph_p = np.nan
    try:
        ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_p = float(ph.summary["p"].iloc[0])
    except Exception:
        pass
    return {
        "cox_hr":        float(row["exp(coef)"]),
        "cox_ci_lo":     float(row["exp(coef) lower 95%"]),
        "cox_ci_hi":     float(row["exp(coef) upper 95%"]),
        "cox_p":         float(row["p"]),
        "cox_concordance": float(cph.concordance_index_),
        "cox_ph_p":      ph_p,
        "cox_ph_ok":     bool(ph_p > 0.05) if np.isfinite(ph_p) else None,
        "_cph_obj":      cph,
    }


# ── figures ───────────────────────────────────────────────────────────────────

def _plot_mixture_fit(
    km_ne: KaplanMeierFitter,
    km_ac: KaplanMeierFitter,
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    out: Path,
) -> None:
    t_max = max(
        float(km_ne.survival_function_.index.max()),
        float(km_ac.survival_function_.index.max()),
    )
    t_grid = np.linspace(0, t_max, 400)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, kmf, model, label, colour in [
        (axes[0], km_ne, m_ne, _LABEL_NE, _C["neutral"]),
        (axes[1], km_ac, m_ac, _LABEL_AC, _C["acidic"]),
    ]:
        # KM
        ax.step(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            where="post",
            color=colour,
            alpha=0.55,
            linewidth=1.4,
            label="Kaplan-Meier",
        )
        # KM CI
        try:
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(
                ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1],
                step="post",
                alpha=0.12,
                color=colour,
            )
        except Exception:
            pass

        # Mixture overall
        s_mix = np.asarray(latent_survival(t_grid, model), dtype=float)
        ax.plot(t_grid, s_mix, color=colour, linewidth=2.2, label="Mixture S(t)")

        # Components
        from analysis.models.survival.latent_weibull_competing_risks import weibull_survival as wb_surv
        s1 = np.asarray(wb_surv(t_grid, model.component_1), dtype=float)
        s2 = np.asarray(wb_surv(t_grid, model.component_2), dtype=float)
        ax.plot(t_grid, s1, color=colour, linewidth=1.0, linestyle="--",
                alpha=0.7, label=f"C1: β={model.component_1.beta:.2f}, η={model.component_1.eta:.0f}d")
        ax.plot(t_grid, s2, color=colour, linewidth=1.0, linestyle=":",
                alpha=0.7, label=f"C2: β={model.component_2.beta:.2f}, η={model.component_2.eta:.0f}d")

        ax.set_title(f"{label}\nw₁={model.weight_1:.2f}  w₂={model.weight_2:.2f}", fontsize=10)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("S(t)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.22)

    fig.suptitle("Latent Weibull mixture fit — H₂S groups (Vt)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "p8c_mixture_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_survival(
    km_ne: KaplanMeierFitter,
    km_ac: KaplanMeierFitter,
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    """All curves on one panel: KM + mixture + both components for each H2S group."""
    from analysis.models.survival.latent_weibull_competing_risks import weibull_survival as wb_surv

    fig, ax = plt.subplots(figsize=(10, 6))

    for kmf, model, label, colour in [
        (km_ne, m_ne, _LABEL_NE, _C["neutral"]),
        (km_ac, m_ac, _LABEL_AC, _C["acidic"]),
    ]:
        short = "Non-sour" if "Некислый" in label else "Sour"

        # KM step curve
        ax.step(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            where="post",
            color=colour,
            linewidth=2.2,
            alpha=0.85,
            label=f"{short} — KM",
        )
        # KM 95% CI band
        try:
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(
                ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1],
                step="post",
                alpha=0.10,
                color=colour,
            )
        except Exception:
            pass

        # Mixture overall
        s_mix = np.asarray(latent_survival(t_grid, model), dtype=float)
        ax.plot(
            t_grid, s_mix,
            color=colour, linewidth=2.0, linestyle="-",
            alpha=0.6,
            label=f"{short} — mixture",
        )

        # Component 1 (infant / early-failure mode)
        s1 = np.asarray(wb_surv(t_grid, model.component_1), dtype=float)
        ax.plot(
            t_grid, s1,
            color=colour, linewidth=1.3, linestyle="--", alpha=0.65,
            label=(
                f"{short} C1 (infant)  "
                f"β={model.component_1.beta:.2f}  η={model.component_1.eta:.0f}d  "
                f"w={model.weight_1:.2f}"
            ),
        )

        # Component 2 (wear-out mode)
        s2 = np.asarray(wb_surv(t_grid, model.component_2), dtype=float)
        ax.plot(
            t_grid, s2,
            color=colour, linewidth=1.3, linestyle=":", alpha=0.65,
            label=(
                f"{short} C2 (wear-out)  "
                f"β={model.component_2.beta:.2f}  η={model.component_2.eta:.0f}d  "
                f"w={model.weight_2:.2f}"
            ),
        )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival S(t)", fontsize=11)
    ax.set_title(
        "Survival by H₂S group — KM, mixture S(t) and failure modes (Vt)\n"
        "Solid thick = KM  ·  Solid thin = mixture  ·  Dashed = infant mode  ·  Dotted = wear-out mode",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out / "p8c_combined_survival.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_bars(
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    out: Path,
) -> None:
    params = {
        "β₁ (infant shape)": (m_ne.component_1.beta, m_ac.component_1.beta),
        "η₁ (infant scale, d)": (m_ne.component_1.eta, m_ac.component_1.eta),
        "w₁ (infant weight)": (m_ne.weight_1, m_ac.weight_1),
        "β₂ (wear-out shape)": (m_ne.component_2.beta, m_ac.component_2.beta),
        "η₂ (wear-out scale, d)": (m_ne.component_2.eta, m_ac.component_2.eta),
    }
    names = list(params)
    vals_ne = [v[0] for v in params.values()]
    vals_ac = [v[1] for v in params.values()]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, vals_ne, w, color=_C["neutral"], alpha=0.75, label=_LABEL_NE)
    ax.bar(x + w / 2, vals_ac, w, color=_C["acidic"],  alpha=0.75, label=_LABEL_AC)
    for xi, (ne, ac) in enumerate(zip(vals_ne, vals_ac)):
        ax.text(xi - w / 2, ne + 0.01 * max(max(vals_ne), max(vals_ac)), f"{ne:.2f}",
                ha="center", va="bottom", fontsize=7)
        ax.text(xi + w / 2, ac + 0.01 * max(max(vals_ne), max(vals_ac)), f"{ac:.2f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title("Mixture component parameters — H₂S groups (Vt)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8c_component_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loglog_mixture(
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model, label, colour in [
        (m_ne, _LABEL_NE, _C["neutral"]),
        (m_ac, _LABEL_AC, _C["acidic"]),
    ]:
        s = np.asarray(latent_survival(t_grid, model), dtype=float)
        valid = (s > 0) & (s < 1)
        ax.plot(
            np.log(t_grid[valid]),
            np.log(-np.log(s[valid])),
            label=label,
            color=colour,
            linewidth=2,
        )

    ax.set_xlabel("log(t)  [t in days]")
    ax.set_ylabel("log(−log S_mix(t))")
    ax.set_title(
        "Log-log plot on mixture curves — H₂S groups (Vt)\n"
        "Parallel lines ⇒ proportional hazards"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8c_loglog_mixture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_y_regression_mixture(res, out: Path) -> None:
    times = res.times
    t_grid = np.linspace(times.min(), times.max(), 250)
    hr_lo, hr_hi = res.hr_ci(t_grid)
    y_lo = np.log(np.clip(hr_lo, 1e-9, None))
    y_hi = np.log(np.clip(hr_hi, 1e-9, None))
    y_grid = res.beta_star + res.gamma * np.log(t_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log(times), res.y_observed, s=18, alpha=0.6, color="#555",
               label="y(t) mixture")
    ax.plot(np.log(t_grid), y_grid, color="#1f77b4", linewidth=2,
            label=f"β* + γ·log(t)  γ={res.gamma:+.3f}  p={res.p_gamma:.3f}")
    ax.fill_between(np.log(t_grid), y_lo, y_hi, alpha=0.15, color="#1f77b4",
                    label="95% CI")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="y=0  (proportional hazards)")

    ax.set_xlabel("log(t)  [t in days]")
    ax.set_ylabel("y(t) = log(−log S₁_mix) − log(−log S₀_mix)")
    ax.set_title(
        f"Extended Cox on mixture — H₂S effect (Vt)\n"
        f"β*={res.beta_star:+.3f}  γ={res.gamma:+.3f}  R²={res.r_squared:.3f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8c_y_regression_mixture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tv_hr(res, cox_hr: float, cox_ci: tuple[float, float], out: Path) -> None:
    t_grid = np.linspace(1, float(res.times.max()), 300)
    hr_tv = res.hr_profile(t_grid)
    hr_lo, hr_hi = res.hr_ci(t_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_grid, hr_tv, color="#d62728", linewidth=2,
            label="Extended Cox HR(t)  [mixture]")
    ax.fill_between(t_grid, hr_lo, hr_hi, alpha=0.15, color="#d62728",
                    label="95% CI")
    ax.axhline(cox_hr, color="#1f77b4", linewidth=1.8, linestyle="--",
               label=f"Conventional Cox HR={cox_hr:.3f}")
    ax.axhspan(cox_ci[0], cox_ci[1], alpha=0.10, color="#1f77b4")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Hazard Ratio  (Кислый vs Некислый)")
    ax.set_title("Time-varying HR for H₂S (Vt) — mixture-based Extended Cox")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8c_tv_hr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── hazard comparison ─────────────────────────────────────────────────────────

def _plot_mixture_hazard(
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    h_ne = np.asarray(latent_hazard(t_grid, m_ne), dtype=float)
    h_ac = np.asarray(latent_hazard(t_grid, m_ac), dtype=float)
    hr_raw = h_ac / np.clip(h_ne, 1e-12, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(t_grid, h_ne, color=_C["neutral"], linewidth=1.8, label=_LABEL_NE)
    axes[0].plot(t_grid, h_ac, color=_C["acidic"],  linewidth=1.8, label=_LABEL_AC)
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("h(t)  mixture hazard")
    axes[0].set_title("Mixture hazard by H₂S group (Vt)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_grid, hr_raw, color="#7B2D8B", linewidth=2)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Hazard Ratio  h_acidic / h_neutral")
    axes[1].set_title("Observed (mixture) HR over time — H₂S (Vt)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "p8c_mixture_hazard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── classical Weibull ─────────────────────────────────────────────────────────

def _plot_classical_weibull(
    km_ne: KaplanMeierFitter,
    km_ac: KaplanMeierFitter,
    neutral_sub: pd.DataFrame,
    acidic_sub: pd.DataFrame,
    t_grid: np.ndarray,
    out: Path,
) -> dict:
    """Fit one Weibull per H2S group via MLE and plot KM + Weibull survival."""
    from analysis.models.survival.weibull_model import fit_basic_weibull

    fits = {}
    for sub, label, short in [
        (neutral_sub, _LABEL_NE, "non-sour"),
        (acidic_sub,  _LABEL_AC, "sour"),
    ]:
        res = fit_basic_weibull(
            sub["duration"].to_numpy(float),
            sub["event"].to_numpy(int),
        )
        fits[short] = res
        print(
            f"  [{short}] β={res['beta']:.3f}  η={res['eta']:.1f}d  "
            f"B50={res['eta'] * (np.log(2) ** (1/res['beta'])):.1f}d  "
            f"AIC={res['aic']:.1f}"
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    for kmf, sub, short, colour in [
        (km_ne, neutral_sub, "non-sour", _C["neutral"]),
        (km_ac, acidic_sub,  "sour",     _C["acidic"]),
    ]:
        res = fits[short]
        beta, eta = res["beta"], res["eta"]
        b50 = eta * (np.log(2) ** (1 / beta))
        label_km = "Non-sour — KM" if short == "non-sour" else "Sour — KM"
        label_wb = (
            f"{'Non-sour' if short == 'non-sour' else 'Sour'} — Weibull"
            f"  β={beta:.2f}  η={eta:.0f}d  B50={b50:.0f}d"
        )

        # KM
        ax.step(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            where="post",
            color=colour, linewidth=2.2, alpha=0.85,
            label=label_km,
        )
        try:
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(
                ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                step="post", alpha=0.10, color=colour,
            )
        except Exception:
            pass

        # Weibull S(t) = exp(-(t/eta)^beta)
        s_wb = np.exp(-np.power(t_grid / eta, beta))
        ax.plot(t_grid, s_wb, color=colour, linewidth=2.0, linestyle="--",
                alpha=0.85, label=label_wb)

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival S(t)", fontsize=11)
    ax.set_title(
        "Classical Weibull vs KM — Sour / Non-sour (Vt)\n"
        "Solid = KM  ·  Dashed = Weibull MLE fit",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out / "p8c_classical_weibull.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fits


# ── component-specific Extended Cox ──────────────────────────────────────────

def _component_extended_cox_params(
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
) -> dict:
    """
    For each Weibull component k, recover the Extended Cox parameters (β*_k, γ_k)
    by comparing the fitted parameters across H2S groups.

    For a Weibull h_k(t) = (β_k/η_k^β_k) · t^(β_k−1):

        log HR_k(t) = log h_k(t|sour) − log h_k(t|non-sour)
                    = β*_k + γ_k · log(t)

    where:
        γ_k  = β_k^sour − β_k^non-sour          (change in shape = time coefficient)
        β*_k = log(β_k^s) − β_k^s·log(η_k^s)
             − log(β_k^n) + β_k^n·log(η_k^n)    (intercept = scale + shape offset)

    HR_k(t) = exp(β*_k) · t^γ_k
    HR_k(1) = exp(β*_k)                          (H2S effect at t=1 day for mode k)
    """
    results = {}
    for k, (comp_ne, comp_ac) in enumerate(
        [
            (m_ne.component_1, m_ac.component_1),
            (m_ne.component_2, m_ac.component_2),
        ],
        start=1,
    ):
        bn, hn = comp_ne.beta, comp_ne.eta
        bs, hs = comp_ac.beta, comp_ac.eta

        gamma_k  = bs - bn
        beta_star_k = (
            np.log(bs) - bs * np.log(hs)
            - np.log(bn) + bn * np.log(hn)
        )
        hr_at_t1 = float(np.exp(beta_star_k))
        results[k] = {
            "mode": comp_ne.label,
            "beta_non_sour": bn,
            "eta_non_sour":  hn,
            "beta_sour":     bs,
            "eta_sour":      hs,
            "gamma_k":       gamma_k,
            "beta_star_k":   beta_star_k,
            "hr_at_t1":      hr_at_t1,
        }
    return results


def _plot_component_hr(
    comp_params: dict,
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    """
    Two-panel figure:
      Left  — HR_k(t) for each component + overall mixture HR(t)
      Right — log HR_k(t) vs log(t) showing the linear structure of each mode
    """
    _mode_colours = {1: "#E07B39", 2: "#7B2D8B"}   # orange = infant, purple = wear-out
    _mode_labels  = {1: "Infant mode (C1)", 2: "Wear-out mode (C2)"}

    # overall mixture HR from hazard functions
    h_ne  = np.asarray(latent_hazard(t_grid, m_ne), dtype=float)
    h_ac  = np.asarray(latent_hazard(t_grid, m_ac), dtype=float)
    hr_mix = h_ac / np.clip(h_ne, 1e-12, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: HR(t) linear scale ─────────────────────────────────────────────
    ax = axes[0]
    for k, p in comp_params.items():
        hr_k = np.exp(p["beta_star_k"] + p["gamma_k"] * np.log(np.clip(t_grid, 1e-9, None)))
        ax.plot(
            t_grid, hr_k,
            color=_mode_colours[k], linewidth=2.2,
            label=(
                f"{_mode_labels[k]}\n"
                f"  γ={p['gamma_k']:+.3f}  β*={p['beta_star_k']:+.3f}\n"
                f"  HR(1d)={p['hr_at_t1']:.2f}"
            ),
        )

    ax.plot(t_grid, hr_mix, color="#1f77b4", linewidth=2.0, linestyle="--",
            alpha=0.8, label="Overall mixture HR(t)\n  (weighted combination)")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel("Hazard Ratio  (Sour / Non-sour)", fontsize=10)
    ax.set_title("Component-specific H₂S hazard ratios\nExtended Cox γ_k = Δβ_k", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22)

    # ── right: log HR(t) vs log(t) — linear structure ───────────────────────
    ax2 = axes[1]
    log_t = np.log(np.clip(t_grid, 1, None))

    for k, p in comp_params.items():
        log_hr_k = p["beta_star_k"] + p["gamma_k"] * log_t
        ax2.plot(
            log_t, log_hr_k,
            color=_mode_colours[k], linewidth=2.2,
            label=f"{_mode_labels[k]}  γ={p['gamma_k']:+.3f}",
        )

    log_hr_mix = np.log(np.clip(hr_mix, 1e-12, None))
    valid = np.isfinite(log_hr_mix)
    ax2.plot(log_t[valid], log_hr_mix[valid], color="#1f77b4", linewidth=1.8,
             linestyle="--", alpha=0.8, label="Mixture log HR(t)")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    # annotate slope = gamma
    for k, p in comp_params.items():
        x_ann = log_t[len(log_t) // 2]
        y_ann = p["beta_star_k"] + p["gamma_k"] * x_ann
        ax2.annotate(
            f"slope = γ_{k} = {p['gamma_k']:+.3f}",
            xy=(x_ann, y_ann),
            xytext=(x_ann - 0.8, y_ann + 0.3),
            fontsize=8,
            color=_mode_colours[k],
            arrowprops=dict(arrowstyle="->", color=_mode_colours[k], lw=1.0),
        )

    ax2.set_xlabel("log(t)  [t in days]", fontsize=10)
    ax2.set_ylabel("log HR_k(t)", fontsize=10)
    ax2.set_title(
        "log HR vs log(t) per component\n"
        "Straight line confirms Extended Cox structure: log HR = β* + γ·log(t)",
        fontsize=10,
    )
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.22)

    fig.suptitle("H₂S effect decomposed by failure mode (Vt) — Extended Cox per latent component", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "p8c_component_hr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_survival_shift(
    m_ne: TwoComponentLatentWeibullModel,
    m_ac: TwoComponentLatentWeibullModel,
    comp_params: dict,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    """Show sour vs non-sour survival within each component, plus weight shift."""
    from analysis.models.survival.latent_weibull_competing_risks import weibull_survival as wb_surv

    _mode_colours = {1: "#E07B39", 2: "#7B2D8B"}
    _mode_labels  = {1: "Infant (C1)", 2: "Wear-out (C2)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    comps = [
        (1, m_ne.component_1, m_ac.component_1),
        (2, m_ne.component_2, m_ac.component_2),
    ]
    weights = {
        1: (m_ne.weight_1, m_ac.weight_1),
        2: (m_ne.weight_2, m_ac.weight_2),
    }

    for ax, (k, comp_ne, comp_ac) in zip(axes, comps):
        p = comp_params[k]
        s_ne = np.asarray(wb_surv(t_grid, comp_ne), dtype=float)
        s_ac = np.asarray(wb_surv(t_grid, comp_ac), dtype=float)
        w_ne, w_ac = weights[k]
        colour = _mode_colours[k]

        ax.plot(t_grid, s_ne, color=_C["neutral"], linewidth=2.2,
                label=f"Non-sour  β={comp_ne.beta:.3f}  η={comp_ne.eta:.0f}d  w={w_ne:.2f}")
        ax.plot(t_grid, s_ac, color=_C["acidic"],  linewidth=2.2,
                label=f"Sour      β={comp_ac.beta:.3f}  η={comp_ac.eta:.0f}d  w={w_ac:.2f}")
        ax.fill_between(t_grid, s_ac, s_ne, alpha=0.12, color=colour,
                        label="H₂S deficit")

        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel("S_k(t)  component survival", fontsize=10)
        ax.set_title(
            f"{_mode_labels[k]} — Sour vs Non-sour\n"
            f"γ_k = Δβ = {p['gamma_k']:+.3f}   β*_k = {p['beta_star_k']:+.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        "Within-mode survival shift due to H₂S (Vt)\n"
        "Weight shift:  Infant w₁  0.44 → 0.33   Wear-out w₂  0.56 → 0.67",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "p8c_component_survival_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, out: Path) -> dict:
    tbl_out = out
    fig_out = out.parent.parent / "figures" / "phase8c_latent_cox"
    tbl_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # ── data prep ─────────────────────────────────────────────────────────────
    sub = _prepare_vt_h2s(df)
    neutral_sub = sub[sub["is_acidic"] == 0]
    acidic_sub  = sub[sub["is_acidic"] == 1]

    n_ac_fail = int(acidic_sub["event"].sum())
    n_ne_fail = int(neutral_sub["event"].sum())

    print(f"\n{'=' * 65}")
    print("PHASE 8c — LATENT WEIBULL + COX  (H₂S, Vt, raw baseline)")
    print(f"{'=' * 65}")
    print(f"  Кислый  (X=1) : {len(acidic_sub)}  ({n_ac_fail} failures)")
    print(f"  Некислый (X=0)  : {len(neutral_sub)} ({n_ne_fail} failures)")

    if n_ac_fail < 10 or n_ne_fail < 10:
        print("  ⚠ Too few failures — skipping phase.")
        return results

    # ── KM fits ───────────────────────────────────────────────────────────────
    km_ne = _fit_km(neutral_sub)
    km_ac = _fit_km(acidic_sub)

    # ── latent Weibull mixtures ───────────────────────────────────────────────
    print("\n  Fitting K=2 Weibull mixtures …")
    m_ne = _fit_mixture(km_ne, "Некислый")
    m_ac = _fit_mixture(km_ac, "Кислый")

    # save component params
    params_rows = [
        _mixture_params_row("Некислый", m_ne),
        _mixture_params_row("Кислый",   m_ac),
    ]
    params_df = pd.DataFrame(params_rows)
    params_df.to_csv(tbl_out / "p8c_mixture_params.csv", index=False, encoding="utf-8-sig")
    results["mixture_params"] = params_df

    # ── time grid ─────────────────────────────────────────────────────────────
    t_max = float(np.quantile(
        np.concatenate([neutral_sub["duration"].values, acidic_sub["duration"].values]),
        0.97,
    ))
    t_grid = np.linspace(1.0, t_max, 300)

    # ── conventional Cox ──────────────────────────────────────────────────────
    cox = _fit_conventional_cox(sub)
    results["conventional_cox"] = {k: v for k, v in cox.items() if not k.startswith("_")}

    # ── Extended Cox on mixture curves ────────────────────────────────────────
    # Build pseudo-KM DataFrames from the mixture survival on the fine grid
    sf_ne_df = _mixture_sf_df(m_ne, t_grid)
    sf_ac_df = _mixture_sf_df(m_ac, t_grid)

    exc_res = fit_extended_cox_log_log(
        km_baseline_sf=sf_ne_df,
        km_treatment_sf=sf_ac_df,
        baseline_weibull_beta=None,   # mixture baseline — skip Weibull correction
        min_time=5.0,
        trim_quantile=0.97,
    )
    results["extended_cox_mixture"] = exc_res.summary_dict()

    # ── comparison table ──────────────────────────────────────────────────────
    rows = [
        {
            "model": "Conventional Cox PH",
            "parameter": "HR (is_acidic)",
            "estimate": round(cox.get("cox_hr", np.nan), 3),
            "ci_lo": round(cox.get("cox_ci_lo", np.nan), 3),
            "ci_hi": round(cox.get("cox_ci_hi", np.nan), 3),
            "p_value": round(cox.get("cox_p", np.nan), 4),
            "note": "Assumes constant HR",
        },
        {
            "model": "Extended Cox — mixture",
            "parameter": "γ  (log-t interaction)",
            "estimate": round(exc_res.gamma, 3),
            "ci_lo": round(exc_res.gamma - 1.96 * exc_res.se_gamma, 3),
            "ci_hi": round(exc_res.gamma + 1.96 * exc_res.se_gamma, 3),
            "p_value": round(exc_res.p_gamma, 4),
            "note": "γ≠0 ⇒ time-varying HR",
        },
        {
            "model": "Extended Cox — mixture",
            "parameter": "β* (intercept, log-scale offset)",
            "estimate": round(exc_res.beta_star, 3),
            "ci_lo": round(exc_res.beta_star - 1.96 * exc_res.se_beta_star, 3),
            "ci_hi": round(exc_res.beta_star + 1.96 * exc_res.se_beta_star, 3),
            "p_value": round(exc_res.p_beta_star, 4),
            "note": "HR(t=1) = exp(β*)",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tbl_out / "p8c_summary.csv", index=False, encoding="utf-8-sig")
    results["summary_df"] = summary_df

    audit_df = pd.DataFrame({
        "time_days": exc_res.times,
        "y_observed": exc_res.y_observed,
        "y_fitted": exc_res.y_fitted,
    })
    audit_df.to_csv(tbl_out / "p8c_regression_points.csv", index=False, encoding="utf-8-sig")

    # ── figures ───────────────────────────────────────────────────────────────
    _plot_mixture_fit(km_ne, km_ac, m_ne, m_ac, fig_out)
    _plot_combined_survival(km_ne, km_ac, m_ne, m_ac, t_grid, fig_out)
    _plot_component_bars(m_ne, m_ac, fig_out)
    _plot_loglog_mixture(m_ne, m_ac, t_grid, fig_out)
    _plot_y_regression_mixture(exc_res, fig_out)
    _plot_tv_hr(
        exc_res,
        cox_hr=cox.get("cox_hr", 1.0),
        cox_ci=(cox.get("cox_ci_lo", 1.0), cox.get("cox_ci_hi", 1.0)),
        out=fig_out,
    )
    _plot_mixture_hazard(m_ne, m_ac, t_grid, fig_out)

    # ── component-specific Extended Cox ──────────────────────────────────────
    comp_params = _component_extended_cox_params(m_ne, m_ac)
    comp_rows = []
    print("\n  Component-specific Extended Cox parameters:")
    print(f"  {'Mode':<14}  {'γ_k=Δβ':>8}  {'β*_k':>8}  {'HR(1d)':>8}  {'Δβ note'}")
    print("  " + "-" * 70)
    for k, p in comp_params.items():
        print(
            f"  {p['mode']:<14}  {p['gamma_k']:>+8.3f}  {p['beta_star_k']:>+8.3f}"
            f"  {p['hr_at_t1']:>8.3f}  "
            f"β: {p['beta_non_sour']:.3f}→{p['beta_sour']:.3f}  "
            f"η: {p['eta_non_sour']:.0f}d→{p['eta_sour']:.0f}d"
        )
        comp_rows.append({
            "mode": p["mode"], "k": k,
            "beta_non_sour": round(p["beta_non_sour"], 4),
            "eta_non_sour":  round(p["eta_non_sour"],  1),
            "beta_sour":     round(p["beta_sour"],      4),
            "eta_sour":      round(p["eta_sour"],       1),
            "gamma_k":       round(p["gamma_k"],        4),
            "beta_star_k":   round(p["beta_star_k"],    4),
            "hr_at_t1":      round(p["hr_at_t1"],       4),
        })
    pd.DataFrame(comp_rows).to_csv(
        tbl_out / "p8c_component_extended_cox.csv", index=False, encoding="utf-8-sig"
    )
    _plot_component_hr(comp_params, m_ne, m_ac, t_grid, fig_out)
    _plot_component_survival_shift(m_ne, m_ac, comp_params, t_grid, fig_out)
    results["component_extended_cox"] = comp_params

    print("\n  Fitting classical (single) Weibull per group …")
    wb_fits = _plot_classical_weibull(km_ne, km_ac, neutral_sub, acidic_sub, t_grid, fig_out)
    results["classical_weibull"] = {
        k: {p: round(v, 4) for p, v in f.items() if isinstance(v, float)}
        for k, f in wb_fits.items()
    }

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'Model':<30}  {'Parameter':<32}  {'Est':>8}  {'95%CI':>20}  {'p':>8}")
    print("  " + "-" * 108)
    for row in rows:
        lo, hi = row["ci_lo"], row["ci_hi"]
        print(
            f"  {row['model']:<30}  {row['parameter']:<32}  "
            f"{row['estimate']:>8.3f}  [{lo:.3f}, {hi:.3f}]  {row['p_value']:>8.4f}"
        )

    print(f"\n  Mixture parameters:")
    print(f"  {'Group':<12}  β₁      η₁      w₁      β₂      η₂      w₂")
    print(f"  " + "-" * 68)
    for row in params_rows:
        g = row["group"]
        print(
            f"  {g:<12}  {row['beta1']:.3f}  {row['eta1_days']:6.1f}d  "
            f"{row['w1']:.3f}  {row['beta2']:.3f}  {row['eta2_days']:6.1f}d  {row['w2']:.3f}"
        )

    gamma = exc_res.gamma
    if exc_res.p_gamma < 0.05:
        direction = "decreasing" if gamma < 0 else "increasing"
        print(f"\n  γ={gamma:+.4f} (p={exc_res.p_gamma:.4f}): H₂S effect is {direction} over time.")
    else:
        print(f"\n  γ={gamma:+.4f} (p={exc_res.p_gamma:.4f}): no significant time interaction.")

    ph_ok = cox.get("cox_ph_ok")
    if ph_ok is False:
        print(f"  Schoenfeld p={cox.get('cox_ph_p', np.nan):.3f}: PH assumption REJECTED.")
    elif ph_ok is True:
        print(f"  Schoenfeld p={cox.get('cox_ph_p', np.nan):.3f}: PH assumption not rejected.")

    print(f"\n  Extended Cox on mixture: n_grid={exc_res.n_points}  R²={exc_res.r_squared:.4f}")
    print(f"  Outputs → {tbl_out}  &  {fig_out}")
    print(f"{'=' * 65}\n")

    return results
