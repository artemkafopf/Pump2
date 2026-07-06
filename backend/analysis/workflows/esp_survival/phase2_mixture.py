"""Phase 2 — K=2 latent Weibull mixture per stratum via EM on raw data.

Strategy:
  ≥ 40 observed failures  → independent K=2 EM fit (5 free params)
  20–39 observed failures → two-stage: global (β₁, β₂) fixed; fit (w₁, η₁, η₂)
  < 20 observed failures  → skip (single Weibull from Phase 1 is used)

Physical constraints (enforced inside weibull_em):
  C1 (early failure): β₁ > 0 (free — no upper bound required)
  C2 (wear-out):      β₂ > 1.0  (hard)
  Scale separation:   η₂ ≥ 2.0 · η₁  (hard)
  Weight:             w₁ < 0.75 preferred but soft (degenerate flag only)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_survival,
    latent_life_quantile,
)
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.models.survival.weibull_model import fit_basic_weibull
from .data import MIN_FAILURES_INDEPENDENT, MIN_FAILURES_TWO_STAGE, stratum_summary


def _save_models(
    models: dict[str, TwoComponentLatentWeibullModel],
    global_beta1: float,
    global_beta2: float,
    two_stage_strata: list[str],
    out_dir: Path,
) -> None:
    payload = {
        "global_beta1": global_beta1,
        "global_beta2": global_beta2,
        "two_stage_strata": two_stage_strata,
        "models": {
            stratum: {
                "weight_1": m.weight_1,
                "beta1": m.component_1.beta,
                "eta1": m.component_1.eta,
                "label1": m.component_1.label,
                "beta2": m.component_2.beta,
                "eta2": m.component_2.eta,
                "label2": m.component_2.label,
            }
            for stratum, m in models.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2_models.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_models(json_path: Path) -> tuple[
    dict[str, TwoComponentLatentWeibullModel], float, float, list[str]
]:
    """Deserialize Phase 2 models from a saved JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    models: dict[str, TwoComponentLatentWeibullModel] = {}
    for stratum, d in data["models"].items():
        models[stratum] = TwoComponentLatentWeibullModel(
            weight_1=d["weight_1"],
            component_1=WeibullParameters(
                beta=d["beta1"], eta=d["eta1"], label=d.get("label1", "C1_early")
            ),
            component_2=WeibullParameters(
                beta=d["beta2"], eta=d["eta2"], label=d.get("label2", "C2_wearout")
            ),
        )
    return (
        models,
        float(data["global_beta1"]),
        float(data["global_beta2"]),
        list(data["two_stage_strata"]),
    )


# Default initial models — bias only; EM uses Halton for the other starts
_DEFAULT_INIT = TwoComponentLatentWeibullModel(
    weight_1=0.35,
    component_1=WeibullParameters(beta=0.75, eta=60.0, label="C1_early"),
    component_2=WeibullParameters(beta=2.00, eta=300.0, label="C2_wearout"),
)
_SOUR_INIT = TwoComponentLatentWeibullModel(
    weight_1=0.33,
    component_1=WeibullParameters(beta=0.90, eta=15.0, label="C1_early"),
    component_2=WeibullParameters(beta=1.80, eta=120.0, label="C2_wearout"),
)

NUM_STARTS = 12
MIN_ETA_RATIO = 2.0


def _derive_global_shapes(
    independent_fits: dict[str, tuple["LatentWeibullCurveFitResult", bool]],  # noqa: F821
) -> tuple[float, float]:
    """Derive (β₁_global, β₂_global) as the median of non-degenerate independent fits.

    Excludes fits flagged DEGENERATE (w₁>0.75) or with implausibly high β₂ (≥8).
    Falls back to domain-knowledge defaults if too few fits qualify.
    """
    beta1_vals: list[float] = []
    beta2_vals: list[float] = []
    for result, is_degenerate in independent_fits.values():
        if is_degenerate:
            continue
        b1 = result.model.component_1.beta
        b2 = result.model.component_2.beta
        if b2 >= 8.0:  # likely boundary artefact
            continue
        beta1_vals.append(b1)
        beta2_vals.append(b2)
    if len(beta2_vals) >= 2:
        return float(np.median(beta1_vals)), float(np.median(beta2_vals))
    # Fallback: domain knowledge for oilfield ESPs
    return 0.90, 1.60


def _fit_stratum(
    durations: np.ndarray,
    events: np.ndarray,
    stratum: str,
    global_beta1: float | None = None,
    global_beta2: float | None = None,
) -> "LatentWeibullCurveFitResult":  # noqa: F821
    """EM fit for one stratum. Two-stage mode when global shapes are provided."""
    is_sour = "sour" in stratum
    init_model = _SOUR_INIT if is_sour else _DEFAULT_INIT
    return fit_latent_weibull_em(
        durations, events,
        initial_model=init_model,
        num_starts=NUM_STARTS,
        min_eta_ratio=MIN_ETA_RATIO,
        fix_beta1=global_beta1,
        fix_beta2=global_beta2,
    )


def _model_to_row(
    stratum: str,
    g: pd.DataFrame,
    model: TwoComponentLatentWeibullModel,
    result: "LatentWeibullCurveFitResult",  # noqa: F821
    fit_mode: str,
) -> dict:
    n_fail = int(g["event"].sum())
    n_obs = len(g)
    n_params = 5 if "independent" in fit_mode else 3  # free params for AIC
    nll_k2 = float(result.objective_value)
    aic = aic_k2 = 2.0 * n_params + 2.0 * nll_k2

    # ── T3: K=1 (single Weibull) vs K=2 (mixture) via ΔAIC ───────────────────
    # RMSE-to-KM is not a proper score under censoring; ΔAIC is the principled
    # K=1 vs K=2 criterion (both NLLs use the same censored likelihood form).
    try:
        k1 = fit_basic_weibull(
            g["tte"].to_numpy(dtype=float), g["event"].to_numpy(dtype=int)
        )
        nll_k1 = float(k1["nll"])
        aic_k1 = float(k1["aic"])  # 2·2 + 2·nll_k1
    except Exception:
        nll_k1 = float("nan")
        aic_k1 = float("nan")
    delta_aic = aic_k1 - aic_k2  # > 0 ⇒ mixture (K=2) preferred

    b10 = latent_life_quantile(0.10, model)
    b50 = latent_life_quantile(0.50, model)
    eta_ratio = model.component_2.eta / max(model.component_1.eta, 1e-9)
    return {
        "stratum": stratum,
        "fit_mode": fit_mode,
        "n_obs": n_obs,
        "n_failures": n_fail,
        "w1": round(model.weight_1, 4),
        "w2": round(model.weight_2, 4),
        "beta1": round(model.component_1.beta, 4),
        "eta1_days": round(model.component_1.eta, 1),
        "beta2": round(model.component_2.beta, 4),
        "eta2_days": round(model.component_2.eta, 1),
        "eta_ratio": round(eta_ratio, 2),
        "B10_days": round(b10, 1) if np.isfinite(b10) else None,
        "B50_days": round(b50, 1) if np.isfinite(b50) else None,
        "nll": round(result.objective_value, 2),
        "aic": round(aic, 1),
        "nll_k1": round(nll_k1, 2) if np.isfinite(nll_k1) else None,
        "nll_k2": round(nll_k2, 2),
        "aic_k1": round(aic_k1, 1) if np.isfinite(aic_k1) else None,
        "aic_k2": round(aic_k2, 1),
        "delta_aic": round(delta_aic, 1) if np.isfinite(delta_aic) else None,
        "mixture_wins_aic": bool(delta_aic > 0) if np.isfinite(delta_aic) else None,
        "rmse": round(result.rmse, 6),
        "weighted_rmse": round(result.weighted_rmse, 6),
        "n_starts": result.n_starts,
        "converged": result.success,
        "degenerate": "DEGENERATE" in result.message,
        "message": result.message,
        "beta2_gt1": model.component_2.beta > 1.0,
    }


def _plot_stratum(
    kmf: KaplanMeierFitter,
    model: TwoComponentLatentWeibullModel,
    stratum: str,
    ax: plt.Axes,
) -> None:
    t_max = max(
        float(kmf.survival_function_.index.max()) * 1.1,
        float(model.component_2.eta) * 2.5,
    )
    t_grid = np.linspace(1e-3, t_max, 400)

    mix_s = np.asarray(latent_survival(t_grid, model), dtype=float)
    c1_s = np.exp(-(t_grid / model.component_1.eta) ** model.component_1.beta)
    c2_s = np.exp(-(t_grid / model.component_2.eta) ** model.component_2.beta)

    # KM step with CI
    try:
        ci = kmf.confidence_interval_.reset_index()
        ci.columns = ["time", "lo", "hi"]
        ci = ci[ci["time"] > 0]
        ax.fill_between(ci["time"], ci["lo"], ci["hi"], alpha=0.15, color="steelblue")
    except Exception:
        pass

    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]
    sf = sf[sf["time"] > 0]
    ax.step(sf["time"], sf["survival"], where="post", color="steelblue", lw=2, label="KM")

    ax.plot(t_grid, mix_s, "k-", lw=2.0,
            label=f"Mixture w₁={model.weight_1:.2f}")
    ax.plot(t_grid, model.weight_1 * c1_s, "--", color="crimson", lw=1.2,
            label=f"C1: β={model.component_1.beta:.2f} η={model.component_1.eta:.0f}d")
    ax.plot(t_grid, model.weight_2 * c2_s, "--", color="darkorange", lw=1.2,
            label=f"C2: β={model.component_2.beta:.2f} η={model.component_2.eta:.0f}d")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival")
    ax.set_title(stratum, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)


def run(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    preset_global_beta1: float | None = None,
    preset_global_beta2: float | None = None,
) -> dict:
    """Fit K=2 EM per stratum.

    ``preset_global_beta1/2``: skip shape derivation and use these values directly
    for all two-stage strata.  Useful when the caller already knows the global shapes
    from a parent analysis (e.g. M3 shapes for Vt_sour contractor sub-strata).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = stratum_summary(df)
    independent = summary[summary["n_failures"] >= MIN_FAILURES_INDEPENDENT]["stratum"].tolist()
    two_stage = summary[
        (summary["n_failures"] >= MIN_FAILURES_TWO_STAGE)
        & (summary["n_failures"] < MIN_FAILURES_INDEPENDENT)
    ]["stratum"].tolist()

    print(f"\n[Phase 2] Independent K=2 strata ({len(independent)}): {independent}")
    print(f"\n[Phase 2] Two-stage strata ({len(two_stage)}): {two_stage}")

    results: list[dict] = []
    models: dict[str, TwoComponentLatentWeibullModel] = {}
    kmf_cache: dict[str, KaplanMeierFitter] = {}

    all_viable = independent + two_stage
    ncols = min(3, len(all_viable))
    nrows = int(np.ceil(len(all_viable) / ncols)) if all_viable else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    # ── Pass 1: fit independent strata ────────────────────────────────────────
    print("\n[Phase 2] Pass 1 — independent fits:")
    independent_fit_results: dict[str, tuple] = {}  # stratum → (result, is_degen)

    for idx, stratum in enumerate(independent):
        g = df[df["stratum"] == stratum]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations, events)
        kmf_cache[stratum] = kmf

        try:
            fit_result = _fit_stratum(durations, events, stratum)
            model = fit_result.model
            models[stratum] = model
            row = _model_to_row(stratum, g, model, fit_result, "independent_K2")
            results.append(row)
            independent_fit_results[stratum] = (fit_result, row["degenerate"])

            degen_flag = " [DEGENERATE]" if row["degenerate"] else ""
            print(
                f"  {stratum}: "
                f"w₁={model.weight_1:.3f}  "
                f"β₁={model.component_1.beta:.3f} η₁={model.component_1.eta:.0f}d  "
                f"β₂={model.component_2.beta:.3f} η₂={model.component_2.eta:.0f}d  "
                f"η₂/η₁={row['eta_ratio']:.1f}  "
                f"NLL={fit_result.objective_value:.1f}  "
                f"RMSE={fit_result.rmse:.5f}"
                + (" ✓" if model.component_2.beta > 1.0 else " ✗ β₂≤1")
                + degen_flag
            )
            _plot_stratum(kmf, model, stratum, axes_flat[idx])

        except Exception as exc:
            print(f"  {stratum}: FAILED — {exc}")
            axes_flat[idx].set_visible(False)

    # ── Derive global shapes from non-degenerate independent fits ─────────────
    if preset_global_beta1 is not None and preset_global_beta2 is not None:
        global_beta1, global_beta2 = preset_global_beta1, preset_global_beta2
        print(
            f"\n[Phase 2] Global shapes (caller-supplied): "
            f"β₁={global_beta1:.3f}  β₂={global_beta2:.3f}"
        )
    else:
        global_beta1, global_beta2 = _derive_global_shapes(independent_fit_results)
        print(
            f"\n[Phase 2] Global shapes (median of non-degenerate independent fits): "
            f"β₁={global_beta1:.3f}  β₂={global_beta2:.3f}"
        )

    # ── Pass 2: two-stage strata ──────────────────────────────────────────────
    if two_stage:
        print("\n[Phase 2] Pass 2 — two-stage fits (β₁, β₂ fixed):")
    for i, stratum in enumerate(two_stage):
        idx = len(independent) + i
        g = df[df["stratum"] == stratum]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations, events)
        kmf_cache[stratum] = kmf

        try:
            fit_result = _fit_stratum(
                durations, events, stratum,
                global_beta1=global_beta1,
                global_beta2=global_beta2,
            )
            model = fit_result.model
            models[stratum] = model
            row = _model_to_row(stratum, g, model, fit_result, "two_stage_K2")
            results.append(row)

            degen_flag = " [DEGENERATE]" if row["degenerate"] else ""
            print(
                f"  {stratum}: "
                f"w₁={model.weight_1:.3f}  "
                f"η₁={model.component_1.eta:.0f}d  "
                f"η₂={model.component_2.eta:.0f}d  "
                f"η₂/η₁={row['eta_ratio']:.1f}  "
                f"NLL={fit_result.objective_value:.1f}  "
                f"RMSE={fit_result.rmse:.5f}"
                + degen_flag
            )
            _plot_stratum(kmf, model, stratum, axes_flat[idx])

        except Exception as exc:
            print(f"  {stratum}: FAILED — {exc}")
            axes_flat[idx].set_visible(False)

    for idx in range(len(all_viable), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Phase 2 — K=2 Latent Weibull Mixture (EM on raw data)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "phase2_mixture_grid.png", dpi=150)
    plt.close(fig)

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "phase2_mixture_params.csv", index=False, encoding="utf-8-sig")

    _save_models(models, global_beta1, global_beta2, two_stage, out_dir.parent / "models")

    n_ok = sum(1 for r in results if r.get("beta2_gt1", False))
    n_degen = sum(1 for r in results if r.get("degenerate", False))
    print(
        f"\n[Phase 2] Complete. β₂>1: {n_ok}/{len(results)} strata. "
        f"Degenerate (w₁>0.75): {n_degen}/{len(results)}."
    )

    return {
        "results": result_df,
        "models": models,
        "kmf_cache": kmf_cache,
        "global_beta1": global_beta1,
        "global_beta2": global_beta2,
        "two_stage_strata": two_stage,
    }
