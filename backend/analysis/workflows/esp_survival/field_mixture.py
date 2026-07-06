"""Field-level K=2 latent Weibull mixture — no sour/non-sour split.

Fits the same EM estimator as phase2_mixture.py, but groups data by
`field_clean` rather than `stratum`.  Useful for:
  - Comparing field-level pooled fit to the stratified (sour/non-sour) result
  - Understanding whether the H2S split is necessary for the mixture shape
  - Providing a simpler model for fields where sour wells are rare

Stratification thresholds are the same as Phase 2:
  ≥ 40 failures  → independent K=2 EM (5 free params)
  20–39 failures → two-stage: global (β₁, β₂) fixed
  < 20 failures  → skip

Outputs
-------
results/esp_survival_field_mixture/<date>/
  figures/field_mixture_grid.png
  figures/field_mixture_params.csv
  models/field_models.json
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
from .data import MIN_FAILURES_INDEPENDENT, MIN_FAILURES_TWO_STAGE
from .phase2_mixture import (
    _DEFAULT_INIT,
    _SOUR_INIT,
    _derive_global_shapes,
    _model_to_row,
    _plot_stratum,
    _save_models,
    load_models,
    NUM_STARTS,
    MIN_ETA_RATIO,
)


def _field_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, g in df.groupby("field_clean"):
        n_fail = int(g["event"].sum())
        b50 = g.loc[g["event"] == 1, "tte"].median()
        if n_fail >= MIN_FAILURES_INDEPENDENT:
            fit_mode = "independent_K2"
        elif n_fail >= MIN_FAILURES_TWO_STAGE:
            fit_mode = "two_stage_K2"
        else:
            fit_mode = "skip"
        sour_frac = float(g.loc[g["event"] == 1, "h2s_class"].eq("sour").mean())
        rows.append({
            "field": field,
            "n_total": len(g),
            "n_failures": n_fail,
            "sour_failure_frac": round(sour_frac, 3),
            "empirical_B50_days": round(b50, 1) if not np.isnan(b50) else None,
            "fit_mode": fit_mode,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("n_failures", ascending=False)
        .reset_index(drop=True)
    )


def _fit_field(
    durations: np.ndarray,
    events: np.ndarray,
    field: str,
    global_beta1: float | None = None,
    global_beta2: float | None = None,
) -> "LatentWeibullCurveFitResult":  # noqa: F821
    # Sour-dominant fields (Vt) may need a different initial model
    sour_frac = 0.0  # unknown at this point; use default
    init_model = _DEFAULT_INIT
    return fit_latent_weibull_em(
        durations, events,
        initial_model=init_model,
        num_starts=NUM_STARTS,
        min_eta_ratio=MIN_ETA_RATIO,
        fix_beta1=global_beta1,
        fix_beta2=global_beta2,
    )


def run(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _field_summary(df)
    independent = summary[summary["fit_mode"] == "independent_K2"]["field"].tolist()
    two_stage = summary[summary["fit_mode"] == "two_stage_K2"]["field"].tolist()
    skip = summary[summary["fit_mode"] == "skip"]["field"].tolist()

    print(f"\n[Field] Independent K=2 fields ({len(independent)}): {independent}")
    print(f"[Field] Two-stage fields ({len(two_stage)}): {two_stage}")
    if skip:
        print(f"[Field] Skipped ({len(skip)}): {skip}")

    results: list[dict] = []
    models: dict[str, TwoComponentLatentWeibullModel] = {}
    kmf_cache: dict[str, KaplanMeierFitter] = {}

    all_viable = independent + two_stage
    ncols = min(3, len(all_viable))
    nrows = int(np.ceil(len(all_viable) / ncols)) if all_viable else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    # ── Pass 1: independent fields ────────────────────────────────────────────
    print("\n[Field] Pass 1 — independent fits:")
    independent_fit_results: dict[str, tuple] = {}

    for idx, field in enumerate(independent):
        g = df[df["field_clean"] == field]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations, events)
        kmf_cache[field] = kmf

        try:
            fit_result = _fit_field(durations, events, field)
            model = fit_result.model
            models[field] = model

            row = _model_to_row(field, g, model, fit_result, "independent_K2")
            results.append(row)
            independent_fit_results[field] = (fit_result, row["degenerate"])

            degen_flag = " [DEGENERATE]" if row["degenerate"] else ""
            sour_pct = float(g.loc[g["event"] == 1, "h2s_class"].eq("sour").mean()) * 100
            print(
                f"  {field}: "
                f"w₁={model.weight_1:.3f}  "
                f"β₁={model.component_1.beta:.3f} η₁={model.component_1.eta:.0f}d  "
                f"β₂={model.component_2.beta:.3f} η₂={model.component_2.eta:.0f}d  "
                f"η₂/η₁={row['eta_ratio']:.1f}  "
                f"NLL={fit_result.objective_value:.1f}  "
                f"RMSE={fit_result.rmse:.5f}  "
                f"[sour={sour_pct:.0f}%]"
                + (" ✓" if model.component_2.beta > 1.0 else " ✗ β₂≤1")
                + degen_flag
            )
            _plot_stratum(kmf, model, field, axes_flat[idx])

        except Exception as exc:
            print(f"  {field}: FAILED — {exc}")
            axes_flat[idx].set_visible(False)

    # ── Global shapes ─────────────────────────────────────────────────────────
    global_beta1, global_beta2 = _derive_global_shapes(independent_fit_results)
    print(
        f"\n[Field] Global shapes (median of non-degenerate independent fits): "
        f"β₁={global_beta1:.3f}  β₂={global_beta2:.3f}"
    )

    # ── Pass 2: two-stage fields ───────────────────────────────────────────────
    if two_stage:
        print("\n[Field] Pass 2 — two-stage fits (β₁, β₂ fixed):")
    for i, field in enumerate(two_stage):
        idx = len(independent) + i
        g = df[df["field_clean"] == field]
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations, events)
        kmf_cache[field] = kmf

        try:
            fit_result = _fit_field(
                durations, events, field,
                global_beta1=global_beta1,
                global_beta2=global_beta2,
            )
            model = fit_result.model
            models[field] = model

            row = _model_to_row(field, g, model, fit_result, "two_stage_K2")
            results.append(row)

            degen_flag = " [DEGENERATE]" if row["degenerate"] else ""
            print(
                f"  {field}: "
                f"w₁={model.weight_1:.3f}  "
                f"η₁={model.component_1.eta:.0f}d  "
                f"η₂={model.component_2.eta:.0f}d  "
                f"η₂/η₁={row['eta_ratio']:.1f}  "
                f"NLL={fit_result.objective_value:.1f}  "
                f"RMSE={fit_result.rmse:.5f}"
                + degen_flag
            )
            _plot_stratum(kmf, model, field, axes_flat[idx])

        except Exception as exc:
            print(f"  {field}: FAILED — {exc}")
            axes_flat[idx].set_visible(False)

    for idx in range(len(all_viable), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Field-level K=2 Latent Weibull Mixture (no sour/non-sour split)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "field_mixture_grid.png", dpi=150)
    plt.close(fig)

    # ── Comparison plot: field vs stratum β₂ and η₁ ──────────────────────────
    _plot_comparison(results, df, out_dir)

    result_df = pd.DataFrame(results)
    result_df = result_df.rename(columns={"stratum": "field"})
    result_df.to_csv(out_dir / "field_mixture_params.csv", index=False, encoding="utf-8-sig")

    _save_models(models, global_beta1, global_beta2, two_stage, out_dir.parent / "models")

    n_ok = sum(1 for r in results if r.get("beta2_gt1", False))
    n_degen = sum(1 for r in results if r.get("degenerate", False))
    print(
        f"\n[Field] Complete. β₂>1: {n_ok}/{len(results)} fields. "
        f"Degenerate (w₁>0.75): {n_degen}/{len(results)}."
    )

    return {
        "results": result_df,
        "models": models,
        "kmf_cache": kmf_cache,
        "global_beta1": global_beta1,
        "global_beta2": global_beta2,
        "two_stage_fields": two_stage,
        "summary": summary,
    }


def _plot_comparison(field_results: list[dict], df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side comparison: field-level vs stratum-level β₂, η₁, w₁.

    Loads Phase 2 stratum results from the most recent results directory.
    """
    from analysis.paths import RESULTS_ROOT
    # Try to load Phase 2 stratum-level results for comparison
    stratum_df: pd.DataFrame | None = None
    phase2_root = RESULTS_ROOT / "esp_survival_phase2_mixture"
    if phase2_root.exists():
        dated = sorted(phase2_root.glob("????-??-??"), reverse=True)
        for d in dated:
            csv = d / "figures" / "phase2_mixture_params.csv"
            if csv.exists():
                stratum_df = pd.read_csv(csv)
                # Map stratum → field by stripping _sour/_nonsour suffix
                stratum_df["field"] = stratum_df["stratum"].str.replace(
                    r"_(sour|nonsour)$", "", regex=True
                )
                break

    field_df = pd.DataFrame(field_results).rename(columns={"stratum": "field"})
    fields = field_df["field"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = [
        ("beta2", "β₂ (wear-out shape)"),
        ("eta1_days", "η₁ (early-failure scale, d)"),
        ("w1", "w₁ (early-failure fraction)"),
    ]

    x = np.arange(len(fields))
    width = 0.35

    for ax, (col, label) in zip(axes, params):
        field_vals = field_df.set_index("field")[col].reindex(fields).fillna(0).tolist()
        bars1 = ax.bar(x - width / 2, field_vals, width, label="Field (pooled)", color="steelblue", alpha=0.8)

        if stratum_df is not None:
            # For each field, compute weighted-average of stratum params
            wavg = []
            for field in fields:
                sub = stratum_df[stratum_df["field"] == field]
                if sub.empty:
                    wavg.append(float("nan"))
                    continue
                weights = sub["n_failures"].astype(float)
                wavg.append(float(np.average(sub[col], weights=weights)) if weights.sum() > 0 else float("nan"))
            bars2 = ax.bar(x + width / 2, wavg, width, label="Stratum (n-wtd avg)", color="darkorange", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(fields, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("Field-level vs stratum-level mixture parameters", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "field_vs_stratum_comparison.png", dpi=150)
    plt.close(fig)
