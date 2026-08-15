"""End-to-end run: build the panel, score every realization, impute, propagate into β.

Writes to ``results/intake_pressure_reconstruction/<date>/``.  Nothing here holds analysis
logic — the physics is in :mod:`analysis.features.intake_pressure`, the ML in
:mod:`analysis.models.ml.intake_pressure_ml`, the scoring in
:mod:`analysis.workflows.intake_pressure.validate`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.features.intake_pressure import PhysicalIntakeModel, WellMedianBaseline
from analysis.models.ml.intake_pressure_ml import IntakeMLModel
from analysis.paths import results_dir
from analysis.workflows.intake_pressure import validate as V
from analysis.workflows.intake_pressure.data import build_panel, gap_profile
from analysis.workflows.intake_pressure.gas_at_intake import (
    attach_beta,
    beta_error_profile,
    beta_rank_stability,
    pvt_sweep_beta,
)

SLUG = "intake_pressure_reconstruction"


def _chosen_model(name: str):
    if name == "well_median":
        return WellMedianBaseline()
    if name == "phys_linear":
        return PhysicalIntakeModel(mode="linear")
    if name == "phys_vogel":
        return PhysicalIntakeModel(mode="vogel")
    variant = {"cb_plain": "plain", "cb_physics": "physics", "cb_residual": "residual"}[name]
    return IntakeMLModel(variant=variant, use_history=True)


def run(
    db_path=None,
    *,
    include_leak_demo: bool = True,
    run_pvt_sweep: bool = True,
    score: bool = True,
) -> dict:
    """Score, impute with the two headline realizations, and report β impact.

    Both realizations are written to the output, not just the winner: the whole point of
    two independent reconstructions is that their **disagreement** is a usable uncertainty
    signal, and collapsing to one number throws that away.  ``p_intake_spread`` in the
    imputed table is that disagreement per row.

    ``score=False`` reuses the ``model_scores.csv`` already in today's output directory
    instead of recomputing it.  Scoring dominates the runtime (~20 min for six models over
    warm and cold folds) and does not change when only the imputation or its output schema
    does, so this exists to keep those iterations cheap.  It raises if no scores are there
    to reuse — silently shipping a summary with no validation behind it would be worse
    than failing.
    """
    out = results_dir(SLUG)
    tab, fig, rep = out / "tables", out / "figures", out / "reports"

    panel = build_panel(db_path)
    gp = gap_profile(panel)
    gp.to_csv(tab / "well_gap_profile.csv", index=False)

    scores_path = tab / "model_scores.csv"
    if score:
        scores = V.evaluate(panel, include_leak_demo=include_leak_demo)
        scores.to_csv(scores_path, index=False)
    else:
        if not scores_path.exists():
            raise FileNotFoundError(
                f"score=False needs an existing {scores_path}; run once with score=True"
            )
        scores = pd.read_csv(scores_path)

    # ── the two independent realizations, fit on every observed day ──────────
    # variant="plain": measured best in BOTH regimes, and 23 % better than the physics
    # hybrids on cold wells, where those inherit the physical model's field-pooled
    # extrapolation.  See analysis.models.ml.intake_pressure_ml.
    phys = PhysicalIntakeModel(mode="linear").fit(panel)
    ml = IntakeMLModel(variant="plain", use_history=True).fit(panel)
    phys.calibration_table().to_csv(tab / "physical_calibration.csv", index=False)
    ml.importances().rename("importance").to_csv(tab / "ml_feature_importance.csv")

    panel["p_intake_phys"] = phys.predict(panel)
    panel["p_intake_ml"] = ml.predict(panel)
    # Named for the physical arm because that is where it is computed, but it is the
    # warm/cold flag for BOTH: a well with no per-well physical calibration is exactly a
    # well with no observed history for the ML arm either.  Anything other than "well" is
    # an extrapolation, and cold-regime error is ~1.2x the warm one.
    panel["phys_calib_tier"] = phys.predict_source(panel)
    panel["p_intake_spread"] = (panel["p_intake_phys"] - panel["p_intake_ml"]).abs()
    # Deployable column: measurement wherever it exists, ML elsewhere, physics as the
    # last resort where the ML row lacks its inputs.
    panel["p_intake_final"] = panel["rpump_intake"]
    fill = panel["p_intake_final"].isna()
    panel.loc[fill, "p_intake_final"] = (
        panel.loc[fill, "p_intake_ml"].fillna(panel.loc[fill, "p_intake_phys"])
    )
    panel["p_intake_imputed"] = fill & panel["p_intake_final"].notna()

    # ── β, and the honest end-to-end error in β ─────────────────────────────
    panel = attach_beta(panel, "p_intake_final", out_col="beta_final")

    # The β error must be measured out-of-sample.  ``ml`` above saw every observed day,
    # so scoring it on held-out rows would be scoring it on its own training data and the
    # β error would come back flattering and meaningless.  Refit on the warm-training ends
    # only, then predict the hidden block.
    hide = V.warm_split(panel)
    ml_oos = IntakeMLModel(variant="plain", use_history=True).fit(
        panel[panel["is_observed"] & ~hide]
    )
    held = panel[hide].copy()
    held["p_intake_hat"] = ml_oos.predict(held)
    prof = beta_error_profile(held)
    prof.to_csv(tab / "beta_error_profile.csv", index=False)
    rank = beta_rank_stability(held)

    if run_pvt_sweep:
        pvt_sweep_beta(panel, "p_intake_final").to_csv(tab / "beta_pvt_sweep.csv", index=False)

    keep = ["well_key", "field", "dt", "rpump_intake", "p_intake_phys", "p_intake_ml",
            "p_intake_final", "p_intake_imputed", "p_intake_spread", "phys_calib_tier",
            "beta_final"]
    panel[keep].to_parquet(tab / "daily_intake_imputed.parquet", index=False)

    summary = {
        "panel_rows": int(len(panel)),
        "wells": int(panel["well_key"].nunique()),
        "observed_share": float(panel["is_observed"].mean()),
        "gap_rows": int((~panel["is_observed"]).sum()),
        "gap_rows_imputed": int(panel["p_intake_imputed"].sum()),
        "gap_imputed_share": float(
            panel.loc[~panel["is_observed"], "p_intake_final"].notna().mean()
        ),
        "median_realization_spread_atm": float(panel["p_intake_spread"].median()),
        "beta_rank_stability": rank,
        "cold_wells": gp.loc[gp["n_observed"] == 0, "well_key"].tolist(),
    }
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    _plot(held, prof, scores, fig)
    return {"out_dir": out, "scores": scores, "summary": summary, "beta_profile": prof}


def _plot(held, prof, scores, fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].scatter(held["rpump_intake"], held["p_intake_hat"], s=1, alpha=0.02,
                  color="#3b6ea5")
    lim = [0, 250]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="observed Рприем, atm",
              ylabel="ML reconstruction, atm",
              title="Held-out blocks (out-of-sample)")

    w = scores[scores["regime"] == "warm"].sort_values("mae")
    c = scores[scores["regime"] == "cold"].set_index("model")["mae"]
    y = np.arange(len(w))
    ax[1].barh(y - 0.2, w["mae"], height=0.4, label="warm (masked block)", color="#3b6ea5")
    ax[1].barh(y + 0.2, [c.get(m, np.nan) for m in w["model"]], height=0.4,
               label="cold (unseen well)", color="#c46a34")
    ax[1].set(yticks=y, yticklabels=w["model"], xlabel="MAE, atm",
              title="Error by regime")
    ax[1].legend(fontsize=8)
    ax[1].invert_yaxis()

    if len(prof):
        ax[2].bar(range(len(prof)), prof["beta_mae"], color="#4a7c59")
        ax[2].set(xticks=range(len(prof)),
                  xticklabels=[str(b) for b in prof["band"]],
                  ylabel="|Δβ|", xlabel="true Рприем band, atm",
                  title="β error induced by the reconstruction")
        ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "intake_reconstruction.png", dpi=140)
    plt.close(fig)


__all__ = ["run", "SLUG"]
