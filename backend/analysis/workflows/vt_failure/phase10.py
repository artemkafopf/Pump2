"""Phase 10 — CatBoost Infant Mortality Classifier."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from .config import INFANT_THRESHOLD_PRIMARY, MIN_FAILURES_CIF, VT_FIELD


INFANT_THR = INFANT_THRESHOLD_PRIMARY  # 90 days


def _prepare_catboost_data(df: pd.DataFrame, threshold: int = INFANT_THR) -> tuple[pd.DataFrame, pd.Series] | None:
    """
    Build feature matrix X and binary label y for infant mortality classification.

    Target: infant failure = event==1 AND duration < threshold.
    Reference class: runs where duration >= threshold (mature failures or long-censored).
    Exclude: censored runs with duration < threshold (outcome unknown).
    """
    # Apply denominator rule from plan
    eligible = df[(df["duration"] >= threshold) | (df["event"] == 1)].copy()
    n_excluded = len(df) - len(eligible)
    print(f"  CatBoost: {len(eligible)} eligible runs ({n_excluded} excluded — censored before {threshold}d)")

    eligible["is_infant"] = ((eligible["event"] == 1) & (eligible["duration"] < threshold)).astype(int)
    n_infant = int(eligible["is_infant"].sum())
    n_ref = int((eligible["is_infant"] == 0).sum())
    print(f"  Infant failures: {n_infant}  |  Reference class: {n_ref}")

    if n_infant < MIN_FAILURES_CIF:
        print(f"  n_infant={n_infant} < {MIN_FAILURES_CIF}, skipping CatBoost.")
        return None

    # Feature selection (NO post-outcome variables: TRF, TLF, TTF, failure_category)
    feature_cols = [
        "freq_signed_exposure", "freq_w_mean", "freq_above_55hz_pct",
        "field",
        "h2s_proxy_mg_l",
        "avg_glf", "avg_kpod",
        "mount_year",
        "contractor",
        "nominal_freq_hz",
        "pbubble_atm",
    ]
    # Ion proxies if coverage >= 60%
    for ion_col in ["cum_calcium_load_kg", "cum_chloride_load_kg", "cum_sulfate_load_kg", "cum_gypsum_scale_proxy"]:
        if ion_col in eligible.columns and eligible[ion_col].notna().mean() >= 0.6:
            feature_cols.append(ion_col)

    # Only keep available columns
    feature_cols = [c for c in feature_cols if c in eligible.columns]
    cat_features = [c for c in ["field", "contractor"] if c in feature_cols]

    X = eligible[feature_cols].copy()
    y = eligible["is_infant"].copy()

    # Fill NaN for cat features
    for c in cat_features:
        X[c] = X[c].fillna("<missing>").astype(str)

    return X, y, cat_features


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("\n" + "=" * 60)
    print("PHASE 10 — CATBOOST INFANT MORTALITY CLASSIFIER")
    print("=" * 60)

    prepared = _prepare_catboost_data(df, threshold=INFANT_THR)
    if prepared is None:
        return results
    X, y, cat_features = prepared

    # ------------------------------------------------------------------
    # 5-fold stratified CV (stratify by field + infant label)
    # ------------------------------------------------------------------
    strat_labels = df.loc[X.index, "field"].fillna("unknown").astype(str) + "_" + y.astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    all_preds = np.zeros(len(y))
    all_true  = y.values.copy()

    cat_idx = [list(X.columns).index(c) for c in cat_features]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42 + fold,
            verbose=0,
            class_weights={0: 1.0, 1: max(1.0, (y_tr == 0).sum() / (y_tr == 1).sum() + 0.1)},
        )
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool   = Pool(X_val, y_val, cat_features=cat_idx)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)

        val_preds = model.predict_proba(val_pool)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        auc_scores.append(fold_auc)
        all_preds[val_idx] = val_preds
        print(f"  Fold {fold+1}/5 — AUC={fold_auc:.4f}")

    mean_auc = np.mean(auc_scores)
    std_auc  = np.std(auc_scores)
    results["cv_auc_mean"] = mean_auc
    results["cv_auc_std"]  = std_auc
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#D65F5F", linewidth=2, label=f"OOF ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Phase 10 — CatBoost infant mortality (threshold={INFANT_THR}d)\n5-fold stratified CV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p10_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final model on full data for SHAP/importance
    # ------------------------------------------------------------------
    final_model = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=5,
        loss_function="Logloss", random_seed=42, verbose=0,
    )
    full_pool = Pool(X, y, cat_features=cat_idx)
    final_model.fit(full_pool)

    # Feature importance
    fi = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    fi.to_csv(out / "p10_feature_importance.csv", index=False, encoding="utf-8-sig")
    results["feature_importance"] = fi

    fig2, ax2 = plt.subplots(figsize=(9, max(4, len(fi) * 0.4)))
    ax2.barh(fi["feature"][::-1], fi["importance"][::-1], color="#4878CF", alpha=0.8)
    ax2.set_xlabel("Feature importance (CatBoost default)")
    ax2.set_title("Phase 10 — CatBoost feature importance\n(NOT causal — SHAP preferred for interpretation)")
    ax2.grid(True, axis="x", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "p10_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # SHAP values
    # ------------------------------------------------------------------
    try:
        import shap
        explainer = shap.TreeExplainer(final_model)
        X_sample = X.sample(min(500, len(X)), random_state=42)
        for c in cat_features:
            X_sample[c] = X_sample[c].astype(str)
        shap_values = explainer.shap_values(Pool(X_sample, cat_features=cat_idx))
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        shap_mean = pd.DataFrame({
            "feature": X.columns.tolist(),
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_mean.to_csv(out / "p10_shap_importance.csv", index=False, encoding="utf-8-sig")
        results["shap_importance"] = shap_mean

        fig3, ax3 = plt.subplots(figsize=(9, max(4, len(shap_mean) * 0.4)))
        ax3.barh(shap_mean["feature"][::-1], shap_mean["mean_abs_shap"][::-1], color="#6ACC65", alpha=0.85)
        ax3.set_xlabel("Mean |SHAP value|")
        ax3.set_title("Phase 10 — SHAP feature importance (mean |SHAP| on 500-sample subset)")
        ax3.grid(True, axis="x", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(out / "p10_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print("  SHAP importance computed.")

        # Predicted infant probability by freq group
        X_all = X.copy()
        for c in cat_features:
            X_all[c] = X_all[c].astype(str)
        all_proba = final_model.predict_proba(Pool(X_all, cat_features=cat_idx))[:, 1]
        freq_grps = df.loc[X.index, "freq_group"].values
        pred_by_grp = pd.DataFrame({"freq_group": freq_grps, "prob_infant": all_proba})
        pred_summary = pred_by_grp.groupby("freq_group", observed=True)["prob_infant"].agg(["mean", "median"]).reset_index()
        pred_summary.to_csv(out / "p10_predicted_prob_by_freq.csv", index=False, encoding="utf-8-sig")
        results["pred_by_freq"] = pred_summary
        print("  Predicted infant probability by freq group:")
        print(pred_summary.to_string(index=False))

    except Exception as e:
        print(f"  SHAP skipped: {e}")

    print(f"\n  Phase 10 complete. CV AUC={mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Top features:")
    print(fi.head(10).to_string(index=False))

    return results
