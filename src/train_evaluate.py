"""
train_evaluate.py
------------------
End-to-end driver: load -> audit -> clean -> nested-CV evaluate (LR + XGB)
-> calibration -> final fit -> save models + metrics + figures.

Run from the heart_project/ directory: `python src/train_evaluate.py`
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

from data import (
    load_raw, audit_data, write_audit_report, write_data_dictionary,
    clean_data, get_feature_target,
)
from modeling import (
    build_lr_pipeline, build_xgb_pipeline, LR_PARAM_GRID, XGB_PARAM_GRID,
    nested_cv_oof_predictions, bootstrap_auroc_ci,
    classification_metrics_at_threshold, calibration_slope_intercept,
    fit_final_pipeline, MODELS_DIR, SEED,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def main():
    np.random.seed(SEED)

    print("=" * 70)
    print("STEP 1: Load + audit + clean")
    print("=" * 70)
    raw = load_raw()
    audit = audit_data(raw)
    write_audit_report(audit)
    write_data_dictionary()
    cleaned = clean_data(raw)
    X, y = get_feature_target(cleaned)
    print(f"Raw rows: {audit['n_rows_raw']}  ->  cleaned rows: {len(cleaned)}")
    print(f"chol==0 recoded to missing: {audit['chol_zero_count']}")
    print(f"Class balance: {audit['target_balance']}")

    results = {}
    all_metrics_rows = []

    models = {
        "Logistic Regression": (build_lr_pipeline, LR_PARAM_GRID),
        "XGBoost": (build_xgb_pipeline, XGB_PARAM_GRID),
    }

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, (build_fn, grid) in models.items():
        print("\n" + "=" * 70)
        print(f"STEP 2: Nested cross-validation -- {name}")
        print("=" * 70)
        oof_proba, fold_params = nested_cv_oof_predictions(
            X, y, build_fn, grid, n_outer=5, n_inner=3, seed=SEED
        )
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y, oof_proba)
        metrics_50 = classification_metrics_at_threshold(y, oof_proba, 0.5)
        slope, intercept = calibration_slope_intercept(y.values, oof_proba)

        print(f"AUROC: {auc_mean:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")
        print(f"At threshold 0.5: sensitivity={metrics_50['sensitivity']:.3f} "
              f"specificity={metrics_50['specificity']:.3f} "
              f"precision={metrics_50['precision']:.3f} "
              f"f1={metrics_50['f1']:.3f}")
        print(f"Calibration slope={slope:.3f} intercept={intercept:.3f}")
        print(f"Per-fold best hyperparameters: {fold_params}")

        all_metrics_rows.append({
            "model": name,
            "auroc": auc_mean, "auroc_ci_low": auc_lo, "auroc_ci_high": auc_hi,
            "sensitivity": metrics_50["sensitivity"],
            "specificity": metrics_50["specificity"],
            "precision": metrics_50["precision"],
            "f1": metrics_50["f1"],
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        })

        frac_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

        results[name] = {
            "oof_proba": oof_proba,
            "fold_params": fold_params,
        }

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (observed)")
    plt.title("Calibration curve (out-of-fold predictions, nested CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curves.png", dpi=150)
    plt.close()
    print(f"\nSaved {FIGURES_DIR / 'calibration_curves.png'}")

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(REPORTS_DIR / "evaluation_metrics.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'evaluation_metrics.csv'}")
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3: Fit final deployment models on ALL cleaned data")
    print("=" * 70)
    final_models = {}
    for name, (build_fn, grid) in models.items():
        best_pipe, best_params = fit_final_pipeline(X, y, build_fn, grid,
                                                      n_inner=5, seed=SEED)
        print(f"{name} final hyperparameters: {best_params}")
        final_models[name] = best_pipe

    joblib.dump(final_models["Logistic Regression"],
                MODELS_DIR / "lr_pipeline.joblib")
    joblib.dump(final_models["XGBoost"], MODELS_DIR / "xgb_pipeline.joblib")
    print(f"Saved models to {MODELS_DIR}")

    # Save cleaned data + oof predictions for notebook reuse
    cleaned.to_csv(PROJECT_ROOT / "data" / "cleaned.csv", index=False)
    np.save(REPORTS_DIR / "lr_oof_proba.npy", results["Logistic Regression"]["oof_proba"])
    np.save(REPORTS_DIR / "xgb_oof_proba.npy", results["XGBoost"]["oof_proba"])
    y.to_csv(REPORTS_DIR / "y_full.csv", index=False)

    print("\nDone. All artifacts written under reports/, figures/, models/.")


if __name__ == "__main__":
    main()
