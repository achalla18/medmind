"""
brfss_train.py
---------------
End-to-end driver for the BRFSS analysis: load -> audit -> clean -> split
-> tune (train-set-only CV) -> evaluate once on held-out test -> final
refit on all data -> save models + metrics + figures.

Run in two stages (kept separate only to keep each run short):
  python brfss_train.py evaluate   # tune on train, evaluate once on test
  python brfss_train.py finalize   # refit best config on ALL data, save models
"""
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

from brfss_data import (
    load_raw, audit_data, write_audit_report, write_data_dictionary,
    clean_data, get_feature_target,
)
from brfss_modeling import (
    build_lr_pipeline, build_xgb_pipeline, LR_PARAM_GRID, XGB_PARAM_GRID,
    train_test_split_data, tune_and_fit, SEED,
)
from modeling import (
    bootstrap_auroc_ci, classification_metrics_at_threshold,
    calibration_slope_intercept,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

MODELS = {
    "Logistic Regression": (build_lr_pipeline, LR_PARAM_GRID),
    "XGBoost": (build_xgb_pipeline, XGB_PARAM_GRID),
}


def load_clean():
    raw = load_raw()
    audit = audit_data(raw)
    write_audit_report(audit)
    write_data_dictionary()
    cleaned = clean_data(raw)
    X, y = get_feature_target(cleaned)
    cleaned.to_csv(PROJECT_ROOT / "data" / "brfss_cleaned.csv", index=False)
    return X, y, audit


def stage_evaluate():
    t0 = time.time()
    X, y, audit = load_clean()
    print(f"Rows: {len(X)}  Class balance: {audit['target_balance']}  [{time.time()-t0:.1f}s]")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    all_metrics_rows = []
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, (build_fn, grid) in MODELS.items():
        best_pipe, best_params = tune_and_fit(X_train, y_train, build_fn, grid,
                                               cv_folds=3, seed=SEED)
        print(f"{name} best params: {best_params}  [{time.time()-t0:.1f}s]")

        test_proba = best_pipe.predict_proba(X_test)[:, 1]
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_test, test_proba, n_boot=1000)
        m50 = classification_metrics_at_threshold(y_test, test_proba, 0.5)
        slope, intercept = calibration_slope_intercept(y_test.values, test_proba)

        print(f"  AUROC {auc_mean:.3f} ({auc_lo:.3f}-{auc_hi:.3f}) "
              f"sens={m50['sensitivity']:.3f} spec={m50['specificity']:.3f} "
              f"prec={m50['precision']:.3f} f1={m50['f1']:.3f} "
              f"cal_slope={slope:.3f} cal_int={intercept:.3f}")

        all_metrics_rows.append({
            "model": name, "auroc": auc_mean, "auroc_ci_low": auc_lo,
            "auroc_ci_high": auc_hi, "sensitivity": m50["sensitivity"],
            "specificity": m50["specificity"], "precision": m50["precision"],
            "f1": m50["f1"], "calibration_slope": slope,
            "calibration_intercept": intercept, "best_params": str(best_params),
        })
        frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (observed)")
    plt.title("BRFSS calibration curve (held-out test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "brfss_calibration_curves.png", dpi=150)
    plt.close()

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(REPORTS_DIR / "brfss_evaluation_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    print(f"stage_evaluate done in {time.time()-t0:.1f}s")


def stage_finalize():
    t0 = time.time()
    X, y, _ = load_clean()
    for name, (build_fn, grid) in MODELS.items():
        best_pipe, best_params = tune_and_fit(X, y, build_fn, grid, cv_folds=3, seed=SEED)
        print(f"{name} final params: {best_params}  [{time.time()-t0:.1f}s]")
        fname = "brfss_lr_pipeline.joblib" if "Logistic" in name else "brfss_xgb_pipeline.joblib"
        joblib.dump(best_pipe, MODELS_DIR / fname)
        print(f"saved {fname}")
    print(f"stage_finalize done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if stage == "evaluate":
        stage_evaluate()
    elif stage == "finalize":
        stage_finalize()
    else:
        raise SystemExit(f"Unknown stage: {stage}")
