"""
Full evaluation of out-of-fold (nested-CV) predictions:
  - AUROC with bootstrapped 95% CI
  - Sensitivity, specificity, PPV, NPV, F1 at threshold=0.5 AND at a
    clinically-motivated threshold tuned for higher sensitivity (Youden's J)
  - Calibration curve + calibration slope/intercept
Never touches the test set more than once; all of this operates on the
out-of-fold predictions produced by the nested CV in train.py, so every
prediction was made by a model that never saw that row during training.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
FIGURES = BASE / "figures"
SEED = 42
N_BOOT = 2000


def bootstrap_auroc_ci(y_true, y_proba, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return np.mean(aucs), lo, hi


def sens_spec_ppv_npv_f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1)


def youdens_threshold(y_true, y_proba):
    fpr, tpr, thresh = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return thresh[np.argmax(j)]


def calibration_slope_intercept(y_true, y_proba, eps=1e-6):
    p = np.clip(y_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def evaluate_model(name):
    df = pd.read_csv(REPORTS / f"oof_predictions_{name}.csv")
    y_true, y_proba = df["y_true"].values, df["y_proba"].values

    auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_proba)

    thresh_default = 0.5
    thresh_youden = youdens_threshold(y_true, y_proba)

    metrics_default = sens_spec_ppv_npv_f1(y_true, (y_proba >= thresh_default).astype(int))
    metrics_youden = sens_spec_ppv_npv_f1(y_true, (y_proba >= thresh_youden).astype(int))

    slope, intercept = calibration_slope_intercept(y_true, y_proba)

    # --- calibration plot ---
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, "o-", label=f"{name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve -- {name}\nslope={slope:.2f}, intercept={intercept:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"calibration_{name}.png", dpi=150)
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUROC={auc_mean:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate (1-Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC curve -- {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"roc_{name}.png", dpi=150)
    plt.close()

    return {
        "model": name,
        "auroc_mean": auc_mean, "auroc_ci_lo": auc_lo, "auroc_ci_hi": auc_hi,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "threshold_0.5_sensitivity": metrics_default["sensitivity"],
        "threshold_0.5_specificity": metrics_default["specificity"],
        "threshold_0.5_ppv": metrics_default["ppv"],
        "threshold_0.5_npv": metrics_default["npv"],
        "threshold_0.5_f1": metrics_default["f1"],
        "youden_threshold": thresh_youden,
        "youden_sensitivity": metrics_youden["sensitivity"],
        "youden_specificity": metrics_youden["specificity"],
        "youden_ppv": metrics_youden["ppv"],
        "youden_npv": metrics_youden["npv"],
        "youden_f1": metrics_youden["f1"],
    }


if __name__ == "__main__":
    rows = [evaluate_model("logreg"), evaluate_model("xgboost")]
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "evaluation_metrics.csv", index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(out.T)
