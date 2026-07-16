"""
Same rigorous methodology as train.py/evaluate.py, applied to the cardio70k
dataset: nested stratified CV, tuned logistic regression + XGBoost,
AUROC with bootstrapped CI, calibration, SHAP. See cardio70k_prep.py for why
this is a closer (but still imperfect) match to the primary UCI task than
Framingham is.
"""
import time
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cardio70k_prep import load_clean_cardio70k, NUMERIC_FEATURES, CATEGORICAL_FEATURES

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
FIGURES = BASE / "figures"
MODELS = BASE / "models"
SEED = 42


def make_pipeline(model):
    numeric_pipe = SkPipeline([("impute", IterativeImputer(random_state=SEED, max_iter=10)),
                                ("scale", StandardScaler())])
    categorical_pipe = SkPipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, NUMERIC_FEATURES),
                              ("cat", categorical_pipe, CATEGORICAL_FEATURES)])
    return SkPipeline([("preprocess", pre), ("model", model)])


def bootstrap_auroc_ci(y_true, y_proba, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return np.mean(aucs), lo, hi


def calibration_slope_intercept(y_true, y_proba, eps=1e-6):
    p = np.clip(y_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def run_one(name, model, grid, X, y, outer_cv, inner_cv):
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in outer_cv.split(X, y):
        pipe = make_pipeline(model)
        gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=2, refit=True)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[test_idx] = gs.predict_proba(X.iloc[test_idx])[:, 1]

    auc_mean, lo, hi = bootstrap_auroc_ci(y, oof_proba)
    slope, intercept = calibration_slope_intercept(y.values, oof_proba)
    pred = (oof_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)

    result = {"model": name, "auroc_mean": auc_mean, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
              "calibration_slope": slope, "calibration_intercept": intercept,
              "sensitivity_0.5": sens, "specificity_0.5": spec}
    print(name + ": AUROC " + str(round(auc_mean,3)) + " (" + str(round(lo,3)) + "-" + str(round(hi,3)) + ") slope=" + str(round(slope,2)) + " intercept=" + str(round(intercept,2)))

    frac_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(mean_pred, frac_pos, "o-")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration -- cardio70k " + name)
    plt.tight_layout()
    plt.savefig(FIGURES / ("calibration_cardio70k_" + name + ".png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y, oof_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="AUROC=" + str(round(auc_mean,3)))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC -- cardio70k " + name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / ("roc_cardio70k_" + name + ".png"), dpi=150)
    plt.close()

    final_pipe = make_pipeline(model)
    final_gs = GridSearchCV(final_pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=2, refit=True)
    final_gs.fit(X, y)
    joblib.dump(final_gs.best_estimator_, MODELS / ("cardio70k_" + name + "_final.joblib"))

    return result


def main():
    t0 = time.time()
    X, y, audit = load_clean_cardio70k()
    print("Loaded cardio70k:", X.shape, "audit:", audit)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)

    configs = {
        "logreg": (LogisticRegression(max_iter=2000, random_state=SEED), {"model__C": [0.01, 0.1, 1.0]}),
        "xgboost": (XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=2),
                    {"model__n_estimators": [200], "model__max_depth": [3, 4], "model__learning_rate": [0.1]}),
    }

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        configs = {only: configs[only]}

    results_path = REPORTS / "cardio70k_results.csv"
    prior_results = []
    if results_path.exists():
        prior_results = pd.read_csv(results_path).to_dict("records")
        prior_results = [r for r in prior_results if r["model"] not in configs]

    new_results = []
    for name, (model, grid) in configs.items():
        new_results.append(run_one(name, model, grid, X, y, outer_cv, inner_cv))

    pd.DataFrame(prior_results + new_results).to_csv(results_path, index=False)
    print("Total time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
