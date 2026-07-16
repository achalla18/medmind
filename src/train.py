"""
Nested cross-validation training for the UCI combined Heart Disease dataset.

- Outer loop (5-fold stratified): honest performance estimate.
- Inner loop (5-fold stratified, via GridSearchCV): hyperparameter tuning.
Nothing about the outer test fold ever touches preprocessing, resampling,
or hyperparameter selection -- everything is inside the imblearn Pipeline
and refit per outer-training-fold only.

Also runs a "with vs without" class-imbalance-handling ablation
(none / class_weight / SMOTE) for both model families, as required by the
Gold-Tier checklist.

Outputs:
  reports/imbalance_comparison.csv
  reports/model_comparison.csv
  reports/oof_predictions_logreg.csv
  reports/oof_predictions_xgboost.csv
  models/logreg_final.joblib
  models/xgboost_final.joblib
"""
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from xgboost import XGBClassifier

from data_prep import load_clean_uci
from model_pipeline import make_pipeline

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
MODELS = BASE / "models"
SEED = 42

X, y, groups = load_clean_uci()
n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
scale_pos_weight = n_neg / n_pos

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)


def get_model_and_grid(family, imbalance):
    if family == "logreg":
        if imbalance == "class_weight":
            model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        else:
            model = LogisticRegression(max_iter=2000, random_state=SEED)
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    elif family == "xgboost":
        spw = scale_pos_weight if imbalance == "class_weight" else 1.0
        model = XGBClassifier(eval_metric="logloss", random_state=SEED,
                               scale_pos_weight=spw, n_jobs=2)
        grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.1],
        }
    else:
        raise ValueError(family)

    imb_strategy = "smote" if imbalance == "smote" else "none"
    pipe = make_pipeline(model, imbalance_strategy=imb_strategy)
    return pipe, grid


def nested_cv(family, imbalance, X, y):
    fold_auc, fold_f1, fold_recall = [], [], []
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe, grid = get_model_and_grid(family, imbalance)
        gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=2, refit=True)
        gs.fit(X_tr, y_tr)

        proba = gs.predict_proba(X_te)[:, 1]
        oof_proba[test_idx] = proba
        pred = (proba >= 0.5).astype(int)

        fold_auc.append(roc_auc_score(y_te, proba))
        fold_f1.append(f1_score(y_te, pred))
        fold_recall.append(recall_score(y_te, pred))

    return {
        "family": family, "imbalance": imbalance,
        "auroc_mean": np.mean(fold_auc), "auroc_std": np.std(fold_auc),
        "f1_mean": np.mean(fold_f1), "recall_mean": np.mean(fold_recall),
        "fold_aurocs": fold_auc,
    }, oof_proba


if __name__ == "__main__":
    t0 = time.time()
    results = []
    oof_store = {}

    for family in ["logreg", "xgboost"]:
        for imbalance in ["none", "class_weight", "smote"]:
            res, oof = nested_cv(family, imbalance, X, y)
            results.append(res)
            oof_store[(family, imbalance)] = oof
            print(f"{family:8s} | {imbalance:13s} | AUROC {res['auroc_mean']:.3f} +/- {res['auroc_std']:.3f} "
                  f"| F1 {res['f1_mean']:.3f} | Recall {res['recall_mean']:.3f}")

    comp_df = pd.DataFrame(results)[["family", "imbalance", "auroc_mean", "auroc_std", "f1_mean", "recall_mean"]]
    comp_df.to_csv(REPORTS / "imbalance_comparison.csv", index=False)

    # pick the best imbalance strategy per family by mean AUROC
    best = {}
    for family in ["logreg", "xgboost"]:
        sub = comp_df[comp_df.family == family].sort_values("auroc_mean", ascending=False)
        best[family] = sub.iloc[0]["imbalance"]
        print(f"Best imbalance strategy for {family}: {best[family]}")

    # save the chosen out-of-fold predictions for downstream evaluation
    for family in ["logreg", "xgboost"]:
        oof = oof_store[(family, best[family])]
        out = pd.DataFrame({"y_true": y.values, "y_proba": oof, "dataset_site": groups.values})
        name = "logreg" if family == "logreg" else "xgboost"
        out.to_csv(REPORTS / f"oof_predictions_{name}.csv", index=False)

    # refit final pipelines on the FULL dataset (with chosen strategy) for
    # SHAP explanation and for the external-validation script. Hyperparameters
    # are re-selected via the same inner CV grid on the full data.
    final_summary = []
    for family in ["logreg", "xgboost"]:
        pipe, grid = get_model_and_grid(family, best[family])
        gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=2, refit=True)
        gs.fit(X, y)
        joblib.dump(gs.best_estimator_, MODELS / f"{family}_final.joblib")
        final_summary.append({"family": family, "chosen_imbalance": best[family],
                               "best_params": gs.best_params_, "best_cv_auroc": gs.best_score_})
        print(f"Saved final {family} pipeline. Best params: {gs.best_params_}")

    pd.DataFrame(final_summary).to_csv(REPORTS / "model_comparison.csv", index=False)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
