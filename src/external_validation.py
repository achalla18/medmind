"""
Framingham as a second, independent dataset.

IMPORTANT HONESTY NOTE (see reports/external_validation.md for full discussion):
A true clinical external validation would apply the UCI-trained model, unchanged,
to a second dataset predicting the SAME clinical target. That is not possible
here: UCI's target is presence of angiographically significant coronary
narrowing at time of catheterization (a diagnostic snapshot), while
Framingham's target is 10-year incident CHD risk (a prognostic, time-to-event
question) in a population that was disease-free at baseline. The feature sets
also barely overlap (UCI has cp/thal/ca/exang/oldpeak/restecg; Framingham has
smoking/diabetes/BP-meds/education -- none shared). Forcing a direct transfer
and calling it "validated" would be exactly the kind of overclaiming a
Gold-Tier project must avoid.

So we do two honestly-labeled things instead:

  (A) NAIVE TRANSPORTABILITY CHECK: refit a reduced logistic regression using
      only the four features both datasets share in spirit (age, sex,
      systolic BP, total cholesterol) on UCI, and see how it discriminates
      Framingham's TenYearCHD outcome. This is explicitly a transportability
      probe, not a validation -- the two targets are different clinical
      questions, so a low or high AUROC here says more about how much a
      handful of shared risk factors generalize across a diagnostic-vs-
      prognostic gap than about whether the Phase-1 model "works" on new data.

  (B) METHODOLOGICAL REPLICATION: rerun the IDENTICAL rigorous pipeline
      (same imputation strategy, nested stratified CV, calibration,
      thresholding) on Framingham's own native prediction task (its own
      features -> its own TenYearCHD label). This demonstrates the pipeline
      and evaluation methodology -- not the fitted model -- generalizes
      across an independent, larger, differently-imbalanced dataset. This is
      the strongest honest evidence available from two public datasets with
      incompatible targets.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

from data_prep import load_clean_uci

BASE = Path(__file__).resolve().parents[1]
RAW_FHS = BASE / "data" / "raw" / "framingham.csv"
REPORTS = BASE / "reports"
FIGURES = BASE / "figures"
MODELS = BASE / "models"
SEED = 42


def calibration_slope_intercept(y_true, y_proba, eps=1e-6):
    p = np.clip(y_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def bootstrap_auroc_ci(y_true, y_proba, n_boot=2000, seed=SEED):
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


# ---------------------------------------------------------------------------
# (A) Naive transportability check on 4 shared features
# ---------------------------------------------------------------------------
def naive_transport_check():
    X_uci, y_uci, _ = load_clean_uci()
    shared_uci = X_uci[["age", "trestbps", "chol"]].copy()
    shared_uci["sex_male"] = (X_uci["sex"] == "Male").astype(int)

    pre = ColumnTransformer([
        ("num", SkPipeline([("impute", IterativeImputer(random_state=SEED)),
                             ("scale", StandardScaler())]),
         ["age", "trestbps", "chol", "sex_male"]),
    ])
    reduced_model = SkPipeline([("preprocess", pre),
                                 ("model", LogisticRegression(max_iter=2000, random_state=SEED))])
    reduced_model.fit(shared_uci, y_uci)

    fhs = pd.read_csv(RAW_FHS)
    fhs = fhs.dropna(subset=["totChol", "sysBP", "age", "male"]).reset_index(drop=True)
    shared_fhs = pd.DataFrame({
        "age": fhs["age"],
        "trestbps": fhs["sysBP"],
        "chol": fhs["totChol"],
        "sex_male": fhs["male"],
    })
    y_fhs = fhs["TenYearCHD"]

    proba = reduced_model.predict_proba(shared_fhs)[:, 1]
    auc_mean, lo, hi = bootstrap_auroc_ci(y_fhs, proba)
    slope, intercept = calibration_slope_intercept(y_fhs.values, proba)

    result = {
        "check": "naive_transportability_4_shared_features",
        "n_framingham_rows_used": len(fhs),
        "auroc_mean": auc_mean, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "caveat": "Different target definitions (prevalent CAD vs 10-yr incident CHD); "
                  "not a valid clinical external validation, see docstring.",
    }
    return result


# ---------------------------------------------------------------------------
# (B) Full methodological replication on Framingham's own task
# ---------------------------------------------------------------------------
FHS_NUMERIC = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
FHS_CATEGORICAL = ["male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]


def fhs_pipeline(imbalance="class_weight"):
    numeric_pipe = SkPipeline([("impute", IterativeImputer(random_state=SEED, max_iter=15)),
                                ("scale", StandardScaler())])
    categorical_pipe = SkPipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, FHS_NUMERIC),
                              ("cat", categorical_pipe, FHS_CATEGORICAL)])
    model = LogisticRegression(max_iter=2000, random_state=SEED,
                                class_weight="balanced" if imbalance == "class_weight" else None)
    return SkPipeline([("preprocess", pre), ("model", model)])


def methodological_replication():
    fhs = pd.read_csv(RAW_FHS)
    X = fhs[FHS_NUMERIC + FHS_CATEGORICAL].copy()
    y = fhs["TenYearCHD"].copy()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in outer_cv.split(X, y):
        pipe = fhs_pipeline("class_weight")
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
                           scoring="roc_auc", n_jobs=2, refit=True)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[test_idx] = gs.predict_proba(X.iloc[test_idx])[:, 1]

    auc_mean, lo, hi = bootstrap_auroc_ci(y, oof_proba)
    slope, intercept = calibration_slope_intercept(y.values, oof_proba)

    # calibration + ROC figures
    frac_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, "o-", label="Framingham LR (own task)")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration -- Framingham replication\nslope={slope:.2f}, intercept={intercept:.2f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "calibration_framingham.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y, oof_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Framingham LR (AUROC={auc_mean:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC -- Framingham replication"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "roc_framingham.png", dpi=150)
    plt.close()

    # final refit on all Framingham data for completeness
    final_pipe = fhs_pipeline("class_weight")
    final_gs = GridSearchCV(final_pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
                             scoring="roc_auc", n_jobs=2, refit=True)
    final_gs.fit(X, y)
    joblib.dump(final_gs.best_estimator_, MODELS / "framingham_logreg_final.joblib")

    return {
        "check": "methodological_replication_framingham_own_task",
        "n_rows": len(y), "positive_rate": float(y.mean()),
        "auroc_mean": auc_mean, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "best_params": final_gs.best_params_,
    }


if __name__ == "__main__":
    result_a = naive_transport_check()
    print("(A) Naive transportability check:", result_a)
    result_b = methodological_replication()
    print("(B) Methodological replication:", result_b)

    pd.DataFrame([result_a]).to_csv(REPORTS / "external_naive_transport.csv", index=False)
    pd.DataFrame([result_b]).to_csv(REPORTS / "external_methodological_replication.csv", index=False)
