"""
brfss_modeling.py
------------------
Leakage-free pipeline and training for the BRFSS dataset.

METHODOLOGY DIFFERENCE FROM THE UCI MODEL -- disclosed deliberately, not
an inconsistency:
The UCI analysis (src/modeling.py) uses full nested cross-validation
(5 outer x 3 inner folds) because with only 918 rows, a single
train/validation split would be noisy, and hyperparameter tuning on the
same data used for the final performance estimate would risk a small,
optimistic bias that matters at that sample size.

BRFSS has 253,680 rows. At that scale, a single stratified train/test
split produces a test set of ~50,000 respondents -- more than 50x the
entire UCI dataset -- so the extra variance-reduction nested CV buys is
negligible, while its computational cost (order of magnitude more model
fits) is not. This module therefore uses: one stratified train/test
split, hyperparameter tuning via plain k-fold CV on the training set
ONLY, then a single evaluation on the held-out test set the tuning never
touched. This is standard, leakage-free practice for large datasets, not
a shortcut -- the test set is still never used for anything except the
final, one-time evaluation.

CLASS IMBALANCE: this dataset is ~90.6% no-disease / 9.4% disease-history,
much more skewed than the UCI dataset (~55/45). Both models correct for
this: logistic regression via class_weight="balanced", XGBoost via
scale_pos_weight (set to the negative/positive class ratio). Without this,
a threshold-0.5 XGBoost model in early testing had sensitivity of just
0.10 -- it was nearly always predicting "no disease" and still scoring
well on raw accuracy, exactly the accuracy-is-misleading trap this
project's metrics section (AUROC/sensitivity/specificity, not accuracy)
exists to catch.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from brfss_data import NUMERIC_FEATURES, BINARY_FEATURES

SEED = 42

# Overall negative:positive ratio in the full BRFSS dataset (229,787 : 23,893
# from the data audit) -- used as XGBoost's scale_pos_weight. Computed once
# from the full dataset rather than per-fold since stratified folds/splits
# preserve this ratio closely enough that recomputing it per-fold would not
# meaningfully change the result, and using a fixed constant keeps the
# pipeline simple and reproducible.
SCALE_POS_WEIGHT = 229787 / 23893


def build_preprocessor() -> ColumnTransformer:
    # No imputation needed anywhere -- BRFSS has zero missing values
    # (see brfss_data.audit_data). Numeric/ordinal columns are scaled;
    # binary flags are passed through unchanged.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", XGBClassifier(
            random_state=SEED, eval_metric="logloss",
            scale_pos_weight=SCALE_POS_WEIGHT,
        )),
    ])


LR_PARAM_GRID = {"clf__C": [0.01, 0.1, 1]}
XGB_PARAM_GRID = {
    "clf__n_estimators": [150, 300],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.1],
}


def train_test_split_data(X, y, test_size=0.2, seed=SEED):
    return train_test_split(X, y, test_size=test_size, stratify=y,
                             random_state=seed)


def tune_and_fit(X_train, y_train, build_pipeline_fn, param_grid,
                  cv_folds=3, seed=SEED):
    """Single-level (not nested) k-fold CV hyperparameter search on the
    training set only. Returns the refit best pipeline and best params.
    Leakage-free because the held-out test set is never involved here."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pipe = build_pipeline_fn()
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv,
                           n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
