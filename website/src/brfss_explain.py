"""
brfss_explain.py
-----------------
SHAP explainability for the BRFSS model, with a global-importance split
into "lifestyle" (modifiable, behavioral) vs "clinical/demographic"
(largely non-modifiable or already-diagnosed) feature groups -- this is
the specific question this second dataset was added to the project to
answer: which lifestyle factors actually predict heart disease history,
versus factors a person can't simply change?

Practical note: SHAP TreeExplainer on all 253,680 rows is unnecessary and
slow for a *global* importance summary -- a random sample is standard
practice and is what's used here (disclosed explicitly, matching this
project's habit of stating shortcuts rather than hiding them). Per-patient
explanations (e.g. in a dashboard) still explain one real row exactly,
sampling only affects the *aggregate* importance ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from brfss_data import NUMERIC_FEATURES, BINARY_FEATURES, LIFESTYLE_FEATURES, \
    CLINICAL_DEMOGRAPHIC_FEATURES

SAMPLE_SIZE = 5000
SEED = 42


def get_output_feature_names():
    # ColumnTransformer output order: scaled numeric columns first (in
    # NUMERIC_FEATURES order), then passthrough binary columns (in
    # BINARY_FEATURES order) -- no one-hot encoding in this pipeline, so
    # column count and order are unchanged from input.
    return list(NUMERIC_FEATURES) + list(BINARY_FEATURES)


def make_tree_explainer(xgb_pipeline):
    preprocessor = xgb_pipeline.named_steps["preprocess"]
    clf = xgb_pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    return explainer, preprocessor, get_output_feature_names()


def transform_for_shap(preprocessor, X: pd.DataFrame) -> np.ndarray:
    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt)


def explain_sample(xgb_pipeline, X: pd.DataFrame, sample_size=SAMPLE_SIZE, seed=SEED):
    """Global SHAP values on a random sample of rows (see module docstring
    for why sampling is used here, unlike the full-dataset UCI SHAP run)."""
    explainer, preprocessor, feature_names = make_tree_explainer(xgb_pipeline)
    sample = X.sample(n=min(sample_size, len(X)), random_state=seed)
    Xt = transform_for_shap(preprocessor, sample)
    shap_values = explainer.shap_values(Xt)
    return shap_values, Xt, feature_names, sample


def lifestyle_vs_clinical_split(importance_df: pd.DataFrame) -> dict:
    """importance_df must have columns 'feature' and 'mean_abs_shap'.
    Returns the % of total SHAP-explained importance attributable to
    lifestyle features vs clinical/demographic features."""
    total = importance_df["mean_abs_shap"].sum()
    lifestyle_total = importance_df.loc[
        importance_df["feature"].isin(LIFESTYLE_FEATURES), "mean_abs_shap"
    ].sum()
    clinical_total = importance_df.loc[
        importance_df["feature"].isin(CLINICAL_DEMOGRAPHIC_FEATURES), "mean_abs_shap"
    ].sum()
    return {
        "lifestyle_pct": lifestyle_total / total,
        "clinical_demographic_pct": clinical_total / total,
    }


def explain_patient(xgb_pipeline, patient_row: pd.DataFrame):
    explainer, preprocessor, feature_names = make_tree_explainer(xgb_pipeline)
    Xt = transform_for_shap(preprocessor, patient_row)
    shap_values = explainer.shap_values(Xt)[0]
    base_value = explainer.expected_value
    proba = xgb_pipeline.predict_proba(patient_row)[0, 1]
    contributions = sorted(
        zip(feature_names, shap_values, Xt[0]), key=lambda t: -abs(t[1])
    )
    return {
        "predicted_probability": float(proba),
        "base_value": float(base_value),
        "contributions": contributions,
    }
