"""
explain.py
----------
SHAP explainability, shared by the notebook (Part 1) and the Streamlit
app (Part 2) so the "why did the model say that" logic is defined exactly
once.

WHY SHAP (short version for the presentation):
A model that spits out "73% risk" with no explanation is not something a
patient or clinician should trust blindly. SHAP (SHapley Additive
exPlanations) breaks a single prediction down into: "the model's average
prediction across all patients, PLUS how much each of this patient's
specific feature values pushed the prediction up or down." That gives two
things a bare probability can't: (1) a sanity check -- do the top drivers
match known medical risk factors, or is the model keying off something
suspicious? -- and (2) a personalized explanation -- "your risk is high
mainly because of X and Y," which is what the what-if simulator in Part 2
is built around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from data import CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERIC_FEATURES, ALL_FEATURES


def get_output_feature_names(preprocessor) -> list[str]:
    """Recover human-readable feature names after the ColumnTransformer's
    imputation/scaling/one-hot-encoding, so SHAP plots say 'cp_4.0'
    instead of 'x14'."""
    names = []
    names += NUMERIC_FEATURES  # num transformer doesn't change column count
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    names += list(cat_names)
    names += BINARY_FEATURES
    return names


def make_tree_explainer(xgb_pipeline):
    """Build a SHAP TreeExplainer for the XGB step of a fitted pipeline.
    Returns (explainer, preprocessor, feature_names)."""
    preprocessor = xgb_pipeline.named_steps["preprocess"]
    clf = xgb_pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    feature_names = get_output_feature_names(preprocessor)
    return explainer, preprocessor, feature_names


def transform_for_shap(preprocessor, X: pd.DataFrame) -> np.ndarray:
    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return Xt


def explain_dataset(xgb_pipeline, X: pd.DataFrame):
    """Global SHAP values for a whole dataframe. Returns (shap_values,
    Xt, feature_names) for building summary plots."""
    explainer, preprocessor, feature_names = make_tree_explainer(xgb_pipeline)
    Xt = transform_for_shap(preprocessor, X)
    shap_values = explainer.shap_values(Xt)
    return shap_values, Xt, feature_names


def explain_patient(xgb_pipeline, patient_row: pd.DataFrame):
    """SHAP explanation for a single patient (patient_row: 1-row DataFrame
    with the raw, un-preprocessed feature columns -- exactly what the
    Streamlit sliders produce). Returns a dict with the predicted
    probability and a list of (feature, shap_value, feature_value) sorted
    by |impact|, ready to render as a bar chart."""
    explainer, preprocessor, feature_names = make_tree_explainer(xgb_pipeline)
    Xt = transform_for_shap(preprocessor, patient_row)
    shap_values = explainer.shap_values(Xt)[0]
    base_value = explainer.expected_value
    proba = xgb_pipeline.predict_proba(patient_row)[0, 1]

    # Xt columns line up with feature_names; grab the (post-transform)
    # value too, since e.g. a one-hot column's "value" (0 or 1) is more
    # interpretable next to its SHAP contribution than showing nothing.
    contributions = sorted(
        zip(feature_names, shap_values, Xt[0]),
        key=lambda t: -abs(t[1]),
    )
    return {
        "predicted_probability": float(proba),
        "base_value": float(base_value),
        "contributions": contributions,  # list of (name, shap_value, xt_value)
    }
