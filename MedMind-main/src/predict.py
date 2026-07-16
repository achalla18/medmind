"""
Interactive command-line risk estimate for a single patient, using the
trained Phase-1 model.

Run:  python predict.py
Then answer each question (press Enter to accept the bracketed default,
which is the median/most common value from the training data -- useful for
seeing how the prediction changes as you tweak just one or two answers).

IMPORTANT: this is a research/educational prototype, not a diagnosis.
It estimates the probability that a patient's profile resembles those with
angiographically significant coronary artery disease in the training data.
It is not validated for clinical use and should never replace a doctor.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# training-data ranges, used only to warn when an input is well outside what
# the model has ever seen -- extrapolation outside these ranges is unreliable
TRAINING_RANGES = {
    "age": (28, 77), "trestbps": (80, 200), "chol": (85, 603),
    "thalch": (60, 202), "oldpeak": (-2.6, 6.2), "ca": (0, 3),
}

# fbs and exang were stored as native Python booleans (not strings) when the
# model was trained -- inputs must be converted to real bool objects or the
# one-hot encoder silently treats them as an unrecognized category (all-zero),
# which quietly drops the answer from the prediction entirely.
BOOL_FEATURES = {"fbs", "exang"}

QUESTIONS = [
    ("age", "Age (years)", "numeric", 54),
    ("sex", "Sex", "choice", ["Male", "Female"], "Male"),
    ("cp", "Chest pain type", "choice",
     ["typical angina", "atypical angina", "non-anginal", "asymptomatic"], "asymptomatic"),
    ("trestbps", "Resting blood pressure (mmHg)", "numeric", 130),
    ("chol", "Serum cholesterol (mg/dL)", "numeric", 223),
    ("fbs", "Fasting blood sugar > 120 mg/dL?", "choice", ["True", "False"], "False"),
    ("restecg", "Resting ECG result", "choice",
     ["normal", "st-t abnormality", "lv hypertrophy"], "normal"),
    ("thalch", "Max heart rate achieved (bpm)", "numeric", 140),
    ("exang", "Exercise-induced angina?", "choice", ["True", "False"], "False"),
    ("oldpeak", "ST depression induced by exercise", "numeric", 0.5),
    ("slope", "Slope of peak exercise ST segment", "choice",
     ["upsloping", "flat", "downsloping"], "flat"),
    ("ca", "Number of major vessels colored by fluoroscopy (0-3)", "numeric", 0),
    ("thal", "Thallium stress test result", "choice",
     ["normal", "fixed defect", "reversable defect"], "normal"),
]


def ask(name, prompt, kind, *args):
    if kind == "numeric":
        default = args[0]
        raw = input(f"{prompt} [{default}]: ").strip()
        val = float(raw) if raw else float(default)
        lo, hi = TRAINING_RANGES[name]
        if val < lo or val > hi:
            print(f"  [!] {val} is outside the training data's observed range "
                  f"({lo}-{hi}) for {name}. The model has never seen anything "
                  f"like this -- treat the resulting probability as unreliable "
                  f"extrapolation, not a real estimate.")
        return val
    else:
        choices, default = args
        choice_str = "/".join(choices)
        raw = input(f"{prompt} ({choice_str}) [{default}]: ").strip()
        chosen = raw if raw else default
        # case-insensitive match against the allowed choices
        matches = [c for c in choices if c.lower() == chosen.lower()]
        if not matches:
            print(f"  [!] '{chosen}' isn't one of {choices}; using default '{default}' instead.")
            chosen = default
        else:
            chosen = matches[0]
        if name in BOOL_FEATURES:
            return chosen == "True"
        return chosen


def collect_patient():
    print("Enter patient values (press Enter to accept the bracketed default).\n")
    answers = {}
    for item in QUESTIONS:
        name, prompt, kind, *rest = item
        answers[name] = ask(name, prompt, kind, *rest)
    return pd.DataFrame([answers])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def risk_label(p):
    if p < 0.2:
        return "Low"
    elif p < 0.5:
        return "Moderate-low"
    elif p < 0.8:
        return "Moderate-high"
    else:
        return "High"


def explain_patient(pipe, patient_df):
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_trans = preprocess.transform(patient_df)
    feature_names = preprocess.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_trans_df)
    contribs = pd.Series(sv.values[0], index=feature_names).sort_values(key=abs, ascending=False)
    return contribs.head(5)


def main():
    model_path = MODELS / "xgboost_final.joblib"
    if not model_path.exists():
        print(f"Could not find {model_path}. Run train.py first.")
        sys.exit(1)
    pipe = joblib.load(model_path)

    patient_df = collect_patient()
    proba = pipe.predict_proba(patient_df)[0, 1]
    label = risk_label(proba)

    print("\n" + "=" * 50)
    print(f"Estimated probability of significant CAD: {proba*100:.1f}%")
    print(f"Risk category: {label}")
    print("=" * 50)

    print("\nTop factors driving this specific estimate (SHAP):")
    contribs = explain_patient(pipe, patient_df)
    for feat, val in contribs.items():
        direction = "raises" if val > 0 else "lowers"
        print(f"  - {feat:30s} {direction} risk (impact {val:+.3f})")

    print("\nThis is a research prototype, not a medical diagnosis. Discuss any "
          "concerns about heart disease risk with a qualified clinician.")


if __name__ == "__main__":
    main()
