"""
Loading and cleaning for the UCI combined Heart Disease dataset.
No leakage-prone steps (imputation, scaling, resampling) happen here --
those live inside the modeling Pipeline and are fit per-fold only.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "heart_disease_uci_combined.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_clean_uci(random_state: int = 42):
    """Load raw data, drop exact duplicates, recode physiologically
    impossible sentinel values (chol==0, trestbps==0) to NaN, and binarize
    the target. Returns (X, y, groups) where groups is the recruiting site
    (kept only for stratified diagnostics, never used as a feature)."""
    df = pd.read_csv(RAW)

    # drop duplicate patient records (see reports/data_audit.md)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"]).reset_index(drop=True)

    # physiologically impossible sentinels -> missing
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

    # binary target for Phase 1 (num==0 -> no disease, num>0 -> disease present)
    y = (df["num"] > 0).astype(int)
    groups = df["dataset"].copy()

    X = df[ALL_FEATURES].copy()

    # normalize boolean-like categoricals to plain strings (avoids True/False
    # vs "TRUE"/"FALSE" mismatches across sklearn versions/encoders)
    for c in ["fbs", "exang"]:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_clean_uci()
    print("X shape:", X.shape)
    print("y balance:\n", y.value_counts(normalize=True))
    print("dtypes:\n", X.dtypes)
