"""
Loading and cleaning for the Kaggle "Cardiovascular Disease dataset"
(sulianova/cardiovascular-disease-dataset, ~70,000 rows).

Distinct from the primary UCI dataset: this is a Russian medical-exam cohort
where the target (`cardio`) reflects a broader "cardiovascular disease"
diagnosis based on blood pressure/cholesterol/glucose thresholds and exam
findings, NOT specifically angiographically-confirmed coronary artery
disease. Feature overlap with UCI is limited to age, sex, and blood
pressure -- there is no chest-pain, ECG, fluoroscopy, or thallium-test data
here. See reports/second_cardio_dataset.md for the honest discussion of
what this dataset can and cannot be used to claim.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "cardio70k.csv"

NUMERIC_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def load_clean_cardio70k():
    df = pd.read_csv(RAW, sep=";")

    dupe_ids = df["id"].duplicated().sum()
    dupe_rows = df.drop(columns=["id"]).duplicated().sum()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"]).reset_index(drop=True)

    df["age_years"] = (df["age"] / 365.25)

    # physiologically implausible sentinels -> missing (see reports/data audit)
    bad_bp = (df["ap_hi"] <= 0) | (df["ap_hi"] > 260) | (df["ap_lo"] <= 0) | (df["ap_lo"] > 200) | (df["ap_hi"] <= df["ap_lo"])
    df.loc[bad_bp, ["ap_hi", "ap_lo"]] = np.nan
    bad_height = (df["height"] < 130) | (df["height"] > 210)
    df.loc[bad_height, "height"] = np.nan
    bad_weight = (df["weight"] < 30) | (df["weight"] > 200)
    df.loc[bad_weight, "weight"] = np.nan

    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

    df["gender"] = df["gender"].map({1: "Female", 2: "Male"})
    # cholesterol/gluc: 1=normal, 2=above normal, 3=well above normal (ordinal, kept categorical)
    df["cholesterol"] = df["cholesterol"].map({1: "normal", 2: "above normal", 3: "well above normal"})
    df["gluc"] = df["gluc"].map({1: "normal", 2: "above normal", 3: "well above normal"})

    y = df["cardio"].astype(int)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    audit = {
        "n_rows": len(df), "dupe_ids": int(dupe_ids), "dupe_rows": int(dupe_rows),
        "bad_bp_rows_recoded": int(bad_bp.sum()), "bad_height_rows_recoded": int(bad_height.sum()),
        "bad_weight_rows_recoded": int(bad_weight.sum()),
    }
    return X, y, audit


if __name__ == "__main__":
    X, y, audit = load_clean_cardio70k()
    print("shape:", X.shape)
    print("target balance:")
    print(y.value_counts(normalize=True))
    print("audit:", audit)
    print(X.describe(include="all").T)
