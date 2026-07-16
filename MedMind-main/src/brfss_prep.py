"""
Loading for the CDC BRFSS 2015 "Heart Disease Health Indicators" dataset
(253,680 survey respondents, 21 predictors + HeartDiseaseorAttack target).

This is the dataset that most directly matches Project 2's framing: a
population health-survey dataset with clear LIFESTYLE features (smoking,
diet, physical activity, alcohol) alongside clinical/demographic ones
(blood pressure, cholesterol, diabetes, age, income), letting us directly
rank which lifestyle factors are the strongest predictors via SHAP.

No missing values in this cleaned release. ~9.4% duplicate rows exist, but
unlike the UCI dataset's duplicates (which involved continuous lab values
and were clearly data-entry errors), these are expected here: with ~21
mostly binary/low-cardinality survey questions, many distinct respondents
will legitimately share an identical answer pattern by chance across a
253,680-row sample. We do NOT drop them -- doing so would silently bias the
sample toward less "common" respondent profiles.
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "brfss_heart_indicators.csv"

LIFESTYLE_FEATURES = ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "BMI"]
CLINICAL_DEMOGRAPHIC_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Stroke", "Diabetes", "DiffWalk",
    "GenHlth", "MentHlth", "PhysHlth", "AnyHealthcare", "NoDocbcCost",
    "Sex", "Age", "Education", "Income",
]
ALL_FEATURES = LIFESTYLE_FEATURES + CLINICAL_DEMOGRAPHIC_FEATURES


def load_brfss():
    df = pd.read_csv(RAW)
    dupe_rows = int(df.duplicated().sum())
    y = df["HeartDiseaseorAttack"].astype(int)
    X = df[ALL_FEATURES].copy()
    audit = {"n_rows": len(df), "dupe_rows_not_dropped": dupe_rows,
              "positive_rate": float(y.mean()), "n_missing": int(df.isna().sum().sum())}
    return X, y, audit


if __name__ == "__main__":
    X, y, audit = load_brfss()
    print("shape:", X.shape)
    print("audit:", audit)
    print(X.describe().T)
