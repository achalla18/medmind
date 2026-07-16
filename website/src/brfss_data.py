"""
brfss_data.py
-------------
Loading, auditing, and cleaning for the CDC BRFSS 2015 Heart Disease
Health Indicators dataset -- a SEPARATE, second dataset from the UCI
Heart Disease data used elsewhere in this project.

Why a second dataset, and why kept separate rather than combined:
The UCI model (src/data.py) predicts angiographically-confirmed coronary
artery disease from clinical/invasive test results (chest pain type, ECG,
fluoroscopy, etc.) -- it does NOT include smoking, BMI, or a continuous
glucose measurement, so it cannot answer "how does smoking or BMI change
your risk?" This BRFSS dataset is a 253,680-respondent CDC health survey
that DOES include smoking status, BMI, and diabetes status, and predicts
a different (self-reported) target: whether the respondent has EVER been
told they have heart disease or had a heart attack.

These two datasets are analyzed as two SEPARATE models, never pooled,
because pooling would require pretending they measure the same thing:
- UCI target: angiographically-confirmed CAD presence at time of testing
  (a diagnostic snapshot in a hospital-referral population).
- BRFSS target: self-reported lifetime history of heart disease/heart
  attack (a survey response in a general population).
- Feature overlap is minimal (age, sex, high blood pressure are the only
  loosely-shared concepts; BRFSS has no chest-pain-type, ECG, or
  fluoroscopy data, and UCI has no BMI/smoking/diet data).
Forcing these into one model and calling it "more data" would be exactly
the kind of overclaiming a rigorous project should avoid.

Source: CDC BRFSS 2015 survey, redistributed on Kaggle as
"Heart Disease Health Indicators Dataset"
(alexteboul/heart-disease-health-indicators-dataset). Downloaded directly
by the user from Kaggle and supplied as a zip archive.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "brfss2015.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

TARGET = "HeartDiseaseorAttack"

# Split into two groups for the "which lifestyle factors matter most"
# analysis -- the whole reason this dataset was added to the project.
LIFESTYLE_FEATURES = [
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "BMI",
]
CLINICAL_DEMOGRAPHIC_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Stroke", "Diabetes", "DiffWalk",
    "GenHlth", "MentHlth", "PhysHlth", "AnyHealthcare", "NoDocbcCost",
    "Sex", "Age", "Education", "Income",
]
ALL_FEATURES = LIFESTYLE_FEATURES + CLINICAL_DEMOGRAPHIC_FEATURES

# All columns in this dataset are already numeric 0/1 flags, small ordinal
# codes (Age, GenHlth, Education, Income), or continuous (BMI, MentHlth,
# PhysHlth). Only BMI/MentHlth/PhysHlth/Age/GenHlth/Education/Income get
# scaled as "numeric"; the rest are already meaningful 0/1 flags.
NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth", "Age", "GenHlth",
                     "Education", "Income"]
BINARY_FEATURES = [c for c in ALL_FEATURES if c not in NUMERIC_FEATURES]

DATA_DICTIONARY = [
    {"column": "HeartDiseaseorAttack", "type": "binary (label)", "unit": "1=yes, 0=no",
     "description": "Self-reported: has a doctor/nurse ever told you that you had "
                     "coronary heart disease (CHD) or a myocardial infarction (MI)?"},
    {"column": "HighBP", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Told by a health professional you have high blood pressure."},
    {"column": "HighChol", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Told by a health professional you have high cholesterol."},
    {"column": "CholCheck", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Had a cholesterol check within the last 5 years."},
    {"column": "BMI", "type": "numeric", "unit": "kg/m^2",
     "description": "Body Mass Index, self-reported height/weight."},
    {"column": "Smoker", "type": "binary (lifestyle)", "unit": "1=yes, 0=no",
     "description": "Smoked at least 100 cigarettes in your entire life."},
    {"column": "Stroke", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Ever told you had a stroke."},
    {"column": "Diabetes", "type": "categorical (0/1/2)", "unit": "n/a",
     "description": "0=no diabetes, 1=prediabetes, 2=diabetes. The closest proxy "
                     "to a glucose measurement available in this survey (BRFSS "
                     "does not ask for a lab glucose value)."},
    {"column": "PhysActivity", "type": "binary (lifestyle)", "unit": "1=yes, 0=no",
     "description": "Physical activity in past 30 days, outside of a regular job."},
    {"column": "Fruits", "type": "binary (lifestyle)", "unit": "1=yes, 0=no",
     "description": "Consumes fruit 1+ times per day."},
    {"column": "Veggies", "type": "binary (lifestyle)", "unit": "1=yes, 0=no",
     "description": "Consumes vegetables 1+ times per day."},
    {"column": "HvyAlcoholConsump", "type": "binary (lifestyle)", "unit": "1=yes, 0=no",
     "description": "Heavy drinker (adult men >14 drinks/week, women >7 drinks/week)."},
    {"column": "AnyHealthcare", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Has any kind of health care coverage."},
    {"column": "NoDocbcCost", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Needed to see a doctor in the past year but couldn't because of cost."},
    {"column": "GenHlth", "type": "ordinal (1-5)", "unit": "n/a",
     "description": "Self-rated general health: 1=excellent ... 5=poor."},
    {"column": "MentHlth", "type": "numeric", "unit": "days (0-30)",
     "description": "Days of poor mental health in the past 30 days."},
    {"column": "PhysHlth", "type": "numeric", "unit": "days (0-30)",
     "description": "Days of poor physical health in the past 30 days."},
    {"column": "DiffWalk", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Serious difficulty walking or climbing stairs."},
    {"column": "Sex", "type": "binary", "unit": "1=male, 0=female",
     "description": "Respondent sex."},
    {"column": "Age", "type": "ordinal (1-13)", "unit": "13 age bands",
     "description": "13-level age category (1=18-24 ... 13=80+); not raw years."},
    {"column": "Education", "type": "ordinal (1-6)", "unit": "n/a",
     "description": "Highest education level completed, 1 (none) to 6 (college grad)."},
    {"column": "Income", "type": "ordinal (1-8)", "unit": "n/a",
     "description": "Household income bracket, 1 (<$10k) to 8 (>=$75k)."},
]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df[TARGET] = df[TARGET].astype(int)
    return df


def audit_data(df: pd.DataFrame) -> dict:
    audit = {}
    audit["n_rows_raw"] = len(df)
    audit["n_duplicate_rows"] = int(df.duplicated().sum())
    audit["missing_native"] = int(df.isna().sum().sum())
    audit["target_balance"] = df[TARGET].value_counts().to_dict()
    audit["bmi_range"] = (float(df["BMI"].min()), float(df["BMI"].max()))
    audit["bmi_extreme_high"] = int((df["BMI"] > 60).sum())
    return audit


def write_audit_report(audit: dict) -> Path:
    path = REPORTS_DIR / "brfss_data_audit.md"
    dup_pct = audit["n_duplicate_rows"] / audit["n_rows_raw"]
    lines = [
        "# BRFSS data audit\n",
        f"- Raw rows: **{audit['n_rows_raw']}**",
        f"- Native missing values (entire dataframe): "
        f"**{audit['missing_native']}** (this is a pre-cleaned survey "
        f"extract -- no NaNs to impute)",
        f"- Exact duplicate rows: **{audit['n_duplicate_rows']}** "
        f"({dup_pct:.1%} of the data)",
        f"- Target class balance (0=no history, 1=heart disease/attack "
        f"history): `{audit['target_balance']}` -- substantially more "
        f"imbalanced (~90/10) than the UCI dataset (~55/45)",
        f"- BMI range: {audit['bmi_range'][0]:.0f}-{audit['bmi_range'][1]:.0f}, "
        f"with **{audit['bmi_extreme_high']}** respondents above BMI 60 "
        f"(physiologically extreme but not impossible -- left as-is, "
        f"not recoded to missing, since very high BMI is a real "
        f"condition, unlike a cholesterol reading of exactly 0)",
        "",
        "## Why duplicates are NOT dropped here (unlike the UCI dataset)",
        "With 21 mostly-binary/small-ordinal survey questions answered by "
        "253,680 people, many distinct respondents legitimately share an "
        "identical answer pattern purely by chance -- unlike the UCI "
        "dataset's 2 duplicate ROWS out of 920 (which were almost "
        "certainly genuine double-entries of the same patient), treating "
        "a shared answer pattern across a quarter-million survey rows as "
        "an error and dropping it would silently and systematically bias "
        "the sample toward less common response combinations. This is a "
        "deliberate, disclosed methodological difference from how "
        "duplicates were handled in the UCI dataset, not an "
        "inconsistency.",
    ]
    path.write_text("\n".join(lines))
    return path


def write_data_dictionary() -> Path:
    path = REPORTS_DIR / "brfss_data_dictionary.csv"
    pd.DataFrame(DATA_DICTIONARY).to_csv(path, index=False)
    return path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """No impossible values or missingness to fix in this dataset (it's a
    pre-cleaned survey extract). Duplicates are deliberately NOT dropped
    (see write_audit_report). This function exists for API symmetry with
    the UCI src/data.py module and as the single place future cleaning
    rules would go."""
    return df.reset_index(drop=True)


def get_feature_target(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


if __name__ == "__main__":
    raw = load_raw()
    audit = audit_data(raw)
    write_audit_report(audit)
    write_data_dictionary()
    cleaned = clean_data(raw)
    print("Raw shape:", raw.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Audit:", audit)
