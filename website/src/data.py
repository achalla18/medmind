"""
data.py
-------
Loading, renaming, auditing, and cleaning for the full UCI combined Heart
Disease dataset (920 rows, Cleveland + Hungarian + Switzerland + Long Beach
VA), including the `ca` and `thal` features and per-site labels.

Data source
-----------
UCI Machine Learning Repository, "Heart Disease" (DOI 10.24432/C52P4X),
combined release commonly redistributed as `heart_disease_uci.csv`
(e.g. Kaggle: redwankarimsony/heart-disease-data). 920 rows: Cleveland
(304), Hungary (293), Switzerland (123), VA Long Beach (200). Downloaded
directly by the user from Kaggle and supplied as `archive.zip` after
earlier automated fetch attempts (UCI's own archive, GitHub raw mirrors)
were blocked or truncated by network restrictions in the build
environment -- disclosed here because how the data was obtained is part
of an honest methods section, not just what the data contains.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci_full.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# Source column is "thalch"; renamed to the more conventional "thalach"
# used throughout the clinical-ML literature on this dataset.
RENAME_MAP = {"thalch": "thalach"}

# `ca` is a count of vessels (0-3) -- genuinely ordinal/numeric, so it gets
# median imputation + scaling like the other numeric features, not one-hot
# encoding.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
TARGET = "target"
ALL_FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# `dataset` (which hospital/site a row came from) is loaded and used for
# the missingness audit below, but is deliberately EXCLUDED from
# ALL_FEATURES / the model's inputs. Including it would let the model
# learn site-specific measurement conventions (e.g. "rows with no ca/thal
# recorded are probably VA Long Beach") rather than actual clinical risk
# -- a subtle form of leakage even though `dataset` isn't leaking target
# information directly.

DATA_DICTIONARY = [
    {"column": "age", "type": "numeric", "unit": "years",
     "description": "Patient age at the time of the study visit."},
    {"column": "sex", "type": "binary", "unit": "1=male, 0=female",
     "description": "Biological sex as recorded in the source studies."},
    {"column": "cp", "type": "categorical", "unit": "n/a",
     "description": "Chest pain type: typical angina, atypical angina, "
                     "non-anginal pain, or asymptomatic. Asymptomatic "
                     "presentation is counter-intuitively one of the "
                     "strongest predictors of disease in this dataset."},
    {"column": "trestbps", "type": "numeric", "unit": "mm Hg",
     "description": "Resting systolic blood pressure on admission to hospital."},
    {"column": "chol", "type": "numeric", "unit": "mg/dl",
     "description": "Serum cholesterol. A recorded value of 0 is not a "
                     "real cholesterol reading (you cannot have zero serum "
                     "cholesterol and be alive) -- it is a sentinel value "
                     "some sites used for 'not measured'. Treated as "
                     "missing, not as zero, everywhere in this project."},
    {"column": "fbs", "type": "binary", "unit": "1=true, 0=false",
     "description": "Fasting blood sugar > 120 mg/dl."},
    {"column": "restecg", "type": "categorical", "unit": "n/a",
     "description": "Resting electrocardiogram result: normal, ST-T wave "
                     "abnormality, or probable/definite left ventricular "
                     "hypertrophy."},
    {"column": "thalach", "type": "numeric", "unit": "bpm",
     "description": "Maximum heart rate achieved during the exercise stress test."},
    {"column": "exang", "type": "binary", "unit": "1=yes, 0=no",
     "description": "Exercise-induced angina (chest pain during the stress test)."},
    {"column": "oldpeak", "type": "numeric", "unit": "mm ST depression",
     "description": "ST depression induced by exercise relative to rest, "
                     "an ECG marker of reduced blood flow to the heart muscle."},
    {"column": "slope", "type": "categorical", "unit": "n/a",
     "description": "Slope of the peak exercise ST segment: upsloping "
                     "(least concerning), flat, or downsloping (most "
                     "concerning). Missing in about a third of rows -- "
                     "see the data audit for why."},
    {"column": "ca", "type": "numeric (0-3)", "unit": "count of vessels",
     "description": "Number of major coronary vessels colored by "
                     "fluoroscopy, an invasive test. Missing in the large "
                     "majority of non-Cleveland rows (see audit) because "
                     "most sites didn't perform this test."},
    {"column": "thal", "type": "categorical", "unit": "n/a",
     "description": "Thallium stress test result: normal, fixed defect, "
                     "or reversible defect. Also an invasive/specialized "
                     "test, missing in about half the dataset for the "
                     "same site-driven reason as `ca`."},
    {"column": "target", "type": "binary (label)", "unit": "1=disease, 0=no disease",
     "description": "Presence of heart disease, binarized here from the "
                     "source `num` column (an angiographic severity scale "
                     "0-4) into absent (num=0) vs. present (num>=1)."},
]


def load_raw() -> pd.DataFrame:
    """Load the raw CSV, rename thalch -> thalach, and convert the
    True/False text columns to real booleans/ints. The source file
    already uses proper NaN for missing values (not sentinel strings),
    so pandas' default na handling is sufficient here."""
    df = pd.read_csv(RAW_CSV)
    df = df.rename(columns=RENAME_MAP)

    df["sex"] = (df["sex"] == "Male").astype(int)
    df["fbs"] = df["fbs"].map({True: 1, False: 0})
    df["exang"] = df["exang"].map({True: 1, False: 0})

    # num: 0 = no disease, 1-4 = increasing angiographic severity.
    # Binarized here (matches the original assignment scope: presence vs.
    # absence, not severity grading, which is a documented Phase-2-style
    # extension, not attempted in this project).
    df["target"] = (df["num"] > 0).astype(int)

    return df


def audit_data(df: pd.DataFrame) -> dict:
    audit = {}
    audit["n_rows_raw"] = len(df)
    audit["n_duplicate_rows"] = int(df.drop(columns=["id"]).duplicated().sum())
    audit["missing_native"] = df[ALL_FEATURES].isna().sum().to_dict()
    audit["chol_zero_count"] = int((df["chol"] == 0).sum())
    audit["trestbps_zero_count"] = int((df["trestbps"] == 0).sum())
    audit["target_balance"] = df["target"].value_counts().to_dict()
    audit["site_counts"] = df["dataset"].value_counts().to_dict()
    # Site-driven missingness -- the key finding that explains WHY ca/thal
    # are missing so often: it's not random, it's which hospital the
    # patient was seen at.
    audit["missing_by_site"] = (
        df.groupby("dataset")[["ca", "thal", "slope"]]
        .apply(lambda g: g.isna().mean())
        .round(3)
        .to_dict()
    )
    return audit


def write_audit_report(audit: dict) -> Path:
    path = REPORTS_DIR / "data_audit.md"
    missing_by_site_lines = []
    for col, site_dict in audit["missing_by_site"].items():
        parts = ", ".join(f"{site}: {pct:.0%}" for site, pct in site_dict.items())
        missing_by_site_lines.append(f"  - `{col}`: {parts}")

    lines = [
        "# Data audit\n",
        f"- Raw rows: **{audit['n_rows_raw']}** across sites "
        f"`{audit['site_counts']}`",
        f"- Exact duplicate rows found (excluding the `id` column): "
        f"**{audit['n_duplicate_rows']}** (dropped before any train/test "
        f"split, keeping the first occurrence)",
        f"- Natively-missing (NaN) values per model feature: "
        f"`{audit['missing_native']}`",
        f"- `chol == 0` (impossible, recoded to missing): "
        f"**{audit['chol_zero_count']}** rows "
        f"({audit['chol_zero_count'] / audit['n_rows_raw']:.1%} of the data)",
        f"- `trestbps == 0` (impossible, recoded to missing): "
        f"**{audit['trestbps_zero_count']}** rows",
        f"- Target class balance (0=no disease, 1=disease): "
        f"`{audit['target_balance']}`",
        "",
        "## Missingness is driven by site, not chance",
        "`ca` (fluoroscopy) and `thal` (thallium stress test) are both "
        "invasive/specialized tests. Missingness by site:",
        *missing_by_site_lines,
        "",
        "Cleveland performed both tests on almost every patient; the "
        "other three sites performed them rarely or never. This is "
        "**Missing At Random conditional on site**, not Missing "
        "Completely At Random -- dropping every row with a missing `ca` "
        "or `thal` would both shrink the sample sharply and bias it "
        "toward Cleveland-only patients, which is exactly why those "
        "rows are imputed (inside the leakage-free pipeline, fit on "
        "training folds only) rather than dropped.",
        "",
        "## Why the chol==0 finding matters",
        "`chol == 0` affects nearly a fifth of rows with a valid "
        "cholesterol field. If left as-is, a model would learn that a "
        "cholesterol reading of exactly 0 is strongly associated with "
        "disease status -- a data-artifact correlation (which site "
        "didn't measure cholesterol), not a medical one.",
        "",
        "## Feature retained for diagnostics only",
        "The `dataset` (site) column is used above to explain *why* "
        "missingness looks the way it does, but is deliberately excluded "
        "from the model's input features (see `ALL_FEATURES` in "
        "`src/data.py`) -- including it would let the model learn "
        "site-specific measurement conventions instead of clinical risk.",
    ]
    path.write_text("\n".join(lines))
    return path


def write_data_dictionary() -> Path:
    path = REPORTS_DIR / "data_dictionary.csv"
    pd.DataFrame(DATA_DICTIONARY).to_csv(path, index=False)
    return path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix what the audit found. Nothing here drops rows for missingness
    -- impossible values are recoded to NaN and left for the pipeline's
    imputer. Only exact duplicate rows (excluding the meaningless `id`
    column) are dropped, once, before any split."""
    df = df.copy()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
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
    print("Cleaned shape (after dedup):", cleaned.shape)
    print("chol missing after cleaning:", cleaned["chol"].isna().sum())
    print("ca missing:", cleaned["ca"].isna().sum())
    print("thal missing:", cleaned["thal"].isna().sum())
    print("Wrote:", REPORTS_DIR / "data_audit.md")
    print("Wrote:", REPORTS_DIR / "data_dictionary.csv")
