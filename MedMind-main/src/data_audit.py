"""
Data audit for the UCI combined Heart Disease dataset (Cleveland + Hungary +
Switzerland + VA Long Beach, n=920).

Produces:
  - reports/data_dictionary.csv   (per-feature documentation)
  - reports/data_audit.md         (human-readable audit findings)

Run: python3 src/data_audit.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "heart_disease_uci_combined.csv"
OUT_DICT = BASE / "reports" / "data_dictionary.csv"
OUT_MD = BASE / "reports" / "data_audit.md"

df = pd.read_csv(RAW)

# ---- 1. Basic shape / duplicate / unit checks -----------------------------
n_rows, n_cols = df.shape
n_dupe_rows = df.duplicated(subset=[c for c in df.columns if c != "id"]).sum()
n_dupe_ids = df["id"].duplicated().sum()

# ---- 2. Impossible / physiologically implausible values -------------------
# cholesterol of 0 mg/dL and resting BP of 0 mmHg are not physiologically
# possible in a living patient -- these are known encoding artifacts for
# "not measured" in the non-Cleveland UCI sites, not true zero values.
impossible = {
    "chol == 0": int((df["chol"] == 0).sum()),
    "trestbps == 0": int((df["trestbps"] == 0).sum()),
    "age <= 0": int((df["age"] <= 0).sum()),
}

# ---- 3. Missingness by feature and by site ---------------------------------
missing_overall = df.isna().sum()
missing_pct = (missing_overall / n_rows * 100).round(1)
missing_by_site = df.groupby("dataset").apply(lambda g: g.isna().mean() * 100).round(1)

# ---- 4. Data dictionary -----------------------------------------------------
data_dict = [
    {"feature": "id", "type": "int", "unit": "n/a", "valid_range": "1-920",
     "description": "Row identifier. Dropped before modeling.", "missing_pct": 0.0},
    {"feature": "age", "type": "int", "unit": "years", "valid_range": "28-77",
     "description": "Patient age at time of study.", "missing_pct": 0.0},
    {"feature": "sex", "type": "categorical", "unit": "n/a", "valid_range": "Male/Female",
     "description": "Biological sex as recorded in source studies.", "missing_pct": 0.0},
    {"feature": "dataset", "type": "categorical", "unit": "n/a",
     "valid_range": "Cleveland/Hungary/Switzerland/VA Long Beach",
     "description": "Recruiting site/source study. Used for site-level missingness "
                     "analysis only, NOT as a model feature (would leak site-specific "
                     "measurement conventions).", "missing_pct": 0.0},
    {"feature": "cp", "type": "categorical", "unit": "n/a",
     "valid_range": "typical angina/atypical angina/non-anginal/asymptomatic",
     "description": "Chest pain type.", "missing_pct": float(missing_pct.get("cp", 0))},
    {"feature": "trestbps", "type": "float", "unit": "mmHg", "valid_range": "~80-220 physiological; 0 = missing sentinel",
     "description": "Resting systolic blood pressure on admission. 0 mmHg values are "
                     "physiologically impossible and treated as missing.",
     "missing_pct": float(missing_pct.get("trestbps", 0))},
    {"feature": "chol", "type": "float", "unit": "mg/dL", "valid_range": "~100-450 physiological; 0 = missing sentinel",
     "description": "Serum cholesterol. 0 mg/dL values are physiologically impossible "
                     "and treated as missing.", "missing_pct": float(missing_pct.get("chol", 0))},
    {"feature": "fbs", "type": "categorical (bool)", "unit": "n/a", "valid_range": "True/False",
     "description": "Fasting blood sugar > 120 mg/dL.", "missing_pct": float(missing_pct.get("fbs", 0))},
    {"feature": "restecg", "type": "categorical", "unit": "n/a",
     "valid_range": "normal/st-t abnormality/lv hypertrophy",
     "description": "Resting electrocardiographic results.", "missing_pct": float(missing_pct.get("restecg", 0))},
    {"feature": "thalch", "type": "float", "unit": "bpm", "valid_range": "60-202",
     "description": "Maximum heart rate achieved during exercise test.",
     "missing_pct": float(missing_pct.get("thalch", 0))},
    {"feature": "exang", "type": "categorical (bool)", "unit": "n/a", "valid_range": "True/False",
     "description": "Exercise-induced angina.", "missing_pct": float(missing_pct.get("exang", 0))},
    {"feature": "oldpeak", "type": "float", "unit": "mm ST depression", "valid_range": "-2.6 to 6.2",
     "description": "ST depression induced by exercise relative to rest.",
     "missing_pct": float(missing_pct.get("oldpeak", 0))},
    {"feature": "slope", "type": "categorical", "unit": "n/a", "valid_range": "upsloping/flat/downsloping",
     "description": "Slope of peak exercise ST segment. Heavily missing outside Cleveland.",
     "missing_pct": float(missing_pct.get("slope", 0))},
    {"feature": "ca", "type": "float (ordinal count)", "unit": "vessels", "valid_range": "0-3",
     "description": "Number of major vessels colored by fluoroscopy. Heavily missing "
                     "outside Cleveland (only site that performed this test consistently).",
     "missing_pct": float(missing_pct.get("ca", 0))},
    {"feature": "thal", "type": "categorical", "unit": "n/a",
     "valid_range": "normal/fixed defect/reversable defect",
     "description": "Thallium stress test result. Heavily missing outside Cleveland.",
     "missing_pct": float(missing_pct.get("thal", 0))},
    {"feature": "num", "type": "int (target)", "unit": "n/a", "valid_range": "0-4",
     "description": "Angiographic disease severity (0 = no significant narrowing, "
                     "1-4 = increasing severity in vessels with >50% diameter narrowing). "
                     "Phase 1 binarizes this to num>0 = disease present. Phase 2 "
                     "(roadmap) uses the full 0-4 ordinal scale for severity prediction.",
     "missing_pct": 0.0},
]
pd.DataFrame(data_dict).to_csv(OUT_DICT, index=False)

# ---- 5. Write human-readable audit report ----------------------------------
lines = []
lines.append("# Data Audit -- UCI Combined Heart Disease Dataset\n")
lines.append(f"- Source: UCI Heart Disease repository, combined multi-site release "
             f"(Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital "
             f"Zurich/Basel Switzerland, VA Long Beach).\n"
             f"- Rows: {n_rows}, Columns: {n_cols}\n"
             f"- Duplicate rows (excluding id): {n_dupe_rows}\n"
             f"- Duplicate ids: {n_dupe_ids}\n")
lines.append("## Site composition\n")
lines.append(df["dataset"].value_counts().to_frame("n").to_markdown() + "\n")
lines.append("## Target distribution (num, 0-4 severity)\n")
lines.append(df["num"].value_counts().sort_index().to_frame("n").to_markdown() + "\n")
lines.append("Binarized (num>0 = disease present):\n")
lines.append((df["num"] > 0).value_counts().to_frame("n").to_markdown() + "\n")
lines.append("## Physiologically impossible values (treated as missing sentinels)\n")
for k, v in impossible.items():
    lines.append(f"- {k}: {v} rows\n")
lines.append("\n## Missingness by feature (overall %)\n")
lines.append(missing_pct[missing_pct > 0].sort_values(ascending=False).to_frame("pct_missing").to_markdown() + "\n")
lines.append("\n## Missingness by feature and site (%) -- shows missingness is site-driven, "
             "not random (MAR conditional on site, not MCAR)\n")
lines.append(missing_by_site[["trestbps", "chol", "fbs", "restecg", "thalch", "exang",
                                "oldpeak", "slope", "ca", "thal"]].to_markdown() + "\n")
lines.append("\n## Interpretation\n")
lines.append(
"Missingness is heavily concentrated in `ca`, `thal`, and `slope`, and is almost "
"entirely explained by recruiting site: Cleveland performed fluoroscopy and thallium "
"stress testing consistently (near-complete), while Hungary, Switzerland, and VA Long "
"Beach did not. This is Missing At Random (MAR) conditional on site -- not Missing "
"Completely At Random -- so listwise deletion would both shrink the sample sharply "
"and bias it toward Cleveland-only patients. We use multivariate iterative imputation "
"(IterativeImputer) fit on the training folds only, and additionally retain `dataset` "
"as a masking indicator during EDA (but not as a model feature) to make this "
"mechanism auditable.\n"
"\n`chol` and `trestbps` contain 0-valued sentinels that are physiologically "
"impossible (living patients cannot have 0 mg/dL cholesterol or 0 mmHg blood "
"pressure); these are recoded to NaN before imputation.\n"
)
OUT_MD.write_text("".join(lines))
print(f"Wrote {OUT_DICT}")
print(f"Wrote {OUT_MD}")
print("\n--- summary ---")
print(f"rows={n_rows} cols={n_cols} dupe_rows={n_dupe_rows} dupe_ids={n_dupe_ids}")
print(impossible)
