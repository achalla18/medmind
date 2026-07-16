# AI-Powered Early Heart Disease Risk Prediction

A two-dataset, two-part project. **Part 1** trains and rigorously
evaluates heart disease risk models on two independent public datasets
(logistic regression + XGBoost, leakage-free pipelines, honest
cross-validation, SHAP explainability). **Part 2** is a two-tab
Streamlit "what-if" dashboard — one tab per dataset — where you can
enter a patient/respondent, drag sliders on modifiable risk factors, and
watch the predicted risk and its SHAP explanation update live.

Built for a high-school science-lab submission targeting the "Gold Tier"
rubric — see the checklist at the bottom of this file.

## Quick start

```bash
pip install -r requirements.txt

# Part 1: open and run notebooks/part1_heart_disease_model.ipynb
#   (or reproduce all artifacts from the command line:)
cd src
python train_evaluate.py             # UCI clinical model
python brfss_train.py evaluate       # BRFSS lifestyle model — honest eval on held-out test
python brfss_train.py finalize       # BRFSS lifestyle model — final deployed model

# Part 2: launch the two-tab dashboard
cd ..
streamlit run heart_app.py
```

## Project structure

```
heart_project/
├── data/
│   ├── raw/heart_disease_uci_full.csv   UCI raw source data
│   ├── raw/brfss2015.csv                BRFSS raw source data
│   ├── cleaned.csv                      UCI cleaned data
│   └── brfss_cleaned.csv                BRFSS cleaned data
├── src/
│   ├── data.py / brfss_data.py          loading, data dictionary, audit, cleaning (one pair per dataset)
│   ├── modeling.py / brfss_modeling.py  leakage-free pipelines, evaluation
│   ├── explain.py / brfss_explain.py    SHAP logic shared by notebook + app
│   ├── train_evaluate.py                UCI end-to-end driver
│   └── brfss_train.py                   BRFSS end-to-end driver (two stages: evaluate, finalize)
├── notebooks/
│   └── part1_heart_disease_model.ipynb  Part 1 deliverable — both datasets, fully executed with outputs
├── heart_app.py                         Part 2 deliverable: two-tab Streamlit dashboard
├── models/                              saved pipelines (xgb/lr_pipeline.joblib, brfss_xgb/lr_pipeline.joblib)
├── figures/                             calibration curves, SHAP plots (uci_* and brfss_* prefixed)
├── reports/                             data dictionaries, audits, evaluation metrics, SHAP importance (per dataset)
├── build_notebook.py                    script that assembles the .ipynb (for editing/regenerating it)
└── requirements.txt
```

## The two datasets, and why they're kept separate

### 1. UCI Heart Disease (clinical model) — 920 rows, 918 after cleaning

The full, standard UCI Heart Disease dataset (DOI 10.24432/C52P4X),
combining Cleveland Clinic (304), Hungarian Institute of Cardiology (293),
University Hospital Zurich/Basel Switzerland (123), and V.A. Medical
Center Long Beach (200). Widely redistributed as `heart_disease_uci.csv`
(e.g. Kaggle: `redwankarimsony/heart-disease-data`). Downloaded directly
from Kaggle by the user and supplied manually — noted plainly since
earlier automated fetch attempts (UCI's own archive, several GitHub
mirrors) were blocked or returned truncated files.

Predicts **angiographically-confirmed coronary artery disease presence**
from clinical test results: chest pain type, resting ECG, exercise
stress test results (max heart rate, ST depression, ST slope), vessels
on fluoroscopy (`ca`), and thallium test result (`thal`).

### 2. CDC BRFSS 2015 (lifestyle model) — 253,680 rows

The CDC's 2015 Behavioral Risk Factor Surveillance System survey,
redistributed as "Heart Disease Health Indicators"
(Kaggle: `alexteboul/heart-disease-health-indicators-dataset`). Also
downloaded directly from Kaggle and supplied manually.

Predicts **self-reported lifetime history of heart disease or heart
attack** from survey responses: smoking status, BMI, diet, physical
activity, alcohol use, diabetes status, blood-pressure/cholesterol
flags, self-rated general health, and demographics.

### Why not combine them into one bigger model?

They measure genuinely different things:

| | UCI | BRFSS |
|---|---|---|
| Target | Angiographic CAD presence at time of testing | Self-reported lifetime heart disease/attack history |
| Population | Hospital-referral patients (already being worked up for suspected cardiac disease) | General population survey |
| Has smoking/BMI/diet? | No | Yes |
| Has chest pain/ECG/fluoroscopy/thallium? | Yes | No |
| Size | 918 | 253,680 |

Pooling them would mean pretending a diagnostic test result and a
self-reported survey answer, predicting two different outcomes in two
different populations, are interchangeable data points. They're reported
and modeled **separately**, and the dashboard reflects that with two
clearly-labeled tabs rather than one blended (and quietly misleading)
simulator.

## Headline results

### UCI clinical model — 5-fold nested cross-validation, bootstrap 95% CI

| Model | AUROC (95% CI) | Sensitivity | Specificity | Precision | F1 | Calib. slope | Calib. intercept |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.879 (0.856–0.901) | 0.801 | 0.793 | 0.827 | 0.814 | 1.13 | 0.21 |
| XGBoost | 0.883 (0.860–0.904) | 0.843 | 0.759 | 0.812 | 0.827 | 1.07 | −0.02 |

Both land right around **AUROC ≈ 0.88**, matching the widely-cited
published benchmark (0.886) for this exact dataset. Top SHAP drivers:
asymptomatic chest pain, ST depression, sex, cholesterol,
exercise-induced angina, max heart rate, thallium result, age, and
vessels on fluoroscopy — all established cardiac risk factors.

### BRFSS lifestyle model — single held-out 20% test split, tuned on train only

| Model | AUROC (95% CI) | Sensitivity | Specificity | Precision | F1 | Calib. slope | Calib. intercept |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.847 (0.842–0.852) | 0.797 | 0.749 | 0.248 | 0.379 | 0.91 | −2.24 |
| XGBoost | 0.850 (0.845–0.855) | 0.817 | 0.734 | 0.242 | 0.373 | 1.00 | −2.25 |

AUROC ≈0.85 matches the published benchmark (0.848) for this dataset.
Both models correct for the dataset's ~90/10 class imbalance
(`class_weight="balanced"` / `scale_pos_weight`) — without this, an
uncorrected XGBoost model reached similar AUROC but sensitivity of just
0.10, silently useless as a screening tool despite a deceptively
reasonable-looking headline number. Both calibration intercepts are
meaningfully negative (a textbook side effect of class-weighting,
disclosed in the notebook) — a deployed version would need post-hoc
recalibration before showing raw probabilities to users.

**Which lifestyle factors matter most?** Ranked by SHAP importance: age,
self-rated general health, high blood pressure, sex, and high
cholesterol are the five strongest predictors — all clinical/demographic.
Smoking status, the strongest lifestyle factor, ranks 6th overall.
**Lifestyle factors (smoking, diet, activity, alcohol, BMI) account for
only ~7% of total SHAP-explained importance**, with smoking the
strongest single lifestyle lever. This is a genuine finding (matching
published work on this dataset), not a null result — see the notebook
(Section 10) for why it doesn't mean lifestyle change is ineffective.

## Data audits (full detail in `reports/*_audit.md`)

**UCI:** 2 exact duplicate rows dropped; `cholesterol == 0` in 172 rows
(18.7%, recoded to missing); `resting blood pressure == 0` in 1 row;
heavy **site-driven** missingness in `ca` (66%) and `thal` (53%) —
Cleveland recorded both for ~99% of patients, the other three sites for
as few as 1–10%. Missing At Random conditional on site, imputed inside
the leakage-free pipeline, never dropped.

**BRFSS:** zero native missing values (pre-cleaned survey extract);
~23,900 duplicate rows (9.4%) found but **deliberately not dropped** —
with 21 mostly-binary questions across 253,680 respondents, many people
legitimately share an identical answer pattern by chance, and dropping
those rows would bias the sample toward less common response patterns.
This is a disclosed methodological difference from how UCI's duplicates
were handled, not an inconsistency — the underlying principle (don't let
data handling introduce an unjustified bias) is the same in both cases.

## Part 2: two-tab what-if simulator

**Tab 1 (Clinical Risk, UCI):** sliders for cholesterol, resting blood
pressure, and fasting blood sugar — the modifiable factors that dataset
has. `ca` and `thal` are patient-profile inputs, not sliders, since
they're invasive test results.

**Tab 2 (Lifestyle Risk, BRFSS):** sliders for BMI, smoking status,
physical activity, fruit/vegetable intake, and heavy alcohol use — the
genuine lifestyle sliders the original brief asked for, now backed by a
dataset that actually has them, plus a diabetes-status field as the
closest available glucose proxy (BRFSS doesn't ask for a lab glucose
value).

Both tabs share the same design: patient/respondent profile on the left,
live predicted risk + SHAP waterfall explanation on the right, updating
as sliders move.

## How to defend this project (summary for presentations)

- **Two independent, honestly-scoped datasets**, each disclosed with how
  it was obtained (both downloaded directly from Kaggle after automated
  fetch attempts failed), analyzed separately rather than pooled, with
  the reasoning for that separation stated explicitly.
- **Leakage prevention** in both: UCI uses full nested cross-validation
  (justified by its small size); BRFSS uses a single train/test split
  with tuning confined to the training set only (justified by its size
  making nested CV's extra rigor unnecessary while its computational
  cost is not) — the trade-off is disclosed, not hidden.
- **Class imbalance handled explicitly** in both models on both
  datasets, with a concrete before/after example (BRFSS XGBoost
  sensitivity 0.10 → 0.82) showing why this matters, not just asserting
  that it does.
- **Full metrics, not just accuracy**: AUROC with bootstrap CI,
  sensitivity/specificity/precision/F1, and calibration slope/intercept
  for every model.
- **SHAP explainability** for both datasets, including a genuine
  "lifestyle vs. clinical factors" quantitative finding for BRFSS.
- **Two-tab dashboard**, each tab honestly scoped to what its underlying
  dataset can support, rather than one simulator promising more than any
  single dataset delivers.
- **Honest limitations, stated up front:** research/educational
  prototypes, not diagnostic devices; not validated prospectively or
  reviewed by any regulator; UCI's `ca`/`thal` are missing for most
  non-Cleveland patients; BRFSS is a cross-sectional self-report survey
  that cannot establish causation. Neither should be used to make an
  unsupervised clinical decision.

## Gold-Tier checklist status

| Item | Status |
|---|---|
| Named, well-described datasets with disclosed limitations | Done (two datasets, both sourced and documented) |
| Data dictionaries | Done (`reports/data_dictionary.csv`, `reports/brfss_data_dictionary.csv`) |
| Data audits (duplicates, impossible values, missingness patterns) | Done (`reports/data_audit.md`, `reports/brfss_data_audit.md`) |
| Missing-data handling (no silent row-dropping) | Done for both |
| Leakage-free pipelines (fit on train folds/split only) | Done (`src/modeling.py`, `src/brfss_modeling.py`) |
| Tuned logistic regression baseline | Done for both datasets |
| Strong model (XGBoost), tuned | Done for both datasets |
| Cross-validation appropriate to dataset size | Done — nested CV (UCI), train/test + CV tuning (BRFSS), difference disclosed |
| Class imbalance measured + handled, with a concrete impact example | Done (BRFSS) |
| AUROC with 95% CI | Done for both |
| Sensitivity / specificity / precision / F1 | Done for both |
| Calibration plot + slope/intercept | Done for both (`figures/*calibration*`) |
| SHAP global + per-patient explanations | Done for both (`figures/*shap*`) |
| Lifestyle vs. clinical/demographic feature importance | Done (BRFSS, Section 10) |
| Interactive what-if simulator with live SHAP | Done — two tabs (`heart_app.py`) |
| Multiple independent datasets, honestly compared, not pooled | Done |
| Decision-support framing, limitations, honesty notes | Done (this file + notebook) |
| Fixed seeds, pinned requirements, runnable code | Done (`requirements.txt`, `SEED=42` throughout) |
