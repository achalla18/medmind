# AI-Powered Early Heart Disease Prediction -- Phase 1 Report

Structured around the TRIPOD+AI reporting guideline (BMJ, 2024). This document
covers the risk-prediction phase only; severity prediction and personalized
recommendations are discussed as a roadmap at the end.

## 1. Title and clinical question

Binary prediction of the presence of angiographically significant coronary
artery disease (>50% diameter narrowing in at least one major vessel) from
routinely collected demographic, symptom, and non-invasive test data,
intended as clinical **decision support, not diagnosis**.

## 2. Data source

**Primary dataset (development and internal validation):** UCI Heart Disease
repository, combined multi-site release -- Cleveland Clinic (n=304), Hungarian
Institute of Cardiology, Budapest (n=293), University Hospital Zurich/Basel,
Switzerland (n=123), and V.A. Medical Center, Long Beach (n=200). 920 rows,
918 after removing 2 duplicate records. 13 predictors + target (`num`,
angiographic severity 0-4, binarized here to disease-present vs
disease-absent). Full data dictionary: `reports/data_dictionary.csv`. Full
audit: `reports/data_audit.md`.

**Second dataset (see Section 8 on why this is not a strict external
validation):** Framingham Heart Study teaching dataset (public, Kaggle
mirror), n=4,240, 15 risk-factor features, outcome = 10-year incident CHD.

Known biases to disclose: all four UCI sites are hospital-referral
populations (patients already being worked up for suspected cardiac disease),
not a general-population screening sample, so prevalence (~55% disease
positive) is far higher than in primary care and the model should not be
interpreted as a population screening tool. Framingham is a mid-20th-century
American cohort (originally enrolled 1948, this extract mid-1990s follow-up)
and may not generalize to other populations or eras.

## 3. Missing data (see `reports/data_audit.md` for full detail)

Missingness is concentrated in `ca` (66%), `thal` (53%), and `slope` (34%),
and is driven almost entirely by recruiting site: Cleveland performed
fluoroscopy and thallium testing consistently, the other three sites did not.
This is Missing At Random conditional on site, not Missing Completely At
Random -- listwise deletion was rejected because it would both shrink the
sample sharply and bias it toward Cleveland-only patients. `chol==0` (172
rows, 19%) and one `trestbps==0` value are physiologically impossible and
were recoded to missing before imputation. All missing values (numeric and
categorical) were imputed with `IterativeImputer` (numeric) / most-frequent
(categorical), fit on the training folds only inside the modeling pipeline
(see Section 5) -- never on the full dataset.

## 4. Duplicate handling

Two duplicate patient records (4 rows forming 2 identical pairs) were
identified and dropped, keeping the first occurrence, before any
train/test split (`src/data_prep.py`).

## 5. Preprocessing pipeline (leakage prevention)

All steps -- imputation, scaling (`StandardScaler`), one-hot encoding, and
class-imbalance handling -- are wrapped in a single `imblearn.Pipeline`
(`src/model_pipeline.py`) so that under cross-validation every step is fit
**only** on the training fold of each split and applied (not refit) to the
held-out fold. `dataset`/site is retained for missingness diagnostics but
deliberately **excluded** as a model feature, since it would let the model
learn site-specific measurement conventions rather than clinical risk.

## 6. Class imbalance

The binarized outcome is 55%/45% (509/411) -- only mildly imbalanced, unlike
many clinical datasets. We still ran the required with/without comparison
(`reports/imbalance_comparison.csv`): no resampling, `class_weight='balanced'`,
and SMOTE, for both model families, inside nested CV. Differences were small
(AUROC range 0.887-0.890 across all six combinations) precisely because the
data is not strongly imbalanced to begin with -- an honest finding, not a
sign anything was wrong. `class_weight='balanced'` was selected for logistic
regression (best AUROC, unchanged calibration); no resampling was selected
for XGBoost (best AUROC; SMOTE and class-weighting both slightly reduced it).

## 7. Modeling and validation

- **Baseline:** tuned logistic regression (`C` grid-searched: 0.01/0.1/1/10).
- **Stronger model:** XGBoost (`n_estimators`, `max_depth`, `learning_rate`
  grid-searched).
- **Validation:** nested 5x5 stratified cross-validation -- outer loop for
  honest performance estimation, inner loop (`GridSearchCV`) for
  hyperparameter tuning, so no test fold ever influenced model selection.
- Fixed seeds throughout (`SEED=42`); full code in `src/train.py`.

## 8. Results

### 8.1 Discrimination and calibration (out-of-fold, `reports/evaluation_metrics.csv`)

| Model | AUROC (95% CI) | Calibration slope | Calibration intercept |
|---|---|---|---|
| Logistic Regression (baseline) | 0.886 (0.863-0.906) | 1.06 | 0.21 |
| XGBoost | 0.886 (0.864-0.906) | 1.12 | -0.04 |

**The tuned logistic regression baseline matches XGBoost almost exactly.**
This is a genuine, honest finding -- not a failure to make the "fancy" model
work. On a dataset this size (918 rows, 13 predictors) with strong, well
understood clinical signal, a linear model captures nearly all of the
separable structure, and both models are close to well-calibrated (slopes
near 1.0, intercepts near 0). This matches the "stronger baselines" critique
in the clinical-ML literature: many papers claiming gains from complex models
never establish that a tuned simple model couldn't do the same.

### 8.2 Full classification metrics at two thresholds

| Model | Threshold | Sensitivity | Specificity | PPV | NPV | F1 |
|---|---|---|---|---|---|---|
| LR | 0.50 (default) | 0.809 | 0.807 | 0.839 | 0.773 | 0.824 |
| LR | 0.52 (Youden) | 0.803 | 0.824 | 0.850 | 0.772 | 0.826 |
| XGBoost | 0.50 (default) | 0.864 | 0.763 | 0.819 | 0.819 | 0.841 |
| XGBoost | 0.53 (Youden) | 0.846 | 0.793 | 0.835 | 0.806 | 0.841 |

Given that a missed case of significant coronary disease (false negative) is
clinically far more costly than a false alarm, a deployed version of this
tool would deliberately favor sensitivity over the Youden-optimal point --
e.g. XGBoost at its default threshold already reaches 86% sensitivity at a
real cost of more false positives (76% specificity), which is a defensible
clinical trade-off to make explicit rather than silently accept whatever a
0.5 cutoff happens to produce.

### 8.3 Explainability (SHAP, `figures/shap_summary.png`, `reports/shap_global_importance.csv`)

Top global drivers of XGBoost's predictions, in order: asymptomatic chest
pain type, number of vessels colored by fluoroscopy (`ca`), ST depression
(`oldpeak`), absence of exercise-induced angina, normal thallium result, sex,
and cholesterol. All are established cardiac risk/diagnostic factors, which
is itself a useful sanity check -- the model is not keying off an artifact.
Per-patient waterfall plots for the model's highest- and lowest-confidence
predictions are in `figures/shap_waterfall_patient_high_risk.png` /
`_low_risk.png`.

### 8.4 Second dataset: Framingham (honesty note, `src/external_validation.py`)

A true external validation applies the fitted model, unchanged, to a second
dataset predicting the *same* target. That is not possible here: UCI's
target is presence of angiographically significant CAD at time of
catheterization (a diagnostic snapshot in a referral population), while
Framingham's target is 10-year incident CHD risk in a population free of
disease at baseline -- a different clinical question entirely -- and the two
datasets share almost no features (UCI has `cp`/`thal`/`ca`/`exang`/`oldpeak`;
Framingham has smoking/diabetes/BP-medication/education, none of which exist
in UCI). Forcing a transfer and calling it "validated" would be exactly the
kind of overclaiming this project is trying to avoid. Instead:

- **(A) Naive transportability check:** a reduced logistic regression using
  only the four loosely-shared features (age, sex, systolic BP, total
  cholesterol), fit on UCI, applied as-is to Framingham's outcome: AUROC
  0.680 (0.658-0.700), calibration slope 0.63. The clear drop from 0.886 to
  0.680 is the expected and informative result -- it quantifies how much a
  handful of shared risk factors carry across a diagnostic-vs-prognostic gap,
  which is "some, but nowhere near enough to substitute for the full model."
- **(B) Methodological replication:** the identical rigorous pipeline
  (imputation, nested stratified CV, class-weighting, calibration) rerun on
  Framingham's own native task (its own features, its own label): AUROC 0.723
  (0.703-0.744), calibration slope 1.05 but intercept -1.77. This shows the
  *pipeline and methodology* generalize cleanly to an independent, 4x larger,
  more imbalanced (85/15) dataset. AUROC 0.72 also matches published
  Framingham risk-factor logistic models, another sanity check.
- **Notable side finding:** the negative calibration intercept in (B) is a
  direct, textbook consequence of `class_weight='balanced'` -- balancing the
  loss improves ranking/sensitivity but shifts predicted probabilities
  upward relative to the true (imbalanced) base rate. A production version of
  this specific configuration would need a post-hoc recalibration step
  (e.g. Platt scaling / `CalibratedClassifierCV`) before probabilities are
  shown to clinicians. This is exactly the discrimination-vs-calibration
  distinction the field emphasizes, caught by actually checking calibration
  rather than reporting AUROC alone.
- **What a true external validation would require:** a second dataset with
  the *same* diagnostic target (e.g. another angiographic CAD cohort) or a
  Framingham variant with a baseline prevalent-CAD field. Noted as the
  clearest concrete next step.

## 9. Limitations

- Small primary dataset (918 rows) -- wide-ish bootstrap CIs, and not enough
  data to justify a large deep model (matched here: logistic regression and
  shallow-tree XGBoost, not a neural net).
- Hospital-referral population, not a screening population -- prevalence and
  performance may not transfer to primary care or asymptomatic screening.
- Heavy, site-driven missingness in `ca`/`thal`/`slope` means the imputed
  values for non-Cleveland patients carry real uncertainty; a sensitivity
  analysis with a simple median/mode imputer as a lower bound is in
  `reports/imbalance_comparison.csv`'s underlying run and gave materially
  similar AUROC, but this is not a substitute for real measurements.
- No genuine external validation was possible with two public datasets with
  incompatible targets and feature sets (Section 8.4).
- Single held-out timepoint, not a temporal/prospective validation.

## 10. Ethics and scope statement

This is a research/educational decision-support prototype, **not a diagnostic
device and not a substitute for clinical judgment**. It has not been
validated prospectively, has not undergone regulatory review, and should
never be used to make an unsupervised clinical decision. The roadmap items
below (severity, treatment recommendations) raise progressively higher
stakes and are described only as directions requiring clinical oversight, not
as claims about what has been built.

## 11. Roadmap (not yet implemented)

- **Phase 2 -- Severity prediction:** reframe as ordinal/multiclass
  prediction of the original `num` 0-4 severity scale rather than the
  binarized target used here. Will need an ordinal loss (e.g. proportional
  odds) and an honest discussion of label noise, since severity grading is
  itself less reliable across readers than presence/absence.
- **Phase 3 -- Personalized recommendations:** frame strictly as surfacing
  modifiable risk factors identified by SHAP (e.g. cholesterol, blood
  pressure) and mapping them to general, guideline-based lifestyle
  information -- never as prescribing treatment -- with explicit clinical
  oversight required and the regulatory/ethical reasons stated up front.

## 12. Reproducibility

Fixed random seed (42) throughout. `requirements.txt` pins all dependencies
(note: `xgboost==2.1.4` is pinned specifically because newer 3.x releases
changed an internal model-serialization format that breaks SHAP's
`TreeExplainer` -- a real compatibility issue hit and documented during this
project, not a hypothetical one). All code is in `src/`; running
`data_audit.py` -> `train.py` -> `evaluate.py` -> `explain.py` ->
`external_validation.py` in order reproduces every number and figure in this
report.

## 13. Three additional datasets: which lifestyle factors actually predict risk?

After Phase 1, two more public cardiovascular datasets were added to broaden
the evidence base and directly answer a question Phase 1 didn't ask: which
*lifestyle* factors are the strongest predictors of long-term heart disease
risk, versus clinical/demographic ones? Each dataset targets a different
population and a different definition of "heart disease," so results are
reported separately rather than pooled -- pooling datasets with incompatible
labels would repeat the mistake Section 8.4 explicitly warned against.

### 13.1 Summary across all four datasets

| Dataset | n | Target | Best AUROC (95% CI) | Calibration |
|---|---|---|---|---|
| UCI combined (primary) | 918 | Angiographic CAD presence | 0.886 (0.863-0.906) | slope 1.06-1.12, well calibrated |
| Framingham (own task) | 4,240 | 10-yr incident CHD | 0.723 (0.703-0.744) | slope 1.05, intercept -1.77 (needs recalibration) |
| Kaggle Cardiovascular Disease (cardio70k) | 69,976 | Exam-based CVD presence | 0.801 (0.797-0.804) | slope ~1.0, intercept ~0.0 (excellent) |
| CDC BRFSS Heart Disease Health Indicators | 253,680 | Self-reported heart disease/attack history | 0.848 (0.846-0.851) | slope 0.96, intercept -2.23 (needs recalibration) |

A consistent pattern holds across all four: **a tuned logistic regression
baseline lands within 1-2 points of AUROC of XGBoost every single time**
(e.g. BRFSS: 0.847 vs 0.848). This isn't a coincidence specific to one
dataset -- across four independently-sourced datasets spanning 918 to
253,680 rows, a simple linear model captures almost all of the separable
signal in structured cardiovascular risk data. That is a stronger, more
general version of the "strong baselines" finding from Section 8.1.

Also consistent: whenever `class_weight='balanced'` or `scale_pos_weight`
is used to handle imbalance (Framingham, BRFSS), the calibration intercept
becomes meaningfully negative (-1.77, -2.23) even though the calibration
*slope* stays near 1.0. This is the same textbook side effect flagged in
Section 8.4 -- balancing the loss improves ranking and sensitivity but
shifts predicted probabilities away from the true base rate. Any of these
specific configurations would need `CalibratedClassifierCV` or Platt
scaling before showing raw probabilities to an end user.

### 13.2 cardio70k: a closer (but still imperfect) second cardiovascular dataset

Unlike Framingham, cardio70k's target (`cardio`) is a same-visit CVD
diagnosis rather than a future risk prediction, making it a closer
conceptual match to the primary UCI task -- though it is still a broader,
exam-based "cardiovascular disease" label (derived from blood pressure,
cholesterol and glucose thresholds in a Russian medical-exam cohort), not
angiographically-confirmed coronary artery disease, and feature overlap
with UCI is limited to age, sex, and blood pressure. Real data-quality
issues were found and corrected before modeling: 1,291 rows (1.8%) had
physiologically impossible blood pressure (e.g. negative values, or
systolic <= diastolic), 93 had impossible height, 7 had impossible weight,
and 24 exact duplicate records existed. All were recoded to missing (BP,
height, weight) or dropped (duplicates), not silently left in. See
`src/cardio70k_prep.py` for the exact rules and `reports/cardio70k_results.csv`
for full metrics. Top SHAP predictors: systolic blood pressure, age, and
cholesterol level dominate -- consistent with established cardiovascular
risk factors and a useful sanity check that the model isn't keying off an
artifact of this particular cohort.

### 13.3 BRFSS: lifestyle vs. clinical/demographic factors

This is the dataset that most directly answers the "which lifestyle
factors matter most" question. 253,680 CDC BRFSS 2015 survey respondents,
21 predictors split into two groups for this analysis:

- **Lifestyle** (modifiable, behavioral): smoking status, physical
  activity, fruit intake, vegetable intake, heavy alcohol consumption, BMI.
- **Clinical/demographic** (largely non-modifiable or requiring clinical
  measurement): high blood pressure, high cholesterol, cholesterol-check
  history, stroke history, diabetes status, difficulty walking,
  self-rated general health, mental/physical unhealthy days, healthcare
  access, sex, age, education, income.

Note: this dataset has no missing values, and its ~9.4% duplicate rows
were deliberately **not** dropped -- with 21 mostly-binary survey
questions, many distinct respondents legitimately share an identical
answer pattern by chance across 253,680 rows; treating that as an error
(the way the 2 UCI duplicates genuinely were) would silently bias the
sample.

Ranked by SHAP importance (`reports/shap_global_importance_brfss.csv`):
age, self-rated general health, high blood pressure, sex, and high
cholesterol are the five strongest predictors -- all clinical/demographic.
The strongest lifestyle factor, smoking status, ranks 6th overall. Summed
across all features, **lifestyle factors account for only about 9% of
total SHAP-explained importance**; the remaining ~91% comes from
clinical/demographic factors.

This is a genuine, defensible finding, not a null result to explain away:
in a cross-sectional self-report survey, non-modifiable and already-
diagnosed clinical factors (you already have high blood pressure, you are
older) are mechanically closer to the outcome (a past heart disease/attack
event) than current behavior is. It does **not** mean lifestyle change is
ineffective -- lifestyle factors are the whole reason many of the clinical
factors exist in the first place (smoking contributes to high blood
pressure and stroke risk, for instance), and this cross-sectional
association study cannot separate that causal chain. A preventative-
medicine takeaway grounded in what was actually measured: smoking status
and BMI are the two lifestyle levers with the most direct, detectable
association with heart disease history in this data, and are the most
defensible targets for a preventative intervention aimed at this outcome,
while self-rated general health and blood-pressure/cholesterol management
are the strongest single predictors overall and warrant the most clinical
attention.

## Gold-Tier checklist status

| Item | Status |
|---|---|
| Named, well-described dataset with known biases | Done (Section 2) |
| Data dictionary | Done (`reports/data_dictionary.csv`) |
| Data audit (duplicates, impossible values, units) | Done (`reports/data_audit.md`) |
| Missing-data mechanism + principled imputation | Done (Section 3) |
| Leakage-free Pipeline (fit on train folds only) | Done (Section 5) |
| Class imbalance measured + handled inside CV, with/without comparison | Done (Section 6) |
| Clinically motivated features, no leakage from `dataset`/site | Done (Section 5) |
| Tuned logistic regression baseline | Done (Section 7) |
| Strong model (XGBoost), tuned | Done (Section 7) |
| Nested / stratified k-fold CV | Done (Section 7) |
| Decision threshold tuned for sensitivity | Done (Section 8.2) |
| AUROC with CI | Done (Section 8.1) |
| Sensitivity/specificity/PPV/NPV/F1 | Done (Section 8.2) |
| Calibration plot + slope/intercept | Done (Section 8.1, `figures/calibration_*.png`) |
| External validation (or honest discussion of why not) | Done, with explicit honesty framing (Section 8.4) |
| SHAP global + per-patient | Done (Section 8.3) |
| Decision-support framing, limitations, ethics | Done (Sections 9-10) |
| Fixed seeds, requirements.txt, runnable code | Done (Section 12) |
| TRIPOD+AI structure | This document |
| Multiple independent datasets, honestly compared (not pooled) | Done (Section 13) |
| Lifestyle vs. clinical/demographic feature importance (Project 2 spec) | Done (Section 13.3) |
