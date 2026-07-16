# MedMind Website

A website for the MedMind heart-disease risk prediction project
(https://github.com/achalla18/medmind). Two parts:

1. **Showcase page** (`/`) — project overview, methodology, headline
   results (AUROC, sensitivity/specificity tables), and evaluation
   figures (ROC curves, calibration, SHAP global importance).
2. **Interactive risk calculator** (`/calculator`) — a browser version of
   the project's Streamlit "what-if" simulator (`heart_app.py`). Two
   tabs, one per dataset (UCI clinical, CDC BRFSS lifestyle survey).
   Enter a profile, drag the sliders, and get a live predicted risk % and
   a SHAP waterfall explanation — powered by the exact same trained
   pipelines (`models/*.joblib`) and SHAP logic (`src/*.py`) as the
   original project, not re-implemented.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in a browser.

## Structure

```
app.py                  Flask backend: serves both pages + /api/predict/uci and /api/predict/brfss
src/                     Copied unchanged from the original repo (data.py, explain.py, brfss_data.py, brfss_explain.py)
models/                  Copied unchanged: xgb_pipeline.joblib, brfss_xgb_pipeline.joblib
templates/index.html    Showcase / landing page
templates/calculator.html   Interactive two-tab risk calculator
static/css/style.css     Shared styling
static/js/calculator.js  Wires the calculator's sliders to the backend and renders results live
static/figures/          Evaluation figures (ROC, calibration, SHAP) shown on the landing page
requirements.txt
```

## Notes

- This is a from-scratch web front end + Flask backend built for this
  project; it is not the original Streamlit app (`heart_app.py`), though
  it reuses that app's exact model-loading and SHAP-explanation logic so
  predictions match exactly.
- Same disclaimer as the original project: educational decision-support
  prototype, not a diagnostic device. Model outputs are not calibrated
  clinical probabilities — see the project's `reports/` for calibration
  analysis.
