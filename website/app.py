"""
app.py
------
Flask backend for the MedMind website's interactive "what-if" risk
calculator. Re-uses the exact trained pipelines and SHAP explain logic
from the original project (src/data.py, src/explain.py, src/brfss_data.py,
src/brfss_explain.py) -- no model logic is re-implemented here, only a
thin web layer around it, mirroring how heart_app.py (the original
Streamlit dashboard) was built.

Run with:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import base64
import io
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import ALL_FEATURES  # noqa: E402
from explain import make_tree_explainer, transform_for_shap  # noqa: E402
from brfss_data import ALL_FEATURES as BRFSS_ALL_FEATURES  # noqa: E402
from brfss_explain import (  # noqa: E402
    make_tree_explainer as brfss_make_tree_explainer,
    transform_for_shap as brfss_transform_for_shap,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Models (loaded once at startup, same joblib pipelines the Streamlit app
# and the training notebook use)
# ---------------------------------------------------------------------------
UCI_MODEL = joblib.load(PROJECT_ROOT / "models" / "xgb_pipeline.joblib")
UCI_EXPLAINER, UCI_PREPROCESSOR, UCI_FEATURE_NAMES = make_tree_explainer(UCI_MODEL)

BRFSS_MODEL = joblib.load(PROJECT_ROOT / "models" / "brfss_xgb_pipeline.joblib")
BRFSS_EXPLAINER, BRFSS_PREPROCESSOR, BRFSS_FEATURE_NAMES = brfss_make_tree_explainer(BRFSS_MODEL)


def waterfall_png_base64(shap_values_row, feature_names) -> str:
    """Render a SHAP waterfall plot to a base64 PNG data URI, same plot
    heart_app.py shows inline, so it can be dropped straight into an
    <img src="..."> tag."""
    shap_values_row.feature_names = feature_names
    plt.figure(figsize=(6.5, 4.4))
    shap.plots.waterfall(shap_values_row, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def risk_band(pct: float, high: float, mid: float):
    if pct >= high:
        return "#c0392b", "Higher predicted risk"
    if pct >= mid:
        return "#e08e0b", "Moderate predicted risk"
    return "#1e8449", "Lower predicted risk"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculator")
def calculator():
    return render_template("calculator.html")


@app.route("/api/predict/uci", methods=["POST"])
def predict_uci():
    payload = request.get_json(force=True)
    patient = pd.DataFrame([{
        "age": int(payload["age"]),
        "sex": int(payload["sex"]),
        "cp": payload["cp"],
        "trestbps": int(payload["trestbps"]),
        "chol": int(payload["chol"]),
        "fbs": int(payload["fbs"]),
        "restecg": payload["restecg"],
        "thalach": int(payload["thalach"]),
        "exang": int(payload["exang"]),
        "oldpeak": float(payload["oldpeak"]),
        "slope": payload["slope"],
        "ca": int(payload["ca"]),
        "thal": payload["thal"],
    }])[ALL_FEATURES]

    proba = float(UCI_MODEL.predict_proba(patient)[0, 1])
    risk_pct = proba * 100
    color, band = risk_band(risk_pct, 66, 33)

    Xt_row = transform_for_shap(UCI_PREPROCESSOR, patient)
    sv = UCI_EXPLAINER(Xt_row)
    chart = waterfall_png_base64(sv[0], UCI_FEATURE_NAMES)

    return jsonify({
        "risk_pct": round(risk_pct, 1),
        "color": color,
        "band": band,
        "chart": chart,
    })


@app.route("/api/predict/brfss", methods=["POST"])
def predict_brfss():
    payload = request.get_json(force=True)
    respondent = pd.DataFrame([{
        "Smoker": int(payload["smoker"]),
        "PhysActivity": int(payload["physact"]),
        "Fruits": int(payload["fruits"]),
        "Veggies": int(payload["veggies"]),
        "HvyAlcoholConsump": int(payload["hvyalcohol"]),
        "BMI": int(payload["bmi"]),
        "HighBP": int(payload["highbp"]),
        "HighChol": int(payload["highchol"]),
        "CholCheck": 1,
        "Stroke": int(payload["stroke"]),
        "Diabetes": int(payload["diabetes"]),
        "DiffWalk": int(payload["diffwalk"]),
        "GenHlth": int(payload["genhlth"]),
        "MentHlth": 0,
        "PhysHlth": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "Sex": int(payload["sex"]),
        "Age": int(payload["age_band"]),
        "Education": int(payload["education"]),
        "Income": int(payload["income"]),
    }])[BRFSS_ALL_FEATURES]

    proba = float(BRFSS_MODEL.predict_proba(respondent)[0, 1])
    risk_pct = proba * 100
    color, band = risk_band(risk_pct, 40, 15)

    Xt_row = brfss_transform_for_shap(BRFSS_PREPROCESSOR, respondent)
    sv = BRFSS_EXPLAINER(Xt_row)
    chart = waterfall_png_base64(sv[0], BRFSS_FEATURE_NAMES)

    return jsonify({
        "risk_pct": round(risk_pct, 1),
        "color": color,
        "band": band,
        "chart": chart,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
