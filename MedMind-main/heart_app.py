"""
heart_app.py
------------
Two-tab "what-if" risk simulator, one tab per dataset used in this
project:

  Tab 1 -- Clinical Risk (UCI): trained on the 918-row UCI Heart Disease
  dataset. Sliders cover cholesterol, blood pressure, and fasting blood
  sugar -- the modifiable factors that dataset actually has.

  Tab 2 -- Lifestyle Risk (BRFSS): trained on the 253,680-row CDC BRFSS
  survey. Sliders cover smoking, BMI, physical activity, diet, and heavy
  alcohol use -- the modifiable lifestyle factors THAT dataset has, which
  the UCI dataset does not.

WHY TWO TABS INSTEAD OF ONE BLENDED MODEL:
The original brief asked for one simulator with sliders on cholesterol,
blood pressure, smoking, BMI, and glucose. No single public dataset used
in this project has all five as real, trainable features -- UCI has the
first two plus a glucose flag but no smoking/BMI; BRFSS has smoking/BMI
plus blood-pressure/cholesterol *flags* but no lab values and a
different, self-reported target. Rather than forcing a blended model
that would silently mix a diagnostic clinical measurement with a survey
self-report, this app keeps them as two clearly-labeled, honestly-scoped
tools. See README.md and the Part 1 notebook (Section 10) for the full
reasoning.

Run with:  streamlit run heart_app.py   (from the heart_project/ folder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import DATA_DICTIONARY, ALL_FEATURES  # noqa: E402
from explain import make_tree_explainer, transform_for_shap  # noqa: E402

from brfss_data import (  # noqa: E402
    DATA_DICTIONARY as BRFSS_DICT, ALL_FEATURES as BRFSS_ALL_FEATURES,
)
from brfss_explain import (  # noqa: E402
    make_tree_explainer as brfss_make_tree_explainer,
    transform_for_shap as brfss_transform_for_shap,
)

st.set_page_config(page_title="Heart Disease Risk — What-If Simulator",
                    layout="wide")


@st.cache_resource
def load_uci_model():
    return joblib.load(PROJECT_ROOT / "models" / "xgb_pipeline.joblib")


@st.cache_resource
def load_brfss_model():
    return joblib.load(PROJECT_ROOT / "models" / "brfss_xgb_pipeline.joblib")


@st.cache_resource
def get_uci_explainer(_model):
    return make_tree_explainer(_model)


@st.cache_resource
def get_brfss_explainer(_model):
    return brfss_make_tree_explainer(_model)


st.title("Heart Disease Risk — What-If Simulator")
st.caption(
    "Educational decision-support prototype, not a diagnostic device. "
    "Two separate models, two separate tabs — see the note below for why."
)

with st.expander("Why two tabs instead of one simulator?", expanded=False):
    st.markdown(
        "The original brief asked for one what-if simulator with sliders "
        "on **cholesterol, blood pressure, smoking, BMI, and glucose**. "
        "No single dataset used in this project has all five as real, "
        "trainable features:\n\n"
        "- The **UCI Heart Disease dataset** (918 patients) has "
        "cholesterol, blood pressure, and a glucose *flag* (fasting "
        "blood sugar > 120 mg/dl) — but no smoking status or BMI at all.\n"
        "- The **CDC BRFSS 2015 survey** (253,680 respondents) has "
        "smoking status, BMI, diet, and activity — but only blood-pressure "
        "and cholesterol *flags* (told by a doctor, yes/no), no lab "
        "values, and diabetes status instead of a glucose value.\n\n"
        "Blending these into one model would mean pretending a "
        "diagnostic test result and a self-reported survey answer are "
        "the same kind of measurement, predicting the same target — they "
        "aren't (angiographic CAD presence vs. self-reported lifetime "
        "heart disease/attack history). So this app keeps them as two "
        "honestly-scoped tools instead of one that quietly does less "
        "than it implies."
    )

tab_clinical, tab_lifestyle = st.tabs([
    "🩺 Clinical Risk (UCI, n=918)",
    "🚬 Lifestyle Risk (BRFSS, n=253,680)",
])

# ===========================================================================
# TAB 1 — Clinical risk (UCI)
# ===========================================================================
with tab_clinical:
    model = load_uci_model()
    explainer, preprocessor, feature_names = get_uci_explainer(model)

    st.caption(
        "Model: tuned XGBoost trained on the full UCI Heart Disease "
        "dataset (Cleveland, Hungarian, Switzerland, VA Long Beach; 918 "
        "rows after removing duplicates). Predicts angiographically-"
        "confirmed coronary artery disease presence."
    )

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("1. Patient profile")
        st.caption("Clinical/demographic values — held fixed while you explore the sliders below.")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, key="uci_age")
            sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True, key="uci_sex")
            sex = 1 if sex_label == "Male" else 0
            cp = st.selectbox(
                "Chest pain type",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                index=3, key="uci_cp",
            )
            restecg = st.selectbox(
                "Resting ECG result",
                ["normal", "st-t abnormality", "lv hypertrophy"],
                index=0, key="uci_restecg",
            )
            thal = st.selectbox(
                "Thallium stress test result",
                ["normal", "fixed defect", "reversable defect"],
                index=0, key="uci_thal",
                help="Invasive/specialist test — missing for most non-Cleveland "
                     "patients in the training data (see the notebook's audit).",
            )
        with c2:
            thalach = st.slider("Max heart rate achieved (bpm)", 60, 220, 150, key="uci_thalach")
            exang_label = st.radio("Exercise-induced angina", ["No", "Yes"], horizontal=True, key="uci_exang")
            exang = 1 if exang_label == "Yes" else 0
            oldpeak = st.slider("ST depression (oldpeak, mm)", 0.0, 6.5, 1.0, 0.1, key="uci_oldpeak")
            slope = st.selectbox(
                "ST segment slope",
                ["upsloping", "flat", "downsloping"],
                index=0, key="uci_slope",
                help="upsloping = least concerning, downsloping = most concerning.",
            )
            ca = st.slider(
                "Vessels colored by fluoroscopy (ca)", 0, 3, 0, key="uci_ca",
                help="Invasive test result — count of major coronary vessels showing "
                     "blockage. Missing for most non-Cleveland patients in training data.",
            )

        st.subheader("2. What-if sliders — modifiable risk factors")
        st.caption("Drag these and watch the risk + explanation on the right update live.")

        chol = st.slider("Total cholesterol (mg/dl)", 100, 600, 240, 5, key="uci_chol",
                          help="Desirable: <200. Borderline high: 200-239. High: >=240.")
        trestbps = st.slider("Resting systolic blood pressure (mm Hg)", 80, 220, 130, 2, key="uci_trestbps",
                              help="Normal: <120. Elevated: 120-129. High (stage 1): 130-139.")
        fbs_label = st.radio(
            "Fasting blood sugar > 120 mg/dl (elevated glucose flag)",
            ["No", "Yes"], horizontal=True, key="uci_fbs",
            help="The only glucose-related field in this dataset — a yes/no flag, not a lab value.",
        )
        fbs = 1 if fbs_label == "Yes" else 0

    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])[ALL_FEATURES]

    proba = model.predict_proba(patient)[0, 1]

    with right:
        st.subheader("3. Predicted risk")
        risk_pct = proba * 100
        if risk_pct >= 66:
            color = "#c0392b"; band = "Higher predicted risk"
        elif risk_pct >= 33:
            color = "#e08e0b"; band = "Moderate predicted risk"
        else:
            color = "#1e8449"; band = "Lower predicted risk"

        st.markdown(
            f"<div style='padding:1.2rem;border-radius:0.6rem;background:{color}22;"
            f"border:2px solid {color};'>"
            f"<span style='font-size:2.6rem;font-weight:700;color:{color}'>{risk_pct:.1f}%</span>"
            f"<span style='font-size:1.1rem;color:{color};margin-left:0.6rem'>{band}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Model estimate from 918 training patients, not a clinical "
            "probability — see the notebook's calibration analysis."
        )

        st.subheader("4. Why the model said that (SHAP)")
        Xt_row = transform_for_shap(preprocessor, patient)
        sv = explainer(Xt_row)
        sv.feature_names = feature_names

        plt.figure(figsize=(6.5, 4.2))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        st.caption(
            "Each bar shows how much that feature value pushed this "
            "patient's risk up (red) or down (blue) from the model's "
            "average prediction, ending at the final probability."
        )

# ===========================================================================
# TAB 2 — Lifestyle risk (BRFSS)
# ===========================================================================
AGE_BANDS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
    12: "75-79", 13: "80+",
}
GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_LABELS = {
    1: "Never attended / kindergarten only", 2: "Elementary (grades 1-8)",
    3: "Some high school", 4: "High school graduate",
    5: "Some college / technical school", 6: "College graduate",
}
INCOME_LABELS = {
    1: "< $10,000", 2: "$10,000-14,999", 3: "$15,000-19,999",
    4: "$20,000-24,999", 5: "$25,000-34,999", 6: "$35,000-49,999",
    7: "$50,000-74,999", 8: ">= $75,000",
}
DIABETES_LABELS = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}

with tab_lifestyle:
    brfss_model = load_brfss_model()
    b_explainer, b_preprocessor, b_feature_names = get_brfss_explainer(brfss_model)

    st.caption(
        "Model: tuned XGBoost trained on 253,680 respondents to the CDC's "
        "2015 BRFSS survey. Predicts self-reported lifetime history of "
        "heart disease or heart attack — a different, broader question "
        "than the UCI model's diagnostic CAD prediction (see the "
        "'Why two tabs' note above)."
    )

    bleft, bright = st.columns([1, 1.3])

    with bleft:
        st.subheader("1. Respondent profile")
        st.caption("Clinical/demographic values — held fixed while you explore the sliders below.")

        c1, c2 = st.columns(2)
        with c1:
            age_band = st.selectbox("Age group", list(AGE_BANDS.keys()),
                                     format_func=lambda k: AGE_BANDS[k], index=6, key="brfss_age")
            sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True, key="brfss_sex")
            b_sex = 1 if sex_label == "Male" else 0
            genhlth = st.selectbox("Self-rated general health", list(GENHLTH_LABELS.keys()),
                                    format_func=lambda k: GENHLTH_LABELS[k], index=2, key="brfss_genhlth")
            education = st.selectbox("Education level", list(EDUCATION_LABELS.keys()),
                                      format_func=lambda k: EDUCATION_LABELS[k], index=4, key="brfss_edu")
            income = st.selectbox("Household income", list(INCOME_LABELS.keys()),
                                   format_func=lambda k: INCOME_LABELS[k], index=5, key="brfss_income")
        with c2:
            highbp_label = st.radio("Told you have high blood pressure", ["No", "Yes"], horizontal=True, key="brfss_highbp")
            highbp = 1 if highbp_label == "Yes" else 0
            highchol_label = st.radio("Told you have high cholesterol", ["No", "Yes"], horizontal=True, key="brfss_highchol")
            highchol = 1 if highchol_label == "Yes" else 0
            diabetes = st.selectbox("Diabetes status", list(DIABETES_LABELS.keys()),
                                     format_func=lambda k: DIABETES_LABELS[k], index=0, key="brfss_diabetes",
                                     help="Closest proxy to a glucose measurement in this survey.")
            diffwalk_label = st.radio("Serious difficulty walking/climbing stairs", ["No", "Yes"], horizontal=True, key="brfss_diffwalk")
            diffwalk = 1 if diffwalk_label == "Yes" else 0
            stroke_label = st.radio("Ever told you had a stroke", ["No", "Yes"], horizontal=True, key="brfss_stroke")
            stroke = 1 if stroke_label == "Yes" else 0

        st.subheader("2. What-if sliders — modifiable lifestyle factors")
        st.caption("The sliders the UCI model couldn't offer — smoking, BMI, diet, and activity.")

        bmi = st.slider("BMI (kg/m²)", 12, 60, 27, key="brfss_bmi",
                         help="Underweight <18.5, healthy 18.5-24.9, overweight 25-29.9, obese >=30.")
        smoker_label = st.radio("Smoked at least 100 cigarettes in your life", ["No", "Yes"], horizontal=True, key="brfss_smoker")
        smoker = 1 if smoker_label == "Yes" else 0
        physact_label = st.radio("Physical activity in past 30 days (outside work)", ["No", "Yes"], horizontal=True, key="brfss_physact")
        physact = 1 if physact_label == "Yes" else 0
        c3, c4 = st.columns(2)
        with c3:
            fruits_label = st.radio("Eats fruit 1+ times/day", ["No", "Yes"], horizontal=True, key="brfss_fruits")
            fruits = 1 if fruits_label == "Yes" else 0
        with c4:
            veggies_label = st.radio("Eats vegetables 1+ times/day", ["No", "Yes"], horizontal=True, key="brfss_veggies")
            veggies = 1 if veggies_label == "Yes" else 0
        hvyalcohol_label = st.radio("Heavy alcohol consumption", ["No", "Yes"], horizontal=True, key="brfss_alcohol",
                                     help="Adult men >14 drinks/week, women >7 drinks/week.")
        hvyalcohol = 1 if hvyalcohol_label == "Yes" else 0

    respondent = pd.DataFrame([{
        "Smoker": smoker, "PhysActivity": physact, "Fruits": fruits, "Veggies": veggies,
        "HvyAlcoholConsump": hvyalcohol, "BMI": bmi,
        "HighBP": highbp, "HighChol": highchol, "CholCheck": 1, "Stroke": stroke,
        "Diabetes": diabetes, "DiffWalk": diffwalk, "GenHlth": genhlth,
        "MentHlth": 0, "PhysHlth": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "Sex": b_sex, "Age": age_band, "Education": education, "Income": income,
    }])[BRFSS_ALL_FEATURES]

    b_proba = brfss_model.predict_proba(respondent)[0, 1]

    with bright:
        st.subheader("3. Predicted risk")
        b_risk_pct = b_proba * 100
        if b_risk_pct >= 40:
            color = "#c0392b"; band = "Higher predicted risk"
        elif b_risk_pct >= 15:
            color = "#e08e0b"; band = "Moderate predicted risk"
        else:
            color = "#1e8449"; band = "Lower predicted risk"

        st.markdown(
            f"<div style='padding:1.2rem;border-radius:0.6rem;background:{color}22;"
            f"border:2px solid {color};'>"
            f"<span style='font-size:2.6rem;font-weight:700;color:{color}'>{b_risk_pct:.1f}%</span>"
            f"<span style='font-size:1.1rem;color:{color};margin-left:0.6rem'>{band}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Risk bands are lower here than the clinical tab on purpose — only "
            "~9.4% of BRFSS respondents report a heart disease/attack history, "
            "vs. ~55% disease prevalence in the UCI hospital-referral sample, "
            "so a 'high' predicted probability looks different on each scale. "
            "Model estimate from 253,680 survey respondents, not a clinical "
            "probability — see the notebook's calibration analysis (calibration "
            "intercept is meaningfully negative here due to class-imbalance "
            "correction, disclosed in Section 10)."
        )

        st.subheader("4. Why the model said that (SHAP)")
        Xt_row = brfss_transform_for_shap(b_preprocessor, respondent)
        sv = b_explainer(Xt_row)
        sv.feature_names = b_feature_names

        plt.figure(figsize=(6.5, 4.2))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        st.caption(
            "Each bar shows how much that feature value pushed this "
            "respondent's risk up (red) or down (blue) from the model's "
            "average prediction. Notebook Section 10 found lifestyle "
            "factors (smoking, diet, activity, BMI) account for only "
            "~7% of total SHAP importance in this dataset overall — "
            "clinical/demographic factors like age and general health "
            "dominate — so don't expect the lifestyle sliders to move "
            "the needle as much as blood pressure or age will."
        )

st.divider()
st.caption(
    "Two-dataset project. Part 1 (notebooks/part1_heart_disease_model.ipynb) "
    "covers data sourcing, cleaning, leakage-free modeling, and evaluation for "
    "both the UCI clinical dataset and the CDC BRFSS lifestyle survey. This app "
    "reuses the exact same trained pipelines and SHAP logic from src/ — no "
    "duplicated or re-implemented model logic between the notebook and the dashboard."
)
