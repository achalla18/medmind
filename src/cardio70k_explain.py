import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from cardio70k_prep import load_clean_cardio70k

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"
MODELS = BASE / "models"

X, y, audit = load_clean_cardio70k()
pipe = joblib.load(MODELS / "cardio70k_xgboost_final.joblib")
preprocess = pipe.named_steps["preprocess"]
model = pipe.named_steps["model"]

# subsample for SHAP speed (70k rows is more than needed for a stable summary)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X), size=5000, replace=False)
X_sample = X.iloc[sample_idx]

X_trans = preprocess.transform(X_sample)
feature_names = preprocess.get_feature_names_out()
X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_trans_df)

plt.figure()
shap.summary_plot(shap_values, X_trans_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIGURES / "shap_summary_cardio70k.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
imp_df.to_csv(REPORTS / "shap_global_importance_cardio70k.csv", index=False)
print(imp_df.head(12).to_string(index=False))
