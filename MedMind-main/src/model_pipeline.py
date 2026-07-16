"""
Leakage-free preprocessing + modeling pipeline builder.

Every step (imputation, scaling, encoding, resampling) is wrapped inside a
single imblearn Pipeline so that, under cross-validation, each step is
fit ONLY on the training fold and applied (not refit) to the held-out fold.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_prep import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor(numeric_impute="iterative"):
    if numeric_impute == "iterative":
        num_imputer = IterativeImputer(random_state=42, max_iter=15, sample_posterior=False)
    else:
        num_imputer = SimpleImputer(strategy="median")

    numeric_pipe = SkPipeline([
        ("impute", num_imputer),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return pre


def make_pipeline(model, imbalance_strategy="none", numeric_impute="iterative", random_state=42):
    """
    imbalance_strategy: "none" | "class_weight" (handled by caller passing a
    model already configured with class_weight='balanced') | "smote"
    """
    pre = make_preprocessor(numeric_impute=numeric_impute)
    steps = [("preprocess", pre)]
    if imbalance_strategy == "smote":
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))
    return ImbPipeline(steps)
