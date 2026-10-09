"""All learned transformations live inside a fitted pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import NUMERIC, CATEGORICAL
from .features import CustomerFeatures

DERIVED_NUMERIC = NUMERIC + ["service_count", "month_to_month"]
DERIVED_CATEGORICAL = CATEGORICAL + ["tenure_band"]

def make_preprocessor(scale: bool = False):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return Pipeline([
        ("derive", CustomerFeatures()),
        ("encode", ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), DERIVED_NUMERIC),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), DERIVED_CATEGORICAL),
        ], remainder="drop", verbose_feature_names_out=False)),
    ])
