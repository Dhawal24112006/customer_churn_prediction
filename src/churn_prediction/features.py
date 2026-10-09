"""Features computable from an account snapshot; no target-derived columns."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

class CustomerFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ordered bands capture nonlinearity; tenure itself is retained.
        t = df["tenure"]
        df["tenure_band"] = np.select([t < 12, t < 24, t < 48], ["0-11", "12-23", "24-47"], default="48+")
        df.loc[t.isna(), "tenure_band"] = "unknown"
        # Count of optional subscribed services, not an inferred behavioral cause.
        df["service_count"] = df[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype(float)
        # Useful indicator because month-to-month billing has distinctive churn association.
        df["month_to_month"] = (df["Contract"] == "Month-to-month").astype(float)
        return df
