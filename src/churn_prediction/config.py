"""Centralized paths and schema for the original 21-column IBM sample."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ARTIFACT = MODELS / "churn_pipeline.joblib"
METADATA = MODELS / "metadata.json"
SEED = 42
TARGET = "Churn"
ID = "customerID"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
REQUIRED = [ID] + FEATURES + [TARGET]
IBM_CSV_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
