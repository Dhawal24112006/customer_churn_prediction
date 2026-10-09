"""Strict input checking; optional IBM download keeps the dataset uncommitted."""
import argparse
import logging
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA, FEATURES, ID, TARGET, NUMERIC, REQUIRED, IBM_CSV_URL

LOG = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Unusable dataset or inference input."""


def download_data(destination: Path = DATA) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    try:
        urllib.request.urlretrieve(IBM_CSV_URL, destination)
        validate_frame(pd.read_csv(destination), labeled=True)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download/validate the IBM CSV: {exc}. Download the CSV manually from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and save it as {destination}"
        ) from exc
    return destination


def validate_frame(frame: pd.DataFrame, labeled: bool = True, require_id: bool = True) -> pd.DataFrame:
    """Return cleaned copy with numeric columns coerced, without fitting statistics."""
    df = frame.copy()
    required = FEATURES + ([ID] if require_id else []) + ([TARGET] if labeled else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("CSV contains no customers")
    if require_id and (df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any()):
        raise DataValidationError("Missing customerID")
    if require_id and df[ID].duplicated().any():
        raise DataValidationError(f"Duplicate customerID rows: {df[ID].duplicated().sum()}")
    for col in NUMERIC:
        raw = df[col].astype("string").str.strip().replace("", pd.NA)
        converted = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & converted.isna()
        if bad.any():
            raise DataValidationError(f"Non-numeric values in {col}: {raw[bad].head().tolist()}")
        df[col] = converted.astype(float)
    for col in FEATURES:
        if col not in NUMERIC:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = df[col].astype("object")
    if (df["tenure"].dropna() < 0).any() or (df["MonthlyCharges"].dropna() < 0).any():
        raise DataValidationError("Negative tenure or MonthlyCharges")
    if labeled:
        values = set(df[TARGET].dropna().astype(str).str.strip())
        if values - {"Yes", "No"} or df[TARGET].isna().any():
            raise DataValidationError(f"Churn must be exactly Yes/No (no missing); got {values}")
        df[TARGET] = df[TARGET].astype(str).str.strip()
    if df[FEATURES].isna().any().any():
        LOG.warning("Missing feature values: %s", df[FEATURES].isna().sum()[lambda s:s>0].to_dict())
    return df


def load_data(path: Path = DATA, labeled: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python -m churn_prediction.data_validation --download or copy the Kaggle CSV there.")
    return validate_frame(pd.read_csv(path), labeled=labeled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--path", type=Path, default=DATA)
    args = parser.parse_args()
    if args.download:
        download_data(args.path)
    df = load_data(args.path)
    print(f"Validated {len(df)} rows, {len(df.columns)} columns; churn rate={(df.Churn=='Yes').mean():.2%}")
    print("Missing values:", df.isna().sum()[lambda s:s>0].to_dict())
    print("Full-row duplicates:", int(df.duplicated().sum()))

if __name__ == "__main__":
    main()
