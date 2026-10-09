"""Trusted local joblib artifact, validated batch and single inference."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from .config import ARTIFACT,METADATA,ID
from .data_validation import validate_frame
from .retention import risk_category

def load_model(path=ARTIFACT):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model missing: {path}. Run python -m churn_prediction.train first")
    # joblib uses pickle; load only artifacts that you generated and trust.
    model=joblib.load(path)
    if not hasattr(model,"predict_proba") or not hasattr(model,"classes_"):
        raise ValueError("Artifact does not contain a fitted probabilistic classifier")
    if 1 not in model.classes_:
        raise ValueError("Model does not provide class 1 (churn) probabilities")
    return model

def predict_frame(frame,model=None):
    df=validate_frame(frame,labeled=False)
    model=model if model is not None else load_model()
    p=model.predict_proba(df)[:,list(model.classes_).index(1)]
    result=pd.DataFrame({ID:df[ID].to_numpy(),"churn_probability":p})
    result["risk_category"]=[risk_category(float(v)) for v in p]
    return result

def predict_one(customer:dict,model=None):
    return predict_frame(pd.DataFrame([customer]),model).iloc[0].to_dict()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("reports/predictions.csv"))
    args=parser.parse_args()
    results=predict_frame(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    results.to_csv(args.output,index=False)
    print(f"Saved {len(results)} predictions to {args.output}")
if __name__=="__main__":main()
