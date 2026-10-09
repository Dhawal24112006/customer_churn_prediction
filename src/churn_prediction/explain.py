"""Optional SHAP for transformed tabular data; observational feature attribution."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import DATA,FEATURES,REPORTS,METADATA
from .data_validation import load_data
from .predict import load_model


def shap_explanations(model, frame, output=REPORTS, max_rows=120):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is optional: pip install -r requirements-optional.txt") from exc
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Explaining predicted churn probability, model-agnostic to all classifier types.
    # Permutation SHAP can be expensive: limit background & number of explanations.
    transform=model.named_steps["preprocess"]
    x=frame[FEATURES].head(max_rows).copy()
    z=np.asarray(transform.transform(x),dtype=float)
    names=transform.named_steps["encode"].get_feature_names_out().tolist()
    zdf=pd.DataFrame(z,columns=names)
    predictor=lambda a: model.named_steps["model"].predict_proba(np.asarray(a))[:,list(model.classes_).index(1)]
    background=zdf.head(min(20,len(zdf)))
    explainer=shap.Explainer(predictor,background,algorithm="permutation",seed=42)
    exp=explainer(zdf.iloc[:min(20,len(zdf))],max_evals=max(2*z.shape[1]+1,201))
    plt.figure();shap.summary_plot(exp.values,zdf.iloc[:len(exp.values)],show=False,max_display=15)
    plt.tight_layout();plt.savefig(output/"shap_summary.png",bbox_inches="tight",dpi=130);plt.close()
    strength=np.abs(exp.values).mean(axis=0)
    top=int(np.argmax(strength))
    plt.figure();shap.dependence_plot(top,exp.values,zdf.iloc[:len(exp.values)],show=False,interaction_index=None)
    plt.tight_layout();plt.savefig(output/"shap_dependence.png",bbox_inches="tight",dpi=130);plt.close()
    explanation=pd.DataFrame({"feature":names,"shap_value":exp.values[0],"feature_value":zdf.iloc[0].to_numpy()})
    explanation.reindex(explanation.shap_value.abs().sort_values(ascending=False).index).to_csv(output/"shap_local_customer.csv",index=False)
    return explanation


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--max-rows",type=int,default=100);a=parser.parse_args()
    df=load_data(DATA)
    result=shap_explanations(load_model(),df,max_rows=a.max_rows)
    print("SHAP is association, not causal effect. Strongest local features:")
    print(result.reindex(result.shap_value.abs().sort_values(ascending=False).index).head(8).to_string(index=False))
if __name__=="__main__":main()
