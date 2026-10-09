"""3-way stratified split; training-only CV; validation selection; one final test."""
import argparse
import json
from pathlib import Path
import logging
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import DATA, FEATURES, ID, TARGET, SEED, ARTIFACT, METADATA, REPORTS
from .data_validation import load_data
from .preprocessing import make_preprocessor
from .evaluate import scores, save_evaluation
from .retention import Assumptions, threshold_table, choose_threshold

LOG = logging.getLogger(__name__)


def candidates(include_xgboost=True):
    options={
        "dummy": (DummyClassifier(strategy="prior"), {}, False),
        "logistic_regression": (LogisticRegression(max_iter=2000,random_state=SEED), {"model__C":[0.1,1,10], "model__class_weight":[None,"balanced"]}, True),
        "random_forest": (RandomForestClassifier(n_estimators=150,n_jobs=1,random_state=SEED), {"model__max_depth":[6,None],"model__min_samples_leaf":[1,5],"model__class_weight":[None,"balanced"]},False),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            options["xgboost"]=(XGBClassifier(n_estimators=150,objective="binary:logistic",tree_method="hist",eval_metric="logloss",random_state=SEED,n_jobs=1),
                  {"model__max_depth":[3,5],"model__learning_rate":[0.05,0.15],"model__subsample":[0.8,1.0]},False)
        except ImportError:
            LOG.warning("XGBoost unavailable. Install: pip install -r requirements-optional.txt")
    return options


def split_data(df):
    x=df[FEATURES].copy(); y=df[TARGET].eq("Yes").astype(int)
    x_trainval,x_test,y_trainval,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=SEED)
    x_train,x_val,y_train,y_val=train_test_split(x_trainval,y_trainval,test_size=.25,stratify=y_trainval,random_state=SEED)
    return x_train,x_val,x_test,y_train,y_val,y_test


def train(path=DATA, include_xgboost=True, fast=False, assumptions=None):
    assumptions=assumptions or Assumptions()
    df=load_data(Path(path))
    x_train,x_val,x_test,y_train,y_val,y_test=split_data(df)
    cv=StratifiedKFold(n_splits=3 if fast else 5,shuffle=True,random_state=SEED)
    fitted={}; val_probas={}; cv_rows=[]
    for name,(estimator,params,scale) in candidates(include_xgboost).items():
        pipeline=Pipeline([("preprocess",make_preprocessor(scale)),("model",estimator)])
        if fast or not params:
            model=pipeline.fit(x_train,y_train)
            cv_mean=cv_sd=None
            best_params={}
        else:
            search=GridSearchCV(pipeline,params,scoring="average_precision",cv=cv,n_jobs=1,refit=True)
            search.fit(x_train,y_train)
            model=search.best_estimator_;cv_mean=float(search.best_score_)
            cv_sd=float(search.cv_results_["std_test_score"][search.best_index_]);best_params=search.best_params_
        fitted[name]=model
        val_probas[name]=model.predict_proba(x_val)[:,list(model.classes_).index(1)]
        cv_rows.append({"model":name,"cv_average_precision_mean":cv_mean,"cv_average_precision_sd":cv_sd,
                        "best_parameters":json.dumps(best_params),**{k:v for k,v in scores(y_val,val_probas[name]).items() if k!="confusion_matrix"}})
        LOG.info("Fitted %s",name)
    comparison=pd.DataFrame(cv_rows)
    # Select by validation average precision; tie break with validation log loss then name.
    chosen=comparison.sort_values(["average_precision","log_loss","model"],ascending=[False,True,True]).iloc[0]["model"]
    table=threshold_table(y_val,val_probas[chosen],assumptions)
    selected_threshold=choose_threshold(table)
    REPORTS.mkdir(parents=True,exist_ok=True);ARTIFACT.parent.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(REPORTS/"validation_model_comparison.csv",index=False)
    table.to_csv(REPORTS/"validation_thresholds.csv",index=False)
    # Final held-out evaluation is reached only after selection and threshold are frozen.
    test_probs={name:model.predict_proba(x_test)[:,list(model.classes_).index(1)] for name,model in fitted.items()}
    save_evaluation(y_test,{chosen:test_probs[chosen]},chosen,selected_threshold,REPORTS)
    model=fitted[chosen]
    joblib.dump(model,ARTIFACT)
    metadata={"artifact_version":"1.0","created_utc":datetime.now(timezone.utc).isoformat(),
        "sklearn_version":sklearn.__version__,"chosen_model":chosen,"threshold":selected_threshold,
        "criterion":"highest validation average precision, then lowest validation log loss",
        "split":"60% train / 20% validation / 20% final test, stratified, seed 42",
        "validation_metrics":scores(y_val,val_probas[chosen],selected_threshold),
        "final_test_metrics":scores(y_test,test_probs[chosen],selected_threshold),
        "assumptions":assumptions.__dict__,"rows":len(df),"churn_rate":float(df.Churn.eq('Yes').mean())}
    METADATA.write_text(json.dumps(metadata,indent=2))
    # Logistic coefficients and conditional odds ratios, on encoded feature scale.
    logistic=fitted.get("logistic_regression")
    if logistic is not None:
        names=logistic.named_steps["preprocess"].named_steps["encode"].get_feature_names_out()
        coefs=logistic.named_steps["model"].coef_[0]
        pd.DataFrame({"encoded_feature":names,"log_odds_coefficient":coefs,
                      "conditional_odds_ratio":np.exp(coefs)}).to_csv(REPORTS/"logistic_coefficients.csv",index=False)
    # Feature-level validation permutation importance, held-out of model fit.
    try:
        imp=permutation_importance(model,x_val,y_val,n_repeats=3,random_state=SEED,scoring="average_precision")
        importances=pd.Series(imp.importances_mean,index=FEATURES).sort_values(ascending=False)
        importances.to_csv(REPORTS/"permutation_importance.csv",header=["average_precision_drop"])
        fig,ax=plt.subplots(figsize=(9,6));importances.head(15).sort_values().plot.barh(ax=ax)
        ax.set(xlabel="Mean validation AP decrease after shuffling",title="Permutation importance (associational)")
        fig.tight_layout();fig.savefig(REPORTS/"feature_importance.png",dpi=150);plt.close(fig)
    except Exception as exc:
        LOG.warning("Permutation importance unavailable: %s",exc)
    print(f"Selected: {chosen}; threshold: {selected_threshold:.2f}; test AP: {metadata['final_test_metrics']['average_precision']:.4f}")
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=str,default=str(DATA))
    parser.add_argument("--no-xgboost",action="store_true")
    parser.add_argument("--fast",action="store_true",help="Smoke mode: no grid search; still real held-out evaluation")
    parser.add_argument("--contact-cost",type=float,default=10.)
    parser.add_argument("--retained-value",type=float,default=200.)
    parser.add_argument("--effectiveness",type=float,default=.1)
    parser.add_argument("--budget",type=float,default=1000.)
    a=parser.parse_args();logging.basicConfig(level=logging.INFO)
    train(a.data,not a.no_xgboost,a.fast,Assumptions(a.contact_cost,a.retained_value,a.effectiveness,a.budget))

if __name__=="__main__":main()
