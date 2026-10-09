"""Interactive model inspection and hypothetical outreach planning."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Works when launched from repository root without needing environment hacks.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from churn_prediction.config import DATA,FEATURES,ID,METADATA,REPORTS,NUMERIC,CATEGORICAL
from churn_prediction.data_validation import load_data
from churn_prediction.predict import load_model,predict_frame,predict_one
from churn_prediction.retention import Assumptions,prioritize,risk_category

st.set_page_config(page_title="Churn & Retention Intelligence",layout="wide")
st.title("Customer Churn & Retention Intelligence")
st.caption("Predictive associations and *illustrative* campaign values; not established causal treatment effects")

@st.cache_resource
def get_model():return load_model()
@st.cache_data
def get_data():return load_data()

try:
    model=get_model();df=get_data();meta=json.loads(METADATA.read_text())
except (FileNotFoundError,ValueError,KeyError) as e:
    st.error(f"Missing input or trained model: {e}")
    st.code("python -m churn_prediction.data_validation --download\npython -m churn_prediction.train")
    st.stop()

@st.cache_data
def probability_cache(_model,data):return predict_frame(data,_model)

predictions=probability_cache(model,df)
joined=df.merge(predictions,on=ID,validate="one_to_one")
page=st.sidebar.radio("Section",["Executive Overview","Customer Risk Explorer","Model Evaluation","Explainability","Retention Planning","Single-Customer Prediction"])

def show_image(name):
    file=REPORTS/name
    if file.exists():st.image(str(file))
    else:st.info(f"{name} not generated yet; run the documented command.")

if page=="Executive Overview":
    a,b,c=st.columns(3)
    a.metric("Customers",f"{len(df):,}");b.metric("Observed churn rate",f"{df.Churn.eq('Yes').mean():.1%}")
    c.metric("Selected model",meta["chosen_model"])
    st.json({k:v for k,v in meta["final_test_metrics"].items() if k!="confusion_matrix"},expanded=False)
    show_image("eda_churn_patterns.png")
    st.warning("IBM example sample. Measured associations are not explanations of customer behavior.")
elif page=="Customer Risk Explorer":
    contract=st.multiselect("Contract",sorted(df.Contract.dropna().unique()),default=sorted(df.Contract.dropna().unique()))
    tenure=st.slider("Tenure range (months)",0,int(df.tenure.max()),(0,int(df.tenure.max())))
    service=st.multiselect("Internet service",sorted(df.InternetService.dropna().unique()),default=sorted(df.InternetService.dropna().unique()))
    categories=st.multiselect("Risk category",["Low","Medium","High"],default=["Low","Medium","High"])
    view=joined[joined.Contract.isin(contract)&joined.tenure.between(*tenure)&joined.InternetService.isin(service)&joined.risk_category.isin(categories)]
    st.dataframe(view[[ID,"Contract","tenure","InternetService","churn_probability","risk_category"]].sort_values("churn_probability",ascending=False),width="stretch")
    if len(view):
        customer=st.selectbox("Inspect customer",view[ID].astype(str).tolist())
        st.json(view.loc[view[ID].astype(str)==customer].iloc[0][[ID,"churn_probability","risk_category"]].to_dict())
elif page=="Model Evaluation":
    st.write("Threshold selected on validation; test reserved until final evaluation.")
    st.metric("Decision threshold",f"{meta['threshold']:.2f}")
    for csv in ["validation_model_comparison.csv","test_model_comparison.csv","validation_thresholds.csv"]:
        file=REPORTS/csv
        if file.exists():st.subheader(csv);st.dataframe(pd.read_csv(file),width="stretch")
    for name in ["curves.png","confusion_matrix.png","calibration.png"]:show_image(name)
elif page=="Explainability":
    show_image("feature_importance.png")
    st.write("Permutation importance shows reduction in validation average precision; correlated features can share importance.")
    for name in ["shap_summary.png","shap_dependence.png"]:show_image(name)
    local=REPORTS/"shap_local_customer.csv"
    if local.exists():st.dataframe(pd.read_csv(local).head(15))
    else:st.code("python -m churn_prediction.explain")
    st.caption("SHAP explains fitted predictions, not the causes of churn.")
elif page=="Retention Planning":
    col1,col2,col3,col4=st.columns(4)
    cost=col1.number_input("Cost per contact",min_value=.01,value=10.)
    value=col2.number_input("Value if retained",min_value=0.,value=200.)
    effectiveness=col3.number_input("Hypothetical intervention success",min_value=0.,max_value=1.,value=.10,step=.01)
    budget=col4.number_input("Campaign budget",min_value=0.,value=1000.)
    t=st.slider("Minimum churn probability",0.,1.,float(meta["threshold"]),.01)
    plan=prioritize(joined,predictions.churn_probability,Assumptions(cost,value,effectiveness,budget),t)
    chosen=plan[plan.selected]
    st.metric("Selected",len(chosen));st.metric("Illustrative total cost",f"{chosen.intervention_cost.sum():,.2f}")
    st.metric("Illustrative expected net",f"{chosen.estimated_expected_net.sum():,.2f}")
    st.dataframe(plan,width="stretch")
    st.download_button("Export outreach plan",plan.to_csv(index=False),"retention_plan.csv")
    st.info("This planning model assumes a uniform hypothetical intervention effect. Run an RCT before interpreting these as realizable savings.")
else:
    st.write("Set inputs for one hypothetical customer, using observed categories as defaults.")
    values={ID:"NEW-CUSTOMER"}
    with st.form("single_predict"):
        for feature in NUMERIC:
            values[feature]=st.number_input(feature,min_value=0.,value=float(df[feature].median()),step=1. if feature in ["SeniorCitizen","tenure"] else .1)
        for feature in CATEGORICAL:
            options=sorted(df[feature].dropna().astype(str).unique().tolist())
            values[feature]=st.selectbox(feature,options)
        submitted=st.form_submit_button("Predict churn")
    if submitted:
        try:
            output=predict_one(values,model)
            st.metric("Predicted churn probability",f"{output['churn_probability']:.1%}")
            st.write("Risk category:",output["risk_category"])
            st.caption("The numeric probability is an estimate, not a certainty. Run SHAP for feature attributions.")
        except ValueError as exc:st.error(str(exc))
