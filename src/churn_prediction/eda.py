"""Reproducible EDA figures, tests and summary without fitting a classifier."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from .config import DATA,REPORTS,NUMERIC,CATEGORICAL
from .data_validation import load_data

def run_eda(path=DATA):
    df=load_data(path);REPORTS.mkdir(parents=True,exist_ok=True)
    df["churn_binary"]=df.Churn.eq("Yes").astype(int)
    df["tenure_group"]=pd.cut(df.tenure,[-.01,12,24,48,np.inf],labels=["0-12","13-24","25-48","49+"])
    df["monthly_band"]=pd.qcut(df.MonthlyCharges,q=4,duplicates="drop")
    findings={"rows":len(df),"columns_original":len(df.columns)-2,"churn_rate":float(df.churn_binary.mean()),
              "missing":df.isna().sum()[lambda s:s>0].astype(int).to_dict(),"duplicate_rows":int(df.duplicated().sum()),
              "numeric_summary":df[NUMERIC].describe().round(2).to_dict()}
    # Wilson 95% CIs summarize uncertainty around each observed subgroup rate.
    group_counts=df.groupby("Contract",observed=True).churn_binary.agg(["sum","count"])
    lower,upper=proportion_confint(group_counts["sum"],group_counts["count"],alpha=.05,method="wilson")
    group_counts["churn_rate"]=group_counts["sum"]/group_counts["count"]
    group_counts["wilson_ci_low"]=lower
    group_counts["wilson_ci_high"]=upper
    group_counts.to_csv(REPORTS/"eda_contract_rates_ci.csv")
    cols=["Contract","PaymentMethod","InternetService","tenure_group","monthly_band"]
    fig,axes=plt.subplots(3,2,figsize=(13,13));axes=axes.ravel()
    sns.countplot(data=df,x="Churn",ax=axes[0],color="steelblue");axes[0].set_title("Class distribution (observed)")
    for i,col in enumerate(cols,1):
        rates=df.groupby(col,observed=True).churn_binary.mean().sort_values(ascending=False)
        rates.plot.bar(ax=axes[i]);axes[i].set(title=f"Observed churn rate by {col}",ylabel="Churn fraction",xlabel=col,ylim=(0,1));axes[i].tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(REPORTS/"eda_churn_patterns.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,col in zip(axes,["tenure","MonthlyCharges","TotalCharges"]):
        sns.histplot(data=df,x=col,hue="Churn",ax=ax,element="step",stat="density",common_norm=False)
        ax.set_title(f"{col} distribution")
    fig.tight_layout();fig.savefig(REPORTS/"eda_numeric_distributions.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));sns.heatmap(df[NUMERIC+["churn_binary"]].corr(numeric_only=True),annot=True,cmap="coolwarm",vmin=-1,vmax=1,ax=ax)
    ax.set_title("Pearson correlations (not causal)");fig.tight_layout();fig.savefig(REPORTS/"eda_correlations.png",dpi=150);plt.close(fig)
    tests=[]
    for col in ["Contract","PaymentMethod","InternetService","Partner","PaperlessBilling"]:
        table=pd.crosstab(df[col],df.Churn)
        chi2,p,_,expected=chi2_contingency(table)
        n=table.to_numpy().sum();v=float(np.sqrt(chi2/(n*min(table.shape[0]-1,table.shape[1]-1))))
        tests.append({"feature":col,"test":"chi_square","p_value":float(p),"effect_size_cramers_v":v,"minimum_expected":float(expected.min())})
    for col in ["tenure","MonthlyCharges","TotalCharges"]:
        a=df.loc[df.churn_binary==1,col].dropna();b=df.loc[df.churn_binary==0,col].dropna()
        stat,p=mannwhitneyu(a,b,alternative="two-sided")
        rank_biserial=2*stat/(len(a)*len(b))-1
        tests.append({"feature":col,"test":"mann_whitney_u","p_value":float(p),"effect_size_rank_biserial":float(rank_biserial)})
    adjusted=multipletests([x["p_value"] for x in tests],method="fdr_bh")[1]
    for t,p in zip(tests,adjusted):t["p_value_fdr_bh"]=float(p)
    pd.DataFrame(tests).to_csv(REPORTS/"eda_hypothesis_tests.csv",index=False)
    (REPORTS/"eda_summary.json").write_text(json.dumps(findings,indent=2,default=str))
    (REPORTS/"eda_summary.md").write_text(
        f"# EDA summary\n\nCustomers: {len(df)}; observed churn rate: {findings['churn_rate']:.2%}. "
        f"Missing entries: {findings['missing']}. Full-row duplicates: {findings['duplicate_rows']}.\n\n"
        "Plots: eda_churn_patterns.png, eda_numeric_distributions.png, eda_correlations.png.\n\n"
        "Tests: two-sided chi-square (categorical) and Mann–Whitney U (numeric), "
        "with Benjamini–Hochberg FDR-adjusted p-values in eda_hypothesis_tests.csv. "
        "H0: no association / identical rank distribution; H1: association / differing distributions. "
        "Check expected chi-square cell counts in the table. Significance is not causality. "
        "Cramér's V and rank-biserial correlation describe effect magnitude. "
        "Snapshot sampling and correlated observations may violate inferential assumptions.\n"
    )
    print(f"EDA saved under {REPORTS}; {len(df)} customers")
    return findings
if __name__=="__main__":run_eda()
