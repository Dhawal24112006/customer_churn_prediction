"""Held-out classification, calibration and diagnostic charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve


def scores(y, p, threshold=0.5):
    label = np.asarray(p) >= threshold
    return dict(accuracy=float(accuracy_score(y,label)), precision=float(precision_score(y,label,zero_division=0)),
                recall=float(recall_score(y,label,zero_division=0)), f1=float(f1_score(y,label,zero_division=0)),
                roc_auc=float(roc_auc_score(y,p)) if len(set(y)) > 1 else None,
                average_precision=float(average_precision_score(y,p)) if len(set(y)) > 1 else None,
                log_loss=float(log_loss(y,p,labels=[0,1])), threshold=float(threshold),
                confusion_matrix=confusion_matrix(y,label,labels=[0,1]).tolist())


def save_evaluation(y, prob_by_model: dict, chosen: str, threshold: float, directory: Path):
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,p in prob_by_model.items():
        row={"model":name, **scores(y,p, threshold if name==chosen else 0.5)}
        rows.append({k:v for k,v in row.items() if k!="confusion_matrix"})
    pd.DataFrame(rows).to_csv(directory/"test_model_comparison.csv",index=False)
    (directory/"final_test_metrics.json").write_text(json.dumps(scores(y,prob_by_model[chosen],threshold),indent=2))
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for name,p in prob_by_model.items():
        fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
        ax[0].plot(fpr,tpr,label=name); ax[1].plot(recall,precision,label=name)
    ax[0].plot([0,1],[0,1],"k--",alpha=.4)
    for a,title,xlab,ylab in [(ax[0],"Test ROC","False positive rate","True positive rate"),(ax[1],"Test precision–recall","Recall","Precision")]:
        a.set(title=title,xlabel=xlab,ylabel=ylab); a.legend(fontsize=8); a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directory/"curves.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(y, np.asarray(prob_by_model[chosen])>=threshold, ax=ax, display_labels=["Stay","Churn"],colorbar=False)
    ax.set_title(f"Test confusion matrix: {chosen}"); fig.tight_layout();fig.savefig(directory/"confusion_matrix.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    obs,pred=calibration_curve(y,prob_by_model[chosen],n_bins=8,strategy="quantile")
    ax.plot(pred,obs,"o-",label=chosen);ax.plot([0,1],[0,1],"k--",label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability",ylabel="Observed churn fraction",title="Test-set calibration")
    ax.legend();fig.tight_layout();fig.savefig(directory/"calibration.png",dpi=150);plt.close(fig)
