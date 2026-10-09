import numpy as np
import pandas as pd
import pytest
from churn_prediction.retention import Assumptions,threshold_table,choose_threshold,prioritize

def test_threshold_confusion():
    result=threshold_table([1,0,1,0],[.9,.8,.3,.1],Assumptions(budget=100),thresholds=[.5])
    row=result.iloc[0]
    assert (row.tp,row.fp,row.fn,row.tn)==(1,1,1,1)
    assert row.precision==.5 and row.recall==.5
    assert choose_threshold(result)==.5

def test_budget_never_exceeded():
    frame=pd.DataFrame({"customerID":[str(i) for i in range(8)]})
    plan=prioritize(frame,np.full(8,.9),Assumptions(contact_cost=12,retained_value=200,effectiveness=.5,budget=25))
    assert plan.loc[plan.selected,"intervention_cost"].sum()<=25
    assert plan.selected.sum()==2

def test_invalid_costs():
    with pytest.raises(ValueError):Assumptions(contact_cost=0)
    with pytest.raises(ValueError):Assumptions(effectiveness=1.2)
