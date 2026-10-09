import pytest
import pandas as pd
from churn_prediction.config import FEATURES,NUMERIC,CATEGORICAL,ID,TARGET
@pytest.fixture
def sample():
    rows=[]
    for i in range(30):
        d={ID:f"TEST-{i}",TARGET:"Yes" if i%3==0 else "No"}
        d.update({c:"Yes" for c in CATEGORICAL})
        d.update({"SeniorCitizen":i%2,"tenure":i+1,"MonthlyCharges":30+i,"TotalCharges":str((30+i)*(i+1))})
        d["Contract"]=["Month-to-month","One year","Two year"][i%3]
        rows.append(d)
    return pd.DataFrame(rows)
