from churn_prediction.preprocessing import make_preprocessor
from churn_prediction.data_validation import validate_frame

def test_unknown_category_and_missing_value(sample):
    data=validate_frame(sample)
    pre=make_preprocessor(True).fit(data.iloc[:20])
    holdout=data.iloc[20:].copy();holdout.loc[20,"Contract"]="Unseen contract"
    holdout.loc[21,"MonthlyCharges"]=float("nan")
    transformed=pre.transform(holdout)
    assert transformed.shape[0]==10
    assert transformed.shape[1]==len(pre.named_steps["encode"].get_feature_names_out())
    assert not __import__("numpy").isnan(transformed).any()
