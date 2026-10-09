import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from churn_prediction.preprocessing import make_preprocessor
from churn_prediction.data_validation import validate_frame
from churn_prediction.predict import load_model,predict_frame,predict_one

def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError,match="Trained model missing"):
        load_model(tmp_path/"absent.joblib")

def test_probabilities_and_single_prediction(sample):
    df=validate_frame(sample)
    model=Pipeline([("preprocess",make_preprocessor()),("model",DummyClassifier(strategy="prior"))])
    model.fit(df,df.Churn.eq("Yes").astype(int))
    result=predict_frame(df,model)
    assert len(result)==30 and result.churn_probability.between(0,1).all()
    assert list(result.columns)==["customerID","churn_probability","risk_category"]
    assert 0<=predict_one(df.iloc[0].to_dict(),model)["churn_probability"]<=1

def test_reproducible_features(sample):
    df=validate_frame(sample)
    p=make_preprocessor()
    assert np.array_equal(p.fit_transform(df),p.fit_transform(df))
