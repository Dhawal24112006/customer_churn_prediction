import pytest
from churn_prediction.data_validation import validate_frame,DataValidationError

def test_missing_column(sample):
    with pytest.raises(DataValidationError,match="Missing required"):
        validate_frame(sample.drop(columns="Contract"))

def test_bad_target(sample):
    sample.loc[0,"Churn"]="Maybe"
    with pytest.raises(DataValidationError,match="Churn must"):
        validate_frame(sample)

def test_duplicate_id(sample):
    sample.loc[1,"customerID"]=sample.loc[0,"customerID"]
    with pytest.raises(DataValidationError,match="Duplicate"):
        validate_frame(sample)

def test_blank_totalcharges_and_bad_numeric(sample):
    sample.loc[0,"TotalCharges"]=" "
    assert validate_frame(sample).TotalCharges.isna().sum()==1
    sample.loc[1,"TotalCharges"]="not numeric"
    with pytest.raises(DataValidationError,match="Non-numeric"):
        validate_frame(sample)
