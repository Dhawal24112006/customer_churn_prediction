# IBM Telco sample data (not distributed)

The expected file is `data/Telco-Customer-Churn.csv`, containing `Churn` (Yes/No), `customerID`, and the 19 raw input attributes. The `Churn` label indicates whether a customer left within the last month, *not* who will necessarily leave next month in a deployed prospective system.

**Primary source:** IBM's archived sample repository: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

**Public alternative:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). Rename the downloaded file to `Telco-Customer-Churn.csv`.

Automatically fetch the IBM sample with `python -m churn_prediction.data_validation --download`. If unavailable, download manually using either link, place CSV at the stated path and run `python -m churn_prediction.data_validation`.

The raw data is excluded from Git because public accessibility is not permission to redistribute. IBM data terms are independent from the code's MIT license. Data is a fictional/demo snapshot, not a live feed or representative deployment cohort. Blank `TotalCharges` is converted to missing and imputed inside the training pipeline.
