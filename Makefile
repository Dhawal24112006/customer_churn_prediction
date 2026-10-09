.PHONY: install optional data validate eda train explain predict dashboard test
install:
	python -m pip install -e .
optional:
	python -m pip install -r requirements-optional.txt
data:
	python -m churn_prediction.data_validation --download
validate:
	python -m churn_prediction.data_validation
da:
	python -m churn_prediction.eda
eda:
	python -m churn_prediction.eda
train:
	python -m churn_prediction.train
explain:
	python -m churn_prediction.explain
predict:
	python -m churn_prediction.predict --input data/Telco-Customer-Churn.csv --output reports/predictions.csv
dashboard:
	streamlit run app/streamlit_app.py
test:
	python -m pytest -q
