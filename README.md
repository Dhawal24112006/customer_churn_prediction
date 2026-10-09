# Customer Churn Prediction & Retention Intelligence Platform

An end-to-end Python portfolio project for the **IBM Telco Customer Churn** sample. It validates data, explores statistical patterns, compares four classifiers, measures held-out discrimination/calibration, explains predictions and ranks hypothetical retention contacts under a budget. **Churn risk is not treatment responsiveness; correlations and SHAP scores are not causal explanations.**

## Business question and capabilities
A telecom operator wants to anticipate cancellations and allocate a limited outreach budget. This system produces per-account probabilities (`Churn == Yes`), an interpretable logistic baseline, optional XGBoost, validation-selected policy and an interactive Streamlit application. It includes EDA inference, class-imbalance-aware metrics, leakage-safe transformations, tests, CI and explainability.

## Architecture
`CSV → strict schema validation → train/validation/test stratification → [feature construction + column imputation/encoding] → classifier/CV → validation model and cost-policy selection → one final test → serialized pipeline → inference/dashboard/SHAP`.

The same train-fitted preprocessing and classifier are used for all new predictions. **No unseen data is used for fitting transforms or choosing thresholds.**

## Repository map
- `src/churn_prediction/`: validation, preprocessing, features, EDA, train, evaluation, SHAP, prediction, retention
- `app/streamlit_app.py`: six interactive pages
- `notebooks/01_exploratory_data_analysis.ipynb`: reproducible exploration and chart interpretation
- `tests/`: schema, pipelines, probabilities and budget checks
- `data/`: instructions, dataset ignored by Git
- `reports/`, `models/`: output folders (generated artifacts ignored)
- `docs/`: methodological decisions, model card and business formulas
- `.github/workflows/ci.yml`: pytest on push/pull request

## Dataset: attribution and access
Primary download is the [IBM archived original GitHub sample](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). Alternative: [Kaggle listing](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), whose filename is `WA_Fn-UseC_-Telco-Customer-Churn.csv`; rename to `Telco-Customer-Churn.csv`. Target `Churn` means whether the customer left in the last month in this illustrative sample. Dataset licensing is separate from MIT code; **the CSV is not distributed in this repository**. See [data instructions](data/README.md).

## Setup (Python 3.10–3.12 recommended)
Commands from repository root:

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```
**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```
Optional explainability/XGBoost: `python -m pip install -r requirements-optional.txt`. If PowerShell restricts activation, use `.venv\Scripts\python.exe -m pip install -e .` and prefix commands with that Python.

## Exact execution workflow
```bash
python -m churn_prediction.data_validation --download
python -m churn_prediction.data_validation
python -m churn_prediction.eda
python -m churn_prediction.train
python -m churn_prediction.predict --input data/Telco-Customer-Churn.csv --output reports/predictions.csv
python -m pip install -r requirements-optional.txt
python -m churn_prediction.explain
streamlit run app/streamlit_app.py
python -m pytest -q
```
If the dataset download fails, follow `data/README.md` and place the CSV manually. If XGBoost is absent, training automatically uses the other three classifiers. To skip it explicitly use `python -m churn_prediction.train --no-xgboost`. SHAP is optional and is slower on CPU. Quick local smoke mode: `python -m churn_prediction.train --fast --no-xgboost` (no CV search; for smoke tests only). To change hypothetical economics: `python -m churn_prediction.train --contact-cost 10 --retained-value 200 --effectiveness 0.1 --budget 1000`.

For the notebook: `python -m pip install jupyterlab` then `jupyter lab notebooks/01_exploratory_data_analysis.ipynb`; or run the source-controlled EDA directly using the command above. Visuals are saved to `reports/`.

## Evaluation and statistical method
- **Accuracy:** fraction correctly classified, insufficient alone for imbalanced churn.
- **Precision:** of contacted accounts flagged as churn, fraction actually labeled churn in the retrospective sample.
- **Recall:** fraction of labeled churners flagged.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** discrimination across all FPR thresholds.
- **Average precision/PR-AUC:** precision–recall ranking, especially informative when positives are less prevalent.
- **Log loss/calibration:** assess probability quality, not only ranking.

We train with stratified 60/20/20 splits, stratified CV on **training only**, hyperparameter search ranked by mean CV average precision, then select model by held-out **validation average precision**, with log loss as tie breaker. Threshold chosen on validation estimates using configurable contact cost, value, effectiveness and budget. Finally evaluate once on untouched test. Logistic encoded-scale coefficients and conditional odds ratios are saved to `reports/logistic_coefficients.csv` (observational associations, not causal risk ratios). Cross-validation variability is recorded in `reports/validation_model_comparison.csv`. The test comparison file intentionally contains **only the locked selected model**. Validation diagnostic and test figures appear only after running. No results in this README are invented. Real forward-looking use needs temporal splits and probability recalibration where needed.

## Explainability and policy
`python -m churn_prediction.explain` computes model-agnostic permutation SHAP for a small sampled set of transformed inputs and saves summary, dependence and local attributions. `feature_importance.png` shows validation-set permutation importance, which can understate correlated attributes. An odds ratio from logistic coefficients is conditional on other inputs, not an intervention effect. See [methodology](docs/methodology.md).

For risk bands: Low `<0.30`, Medium `0.30–<0.70`, High `>=0.70` (descriptive only). Priority uses `p(churn) × assumed incremental effectiveness × retained value − contact cost`, positive values first under budget. **The dataset has no outreach experiment, so expected savings are strictly hypothetical.** See [business impact](docs/business_impact.md).

## Dashboard
`streamlit run app/streamlit_app.py` opens six sections: overview, filtered customer explorer, evaluation, explainability, budgeted retention plan, and single-customer scoring form. Missing CSV/model is reported with actionable commands. The dashboard loads a saved artifact; it never retrains on startup.

## Screenshots
Screenshots must be captured from a real run before sharing: `docs/screenshots/overview.png`, `docs/screenshots/evaluation.png`, `docs/screenshots/retention.png` (**placeholders only; none supplied**).

## Testing / quality / limitations
Run `python -m pytest -q`. CI installs the package and runs tests on Python 3.10 and 3.12. Model artifacts are local joblib pickle objects and must only be loaded from trusted sources. The data are illustrative, at one snapshot, with non-prospective labels; sensitive/demographic attributes may drive unfair decisions; temporal, consent, privacy, subgroup, calibration and causal testing are necessary for business use. See [model card](docs/model_card.md). Do not interpret numeric effect tests as proof of churn causes.

## Publishing to GitHub
Create a **new empty public repository** on GitHub named `customer-churn-prediction` (do not initialize with a README/license since local versions exist), then from this folder run:
```bash
git init
git branch -M main
git add .
git status
git commit -m "Build end-to-end churn and retention intelligence platform"
git remote add origin https://github.com/<YOUR_USERNAME>/customer-churn-prediction.git
git push -u origin main
```
GitHub will handle your own authentication (browser/credential manager or SSH). Never paste a token into code. Verify files under the repository's **Code** tab, confirm `data/*.csv`, `models/*.joblib`, secrets and generated reports were excluded, and check **Actions** for test status. Run locally and add *real* screenshots only after verifying them. Add a concise README summary and live dashboard link to your resume once deployed.

## Future work
Out-of-time testing, known-at-scoring-time feature registry, bootstrapped uncertainty intervals, formal calibration, segment/fairness audits, uplift/RCT experiments, monitored drift and MLOps artifact lineage. These are future enhancements, not completed features.

## License
MIT for project code (`LICENSE`). IBM/Kaggle data terms remain separate. Credit the [IBM sample](https://github.com/IBM/telco-customer-churn-on-icp4d) and [Kaggle mirror](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
