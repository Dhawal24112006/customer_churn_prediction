# Model card — Telco churn benchmark

**Status:** Educational, research/demonstration only. **Version:** 0.1.0. **Target:** binary `Churn == Yes` in the IBM illustrative snapshot. **Input:** 19 tabular attributes; exclude `customerID`, target, and post-churn fields. **Output:** score between zero and one; configured classification and high/medium/low risk bins.

**Intended users:** internship reviewers, data scientists learning churn risk classification. **Not intended for:** automated customer denial, legal eligibility, live targeting, or assertion of causal intervention impact.

**Training:** training-only stratified 60/20/20 split; training-only stratified CV; compare dummy, logistic, random forest, optional XGBoost. Validation AP selects classifier; validation hypothetical campaign net determines threshold. Final test is accessed after both choices. Versioned fitted pipeline serialized to `models/churn_pipeline.joblib`; companion `metadata.json` includes runtime versions, metrics, timestamp and assumptions. Predictions may change across library versions.

**Dataset and provenance:** IBM demo source and Kaggle mirror documented in `data/README.md`. 7,043 rows in the source version; actual counts derived at runtime. No evidence that sample matches a real deployment region, temporal distribution, protected groups or current churn base rate.

**Evaluation:** measured results are **not filled in here**, to avoid stale/fabricated scores. Run training; inspect `reports/validation_model_comparison.csv`, `reports/final_test_metrics.json`, ROC/PR plots and calibration. Training uses no estimates from the test for model/threshold selection. Non-temporal random split may overstate prospective generalization.

**Known limitations:** selection bias, class imbalance, unmeasured competitors and pricing, missing time axes and reasons for churn, feature dependency, changing promotions, demographic attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) and possible disparate impacts. Calibration and performance must be checked by subgroup and period; fairness and privacy review required before deployment. IBM sample includes no experimental outreach outcomes or counterfactual retention values.

**Security and governance:** joblib/pickle can execute code at load time; load only trusted local files. Do not commit actual identifiable customers, CSV data, credentials or artifacts. Monitor drift, threshold stability, interventions and complaints in any ethically reviewed deployment.
