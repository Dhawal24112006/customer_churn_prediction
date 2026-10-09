# Illustrative retention economics

## Parameters
- `c`: contact cost per customer (must be positive, in one consistent currency).
- `V`: estimated gross value of a customer retained due to outreach (nonnegative).
- `e`: assumed *incremental* chance outreach prevents churn **conditional on the customer otherwise churning** (between 0 and 1).
- `B`: campaign budget, nonnegative.
- `p_i`: estimated churn probability without intervention. No individual treatment effectiveness is estimated from this dataset.

## Calculation
For customer i: hypothetical benefit `p_i × e × V`; cost `c`; expected net `p_i × e × V − c`. Rank by expected net descending. Select only positive-net accounts with `p_i ≥ threshold`, respecting `floor(B/c)` contact slots. Sum selected costs cannot exceed B. When testing a labeled validation set at threshold t, `TP` customers flagged who historically churned; `FP` flagged who did not; `FN` churned but not flagged; `TN` otherwise. An illustrative retrospective benefit proxy is `TP × e × V`; hypothetical net is this minus `(TP+FP) × c`. It is *not* a measurement of customers retained.

## Important mismatch between policy and evaluation
The validation threshold table assumes **all customers above a threshold are contacted**, and only compares thresholds whose total contacts fit its validation-set budget. Deployment ranks customers by estimated expected net and caps contacts to the specified budget. Because validation set and scoring population differ in size, a threshold selected under a validation-set total budget may not be optimal in the full customer universe. Re-tune policy constraints to the actual campaign cohort rather than transferring fixed counts mechanically.

## Why assumptions are speculative
The IBM snapshot has no historical experiment, offer acceptance or treatment-control outcome. A probability of churn is not a probability of persuasion. An RCT comparing pre-specified treatment and control cohorts, coupled with incremental net benefit estimates, is necessary to establish campaign lift. Customer acquisition costs, cannibalization, fatigue, discounts and consent constraints are excluded. Baseline demo numbers (`c=10`, `V=200`, `e=0.10`, `B=1000`) are configurable **illustrative units**, never realized cash flow. Currency is user-defined.

## Recommended pilot
Pre-register outcome horizon, arm assignment, unit economics and primary incremental retention metric; randomize contact and control; estimate average and subgroup uplift with confidence intervals; revisit ethical constraints, consent and calibration; only then consider automated targeting.
