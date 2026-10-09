"""Illustrative cost-sensitive planning, not a causal treatment-effect model."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Assumptions:
    contact_cost: float = 10.0
    retained_value: float = 200.0
    effectiveness: float = 0.10
    budget: float = 1000.0

    def __post_init__(self):
        if self.contact_cost <= 0 or self.retained_value < 0 or self.budget < 0 or not 0 <= self.effectiveness <= 1:
            raise ValueError("Requires contact_cost > 0, value/budget >= 0 and effectiveness between 0 and 1")


def risk_category(probability: float) -> str:
    return "High" if probability >= 0.7 else "Medium" if probability >= 0.3 else "Low"


def threshold_table(y_true, probabilities, assumptions: Assumptions, thresholds=None) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Valid matched labels and probabilities in [0,1] required")
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for threshold in thresholds:
        flagged = p >= threshold
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())
        n = int(flagged.sum())
        cost = n * assumptions.contact_cost
        # Expected value from observed validation classifications; non-causal hypothetical scenario.
        benefit = tp * assumptions.effectiveness * assumptions.retained_value
        rows.append(dict(threshold=float(threshold), flagged=n, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=tp/n if n else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                         campaign_cost=cost, estimated_benefit=benefit, estimated_net=benefit-cost,
                         within_budget=bool(cost <= assumptions.budget)))
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame) -> float:
    feasible = table.loc[table.within_budget]
    if feasible.empty:
        return 1.0  # No contacts when costs exceed available budget.
    return float(feasible.sort_values(["estimated_net", "threshold"], ascending=[False, False]).iloc[0].threshold)


def prioritize(customers: pd.DataFrame, probabilities, assumptions: Assumptions, threshold: float = 0.5) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    if len(customers) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    result = pd.DataFrame({"customerID": customers["customerID"].to_numpy(), "churn_probability": p})
    result["risk_category"] = [risk_category(x) for x in p]
    # Descriptive account context, not a causal explanation or individualized SHAP ranking.
    descriptors = [c for c in ["Contract", "tenure", "InternetService"] if c in customers.columns]
    if descriptors:
        result["observed_account_indicators"] = ["; ".join(f"{c}={customers.iloc[i][c]}" for c in descriptors) for i in range(len(customers))]
    else:
        result["observed_account_indicators"] = "Not supplied"
    result["intervention_cost"] = assumptions.contact_cost
    result["estimated_expected_benefit"] = p * assumptions.effectiveness * assumptions.retained_value
    result["estimated_expected_net"] = result.estimated_expected_benefit - assumptions.contact_cost
    result = result.sort_values(["estimated_expected_net", "churn_probability"], ascending=False).reset_index(drop=True)
    eligible = (result.churn_probability >= threshold) & (result.estimated_expected_net > 0)
    slots = int(assumptions.budget // assumptions.contact_cost)
    result["selected"] = False
    indices = result.index[eligible][:slots]
    result.loc[indices, "selected"] = True
    result["suggested_next_step"] = np.where(result.selected, "Offer controlled-test outreach", "No outreach under current assumptions")
    return result
