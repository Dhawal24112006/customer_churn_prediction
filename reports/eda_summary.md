# EDA summary

Customers: 7043; observed churn rate: 26.54%. Missing entries: {'TotalCharges': 11}. Full-row duplicates: 0.

Plots: eda_churn_patterns.png, eda_numeric_distributions.png, eda_correlations.png.

Tests: two-sided chi-square (categorical) and Mann–Whitney U (numeric), with Benjamini–Hochberg FDR-adjusted p-values in eda_hypothesis_tests.csv. H0: no association / identical rank distribution; H1: association / differing distributions. Check expected chi-square cell counts in the table. Significance is not causality. Cramér's V and rank-biserial correlation describe effect magnitude. Snapshot sampling and correlated observations may violate inferential assumptions.
