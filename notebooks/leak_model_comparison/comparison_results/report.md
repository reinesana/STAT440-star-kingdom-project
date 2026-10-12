# Leak probability model comparison

Seven supplied single-model procedures and all 120 equal-weight ensembles were compared.
Selection uses only 2023-2025; each year has equal weight. 2026 metrics were computed after selection.json was saved.

Selected: equal__xgboost_raw__catboost__lightgbm_baseline_raw

Members: xgboost_raw, catboost, lightgbm_baseline_raw
Equal weight: 0.333333 each.
Development mean annual Brier: 0.038216.
Best single: lightgbm_baseline_raw; Brier 0.039399.
Development Brier improvement over best single: 3.003%.

Selected procedure on 2026: Brier 0.077797; log loss 0.279958; average precision 0.304193.
2026 expected leaks: 2457.1; actual 3038; bias -19.12%.

## Single models on development years

| Model | Mean Brier | Mean log loss | Mean AP | Worst annual Brier | Mean absolute count bias (%) | Members |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm_baseline_raw | 0.039399 | 0.166386 | 0.342327 | 0.046650 | 40.44% | 1 |
| catboost | 0.040173 | 0.161172 | 0.349909 | 0.050004 | 34.69% | 1 |
| xgboost_calibrated | 0.040589 | 0.161800 | 0.371561 | 0.046368 | 33.59% | 1 |
| lightgbm_baseline_calibrated | 0.040762 | 0.159774 | 0.342327 | 0.048069 | 34.58% | 1 |
| lightgbm_weighted_calibrated | 0.041180 | 0.160703 | 0.346961 | 0.047121 | 37.61% | 1 |
| xgboost_raw | 0.050473 | 0.193553 | 0.349506 | 0.057009 | 135.89% | 1 |
| lightgbm_weighted_raw | 0.050791 | 0.194162 | 0.346961 | 0.059719 | 101.38% | 1 |

## Final 2026 evaluation (not used to reselect)

| Model | Brier | Log loss | AP | Actual leaks | Expected leaks | Count bias (%) |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost_raw | 0.088302 | 0.305532 | 0.292704 | 3038 | 4,602.5 | 51.50% |
| xgboost_calibrated | 0.084677 | 0.327423 | 0.166670 | 3038 | 1,608.5 | -47.05% |
| catboost | 0.082738 | 0.330567 | 0.306619 | 3038 | 1,462.6 | -51.86% |
| lightgbm_weighted_raw | 0.095177 | 0.325910 | 0.313565 | 3038 | 5,629.9 | 85.32% |
| lightgbm_weighted_calibrated | 0.076671 | 0.271479 | 0.313565 | 3038 | 2,157.2 | -28.99% |
| lightgbm_baseline_raw | 0.078871 | 0.325873 | 0.316176 | 3038 | 1,306.3 | -57.00% |
| lightgbm_baseline_calibrated | 0.076807 | 0.272025 | 0.316176 | 3038 | 2,173.4 | -28.46% |
| equal__xgboost_raw__catboost__lightgbm_baseline_raw | 0.077797 | 0.279958 | 0.304193 | 3038 | 2,457.1 | -19.12% |

## Limits and deployment

The candidate search is exploratory and its development scores are selection-optimistic. No claim of statistical superiority or global optimality is made.
Prior team exposure to 2026 means this is a temporal evaluation, not a guaranteed untouched final test.
New calibrated XGBoost holds the base estimator fixed and reserves the preceding year for calibration. Mar's calibrated LightGBM transfers a prior-year calibration mapping to a refitted estimator. These are distinct procedures.
XGBoost retains Amy's weighted configuration; models are compared as supplied, not with equal tuning budgets. Correlated LightGBM variants can give that model family extra representation in an ensemble.
No future probabilities were generated. Deploy the frozen member/weight choices with the same training/calibration schedule shifted to 2027 and reconstruct the eligible first-leak population.
Long-horizon forecasts require further assumptions about aging, replacement status, and the changing annual leak rate.