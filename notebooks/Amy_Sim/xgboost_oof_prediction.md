# XGBoost OOF Predictions

## Purpose

This script generates held-out XGBoost leak probabilities for model comparison and ensemble testing.

It evaluates XGBoost on known outcomes from 2023-2026.

It does not generate 2027 predictions.

---

## Data

Input:

`data/leak_prediction_master.csv`

One observation is one pipe-year while the pipe is still at risk of its first leak.

After a pipe experiences its first leak, it is removed from later risk-set observations.

Target:

- `Y = 1`: first leak occurs in that year
- `Y = 0`: no first leak occurs in that year

Features:

- `material`
- `surface`
- `pipe_length`
- `pipe_age`

`pipe_id` is used only as an identifier.

`year` is used for temporal splitting.

---

## Temporal Validation

Four expanding-window folds are used:

| Fold | Training  | Test |
| ---- | --------- | ---- |
| 1    | 2019–2022 | 2023 |
| 2    | 2019–2023 | 2024 |
| 3    | 2019–2024 | 2025 |
| 4    | 2019–2025 | 2026 |

The model never trains on the year it predicts.

The four test years contain:

- 138,833 pipe-year observations
- 8,586 actual first leaks

---

## Preprocessing

Categorical features are one-hot encoded.

Numeric features are passed through unchanged.

Preprocessing is fitted only on each fold's training data.

The final transformed dataset has 14 features.

---

## XGBoost

Parameters:

- `objective = binary:logistic`
- `eval_metric = logloss`
- `scale_pos_weight = 10`
- `max_depth = 4`
- `learning_rate = 0.10`
- `n_estimators = 100`
- `random_state = 440`

Each fold outputs the predicted probability of a first leak.

---

## OOF Results

Combined held-out observations:

`138,833`

Actual leaks:

`8,586`

Metrics:

| Metric                |   Result |
| --------------------- | -------: |
| PR-AUC                | 0.318501 |
| Brier Score           | 0.059042 |
| Precision @ 0.5       | 0.452026 |
| Recall @ 0.5          | 0.493827 |
| F1 @ 0.5              | 0.472003 |
| Predicted Leaks @ 0.5 |    9,380 |

The threshold is used only for classification metrics.

PR-AUC and Brier Score use the predicted probabilities directly.

---

## Output

Output file:

`predictions/xgboost_oof_predictions.csv`

Columns:

- `pipe_id`
- `year`
- `y_true`
- `predicted_probability`

The file contains 138,833 rows.

There are no duplicate `pipe_id + year` observations.

There are no missing predicted probabilities.

---

## Next Step

Compare this file with the CatBoost and LightGBM OOF prediction files.

Merge the files by `pipe_id + year`.

Check that all models use the same observations and `y_true`.

Then evaluate individual models and probability ensembles.
