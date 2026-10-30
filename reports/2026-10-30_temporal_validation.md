# Temporal validation and public repeat — October 30, 2026

## Decision

Retain the existing integrated-coverage plan ($77.46M public). No new production candidate was rebuilt and no additional submission is recommended. [Source and evidence](../submissions/2026-10-30_temporal_validation/).

## Prediction comparison

Use 2024–2025 for development and 2026 for confirmation, with training restricted to earlier years. Existing temporal ML forecasts are reused. The fixed ML comparator averages XGBoost, CatBoost and baseline LightGBM; it does not reproduce Takito's manually staged p17b cohort overrides. Original polyurethane pipes are excluded from the common eligible prediction population.

| Year | Fine-age Brier | ML ensemble Brier | ML minus fine-age policy cost |
|---|---:|---:|---:|
| 2024 | 0.042188 | 0.035003 | -$573,279.15 |
| 2025 | 0.047959 | 0.048990 | +$3,102,686.33 |
| 2026 | 0.079419 | 0.081792 | +$1,485,923.17 |

Lower is better in both comparisons. The development-selected ML ensemble has worse 2026 Brier and log loss than the fine-age baseline. A fixed 50/50 blend has better 2026 Brier, but it was not the development winner; selecting it after inspecting confirmation results would reuse those labels. Both approaches underpredict 2026 leak counts: approximately 54% for fine-age and 22% for ML. This indicates calibration weakness, not proof of underfitting.

Conditional repair-cost comparisons use matching observed repairs and past-only predictions. CatBoost and the past-only two-model stack improve RMSE over material/surface means in all three years; corrected log RF has the lowest MAE. Cost bias and MAE are not uniformly improved. Final 2027 weights are not retroactively used in historical tests. See [annual cost metrics](../submissions/2026-10-30_temporal_validation/evidence/cost_annual_metrics.csv).

## Controlled planning checks

Six one-year historical replays hold inventory midpoint-circle geometry, past-only material/surface cost forecasts, shared dynamic ordering, and the $10M construction budget fixed. Only leak forecasts vary. Actual held-year repair records measure counterfactual avoided repairs. ML performs better in 2024; fine-age performs better in 2025 and 2026. These are retrospective one-year policy costs, not official scores or a 25-year backtest. Missing ML inputs use an identical fine-age fallback. Alternative cost models were not varied in this policy comparison.

Future-label/cost mutation checks passed for Ryota fits at all three origins. Saved ML predictions were audited against source definitions, but all base estimators were not retrained and no complete future-feature invariance proof is claimed for them.

## Submission and limitations

The existing CSV was revalidated with `2026-10-26-area-exponent`: 30,094 units, no uncovered original iron pipes, and no-future-leak completion in 2044. Its SHA-256 remains `fe3a108ca087fa89431a88ed53d7d93c6201b8381fccf270b02d12d208209f10`.

The user subsequently supplied a repeat public result displayed as $77.46M at 16:06 on October 30, with UniUni second. The annual breakdown visually matches the earlier result. This is repeatability evidence, not a new independent test; uploaded-file identity was not independently verified.

The team had already inspected these historical years. Private performance, absence of overfitting, and 25-year generalization remain unestablished. Do not tune another candidate to the leaderboard gap or reselect on these confirmation results.
