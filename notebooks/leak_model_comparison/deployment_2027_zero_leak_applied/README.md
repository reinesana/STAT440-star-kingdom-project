# 2027 ensemble forecast with confirmed zero-leak rules

Run from the project repository:

```powershell
python notebooks/leak_model_comparison/predict_2027_zero_leak_applied.py
```

Dependencies: numpy, pandas, scikit-learn, xgboost, catboost, lightgbm, joblib.

## What changed compared with the previous forecast

- **Purpose:** apply the confirmed zero-leak rules to the 2027 forecast; this is a rule-based correction, not a new model-selection experiment.
- **Originally polyurethane pipes:** 1,678 pipes previously had model-generated probabilities; all three component probabilities and the ensemble probability are now exactly zero.
- **Previously leaked/replaced pipes:** 14,113 pipes were excluded from the earlier prediction population. They are now explicitly included with zero probability in the new full-inventory file; the cost-join file still excludes them.
- **Other predicted pipes:** probabilities are exactly unchanged from the previous export.
- **Missing-input pipes:** the same 17 pipes remain unknown, not zero. They still need a separate missing-data policy.
- **Expected leak count:** decreased from 3,113.3 to 3,080.4, a reduction of approximately 32.9 expected leaks (1.06%). These totals exclude the unknown contribution of the 17 missing-input pipes.
- **Unchanged modeling:** the three model configurations, equal one-third weights, historical training data, and historical selection/evaluation results are unchanged. The script refits the same models and then applies the zero-risk override.
- **Preserved files:** the original predict_2027.py and deployment_2027/ outputs remain unchanged; corrected outputs are saved separately in deployment_2027_zero_leak_applied/.
- **Historical metrics:** the reported 2023–2025 and 2026 metrics belong to the earlier workflow and have not been recalculated with the zero-risk override.

## Counts and expected leaks

- Historical training: 305,391 pipe-year rows, 2019–2026.
- Full inventory: 43,039 pipes.
- Cost-join forecast: 28,909 pipes without historical leaks and with complete inputs.
- Originally polyurethane pipes zeroed in that forecast: 1,678.
- Historically leaked/replaced inventory pipes assigned zero: 14,113.
- Full-inventory zero-risk pipes, counted once: 15,791.
- Missing-input pipes with unknown probability: 17.
- Revised expected leaks across known probabilities: 3,080.3585. This excludes the unknown contribution of the 17 missing-input pipes.
- Components after the rule: XGBoost 6,207.1900; CatBoost 1,529.1820; LightGBM 1,504.7036.

## Files

| File | Purpose |
|---|---|
| ensemble_2027_probabilities.csv | Component and ensemble probabilities for the 28,909 complete-input, no-history pipes; includes zero-risk original polyurethane and probability_status. |
| ensemble_2027_for_cost_join.csv | pipe_id, year, leak_probability for joining to the cost team. |
| full_inventory_2027_probabilities.csv | All inventory IDs, explicit zero risks, missing-input unknowns, and reasons/status. |
| missing_input_17_pipes.csv | Pipes whose risk remains unknown. |
| prediction_features_2027.csv | Features used for the complete-input population. |
| inventory_scope_audit.csv | Input eligibility; eligible means complete input/no history, not necessarily nonzero risk. |
| ensemble_models.joblib | Fitted estimators and encoding; the zero rule must still be applied when using them directly. |
| manifest.json | Settings, hashes, versions, counts, confirmed rule and totals. |

## Difference from cost-team pipe count

The cost team has 28,926 IDs; the cost-join export has 28,909. All exported IDs match the cost population. The 17 missing probabilities are not zero: nine wrought-iron pipes lack installation dates (P10869, P12275–P12282); eight lack material (P09986, P09987, P42036–P42041). None has a recorded historical leak.
Left-join from the cost table, preserve these 17, and flag missing probabilities rather than filling them with zero. Expected leak loss equals annual leak probability times conditional repair cost, not construction cost.

## Validation and limitations

Validated IDs, bounds, exact zero-rule application, unchanged non-polyurethane component values before/after the override, ensemble arithmetic, reconstructed historical features, and CSV round-trip contents.
The original approximately 3,113 expected-leak total is superseded by this rule-adjusted forecast. Historical selection and 2026 metrics were not recalculated; they describe the earlier workflow without this explicit override.
No 2027 labels exist. Substantial model disagreement remains. The unknown contribution of missing-input pipes is not included in the total. This is not a full 2027–2051 replacement simulation.
When planned replacement occurs, newly polyurethane pipes must also receive zero future leak risk.