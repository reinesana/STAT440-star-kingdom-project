# Takito October 13 submission candidates

Six headerless CSVs contain the submitted construction-circle lists. Coordinates and radii are metres. Every file has 38,455 circles, with the same geometry as the oldest-first baseline and a different ordering.

Read the complete English report followed by Japanese: [experiments_10_13_takito](../../reports/experiments_10_13_takito.md). A [formatted HTML version](../../reports/experiments_10_13_takito.html) can be opened locally and copied into Google Docs.

| File | Ordering | Submitted score M$ |
| --- | --- | ---: |
| [01_catboost_risk.csv](01_catboost_risk.csv) | CatBoost cost × two-year leak probability | -27.54 |
| [02_stable_risk.csv](02_stable_risk.csv) | Shrinkage ensemble cost × two-year leak probability | -25.70 |
| [03_equal_risk.csv](03_equal_risk.csv) | Equal-weight ensemble cost × two-year leak probability | -25.58 |
| [04_rmse_weighted_risk.csv](04_rmse_weighted_risk.csv) | RMSE-weighted ensemble cost × two-year leak probability | -27.16 |
| [05_catboost_cost_only.csv](05_catboost_cost_only.csv) | CatBoost cost only | -35.45 |
| [06_stable_risk_per_work_cost.csv](06_stable_risk_per_work_cost.csv) | Shrinkage expected repair cost / initial construction cost | -15.61 |

Baseline score: -25.58 M$. Scores are rounded values from the [submission screenshot](../../reports/assets/experiments_10_13_takito_scores.png), matched bottom to top as baseline then 01–06. They are not calculated from the CSVs or local no-future-leak simulation.

Cost predictions use the saved full-history 2027 models in [ensemble_for_cost_prediction](../../notebooks/ensemble_for_cost_prediction). Leak probabilities use the separate saved [risk.txt](../risk_priority_models/risk.txt) model, not the newer leak-model ensemble. See [risk metadata](../risk_priority_models/metadata.json).

[comparison.csv](comparison.csv) contains construction costs and completion years **under no future leaks**, not actual evaluator cost breakdowns. The six `*.validation.json` files document exported-circle coverage and budget checks. [manifest.json](manifest.json) records source and submission hashes. [cost_prediction_provenance_check.json](cost_prediction_provenance_check.json) confirms full-history training metadata, ensemble calculations, prediction hashes and agreement with the costs used in the submission audits.

Generate and validate the CSVs from the repository root:

```powershell
python src/build_cost_submissions.py
```

Requires numpy, pandas, scipy and lightgbm. The command writes the same dated candidate paths. Predictions, source data and the saved risk model are read without modification.

Only the six numbered CSVs are submission files. Reports, JSON validation records, and comparison tables are supporting material.
