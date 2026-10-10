# Repair cost prediction: single-model and ensemble candidates

CatBoost is now an explicit candidate: it has the best single-model RMSE. The shrinkage ensemble remains the budget-focused default because it has smaller annual total bias. The candidate set has been expanded without automatically replacing that default.

## Observed evaluation results

| candidate              |   mean_annual_RMSE |   pooled_RMSE |   annual_RMSE_sd |   mean_absolute_annual_total_bias_pct |   worst_absolute_annual_total_bias_pct |
|:-----------------------|-------------------:|--------------:|-----------------:|--------------------------------------:|---------------------------------------:|
| takito_catboost        |          124569.85 |     126180.72 |         25624.46 |                                 20.50 |                                  34.45 |
| ensemble_equal         |          125151.59 |     127044.21 |         25702.59 |                                  6.47 |                                  12.39 |
| ensemble_stable        |          125230.52 |     127113.11 |         25777.62 |                                  4.61 |                                  10.02 |
| ensemble_balanced      |          125255.35 |     127124.82 |         25856.33 |                                  6.38 |                                  13.33 |
| ensemble_rmse_weighted |          125329.07 |     127093.90 |         25167.36 |                                  7.67 |                                  21.30 |
| prior_surface_mean     |          125255.32 |     127022.98 |         25188.50 |                                  8.63 |                                  21.22 |

Evaluation covers 8,592 observed repair events in 2023–2026. RMSE in original cost units is primary. Mean annual RMSE gives each year equal weight; pooled RMSE combines all events; annual_RMSE_sd is the sample standard deviation across four years. Annual total bias is 100 × (predicted total / observed total − 1); the table reports the mean and maximum absolute annual bias.

CatBoost mean annual RMSE is 124,569.85 versus 125,230.52 for shrinkage (about 0.53% higher). Worst absolute annual total bias is 34.45% versus 10.02%. Shrinkage annual RMSE standard deviation is about 2.34% higher than the reference: annual RMSE stabilization has not been established. Budget-bias improvement and RMSE variation should be assessed separately.

## Candidate explanations

### Best single model: CatBoost
One raw-cost CatBoost model. Lowest mean annual and pooled RMSE among evaluated single models; higher budget bias.

CSV: [predictions/takito_catboost.csv](predictions/takito_catboost.csv)

### Equal-weight ensemble
Ten components, each weighted 10%. Lowest mean annual RMSE among the ensembles.

CSV: [predictions/ensemble_equal.csv](predictions/ensemble_equal.csv)

### Shrinkage ensemble (budget-focused default)
50% regularized past-OOF RMSE weights + 50% equal weights. Every component has at least 5% weight. Lowest mean and worst absolute annual total bias among compared candidates.

CSV: [predictions/ensemble_stable.csv](predictions/ensemble_stable.csv)

### Constrained balance ensemble
Past-only optimization considers annual total bias, relative annual RMSE variation and a uniform-weight penalty, with 1–70% component weights and an RMSE constraint. Its observed bias is worse than shrinkage.

CSV: [predictions/ensemble_balanced.csv](predictions/ensemble_balanced.csv)

### RMSE-optimized ensemble
Nonnegative weights summing to one minimize past mean annual RMSE. Zero weights are allowed; concentration risk remains.

CSV: [predictions/ensemble_rmse_weighted.csv](predictions/ensemble_rmse_weighted.csv)

### Surface mean (reference)
All-history surface-specific mean, shrunk toward the global mean using an Optuna-selected smoothing strength.

CSV: [predictions/prior_surface_mean.csv](predictions/prior_surface_mean.csv)

## Components and final deployment weights

| component               |   takito_catboost |   ensemble_equal |   ensemble_stable |   ensemble_balanced |   ensemble_rmse_weighted |   prior_surface_mean |
|:------------------------|------------------:|-----------------:|------------------:|--------------------:|-------------------------:|---------------------:|
| prior_surface_mean      |              0.00 |            10.00 |             19.61 |               61.56 |                    74.44 |               100.00 |
| anthony_linear_raw      |              0.00 |            10.00 |              9.86 |                6.58 |                     0.00 |                 0.00 |
| anthony_gamma           |              0.00 |            10.00 |             11.08 |                1.00 |                     1.34 |                 0.00 |
| anthony_rf_log_mean     |              0.00 |            10.00 |             10.39 |                1.00 |                     5.68 |                 0.00 |
| takito_xgboost          |              0.00 |            10.00 |              6.05 |                1.00 |                     4.55 |                 0.00 |
| takito_catboost         |            100.00 |            10.00 |              9.97 |                1.00 |                     4.50 |                 0.00 |
| takito_lightgbm         |              0.00 |            10.00 |              5.00 |                1.00 |                     0.00 |                 0.00 |
| rion_gamma_refit        |              0.00 |            10.00 |              5.32 |                1.00 |                     0.00 |                 0.00 |
| rion_rf_raw_refit       |              0.00 |            10.00 |              7.36 |                1.00 |                     0.00 |                 0.00 |
| rion_lgb_log_mean_refit |              0.00 |            10.00 |             15.35 |               24.86 |                     9.48 |                 0.00 |

Weights are percentages. Ensembles combine the same ten components: surface mean; Anthony linear/Gamma/log-RF; Takito XGBoost/CatBoost/LightGBM; Rion Gamma/raw-RF/log-LightGBM. Log-model back-transform corrections were also selected using past data only.

## Work performed and leakage controls

1. Retained all 14,113 train.csv events from 2019–2026 and joined inventory by ID. Restored six missing-date records; no expensive events were removed or capped.
2. Built January-1 age, material/surface, metric distances, inventory neighborhoods, endpoint routes and past-cost aggregates using labels strictly before each prediction year. Coordinates are meters; planar crossings do not imply connectivity.
3. Ran 30 Optuna trials per model/origin in 2021–2027, with up to two latest preceding inner validation years and six feature packs. The 2020 cold start uses fixed settings. Of 2,100 trial slots, 1,978 completed and 122 failed/were interrupted.
4. Compared OLS/Ridge/Lasso/ElasticNet; searched L1/L2 for XGBoost/LightGBM, L2 for Gamma/CatBoost, and structural controls for RF. Final Anthony linear regression selected no regularization. Gamma L1 was not evaluated.
5. Learned ensemble weights only from preceding OOF years. Final 2027 components were trained on all 14,113 events with seed 440. Excluded existing Shana predictions and models requiring unavailable realized failure time/weekday.
6. Verified ten saved models, 80 temporal configurations, independent metric/blend recomputation and invariance to future cost/date changes.

## Best single-model settings

```json
{
  "feature_pack": "topology",
  "iterations": 150,
  "depth": 2,
  "learning_rate": 0.02932217209384241,
  "l2_leaf_reg": 1.0143268892104178,
  "random_strength": 0.00153216083646107,
  "rsm": 0.7357430693448022,
  "border_count": 128,
  "one_hot_max_size": 6,
  "leaf_estimation_iterations": 4,
  "bootstrap_type": "MVS",
  "subsample": 0.99792074633296
}
```

Final CatBoost selected 49 topology-pack features, depth 2 and 150 iterations. These are deployment settings for 2027; each historical fold used separately tuned past-only settings. All ten final configurations are in configs/ and evidence/final_model_settings.csv.

## Prediction population and interpretation

The actual inventory filename is pipes.csv. All 43,039 IDs were considered and the 14,113 IDs in train.csv skipped, leaving 28,926 predictions per candidate. CSVs contain exactly id,predicted_cost, sorted by decreasing cost and then ascending ID. Scenario year is 2027 with history frozen at end-2026. Predictions for 27,240 supported-material pipes come from saved full-data fits; the same models additionally predict 1,686 pipes (1,678 PU and eight missing-material rows). No cost labels exist for these materials. These predictions use native unknown-category handling and are unvalidated extrapolations, outside the OOF evaluation. They are numeric estimates, not zero assignments, but should not be budgeted with the same confidence as supported-material estimates. Identify them in evidence/prediction_scope.csv.

Costs are conditional on a failure, not repair necessity or failure probability. Summing every CSV row does not produce an annual budget: failure probabilities/counts are needed. Monetary units follow the source data and are not assumed to be yen. Candidate selection follows inspection of historical results; this is not an untouched final holdout guarantee.

## Annual results

| model                  |   year |   n_test |      RMSE |   total_bias_pct |
|:-----------------------|-------:|---------:|----------:|-----------------:|
| prior_surface_mean     |   2023 |     1826 | 158531.63 |             4.93 |
| takito_catboost        |   2023 |     1826 | 158723.20 |            10.34 |
| ensemble_equal         |   2023 |     1826 | 159167.41 |             3.47 |
| ensemble_rmse_weighted |   2023 |     1826 | 158626.76 |             5.33 |
| ensemble_stable        |   2023 |     1826 | 159453.14 |            -1.95 |
| prior_surface_mean     |   2024 |     1955 |  99232.67 |            -1.01 |
| takito_catboost        |   2024 |     1955 |  97780.40 |            28.78 |
| ensemble_equal         |   2024 |     1955 |  99065.40 |             5.96 |
| ensemble_rmse_weighted |   2024 |     1955 |  99458.85 |            -2.62 |
| ensemble_stable        |   2024 |     1955 |  99245.82 |             2.59 |
| prior_surface_mean     |   2025 |     1773 | 114824.73 |            21.22 |
| takito_catboost        |   2025 |     1773 | 115475.00 |            34.45 |
| ensemble_equal         |   2025 |     1773 | 113674.62 |            12.39 |
| ensemble_rmse_weighted |   2025 |     1773 | 114737.81 |            21.30 |
| ensemble_stable        |   2025 |     1773 | 113590.50 |            10.02 |
| prior_surface_mean     |   2026 |     3038 | 128432.27 |             7.36 |
| takito_catboost        |   2026 |     3038 | 126300.80 |             8.43 |
| ensemble_equal         |   2026 |     3038 | 128698.93 |            -4.06 |
| ensemble_rmse_weighted |   2026 |     3038 | 128492.84 |             1.43 |
| ensemble_stable        |   2026 |     3038 | 128632.64 |            -3.90 |
| ensemble_balanced      |   2023 |     1826 | 159543.95 |            -9.96 |
| ensemble_balanced      |   2024 |     1955 |  98941.25 |             1.75 |
| ensemble_balanced      |   2025 |     1773 | 113997.86 |            13.33 |
| ensemble_balanced      |   2026 |     3038 | 128538.33 |            -0.50 |

## Sources and validation

Source snapshot: `a09355d835d9acbb5746f11edcddd0cfd5e8ec3e`. [Training labels](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/train.csv), [Inventory](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/pipes.csv).
[Sources receipt](evidence/sources.html) · [Temporal validation](evidence/validation.json) · [Export checks](evidence/export_validation.json) · [Per-pipe support](evidence/prediction_scope.csv)
