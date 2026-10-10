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


**Why existing Shana submissions were excluded (temporal leakage):** Outcomes from evaluation years 2023–2025 were used for feature/model selection, so those predictions cannot serve as clean OOF ensemble inputs. This temporal validation contamination is distinct from using realized future failure hour/weekday, which is unavailable at forecast time. The architectures themselves can be candidates if refitted with selection entirely inside past-only folds.

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

## Detailed handling of missing, unseen and invalid values

### What was actually found

| Item | Inventory: 43,039 | Prediction targets: 28,926 | Handling |
|:---|---:|---:|:---|
| PU (`polyurethane`) | 1,678 | 1,678 | Material is known, but has no cost labels; use each model's unseen-category handling |
| Missing material | 8 | 8 | Replace with `__unknown__`, without asserting iron or PU |
| Missing lay date | 15 | 9 | Keep age NaN, set `age_missing=1`; retain the six training rows too |
| Missing surface | 0 | 0 | No observed replacements required; categorical blanks become `__unknown__` |
| Missing/non-numeric/nonfinite coordinates | 0 | 0 | Use observed coordinates; no automatic repair of invalid new coordinates |
| Zero endpoint length | 0 | 0 | None observed; new zero-length input requires correction before directional features |

train.csv has no missing ID, Date, Time or Cost and no nonpositive costs. The six training records with missing inventory lay date were restored from raw train.csv rather than dropped with an incomplete enriched table. All 14,113 labels were retained. Training materials are wrought iron, cast iron, gray iron, brass and copper; neither PU nor missing-material costs were observed. Expensive events, including the maximum observed cost of 7,257,576.36, were not deleted or capped as presumed outliers.

### How each model calculated PU and missing-material costs

PU retains the string `polyurethane`; missing material becomes `__unknown__`. The material–surface composite category carries the same values. Neither is relabeled as a supported material in the source features, although the encoder introduces the limitations below.

| Family | Actual unseen-category handling | Basis and limitation |
|:---|:---|:---|
| CatBoost | Pass PU/`__unknown__` as strings through CatBoost categorical processing | Learned trees and selected age, surface, length, route and other inputs; no PU-specific price relationship was learned |
| XGBoost/LightGBM | Convert categories outside fitted category levels to category missing | Learned missing-category behavior plus remaining features; that path is not validated for PU |
| Linear/Gamma/Random Forest | `OneHotEncoder(handle_unknown='ignore', drop='first')` makes every dummy in the unseen category block zero | Encoding collides with the dropped reference category; other inputs remain, but this does not establish an appropriate PU material effect |
| Surface mean | Ignore material; shrink surface-specific average toward global mean | `(surface cost sum + α × global mean)/(surface count + α)`, with Optuna-selected α; unseen surface falls back to global mean |
| Ensembles | Weighted average of the ten saved component predictions | Averaging does not remove unsupported-material uncertainty or substitute for a PU-specific fit |

### Missing age and missing historical statistics

Age is `(2027-01-01 − Lay date in days)/365.25`. Missing lay date produces NaN age/log-age and `age_missing=1`; the age-band aggregation key is the separate missing group `-1`. Age was not assumed to be zero.

Linear, Gamma and RF pipelines use `SimpleImputer(strategy='median', add_indicator=True)` with medians fitted only on their training period, then standardize. Indicators are added for columns missing during fit, not automatically for every column that might become missing later. CatBoost, XGBoost and LightGBM retain numeric NaN and use their learned native missing-value behavior.

For material, material–surface, route and similar groups with no historical repairs, aggregation count and sum are set to zero. This does not assign zero repair cost. Mean features are `(past sum + 20 × past global mean)/(past count + 20)`; zero count gives the global past mean. Median, maximum and standard deviation remain NaN if unsupported. High-cost rates are shrunk as `(past high-cost count + 50 × past global rate)/(past count + 50)`. Radius aggregates follow the same approach: nearby costs from other materials may be available, but they are not PU observations.

Inventory neighborhoods and endpoint routes use the same pipes.csv; repair-state features use only dates preceding the forecast origin. Neighborhood inventory candidates require known lay dates, so missing-lay-date pipes are excluded as neighbors. Endpoint connections are distinguished from planar crossings; geometry is a proxy, not excavated length or verified physical connectivity.

### Unusual values and limits on new inputs

Observed coordinates are finite, endpoint lengths are nonzero, and known lay dates precede 2019. These were checked before using the data; there is no universal invalid-value sanitization. Nonblank unseen strings follow category rules above; typos are not automatically distinguished from genuinely new categories. Duplicate IDs, unparseable dates, nonfinite coordinates, zero-length geometry or missing inventory IDs require correction and revalidation. Missing coordinates were not replaced with zero or another location.

Negative model predictions are clipped at zero with `max(prediction, 0)`. Log models are first transformed back with `exp`/`expm1` and their fitted correction factor, then clipped. Nonfinite predictions fail validation and are not saved; large predictions are not capped. CatBoost's saved CSV has 35 zero predictions; the other five candidates have none. These arise from model inference and nonnegative constraints, not blanket zero assignment to unseen materials; zero is not evidence of free repair. Counts are recorded in [input_quality_audit.json](evidence/input_quality_audit.json).

See [input_quality_by_pipe.csv](evidence/input_quality_by_pipe.csv) for per-row flags and [input_quality_audit.json](evidence/input_quality_audit.json) for counts. Material support and missing lay date are separate dimensions: a supported-material row can still have missing age.

## Model inputs, outputs and coordination with failure-timing models

### Shared inputs

- Inventory: `Pipe ID`, `Lay date`, `Material`, `Surface`, `GPS x1`, `GPS y1`, `GPS x2`, `GPS y2`. ID is a join/output key, not a predictor.
- Training/history: train.csv `Pipe ID`, `Date`, `Cost`. Date establishes historical years and cutoff. `Time` exists in the file but is not an input to these models.
- Scenario: year 2027, age reference 2027-01-01, observed history strictly before 2027-01-01. Full inventory and past history, not just one pipe row, are needed for neighborhood and route features.
- Estimator inputs: only selected derived columns, explicitly listed in [model_input_contract.json](evidence/model_input_contract.json) and each configs JSON `features`. Ensembles consume the same ten component predictions and apply their saved weights.

### Features and outputs by component

| model                   | feature_pack   |   selected_columns | output                |
|:------------------------|:---------------|-------------------:|:----------------------|
| anthony_gamma           | inventory      |                 52 | Original-unit cost    |
| anthony_linear_raw      | inventory      |                 52 | Original-unit cost    |
| anthony_rf_log_mean     | inventory      |                 57 | Log prediction → cost |
| prior_surface_mean      | surface only   |                  1 | Surface mean cost     |
| rion_gamma_refit        | inventory      |                 54 | Original-unit cost    |
| rion_lgb_log_mean_refit | all            |                309 | Log prediction → cost |
| rion_rf_raw_refit       | local_history  |                 61 | Original-unit cost    |
| takito_catboost         | topology       |                 49 | Original-unit cost    |
| takito_lightgbm         | inventory      |                 57 | Original-unit cost    |
| takito_xgboost          | base           |                 11 | Original-unit cost    |

base covers year, material, surface, year-start age and endpoint length; inventory adds neighborhood inventory, past repair state, orientation and interactions; group_history adds material/surface/grid/age-band/route historical costs; local_history adds radius/nearest-neighbor historical costs; topology adds endpoint routes and their history; all combines every group. Exact columns are selected/removed by family, so pack names alone are not a full schema. The surface-mean configuration contains common columns, but actual inference only reads surface.

### Outputs

Each candidate returns one nonnegative point estimate of repair cost conditional on a failure in the 2027 scenario, in original cost units. Submitted CSVs contain only `id,predicted_cost`. They do not predict failure time, probability, uncertainty intervals, PU reliability or intervention priority. Input-quality and material-support metadata are separate files.

### Alignment with the failure-timing team

Share ID keys, inventory snapshot, observation cutoff, excluded IDs and material/lay-date missing flags. If that team retains repaired training IDs, repeat failures and material replacement need a separate state definition. For an annual budget, obtain failure probabilities `p_i` for the same population and interval and form `Σ p_i × predicted_cost_i`. If multiple failures are allowed, use expected event counts only after checking that this severity estimate applies to each event. The combined budget model has not thereby been independently validated.

The current severity models do not require realized failure month, weekday or hour; predicted failure dates were not inputs to these CSVs either. Substituting a predicted incident-date age changes the year-start specification and requires validation. For later years, update scenario year/year-start age and re-run inference without adding unobserved future repair history. Track unsupported-material coverage separately in the timing and cost models.


## Total cost if every target pipe is repaired in 2027

| Candidate              |   Supported: 27,240 pipes |   PU: 1,678 (extrapolated) |   Missing material: 8 (extrapolated) |   All 28,926 targets |
|:-----------------------|--------------------------:|---------------------------:|-------------------------------------:|---------------------:|
| takito_catboost        |            678,080,616.66 |              60,261,238.25 |                           553,308.96 |       738,895,163.87 |
| ensemble_equal         |            598,280,863.28 |              44,285,679.47 |                           290,308.64 |       642,856,851.39 |
| ensemble_stable        |            590,257,789.64 |              33,788,551.47 |                           220,471.30 |       624,266,812.40 |
| ensemble_balanced      |            590,650,232.16 |              16,542,838.04 |                            80,416.31 |       607,273,486.51 |
| ensemble_rmse_weighted |            606,600,766.13 |              28,702,046.70 |                           149,819.68 |       635,452,632.51 |
| prior_surface_mean     |            628,897,965.63 |              15,064,555.23 |                            48,285.75 |       644,010,806.62 |

This table sums saved CSV predictions under the explicit assumption that every target pipe is repaired once in 2027. The population is 28,926 pipes after excluding the 14,113 train.csv IDs as requested, not all 43,039 inventory pipes. Units are those of the source costs; neither yen nor dollars is asserted. PU and missing-material contributions are separate to expose reliance on unvalidated extrapolation.

No failure probabilities are applied. This is a repair-all scenario, not the usual annual failure budget. Models were trained on costs incurred at failures, not planned preventive replacement; shared mobilization costs, coordinated-work discounts and inflation have not been modeled separately. These totals are not calibrated quotes for a coordinated replacement project. The earlier RMSE and total-bias explanations remain as historical validation on actually failed pipes.

Totals were calculated with decimal arithmetic from the saved CSV values. The same figures are available in the [machine-readable totals CSV](evidence/repair_all_2027_totals.csv).

## Sources and validation

Source snapshot: `a09355d835d9acbb5746f11edcddd0cfd5e8ec3e`. [Training labels](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/train.csv), [Inventory](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/pipes.csv).
[Sources receipt](evidence/sources.html) · [Temporal validation](evidence/validation.json) · [Export checks](evidence/export_validation.json) · [Per-pipe support](evidence/prediction_scope.csv)
