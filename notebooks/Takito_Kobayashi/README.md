# Takito Kobayashi: construction cost prediction

Single-model XGBoost, CatBoost and LightGBM experiments by Shana / Takito.

## Prediction CSVs

- `output_xgboost.csv`
- `output_catboost.csv`
- `output_lightgbm.csv`

Each CSV contains **8,586 held-out predictions** with exactly two columns:
`pipe_id,predicted_cost`. Costs use the original dataset's monetary unit.
These are test predictions for 2023–2026, not fitted predictions for the training
rows. The year for each pipe can be looked up in `data/train_predict_cost.csv`.

| Fold | Training years | Test year | Test rows |
|---|---|---|---:|
| 1 | 2019–2022 | 2023 | 1,824 |
| 2 | 2019–2023 | 2024 | 1,951 |
| 3 | 2019–2024 | 2025 | 1,773 |
| 4 | 2019–2025 | 2026 | 3,038 |

## Results

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| XGBoost | 29,878.19 | 130,462.53 |
| CatBoost | 29,803.85 | 130,560.76 |
| LightGBM | 28,170.72 | 130,766.84 |

Means give each of the four test years equal weight. Per-fold test and training
metrics are in `results/fold_metrics.csv`; the full report, feature catalogue,
feature adoption paths and validation evidence are in `results/`.

## Code and reproducibility

`executed_code/` contains byte-for-byte copies of the Python scripts used in the
experiment, including feature engineering, selection, Optuna tuning, training
and independent verification. `executed_code_sha256.json` records their hashes.
The geometry helpers also include their import dependencies; their legacy model
training routines were not used. Some original scripts expect the original
workspace layout; use the portable runner below to reproduce the saved outputs.
`selected_configs/` contains the final features, parameters, tree counts and
thread counts for all 12 model/fold combinations.

From the repository root, with Python 3.12 or a compatible Python installation:

```bash
python -m pip install -r notebooks/Takito_Kobayashi/requirements.txt
python notebooks/Takito_Kobayashi/reproduce.py --output takito_reproduction
```

The runner verifies the input CSV hashes, rebuilds all 470 features using the
archived code, refits the frozen configurations, writes the three prediction
CSVs, and compares them with the published predictions. It does not repeat the
4,200-trial search. Use a fresh output directory. The runner preserves the
original training thread counts (XGBoost/LightGBM: 2; CatBoost: 3) and seed 440.
Numerical reproduction can depend on library versions and the machine.

## Selection and availability assumptions

The only labelled source is `data/train_predict_cost.csv` (14,107 rows).
`data/pipes.csv` supplies supplementary inventory geometry and attributes.
All 12 configurations include historical cost features. Historical costs and
repair records use strictly prior calendar years, frozen at January 1; the
current year's and future costs are excluded. Prior-year final costs are assumed
available at that time. Inventory pipes with unknown lay dates or lay dates after
the incident are excluded. Inventory attributes are assumed valid at the incident
date; no retirement or attribute-change history is available.

Within each outer training period, the two latest years form expanding temporal
validation splits. Selection minimizes their mean RMSE. Starting from the current
baseline, all single additions/deletions, strong-correlation replacements
(absolute training-only Spearman correlation >= 0.98), equivalent categorical
encodings, and related feature-family additions/deletions are compared. Improving
changes update the baseline, and comparisons repeat. Parameter/tree-count
optimization and feature selection repeat until the final feature set is stable.
This is the best observed result from finite searches, not a global optimality
guarantee. Outer test errors were not used for this feature selection or tuning;
earlier project experiments had examined these years, so the project does not
have a completely untouched final test set.

No ensembles, cost caps, outlier removal or log-target transformation were used.
Negative predictions are clipped to zero. Coordinates and Euclidean distances
are meters. Straight endpoint lengths are not actual curved or excavation
lengths; planar crossings do not establish physical connectivity, and pipe
proximity does not establish the repair location.

Verification independently recomputed all 12 test/train metric pairs and their
averages. Mutating each test year's and all future costs and rebuilding the
cost-dependent features left that test year's inputs unchanged for all four
folds. Inventory calculations were independently checked on sampled rows.
