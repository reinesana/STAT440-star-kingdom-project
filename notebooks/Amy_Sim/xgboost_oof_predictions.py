"""
Generate out-of-fold (OOF) XGBoost leak probabilities for model
comparison and ensemble experiments.

Purpose
-------
Generate held-out leak probabilities for 2023-2026 using expanding
temporal validation.

These predictions will later be compared with CatBoost and LightGBM
predictions and used to test probability ensembles.

This script does NOT generate 2027 predictions.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR.parent.parent
    / "data"
    / "leak_prediction_master.csv"
)

OUTPUT_PATH = BASE_DIR / "xgboost_oof_predictions.csv"


# Features used by the finalized XGBoost model
FEATURES = [
    "material",
    "surface",
    "pipe_length",
    "pipe_age",
]

CATEGORICAL_FEATURES = [
    "material",
    "surface",
]

NUMERIC_FEATURES = [
    "pipe_length",
    "pipe_age",
]


# Final selected XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": 10,
    "max_depth": 4,
    "learning_rate": 0.10,
    "n_estimators": 100,
    "random_state": 440,
}


# ============================================================
# 2. Load and inspect master dataset
# ============================================================

master = pd.read_csv(DATA_PATH)

print("=" * 60)
print("XGBOOST OOF PREDICTION")
print("=" * 60)

print("\nData path:")
print(DATA_PATH)

print("\nMaster dataset shape:")
print(master.shape)

print("\nColumns:")
print(master.columns.tolist())

print("\nYears:")
print(sorted(master["year"].unique()))

print("\nRows by year:")
print(master["year"].value_counts().sort_index())

print("\nLeaks by year:")
print(master.groupby("year")["Y"].sum())

# ============================================================
# 3. Define expanding temporal folds
# ============================================================

TEMPORAL_FOLDS = [
    {
        "fold": 1,
        "train_start": 2019,
        "train_end": 2022,
        "test_year": 2023,
    },
    {
        "fold": 2,
        "train_start": 2019,
        "train_end": 2023,
        "test_year": 2024,
    },
    {
        "fold": 3,
        "train_start": 2019,
        "train_end": 2024,
        "test_year": 2025,
    },
    {
        "fold": 4,
        "train_start": 2019,
        "train_end": 2025,
        "test_year": 2026,
    },
]


print("\n" + "=" * 60)
print("TEMPORAL FOLDS")
print("=" * 60)

total_test_rows = 0
total_test_leaks = 0

for fold_info in TEMPORAL_FOLDS:

    train = master[
        (master["year"] >= fold_info["train_start"])
        & (master["year"] <= fold_info["train_end"])
    ]

    test = master[
        master["year"] == fold_info["test_year"]
    ]

    n_test = len(test)
    n_leaks = int(test["Y"].sum())

    total_test_rows += n_test
    total_test_leaks += n_leaks

    print(
        f"Fold {fold_info['fold']}: "
        f"Train {fold_info['train_start']}-{fold_info['train_end']} "
        f"({len(train):,} rows) -> "
        f"Test {fold_info['test_year']} "
        f"({n_test:,} rows, {n_leaks:,} leaks)"
    )


print("\nTotal held-out observations:", f"{total_test_rows:,}")
print("Total held-out leaks:", f"{total_test_leaks:,}")


# ============================================================
# 4. Preprocessing
# ============================================================

def make_preprocessor():
    """
    Create the preprocessing pipeline used by the XGBoost model.

    Categorical features:
        - material
        - surface
        -> one-hot encoded

    Numeric features:
        - pipe_length
        - pipe_age
        -> passed through unchanged
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


# Quick check using Fold 1 only
fold1 = TEMPORAL_FOLDS[0]

train_check = master[
    (master["year"] >= fold1["train_start"])
    & (master["year"] <= fold1["train_end"])
].copy()

test_check = master[
    master["year"] == fold1["test_year"]
].copy()


preprocessor_check = make_preprocessor()

X_train_check = preprocessor_check.fit_transform(
    train_check[FEATURES]
)

X_test_check = preprocessor_check.transform(
    test_check[FEATURES]
)


print("\n" + "=" * 60)
print("PREPROCESSING CHECK — FOLD 1")
print("=" * 60)

print("Original training shape:", train_check[FEATURES].shape)
print("Transformed training shape:", X_train_check.shape)

print("Original test shape:", test_check[FEATURES].shape)
print("Transformed test shape:", X_test_check.shape)

print(
    "Number of transformed features:",
    len(preprocessor_check.get_feature_names_out()),
)

print("\nTransformed feature names:")
print(preprocessor_check.get_feature_names_out())

# ============================================================
# 5. Train XGBoost — Fold 1 test
# ============================================================

y_train_check = train_check["Y"].astype(int)
y_test_check = test_check["Y"].astype(int)


model_check = XGBClassifier(**XGB_PARAMS)

model_check.fit(
    X_train_check,
    y_train_check,
)


# Probability that Y = 1 (first leak)
test_prob_check = model_check.predict_proba(
    X_test_check
)[:, 1]


print("\n" + "=" * 60)
print("XGBOOST CHECK — FOLD 1")
print("=" * 60)

print("Training observations:", f"{len(y_train_check):,}")
print("Training leaks:", f"{int(y_train_check.sum()):,}")

print("Test observations:", f"{len(y_test_check):,}")
print("Actual test leaks:", f"{int(y_test_check.sum()):,}")

print("\nPredicted probabilities:")
print("Count:", f"{len(test_prob_check):,}")
print("Minimum:", test_prob_check.min())
print("Maximum:", test_prob_check.max())
print("Mean:", test_prob_check.mean())

print("\nFirst 10 probabilities:")
print(test_prob_check[:10])


# ============================================================
# 6. Generate raw OOF predictions — all temporal folds
# ============================================================

oof_results = []


for fold_info in TEMPORAL_FOLDS:

    fold_number = fold_info["fold"]
    train_start = fold_info["train_start"]
    train_end = fold_info["train_end"]
    test_year = fold_info["test_year"]

    print("\n" + "=" * 60)
    print(f"RUNNING FOLD {fold_number}")
    print("=" * 60)

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train_fold = master[
        (master["year"] >= train_start)
        & (master["year"] <= train_end)
    ].copy()

    test_fold = master[
        master["year"] == test_year
    ].copy()


    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_fold[FEATURES]
    y_train = train_fold["Y"].astype(int)

    X_test = test_fold[FEATURES]
    y_test = test_fold["Y"].astype(int)


    # --------------------------------------------------------
    # Fit preprocessing on training data only
    # --------------------------------------------------------

    preprocessor = make_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)


    # --------------------------------------------------------
    # Fit XGBoost
    # --------------------------------------------------------

    model = XGBClassifier(**XGB_PARAMS)

    model.fit(
        X_train_transformed,
        y_train,
    )


    # --------------------------------------------------------
    # Predict raw leak probabilities
    # --------------------------------------------------------

    raw_probability = model.predict_proba(
        X_test_transformed
    )[:, 1]


    # --------------------------------------------------------
    # Store held-out predictions
    # --------------------------------------------------------

    fold_results = pd.DataFrame(
        {
            "pipe_id": test_fold["pipe_id"].values,
            "year": test_fold["year"].values,
            "y_true": y_test.values,
            "xgb_raw_probability": raw_probability,
            "fold": fold_number,
        }
    )

    oof_results.append(fold_results)


    print(
        f"Train {train_start}-{train_end}: "
        f"{len(train_fold):,} observations"
    )

    print(
        f"Test {test_year}: "
        f"{len(test_fold):,} observations"
    )

    print(
        f"Actual leaks: "
        f"{int(y_test.sum()):,}"
    )

    print(
        f"Mean predicted probability: "
        f"{raw_probability.mean():.6f}"
    )


# ============================================================
# Combine all held-out predictions
# ============================================================

xgb_oof = pd.concat(
    oof_results,
    ignore_index=True,
)


print("\n" + "=" * 60)
print("COMBINED XGBOOST OOF RESULTS")
print("=" * 60)

print("Shape:", xgb_oof.shape)

print("\nRows by year:")
print(xgb_oof["year"].value_counts().sort_index())

print("\nActual leaks by year:")
print(xgb_oof.groupby("year")["y_true"].sum())

print(
    "\nTotal actual leaks:",
    f"{int(xgb_oof['y_true'].sum()):,}"
)

print(
    "Duplicate pipe_id + year:",
    xgb_oof.duplicated(
        subset=["pipe_id", "year"]
    ).sum()
)

print(
    "Missing probabilities:",
    xgb_oof["xgb_raw_probability"].isna().sum()
)

print(
    "Probability range:",
    xgb_oof["xgb_raw_probability"].min(),
    "to",
    xgb_oof["xgb_raw_probability"].max(),
)

# ============================================================
# 7. Evaluate raw OOF predictions
# ============================================================

y_true_oof = xgb_oof["y_true"].values
y_prob_oof = xgb_oof["xgb_raw_probability"].values

# Same threshold used in the previous model comparison
THRESHOLD = 0.5

y_pred_oof = (
    y_prob_oof >= THRESHOLD
).astype(int)


pr_auc = average_precision_score(
    y_true_oof,
    y_prob_oof,
)

brier = brier_score_loss(
    y_true_oof,
    y_prob_oof,
)

precision = precision_score(
    y_true_oof,
    y_pred_oof,
)

recall = recall_score(
    y_true_oof,
    y_pred_oof,
)

f1 = f1_score(
    y_true_oof,
    y_pred_oof,
)


print("\n" + "=" * 60)
print("RAW XGBOOST OOF METRICS")
print("=" * 60)

print(f"Threshold: {THRESHOLD}")
print(f"PR-AUC:    {pr_auc:.6f}")
print(f"Brier:     {brier:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1:        {f1:.6f}")

print(
    "Predicted leaks:",
    f"{int(y_pred_oof.sum()):,}"
)

print(
    "Actual leaks:",
    f"{int(y_true_oof.sum()):,}"
)

# ============================================================
# 8. Export final XGBoost OOF prediction CSV
# ============================================================

xgb_export = (
    xgb_oof[
        [
            "pipe_id",
            "year",
            "y_true",
            "xgb_raw_probability",
        ]
    ]
    .rename(
        columns={
            "xgb_raw_probability": "predicted_probability"
        }
    )
    .copy()
)


# ------------------------------------------------------------
# Final validation before export
# ------------------------------------------------------------

expected_columns = [
    "pipe_id",
    "year",
    "y_true",
    "predicted_probability",
]

assert xgb_export.columns.tolist() == expected_columns
assert len(xgb_export) == 138_833
assert xgb_export["y_true"].sum() == 8_586
assert xgb_export.duplicated(
    subset=["pipe_id", "year"]
).sum() == 0

assert xgb_export["predicted_probability"].isna().sum() == 0

assert xgb_export["predicted_probability"].between(
    0, 1
).all()


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

xgb_export.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 60)
print("FINAL XGBOOST OOF CSV")
print("=" * 60)

print("Saved to:")
print(OUTPUT_PATH)

print("\nShape:")
print(xgb_export.shape)

print("\nColumns:")
print(xgb_export.columns.tolist())

print("\nActual leaks:")
print(int(xgb_export["y_true"].sum()))

print("\nFirst 5 rows:")
print(xgb_export.head())