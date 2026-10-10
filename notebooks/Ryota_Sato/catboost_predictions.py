"""Train CatBoost by year and export leak probabilities for 2023-2026.

Colab:
    !pip -q install catboost pandas numpy
    !python catboost_predictions.py

Place leak_prediction_master.csv in the current working directory.
For a different location, specify --input /leak_prediction_master.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# Frozen configuration used in the model comparison.
# No class weighting or additional hyperparameter tuning.
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "random_seed": 440,
    "verbose": False,
    "allow_writing_files": False,
}

# Use the same four predictors as the original code; year only defines the folds.
# Exclude pipe_id (identifier) and Y (target) from the predictors.
FEATURES = ["material", "surface", "pipe_length", "pipe_age"]
CATEGORICAL_FEATURES = ["material", "surface"]
TEST_YEARS = (2023, 2024, 2025, 2026)
EXPECTED_ROWS = 138833
OUTPUT_COLUMNS = ["pipe_id", "year", "y_true", "predicted_probability"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("leak_prediction_master.csv"))
    parser.add_argument("--output", type=Path, default=Path("catboost_predictions.csv"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output CSV paths must be different.")

    # Read identifiers as strings to preserve formatting, including leading zeros.
    data = pd.read_csv(args.input, dtype={"pipe_id": "string"})
    required = ["pipe_id", "year", "Y"] + FEATURES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Missing values found. Apply the same missing-value handling as the original code.")
    for column in ["year", "Y"]:
        numeric = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError(f"{column} must contain integers.")
        data[column] = numeric.astype(int)
    if not data["Y"].isin([0, 1]).all():
        raise ValueError("Y must contain only 0 or 1.")
    if data.duplicated(["pipe_id", "year"]).any():
        raise ValueError("Duplicate pipe_id and year combinations found.")
    for column in ["pipe_length", "pipe_age"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"{column} contains non-finite values.")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype(str)

    target = data[data["year"].isin(TEST_YEARS)]
    if len(target) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected target row count: {len(target):,} (expected {EXPECTED_ROWS:,})")

    predictions = []
    for year in TEST_YEARS:
        # Expanding window: train on all previous years and predict the test year.
        # Training periods: 2023 -> 2019-2022; 2024 -> 2019-2023;
        #                   2025 -> 2019-2024; 2026 -> 2019-2025.
        train = data[data["year"] < year]
        test = data[data["year"] == year]
        if train.empty or test.empty or train["Y"].nunique() != 2:
            raise ValueError(f"Insufficient training data, test data, or training classes for {year}.")
        model = CatBoostClassifier(**MODEL_PARAMS)
        # Do not use test-year labels for training or early stopping.
        model.fit(train[FEATURES], train["Y"], cat_features=CATEGORICAL_FEATURES)
        positive_index = list(model.classes_).index(1)
        probability = model.predict_proba(test[FEATURES])[:, positive_index]
        fold = test[["pipe_id", "year", "Y"]].rename(columns={"Y": "y_true"}).copy()
        fold["predicted_probability"] = probability
        predictions.append(fold)
        print(f"{year}: {len(train):,} training rows / {len(test):,} predictions")

    # Export years in ascending order, preserving input order within each year.
    result = pd.concat(predictions, ignore_index=True)[OUTPUT_COLUMNS]
    if len(result) != EXPECTED_ROWS or result.duplicated(["pipe_id", "year"]).any():
        raise ValueError("Output row count or uniqueness validation failed.")
    probabilities = result["predicted_probability"].to_numpy()
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    expected = target[["pipe_id", "year", "Y"]].rename(columns={"Y": "y_true"})
    keys = ["pipe_id", "year"]
    if not result.set_index(keys)["y_true"].sort_index().equals(expected.set_index(keys)["y_true"].sort_index()):
        raise ValueError("Prediction keys or true labels do not match the input.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    # Validate the saved CSV as well; the row count excludes the header.
    saved = pd.read_csv(args.output, dtype={"pipe_id": "string"})
    if list(saved.columns) != OUTPUT_COLUMNS or len(saved) != EXPECTED_ROWS:
        raise ValueError("Saved CSV columns or row count do not match the expected output.")
    print(f"CSV saved successfully: {args.output} ({len(saved):,} rows; columns: {','.join(saved.columns)})")


if __name__ == "__main__":
    main()
