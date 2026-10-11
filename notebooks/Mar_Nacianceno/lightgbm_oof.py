"""
Generate out-of-fold (OOF) LightGBM leak predictions for 2023-2026.

Supports baseline (unweighted) and class-weighted LightGBM using the
same expanding temporal folds. Weighted mode selects class weights
using historical validation data.

Both modes include historical threshold tuning, sigmoid calibration,
and held-out evaluation. Raw and calibrated predictions are exported
for model comparison and ensemble experiments.

Place leak_prediction_master.csv in the current working directory.
For baseline LightGBM: python lightgbm_oof.py --mode baseline
For class-weighted LightGBM: python lightgbm_oof.py --mode weighted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score,
)

FEATURES = ("material", "surface", "pipe_length", "pipe_age")
TEST_YEARS = (2023, 2024, 2025, 2026)
WEIGHTS = (1, 2, 5, 10, 15, 20)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
OUTPUT_COLUMNS = ["pipe_id", "year", "y_true", "predicted_probability"]


def train_model(X, y, weight):
    # Keep LightGBM settings fixed across years and model comparisons.
    model = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=20, colsample_bytree=1.0,
        subsample=1.0, n_estimators=500, scale_pos_weight=weight,
        random_state=440, verbosity=-1, n_jobs=4,
    )
    model.fit(X, y)
    return model


def best_threshold(y, probability):
    results = [(f1_score(y, probability >= t, zero_division=0), float(t)) for t in THRESHOLDS]
    return max(results, key=lambda pair: pair[0])[1]


def evaluate(y, probability, threshold):
    predicted = probability >= threshold
    return {
        "threshold": threshold,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "average_precision": average_precision_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability, labels=[0, 1]),
        "predicted_leaks": int(predicted.sum()),
        "actual_leaks": int(np.sum(y)),
    }


def fit_calibrator(y, probability):
    # Platt-style sigmoid calibration fitted on the historical validation year.
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=440, max_iter=1000)
    calibrator.fit(logits, y)
    return calibrator


def apply_calibrator(calibrator, probability):
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("leak_prediction_master.csv"))
    parser.add_argument("--mode", choices=("weighted", "baseline"), default="weighted")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype={"pipe_id": "string"})
    required = ["pipe_id", "year", "Y", *FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[required].isna().any().any() or df.duplicated(["pipe_id", "year"]).any():
        raise ValueError("Missing values or duplicate pipe-year observations")
    if not df["Y"].isin([0, 1]).all():
        raise ValueError("Y must contain only 0 and 1")
    if not df["year"].between(2019, 2026).all():
        raise ValueError("Unexpected year outside 2019-2026")
    for col in ("pipe_length", "pipe_age"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values in {col}")

    X = pd.get_dummies(df[list(FEATURES)], columns=["material", "surface"], dtype=float)
    y = df["Y"].astype(int)
    raw_folds, calibrated_folds, metrics, weight_trials = [], [], [], []

    for test_year in TEST_YEARS:
        validation_year = test_year - 1
        inner_train = df["year"] < validation_year
        validation = df["year"] == validation_year
        train = df["year"] < test_year
        test = df["year"] == test_year
        if not all((inner_train.any(), validation.any(), train.any(), test.any())):
            raise ValueError(f"Missing train, validation or test observations for {test_year}")
        if y.loc[inner_train].nunique() < 2 or y.loc[validation].nunique() < 2:
            raise ValueError(f"Insufficient classes for historical validation in {test_year}")

        # Choose weight using historical F1 at 0.5; never use test-year labels.
        candidate_weights = WEIGHTS if args.mode == "weighted" else (1,)
        candidates = []
        for weight in candidate_weights:
            candidate = train_model(X.loc[inner_train], y.loc[inner_train], weight)
            val_probability = candidate.predict_proba(X.loc[validation])[:, 1]
            val_f1 = f1_score(y.loc[validation], val_probability >= 0.5, zero_division=0)
            candidates.append((val_f1, weight, val_probability))
            weight_trials.append({"test_year": test_year, "validation_year": validation_year,
                                  "positive_weight": weight, "validation_f1_at_0_5": val_f1})
        _, selected_weight, val_raw = max(candidates, key=lambda entry: entry[0])

        # Fit sigmoid calibration and choose thresholds on the prior year only.
        calibrator = fit_calibrator(y.loc[validation], val_raw)
        val_calibrated = apply_calibrator(calibrator, val_raw)
        raw_threshold = best_threshold(y.loc[validation], val_raw)
        calibrated_threshold = best_threshold(y.loc[validation], val_calibrated)

        # Refit on all years preceding the held-out test year.
        model = train_model(X.loc[train], y.loc[train], selected_weight)
        raw_probability = model.predict_proba(X.loc[test])[:, 1]
        calibrated_probability = apply_calibrator(calibrator, raw_probability)
        identifiers = df.loc[test, ["pipe_id", "year"]].copy()
        identifiers["y_true"] = y.loc[test].to_numpy()
        raw_fold = identifiers.copy()
        raw_fold["predicted_probability"] = raw_probability
        raw_folds.append(raw_fold)
        calibrated_fold = identifiers.copy()
        calibrated_fold["predicted_probability"] = calibrated_probability
        calibrated_folds.append(calibrated_fold)

        details = {"test_year": test_year, "training_start": int(df.loc[train, "year"].min()),
                   "training_end": test_year - 1, "validation_year": validation_year,
                   "n_train": int(train.sum()), "n_validation": int(validation.sum()),
                   "n_test": int(test.sum()), "selected_positive_weight": selected_weight}
        metrics.append({**details, "probability_type": "raw", **evaluate(y.loc[test], raw_probability, raw_threshold)})
        metrics.append({**details, "probability_type": "calibrated", **evaluate(y.loc[test], calibrated_probability, calibrated_threshold)})
        print(f"{test_year}: Train {int(train.sum()):,} -> Test {int(test.sum()):,} "
              f"| weight={selected_weight} | raw F1={metrics[-2]['f1']:.4f} "
              f"| calibrated F1={metrics[-1]['f1']:.4f} "
              f"| raw Brier={metrics[-2]['brier_score']:.4f} "
              f"| calibrated Brier={metrics[-1]['brier_score']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "lightgbm" if args.mode == "weighted" else "lightgbm_baseline"
    for name, folds in (("oof_predictions", raw_folds), ("calibrated_oof_predictions", calibrated_folds)):
        output = pd.concat(folds, ignore_index=True).sort_values(["year", "pipe_id"], kind="stable")
        assert list(output.columns) == OUTPUT_COLUMNS
        assert len(output) == 138_833 and int(output["y_true"].sum()) == 8_586
        assert not output.duplicated(["pipe_id", "year"]).any()
        assert output["predicted_probability"].between(0, 1).all()
        path = args.output_dir / f"{prefix}_{name}.csv"
        output.to_csv(path, index=False)
        print(f"Saved {len(output):,} predictions to {path}")
    pd.DataFrame(metrics).to_csv(args.output_dir / f"{prefix}_final_metrics.csv", index=False)
    pd.DataFrame(weight_trials).to_csv(args.output_dir / f"{prefix}_weight_selection.csv", index=False)
    print("Historical validation selects weights, thresholds and calibrator; test years remain held out.")
    print("Note: calibration/threshold scores on the selection year are in-sample; use test-year metrics for comparison.")


if __name__ == "__main__":
    main()
