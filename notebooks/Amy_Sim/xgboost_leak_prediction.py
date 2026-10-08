"""
Final XGBoost leak-prediction model for the STAT 440 Star Kingdom project.

Purpose
-------
Fit the selected XGBoost leak model on 2019-2025 pipe-year observations and
fit a sigmoid probability calibrator on 2026 observations.

This script intentionally STOPS before generating 2027 leak probabilities.
The next team step is to construct/confirm the 2027 first-leak risk set and
apply the fitted preprocessor + calibrated model to that dataset.

Selected model
--------------
Features:
    material, surface, pipe_length, pipe_age

XGBoost:
    objective="binary:logistic"
    eval_metric="logloss"
    scale_pos_weight=10
    max_depth=4
    learning_rate=0.10
    n_estimators=100
    random_state=440

Temporal deployment split:
    XGBoost training: 2019-2025
    Calibration:      2026
    Prediction:       2027 (not performed here)

Calibration:
    sigmoid / Platt-style calibration using CalibratedClassifierCV
"""

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "leak_prediction_master.csv"

print("Python file location:", BASE_DIR)
print("Looking for data at:", DATA_PATH)
print("Exists:", DATA_PATH.exists())

FEATURES = [
    "material",
    "surface",
    "pipe_length",
    "pipe_age",
]

CATEGORICAL_FEATURES = ["material", "surface"]
NUMERIC_FEATURES = ["pipe_length", "pipe_age"]

TRAIN_END_YEAR = 2025
CALIBRATION_YEAR = 2026

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": 10,
    "max_depth": 4,
    "learning_rate": 0.10,
    "n_estimators": 100,
    "random_state": 440,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_master(path=DATA_PATH):
    """Load and minimally validate the first-leak pipe-year master dataset."""
    master = pd.read_csv(path)

    required = {"year", "Y", *FEATURES}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return master


def make_final_split(master):
    """
    Create the final temporal deployment split.

    2019-2025 (all observations through 2025): XGBoost training
    2026:                                      calibration only
    """
    final_train = master[master["year"] <= TRAIN_END_YEAR].copy()
    final_cal = master[master["year"] == CALIBRATION_YEAR].copy()

    if final_train.empty:
        raise ValueError("Final training set is empty.")
    if final_cal.empty:
        raise ValueError("2026 calibration set is empty.")

    return final_train, final_cal


def make_preprocessor():
    """Create preprocessing used by the selected XGBoost model."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )


def fit_final_model(final_train, final_cal):
    """
    Fit preprocessing and XGBoost on data through 2025.

    The 2026 observations are transformed but are NOT used to fit either
    the preprocessor or XGBoost model.
    """
    X_train = final_train[FEATURES]
    y_train = final_train["Y"]

    X_cal = final_cal[FEATURES]
    y_cal = final_cal["Y"]

    preprocessor = make_preprocessor()

    X_train_t = preprocessor.fit_transform(X_train)
    X_cal_t = preprocessor.transform(X_cal)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_t, y_train)

    return preprocessor, model, X_train_t, X_cal_t, y_train, y_cal


def fit_sigmoid_calibrator(model, X_cal_t, y_cal):
    """
    Freeze the fitted XGBoost model and fit sigmoid calibration on 2026.

    The underlying XGBoost model is not refitted on 2026.
    """
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
    )
    calibrator.fit(X_cal_t, y_cal)
    return calibrator


def calibration_diagnostics(model, calibrator, X_cal_t, y_cal):
    """
    Report deployment-calibration diagnostics on the 2026 calibration set.

    Note: these are in-sample diagnostics for the final calibrator, not a
    new independent test of generalization. The notebook's earlier rolling
    temporal evaluation provides the held-out evidence for using calibration.
    """
    raw_prob = model.predict_proba(X_cal_t)[:, 1]
    calibrated_prob = calibrator.predict_proba(X_cal_t)[:, 1]

    metrics = {
        "actual_leak_rate": y_cal.mean(),
        "raw_mean_probability": raw_prob.mean(),
        "calibrated_mean_probability": calibrated_prob.mean(),
        "raw_brier": brier_score_loss(y_cal, raw_prob),
        "calibrated_brier": brier_score_loss(y_cal, calibrated_prob),
        "raw_log_loss": log_loss(y_cal, raw_prob),
        "calibrated_log_loss": log_loss(y_cal, calibrated_prob),
        "raw_min_probability": raw_prob.min(),
        "raw_max_probability": raw_prob.max(),
        "calibrated_min_probability": calibrated_prob.min(),
        "calibrated_max_probability": calibrated_prob.max(),
    }

    return metrics, raw_prob, calibrated_prob


def print_summary(master, final_train, final_cal, X_train_t, X_cal_t, metrics):
    """Print a compact reproducibility summary."""
    print("=" * 68)
    print("FINAL XGBOOST LEAK MODEL")
    print("=" * 68)

    print("\nData")
    print(f"Master rows:              {len(master):,}")
    print(
        "Training years:           "
        f"{int(final_train['year'].min())}-{int(final_train['year'].max())}"
    )
    print(f"Calibration year:         {CALIBRATION_YEAR}")
    print(f"Training rows:            {len(final_train):,}")
    print(f"Calibration rows:         {len(final_cal):,}")
    print(f"Transformed features:     {X_train_t.shape[1]}")

    print("\nEvents")
    print(f"Training leaks:           {int(final_train['Y'].sum()):,}")
    print(f"Calibration leaks:        {int(final_cal['Y'].sum()):,}")
    print(f"Training leak rate:       {final_train['Y'].mean():.6f}")
    print(f"2026 leak rate:           {final_cal['Y'].mean():.6f}")

    print("\nSelected XGBoost parameters")
    for key, value in XGB_PARAMS.items():
        print(f"{key:24s} {value}")

    print("\n2026 final-calibrator diagnostics")
    print(f"Actual leak rate:         {metrics['actual_leak_rate']:.6f}")
    print(f"Raw mean probability:     {metrics['raw_mean_probability']:.6f}")
    print(
        f"Calibrated mean prob.:    "
        f"{metrics['calibrated_mean_probability']:.6f}"
    )
    print(f"Raw Brier score:          {metrics['raw_brier']:.6f}")
    print(f"Calibrated Brier score:   {metrics['calibrated_brier']:.6f}")
    print(f"Raw log loss:             {metrics['raw_log_loss']:.6f}")
    print(f"Calibrated log loss:      {metrics['calibrated_log_loss']:.6f}")
    print(
        "Raw probability range:   "
        f"[{metrics['raw_min_probability']:.6f}, "
        f"{metrics['raw_max_probability']:.6f}]"
    )
    print(
        "Calibrated prob. range:   "
        f"[{metrics['calibrated_min_probability']:.6f}, "
        f"{metrics['calibrated_max_probability']:.6f}]"
    )

    print("\nStatus")
    print("Final XGBoost + sigmoid calibrator are ready.")
    print("2027 probabilities are intentionally NOT generated by this script.")
    print(
        "Next step: confirm/build the 2027 first-leak risk set, transform it "
        "with the fitted preprocessor, and call "
        "calibrator.predict_proba(... )[:, 1]."
    )


def main():
    master = load_master()
    final_train, final_cal = make_final_split(master)

    (
        preprocessor,
        model,
        X_train_t,
        X_cal_t,
        y_train,
        y_cal,
    ) = fit_final_model(final_train, final_cal)

    calibrator = fit_sigmoid_calibrator(model, X_cal_t, y_cal)

    metrics, raw_prob, calibrated_prob = calibration_diagnostics(
        model,
        calibrator,
        X_cal_t,
        y_cal,
    )

    print_summary(
        master,
        final_train,
        final_cal,
        X_train_t,
        X_cal_t,
        metrics,
    )

    # Objects intentionally remain available inside main for the eventual
    # 2027 prediction step:
    #   preprocessor
    #   model
    #   calibrator
    #
    # Do not construct the 2027 risk set here until the team confirms the
    # final prediction-population logic.


if __name__ == "__main__":
    main()
