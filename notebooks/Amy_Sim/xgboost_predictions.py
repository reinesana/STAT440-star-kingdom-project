"""Produce comparable raw and calibrated XGBoost first-leak probabilities.

Run from any directory:
    python xgboost_predictions.py
    python xgboost_predictions.py --test-years 2023 2024 2025
    python xgboost_predictions.py --test-years 2026

Dependencies: pip install numpy pandas scikit-learn xgboost

Defaults use the shared pipe-year master and write to predictions/ beside this
script. Select models/ensemble weights on 2023-2025, then freeze the procedure
before evaluating 2026. Existing project exposure means 2026 is not guaranteed
to be an untouched test. This script does not select a winner or forecast 2027.

Raw models train on all years before each test year. For calibrated outputs,
train a separate model through test_year-2, fit sigmoid calibration on the
preceding year, and apply that SAME frozen model and calibrator to the test year.
Thus raw and calibrated outputs represent different training procedures.
Unlike transferring calibration to a refitted model, this keeps the calibration
mapping attached to the estimator whose scores it was fitted on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

FEATURES = ["material", "surface", "pipe_length", "pipe_age"]
KEYS = ["pipe_id", "year"]
OUTPUT_COLUMNS = [*KEYS, "y_true", "predicted_probability"]
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent.parent / "data" / "leak_prediction_master.csv"
PARAMETERS = dict(objective="binary:logistic", eval_metric="logloss",
                  scale_pos_weight=10, max_depth=4, learning_rate=0.10,
                  n_estimators=100, random_state=440, n_jobs=4)


def load_data(path):
    data = pd.read_csv(path, dtype={"pipe_id": "string"})
    required = [*KEYS, "Y", *FEATURES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Missing required values; use the shared cleaned master.")
    if data.pipe_id.str.strip().eq("").any() or data.duplicated(KEYS).any():
        raise ValueError("Blank identifiers or duplicate pipe-year records.")
    for column in ["year", "Y", "pipe_length", "pipe_age"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Non-finite values in {column}")
    if (data.year % 1 != 0).any() or not data.year.between(2019, 2026).all():
        raise ValueError("Expected integer historical years 2019-2026.")
    data["year"] = data.year.astype(int)
    if not data.Y.isin([0, 1]).all():
        raise ValueError("Y must be binary.")
    if (data.pipe_length < 0).any() or (data.pipe_age < 0).any():
        raise ValueError("Negative lengths or ages.")
    data["Y"] = data.Y.astype(int)
    for column in ["material", "surface"]:
        data[column] = data[column].astype(str)
    # First-leak data must not retain a pipe after its observed first event.
    events = data.loc[data.Y.eq(1), KEYS]
    if events.pipe_id.duplicated().any():
        raise ValueError("Multiple first-leak outcomes for a pipe.")
    last_year = data.groupby("pipe_id").year.max()
    if (events.pipe_id.map(last_year).to_numpy() > events.year.to_numpy()).any():
        raise ValueError("Risk set includes observations after first leak.")
    return data


def make_model():
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["material", "surface"]),
        ("numeric", "passthrough", ["pipe_length", "pipe_age"]),
    ])
    return Pipeline([("preprocessing", preprocessing),
                     ("model", XGBClassifier(**PARAMETERS))])


def fit_model(data):
    if data.empty or data.Y.nunique() != 2:
        raise ValueError("Training data must contain both outcome classes.")
    return make_model().fit(data[FEATURES], data.Y)


def logit(probability):
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def prediction_frame(test, probability):
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Invalid predicted probabilities.")
    frame = test[KEYS + ["Y"]].rename(columns={"Y": "y_true"}).copy()
    frame["predicted_probability"] = np.asarray(probability, dtype=np.float64)
    return frame[OUTPUT_COLUMNS]


def metrics(frame, kind, train, calibration_year=None):
    y, p = frame.y_true, frame.predicted_probability
    return dict(probability_type=kind, test_year=int(frame.year.iloc[0]),
                train_start=int(train.year.min()), train_end=int(train.year.max()),
                calibration_year=calibration_year, training_rows=len(train),
                test_rows=len(frame), actual_leaks=int(y.sum()),
                expected_leaks=float(p.sum()), mean_probability=float(p.mean()),
                actual_leak_rate=float(y.mean()),
                brier_score=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                average_precision=float(average_precision_score(y, p)))


def export_predictions(frame, path, expected):
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate exported keys.")
    observed = frame.set_index(KEYS).y_true.sort_index()
    reference = expected.set_index(KEYS).Y.sort_index()
    pd.testing.assert_series_equal(observed, reference, check_names=False)
    frame.to_csv(path, index=False)
    saved = pd.read_csv(path, dtype={"pipe_id": "string"})
    pd.testing.assert_frame_equal(saved, frame.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-15)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "predictions")
    parser.add_argument("--test-years", nargs="+", type=int, choices=[2023, 2024, 2025, 2026],
                        default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()
    years = sorted(set(args.test_years))
    data = load_data(args.input)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ["xgboost_oof_predictions.csv",
             "xgboost_calibrated_oof_predictions.csv", "xgboost_fold_metrics.csv",
             "xgboost_run_manifest.json"]}
    if args.input.resolve() in paths.values():
        raise ValueError("Output must not overwrite the input dataset.")
    raw_frames, calibrated_frames, rows = [], [], []
    for year in years:
        train = data.loc[data.year < year]
        calibration_train = data.loc[data.year < year - 1]
        calibration = data.loc[data.year.eq(year - 1)]
        test = data.loc[data.year.eq(year)]
        if test.empty or calibration.empty or calibration.Y.nunique() != 2:
            raise ValueError(f"Missing test data or calibration classes for {year}.")
        raw_model = fit_model(train)
        raw = prediction_frame(test, raw_model.predict_proba(test[FEATURES])[:, 1])
        frozen_model = fit_model(calibration_train)
        calibration_probability = frozen_model.predict_proba(calibration[FEATURES])[:, 1]
        sigmoid = LogisticRegression(random_state=440, max_iter=1000)
        sigmoid.fit(logit(calibration_probability), calibration.Y)
        test_probability = frozen_model.predict_proba(test[FEATURES])[:, 1]
        calibrated = prediction_frame(test, sigmoid.predict_proba(logit(test_probability))[:, 1])
        raw_frames.append(raw)
        calibrated_frames.append(calibrated)
        rows.extend([metrics(raw, "raw", train),
                     metrics(calibrated, "calibrated", calibration_train, year - 1)])
        print(f"{year}: {len(test):,} held-out rows; raw trained through {year-1}; "
              f"calibrated model trained through {year-2}, calibrated on {year-1}", flush=True)
    expected = data.loc[data.year.isin(years)]
    export_predictions(pd.concat(raw_frames, ignore_index=True),
                       paths["xgboost_oof_predictions.csv"], expected)
    export_predictions(pd.concat(calibrated_frames, ignore_index=True),
                       paths["xgboost_calibrated_oof_predictions.csv"], expected)
    pd.DataFrame(rows).to_csv(paths["xgboost_fold_metrics.csv"], index=False)
    manifest = dict(input_path=str(args.input.resolve()),
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    features=FEATURES, parameters=PARAMETERS, test_years=years,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, sklearn=sklearn.__version__,
                                  xgboost=xgboost.__version__),
                    calibration="LogisticRegression on clipped probability logits; preceding year; frozen estimator",
                    note="No tuning or ensemble selection. Prior project exposure to 2026 is unverified.",
                    output_rows=len(expected))
    paths["xgboost_run_manifest.json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated outputs saved to {output}")


if __name__ == "__main__":
    main()
