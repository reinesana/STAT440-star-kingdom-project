"""Run four expanding-window cost baselines; no outer-year early stopping.

Run from any directory: python lightgbm_cost_model/train_baseline.py
Log model uses log1p/expm1 without a fitted retransformation correction.
"""
from pathlib import Path
import hashlib
import json
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'github-main/data/train_predict_cost.csv'
OUT = Path(__file__).resolve().parent / 'results'
FEATURES = ['material', 'surface', 'pipe_length', 'years_used_at_leak',
            'event_year', 'midpoint_x', 'midpoint_y']
CATEGORIES = ['material', 'surface']
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15,
              max_depth=-1, min_child_samples=50, reg_lambda=1.0,
              colsample_bytree=1.0, subsample=1.0, random_state=440,
              verbosity=-1, n_jobs=4, deterministic=True, force_col_wise=True)


def matrices(train, valid):
    x, v = train[FEATURES].copy(), valid[FEATURES].copy()
    for col in CATEGORIES:
        # Derive levels only from training; unseen categories become missing.
        levels = sorted(x[col].unique())
        x[col] = pd.Categorical(x[col], categories=levels)
        v[col] = pd.Categorical(v[col], categories=levels)
    return x, v


def dollars(pred, target):
    return np.maximum(np.expm1(pred) if target == 'log1p' else pred, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SOURCE)
    years = pd.to_datetime(d.date).dt.year
    assert (years == d.event_year).all()
    assert not d[FEATURES + ['cost']].isna().any().any()
    assert (d.cost >= 0).all() and d.pipe_id.is_unique
    assert set(years) == set(range(2019, 2027))
    scores, predictions, gains = [], [], []
    for test_year in range(2023, 2027):
        train, test = d[years < test_year], d[years == test_year]
        inner_train = train[train.event_year < test_year - 1]
        inner_valid = train[train.event_year == test_year - 1]
        assert inner_train.event_year.max() < inner_valid.event_year.min()
        assert train.event_year.max() < test.event_year.min()
        xi, xv = matrices(inner_train, inner_valid)
        xt, xe = matrices(train, test)
        for target in ['raw', 'log1p']:
            transform = np.log1p if target == 'log1p' else np.asarray
            def dollar_rmse(y_true, pred):
                actual = np.expm1(y_true) if target == 'log1p' else y_true
                return 'dollar_rmse', float(np.sqrt(mean_squared_error(actual, dollars(pred, target)))), False
            tuning = lgb.LGBMRegressor(**PARAMS, n_estimators=3000, metric='None')
            tuning.fit(xi, transform(inner_train.cost),
                       eval_set=[(xv, transform(inner_valid.cost))],
                       eval_metric=dollar_rmse,
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            iterations = tuning.best_iteration_
            model = lgb.LGBMRegressor(**PARAMS, n_estimators=iterations)
            model.fit(xt, transform(train.cost))
            pred = dollars(model.predict(xe), target)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            row = dict(test_year=test_year, target=target, train_n=len(train),
                       test_n=len(test), inner_valid_year=test_year-1,
                       best_iteration=iterations,
                       mae=mean_absolute_error(test.cost, pred),
                       rmse=float(np.sqrt(mean_squared_error(test.cost, pred))),
                       mean_baseline_rmse=float(np.sqrt(mean_squared_error(test.cost, np.full(len(test), train.cost.mean())))))
            scores.append(row)
            predictions.append(pd.DataFrame(dict(pipe_id=test.pipe_id, test_year=test_year,
                                                  target=target, actual=test.cost, prediction=pred)))
            for feature, gain in zip(FEATURES, model.booster_.feature_importance('gain')):
                gains.append(dict(test_year=test_year, target=target, feature=feature, gain=float(gain)))
            model.booster_.save_model(str(OUT / f'model_{target}_{test_year}.txt'))
            print(json.dumps(row), flush=True)
    s = pd.DataFrame(scores)
    p = pd.concat(predictions, ignore_index=True)
    assert len(p) == 2 * int((years >= 2023).sum())
    assert not p.duplicated(['pipe_id', 'target']).any()
    s.to_csv(OUT / 'fold_metrics.csv', index=False)
    p.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(gains).to_csv(OUT / 'gain_importance.csv', index=False)
    summary = s.groupby('target')[['mae', 'rmse', 'mean_baseline_rmse']].mean()
    summary.to_csv(OUT / 'average_metrics.csv')
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    features=FEATURES, params=PARAMS, max_iterations=3000, patience=100,
                    early_stopping_metric='original-dollar RMSE on last training year',
                    aggregation='unweighted mean of four annual metrics',
                    log_transform='log1p/expm1; no smearing correction',
                    clipping='both predictions clipped at zero',
                    python=platform.python_version(), lightgbm=lgb.__version__,
                    pandas=pd.__version__, numpy=np.__version__)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string())


if __name__ == '__main__':
    main()
