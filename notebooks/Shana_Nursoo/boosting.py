"""XGBoost and CatBoost cost models on the same folds and targets as linear_regression.py.

Run from the repository root:
    python notebooks/Shana_Nursoo/boosting.py xgboost
    python notebooks/Shana_Nursoo/boosting.py catboost

Per fold, tree depth and number of trees are tuned on the training years only: fit on all
training years except the last, score dollar RMSE on that last year, then refit on every
training year with the chosen setting. The test year is never used for tuning.

Feature sets are compared on mean 2023-2025 RMSE; 2026 is reported for the chosen set.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folds import SEED, TEST_YEARS, load_data, split_folds  # noqa: E402
from linear_regression import (BASE_NUMERIC, MATERIAL_COLS, REQUIRED, SELECTION_YEARS,  # noqa: E402
                               SURFACE_COLS, UNAVAILABLE, metrics)

HERE = Path(__file__).resolve().parent
TARGETS = ('raw', 'log', 'log_smear')
DEPTHS = (3, 6)
LEARNING_RATE = 0.05
MAX_TREES = 1500
CHECK_EVERY = 25
THREADS = 4


def feature_sets(df, added_columns):
    starting = REQUIRED + BASE_NUMERIC + MATERIAL_COLS + SURFACE_COLS
    usable = [c for c in added_columns
              if c not in UNAVAILABLE and pd.api.types.is_numeric_dtype(df[c])]
    history = [c for c in usable if c.startswith('hist_')]
    return {'starting': starting,
            'starting+history': starting + history,
            'starting+all_new': starting + usable}


def fit(model, depth, trees, X, y):
    if model == 'xgboost':
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=LEARNING_RATE,
                         subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                         random_state=SEED, n_jobs=THREADS)
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=trees, depth=depth, learning_rate=LEARNING_RATE,
                              loss_function='RMSE', random_seed=SEED, thread_count=THREADS,
                              verbose=False, allow_writing_files=False)
    return m.fit(X, y)


def predict(model, m, X, trees):
    if model == 'xgboost':
        return m.predict(X, iteration_range=(0, trees))
    return m.predict(X, ntree_end=trees)


def to_dollars(target, pred, smear=1.0):
    with np.errstate(over='ignore', invalid='ignore'):
        if target == 'raw':
            out = pred
        elif target == 'log':
            out = np.expm1(pred)
        else:
            out = np.exp(pred) * smear - 1
    return np.clip(out, 0, None)


def smear_factor(y_log, fitted_log):
    with np.errstate(over='ignore'):
        return float(np.mean(np.exp(y_log - fitted_log)))


def tune(model, train, cols, test_year):
    """Return {target: (depth, trees)} chosen on the last training year."""
    inner_train = train[train['event_year'] < test_year - 1]
    inner_val = train[train['event_year'] == test_year - 1]
    X, X_val = inner_train[cols], inner_val[cols]
    y = inner_train['cost'].to_numpy(float)
    y_val = inner_val['cost'].to_numpy(float)
    best = {t: (np.inf, None) for t in TARGETS}
    checkpoints = range(CHECK_EVERY, MAX_TREES + 1, CHECK_EVERY)
    for depth in DEPTHS:
        m_raw = fit(model, depth, MAX_TREES, X, y)
        m_log = fit(model, depth, MAX_TREES, X, np.log1p(y))
        for k in checkpoints:
            log_val = predict(model, m_log, X_val, k)
            smear = smear_factor(np.log1p(y), predict(model, m_log, X, k))
            preds = {'raw': to_dollars('raw', predict(model, m_raw, X_val, k)),
                     'log': to_dollars('log', log_val),
                     'log_smear': to_dollars('log_smear', log_val, smear)}
            for target, pred in preds.items():
                rmse = metrics(y_val, pred)[1]
                if rmse < best[target][0]:
                    best[target] = (rmse, (depth, k))
    return {t: setting for t, (_, setting) in best.items()}


def run_fold(model, train, test, cols, year):
    settings = tune(model, train, cols, year)
    X, X_test = train[cols], test[cols]
    y = train['cost'].to_numpy(float)
    y_test = test['cost'].to_numpy(float)
    rows, preds, fitted = [], {}, {}
    for target in TARGETS:
        depth, trees = settings[target]
        family = 'raw' if target == 'raw' else 'log'
        key = (family, depth, trees)
        if key not in fitted:
            fitted[key] = fit(model, depth, trees, X, y if family == 'raw' else np.log1p(y))
        m = fitted[key]
        smear = smear_factor(np.log1p(y), m.predict(X)) if target == 'log_smear' else 1.0
        pred = to_dollars(target, m.predict(X_test), smear)
        mae, rmse = metrics(y_test, pred)
        rows.append({'target': target, 'test_year': year, 'MAE': mae, 'RMSE': rmse,
                     'depth': depth, 'trees': trees})
        preds[target] = pd.DataFrame({'pipe_id': test['pipe_id'].to_numpy(),
                                      'predicted_cost': pred})
    return rows, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['xgboost', 'catboost'])
    model = parser.parse_args().model
    started = time.time()
    np.random.seed(SEED)
    out_dir = HERE / 'results' / model
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((HERE.parents[1] / 'data/train_predict_cost_features_2026-10-05.json').read_text())
    df = load_data()
    sets = feature_sets(df, meta['added_columns'])
    folds = {year: (train, test) for year, train, test in split_folds(df, TEST_YEARS)}

    fold_rows, predictions = [], {}
    for name, cols in sets.items():
        for year in TEST_YEARS:
            train, test = folds[year]
            rows, preds = run_fold(model, train, test, cols, year)
            for r in rows:
                fold_rows.append({'model': model, 'feature_set': name, 'n_features': len(cols), **r})
                print(f'{model} {name:17s} {r["target"]:9s} {year}  RMSE {r["RMSE"]:>10,.0f}  '
                      f'MAE {r["MAE"]:>8,.0f}  depth {r["depth"]} trees {r["trees"]}'
                      f'  ({time.time() - started:.0f}s)', flush=True)
            for target, p in preds.items():
                predictions.setdefault((name, target), []).append(p)

    fold = pd.DataFrame(fold_rows)
    fold.to_csv(out_dir / 'fold_metrics.csv', index=False)
    wide = fold.pivot_table(index=['feature_set', 'target'], columns='test_year', values=['MAE', 'RMSE'])
    summary = pd.DataFrame(index=wide.index)
    for metric in ['MAE', 'RMSE']:
        for year in TEST_YEARS:
            summary[f'{metric}_{year}'] = wide[(metric, year)]
        summary[f'mean_{metric}_2023_2025'] = wide[metric][list(SELECTION_YEARS)].mean(axis=1)
        summary[f'mean_{metric}_2023_2026'] = wide[metric][list(TEST_YEARS)].mean(axis=1)
    summary = summary.sort_values('mean_RMSE_2023_2025').reset_index()
    summary.insert(0, 'model', model)
    summary.to_csv(out_dir / 'average_metrics.csv', index=False)

    chosen = {}
    for target in TARGETS:
        name = summary[summary.target == target].iloc[0].feature_set
        chosen[target] = name
        pd.concat(predictions[(name, target)], ignore_index=True).to_csv(
            HERE / f'output_{model}_{target}.csv', index=False)

    import sklearn
    lib = __import__(model)
    (out_dir / 'selected.json').write_text(json.dumps({
        'selection_rule': 'feature set with lowest mean RMSE over 2023-2025, per target',
        'feature_set_per_target': chosen,
        'feature_sets': {k: v for k, v in sets.items()},
        'tuning': {'depths': DEPTHS, 'learning_rate': LEARNING_RATE, 'max_trees': MAX_TREES,
                   'check_every': CHECK_EVERY, 'threads': THREADS, 'seed': SEED},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                     model: lib.__version__}}, indent=2))

    pd.set_option('display.width', 220)
    cols = ['feature_set', 'target'] + [f'RMSE_{y}' for y in TEST_YEARS] + \
           ['mean_RMSE_2023_2025', 'mean_RMSE_2023_2026', 'mean_MAE_2023_2026']
    print(summary[cols].round(0).to_string(index=False))
    print(f'Done in {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
