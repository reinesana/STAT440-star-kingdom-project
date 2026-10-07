"""Step 2: linear regression cost model (Shana Nursoo, with Anthony).

Run from the repository root:
    python notebooks/Shana_Nursoo/linear_regression.py

Three target versions are kept separate throughout:
    raw        OLS on cost in dollars
    log        OLS on log1p(cost), predictions converted back with np.expm1
    log_smear  same fit as `log`, with Duan's smearing factor from the training residuals

Feature selection uses only the 2023-2025 folds (mean RMSE). The selected setup is then
frozen and fitted once on 2019-2025 to score 2026.
"""
from pathlib import Path
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folds import SEED, TEST_YEARS, load_data, split_folds  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results' / 'linear_regression'
SELECTION_YEARS = (2023, 2024, 2025)
FINAL_YEAR = 2026
TARGETS = ('raw', 'log', 'log_smear')
MIN_GAIN = 50.0
RIDGE_ALPHAS = np.logspace(-2, 4, 13)

MATERIALS = ['brass', 'cast_iron', 'copper', 'gray_iron']
SURFACES = ['farmland', 'grassland', 'road', 'structure', 'swamp']
MATERIAL_COLS = [f'material_{m}' for m in MATERIALS]
SURFACE_COLS = [f'surface_{s}' for s in SURFACES]
BASE_NUMERIC = ['pipe_length', 'years_used_at_leak', 'midpoint_x', 'midpoint_y',
                'season_sin', 'season_cos']
REQUIRED = ['event_year']

# Leak time of day and weekday are not known for future scenario leaks; IDs are not predictors.
UNAVAILABLE = {'event_hour', 'event_night', 'hour_sin', 'hour_cos', 'event_weekday',
               'event_weekend', 'weekday_sin', 'weekday_cos', 'route_id',
               'straight_section_id', 'material_surface'}


def add_engineered(df):
    df = df.copy()
    for m in MATERIAL_COLS + SURFACE_COLS:
        df[f'{m}_x_log_length'] = df[m] * df['log1p_pipe_length']
    pairs = pd.get_dummies(df['material_surface'], prefix='pair', dtype=float)
    pairs.columns = [re.sub(r'\W+', '_', c) for c in pairs.columns]
    reference = pairs.sum().idxmax()
    return pd.concat([df, pairs.drop(columns=reference)], axis=1)


def block_of(col):
    if m := re.fullmatch(r'(hist_.+?_(?:all|last1|last2))_.+', col):
        return m.group(1) + ('_log1p' if col.endswith('_log1p') else '')
    if m := re.fullmatch(r'(hist_(?:radius|nearest)\d+)_.+', col):
        return m.group(1) + ('_log1p' if col.endswith('_log1p') else '')
    if m := re.fullmatch(r'(inventory_\d+m)_.+', col):
        return m.group(1)
    if col.startswith('inventory_'):
        return 'inventory_geometry'
    if col.startswith('route_'):
        return 'route'
    if col.startswith('straight_section_'):
        return 'straight_section'
    if col in {'is_bend', 'is_straight', 'is_branch_adjacent', 'has_continuation',
               'turn_angle', 'endpoint_degree_1', 'endpoint_degree_2', 'topology_eligible'}:
        return 'topology'
    return re.sub(r'_(\d+m?|\d+_\d+)$', '', col)


def candidate_blocks(df, added_columns):
    blocks = {
        'log_length': ['log1p_pipe_length'],
        'log_age': ['log1p_years_used'],
        'material_x_log_length': [f'{m}_x_log_length' for m in MATERIAL_COLS],
        'surface_x_log_length': [f'{s}_x_log_length' for s in SURFACE_COLS],
        'material_surface_pairs': [c for c in df.columns if c.startswith('pair_')],
    }
    for col in added_columns:
        if col in UNAVAILABLE or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        blocks.setdefault(block_of(col), []).append(col)
    return blocks


def removable_groups():
    groups = {c: [c] for c in BASE_NUMERIC}
    groups['material'] = MATERIAL_COLS
    groups['surface'] = SURFACE_COLS
    return groups


def fit_predict(train, test, cols, target, alpha=None):
    model = LinearRegression() if alpha is None else Ridge(alpha=alpha, random_state=SEED)
    pipe = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         StandardScaler(), model)
    X, X_test = train[cols].to_numpy(float), test[cols].to_numpy(float)
    y = train['cost'].to_numpy(float)
    with np.errstate(over='ignore', invalid='ignore'):
        if target == 'raw':
            pred = pipe.fit(X, y).predict(X_test)
        else:
            y_log = np.log1p(y)
            log_pred = pipe.fit(X, y_log).predict(X_test)
            if target == 'log':
                pred = np.expm1(log_pred)
            else:
                smear = np.mean(np.exp(y_log - pipe.predict(X)))
                pred = np.exp(log_pred) * smear - 1
    return np.clip(pred, 0, None)


def metrics(y, pred):
    err = y - pred
    with np.errstate(over='ignore', invalid='ignore'):
        rmse = float(np.sqrt(np.mean(err ** 2)))
    return float(np.mean(np.abs(err))), rmse if np.isfinite(rmse) else np.inf


def inner_alpha(train, cols, target, test_year):
    inner_train = train[train['event_year'] < test_year - 1]
    inner_val = train[train['event_year'] == test_year - 1]
    y = inner_val['cost'].to_numpy(float)
    scores = [metrics(y, fit_predict(inner_train, inner_val, cols, target, a))[1]
              for a in RIDGE_ALPHAS]
    return float(RIDGE_ALPHAS[int(np.argmin(scores))])


class Evaluator:
    def __init__(self, df):
        self.folds = {year: (train, test) for year, train, test in split_folds(df, TEST_YEARS)}

    def run(self, cols, target, years, ridge=False, keep_predictions=False):
        out = {'years': {}, 'predictions': []}
        for year in years:
            train, test = self.folds[year]
            alpha = inner_alpha(train, cols, target, year) if ridge else None
            pred = fit_predict(train, test, cols, target, alpha)
            mae, rmse = metrics(test['cost'].to_numpy(float), pred)
            out['years'][year] = {'MAE': mae, 'RMSE': rmse, 'alpha': alpha}
            if keep_predictions:
                out['predictions'].append(pd.DataFrame({
                    'pipe_id': test['pipe_id'].to_numpy(), 'test_year': year,
                    'cost': test['cost'].to_numpy(float), 'predicted_cost': pred}))
        sel = [out['years'][y] for y in years if y in SELECTION_YEARS]
        out['mean_RMSE'] = float(np.mean([s['RMSE'] for s in sel])) if sel else np.nan
        out['mean_MAE'] = float(np.mean([s['MAE'] for s in sel])) if sel else np.nan
        return out


def log_row(log, target, step, action, block, cols, result, accepted):
    row = {'target': target, 'step': step, 'action': action, 'block': block,
           'n_features': len(cols), 'accepted': accepted,
           'mean_RMSE_2023_2025': result['mean_RMSE'], 'mean_MAE_2023_2025': result['mean_MAE']}
    for year, m in result['years'].items():
        row[f'RMSE_{year}'], row[f'MAE_{year}'] = m['RMSE'], m['MAE']
    log.append(row)


def select_features(ev, target, blocks, log):
    """Greedy forward block selection, then backward removal, on mean 2023-2025 RMSE."""
    groups = removable_groups()
    active = dict(groups)
    blocks = {**groups, **blocks}
    flatten = lambda chosen: REQUIRED + [c for cols in chosen.values() for c in cols]
    best = ev.run(flatten(active), target, SELECTION_YEARS)
    log_row(log, target, 0, 'baseline', '', flatten(active), best, True)
    step = 0
    improved = True
    while improved:
        improved = False
        step += 1
        trials = []
        for name, cols in blocks.items():
            if name in active:
                continue
            trial = dict(active, **{name: cols})
            res = ev.run(flatten(trial), target, SELECTION_YEARS)
            trials.append((res['mean_RMSE'], name, 'add', trial, res))
        for name in list(active):
            trial = {k: v for k, v in active.items() if k != name}
            res = ev.run(flatten(trial), target, SELECTION_YEARS)
            trials.append((res['mean_RMSE'], name, 'remove', trial, res))
        trials.sort(key=lambda t: t[0])
        winner = trials[0]
        gain = best['mean_RMSE'] - winner[0]
        for score, name, action, trial, res in trials:
            log_row(log, target, step, action, name, flatten(trial), res,
                    accepted=(name == winner[1] and action == winner[2] and gain > MIN_GAIN))
        if gain > MIN_GAIN:
            active, best, improved = winner[3], winner[4], True
            print(f'  [{target}] step {step}: {winner[2]} {winner[1]} '
                  f'-> mean RMSE {best["mean_RMSE"]:,.0f}', flush=True)
    return active, flatten(active)


def main():
    started = time.time()
    np.random.seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE.parents[1] / 'data/train_predict_cost_features_2026-10-05.json').read_text())
    df = add_engineered(load_data())
    blocks = candidate_blocks(df, meta['added_columns'])
    ev = Evaluator(df)
    print(f'{len(df):,} rows, {len(blocks)} candidate feature blocks', flush=True)

    reference = []
    for year in TEST_YEARS:
        train, test = ev.folds[year]
        y = test['cost'].to_numpy(float)
        mae, rmse = metrics(y, np.full(len(y), train['cost'].mean()))
        reference.append({'model': 'reference: training-mean constant', 'target': 'raw',
                          'test_year': year, 'MAE': mae, 'RMSE': rmse})

    base_cols = REQUIRED + BASE_NUMERIC + MATERIAL_COLS + SURFACE_COLS
    log, fold_rows, predictions, selected = [], list(reference), [], {}
    for target in TARGETS:
        print(f'Selecting features for target={target}', flush=True)
        groups, cols = select_features(ev, target, blocks, log)
        selected[target] = {'blocks': sorted(groups), 'features': cols}
        variants = [('linear regression (baseline features)', base_cols, False),
                    ('linear regression (selected features)', cols, False),
                    ('ridge (selected features, inner-validated alpha)', cols, True)]
        for model, feature_cols, ridge in variants:
            res = ev.run(feature_cols, target, TEST_YEARS, ridge=ridge, keep_predictions=True)
            for year, m in res['years'].items():
                fold_rows.append({'model': model, 'target': target, 'test_year': year,
                                  'MAE': m['MAE'], 'RMSE': m['RMSE'], 'ridge_alpha': m['alpha'],
                                  'n_features': len(feature_cols)})
            for p in res['predictions']:
                predictions.append(p.assign(model=model, target=target))

    fold = pd.DataFrame(fold_rows)
    fold.to_csv(RESULTS / 'fold_metrics.csv', index=False)
    pd.DataFrame(log).to_csv(RESULTS / 'experiments.csv', index=False)
    preds = pd.concat(predictions, ignore_index=True)

    wide = fold.pivot_table(index=['model', 'target'], columns='test_year', values=['MAE', 'RMSE'])
    summary = pd.DataFrame(index=wide.index)
    for metric in ['MAE', 'RMSE']:
        for year in TEST_YEARS:
            summary[f'{metric}_{year}'] = wide[(metric, year)]
        summary[f'mean_{metric}_2023_2025'] = wide[metric][list(SELECTION_YEARS)].mean(axis=1)
        summary[f'mean_{metric}_2023_2026'] = wide[metric][list(TEST_YEARS)].mean(axis=1)
    summary = summary.sort_values('mean_RMSE_2023_2025').reset_index()
    summary.to_csv(RESULTS / 'average_metrics.csv', index=False)

    candidates = summary[~summary['model'].str.startswith('reference')]
    best = candidates.iloc[0]
    for target in TARGETS:
        model = candidates[candidates.target == target].iloc[0].model
        selected[target]['best_model_for_target'] = model
        chosen = preds[(preds.model == model) & (preds.target == target)]
        chosen[['pipe_id', 'predicted_cost']].to_csv(
            HERE / f'output_linear_regression_{target}.csv', index=False)

    (RESULTS / 'selected_features.json').write_text(json.dumps({
        'chosen_for_output': {'model': best.model, 'target': best.target},
        'selection_rule': 'lowest mean RMSE over 2023-2025; 2026 scored once after freezing',
        'min_gain_dollars': MIN_GAIN, 'ridge_alphas': RIDGE_ALPHAS.tolist(),
        'excluded_columns': sorted(UNAVAILABLE), 'per_target': selected}, indent=2))
    (RESULTS / 'versions.json').write_text(json.dumps({
        'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__, 'seed': SEED}, indent=2))

    pd.set_option('display.width', 200)
    cols = ['model', 'target'] + [f'RMSE_{y}' for y in TEST_YEARS] + \
           ['mean_RMSE_2023_2025', 'mean_RMSE_2023_2026', 'mean_MAE_2023_2026']
    print(summary[cols].round(0).to_string(index=False))
    print(f'Output: {best.model} / {best.target}  ({time.time() - started:.0f}s)')


if __name__ == '__main__':
    main()
