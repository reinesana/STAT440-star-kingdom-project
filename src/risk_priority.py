"""Train chronological candidates, refit all history, and conservatively reorder circles."""
from pathlib import Path
import heapq
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, mean_squared_error, average_precision_score
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submissions/risk_priority_models'
FEATURES = ['material', 'surface', 'pipe_length', 'pipe_age', 'midpoint_x', 'midpoint_y']


def reorder(pipes, history, indices, a, b, mid, lengths):
    OUT.mkdir(parents=True, exist_ok=True)
    events = history.rename(columns={'Pipe ID': 'pipe_id', 'Date': 'date', 'Cost': 'cost'}).copy()
    events['date'] = pd.to_datetime(events.date)
    assert events.pipe_id.is_unique
    base = pd.DataFrame({'pipe_id': pipes['Pipe ID'], 'material': pipes.Material,
                         'surface': pipes.Surface, 'pipe_length': lengths,
                         'midpoint_x': mid[:, 0], 'midpoint_y': mid[:, 1]})
    lay = pd.to_datetime(pipes['Lay date'])
    categories = {}
    for col in ['material', 'surface']:
        categories[col] = sorted(base[col].dropna().unique().tolist())
        base[col] = pd.Categorical(base[col], categories=categories[col])
    leak_dates = base.pipe_id.map(events.set_index('pipe_id').date)
    rows = []
    for year in range(2019, 2027):
        frame = base.copy()
        frame['year'] = year
        frame['pipe_age'] = (pd.Timestamp(year, 1, 1) - lay).dt.days / 365.25
        frame['Y'] = (leak_dates.dt.year == year).astype(int)
        rows.append(frame[leak_dates.isna() | (leak_dates.dt.year >= year)])
    risk_data = pd.concat(rows, ignore_index=True)
    cost_data = base.merge(events[['pipe_id', 'date', 'cost']], on='pipe_id')
    cost_data['pipe_age'] = (cost_data.date - cost_data.pipe_id.map(pd.Series(lay.to_numpy(), index=base.pipe_id))).dt.days / 365.25
    cost_data['year'] = cost_data.date.dt.year
    # Missing age/material retained; LightGBM handles missing features.
    settings = [dict(n_estimators=80, num_leaves=7, min_child_samples=150),
                dict(n_estimators=160, num_leaves=15, min_child_samples=100)]
    validation = []
    chosen = {}
    for role, data, target in [('risk', risk_data, 'Y'), ('cost', cost_data, 'cost')]:
        scores = []
        for candidate, config in enumerate(settings):
            fold_scores = []
            for year in [2025, 2026]:
                train, test = data[data.year < year], data[data.year == year]
                cls = lgb.LGBMClassifier if role == 'risk' else lgb.LGBMRegressor
                model = cls(**config, learning_rate=.04, random_state=440, n_jobs=4, verbosity=-1)
                model.fit(train[FEATURES], train[target])
                pred = model.predict_proba(test[FEATURES])[:, 1] if role == 'risk' else np.maximum(model.predict(test[FEATURES]), 0)
                score = log_loss(test[target], pred) if role == 'risk' else float(np.sqrt(mean_squared_error(test[target], pred)))
                record = dict(role=role, candidate=candidate, year=year, metric='log_loss' if role == 'risk' else 'RMSE', value=score)
                if role == 'risk':
                    record.update(average_precision=average_precision_score(test[target], pred), observed_rate=float(test[target].mean()), predicted_rate=float(pred.mean()))
                validation.append(record)
                fold_scores.append(score)
            scores.append(float(np.mean(fold_scores)))
        best = int(np.argmin(scores))
        cls = lgb.LGBMClassifier if role == 'risk' else lgb.LGBMRegressor
        model = cls(**settings[best], learning_rate=.04, random_state=440, n_jobs=4, verbosity=-1)
        model.fit(data[FEATURES], data[target])
        model.booster_.save_model(str(OUT / f'{role}.txt'))
        chosen[role] = dict(candidate=best, parameters=settings[best], training_rows=len(data))
        chosen[role]['model'] = model
    survival = np.ones(len(base))
    expected = np.zeros(len(base))
    future = base.copy()
    for year in [2027, 2028]:
        future['pipe_age'] = (pd.Timestamp(year, 1, 1) - lay).dt.days / 365.25
        probability = chosen['risk']['model'].predict_proba(future[FEATURES])[:, 1]
        severity = np.maximum(chosen['cost']['model'].predict(future[FEATURES]), 0)
        expected += survival * probability * severity
        survival *= 1 - probability
    replaced = pipes['Pipe ID'].isin(history['Pipe ID']) | pipes.Material.eq('polyurethane')
    expected[replaced] = 0
    tree = cKDTree(mid)
    circle_risk = np.zeros(len(indices))
    for rank, i in enumerate(indices):
        radius = lengths[i] / 2 + .000001
        ids = np.asarray(tree.query_ball_point(mid[i], radius), dtype=int)
        inside = ids[(np.linalg.norm(a[ids]-mid[i], axis=1) <= radius) & (np.linalg.norm(b[ids]-mid[i], axis=1) <= radius)]
        circle_risk[rank] = expected[inside].sum()
    # High-risk fifth may advance by at most 3,000 work units (about one baseline year).
    # All others retain relative oldest-first order through their original release positions.
    threshold = float(np.quantile(circle_risk[circle_risk > 0], .8))
    high = circle_risk >= threshold
    original = np.arange(len(indices))
    release = np.where(high, np.maximum(0, original-3000), original)
    arrivals = np.argsort(release, kind='stable')
    heap, order, cursor = [], [], 0
    for position in original:
        while cursor < len(indices) and release[arrivals[cursor]] <= position:
            rank = int(arrivals[cursor]); cursor += 1
            heapq.heappush(heap, (int(release[rank]), -float(circle_risk[rank]) if high[rank] else 0., rank))
        order.append(heapq.heappop(heap)[2])
    order = np.asarray(order)
    new_rank = np.empty(len(order), dtype=int); new_rank[order] = original
    assert len(np.unique(order)) == len(indices)
    assert np.max(original-new_rank) <= 3000
    audit = pd.DataFrame({'pipe_id': pipes['Pipe ID'].iloc[indices].to_numpy(), 'original_rank': original+1,
                          'new_rank': new_rank+1, 'circle_expected_leak_cost_2027_2028': circle_risk,
                          'priority_eligible': high, 'positions_advanced': original-new_rank,
                          'pipe_leak_probability_2y': (1-survival)[indices], 'pipe_expected_leak_cost_2y': expected[indices]})
    audit.to_csv(OUT.parent / '2026-10-05_risk_priority.audit.csv', index=False)
    metadata = dict(features=FEATURES, categories=categories, history_cutoff='2026-12-31',
                    prediction_window=[2027, 2028], maximum_advance=3000, priority_fraction=.2,
                    priority_threshold=threshold, advanced_units=int((original>new_rank).sum()),
                    validation=validation, models={k: {a:b for a,b in v.items() if a!='model'} for k,v in chosen.items()},
                    limitations=['Fixed 2027-2028 priority window; no annual adaptive reorder.',
                                 'Validation used for model selection, not untouched final test.',
                                 'Priority heuristic; no proven future score improvement.',
                                 'Circle risk may overlap; scoring does not claim independent avoided savings.'])
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return indices[order]
