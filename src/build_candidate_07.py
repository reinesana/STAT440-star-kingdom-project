"""Candidates 07a/07b/09a/09b: candidate 06 ordering with leak-ensemble probabilities.

Run from repository root: python src/build_candidate_07.py
Keeps the baseline circles, the shrinkage repair-cost ensemble and the
priority = benefit / initial construction cost rule from candidate 06.
Varies (a) the 2027 leak probabilities and (b) gross vs net benefit, where
net benefit subtracts the $200/m replacement charge avoided when a pipe leaks first.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree
from build_baseline import RATES, IRON, MARGIN_M
from build_cost_submissions import containment, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submissions/2026-10-18_candidate_07_leak_ensemble'
COST = ROOT / 'notebooks/ensemble_for_cost_prediction/predictions/ensemble_stable.csv'
LEAK = ROOT / 'notebooks/leak_model_comparison/deployment_2027_zero_leak_applied/full_inventory_2027_probabilities.csv'
MODELS = ROOT / 'submissions/risk_priority_models'
CANDIDATE_06 = ROOT / 'submissions/2026-10-13_cost_candidates/06_stable_risk_per_work_cost.csv'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_model_probability(pipes, lengths, mid, years):
    meta = json.loads((MODELS/'metadata.json').read_text())
    model = lgb.Booster(model_file=str(MODELS/'risk.txt'))
    frame = pd.DataFrame(dict(material=pipes.Material, surface=pipes.Surface, pipe_length=lengths,
                              midpoint_x=mid[:,0], midpoint_y=mid[:,1]))
    for col, categories in meta['categories'].items():
        frame[col] = pd.Categorical(frame[col], categories=categories)
    survival = np.ones(len(pipes))
    for year in years:
        frame['pipe_age'] = (pd.Timestamp(year,1,1)-pd.to_datetime(pipes['Lay date'])).dt.days/365.25
        survival *= 1-model.predict(frame[meta['features']], num_threads=4)
    return 1-survival


def order_units(priority, pipe_ids):
    return np.lexsort((pipe_ids, -priority))


def write_units(path, units, order):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        for j in order:
            center, radius = units[j]
            writer.writerow([f'({center[0]:.9f}, {center[1]:.9f})', f'{radius:.9f}'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipes = pd.read_csv(ROOT/'data/pipes.csv', dtype={'Pipe ID': str})
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Pipe ID': str})
    a = pipes[['GPS x1','GPS y1']].to_numpy(float)
    b = pipes[['GPS x2','GPS y2']].to_numpy(float)
    mid = (a+b)/2
    lengths = np.linalg.norm(b-a, axis=1)
    target = pipes.Material.isin(IRON).to_numpy()
    initial = (pipes.Material.eq('polyurethane') | pipes['Pipe ID'].isin(history['Pipe ID'])).to_numpy()
    indices = np.flatnonzero(target)
    unit_ids = pipes['Pipe ID'].iloc[indices].to_numpy()
    units = [(np.round(mid[i], 9), float(f'{lengths[i]/2+MARGIN_M:.9f}')) for i in indices]
    inside = containment(units, cKDTree(mid), a, b)
    construction = []
    for (_, radius), ids in zip(units, inside):
        active = ids[~initial[ids]]
        rate = sum(RATES[s] for s in set(pipes.Surface.iloc[ids]))
        construction.append(0 if not len(active) else 100+rate*(math.pi*radius**2)**.85+200*lengths[active].sum())
    construction = np.array(construction)

    costs = pd.read_csv(COST, dtype={'id': str})
    assert costs.id.is_unique
    severity = pipes['Pipe ID'].map(costs.set_index('id').predicted_cost).to_numpy(copy=True)
    assert np.isfinite(severity[~initial]).all() and (severity[~initial] >= 0).all()
    severity[initial] = 0

    # Reproduce candidate 06 byte-for-byte before changing anything.
    old_2y = old_model_probability(pipes, lengths, mid, [2027, 2028])
    old_2y[initial] = 0
    priority_06 = np.array([(severity*old_2y)[ids].sum() for ids in inside]) / np.maximum(construction, 1)
    check = OUT/'_06_reproduction.tmp.csv'
    write_units(check, units, order_units(priority_06, unit_ids))
    reproduced = digest(check) == digest(CANDIDATE_06)
    check.unlink()
    # Old candidate 06 used the pre-October-26 formula; hash equality is no longer expected.
    print('Historical candidate 06 hash matches:', reproduced)

    leak = pd.read_csv(LEAK, dtype={'pipe_id': str}).set_index('pipe_id')
    assert leak.index.is_unique and set(leak.index) == set(pipes['Pipe ID']) and leak.year.eq(2027).all()
    # Missing-input pipes stay unknown in the ensemble; use the 06 risk model's 2027 probability for them.
    fallback = old_model_probability(pipes, lengths, mid, [2027])
    sources = {
        'full_ensemble': leak.leak_probability,
        'catboost_lightgbm': leak[['catboost_probability','lightgbm_probability']].mean(axis=1, skipna=False),
    }
    probabilities, fallback_counts = {}, {}
    for name, series in sources.items():
        prob = pipes['Pipe ID'].map(series).to_numpy(copy=True)
        missing = np.isnan(prob)
        fallback_counts[name] = int(missing.sum())
        prob[missing] = fallback[missing]
        prob[initial] = 0
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        probabilities[name] = prob

    replacement = 200*lengths
    configurations = [('07a_full_ensemble_gross', 'full_ensemble', 'gross'),
                      ('07b_catboost_lightgbm_gross', 'catboost_lightgbm', 'gross'),
                      ('09a_full_ensemble_net', 'full_ensemble', 'net'),
                      ('09b_catboost_lightgbm_net', 'catboost_lightgbm', 'net')]
    results = []
    for name, source, benefit_type in configurations:
        prob = probabilities[source]
        per_pipe = prob*(severity - replacement) if benefit_type == 'net' else prob*severity
        per_pipe[initial] = 0
        benefit = np.array([per_pipe[ids].sum() for ids in inside])
        priority = benefit / np.maximum(construction, 1)
        order = order_units(priority, unit_ids)
        path = OUT/f'{name}.csv'
        write_units(path, units, order)
        pd.DataFrame(dict(rank=np.arange(1, len(order)+1), pipe_id=unit_ids[order], priority=priority[order],
                          circle_benefit=benefit[order], initial_circle_work_cost=construction[order],
                          pipe_cost_prediction=severity[indices[order]],
                          pipe_leak_probability_2027=prob[indices[order]])
                     ).to_csv(OUT/f'{name}.audit.csv', index=False)
        report = validate(path, pipes, history, a, b, mid, lengths, target)
        rank_06 = np.empty(len(units)); rank_06[order_units(priority_06, unit_ids)] = np.arange(len(units))
        rank_new = np.empty(len(units)); rank_new[order] = np.arange(len(units))
        results.append(dict(candidate=name, leak_probabilities=source, benefit=benefit_type,
                            construction_cost=report['construction_cost'], completion_year=report['completion_year'],
                            uncovered_iron_count=report['uncovered_iron_count'],
                            negative_benefit_circles=int((benefit < 0).sum()),
                            rank_correlation_with_06=float(np.corrcoef(rank_06, rank_new)[0, 1]),
                            submission_sha256=report['sha256']))
        print(json.dumps(results[-1]), flush=True)
    pd.DataFrame(results).to_csv(OUT/'comparison.csv', index=False)
    manifest = dict(
        reference='06_stable_risk_per_work_cost (scored -15.61M)', candidate_06_reproduced=reproduced,
        circles='Baseline: one per original iron pipe, midpoint centre, half endpoint length + MARGIN_M',
        priority='sum over contained unreplaced pipes of benefit / initial circle construction cost',
        gross_benefit='P(leak 2027) * predicted repair cost',
        net_benefit='P(leak 2027) * (predicted repair cost - 200 * endpoint length)',
        leak_window=[2027], probability_fallback='06 LightGBM risk model 2027 probability for ensemble-missing pipes',
        fallback_counts=fallback_counts,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                       [ROOT/'data/pipes.csv', ROOT/'data/train.csv', COST, LEAK,
                        MODELS/'risk.txt', MODELS/'metadata.json', Path(__file__)]},
        candidates=results,
        limitations=['Single-year 2027 probabilities rank the whole 2027-2051 plan; 06 used 2027-2028.',
                     'Net benefit ignores surface fees saved when every pipe in a circle has already leaked.',
                     'Circle priorities overlap and are not re-ranked as work executes.',
                     'Validation assumes no future leaks; it checks feasibility, not the official score.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
