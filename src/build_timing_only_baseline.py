"""One circle per original iron pipe, highest 2027 leak probability first.

Run from repository root: python src/build_timing_only_baseline.py
Requires numpy and pandas. No future leaks are assumed for the budget check.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
class cKDTree:
    def __init__(self, points):
        self.points = points
        self.cells = {}
        for i, p in enumerate(points):
            self.cells.setdefault(tuple(np.floor(p / 100).astype(int)), []).append(i)
    def query_ball_point(self, center, radius):
        lo = np.floor((center-radius)/100).astype(int)
        hi = np.floor((center+radius)/100).astype(int)
        found = []
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):
                found.extend(self.cells.get((x,y), []))
        ids = np.array(found, dtype=int)
        return ids[np.linalg.norm(self.points[ids]-center,axis=1)<=radius].tolist()


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submissions'
RATES = {'grassland': 3, 'farmland': 7, 'swamp': 12,
         'road': 17, 'structure': 50, 'water': 32}
IRON = {'cast iron', 'wrought iron', 'gray iron'}
MARGIN_M = 0.000001


def main():
    pipes = pd.read_csv(ROOT / 'data/pipes.csv', dtype={'Pipe ID': str})
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    assert pipes['Pipe ID'].is_unique
    assert set(history['Pipe ID']) <= set(pipes['Pipe ID'])
    assert set(pipes.Surface) <= set(RATES)
    assert set(pipes.Material.dropna()) <= IRON | {'copper', 'brass', 'polyurethane'}
    dates = pd.to_datetime(history.Date, errors='raise')
    assert dates.between('2019-01-01', '2026-12-31').all()
    a = pipes[['GPS x1', 'GPS y1']].to_numpy(float)
    b = pipes[['GPS x2', 'GPS y2']].to_numpy(float)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    mid = (a + b) / 2
    lengths = np.linalg.norm(b - a, axis=1)
    target = pipes.Material.isin(IRON).to_numpy()
    unknown = pipes.Material.isna().to_numpy()
    pipes['lay_date_parsed'] = pd.to_datetime(pipes['Lay date'], errors='raise')
    forecast_path = ROOT / 'notebooks/leak_model_comparison/deployment_2027_zero_leak_applied/full_inventory_2027_probabilities.csv'
    forecast = pd.read_csv(forecast_path, dtype={'pipe_id': str})
    assert forecast.pipe_id.is_unique and set(forecast.pipe_id) == set(pipes['Pipe ID'])
    assert forecast.year.eq(2027).all()
    assert forecast.leak_probability.dropna().between(0,1).all()
    pipes['probability'] = pipes['Pipe ID'].map(forecast.set_index('pipe_id').leak_probability)
    indices = pipes.loc[target].sort_values(['probability', 'Pipe ID'], ascending=[False, True], na_position='last').index.to_numpy()
    ranked = pipes.loc[indices, ['Pipe ID', 'Material', 'probability']].copy()
    ranked.insert(0, 'submission_row', np.arange(1,len(ranked)+1))
    ranked['probability_status'] = ranked['Pipe ID'].map(forecast.set_index('pipe_id').probability_status)
    ranked['center_x_m'] = mid[indices,0]
    ranked['center_y_m'] = mid[indices,1]
    ranked['radius_m'] = lengths[indices]/2+MARGIN_M
    OUT.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT / '2026-10-14_timing_only_baseline.order_audit.csv', index=False)

    out = OUT / '2026-10-14_timing_only_baseline.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in indices:
            writer.writerow([f'({mid[i, 0]:.9f}, {mid[i, 1]:.9f})',
                             f'{lengths[i] / 2 + MARGIN_M:.9f}'])

    # All validation uses the serialized coordinates/radii, not in-memory circles.
    circles = []
    with out.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            assert len(row) == 2 and row[0].startswith('(') and row[0].endswith(')')
            x, y = map(float, row[0][1:-1].split(','))
            r = float(row[1])
            assert all(map(math.isfinite, (x, y, r))) and r >= 0
            circles.append((np.array([x, y]), r))
    tree = cKDTree(mid)
    covered = np.zeros(len(pipes), dtype=bool)
    contained = []
    for center, radius in circles:
        candidates = np.asarray(tree.query_ball_point(center, radius), dtype=int)
        inside = candidates[(np.linalg.norm(a[candidates] - center, axis=1) <= radius)
                            & (np.linalg.norm(b[candidates] - center, axis=1) <= radius)]
        covered[inside] = True
        contained.append(inside)
    assert covered[target].all(), 'Exported CSV has uncovered iron targets'

    current = (pipes.Material.eq('polyurethane')
               | pipes['Pipe ID'].isin(history['Pipe ID'])).to_numpy(copy=True)
    total_cents = 0
    budget_cents = 0
    next_unit = 0
    skipped = 0
    annual = []
    for year in range(2027, 2052):
        budget_cents += 1_000_000_000
        year_cents = 0
        executed = 0
        while next_unit < len(circles):
            inside = contained[next_unit]
            replace = inside[~current[inside]]
            if not len(replace):
                skipped += 1
                next_unit += 1
                continue
            rate = sum(RATES[s] for s in set(pipes.Surface.iloc[inside]))
            radius = circles[next_unit][1]
            raw = 100 + rate * (math.pi * radius ** 2) ** 0.85 + 200 * lengths[replace].sum()
            cents = int(Decimal(str(raw)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) * 100)
            if cents > budget_cents:
                break
            current[replace] = True
            budget_cents -= cents
            year_cents += cents
            total_cents += cents
            executed += 1
            next_unit += 1
        annual.append({'year': year, 'executed_units': executed,
                       'construction_cost': year_cents / 100,
                       'budget_balance': budget_cents / 100,
                       'remaining_units': len(circles) - next_unit})
        if next_unit == len(circles):
            break
    finished = next_unit == len(circles)
    report = {
        'submission': out.name, 'ordering': '2027 leak_probability descending; Pipe ID ascending ties; unknown probability last. Timing proxy, not predicted date.',
        'radius_margin_m': MARGIN_M, 'coordinate_decimal_places': 9,
        'pipe_count': len(pipes),
        'iron_unknown_probability_count': int(pipes.loc[target, 'probability'].isna().sum()),
        'iron_zero_probability_count': int(pipes.loc[target, 'probability'].eq(0).sum()),
        'model': 'Equal-weight XGBoost, CatBoost, LightGBM; zero-leak rules applied',
        'github_commit': '58fd8a5b4b3f8b5365c525b0156163ece60a8100', 'original_iron_count': int(target.sum()),
        'unknown_material_count': int(unknown.sum()),
        'unknown_policy': 'No circles generated for unknown materials. If contained incidentally, charge replacement unless historically replaced.',
        'iron_missing_lay_date_count': int(pipes.loc[target, 'lay_date_parsed'].isna().sum()),
        'work_unit_count': len(circles), 'uncovered_iron_count': int((target & ~covered).sum()),
        'uncovered_unknown_count': int((unknown & ~covered).sum()),
        'scenario': 'No future leaks; historical leak pipes start as polyurethane.',
        'cost_model_version': '2026-10-26-area-exponent',
        'rounding': 'Nearest cent, ties half up; budget uses integer cents.',
        'completed_by_2051': finished,
        'completion_year': annual[-1]['year'] if finished else None,
        'construction_cost': total_cents / 100,
        'no_future_leak_score': (25_000_000_000 - total_cents) / 100 if finished else None,
        'remaining_original_iron_count': int((target & ~current).sum()),
        'skipped_units': skipped,
        'annual_execution': annual,
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT / 'data/pipes.csv', ROOT / 'data/train.csv', forecast_path, out]},
    }
    report_path = out.with_suffix('.validation.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'annual_execution', 'sha256'}}, indent=2))
    if not finished:
        raise SystemExit('Baseline exceeds no-future-leak budget: do not submit without further review.')


if __name__ == '__main__':
    main()
