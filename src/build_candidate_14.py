"""Candidate 14: Rion's overlap-updating dynamic order + drifting-wave yearly risk.

Circles: 10a grouped circles plus the 94 high-value brass circles (candidate 12).
Order: search_orders.order_dynamic with net (repair minus $200/m) drifting-wave value.
Each CSV is validated with the team validator (build_cost_submissions.validate).

Run from repository root: python src/build_candidate_14.py
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so
from search_dynamic import CIRCLES, team_value
from build_baseline import IRON
from build_cost_submissions import validate

warnings.filterwarnings('ignore')
OUT = sim.ROOT / 'submissions/2026-10-21_candidate_14_dynamic_drift'


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    h = sim.drift_hazard(pipes, sim.fit_drift_wave(pipes))
    yc = so.yearly_cost(h, sim.cost_pools(pipes))
    survive = np.cumprod(1 - h, axis=1)
    p_first = h * np.hstack([np.ones((len(h), 1)), survive[:, :-1]])
    yc_net = np.clip(yc - p_first * 200 * lengths[:, None], 0, None)
    variants = {
        '14a_dynamic_drift_net_decay030': dict(value_by_year=yc_net, decay=0.30),
        '14b_dynamic_drift_net_team25': dict(value_by_year=yc_net, static_value=team_value(pipes, lengths), blend=0.25),
        '14c_dynamic_drift_net_skip1': dict(value_by_year=yc_net, skip=1),
    }
    lines = CIRCLES.read_text(encoding='utf-8').splitlines()
    xy, r = sim.read_units(CIRCLES)
    inside = sim.containment(xy, r, a, b, lengths)
    history = pd.read_csv(sim.ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    target = pipes.Material.isin(IRON).to_numpy()
    OUT.mkdir(exist_ok=True)
    for name, kw in variants.items():
        order = so.order_dynamic(inside, r, pipes, lengths, **kw)
        assert sorted(order) == list(range(len(lines)))
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, target)
        print(name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                    f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}")


if __name__ == '__main__':
    main()
