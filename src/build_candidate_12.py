"""Candidate 12: 10a circles plus high-value brass/copper circles, re-ranked each year.

Orders come from search_orders.py using the cohort-background age-wave hazard.
Each CSV is validated with the team validator (build_cost_submissions.validate).

Run from repository root: python src/build_candidate_12.py
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so
from build_baseline import IRON
from build_cost_submissions import validate

warnings.filterwarnings('ignore')
OUT = sim.ROOT / 'submissions/2026-10-19_candidate_12_rolling_age_wave'
VARIANTS = {
    '12a_rolling_decay030': lambda live, cost, yc: so.order_rolling(live, cost, yc, 25, 0.3),
    '12b_rolling_decay015': lambda live, cost, yc: so.order_rolling(live, cost, yc, 25, 0.15),
    '12c_static_decay015': lambda live, cost, yc: so.order_decay(live, cost, yc, 0.15),
}


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    table, pooled = sim.fit_hazard(pipes)
    pools = sim.cost_pools(pipes)
    yc = so.yearly_cost(sim.cohort_hazard(pipes, table, pooled), pools)

    raw = pd.read_csv(so.SRC, header=None, dtype=str)
    xy, r = sim.read_units(so.SRC)
    inside = sim.containment(xy, r, a, b, lengths)
    ex_xy, ex_r = so.extra_metal_circles(inside, pipes, a, b, yc)
    lines = [f'"{c}",{rad}' for c, rad in zip(raw[0], raw[1])]
    lines += [f'"({x:.9f}, {y:.9f})",{rad:.9f}' for (x, y), rad in zip(ex_xy, ex_r)]
    ex_inside = sim.containment(*sim_read_lines(lines[len(raw):]), a, b, lengths)
    assert all(len(ids) for ids in ex_inside), 'extra circle lost its pipe after rounding'
    inside = inside + ex_inside
    r_all = np.concatenate([r, ex_r])
    live, cost = so.circle_terms(inside, r_all, pipes, lengths)

    history = pd.read_csv(sim.ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    target = pipes.Material.isin(IRON).to_numpy()
    OUT.mkdir(exist_ok=True)
    for name, make in VARIANTS.items():
        order = make(live, cost, yc)
        assert sorted(order) == list(range(len(lines)))
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, target)
        print(name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                    f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}")


def sim_read_lines(lines):
    xy = np.array([[float(v) for v in ln.split('"')[1].strip('()').split(',')] for ln in lines])
    r = np.array([float(ln.rsplit(',', 1)[1]) for ln in lines])
    return xy, r


if __name__ == '__main__':
    main()
