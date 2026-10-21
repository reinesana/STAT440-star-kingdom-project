"""Candidate 13: candidate-12 circles re-ranked yearly under the drifting-wave hazard.

The wear-out wave's mean age drifts earlier with lay date (simulate_score.fit_drift_wave).
`skip` aims each year's work that many years ahead, for worlds where replacement only
protects after a delay.

Run from repository root: python src/build_candidate_13.py
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
SRC = sim.ROOT / 'submissions/2026-10-19_candidate_12_rolling_age_wave/12b_rolling_decay015.csv'
OUT = sim.ROOT / 'submissions/2026-10-21_candidate_13_drift_wave'
VARIANTS = {'13a_drift_rolling_skip0': 0, '13b_drift_rolling_skip1': 1, '13c_drift_rolling_skip2': 2}


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    yc = so.yearly_cost(sim.drift_hazard(pipes, sim.fit_drift_wave(pipes)), sim.cost_pools(pipes))
    lines = SRC.read_text(encoding='utf-8').splitlines()
    xy, r = sim.read_units(SRC)
    inside = sim.containment(xy, r, a, b, lengths)
    live, cost = so.circle_terms(inside, r, pipes, lengths)
    history = pd.read_csv(sim.ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    target = pipes.Material.isin(IRON).to_numpy()
    OUT.mkdir(exist_ok=True)
    for name, skip in VARIANTS.items():
        order = so.order_rolling(live, cost, yc, 25, 0.15, skip=skip)
        assert sorted(order) == list(range(len(lines)))
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, target)
        print(name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                    f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}")


if __name__ == '__main__':
    main()
