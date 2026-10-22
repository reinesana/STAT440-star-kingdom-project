"""Candidate 15: post-bug-fix calibrated orders built to beat Rion's 13a.

15a: dynamic order, value = 50% fixed-age wave yearly risk + 50% team 09a value,
     on 10a circles plus the 94 brass circles.
15b: Rion's 13a value rule (team 09a value) on the same circles, i.e. 13a + brass.

Run from repository root: python src/build_candidate_15.py
"""
from pathlib import Path
import sys
import warnings

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
from search_private import candidates
from build_baseline import IRON
from build_cost_submissions import validate

warnings.filterwarnings('ignore')
OUT = sim.ROOT / 'submissions/2026-10-22_candidate_15_calibrated'
PICK = {
    '15a_fixed_wave50_team50_brass': 'fixed wave 50% + team (+brass circles)',
    '15b_team_value_brass': 'team value yearly (+brass circles)',
}


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    built = candidates(pipes, a, b, lengths, sim.cost_pools(pipes))
    history = pd.read_csv(sim.ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    target = pipes.Material.isin(IRON).to_numpy()
    OUT.mkdir(exist_ok=True)
    for name, key in PICK.items():
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(built[key][2]) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, target)
        print(name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                    f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}")


if __name__ == '__main__':
    main()
