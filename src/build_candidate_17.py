"""Candidate 17: refinements on 16b (public 70.51).

16b = 15b machinery with wave-stage 2027 probabilities for the 1963-1966 cohorts.
Each 17 variant adds exactly one further change so public scores stay interpretable:
  17a: stage also for 1961 (4th wave year, 0.03) and 1962 (3rd wave year, 0.10)
  17b: stronger 2027 wave for 1964 (0.55) and 1965 (0.70)
  17c: 1971 cohort (wave due ~2031-34, 1,546 live pipes) p raised to 0.10
  17d: old copper / gray iron (pre-1956) raised to their observed rates (0.13 / 0.05)

Run from repository root: python src/build_candidate_17.py
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so
from search_dynamic import CIRCLES
from build_baseline import IRON
from build_cost_submissions import validate
from build_candidate_16 import team_inputs, adjusted_p

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
OUT = ROOT / 'submissions/2026-10-22_candidate_17_stage_refinements'


def p17b(pipes, p_ml):
    """2027 probabilities of candidate 17b (public 70.66): 16b plus 1964 -> 0.55, 1965 -> 0.70."""
    p = adjusted_p(pipes, p_ml, stage=True, old=False)
    ly = pipes.lay.dt.year.to_numpy()
    live = (~(pipes.mat.eq('polyurethane') | pipes.leak_date.notna())).to_numpy()
    p[live & (ly == 1964)] = 0.55
    p[live & (ly == 1965)] = 0.70
    return p


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    p_ml, sev = team_inputs(pipes)
    base = adjusted_p(pipes, p_ml, stage=True, old=False)           # == 16b
    ly = pipes.lay.dt.year.to_numpy()
    live = (~(pipes.mat.eq('polyurethane') | pipes.leak_date.notna())).to_numpy()
    mat = pipes.mat.to_numpy()

    def with_(**over):
        p = base.copy()
        for yr, v in over.get('cohort', {}).items():
            p[live & (ly == yr)] = v
        for m, v in over.get('old_material', {}).items():
            p[live & (ly <= 1955) & (mat == m)] = v
        return p

    variants = {
        '17a_stage_1961_1962': with_(cohort={1961: 0.03, 1962: 0.10}),
        '17b_stronger_1964_1965': with_(cohort={1964: 0.55, 1965: 0.70}),
        '17c_1971_p10': with_(cohort={1971: 0.10}),
        '17d_old_copper_gray': with_(old_material={'copper': 0.13, 'gray iron': 0.05}),
    }
    lines = CIRCLES.read_text(encoding='utf-8').splitlines()
    xy, r = sim.read_units(CIRCLES)
    inside = sim.containment(xy, r, a, b, lengths)
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    target = pipes.Material.isin(IRON).to_numpy()
    dummy = np.zeros((len(pipes), len(sim.YEARS)))
    OUT.mkdir(exist_ok=True)
    for name, p in variants.items():
        value = np.clip(p * (sev - 200 * lengths), 0, None)
        order = so.order_dynamic(inside, r, pipes, lengths, value_by_year=dummy, static_value=value, blend=1.0)
        assert sorted(order) == list(range(len(lines)))
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, target)
        print(name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                    f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}", flush=True)


if __name__ == '__main__':
    main()
