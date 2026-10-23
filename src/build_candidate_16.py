"""Candidate 16: 15b's machinery (team value, dynamic overlap updates, 10a circles + brass)
with the 2027 leak probabilities adjusted cohort by cohort from the observed wave stages.

Completed cohorts (laid 1956-1962) show a 2-3 year wear-out wave that drifts earlier
with lay year; the stage each live cohort has reached by end-2026 implies its 2027 hazard:
  1963: third wave year  -> ~0.12 (ML ensemble says 0.64)
  1964: second wave year -> ~0.50 (ML 0.51)
  1965: second wave year after a weak start -> ~0.60 (ML 0.25)
16b (public 70.51) changes only these cohorts. A sibling that also capped pre-1956
cohorts at 1.4% scored 68.95 and was dropped.

Run from repository root: python src/build_candidate_16.py
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

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
OUT = ROOT / 'submissions/2026-10-22_candidate_16_cohort_stage'
STAGE = {1963: 0.12, 1964: 0.50, 1965: 0.60, 1966: 0.30}
OLD_RATE = 0.014


def team_inputs(pipes):
    prob = pd.read_csv(ROOT / 'submissions/2026-10-20_rion_dynamic/source/probabilities.csv').set_index('pipe_id')
    sev = pd.read_csv(ROOT / 'notebooks/ensemble_for_cost_prediction/predictions/ensemble_stable.csv').set_index('id')
    p = np.nan_to_num(pipes['Pipe ID'].map(prob.leak_probability).to_numpy(float))
    s = np.nan_to_num(pipes['Pipe ID'].map(sev.predicted_cost).to_numpy(float))
    return p, s


def adjusted_p(pipes, p_ml, stage=True, old=True):
    p = p_ml.copy()
    ly = pipes.lay.dt.year.to_numpy()
    live = (~(pipes.mat.eq('polyurethane') | pipes.leak_date.notna())).to_numpy()
    if stage:
        for yr, v in STAGE.items():
            m = live & (ly == yr)
            p[m] = v
    if old:
        m = live & (ly <= 1955)
        p[m] = np.minimum(p[m], OLD_RATE)
    return p


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    p_ml, sev = team_inputs(pipes)
    variants = {'16b_stage_only': adjusted_p(pipes, p_ml, stage=True, old=False)}
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
