"""Compare dynamic (overlap-updating) yearly orderings against Rion's 13a.

Run from repository root: python src/search_dynamic.py [--sims N]
"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
CIRCLES = ROOT / 'submissions/2026-10-19_candidate_12_rolling_age_wave/12b_rolling_decay015.csv'
R13A = ROOT / 'submissions/2026-10-20_rion_combined 2/13a_grouped_dynamic_net_benefit_per_cost.csv'


def team_value(pipes, lengths):
    """09a/Rion per-pipe net benefit: p(leak 2027) * (predicted repair - $200/m)."""
    prob = pd.read_csv(ROOT / 'submissions/2026-10-20_rion_dynamic/source/probabilities.csv').set_index('pipe_id')
    sev = pd.read_csv(ROOT / 'notebooks/ensemble_for_cost_prediction/predictions/ensemble_stable.csv').set_index('id')
    p = pipes['Pipe ID'].map(prob.leak_probability).to_numpy(float)
    s = pipes['Pipe ID'].map(sev.predicted_cost).to_numpy(float)
    v = np.nan_to_num(p * (s - 200 * lengths))
    return np.clip(v, 0, None)


def build_all(pipes, a, b, lengths):
    h = sim.drift_hazard(pipes, sim.fit_drift_wave(pipes))
    pools = sim.cost_pools(pipes)
    yc = so.yearly_cost(h, pools)
    survive = np.cumprod(1 - h, axis=1)
    p_first = h * np.hstack([np.ones((len(h), 1)), survive[:, :-1]])
    yc_net = np.clip(yc - p_first * 200 * lengths[:, None], 0, None)
    xy, r = sim.read_units(CIRCLES)
    inside = sim.containment(xy, r, a, b, lengths)
    tv = team_value(pipes, lengths)
    variants = {
        'dyn gross d.15': dict(value_by_year=yc),
        'dyn net d.15': dict(value_by_year=yc_net),
        'dyn net d.30': dict(value_by_year=yc_net, decay=0.30),
        'dyn net d.15 skip1': dict(value_by_year=yc_net, skip=1),
        'dyn net + team 50%': dict(value_by_year=yc_net, static_value=tv, blend=0.5),
        'dyn net + team 25%': dict(value_by_year=yc_net, static_value=tv, blend=0.25),
    }
    orders = {name: so.order_dynamic(inside, r, pipes, lengths, **kw) for name, kw in variants.items()}
    return xy, r, inside, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=16)
    ap.add_argument('--seed', type=int, default=31)
    args = ap.parse_args()
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    xy, r, inside, orders = build_all(pipes, a, b, lengths)
    h = sim.drift_hazard(pipes, sim.fit_drift_wave(pipes))
    pools = sim.cost_pools(pipes)
    cands = {}
    for name in ('Rion 13a', 'my 13a drift'):
        path = R13A if name == 'Rion 13a' else ROOT / 'submissions/2026-10-21_candidate_13_drift_wave/13a_drift_rolling_skip0.csv'
        cxy, cr = sim.read_units(path)
        cands[name] = (sim.containment(cxy, cr, a, b, lengths), cr)
    for name, order in orders.items():
        cands[name] = ([inside[k] for k in order], r[order])
    res = {name: [sim.simulate(ins, rr, pipes, lengths, h, pools, args.sims, seed=args.seed, lag=g).score.values / 1e6
                  for g in (0, 1, 2)] for name, (ins, rr) in cands.items()}
    ref = res['Rion 13a']
    print('Gain over Rion 13a (M$), same simulated futures; ± is the standard error of the mean')
    for name, sc in res.items():
        d = [sc[g] - ref[g] for g in range(3)]
        print(f'{name:22s} ' + '  '.join(f'lag{g} {d[g].mean():+6.2f}±{d[g].std()/np.sqrt(len(d[g])):.2f}' for g in range(3)))


if __name__ == '__main__':
    main()
