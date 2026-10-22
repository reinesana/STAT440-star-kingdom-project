"""Search orders that beat Rion's 13a across several post-bug-fix calibrated worlds.

Worlds are the hazard models that best match the post-fix public scores (relative
to 13a). Candidates are judged by their worst-case gain over 13a across worlds, to
avoid exploiting any one model's mistakes.

Run from repository root: python src/search_private.py [--sims N]
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
from search_dynamic import CIRCLES, R13A, team_value

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
TEAM_CIRCLES = ROOT / 'submissions/2026-10-19_rion_grouping/10a_09a_nearby_pairs.csv'


def team_p27(pipes):
    pr = pd.read_csv(ROOT / 'submissions/2026-10-20_rion_dynamic/source/probabilities.csv')
    return np.nan_to_num(pipes['Pipe ID'].map(pr.set_index('pipe_id').leak_probability).to_numpy(float))


def worlds(pipes):
    w = sim.fit_drift_wave(pipes)
    table, pooled = sim.fit_hazard(pipes)
    fixed = sim.drift_hazard(pipes, w, slope_scale=0)
    return {
        'fixed wave + team floor': np.maximum(fixed, 0.5 * team_p27(pipes)[:, None]),
        'fixed wave q=.8': sim.drift_hazard(pipes, dict(w, q=0.8), slope_scale=0),
        'empirical age': sim.cohort_hazard(pipes, table, pooled),
    }


def net_value(h, pools, lengths):
    yc = so.yearly_cost(h, pools)
    survive = np.cumprod(1 - h, axis=1)
    p_first = h * np.hstack([np.ones((len(h), 1)), survive[:, :-1]])
    return np.clip(yc - p_first * 200 * lengths[:, None], 0, None)


def candidates(pipes, a, b, lengths, pools):
    w = sim.fit_drift_wave(pipes)
    fixed = sim.drift_hazard(pipes, w, slope_scale=0)
    tv = team_value(pipes, lengths)
    v_fixed = net_value(fixed, pools, lengths)
    v_floor = net_value(np.maximum(fixed, 0.5 * team_p27(pipes)[:, None]), pools, lengths)
    out = {}
    for label, path in (('team circles', TEAM_CIRCLES), ('+brass circles', CIRCLES)):
        xy, r = sim.read_units(path)
        inside = sim.containment(xy, r, a, b, lengths)
        lines = path.read_text(encoding='utf-8').splitlines()
        specs = {
            f'team value yearly ({label})': dict(value_by_year=v_fixed, static_value=tv, blend=1.0),
            f'fixed wave 50% + team ({label})': dict(value_by_year=v_fixed, static_value=tv, blend=0.5),
            f'fixed wave 75% + team ({label})': dict(value_by_year=v_fixed, static_value=tv, blend=0.25),
            f'floor model d.3 ({label})': dict(value_by_year=v_floor, decay=0.3),
            f'floor model 50% + team ({label})': dict(value_by_year=v_floor, static_value=tv, blend=0.5),
        }
        for name, kw in specs.items():
            order = so.order_dynamic(inside, r, pipes, lengths, **kw)
            out[name] = ([inside[k] for k in order], r[order], [lines[k] for k in order])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=12)
    ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args()
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    pools = sim.cost_pools(pipes)
    W = worlds(pipes)
    cands = candidates(pipes, a, b, lengths, pools)
    xy, r = sim.read_units(R13A)
    ref_inside = sim.containment(xy, r, a, b, lengths)
    ref = {k: sim.simulate(ref_inside, r, pipes, lengths, h, pools, args.sims, seed=args.seed).score.values / 1e6
           for k, h in W.items()}
    rows = []
    for name, (ins, rr, _) in cands.items():
        gains = {k: (sim.simulate(ins, rr, pipes, lengths, h, pools, args.sims, seed=args.seed).score.values / 1e6
                     - ref[k]) for k, h in W.items()}
        row = {'candidate': name, **{k: g.mean() for k, g in gains.items()}}
        row['worst'] = min(g.mean() for g in gains.values())
        row['se'] = max(g.std() / np.sqrt(len(g)) for g in gains.values())
        rows.append(row)
        print(row, flush=True)
    print('\nGain over Rion 13a (M$), same futures; worst = minimum across worlds')
    print(pd.DataFrame(rows).sort_values('worst', ascending=False).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
