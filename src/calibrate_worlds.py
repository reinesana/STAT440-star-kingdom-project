"""Score alternative leak-hazard worlds against the public leaderboard.

Run from repository root: python src/calibrate_worlds.py [--sims N]
"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
from calibrate_simulator import PUBLIC

warnings.filterwarnings('ignore')
ONLY_COHORT = '--cohort-only' in sys.argv
if ONLY_COHORT:
    sys.argv.remove('--cohort-only')


def shifted(pipes, table, pooled, years):
    lay = pipes.lay
    p = pipes.copy()
    p['lay'] = lay - pd.DateOffset(years=years)
    return sim.hazard_matrix(p, table, pooled)


def worlds(pipes, table, pooled):
    base = sim.hazard_matrix(pipes, table, pooled)
    metal = ~pipes.mat.eq('polyurethane').to_numpy()
    yield 'age model', base
    for g in (0.0, 0.05, 0.1):
        yield f'cohort bg growth {g}', sim.cohort_hazard(pipes, table, pooled, growth=g)
    if ONLY_COHORT:
        return
    for k in (1, 2):
        yield f'wave {k}y earlier', shifted(pipes, table, pooled, k)
    yield 'wave 1y later', shifted(pipes, table, pooled, -1)
    for f in (2.0, 3.0):
        yield f'age model x{f}', np.minimum(base * f, 0.95)
    flat = np.where(metal[:, None], 0.06, 0.0) * np.ones_like(base)
    yield 'flat 6%/yr', flat
    yield 'age model + flat 3%', np.minimum(base + np.where(metal[:, None], 0.03, 0), 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=6)
    args = ap.parse_args()
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    table, pooled = sim.fit_hazard(pipes)
    pools = sim.cost_pools(pipes)
    subs = {p: sim.read_units(p) for p in PUBLIC}
    inside = {p: sim.containment(*subs[p], a, b, lengths) for p in PUBLIC}
    for name, h in worlds(pipes, table, pooled):
        out = pd.DataFrame([dict(sub=p.stem[:12], public=v,
                                 sim=sim.simulate(inside[p], subs[p][1], pipes, lengths, h, pools,
                                                  args.sims).score.mean() / 1e6)
                            for p, v in PUBLIC.items()])
        err = out.sim - out.public
        print(f'{name:22s} rmse {np.sqrt((err**2).mean()):6.1f}  bias {err.mean():6.1f}  '
              f'sd(err) {err.std():5.1f}  ' + ' '.join(f'{s:7.1f}' for s in out.sim))


if __name__ == '__main__':
    main()
