"""Compare simulated mean scores with the public leaderboard for past submissions.

Run from repository root: python src/calibrate_simulator.py [--sims N] [--rates r1 r2 ...]
"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim

warnings.filterwarnings('ignore')
S = sim.ROOT / 'submissions'
PUBLIC = {
    S / '2026-10-05_baseline_v1.csv': -25.58,
    S / '2026-10-13_cost_candidates/06_stable_risk_per_work_cost.csv': -15.61,
    S / '2026-10-18_candidate_07_leak_ensemble/07a_full_ensemble_gross.csv': -11.71,
    S / '2026-10-18_candidate_07_leak_ensemble/07b_catboost_lightgbm_gross.csv': -23.39,
    S / '2026-10-18_candidate_07_leak_ensemble/09a_full_ensemble_net.csv': -6.79,
    S / '2026-10-18_candidate_07_leak_ensemble/09b_catboost_lightgbm_net.csv': -14.40,
    S / '2026-10-19_rion_grouping/10a_09a_nearby_pairs.csv': -5.98,
    S / '2026-10-20_rion_dynamic/12a_dynamic_net_benefit_per_cost.csv': 22.50,
    S / '2026-10-20_rion_combined 2/13a_grouped_dynamic_net_benefit_per_cost.csv': 23.29,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=10)
    ap.add_argument('--rates', type=float, nargs='+', default=[None])
    ap.add_argument('--starts', type=float, nargs='+', default=[68.0])
    args = ap.parse_args()
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    table, pooled = sim.fit_hazard(pipes)
    pools = sim.cost_pools(pipes)
    subs = {p: sim.read_units(p) for p in PUBLIC}
    inside = {p: sim.containment(*subs[p], a, b, lengths) for p in PUBLIC}
    for start, rate in [(s, r) for s in args.starts for r in args.rates]:
        h = sim.hazard_matrix(pipes, table, pooled, tail_start=start, tail_rate=rate)
        rows = []
        for p, public in PUBLIC.items():
            df = sim.simulate(inside[p], subs[p][1], pipes, lengths, h, pools, args.sims)
            rows.append(dict(sub=p.stem[:28], public=public, sim=df.score.mean() / 1e6,
                             constr=df.construction.mean() / 1e6, repair=df.repair.mean() / 1e6))
        out = pd.DataFrame(rows)
        bias = (out.sim - out.public).mean()
        corr = np.corrcoef(out.sim, out.public)[0, 1]
        rmse = np.sqrt(((out.sim - out.public)**2).mean())
        print(f'\ntail_start={start} tail_rate={rate}  rmse={rmse:.1f}M  '
              f'mean(sim-public)={bias:.1f}M  corr={corr:.2f}')
        print(out.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
