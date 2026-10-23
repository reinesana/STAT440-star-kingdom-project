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
# Post-evaluator-bug-fix public scores (M$), as displayed on the leaderboard.
# Deleted dead ends (Oct 22), for the record: net-value/age-model order 67.84; free circle
# merging 62.6-63.8; cold-only merging 69.4-69.9; old cohorts capped at 1.4% 68.95; old-iron p
# lowered 69.6-69.8 and x1.5 70.46; old copper/gray raised 70.73; 1961/62 staged 70.48;
# 17b probabilities with the 12b order 61.32; second same-year merge pass 73.90; merges across
# adjacent planned years 73.60.
PUBLIC = {
    S / '2026-10-18_candidate_07_leak_ensemble/09a_full_ensemble_net.csv': 64.56,
    S / '2026-10-18_candidate_07_leak_ensemble/09b_catboost_lightgbm_net.csv': 59.32,
    S / '2026-10-19_rion_grouping/10a_09a_nearby_pairs.csv': 64.62,
    S / '2026-10-20_rion_dynamic/12a_dynamic_net_benefit_per_cost.csv': 69.54,
    S / '2026-10-20_rion_combined 2/13a_grouped_dynamic_net_benefit_per_cost.csv': 69.60,
    S / '2026-10-21_candidate_14_dynamic_drift/14a_dynamic_drift_net_decay030.csv': 65.10,
    S / '2026-10-21_candidate_14_dynamic_drift/14b_dynamic_drift_net_team25.csv': 66.07,
    S / '2026-10-21_candidate_14_dynamic_drift/14c_dynamic_drift_net_skip1.csv': 56.95,
    S / '2026-10-22_candidate_15_calibrated/15a_fixed_wave50_team50_brass.csv': 68.96,
    S / '2026-10-22_candidate_15_calibrated/15b_team_value_brass.csv': 69.56,
    S / '2026-10-22_candidate_16_cohort_stage/16b_stage_only.csv': 70.51,
    S / '2026-10-22_candidate_17_stage_refinements/17b_stronger_1964_1965.csv': 70.66,
    S / '2026-10-22_candidate_18_shifted_circles/18a_shifted_circles_17b_order.csv': 73.14,
    S / '2026-10-22_candidate_20_sameyear_merge/20a_sameyear_merge_17b_order.csv': 74.03,
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
