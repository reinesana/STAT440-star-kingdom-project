"""Candidate 18: 17b (public 70.66) with single-pipe circles shifted off the midpoint
where that excludes an expensive neighbour.

A pipe's minimal circle (midpoint, r = L/2) sometimes wholly contains a neighbouring
pipe with a costlier surface (or live copper/brass), so the whole area is charged at
the summed rate. Moving the centre along the perpendicular bisector by t gives radius
sqrt((L/2)^2 + t^2) and can exclude the neighbour; we keep the cheapest. Only circles
that are exactly one pipe's minimal circle in the 10a geometry are replaced, and the
ordering rule is unchanged, so there is no timing side-effect.

Run from repository root: python src/build_candidate_18.py
"""
from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so
from search_dynamic import CIRCLES
from build_baseline import RATES, IRON, MARGIN_M
from build_cost_submissions import validate
from build_candidate_16 import team_inputs
from build_candidate_17 import p17b

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
OUT = ROOT / 'submissions/2026-10-22_candidate_18_shifted_circles'


def shifted_circles(pipes, a, b, lengths, min_saving=1.0):
    """{pipe index: (centre, radius, saving)} for live iron pipes whose shifted circle is cheaper."""
    surf = pipes.Surface.to_numpy()
    live = (~(pipes.mat.eq('polyurethane') | pipes.leak_date.notna())).to_numpy()
    iron = pipes.mat.isin(IRON).to_numpy()
    mid = (a + b) / 2
    tree = cKDTree(mid)

    def contained(c, r):
        ids = np.asarray(tree.query_ball_point(c, r + 1e-9), int)
        return ids[(np.linalg.norm(a[ids] - c, axis=1) <= r) & (np.linalg.norm(b[ids] - c, axis=1) <= r)]

    def cost(ids, r):
        return 100 + (sum(RATES[s] for s in set(surf[ids])) * math.pi * r * r) ** 0.85 \
            + 200 * lengths[ids[live[ids] & ~iron[ids]]].sum()

    out = {}
    for i in np.where(live & iron)[0]:
        d = b[i] - a[i]
        n = np.array([-d[1], d[0]]) / max(lengths[i], 1e-9)
        r0 = lengths[i] / 2 + MARGIN_M
        ids0 = contained(mid[i], r0)
        if len(set(surf[ids0])) == 1 and not (live[ids0] & ~iron[ids0]).any():
            continue
        c0 = cost(ids0, r0)
        best = (c0, None, None)
        grid = np.geomspace(0.05, 2 * lengths[i] + 5, 40)
        for t in np.concatenate([-grid, grid]):
            c = mid[i] + t * n
            r = math.sqrt((lengths[i] / 2) ** 2 + t * t) + MARGIN_M
            ids = contained(c, r)
            assert i in ids
            cc = cost(ids, r)
            if cc < best[0]:
                best = (cc, c, r)
        if best[1] is not None and c0 - best[0] >= min_saving:
            out[i] = (best[1], best[2], c0 - best[0])
    return out


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    mid = (a + b) / 2
    p_ml, sev = team_inputs(pipes)
    value = np.clip(p17b(pipes, p_ml) * (sev - 200 * lengths), 0, None)
    shifted = shifted_circles(pipes, a, b, lengths)
    print(f'{len(shifted)} pipes with cheaper shifted circles; saving {sum(v[2] for v in shifted.values())/1e6:.2f}M')
    lines = CIRCLES.read_text(encoding='utf-8').splitlines()
    xy, r = sim.read_units(CIRCLES)
    # identify each line that is exactly pipe i's minimal circle
    tree = cKDTree(mid)
    replaced = 0
    for k in range(len(lines)):
        dist, i = tree.query(xy[k])
        if dist < 1e-6 and abs(r[k] - (lengths[i] / 2 + MARGIN_M)) < 1e-6 and i in shifted:
            c, rad, _ = shifted[i]
            lines[k] = f'"({c[0]:.9f}, {c[1]:.9f})",{rad:.9f}'
            replaced += 1
    print('circles replaced:', replaced)
    tmp = OUT / 'circles_shifted.csv'
    OUT.mkdir(exist_ok=True)
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    xy, r = sim.read_units(tmp)
    inside = sim.containment(xy, r, a, b, lengths)
    dummy = np.zeros((len(pipes), len(sim.YEARS)))
    order = so.order_dynamic(inside, r, pipes, lengths, value_by_year=dummy, static_value=value, blend=1.0)
    assert sorted(order) == list(range(len(lines)))
    path = OUT / '18a_shifted_circles_17b_order.csv'
    path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    rep = validate(path, pipes, history, a, b, mid, lengths, pipes.Material.isin(IRON).to_numpy())
    print(path.name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                     f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}")


if __name__ == '__main__':
    main()
