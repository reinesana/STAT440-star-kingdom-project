"""Search circle orderings and score them in several calibrated leak worlds.

Worlds: cohort-background hazard with protection lag 0, 1 and 2 years. The lag
worlds are the ones that best match the public leaderboard (see calibrate_worlds.py).

Run from repository root: python src/search_orders.py [--sims N]
"""
from pathlib import Path
import argparse
import math
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
from build_baseline import RATES

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
SRC = ROOT / 'submissions/2026-10-19_rion_grouping/10a_09a_nearby_pairs.csv'
LAGS = (0, 1, 2)


def yearly_cost(h, pools):
    survive = np.cumprod(1 - h, axis=1)
    p_first = h * np.hstack([np.ones((len(h), 1)), survive[:, :-1]])
    return p_first * np.array([p.mean() for p in pools])[:, None]


def circle_terms(inside, r, pipes, lengths):
    poly = (pipes.mat.eq('polyurethane') | pipes.leak_date.notna()).to_numpy()
    surf = pipes.Surface.to_numpy()
    live = [ids[~poly[ids]] for ids in inside]
    cost = np.array([100 + (sum(RATES[s] for s in set(surf[ids])) * math.pi * rad**2)**0.85
                     + 200 * lengths[lv].sum() for ids, lv, rad in zip(inside, live, r)])
    return live, cost


def order_by(live, cost, yc, horizon, skip=0):
    """Benefit = expected repair cost in project years [skip, skip+horizon)."""
    window = yc[:, skip:skip + horizon].sum(1)
    tail = yc.sum(1)
    near = np.array([window[lv].sum() for lv in live]) / cost
    full = np.array([tail[lv].sum() for lv in live]) / cost
    return np.lexsort((-full, -near))


def order_decay(live, cost, yc, decay, skip=0):
    """Benefit = expected repair cost weighted by exp(-decay * years after skip)."""
    w = np.exp(-decay * np.clip(np.arange(yc.shape[1]) - skip, 0, None))
    w[:skip] = 0
    score = yc @ w
    return np.argsort(-np.array([score[lv].sum() for lv in live]) / cost, kind='stable')


def order_rolling(live, cost, yc, horizon, decay=0.0, budget=10_000_000, skip=0):
    """Re-rank each project year by expected repair cost over the next `horizon` years,
    starting `skip` years ahead."""
    remaining = np.ones(len(live), bool)
    order, carry = [], 0.0
    for yi in range(yc.shape[1]):
        if not remaining.any():
            break
        w = np.zeros(yc.shape[1])
        start = min(yi + skip, yc.shape[1] - 1)
        w[start:start + horizon] = np.exp(-decay * np.arange(min(horizon, yc.shape[1] - start)))
        score = yc @ w
        idx = np.where(remaining)[0]
        pri = np.array([score[live[k]].sum() for k in idx]) / cost[idx]
        tail = np.array([yc[live[k], yi:].sum() for k in idx]) / cost[idx]
        ranked = idx[np.lexsort((-tail, -pri))]
        spend = carry + budget
        take = ranked[np.cumsum(cost[ranked]) <= spend]
        carry = spend - cost[take].sum()
        order.extend(take)
        remaining[take] = False
    order.extend(np.where(remaining)[0])
    return np.array(order)


def order_dynamic(inside, r, pipes, lengths, value_by_year, decay=0.15, skip=0,
                  budget=10_000_000, static_value=None, blend=0.0):
    """Year-by-year greedy that also updates overlapping circles after each pick.

    Within a year, circles are chosen by (value of still-unreplaced pipes) /
    (current construction cost); after each pick its pipes count as replaced, so
    overlapping circles lose value and cost. A circle that doesn't fit the remaining
    budget opens the next year, matching the evaluator's head-of-line rule.
    `static_value` (per pipe) is mixed in with weight `blend`.
    """
    import heapq
    surf = pipes.Surface.to_numpy()
    area = np.array([sum(RATES[s] for s in set(surf[ids])) * math.pi * rad**2
                     for ids, rad in zip(inside, r)])**0.85 + 100
    state = (pipes.mat.eq('polyurethane') | pipes.leak_date.notna()).to_numpy().copy()
    reverse = [[] for _ in range(len(pipes))]
    for j, ids in enumerate(inside):
        for i in ids:
            reverse[i].append(j)
    done = np.zeros(len(inside), bool)
    order, carry, pending = [], 0.0, None
    n_years = value_by_year.shape[1]
    for yi in range(n_years):
        if done.all():
            break
        start = min(yi + skip, n_years - 1)
        w = np.zeros(n_years)
        w[start:] = np.exp(-decay * np.arange(n_years - start))
        val = value_by_year @ w
        if static_value is not None:
            val = (1 - blend) * val / max(val.mean(), 1e-9) + blend * static_value / max(static_value.mean(), 1e-9)
        version = np.zeros(len(inside), int)
        heap = []

        def push(j):
            ids = inside[j]
            act = ids[~state[ids]]
            cost = area[j] + 200 * lengths[act].sum() if len(act) else 0.0
            pri = np.inf if not len(act) else val[act].sum() / cost
            heapq.heappush(heap, (-pri, j, version[j]))

        for j in np.where(~done)[0]:
            push(j)
        spend = carry + budget
        if pending is not None:
            heap.append((-np.inf, pending, version[pending]))
            heapq.heapify(heap)
            pending = None
        while heap:
            _, j, v = heapq.heappop(heap)
            if done[j] or v != version[j]:
                continue
            ids = inside[j]
            act = ids[~state[ids]]
            cost = area[j] + 200 * lengths[act].sum() if len(act) else 0.0
            if cost > spend:
                pending = j
                break
            spend -= cost
            done[j] = True
            order.append(j)
            state[act] = True
            for k in {k for i in act for k in reverse[i] if not done[k]}:
                version[k] += 1
                push(k)
        carry = spend
    order.extend(np.where(~done)[0])
    return np.array(order)


def extra_metal_circles(inside, pipes, a, b, yc, margin=1e-6):
    """Single-pipe circles for uncovered copper/brass whose expected repair exceeds cost."""
    covered = np.zeros(len(pipes), bool)
    for ids in inside:
        covered[ids] = True
    lengths = pipes.len.to_numpy()
    rate = pipes.Surface.map(RATES).to_numpy()
    cost = 100 + (rate * math.pi * (lengths / 2)**2)**0.85 + 200 * lengths
    pick = (pipes.mat.isin(['copper', 'brass']) & pipes.leak_date.isna()).to_numpy() & ~covered
    pick &= yc.sum(1) > cost
    idx = np.where(pick)[0]
    return (a[idx] + b[idx]) / 2, lengths[idx] / 2 + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=8)
    ap.add_argument('--model', choices=['cohort', 'drift'], default='drift')
    args = ap.parse_args()
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    table, pooled = sim.fit_hazard(pipes)
    pools = sim.cost_pools(pipes)
    if args.model == 'drift':
        h = sim.drift_hazard(pipes, sim.fit_drift_wave(pipes))
    else:
        h = sim.cohort_hazard(pipes, table, pooled)
    yc = yearly_cost(h, pools)
    xy, r = sim.read_units(SRC)
    inside = sim.containment(xy, r, a, b, lengths)
    n_base = len(inside)
    ex_xy, ex_r = extra_metal_circles(inside, pipes, a, b, yc)
    xy, r = np.vstack([xy, ex_xy]), np.concatenate([r, ex_r])
    inside = inside + sim.containment(ex_xy, ex_r, a, b, lengths)
    live, cost = circle_terms(inside, r, pipes, lengths)
    base_only = np.arange(n_base)

    candidates = {'10a (as submitted)': base_only}
    c12 = ROOT / 'submissions/2026-10-19_candidate_12_rolling_age_wave'
    for name in ('12a_rolling_decay030', '12b_rolling_decay015'):
        cxy, cr = sim.read_units(c12 / f'{name}.csv')
        key = {(round(x, 6), round(y, 6), round(rad, 6)): k for k, ((x, y), rad) in enumerate(zip(xy, r))}
        candidates[name] = np.array([key[(round(x, 6), round(y, 6), round(rad, 6))]
                                     for (x, y), rad in zip(cxy, cr)])
    for skip in (0, 1, 2):
        for decay in (0.15, 0.3):
            candidates[f'rolling skip {skip} decay {decay}'] = order_rolling(live, cost, yc, 25, decay, skip=skip)
        candidates[f'rolling skip {skip} horizon 3y'] = order_rolling(live, cost, yc, 3, skip=skip)

    rows = []
    for name, order in candidates.items():
        ins = [inside[k] for k in order]
        row = {'candidate': name}
        for lag in LAGS:
            df = sim.simulate(ins, r[order], pipes, lengths, h, pools, args.sims, lag=lag)
            row[f'lag{lag}'] = df.score.mean() / 1e6
            row['feasible'] = bool(df.feasible.all())
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    for l in LAGS:
        out[f'lag{l}'] -= out[f'lag{l}'].iloc[0]
    out['worst'] = out[[f'lag{l}' for l in LAGS]].min(1)
    out['avg'] = out[[f'lag{l}' for l in LAGS]].mean(1)
    print('Scores below are gains over 10a in M$ (same simulated futures for every candidate).')
    print(out.sort_values('avg', ascending=False).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
