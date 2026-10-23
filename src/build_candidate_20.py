"""Candidate 20: 18a (public 73.14) with circles merged only among pipes that 18a's
order replaces in the SAME year.

Why: earlier free merging cut construction but lost 6-9M publicly, because a merged
circle's priority (value / cost) is diluted when hot and cold pipes share it, delaying
the hot ones. Requiring one planned year per merged circle removes that effect, so the
construction saving should carry through.

Accounting (no future leaks): total = sum over executed circles of (100 + area term)
+ $200/m of every live pipe covered at least once. A merge saves the parents' fixed
terms (plus those of circles made redundant because the merged circle covers all their
required pipes) minus the merged circle's fixed term and any live non-iron newly covered.

Run from repository root: python src/build_candidate_20.py
"""
from pathlib import Path
import heapq
import itertools
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_score as sim
import search_orders as so
from build_baseline import RATES, IRON, MARGIN_M
from build_cost_submissions import validate
from build_candidate_16 import team_inputs
from build_candidate_17 import p17b

warnings.filterwarnings('ignore')
ROOT = sim.ROOT
OUT = ROOT / 'submissions/2026-10-22_candidate_20_sameyear_merge'
BASE = ROOT / 'submissions/2026-10-22_candidate_18_shifted_circles/18a_shifted_circles_17b_order.csv'


# ---------------------------------------------------------------- geometry helpers
def _circle3(x, y, z):
    (ax, ay), (bx, by), (cx, cy) = x, y, z
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array([ux, uy]), math.hypot(ax - ux, ay - uy)


def mec(points):
    """Minimum enclosing circle by brute force over 2- and 3-point subsets (small sets)."""
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) == 1:
        return pts[0], 0.0
    best = None
    for x, y in itertools.combinations(pts, 2):
        c = (x + y) / 2
        rad = float(np.linalg.norm(x - c))
        if (best is None or rad < best[1]) and (np.linalg.norm(pts - c, axis=1) <= rad + 1e-7).all():
            best = (c, rad)
    for x, y, z in itertools.combinations(pts, 3):
        res = _circle3(x, y, z)
        if res and (best is None or res[1] < best[1]) and (np.linalg.norm(pts - res[0], axis=1) <= res[1] + 1e-7).all():
            best = res
    return best


def planned_years(path, pipes, a, b, lengths):
    """Year index (0 = 2027) in which each pipe is replaced under `path`'s order, no future leaks."""
    surf = pipes.Surface.to_numpy()
    poly = (pipes.mat.eq('polyurethane') | pipes.leak_date.notna()).to_numpy().copy()
    xy, r = sim.read_units(path)
    inside = sim.containment(xy, r, a, b, lengths)
    yr = np.full(len(pipes), -1)
    executes = np.zeros(len(inside), bool)
    budget, cur = 0.0, 0
    for yi in range(25):
        budget += 1e7
        while cur < len(inside):
            ids = inside[cur]
            rep = ids[~poly[ids]]
            if not len(rep):
                cur += 1
                continue
            cost = 100 + (sum(RATES[s] for s in set(surf[ids])) * math.pi * r[cur] ** 2) ** 0.85 + 200 * lengths[rep].sum()
            if cost > budget:
                break
            budget -= cost
            poly[rep] = True
            yr[rep] = yi
            executes[cur] = True
            cur += 1
        if cur == len(inside):
            break
    return yr, executes


# ---------------------------------------------------------------- merger
def merge_same_year(pipes, a, b, lengths, xy, r, year, executes, slack=1.3, max_pipes=10, min_saving=50.0,
                    max_gap=0, verbose=True):
    """Start from the given circles; greedily merge pairs whose live pipes share one planned year."""
    surf = pipes.Surface.to_numpy()
    live = (~(pipes.mat.eq('polyurethane') | pipes.leak_date.notna())).to_numpy()
    iron = pipes.mat.isin(IRON).to_numpy()
    mid = (a + b) / 2
    ptree = cKDTree(mid)

    def contained(c, rad):
        ids = np.asarray(ptree.query_ball_point(c, rad + 1e-9), int)
        return ids[(np.linalg.norm(a[ids] - c, axis=1) <= rad) & (np.linalg.norm(b[ids] - c, axis=1) <= rad)]

    def fixed_term(ids, rad):
        return 0.0 if not live[ids].any() else 100 + (sum(RATES[s] for s in set(surf[ids])) * math.pi * rad ** 2) ** 0.85

    inside = [contained(c, rad) for c, rad in zip(xy, r)]
    cover = np.zeros(len(pipes), int)
    for ids in inside:
        cover[ids[iron[ids]]] += 1
    cnt = np.zeros(len(pipes), int)          # live pipes: number of alive circles covering them
    circles = []
    by_req = {}
    for k, ids in enumerate(inside):
        ir = ids[iron[ids]]
        req = ir[cover[ir] == 1]             # iron this circle alone covers
        # a circle skipped in the base order (its pipes already replaced by an earlier circle)
        # pays nothing, so retiring it saves nothing
        circles.append(dict(c=xy[k], r=r[k], req=req, ids=ids, fixed=fixed_term(ids, r[k]) if executes[k] else 0.0, alive=True))
        cnt[ids[live[ids]]] += 1
        for p in req.tolist():
            by_req.setdefault(p, []).append(k)
    ctree = cKDTree(xy)
    rmax = r.max()
    CELL = 100.0
    grid = {}

    def cells(c, rad):
        lo = np.floor((c - rad) / CELL).astype(int)
        hi = np.floor((c + rad) / CELL).astype(int)
        return [(x, y) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)]

    def evaluate(j, k):
        cj, ck = circles[j], circles[k]
        req = np.union1d(cj['req'], ck['req'])
        if not len(req) or len(req) > max_pipes:
            return None
        c, rad = mec(np.vstack([a[req], b[req]]))
        rad += MARGIN_M
        ids = contained(c, rad)
        yl = year[ids[live[ids]]]
        if len(yl) and (yl.max() - yl.min() > max_gap):
            return None                      # live pipes from planned years too far apart
        # redundant circles: alive, not j/k, all their required pipes inside the merged circle
        s = set(ids.tolist())
        red = []
        for i in ids:
            for q in by_req.get(int(i), ()):
                if q in (j, k) or not circles[q]['alive'] or q in red:
                    continue
                if all(p in s for p in circles[q]['req'].tolist()):
                    red.append(q)
        # no iron pipe may lose its last cover (incidental iron in retired circles)
        retired = [j, k] + red
        lost = {}
        for q in retired:
            qi = circles[q]['ids']
            for p in qi[iron[qi]].tolist():
                if p not in s:
                    lost[p] = lost.get(p, 0) + 1
        if any(cover[p] - n_ <= 0 for p, n_ in lost.items()):
            return None
        saving = cj['fixed'] + ck['fixed'] + sum(circles[q]['fixed'] for q in red) - fixed_term(ids, rad)
        lv = ids[live[ids] & ~iron[ids]]
        saving -= 200 * lengths[lv[cnt[lv] == 0]].sum()       # newly covered live copper/brass
        if saving < min_saving:
            return None
        req_all = np.union1d(req, np.concatenate([circles[q]['req'] for q in red])) if red else req
        return saving, dict(c=c, r=rad, req=req_all, ids=ids, fixed=fixed_term(ids, rad), alive=True), red

    def partners(j):
        cj = circles[j]
        reach = slack * (cj['r'] + rmax)
        cand = set(ctree.query_ball_point(cj['c'], reach))
        for cell in cells(cj['c'], reach):
            cand.update(grid.get(cell, ()))
        return [k for k in cand if k != j and circles[k]['alive'] and circles[k]['fixed'] > 0 and len(circles[k]['req'])
                and np.linalg.norm(circles[k]['c'] - cj['c']) <= slack * (cj['r'] + circles[k]['r'])]

    heap, seq = [], itertools.count()
    active = [j for j, x in enumerate(circles) if x['fixed'] > 0 and len(x['req'])]
    for n, j in enumerate(active):
        for k in partners(j):
            if k > j:
                ev = evaluate(j, k)
                if ev:
                    heapq.heappush(heap, (-ev[0], next(seq), j, k))
        if verbose and n % 10000 == 0:
            print(f'  scanned {n}/{len(active)}  candidates {len(heap)}', flush=True)
    total, merges = 0.0, 0
    while heap:
        neg, _, j, k = heapq.heappop(heap)
        if not (circles[j]['alive'] and circles[k]['alive']):
            continue
        ev = evaluate(j, k)                  # lazy re-evaluation
        if not ev:
            continue
        if ev[0] < -neg - 1e-6:
            heapq.heappush(heap, (-ev[0], next(seq), j, k))
            continue
        saving, new, red = ev
        for q in [j, k] + red:
            circles[q]['alive'] = False
            qi = circles[q]['ids']
            cnt[qi[live[qi]]] -= 1
            cover[qi[iron[qi]]] -= 1
        circles.append(new)
        m = len(circles) - 1
        cnt[new['ids'][live[new['ids']]]] += 1
        cover[new['ids'][iron[new['ids']]]] += 1
        assert (cover[iron] >= 1).all()
        for cell in cells(new['c'], new['r']):
            grid.setdefault(cell, []).append(m)
        for p in new['req'].tolist():
            by_req.setdefault(p, []).append(m)
        total += saving
        merges += 1
        for q in partners(m):
            ev = evaluate(m, q)
            if ev:
                heapq.heappush(heap, (-ev[0], next(seq), m, q))
        if verbose and merges % 500 == 0:
            print(f'  merges {merges}  claimed saving {total/1e6:.2f}M  heap {len(heap)}', flush=True)
    out = [x for x in circles if x['alive']]
    if verbose:
        print(f'merged {merges} times; claimed no-leak saving {total/1e6:.2f}M; circles {len(out)}')
    return out


def main():
    pipes, a, b = sim.load()
    lengths = pipes.len.to_numpy()
    p_ml, sev = team_inputs(pipes)
    value = np.clip(p17b(pipes, p_ml) * (sev - 200 * lengths), 0, None)
    year, executes = planned_years(BASE, pipes, a, b, lengths)
    xy, r = sim.read_units(BASE)
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    OUT.mkdir(exist_ok=True)
    # A wider search (slack 2.0, 16 pipes) gave the identical result; a second pass (-0.23M
    # construction) and a one-year gap (-1.37M) both scored lower publicly, so this is final.
    variants = {'20a_sameyear_merge_17b_order': dict(slack=1.3, max_pipes=10)}
    for name, kw in variants.items():
        circles = merge_same_year(pipes, a, b, lengths, xy, r, year, executes, **kw)
        lines = [f'"({x["c"][0]:.9f}, {x["c"][1]:.9f})",{x["r"]:.9f}' for x in circles]
        tmp = OUT / f'{name}.circles.csv'
        tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        cxy, cr = sim.read_units(tmp)
        tmp.unlink()        # geometry-only file: never submit it unordered
        inside = sim.containment(cxy, cr, a, b, lengths)
        dummy = np.zeros((len(pipes), len(sim.YEARS)))
        order = so.order_dynamic(inside, cr, pipes, lengths, value_by_year=dummy, static_value=value, blend=1.0)
        assert sorted(order) == list(range(len(lines)))
        path = OUT / f'{name}.csv'
        path.write_text('\n'.join(lines[k] for k in order) + '\n', encoding='utf-8')
        rep = validate(path, pipes, history, a, b, (a + b) / 2, lengths, pipes.Material.isin(IRON).to_numpy())
        print(path.name, f"units {rep['work_unit_count']}  uncovered iron {rep['uncovered_iron_count']}  "
                         f"no-leak construction {rep['construction_cost']/1e6:.2f}M  finish {rep['completion_year']}", flush=True)


if __name__ == '__main__':
    main()
