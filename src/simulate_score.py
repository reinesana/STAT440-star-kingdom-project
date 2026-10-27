"""Monte Carlo estimate of the official score for one or more submission CSVs.

Leak model: each unreplaced metal pipe leaks at most once, with an annual hazard
that depends only on material and age (0.5-year bins), estimated from 2019-2026.
Leak costs are bootstrapped from historical leaks of the same material and surface.

Run from repository root: python src/simulate_score.py SUB.csv [SUB2.csv ...] [--sims N]
"""
from pathlib import Path
import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_baseline import RATES, IRON

ROOT = Path(__file__).resolve().parents[1]
YEARS = np.arange(2027, 2052)
BIN = 0.5


def load():
    pipes = pd.read_csv(ROOT / 'data/pipes.csv', dtype={'Pipe ID': str})
    hist = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    pipes['lay'] = pd.to_datetime(pipes['Lay date'])
    pipes['mat'] = pipes.Material.fillna('unknown')
    a = pipes[['GPS x1', 'GPS y1']].to_numpy(float)
    b = pipes[['GPS x2', 'GPS y2']].to_numpy(float)
    pipes['len'] = np.linalg.norm(b - a, axis=1)
    leak = hist.set_index('Pipe ID')
    pipes['leak_date'] = pd.to_datetime(pipes['Pipe ID'].map(leak.Date))
    pipes['leak_cost'] = pipes['Pipe ID'].map(leak.Cost)
    return pipes, a, b


def lay_years(pipes):
    return pipes.lay.dt.year + (pipes.lay.dt.dayofyear - 1) / 365.25


def fit_hazard(pipes):
    """Annual hazard table indexed by (material, age bin) using mid-year age."""
    lay = lay_years(pipes).to_numpy()
    leak_year = pipes.leak_date.dt.year.to_numpy(float)
    mats = pipes.mat.to_numpy()
    rows = []
    for y in range(2019, 2027):
        alive = np.isnan(leak_year) | (leak_year >= y)
        age = y + 0.5 - lay[alive]
        rows.append(pd.DataFrame({'mat': mats[alive], 'bin': np.floor(age / BIN).astype(int),
                                  'k': leak_year[alive] == y}))
    d = pd.concat(rows)
    pooled = d[d.mat.isin(IRON | {'brass', 'copper'})].groupby('bin').k.agg(['sum', 'size'])
    table = {}
    for m, g in d.groupby('mat'):
        s = g.groupby('bin').k.agg(['sum', 'size'])
        # Shrink sparse material-specific bins toward the pooled metal hazard.
        s = s.join(pooled, rsuffix='_p', how='outer').fillna(0)
        prior = s['sum_p'] / s['size_p'].clip(lower=1)
        table[m] = ((s['sum'] + 20 * prior) / (s['size'] + 20)).to_dict()
    pooled_h = (pooled['sum'] / pooled['size']).to_dict()
    return table, pooled_h


def hazard_matrix(pipes, table, pooled, tail_start=68.0, tail_rate=None, cap=0.6):
    """P(leak in calendar year Y | not yet leaked) for every pipe x project year.

    Ages past tail_start follow a Gompertz curve fitted to the observed old-age
    hazard; tail_rate overrides the fitted log-slope per year of age.
    """
    lay = lay_years(pipes).to_numpy()
    lay = np.where(np.isnan(lay), np.nanmedian(lay), lay)
    age = YEARS[None, :] + 0.5 - lay[:, None]
    bins = np.floor(age / BIN).astype(int)
    lo, hi = min(pooled), max(pooled)
    old = {k * BIN: v for k, v in pooled.items() if k * BIN >= 68 and v > 0}
    fitted, icpt = np.polyfit(list(old), np.log(list(old.values())), 1)
    slope = fitted if tail_rate is None else tail_rate
    anchor = np.exp(icpt + fitted * tail_start)
    gompertz = np.minimum(cap, anchor * np.exp(slope * (age - tail_start)))
    h = np.zeros(bins.shape)
    mats = pipes.mat.to_numpy()
    for m in pipes.mat.unique():
        rows = np.where(mats == m)[0]
        tab = table.get(m, pooled)
        for i in rows:
            h[i] = [tab.get(k, pooled.get(k, pooled[lo])) for k in np.clip(bins[i], lo, hi)]
    h = np.where(age >= tail_start, gompertz, h)
    h[mats == 'polyurethane'] = 0
    return h


WAVE = (60.0, 67.5)


def cohort_background(pipes, recent=(2023, 2026), shrink=200):
    """Per (material, lay year) leak rate in `recent` years at ages outside WAVE."""
    lay = lay_years(pipes).to_numpy()
    lay = np.where(np.isnan(lay), np.nanmedian(lay), lay)
    mats = pipes.mat.to_numpy()
    ly = np.floor(lay).astype(int)
    leak_year = pipes.leak_date.dt.year.to_numpy(float)
    k = np.zeros(len(pipes))
    n = np.zeros(len(pipes))
    for y in range(recent[0], recent[1] + 1):
        alive = np.isnan(leak_year) | (leak_year >= y)
        use = alive & ~((y + 0.5 - lay >= WAVE[0]) & (y + 0.5 - lay < WAVE[1]))
        n += use
        k += use & (leak_year == y)
    df = pd.DataFrame({'m': mats, 'ly': ly, 'k': k, 'n': n})
    mat_rate = df.groupby('m').k.sum() / df.groupby('m').n.sum().clip(lower=1)
    coh = df.groupby(['m', 'ly'])[['k', 'n']].sum()
    prior = mat_rate.reindex(coh.index.get_level_values(0)).to_numpy()
    coh['rate'] = (coh.k + shrink * prior) / (coh.n + shrink)
    return coh.rate.reindex(pd.MultiIndex.from_arrays([mats, ly])).fillna(0).to_numpy()


def fit_drift_wave(pipes, lay_range=(1952, 1967)):
    """MLE of a normal wear-out wave whose mean age drifts linearly with lay date.

    Wave time = lay + A + B*(lay-1960) + C*[copper] + sigma*Z, reached by a share q
    of pipes; leaks at ages 59-68 are wave events, other leaks censor.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm
    lay = lay_years(pipes).to_numpy()
    lt = (pipes.leak_date.dt.year + (pipes.leak_date.dt.dayofyear - 1) / 365.25).to_numpy()
    use = pipes.mat.isin(IRON | {'brass', 'copper'}).to_numpy() & (lay >= lay_range[0]) & (lay <= lay_range[1])
    lay, lt = lay[use], lt[use]
    cop = pipes.mat.eq('copper').to_numpy()[use].astype(float)
    leak = ~np.isnan(lt)
    wave = leak & (lt - lay >= 59) & (lt - lay < 68)
    cens = np.where(leak & ~wave, lt, 2027.0)

    def nll(th):
        A, B, C, ls, lq = th
        s, q = np.exp(ls), 1 / (1 + np.exp(-lq))
        mu = lay + A + B * (lay - 1960) + C * cop
        s0 = 1 - q * norm.cdf((2019 - mu) / s)
        ll = np.where(wave, np.log(q * norm.pdf((lt - mu) / s) / s + 1e-300),
                      np.log(np.clip(1 - q * norm.cdf((cens - mu) / s), 1e-300, None)))
        return -(ll - np.log(np.clip(s0, 1e-300, None))).sum()

    best = min((minimize(nll, [a0, b0, 0.5, np.log(0.6), 1.0], method='Nelder-Mead',
                         options=dict(maxiter=6000, xatol=1e-5, fatol=1e-4))
                for a0 in (62, 63, 64) for b0 in (-0.4, -0.2, 0.0)), key=lambda r: r.fun)
    A, B, C, ls, lq = best.x
    return dict(A=A, B=B, C=C, sigma=float(np.exp(ls)), q=float(1 / (1 + np.exp(-lq))))


def drift_hazard(pipes, wave, bg_scale=1.0, slope_scale=1.0):
    """Annual hazard from the drifting wave combined with the cohort background."""
    from scipy.stats import norm
    lay = lay_years(pipes).to_numpy()
    lay = np.where(np.isnan(lay), np.nanmedian(lay), lay)
    mats = pipes.mat.to_numpy()
    mu = lay + wave['A'] + slope_scale * wave['B'] * (lay - 1960) + wave['C'] * (mats == 'copper')
    edges = np.append(YEARS, YEARS[-1] + 1)[None, :]
    F = wave['q'] * norm.cdf((edges - mu[:, None]) / wave['sigma'])
    hw = (F[:, 1:] - F[:, :-1]) / np.clip(1 - F[:, :-1], 1e-12, None)
    bg = cohort_background(pipes) * bg_scale
    h = 1 - (1 - hw) * (1 - bg[:, None])
    h[mats == 'polyurethane'] = 0
    h[pipes.mat.eq('unknown').to_numpy()] = bg[pipes.mat.eq('unknown').to_numpy(), None]
    return np.minimum(h, 0.95)


def cohort_hazard(pipes, table, pooled, recent=(2023, 2026), shrink=200, growth=0.0):
    """Age-wave excess hazard plus a per (material, lay year) background rate.

    The background is the cohort's observed leak rate in `recent` years at ages
    outside WAVE, shrunk toward the material's rate; `growth` compounds it yearly.
    """
    lay = lay_years(pipes).to_numpy()
    lay = np.where(np.isnan(lay), np.nanmedian(lay), lay)
    mats = pipes.mat.to_numpy()
    ly = np.floor(lay).astype(int)
    leak_year = pipes.leak_date.dt.year.to_numpy(float)
    k = np.zeros(len(pipes))
    n = np.zeros(len(pipes))
    for y in range(recent[0], recent[1] + 1):
        alive = np.isnan(leak_year) | (leak_year >= y)
        out_wave = ~((y + 0.5 - lay >= WAVE[0]) & (y + 0.5 - lay < WAVE[1]))
        use = alive & out_wave
        n += use
        k += use & (leak_year == y)
    df = pd.DataFrame({'m': mats, 'ly': ly, 'k': k, 'n': n})
    mat_rate = df.groupby('m').k.sum() / df.groupby('m').n.sum().clip(lower=1)
    coh = df.groupby(['m', 'ly'])[['k', 'n']].sum()
    prior = mat_rate.reindex(coh.index.get_level_values(0)).to_numpy()
    coh['rate'] = (coh.k + shrink * prior) / (coh.n + shrink)
    bg = coh.rate.reindex(pd.MultiIndex.from_arrays([mats, ly])).fillna(0).to_numpy()
    base_level = np.median([v for b, v in pooled.items() if not WAVE[0] <= b * BIN < WAVE[1]])
    age = YEARS[None, :] + 0.5 - lay[:, None]
    bins = np.clip(np.floor(age / BIN).astype(int), min(pooled), max(pooled))
    excess = np.vectorize(lambda b: max(pooled.get(b, 0) - base_level, 0))(bins)
    excess = np.where((age >= WAVE[0]) & (age < WAVE[1]), excess, 0)
    years_out = np.arange(len(YEARS))[None, :]
    h = np.minimum(0.95, excess + bg[:, None] * (1 + growth) ** years_out)
    h[mats == 'polyurethane'] = 0
    return h


def cost_pools(pipes):
    leaked = pipes.dropna(subset=['leak_cost'])
    group = lambda df: df.mat.where(df.mat.isin(IRON), df.mat).replace(dict.fromkeys(IRON, 'iron'))
    pools = {k: g.leak_cost.to_numpy() for k, g in leaked.groupby([group(leaked), leaked.Surface])}
    by_surface = {s: g.leak_cost.to_numpy() for s, g in leaked.groupby('Surface')}
    keys = list(zip(group(pipes), pipes.Surface))
    return [pools.get(k, by_surface[k[1]]) for k in keys]


def pipe_cost_pools(pipes, pools, strength=1.0):
    """Rescale each pipe's surface/material cost pool toward the team's per-pipe
    severity prediction (ensemble_stable.csv); strength 0 keeps the pooled mean."""
    pred = pd.read_csv(ROOT / 'notebooks/ensemble_for_cost_prediction/predictions/ensemble_stable.csv')
    pred = pipes['Pipe ID'].map(pred.set_index('id').predicted_cost).to_numpy()
    out = []
    for pool, pr in zip(pools, pred):
        m = pool.mean()
        out.append(pool * (pr / m) ** strength if np.isfinite(pr) and pr > 0 else pool)
    return out


def read_units(path):
    s = pd.read_csv(path, header=None)
    xy = s[0].str.strip('()').str.split(',', expand=True).astype(float).to_numpy()
    return xy, s[1].to_numpy(float)


def containment(xy, r, a, b, lengths):
    mid = (a + b) / 2
    tree = cKDTree(mid)
    out = []
    for c, rad in zip(xy, r):
        ids = np.asarray(tree.query_ball_point(c, rad + 1e-9), dtype=int)
        ok = (np.linalg.norm(a[ids] - c, axis=1) <= rad) & (np.linalg.norm(b[ids] - c, axis=1) <= rad)
        out.append(ids[ok])
    return out


def simulate(inside, r, pipes, lengths, h, pools, sims, seed=0, lag=0):
    rng = np.random.default_rng(seed)
    surf = pipes.Surface.to_numpy()
    area_rate = np.array([sum(RATES[s] for s in set(surf[ids])) * (math.pi * rad**2)**0.85
                          for ids, rad in zip(inside, r)])
    surface_term = area_rate  # 2026-10-26: exponent applies to area only
    start_poly = (pipes.mat.eq('polyurethane') | pipes.leak_date.notna()).to_numpy()
    metal = ~pipes.mat.eq('polyurethane').to_numpy()
    iron = pipes.mat.isin(IRON).to_numpy()
    n = len(pipes)
    results = []
    for _ in range(sims):
        u = rng.random(h.shape)
        draw = rng.random(n)
        leak_cost = np.array([pool[int(d * len(pool))] for pool, d in zip(pools, draw)])
        hit = (u < h) & ~start_poly[:, None]
        first = np.where(hit.any(1), hit.argmax(1), len(YEARS))
        replaced_year = np.full(n, len(YEARS))
        poly = start_poly.copy()
        budget = construction = 0.0
        cursor = 0
        for yi in range(len(YEARS)):
            budget += 10_000_000
            poly |= first < yi
            while cursor < len(inside):
                ids = inside[cursor]
                rep = ids[~poly[ids]]
                if not len(rep):
                    cursor += 1
                    continue
                cost = round(100 + surface_term[cursor] + 200 * lengths[rep].sum(), 2)
                if cost > budget:
                    break
                budget -= cost
                construction += cost
                poly[rep] = True
                replaced_year[rep] = np.minimum(replaced_year[rep], yi)
                cursor += 1
            if cursor == len(inside):
                break
        feasible = cursor == len(inside)
        leaks = np.where((first < replaced_year + lag) & (first < len(YEARS)) & metal)[0]
        repair = leak_cost[leaks].sum()
        results.append(dict(cost_model_version='2026-10-26-area-exponent', construction=construction, repair=repair, leaks=len(leaks),
                            score=250_000_000 - construction - repair, feasible=feasible,
                            iron_leaks=int(iron[leaks].sum()), finish=2027 + yi))
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subs', nargs='+')
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--tail-rate', type=float, default=None)
    args = ap.parse_args()
    pipes, a, b = load()
    lengths = pipes.len.to_numpy()
    table, pooled = fit_hazard(pipes)
    h = hazard_matrix(pipes, table, pooled, tail_rate=args.tail_rate)
    pools = cost_pools(pipes)
    for path in args.subs:
        xy, r = read_units(path)
        inside = containment(xy, r, a, b, lengths)
        df = simulate(inside, r, pipes, lengths, h, pools, args.sims)
        q = df.score.quantile([.1, .5, .9]) / 1e6
        print(f'{Path(path).name:45s} mean {df.score.mean()/1e6:7.2f}M  median {q[.5]:7.2f}  '
              f'p10 {q[.1]:7.2f}  p90 {q[.9]:7.2f}  constr {df.construction.mean()/1e6:6.2f}  '
              f'repair {df.repair.mean()/1e6:6.2f}  leaks {df.leaks.mean():6.0f}  '
              f'finish {df.finish.max()}  feasible {df.feasible.mean():.2f}')


if __name__ == '__main__':
    main()
