"""Past-only annual landmarks; no cost-model feature tables are trusted inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
SEED = 440


def raw():
    p = pd.read_csv(REPO / 'data/pipes.csv')
    h = pd.read_csv(REPO / 'data/train.csv')
    h['Date'] = pd.to_datetime(h.Date)
    p['lay'] = pd.to_datetime(p['Lay date'])
    assert p['Pipe ID'].is_unique and h['Pipe ID'].is_unique
    assert set(h['Pipe ID']) <= set(p['Pipe ID'])
    assert p.lay.dropna().lt('2019-01-01').all()
    assert h.Date.between('2019-01-01', '2026-12-31').all()
    assert not set(h['Pipe ID']) & set(p.loc[p.Material.eq('polyurethane'), 'Pipe ID'])
    return p, h


class Features:
    def __init__(self, p, h):
        self.p, self.h = p.copy(), h.copy()
        self.ids = p['Pipe ID'].to_numpy()
        a = p[['GPS x1', 'GPS y1']].to_numpy(float)
        b = p[['GPS x2', 'GPS y2']].to_numpy(float)
        mid = (a + b) / 2
        delta = b - a
        length = np.linalg.norm(delta, axis=1)
        assert np.isfinite(mid).all() and (length > 0).all()
        f = pd.DataFrame(dict(material=p.Material.fillna('__unknown__'), surface=p.Surface,
                              lay_year=p.lay.dt.year, lay_missing=p.lay.isna().astype(int),
                              length_m=length, log_length=np.log1p(length),
                              mid_x=mid[:, 0], mid_y=mid[:, 1],
                              abs_dx=np.abs(delta[:, 0]), abs_dy=np.abs(delta[:, 1]),
                              orientation_sin=delta[:, 1] / length,
                              orientation_cos=delta[:, 0] / length))
        self.groups = {'basic': ['material', 'surface', 'length_m', 'log_length', 'age', 'lay_missing'],
                       'cohort': ['lay_year', 'lay_decade', 'material_cohort'],
                       'space': ['mid_x', 'mid_y', 'abs_dx', 'abs_dy', 'orientation_sin', 'orientation_cos'],
                       'topology': [], 'neighbors': [], 'history': [], 'calendar': ['origin_year']}
        f['lay_decade'] = (p.lay.dt.year // 10).fillna(-1).astype(int).astype(str)
        f['material_cohort'] = f.material + '_' + p.lay.dt.year.fillna(-1).astype(int).astype(str)
        tree = cKDTree(mid)
        self.adj = {}
        for radius in (50, 100, 250):
            neighborhoods = tree.query_ball_point(mid, radius)
            row = np.repeat(np.arange(len(p)), [len(x) for x in neighborhoods])
            col = np.concatenate(neighborhoods)
            keep = row != col
            adj = csr_matrix((np.ones(keep.sum()), (row[keep], col[keep])), shape=(len(p), len(p)))
            self.adj[radius] = adj
            n = np.asarray(adj.sum(axis=1)).ravel()
            denom = np.maximum(n, 1)
            values = {'count': n, 'mean_length_m': adj @ length / denom}
            for mat in sorted(f.material.unique()):
                values['material_' + mat.replace(' ', '_') + '_fraction'] = adj @ f.material.eq(mat).to_numpy(float) / denom
            for surface in sorted(f.surface.unique()):
                values['surface_' + surface + '_fraction'] = adj @ f.surface.eq(surface).to_numpy(float) / denom
            for name, values_ in values.items():
                key = f'neighbor_{radius}m_{name}'
                f[key] = values_
                self.groups['neighbors'].append(key)
            for name in ('prior_leak_count', 'last1_leak_count', 'last2_leak_count', 'prior_leak_fraction'):
                self.groups['history'].append(f'history_{radius}m_{name}')
        # Exact endpoint equality is only a geometric adjacency proxy.
        endpoints = {}
        for i, pair in enumerate(zip(a, b)):
            for point in pair:
                endpoints.setdefault(tuple(point), []).append(i)
        for end, points in enumerate((a, b), 1):
            counts = np.array([len(endpoints[tuple(x)]) - 1 for x in points])
            for suffix, values in [('degree', counts), ('terminal', counts == 0), ('branch', counts > 1)]:
                key = f'endpoint{end}_{suffix}'
                f[key] = values.astype(float)
                self.groups['topology'].append(key)
        for name in ('nearest_prior_leak_m', 'no_prior_leak', 'cohort_prior_rate', 'material_prior_rate'):
            self.groups['history'].append(name)
        self.static, self.mid = f, mid

    def at(self, origin):
        f = self.static.copy()
        f['age'] = (pd.Timestamp(f'{origin}-01-01') - self.p.lay).dt.days / 365.2425
        f['origin_year'] = origin
        # This is the sole history boundary; events at/after origin never enter X.
        past = self.h.loc[self.h.Date < pd.Timestamp(f'{origin}-01-01')]
        prior = np.isin(self.ids, past['Pipe ID']).astype(float)
        last1 = np.isin(self.ids, past.loc[past.Date.dt.year >= origin - 1, 'Pipe ID']).astype(float)
        last2 = np.isin(self.ids, past.loc[past.Date.dt.year >= origin - 2, 'Pipe ID']).astype(float)
        for radius, adj in self.adj.items():
            for name, value in [('prior_leak_count', prior), ('last1_leak_count', last1), ('last2_leak_count', last2)]:
                f[f'history_{radius}m_{name}'] = adj @ value
            n = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)
            f[f'history_{radius}m_prior_leak_fraction'] = adj @ prior / n
        if prior.sum():
            f['nearest_prior_leak_m'] = cKDTree(self.mid[prior.astype(bool)]).query(self.mid)[0]
        else:
            f['nearest_prior_leak_m'] = np.nan
        f['no_prior_leak'] = int(not prior.sum())
        # Smoothed historical group fractions, not future cohort rates.
        for name, key in [('cohort_prior_rate', 'material_cohort'), ('material_prior_rate', 'material')]:
            group = pd.DataFrame({'key': f[key], 'prior': prior}).groupby('key').prior.agg(['sum', 'count'])
            rate = (group['sum'] + prior.mean() * 20) / (group['count'] + 20)
            f[name] = f[key].map(rate)
        f['pipe_id'] = self.ids
        return f.loc[(prior == 0) & ~self.p.Material.eq('polyurethane')].reset_index(drop=True)


def prepare(force=False):
    cache = BASE / 'cache'
    cache.mkdir(exist_ok=True)
    if (cache / 'features.parquet').exists() and not force:
        return
    p, h = raw()
    builder = Features(p, h)
    features = pd.concat([builder.at(y) for y in range(2019, 2028)], keys=range(2019, 2028), names=['origin', 'row']).reset_index(level=0).reset_index(drop=True)
    features.to_parquet(cache / 'features.parquet', index=False)
    h[['Pipe ID', 'Date']].to_parquet(cache / 'events.parquet', index=False)
    (BASE / 'feature_groups.json').write_text(json.dumps(builder.groups, indent=2), encoding='utf8')
    # Full feature mutation test: deleting/replacing current+future outcomes
    # must not change eligible IDs or any feature at the tested origin.
    checks = []
    for origin in (2021, 2023, 2026, 2027):
        original = builder.at(origin)
        builder.h = h.loc[h.Date < pd.Timestamp(f'{origin}-01-01')].copy()
        mutated = builder.at(origin)
        pd.testing.assert_frame_equal(original, mutated)
        builder.h = h
        checks.append({'origin': origin, 'future_event_deletion_invariant': True, 'rows': len(original)})
    audit = dict(rows_inventory=len(p), rows_events=len(h), missing_lay=int(p.lay.isna().sum()),
                 missing_material=int(p.Material.isna().sum()), native_polyurethane=int(p.Material.eq('polyurethane').sum()),
                 events_unique=True, all_inventory_pre2019=True, units='metres', future_mutation_checks=checks,
                 input_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [REPO / 'data/pipes.csv', REPO / 'data/train.csv']})
    (BASE / 'input_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')


class Dataset:
    def __init__(self):
        prepare()
        self.f = pd.read_parquet(BASE / 'cache/features.parquet')
        self.events = pd.read_parquet(BASE / 'cache/events.parquet').set_index('Pipe ID').Date
        self.groups = json.loads((BASE / 'feature_groups.json').read_text())

    def train(self, cutoff):
        f = self.f.loc[self.f.origin <= cutoff].copy().reset_index(drop=True)
        # Only dates <= cutoff are visible to the training target constructor.
        known = self.events.loc[self.events.dt.year <= cutoff]
        event_year = f.pipe_id.map(known).dt.year
        event = event_year.notna().to_numpy()
        duration = np.where(event, event_year - f.origin + 0.5, cutoff + 1 - f.origin).astype(float)
        assert (duration > 0).all()
        return f, duration, event

    def forecast(self, origin):
        return self.f.loc[self.f.origin == origin].copy().reset_index(drop=True)

    def labels(self, frame, origin, horizon=1):
        year = frame.pipe_id.map(self.events).dt.year
        return year.between(origin, origin + horizon - 1).fillna(False).to_numpy(int)
