"""Append historical candidates and endpoint-route features, preserving original cells.

python src/enrich_cost_features.py [--legacy-features /path/to/features.csv]
The optional cache must come from the archived experiment with identical inputs.
Without a cache, rebuild the archived features in an isolated temporary directory.
Requires numpy, pandas, scipy; rebuilding legacy candidates also requires the
versions in notebooks/Takito_Kobayashi/requirements.txt.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
ANGLE_DEG = 10.0
CURVE_TOTAL_DEG = 30.0
RADII = [25, 100, 250]
IRON = {'cast iron', 'gray iron', 'wrought iron'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_rows(path, original_columns):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    ix = [rows[0].index(c) for c in original_columns]
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\r\n').writerows([[r[i] for i in ix] for r in rows])
    return stream.getvalue().encode('utf-8')


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segment_distance(a, b, c, d):
    def point(q, x, y):
        v = y - x
        t = np.clip(np.sum((q - x) * v, axis=-1) / np.sum(v * v, axis=-1), 0, 1)
        return np.linalg.norm(q - (x + t[..., None] * v), axis=-1)
    dist = np.minimum.reduce([point(a, c, d), point(b, c, d), point(c, a, b), point(d, a, b)])
    r, s, delta = b - a, d - c, c - a
    den = cross(r, s)
    safe = np.where(np.abs(den) < 1e-9, 1.0, den)
    t, u = cross(delta, s) / safe, cross(delta, r) / safe
    hit = (np.abs(den) >= 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    dist[hit] = 0
    return dist


def route_features(p):
    """Maximal nonbranching endpoint chains; no interior-crossing connections.

    All known inventory lay dates precede 2019. Unknown-date pipes are excluded
    from historical topology, retained as ineligible rows in the inventory file.
    """
    dates = pd.to_datetime(p['Lay date'], errors='raise')
    assert dates.dropna().lt(pd.Timestamp('2019-01-01')).all()
    a = p[['GPS x1', 'GPS y1']].to_numpy(float)
    b = p[['GPS x2', 'GPS y2']].to_numpy(float)
    lengths = np.linalg.norm(b - a, axis=1)
    assert (lengths > 0).all()
    ids = p['Pipe ID'].to_numpy()
    nodes = [(tuple(x), tuple(y)) for x, y in zip(a, b)]
    incident = defaultdict(list)
    for i in np.flatnonzero(dates.notna()):
        for node in nodes[i]:
            incident[node].append(i)
    parent = list(range(len(p)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    turns = {}
    for node, edges in incident.items():
        if len(edges) == 2:
            i, j = edges
            parent[find(j)] = find(i)
            vi = b[i] - a[i] if nodes[i][0] == node else a[i] - b[i]
            vj = b[j] - a[j] if nodes[j][0] == node else a[j] - b[j]
            turns[node] = float(np.degrees(np.arccos(np.clip(-vi @ vj / lengths[i] / lengths[j], -1, 1))))
    groups = defaultdict(list)
    for i in np.flatnonzero(dates.notna()):
        groups[find(i)].append(i)
    records = {}
    summary = []
    for edges in groups.values():
        group = set(edges)
        ends = sorted({n for i in edges for n in nodes[i] if len(incident[n]) != 2})
        closed = not ends
        start = ends[0] if ends else min(n for i in edges for n in nodes[i])
        route_id = 'R_' + min(ids[edges])
        node = start
        used = set()
        ordered, vectors, positions = [], [], []
        distance = 0.0
        while len(used) < len(edges):
            choices = sorted((i for i in incident[node] if i in group and i not in used), key=lambda i: ids[i])
            assert choices, 'Broken route traversal'
            i = choices[0]
            other = nodes[i][1] if nodes[i][0] == node else nodes[i][0]
            ordered.append(i)
            vectors.append((np.array(other) - np.array(node)) / lengths[i])
            positions.append(distance + lengths[i] / 2)
            distance += lengths[i]
            used.add(i)
            node = other
        route_turns = [turns[n] for n in {n for i in edges for n in nodes[i]} if n in turns]
        total_turn = float(sum(route_turns))
        end_distance = float(np.linalg.norm(np.array(node) - np.array(start)))
        tortuosity = distance / end_distance if end_distance > 1e-9 else np.nan
        curved = int(closed or total_turn >= CURVE_TOTAL_DEG or (np.isfinite(tortuosity) and tortuosity > 1.01))
        # Split straight sections by accumulated deviation from their first edge,
        # even when successive local bends are individually below 10 degrees.
        sections, section = [], []
        reference = None
        for i, vec in zip(ordered, vectors):
            angle = 0 if reference is None else float(np.degrees(np.arccos(np.clip(reference @ vec, -1, 1))))
            if section and angle > ANGLE_DEG:
                sections.append(section)
                section = []
                reference = None
            if reference is None:
                reference = vec
            section.append(i)
        sections.append(section)
        section_map = {}
        for section in sections:
            tag = 'S_' + min(ids[section])
            for i in section:
                section_map[i] = (tag, len(section), float(lengths[section].sum()))
        for k, i in enumerate(ordered):
            local = [turns[n] for n in nodes[i] if n in turns]
            angle = max(local) if local else np.nan
            tag, count, section_length = section_map[i]
            records[i] = {
                'pipe_id': ids[i], 'topology_eligible': 1,
                'route_id': route_id, 'straight_section_id': tag,
                'endpoint_degree_1': len(incident[nodes[i][0]]),
                'endpoint_degree_2': len(incident[nodes[i][1]]),
                'is_branch_adjacent': int(any(len(incident[n]) >= 3 for n in nodes[i])),
                'has_continuation': int(bool(local)), 'turn_angle': angle,
                'is_bend': int(bool(local) and angle > ANGLE_DEG),
                'is_straight': int(bool(local) and angle <= ANGLE_DEG),
                'route_is_curved': curved, 'route_is_closed': int(closed),
                'route_length': distance, 'route_pipe_count': len(edges),
                'route_total_turn_deg': total_turn, 'route_tortuosity': tortuosity,
                'route_position': positions[k], 'route_position_fraction': positions[k] / distance,
                'straight_section_pipe_count': count, 'straight_section_length': section_length,
            }
        summary.append({'route_id': route_id, 'route_length': distance, 'route_pipe_count': len(edges),
                        'route_is_curved': curved, 'route_is_closed': int(closed),
                        'route_total_turn_deg': total_turn, 'route_tortuosity': tortuosity,
                        'straight_section_count': len(sections)})
    result = pd.DataFrame([records.get(i, {'pipe_id': ids[i], 'topology_eligible': 0}) for i in range(len(p))])
    return result, pd.DataFrame(summary).sort_values('route_id')


def legacy_candidates(base_bytes, cache):
    package = ROOT / 'notebooks/Takito_Kobayashi'
    hashes = json.loads((package / 'source_hashes.json').read_text())
    assert hashlib.sha256(base_bytes).hexdigest() == hashes['train_predict_cost.csv']
    assert sha(ROOT / 'data/pipes.csv') == hashes['pipes.csv']
    if cache:
        cache = cache.resolve()
        manifest = json.loads((cache.parent / 'feature_manifest.json').read_text())
        assert manifest['sha256']['train_predict_cost.csv'] == hashes['train_predict_cost.csv']
        assert manifest['supplemental_inventory']['sha256'] == hashes['pipes.csv']
        frame = pd.read_csv(cache)
    else:
        with tempfile.TemporaryDirectory(prefix='stat440_features_') as tmp:
            workspace = Path(tmp)
            for folder in ['model_comparison', 'lightgbm_cost_model']:
                shutil.copytree(package / 'executed_code' / folder, workspace / folder)
            data = workspace / 'github-main/data'
            data.mkdir(parents=True)
            (data / 'train_predict_cost.csv').write_bytes(base_bytes)
            shutil.copy2(ROOT / 'data/pipes.csv', data / 'pipes.csv')
            scripts = workspace / 'model_comparison'
            for script, args in [
                ('run_experiment.py', ['--prepare', '--source', str(data / 'train_predict_cost.csv')]),
                ('expand_features.py', []), ('prepare_extra_spatial.py', []), ('prepare_segment_cost.py', [])]:
                subprocess.run([sys.executable, str(scripts / script), *args], check=True)
            frame = pd.read_csv(scripts / 'features.csv')
            extra = pd.read_csv(scripts / 'extra_spatial_candidates.csv')
            for col in extra:
                if col != 'pipe_id' and col not in frame:
                    frame[col] = extra.set_index('pipe_id')[col].reindex(frame.pipe_id).to_numpy()
    expected = pd.read_csv(package / 'results/feature_catalogue.csv').feature.tolist()
    assert set(expected) <= set(frame), 'Incomplete legacy cache'
    original = pd.read_csv(io.BytesIO(base_bytes))
    assert frame.pipe_id.is_unique
    matched = frame.set_index('pipe_id').reindex(original.pipe_id)
    assert np.allclose(matched.cost, original.cost)
    assert matched.date.to_list() == original.date.to_list()
    return frame[['pipe_id'] + [c for c in expected if c != 'pipe_id']].copy()


def extra_candidates(d, p, history):
    """Earlier candidate families not all in the 470-column catalogue.

    Material state uses the full original train.csv, including rows excluded
    from severity training. All repair cutoffs are strictly before Jan 1.
    """
    a = p[['GPS x1', 'GPS y1']].to_numpy(float)
    b = p[['GPS x2', 'GPS y2']].to_numpy(float)
    lengths = np.linalg.norm(b - a, axis=1)
    mid = (a + b) / 2
    tree = cKDTree(mid)
    lay = pd.to_datetime(p['Lay date']).to_numpy()
    repair_dates = pd.to_datetime(history.Date)
    assert history['Pipe ID'].is_unique
    repaired_at = p['Pipe ID'].map(pd.Series(repair_dates.to_numpy(), index=history['Pipe ID'])).to_numpy(dtype='datetime64[ns]')
    surface = p.Surface.to_numpy()
    original_pu = p.Material.eq('polyurethane').to_numpy()
    iron = p.Material.isin(IRON).to_numpy()
    index = pd.Series(p.index, index=p['Pipe ID'])
    rows = []
    for n, row in enumerate(d.itertuples()):
        i = int(index[row.pipe_id])
        now = np.datetime64(row.date)
        cut = np.datetime64(f'{row.event_year}-01-01')
        ix = np.asarray(tree.query_ball_point(mid[i], (lengths[i] + lengths.max()) / 2 + 250 + 1e-8), int)
        ix = ix[(ix != i) & (lay[ix] <= now)]
        distances = segment_distance(a[i], b[i], a[ix], b[ix])
        midpoint_distances = np.linalg.norm(mid[ix] - mid[i], axis=1)
        r = {}
        for radius in [50, 100, 250]:
            j = ix[midpoint_distances <= radius]
            r[f'neighbor_count_{radius}'] = len(j)
            if radius == 100:
                r['neighbor_length_sum_100'] = float(lengths[j].sum())
                r['neighbor_road_count_100'] = int((surface[j] == 'road').sum())
        for radius in RADII:
            j = ix[distances <= radius + 1e-8]
            past = repaired_at[j] < cut
            recent = past & (repaired_at[j] >= cut - np.timedelta64(365, 'D'))
            pu = original_pu[j] | past
            remaining = iron[j] & ~past
            r.update({f'prior_repairs_{radius}m': int(past.sum()),
                      f'prior_repair_fraction_{radius}m': float(past.mean()) if len(j) else 0.0,
                      f'no_recorded_repair_{radius}m': int((~past).sum()),
                      f'current_pu_fraction_{radius}m': float(pu.mean()) if len(j) else 0.0,
                      f'remaining_iron_count_{radius}m': int(remaining.sum()),
                      f'remaining_iron_length_{radius}m': float(lengths[j][remaining].sum()),
                      f'repairs_last365d_{radius}m': int(recent.sum()),
                      f'days_since_neighbor_repair_{radius}m': float((cut - repaired_at[j][past].max()) / np.timedelta64(1, 'D')) if past.any() else np.nan})
        for radius in [25, 100]:
            j = ix[distances <= radius + 1e-8]
            r[f'wet_segments_{radius}m'] = int(np.isin(surface[j], ['water', 'swamp']).sum())
        rows.append(r)
        if n % 3000 == 0:
            print(f'Extra geometry/state: {n}/{len(d)}', flush=True)
    result = pd.DataFrame(rows)
    hours = pd.to_timedelta(d.time + ':00').dt.total_seconds() / 3600
    result['event_night'] = ((hours < 6) | (hours >= 22)).astype(int)
    return result


def route_history(d, topology, history):
    lookup = topology.set_index('pipe_id')
    full = history.copy()
    full['year'] = pd.to_datetime(full.Date).dt.year
    full['route_id'] = full['Pipe ID'].map(lookup.route_id)
    full['straight_section_id'] = full['Pipe ID'].map(lookup.straight_section_id)
    result = pd.DataFrame(index=d.index)
    for name in ['route', 'straight_section']:
        key = name + '_id'
        lengths = d[name + '_length']
        for year in sorted(d.event_year.unique()):
            selected = d.event_year == year
            past = full.loc[full.year < year]
            stats = past.groupby(key).agg(n=('Cost', 'size'), cost=('Cost', 'sum'), last=('Date', 'max'))
            recent = past.loc[past.year >= year - 1].groupby(key).size()
            q = d.loc[selected, key]
            n = q.map(stats.n).fillna(0)
            amount = q.map(stats.cost).fillna(0)
            result.loc[selected, name + '_prior_leak_count'] = n
            result.loc[selected, name + '_prior_leak_cost'] = amount
            result.loc[selected, name + '_last1_leak_count'] = q.map(recent).fillna(0)
            # Counts per kilometre are descriptive, not a probability/hazard.
            result.loc[selected, name + '_prior_leaks_per_km'] = n / (lengths[selected] / 1000)
            result.loc[selected, name + '_prior_cost_per_m'] = amount / lengths[selected]
            result.loc[selected, name + '_days_since_leak'] = (pd.Timestamp(f'{year}-01-01') - pd.to_datetime(q.map(stats['last']))).dt.days
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-features', type=Path)
    args = parser.parse_args()
    destination = ROOT / 'data/train_predict_cost.csv'
    metadata_path = ROOT / 'data/train_predict_cost_features_2026-10-05.json'
    if metadata_path.exists():
        original_columns = json.loads(metadata_path.read_text())['original_columns']
    else:
        original_columns = list(pd.read_csv(destination, nrows=0).columns)
    base_bytes = original_rows(destination, original_columns)
    base = pd.read_csv(io.BytesIO(base_bytes))
    assert base.pipe_id.is_unique and len(base) == 14107
    legacy = legacy_candidates(base_bytes, args.legacy_features).set_index('pipe_id').reindex(base.pipe_id).reset_index()
    # The archive stores x/y aliases and two duplicate age/log encodings. Keep
    # existing canonical columns instead; use canonical names for squares too.
    aliases = {'x1': 'gps_x1', 'x2': 'gps_x2', 'y1': 'gps_y1', 'y2': 'gps_y2',
               'years_used': 'years_used_at_leak', 'log_age': 'log1p_years_used',
               'log_length': 'log1p_pipe_length'}
    renamed = {'years_used_squared': 'years_used_at_leak_squared'}
    additions = []
    for col in legacy:
        if col in aliases:
            assert np.allclose(legacy[col], base[aliases[col]], equal_nan=True)
            continue
        if col in base:
            if pd.api.types.is_numeric_dtype(base[col]):
                assert np.allclose(legacy[col], base[col], equal_nan=True)
            else:
                assert legacy[col].equals(base[col])
            continue
        additions.append(legacy[[col]].rename(columns=renamed))
    p = pd.read_csv(ROOT / 'data/pipes.csv', dtype={'Pipe ID': str})
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Pipe ID': str})
    assert set(history['Pipe ID']) <= set(p['Pipe ID'])
    topology, summaries = route_features(p)
    route = topology.set_index('pipe_id').reindex(base.pipe_id).reset_index().drop(columns='pipe_id')
    assert route.topology_eligible.eq(1).all()
    extra = extra_candidates(base, p, history)
    result = pd.concat([base, *additions, extra, route], axis=1)
    rh = route_history(result, topology, history)
    result = pd.concat([result, rh], axis=1)
    assert result.columns.is_unique
    assert not np.isinf(result.select_dtypes(include=np.number).to_numpy()).any()
    new_columns = [c for c in result if c not in original_columns]
    # Preserve the original cells exactly, including original numeric strings.
    original = list(csv.reader(io.StringIO(base_bytes.decode('utf-8'))))
    values = result[new_columns].astype(object).where(result[new_columns].notna(), '').values
    temp = destination.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\r\n')
        writer.writerow(original_columns + new_columns)
        writer.writerows([r + list(v) for r, v in zip(original[1:], values)])
    assert original_rows(temp, original_columns) == base_bytes
    temp.replace(destination)
    topology.to_csv(ROOT / 'data/pipe_route_lookup.csv', index=False)
    summaries.to_csv(ROOT / 'data/pipe_route_summary.csv', index=False)
    pd.DataFrame({'pipe_id': result.pipe_id, 'event_year': result.event_year,
                  'route_id': result.route_id, 'straight_section_id': result.straight_section_id}).to_csv(ROOT / 'data/pipe_route_event_membership.csv', index=False)
    metadata = {
        'added_on': '2026-10-05', 'rows': len(result), 'original_columns': original_columns,
        'added_columns': new_columns, 'column_count': len(result.columns),
        'original_source_sha256': hashlib.sha256(base_bytes).hexdigest(),
        'output_sha256': sha(destination), 'pipes_sha256': sha(ROOT / 'data/pipes.csv'),
        'history_sha256': sha(ROOT / 'data/train.csv'),
        'legacy_candidate_count': 470, 'dropped_aliases': aliases, 'renamed_columns': renamed,
        'legacy_source': 'notebooks/Takito_Kobayashi/executed_code and results/feature_catalogue.csv',
        'legacy_cache_sha256': sha(args.legacy_features) if args.legacy_features else None,
        'history_cutoff': 'Strictly earlier calendar years, frozen January 1; never current-year costs/repairs.',
        'state_and_route_history_source': 'Full data/train.csv (including 6 excluded severity rows)',
        'legacy_cost_history_source': 'Original 14107-row train_predict_cost.csv',
        'topology': {'endpoint_tolerance_m': 0, 'connections': 'Exact equal endpoint coordinates only; no interior crossing connections',
                     'route_rule': 'Maximal chain through degree-2 nodes; split at branches/dead ends',
                     'straight_angle_threshold_deg': ANGLE_DEG, 'curve_total_turn_threshold_deg': CURVE_TOTAL_DEG,
                     'curve_tortuosity_threshold': 1.01, 'unknown_lay_date_policy': 'Excluded from historical topology',
                     'earliest_event_year': 2019, 'latest_known_inventory_lay_date': str(p['Lay date'].dropna().max()),
                     'route_count': len(summaries), 'straight_section_count': int(topology.straight_section_id.nunique()),
                     'excluded_unknown_lay_dates': int(topology.topology_eligible.eq(0).sum())},
        'notes': ['Units: meters; endpoint length is not excavated/curved length.',
                  'Route/section IDs are grouping keys, not ordered numeric predictors.',
                  'Bend flags refer to geometric direction changes, not observed fittings or physical connectivity.',
                  'Local is_straight and route_is_curved may both be 1 for a smooth curve.',
                  'Blank history values mean unavailable; zero counts mean no recorded events.',
                  'No model retraining or feature-selection claim; candidates require validation.'],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['rows', 'column_count', 'topology']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
