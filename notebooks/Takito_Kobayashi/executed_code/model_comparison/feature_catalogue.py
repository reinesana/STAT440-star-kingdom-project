"""Describe feature provenance and missingness without consulting test errors."""
import json
import pandas as pd
from pathlib import Path
OUT=Path(__file__).resolve().parent
m=json.loads((OUT/'feature_manifest.json').read_text());d=pd.read_csv(OUT/'features.csv');rows=[]
added=set(m.get('supplemental_inventory',{}).get('features',[]))
for c in m['features']:
    origin='supplemental_inventory_or_new_transform' if c in added else 'requested_csv_or_initial_derivation'
    if c in m.get('supplemental_cost_features',[]):origin='prior_costs_of_nearby_inventory_segments'
    cost_dependent=c.startswith('hist_') and not c.endswith('_n')
    availability='previous calendar years only' if c.startswith('hist_') or 'repair' in c else 'inventory laid by incident date' if c.startswith('inventory_') else 'incident and pipe attributes'
    if c.startswith('inventory_') and c.endswith('length_sum'):note='Sum of whole endpoint Euclidean lengths of eligible segments within planar proximity radius; not excavated length.'
    elif 'distance_to' in c:note='Shortest planar segment distance in meters; no physical connection inferred.'
    elif 'parallel_projected_length' in c:note='Projected overlap of nearly parallel segments in meters; proximity is not connectivity.'
    elif cost_dependent:note='Historic cost summary from chosen cost CSV; no current-year costs.'
    else:note=''
    row={'feature':c,'origin':origin,'cost_dependent':cost_dependent,'availability':availability,'note':note,'dtype':str(d[c].dtype)}
    for year in range(2023,2027):row[f'train_missing_fraction_before_{year}']=float(d.loc[d.event_year<year,c].isna().mean())
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'feature_catalogue.csv',index=False)
print('Catalogued',len(rows),'features')
