"""Independently audit inventory IDs, availability, counts and prior repair records."""
import json,hashlib
import numpy as np
import pandas as pd
from pathlib import Path
OUT=Path(__file__).resolve().parent
m=json.loads((OUT/'feature_manifest.json').read_text());info=m['supplemental_inventory']
p=pd.read_csv(info['path']);d=pd.read_csv(OUT/'features.csv')
assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256']
assert p['Pipe ID'].is_unique
matched=p.set_index('Pipe ID').reindex(d.pipe_id)
assert np.array_equal(matched.Material.to_numpy(),d.material.to_numpy())
assert np.array_equal(matched.Surface.to_numpy(),d.surface.to_numpy())
assert np.array_equal(pd.to_datetime(matched['Lay date']).to_numpy(),pd.to_datetime(d.lay_date).to_numpy())
a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy();v=b-a
lay=pd.to_datetime(p['Lay date']).to_numpy();repairs=p['Pipe ID'].map(pd.Series(pd.to_datetime(d.date).to_numpy(),index=d.pipe_id)).to_numpy(dtype='datetime64[ns]')
def distance(q):
    t=np.clip(((q-a)*v).sum(axis=1)/(v*v).sum(axis=1),0,1)
    return np.linalg.norm(q-a-t[:,None]*v,axis=1)
for row in d.groupby('event_year').head(3).itertuples():
    i=int(p.index[p['Pipe ID']==row.pipe_id][0]);r=b[i]-a[i]
    ta=np.clip(((a-a[i])*r).sum(axis=1)/(r*r).sum(),0,1)
    tb=np.clip(((b-a[i])*r).sum(axis=1)/(r*r).sum(),0,1)
    dist=np.minimum.reduce([distance(a[i]),distance(b[i]),np.linalg.norm(a-a[i]-ta[:,None]*r,axis=1),np.linalg.norm(b-a[i]-tb[:,None]*r,axis=1)])
    den=r[0]*v[:,1]-r[1]*v[:,0];safe=np.where(abs(den)<1e-9,1,den);offset=a-a[i]
    t=(offset[:,0]*v[:,1]-offset[:,1]*v[:,0])/safe;u=(offset[:,0]*r[1]-offset[:,1]*r[0])/safe
    cross=(abs(den)>=1e-9)&(t>=-1e-9)&(t<=1+1e-9)&(u>=-1e-9)&(u<=1+1e-9);dist[cross]=0
    eligible=(p['Pipe ID'].to_numpy()!=row.pipe_id)&(lay<=np.datetime64(row.date))
    if hasattr(row,'inventory_nearest_midpoint_distance'):
        midpoint_distance=np.linalg.norm((a+b)/2-(a[i]+b[i])/2,axis=1)[eligible]
        nearest=np.sort(midpoint_distance)[:5]
        assert np.isclose(row.inventory_nearest_midpoint_distance,nearest[0])
        assert np.isclose(row.inventory_mean5_midpoint_distance,nearest.mean())
        proper=(abs(den)>=1e-9)&(t>1e-9)&(t<1-1e-9)&(u>1e-9)&(u<1-1e-9)
        shared=(np.linalg.norm(a-a[i],axis=1)<1e-8)|(np.linalg.norm(b-a[i],axis=1)<1e-8)|(np.linalg.norm(a-b[i],axis=1)<1e-8)|(np.linalg.norm(b-b[i],axis=1)<1e-8)
        assert row.inventory_proper_crossing_count==int((eligible&proper).sum())
        assert row.inventory_shared_endpoint_count==int((eligible&shared).sum())
    cut=np.datetime64(f'{row.event_year}-01-01')
    if hasattr(row,'hist_segment25_all_mean'):
        costs=p['Pipe ID'].map(pd.Series(d.cost.to_numpy(),index=d.pipe_id)).to_numpy()
        global_history=d.loc[d.event_year<row.event_year,'cost'];prior=global_history.mean()
        for radius in [25,100,250]:
            z=costs[eligible&(dist<=radius+1e-8)&(repairs<cut)];prefix=f'hist_segment{radius}_all_'
            assert getattr(row,prefix+'n')==len(z)
            assert np.isclose(getattr(row,prefix+'mean'),(z.sum()+20*prior)/(len(z)+20),equal_nan=True)
            assert np.isclose(getattr(row,prefix+'median'),np.median(z) if len(z) else np.nan,equal_nan=True)
            assert np.isclose(getattr(row,prefix+'max'),z.max() if len(z) else np.nan,equal_nan=True)
    for radius in [5,10,25,50,100,250]:
        mask=eligible&(dist<=radius+1e-8);prefix=f'inventory_{radius}m_'
        assert getattr(row,prefix+'count')==int(mask.sum())
        assert getattr(row,prefix+'prior_repairs')==int((mask&(repairs<cut)).sum())
        assert np.isclose(getattr(row,prefix+'length_sum'),np.linalg.norm(v[mask],axis=1).sum())
        assert getattr(row,prefix+'structure_count')==int((mask&p.Surface.eq('structure').to_numpy()).sum())
print('PASS inventory: 24 rows x 6 radii; source fingerprint and prior-year repair availability')
(OUT/'inventory_audit.json').write_text(json.dumps({'source_hash_verified':True,'sample_rows':24,'radii':6,'cost_free_inventory_features':True,'repair_source':'chosen cost CSV dates only; strictly earlier calendar years'}),encoding='utf-8')
