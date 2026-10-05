"""Supplement requested cost CSV with target-free inventory and frozen repair state."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import run_experiment as e
sys.path.insert(0,str(e.ROOT/'lightgbm_cost_model'))
from build_local_geometry import segment_distance

def main():
    d=pd.read_csv(e.OUT/'features.csv')
    source=e.ROOT/'github-main/data/pipes.csv'
    if not source.exists():source=e.ROOT/'data/pipes.csv'
    p=pd.read_csv(source);assert p['Pipe ID'].is_unique
    a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy()
    mid=(a+b)/2;v=b-a;length=np.linalg.norm(v,axis=1);assert (length>0).all()
    tree=cKDTree(mid);lay=pd.to_datetime(p['Lay date']).to_numpy()
    ids=pd.Series(p.index,index=p['Pipe ID']);surf=p.Surface.fillna('unknown').to_numpy();mat=p.Material.fillna('unknown').to_numpy()
    dates=pd.to_datetime(d.date);hist=pd.Series(dates.to_numpy(),index=d.pipe_id)
    repairs=p['Pipe ID'].map(hist).to_numpy(dtype='datetime64[ns]')
    rows=[]
    for idx,row in enumerate(d.itertuples()):
        i=int(ids[row.pipe_id]);now=np.datetime64(row.date);cut=np.datetime64(f'{row.event_year}-01-01')
        assert np.allclose(a[i],[row.x1,row.y1]) and np.allclose(b[i],[row.x2,row.y2])
        ix=np.asarray(tree.query_ball_point(mid[i],length[i]/2+length.max()/2+250+1e-8),int)
        ix=ix[(ix!=i)&(lay[ix]<=now)]
        dist=segment_distance(a[i],b[i],a[ix],b[ix]);unit=v[i]/length[i]
        cosine=np.abs(v[ix]@unit)/length[ix]
        t0=(a[ix]-a[i])@unit;t1=(b[ix]-a[i])@unit
        projection=np.maximum(0,np.minimum(length[i],np.maximum(t0,t1))-np.maximum(0,np.minimum(t0,t1)))
        parallel=(cosine>=np.cos(np.pi/12))&(projection>1e-8)
        r={}
        for radius in [5,10,25,50,100,250]:
            mask=dist<=radius+1e-8;j=ix[mask];n=len(j);tag=f'inventory_{radius}m_'
            r[tag+'count']=n;r[tag+'length_sum']=float(length[j].sum())
            r[tag+'parallel_count']=int((mask&parallel).sum());r[tag+'parallel_projected_length']=float(projection[mask&parallel].sum())
            r[tag+'surface_diversity']=len(np.unique(surf[j]))
            for label in ['road','structure','water','swamp']:r[tag+label+'_count']=int((surf[j]==label).sum())
            r[tag+'same_material_fraction']=float((mat[j]==mat[i]).mean()) if n else 0.
            r[tag+'iron_fraction']=float(np.isin(mat[j],['cast iron','gray iron','wrought iron']).mean()) if n else 0.
            ages=(now-lay[j])/np.timedelta64(1,'D')/365.25
            r[tag+'age_mean']=float(ages.mean()) if n else np.nan
            repaired=repairs[j]<cut;recent=repaired&(repairs[j]>=cut-np.timedelta64(365,'D'))
            r[tag+'prior_repairs']=int(repaired.sum());r[tag+'prior_repair_fraction']=float(repaired.mean()) if n else 0.
            r[tag+'recent_repairs']=int(recent.sum())
            r[tag+'days_since_repair']=float((cut-repairs[j][repaired].max())/np.timedelta64(1,'D')) if repaired.any() else np.nan
        for label in ['road','structure','water']:
            js=np.flatnonzero((surf==label)&(lay<=now)&(np.arange(len(p))!=i))
            r['inventory_distance_to_'+label]=float(segment_distance(a[i],b[i],a[js],b[js]).min()) if len(js) else np.nan
        r['age_x_wet_proximity']=row.years_used*np.log1p(r['inventory_25m_water_count']+r['inventory_25m_swamp_count'])
        r['length_x_structure_exposure']=row.pipe_length*np.log1p(r['inventory_25m_structure_count'])
        r['road_x_structure_exposure']=int(row.surface=='road')*np.log1p(r['inventory_25m_structure_count'])
        rows.append(r)
        if idx%1000==0:print('inventory',idx,flush=True)
    extra=pd.DataFrame(rows)
    for c in [c for c in d if c.startswith('hist_') and c.endswith(('_mean','_median','_max','_std'))]:
        extra[c+'_log1p']=np.log1p(d[c])
    for c in ['pipe_length','years_used','midpoint_x','midpoint_y']:extra[c+'_squared']=d[c]**2
    if not (e.OUT/'features_before_expansion.csv').exists():d.to_csv(e.OUT/'features_before_expansion.csv',index=False)
    added=[c for c in extra if c not in d]
    d=pd.concat([d,extra[added]],axis=1);d.to_csv(e.OUT/'features.csv',index=False)
    m=json.loads((e.OUT/'feature_manifest.json').read_text());m['features']=sorted(set(m['features'])|set(added))
    m['supplemental_inventory']={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'features':added,'availability':'inventory lay date <= incident date; repair records strictly before Jan 1 of incident year; repairs from chosen cost CSV only','geometry':'planar segment proximity in meters; crossings do not imply connectivity'}
    e.dump(e.OUT/'feature_manifest.json',m);print('ADDED',len(added),'TOTAL',len(m['features']),flush=True)

if __name__=='__main__':main()
