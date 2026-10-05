"""Recover remaining historical spatial candidates; stage without changing active searches."""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import run_experiment as e
sys.path.insert(0,str(e.ROOT/'lightgbm_cost_model'))
from spatial_experiment import relations
d=pd.read_csv(e.OUT/'features.csv');p=pd.read_csv(e.ROOT/'github-main/data/pipes.csv')
a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy();mid=(a+b)/2;length=np.linalg.norm(b-a,axis=1)
tree=cKDTree(mid);lay=pd.to_datetime(p['Lay date']).to_numpy();ids=pd.Series(p.index,index=p['Pipe ID']);rows=[]
for idx,row in enumerate(d.itertuples()):
    i=int(ids[row.pipe_id]);date=np.datetime64(row.date)
    k=16
    while True:
        ds,js=tree.query(mid[i],k=min(k,len(p)));valid=(js!=i)&(lay[js]<=date);distances=ds[valid]
        if len(distances)>=5:break
        assert k<len(p);k*=2
    ix=np.asarray(tree.query_ball_point(mid[i],(length[i]+length.max())/2+1e-6),int)
    ix=ix[(ix!=i)&(lay[ix]<=date)]
    r={'pipe_id':row.pipe_id,'inventory_nearest_midpoint_distance':float(distances[0]),'inventory_mean5_midpoint_distance':float(distances[:5].mean())}
    for name,values in zip(['intersection_count','proper_crossing_count','collinear_overlap_count','shared_endpoint_count'],relations(a[i],b[i],a[ix],b[ix])):r['inventory_'+name]=int(values.sum())
    rows.append(r)
    if idx%2000==0:print('extra spatial',idx,flush=True)
pd.DataFrame(rows).to_csv(e.OUT/'extra_spatial_candidates.csv',index=False)
print('Staged six remaining prior-experiment geometry features; no connectivity inferred')
