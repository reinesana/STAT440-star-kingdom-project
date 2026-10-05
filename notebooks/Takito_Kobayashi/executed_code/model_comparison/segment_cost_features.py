"""Prior-year cost summaries for nearby planar segments, not repair locations."""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import run_experiment as e
sys.path.insert(0,str(e.ROOT/'lightgbm_cost_model'))
from build_local_geometry import segment_distance

def build(d):
    p=pd.read_csv(e.ROOT/'github-main/data/pipes.csv');assert p['Pipe ID'].is_unique
    a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy();mid=(a+b)/2;length=np.linalg.norm(b-a,axis=1)
    tree=cKDTree(mid);ids=pd.Series(p.index,index=p['Pipe ID'])
    lay=pd.to_datetime(p['Lay date']).to_numpy()
    years=pd.to_datetime(d.date).dt.year
    times=p['Pipe ID'].map(pd.Series(pd.to_datetime(d.date).to_numpy(),index=d.pipe_id)).to_numpy(dtype='datetime64[ns]')
    costs=p['Pipe ID'].map(pd.Series(d.cost.to_numpy(),index=d.pipe_id)).to_numpy()
    priors={}
    for year in years.unique():
        z=d.loc[years<year,'cost'];threshold=float(z.quantile(.99));priors[year]=(float(z.mean()),threshold,float((z>=threshold).mean()) if len(z) else np.nan)
    rows=[]
    for row,year in zip(d.itertuples(),years):
        i=int(ids[row.pipe_id]);cut=np.datetime64(f'{year}-01-01')
        ix=np.asarray(tree.query_ball_point(mid[i],length[i]/2+length.max()/2+250+1e-8),int)
        ix=ix[(ix!=i)&(times[ix]<cut)&(lay[ix]<=np.datetime64(row.date))]
        dist=segment_distance(a[i],b[i],a[ix],b[ix]);prior,threshold,rate=priors[year];r={'pipe_id':row.pipe_id}
        for radius in [25,100,250]:
            z=costs[ix[dist<=radius+1e-8]];n=len(z);prefix=f'hist_segment{radius}_all_'
            r.update({prefix+'n':n,prefix+'mean':float((z.sum()+20*prior)/(n+20)),prefix+'median':float(np.median(z)) if n else np.nan,prefix+'max':float(z.max()) if n else np.nan,prefix+'std':float(z.std(ddof=1)) if n>1 else np.nan,prefix+'tail_rate':float(((z>=threshold).sum()+50*rate)/(n+50))})
        rows.append(r)
    return pd.DataFrame(rows)
