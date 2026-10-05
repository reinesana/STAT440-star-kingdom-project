"""Inventory-only spatial features and time-respecting model comparisons."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import lightgbm as lgb
from train_baseline import SOURCE
from optimize import diagnose

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'spatial_experiment'
PIPES=SOURCE.parent/'pipes.csv'


def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def relations(a,b,c,e):
    """Exact planar segment classifications, including collinear overlap."""
    r=b-a; s=e-c; ca=c-a
    den=cross(r,s); parallel=np.abs(den)<1e-9
    safe=np.where(parallel,1.,den)
    t=cross(ca,s)/safe; u=cross(ca,r)/safe
    meet=(~parallel)&(t>=-1e-9)&(t<=1+1e-9)&(u>=-1e-9)&(u<=1+1e-9)
    proper=(~parallel)&(t>1e-9)&(t<1-1e-9)&(u>1e-9)&(u<1-1e-9)
    col=parallel&(np.abs(cross(ca,r))<1e-8)
    rr=np.dot(r,r)
    v0=np.sum((c-a)*r,axis=1)/rr; v1=np.sum((e-a)*r,axis=1)/rr
    overlap=np.minimum(1,np.maximum(v0,v1))-np.maximum(0,np.minimum(v0,v1))
    meet|=col&(overlap>=-1e-9)
    positive_overlap=col&(overlap>1e-9)
    shared=(np.linalg.norm(c-a,axis=1)<1e-8)|(np.linalg.norm(e-a,axis=1)<1e-8)|(np.linalg.norm(c-b,axis=1)<1e-8)|(np.linalg.norm(e-b,axis=1)<1e-8)
    return meet,proper,positive_overlap,shared


def main():
    OUT.mkdir(exist_ok=True)
    # Tiny analytic geometry checks: crossing, endpoint touch, overlap, disjoint.
    test=relations(np.array([0.,0.]),np.array([2.,0.]),np.array([[1,-1],[2,0],[1,0],[3,0]]),np.array([[1,1],[3,1],[3,0],[4,0]]))
    assert test[0].tolist()==[True,True,True,False]
    assert test[1].tolist()==[True,False,False,False]
    assert test[2].tolist()==[False,False,True,False]
    p=pd.read_csv(PIPES); d=pd.read_csv(SOURCE)
    assert p['Pipe ID'].is_unique and d.pipe_id.is_unique
    a=p[['GPS x1','GPS y1']].to_numpy(); b=p[['GPS x2','GPS y2']].to_numpy()
    centers=(a+b)/2; lengths=np.linalg.norm(b-a,axis=1)
    assert np.isfinite(centers).all() and (lengths>0).all()
    tree=cKDTree(centers); lay=pd.to_datetime(p['Lay date']).to_numpy()
    ids=pd.Series(np.arange(len(p)),index=p['Pipe ID'])
    rows=[]
    for row in d.itertuples():
        i=int(ids[row.pipe_id]); date=np.datetime64(row.date)
        assert np.allclose(a[i],[row.gps_x1,row.gps_y1]) and np.allclose(b[i],[row.gps_x2,row.gps_y2])
        radius=max(250.,(lengths[i]+lengths.max())/2+1e-6)
        ix=np.asarray(tree.query_ball_point(centers[i],radius),dtype=int)
        ix=ix[(ix!=i)&(lay[ix]<=date)]  # excludes unknown or future lay dates
        distances=np.linalg.norm(centers[ix]-centers[i],axis=1)
        r={'pipe_id':row.pipe_id}
        for rad in [50,100,250]:r[f'neighbor_count_{rad}']=int((distances<=rad).sum())
        near=ix[distances<=100]
        r['neighbor_length_sum_100']=float(lengths[near].sum())
        r['neighbor_road_count_100']=int((p.iloc[near].Surface=='road').sum())
        k=16
        while True:
            ds,js=tree.query(centers[i],k=min(k,len(p)))
            valid=(js!=i)&(lay[js]<=date)
            ds=ds[valid]
            if len(ds)>=5:break
            if k>=len(p):raise ValueError('Fewer than 5 eligible neighbors')
            k*=2
        r['nearest_neighbor_distance']=float(ds[0]); r['mean_5_neighbor_distance']=float(ds[:5].mean())
        meet,proper,overlap,shared=relations(a[i],b[i],a[ix],b[ix])
        for name,val in [('intersection_count',meet),('proper_crossing_count',proper),('collinear_overlap_count',overlap),('shared_endpoint_count',shared)]:r[name]=int(val.sum())
        rows.append(r)
    spatial=pd.DataFrame(rows)
    spatial.to_csv(OUT/'spatial_features.csv',index=False)
    d=d.merge(spatial,on='pipe_id',validate='one_to_one')
    extras=[c for c in spatial if c!='pipe_id']
    spatial[extras].describe().to_csv(OUT/'feature_profile.csv')
    print('SPATIAL '+json.dumps({c:dict(nonzero=int((spatial[c]>0).sum()),max=float(spatial[c].max())) for c in extras}),flush=True)
    cfg=json.loads((ROOT/'optimization/result.json').read_text())['best']
    cache={}; comparisons=[]
    def evaluate(features,save=False):
        features=sorted(features); key='|'.join(features)
        if key in cache and not save:return cache[key]
        folds=[]; predictions=[]
        for year in range(2023,2027):
            tr,te=d[d.event_year<year],d[d.event_year==year]
            it,iv=tr[tr.event_year<year-1],tr[tr.event_year==year-1]
            def matrices(x,y):
                x=x[features].copy();y=y[features].copy()
                for col in set(features)&{'material','surface'}:
                    levels=sorted(x[col].unique());x[col]=pd.Categorical(x[col],categories=levels);y[col]=pd.Categorical(y[col],categories=levels)
                return x,y
            x,v=matrices(it,iv)
            def metric(y,pred):return 'dollar_rmse',float(np.sqrt(np.mean((y-np.maximum(pred,0))**2))),False
            model=lgb.LGBMRegressor(**cfg['params'],n_estimators=3000,metric='None')
            model.fit(x,it.cost,eval_set=[(v,iv.cost)],eval_metric=metric,callbacks=[lgb.early_stopping(100,verbose=False)])
            n=model.best_iteration_;x,z=matrices(tr,te)
            model=lgb.LGBMRegressor(**cfg['params'],n_estimators=n);model.fit(x,tr.cost)
            pred=np.maximum(model.predict(z),0);err=pred-te.cost.to_numpy()
            folds.append(dict(year=year,rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.abs(err).mean()),iterations=n))
            if save:
                model.booster_.save_model(str(OUT/f'model_{year}.txt'))
                predictions.append(pd.DataFrame(dict(pipe_id=te.pipe_id,year=year,actual=te.cost,prediction=pred)))
        result=dict(features=features,rmse=float(np.mean([f['rmse'] for f in folds])),mae=float(np.mean([f['mae'] for f in folds])),folds=folds)
        cache[key]=result
        (OUT/'trials.json').write_text(json.dumps(cache,indent=2))
        if save:
            pred=pd.concat(predictions);pred.to_csv(OUT/'predictions.csv',index=False)
            before=pd.read_csv(ROOT/'optimization/predictions.csv')
            pd.DataFrame(diagnose(before,'previous')+diagnose(pred,'spatial')).to_csv(OUT/'error_diagnostics.csv',index=False)
        return result
    baseline=evaluate(cfg['features']);assert np.isclose(baseline['rmse'],cfg['rmse'])
    current=baseline
    # Compare entire family too, to avoid missing combined effects.
    whole=evaluate(cfg['features']+extras)
    if whole['rmse']<current['rmse']:current=whole
    history=[]
    while True:
        winner=current;action=None
        for col in extras:
            fs=set(current['features']);fs.symmetric_difference_update([col]);r=evaluate(fs)
            comparisons.append(dict(feature=col,action='remove' if col in current['features'] else 'add',reference_rmse=current['rmse'],rmse=r['rmse'],improvement=current['rmse']-r['rmse']))
            if r['rmse']<winner['rmse']-1e-6:winner=r;action=col
        if action is None:break
        current=winner;history.append(dict(changed_feature=action,rmse=current['rmse']))
        print('IMPROVED '+json.dumps(history[-1]),flush=True)
    final=evaluate(current['features'],save=True)
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    result=dict(baseline=baseline,all_spatial=whole,best=final,history=history,params=cfg['params'],candidates=extras,
                evaluated_sets=len(cache),inventory_rows=len(p),unknown_lay_dates=int(pd.isna(lay).sum()),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),pipes_sha256=hashlib.sha256(PIPES.read_bytes()).hexdigest(),
                caveats=['Planar geometry only, not confirmed physical junctions or depth.', 'Distances in source coordinate units, not verified meters.',
                         'Neighbors require known lay date at or before event. Historical retirement/removal is unavailable.',
                         'All four years used for selection; not independent final test.'])
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    if final['rmse']<baseline['rmse']:
        (ROOT/'selected_config.json').write_text(json.dumps(dict(features=final['features'],params=cfg['params'],target='raw',validation_rmse=final['rmse'],spatial_feature_builder='spatial_experiment.py',note=result['caveats']),indent=2))
    print('FINAL '+json.dumps(result),flush=True)


if __name__=='__main__':main()
