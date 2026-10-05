"""Exact planar segment proximity in meters; no cost-derived features."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from spatial_experiment import relations, PIPES


def point_distance(q,a,b):
    v=b-a
    t=np.clip(np.sum((q-a)*v,axis=-1)/np.sum(v*v,axis=-1),0,1)
    return np.linalg.norm(q-(a+t[...,None]*v),axis=-1)


def segment_distance(a,b,c,e):
    dist=np.minimum.reduce([point_distance(a,c,e),point_distance(b,c,e),
                            point_distance(c,a,b),point_distance(e,a,b)])
    dist[relations(a,b,c,e)[0]]=0
    return dist


def build(d):
    p=pd.read_csv(PIPES)
    a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy()
    mid=(a+b)/2;vec=b-a;length=np.linalg.norm(vec,axis=1);half=length/2
    assert (length>0).all()
    lay=pd.to_datetime(p['Lay date']).to_numpy(); tree=cKDTree(mid)
    ids=pd.Series(np.arange(len(p)),index=p['Pipe ID'])
    surfaces={}
    for label in ['road','structure','water']:
        ix=np.flatnonzero(p.Surface.to_numpy()==label)
        surfaces[label]=(ix,cKDTree(mid[ix]),half[ix].max())
    rows=[]
    for row in d.itertuples():
        i=int(ids[row.pipe_id]);date=np.datetime64(row.date)
        near=np.asarray(tree.query_ball_point(mid[i],half[i]+half.max()+25+1e-8),dtype=int)
        near=near[(near!=i)&(lay[near]<=date)]
        dist=segment_distance(a[i],b[i],a[near],b[near])
        unit=vec[i]/length[i]
        angle=np.abs(np.sum(vec[near]*unit,axis=1))/length[near]
        t0=np.sum((a[near]-a[i])*unit,axis=1);t1=np.sum((b[near]-a[i])*unit,axis=1)
        projected=np.maximum(0,np.minimum(length[i],np.maximum(t0,t1))-np.maximum(0,np.minimum(t0,t1)))
        parallel=(angle>=np.cos(np.deg2rad(15)))&(projected>1e-8)
        out={'pipe_id':row.pipe_id}
        for radius in [5,10,25]:
            close=dist<=radius+1e-8
            out[f'segment_neighbors_{radius}m']=int(close.sum())
            out[f'parallel_count_{radius}m']=int((close&parallel).sum())
            out[f'parallel_projected_length_{radius}m']=float(projected[close&parallel].sum())
        for label,(indices,tr,maxhalf) in surfaces.items():
            k=8
            while True:
                _,local=tr.query(mid[i],k=min(k,len(indices)))
                js=indices[np.atleast_1d(local)];js=js[(js!=i)&(lay[js]<=date)]
                if len(js):break
                if k>=len(indices):raise ValueError('No eligible surface neighbors')
                k*=2
            upper=segment_distance(a[i],b[i],a[js],b[js]).min()
            js=indices[tr.query_ball_point(mid[i],half[i]+maxhalf+upper+1e-8)]
            js=js[(js!=i)&(lay[js]<=date)]
            out[f'distance_to_{label}_pipe_m']=float(segment_distance(a[i],b[i],a[js],b[js]).min())
        rows.append(out)
    return pd.DataFrame(rows)
