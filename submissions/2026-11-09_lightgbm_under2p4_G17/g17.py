"""Reorder G17 circles with explicit assignment-preserving precedence edges."""
from pathlib import Path
import sys,csv,ast,json,heapq
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate as ev
a=ev.a;b=ev.b;ROOT=a.ROOT

def geometry(p):
    path=ROOT/'output/2026-11-02_regional_optimization/G17/G17_single_merge.csv'
    assert a.sha(path)=='47a960a0cec73ee2abd1bb5df61891b66f64eca4a05f114e76a25e9b7a6dd86b'
    with path.open(newline='',encoding='utf8') as f:
        units=[(np.asarray(ast.literal_eval(row[0]),float),float(row[1])) for row in csv.reader(f)]
    aa=p[['GPS x1','GPS y1']].to_numpy();bb=p[['GPS x2','GPS y2']].to_numpy();length=np.linalg.norm(bb-aa,axis=1);mid=(aa+bb)/2;tree=cKDTree(mid)
    inside=[];base=[]
    for center,radius in units:
        ids=np.asarray(tree.query_ball_point(center,radius),int)
        ids=ids[(np.linalg.norm(aa[ids]-center,axis=1)<=radius)&(np.linalg.norm(bb[ids]-center,axis=1)<=radius)]
        inside.append(ids);base.append(100+sum(b.RATES[s] for s in set(p.Surface.iloc[ids]))*(np.pi*radius**2)**.85)
    off=np.r_[0,np.cumsum([len(x) for x in inside])].astype(np.int64);flat=np.concatenate(inside).astype(np.int64)
    iron=p.Material.isin(b.IRON).to_numpy();g=(aa,bb,length,iron,np.arange(len(units)),np.array([x[0] for x in units]),np.array([x[1] for x in units]),inside,off,flat,np.array(base))
    return g,path

def reorder(g,state,values):
    owner=np.full(len(state),-1,int);active=[];free=[];cost={};priority={}
    for j,ids in enumerate(g[7]):
        new=ids[(~state[ids])&(owner[ids]<0)]
        if len(new):
            owner[new]=j;active.append(j);cost[j]=int(np.floor((g[10][j]+200*g[2][new].sum())*100+.5));priority[j]=float(values[new].sum())/max(cost[j]/100,1)
        else:free.append(j)
    pending={};children={j:[] for j in active}
    for j in active:
        deps=set(owner[g[7][j]]);deps={int(x) for x in deps if x>=0 and x!=j};assert all(x<j for x in deps)
        pending[j]=len(deps)
        for x in deps: children[x].append(j)
    heap=[(-priority[j],j) for j in active if pending[j]==0];heapq.heapify(heap);order=[]
    while heap:
        _,j=heapq.heappop(heap);order.append(j)
        for child in children[j]:
            pending[child]-=1
            if not pending[child]:heapq.heappush(heap,(-priority[child],child))
    assert len(order)==len(active)
    live=state.copy();total=0
    for j in order:
        ids=g[7][j];new=ids[~live[ids]];assert set(new)==set(np.flatnonzero(owner==j))
        total+=int(np.floor((g[10][j]+200*g[2][new].sum())*100+.5));live[ids]=True
    assert total==sum(cost.values()) and live[g[3]].all()
    return np.array(order+free,dtype=np.int64),dict(active_rows=len(active),free_coverage_rows=len(free),dependency_edges=sum(len(x) for x in children.values()),unchanged_total_construction=total/100,assignment_preserved=True)

def main():
    p=pd.read_csv(ROOT/'data/pipes.csv');h=pd.read_csv(ROOT/'data/train.csv');h['year']=pd.to_datetime(h.Date).dt.year;g,source=geometry(p)
    forecasts=pd.read_csv(HERE/'forecast_costs_2027.csv').set_index('pipe_id').loc[p['Pipe ID']]
    state=ev.initial(p,h,2027);risk=forecasts.leak_probability.to_numpy();orders={'G17_original':np.arange(len(g[4]),dtype=np.int64)};reports=[]
    for label in ['selected','CMS001','best_symmetric']:
        seq,audit=reorder(g,state,risk*forecasts[label].to_numpy());orders[label]=seq
        path=HERE/'submission_files'/f'{label}_G17_circles.csv'
        path.write_text(''.join(f'"({g[5][j,0]:.9f}, {g[5][j,1]:.9f})",{g[6][j]:.9f}\n' for j in seq),encoding='utf8')
        validation=b.validate(path,p,h.drop(columns='year'),g[0],g[1],(g[0]+g[1])/2,g[2],g[3])
        np.testing.assert_allclose(validation['construction_cost'],audit['unchanged_total_construction'],atol=.01)
        reports.append(dict(label=label,path=str(path),sha256=a.sha(path),source_sha256=a.sha(source),**audit,validation=validation))
        print('G17 REORDERED',label,flush=True)
    a.dump(HERE/'g17_validation.json',reports)
    ev.simulations(g,p,h,2027,orders,prefix='G17_')
    print('G17 COMPLETE',flush=True)

if __name__=='__main__':main()
