from pathlib import Path
import sys,json,time,heapq,argparse,importlib.util
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('asym',HERE/'run.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
ROOT=a.ROOT;b=a.bench
from numba import njit
# An archived cache references an unavailable '<dynamic>' module. Recompile the
# unchanged replay function in memory; do not mutate or delete historical caches.
b.evaluate=njit(cache=False)(b.evaluate.py_func)
sys.path.insert(0,str(ROOT/'output/2026-10-27_time_exhaustive'))
# Explicit import avoids accidentally binding another evaluate.py.
spec2=importlib.util.spec_from_file_location('timing_replay',ROOT/'output/2026-10-27_time_exhaustive/evaluate.py');old=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(old)

def load():
    p=pd.read_csv(ROOT/'data/pipes.csv');h=pd.read_csv(ROOT/'data/train.csv');h['year']=pd.to_datetime(h.Date).dt.year
    g=b.geometry(p)
    incidence=csr_matrix((np.ones(len(g[9])),g[9],g[8]),shape=(len(g[4]),len(p)))
    reverse=incidence.T.tocsr()
    return p,h,g,reverse

def initial(p,h,year): return (p.Material.eq('polyurethane')|p['Pipe ID'].isin(h.loc[h.year.lt(year),'Pipe ID'])).to_numpy()

def probability(p,year):
    path=ROOT/'output/2026-10-27_time_exhaustive/forecasts/F168'/str(year)/'curves.npz'
    with np.load(path) as z:
        ids=z['pipe_id'].astype(str);s=z['survival'].astype(float)
    pos=pd.Index(p['Pipe ID']).get_indexer(ids);assert (pos>=0).all()
    risk=np.zeros(len(p));risk[pos]=1-s[:,0]
    assert np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all()
    return risk

def fullcost(p,h,year,ids,pred):
    hist=h[h.year.lt(year)].join(p.set_index('Pipe ID')[['Surface']],on='Pipe ID')
    means=hist.groupby('Surface').Cost.agg(['sum','count']);mean=float(hist.Cost.mean());smoothed=(means['sum']+114.72895299736683*mean)/(means['count']+114.72895299736683)
    val=p['Pipe ID'].map(pd.Series(pred,index=ids));missing=val.isna()&~initial(p,h,year)
    val=val.fillna(p.Surface.map(smoothed)).fillna(mean).to_numpy();val[initial(p,h,year)]=0
    return val,p.loc[missing,'Pipe ID'].tolist()

def cms(p,h,year):
    if year==2027:
        path=ROOT/'output/2026-11-02_robustness_paper/inputs/CMS001_predictions_2027.csv';q=pd.read_csv(path)
        return fullcost(p,h,year,q.pipe_id,q.predicted_repair_cost)
    folder=ROOT/'github-main/notebooks/Takito_Kobayashi/2026-10-28/ThreeModelStacking'
    q=pd.read_csv(ROOT/f'output/2026-10-28_cost_model_comparison/combination_policies/cost_inventory_{year}.csv')
    inv=p.set_index('Pipe ID');q['material_surface']=q.pipe_id.map(inv.Material).fillna('__unknown__')+' | '+q.pipe_id.map(inv.Surface).fillna('__unknown__')
    config=json.loads((folder/f'config_{year}.json').read_text())['models']['conditional_material_surface_rmse']
    base=['takito_catboost','anthony_rf_log_mean','rion_lgb_log_mean_refit'];w=np.tile(config['global_weights'],(len(q),1))
    for key,info in config['groups'].items():
        n=info['n'];factor=n/(n+config['tau']);w[q.material_surface.eq(key)]=factor*np.array(info['local_weights'])+(1-factor)*np.array(config['global_weights'])
    pred=(q[base].to_numpy()*w).sum(1)
    truth=pd.read_csv(folder/'heldout_predictions_2024_2026.csv');truth=truth[truth.event_year.eq(year)]
    matched=pd.Series(pred,index=q.pipe_id).reindex(truth.pipe_id)
    delta=np.abs(matched.to_numpy()-truth.conditional_material_surface_rmse.to_numpy())
    audit_path=HERE/'CMS001_reconstruction_audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audit[str(year)]=dict(n=len(delta),max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean()),rows_difference_over_2c=int((delta>.02).sum()),interpretation='Historical inventory refits with frozen recipes; float/forest threshold differences from original event-only OOF are recorded. No observed event predictions overwritten.')
    a.dump(audit_path,audit)
    np.testing.assert_allclose(matched,truth.conditional_material_surface_rmse,rtol=.01,atol=1.,err_msg='CMS001 inventory reconstruction differs materially from archived OOF')
    return fullcost(p,h,year,q.pipe_id,pred)

def order(g,reverse,state,risk,cost):
    return b.dynamic_order(g[8],g[9],reverse.indptr.astype(np.int64),reverse.indices.astype(np.int64),g[10],g[2],state,risk*cost,1.)

def metric(p,h,g,year,sequence,cost):
    state=initial(p,h,year);events=p['Pipe ID'].isin(h.loc[h.year.eq(year),'Pipe ID']).to_numpy()
    observed=p['Pipe ID'].map(h.loc[h.year.eq(year)].set_index('Pipe ID').Cost).fillna(0).to_numpy()
    z=old.replay(sequence,g[8],g[9],g[10],g[2],state,g[3],events,observed)
    no=old.replay(sequence,g[8],g[9],g[10],g[2],state,g[3],np.zeros(len(p),bool),np.zeros(len(p)))
    # Independent first-year budget trace gives pipe-level prevented outcomes.
    replaced=state.copy();budget=1_000_000_000;construction=0;executed=0
    for j in sequence:
        ids=g[9][g[8][j]:g[8][j+1]];live=ids[~replaced[ids]]
        if not len(live): continue
        cents=int(np.floor((g[10][j]+200*g[2][live].sum())*100+.5))
        if cents>budget: break
        budget-=cents;construction+=cents/100;executed+=1;replaced[live]=True
    saved=events&replaced&~state;amount=float(observed[saved].sum());actual=observed[events];pred=cost[events];err=pred-actual
    np.testing.assert_allclose(amount,observed.sum()-z[1],atol=1e-5)
    hist=h[h.year.lt(year)];tail=float(hist.Cost.quantile(.99));expensive=events&(observed>=tail)
    top=np.argsort(-pred,kind='stable')[:int(np.ceil(.1*len(pred)))]
    high=events&(observed>=np.quantile(actual,.99))
    return dict(year=year,actual_repair_total=float(observed.sum()),prevented_dollars=amount,cost_capture=amount/observed.sum(),leak_recall=float(saved.sum()/events.sum()),
        expensive_prevented_dollars=float(observed[saved&expensive].sum()),expensive_actual_dollars=float(observed[expensive].sum()),tail_prevented_count=int((saved&high).sum()),tail_actual_count=int(high.sum()),
        construction_first_year=construction,executed_first_year=executed,observed_year_only_construction=float(z[0]),no_future_leak_construction=float(no[0]),completion_year=int(year+z[2]),complete=bool(z[5] and no[5]),
        rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),underestimate_fraction=float(np.mean(err<0)),total_bias_pct=float(100*err.sum()/actual.sum()),top10_event_cost_capture=float(actual[top].sum()/actual.sum()))

def evaluate():
    p,h,g,rev=load();rows=[]
    for year in a.YEARS:
        if year==2026:
            past=pd.DataFrame(rows);eligible=past.groupby(['model','k']).complete.min();past=past[past.set_index(['model','k']).index.map(eligible).to_numpy()];dev=past[past.model.ne('CMS001')].groupby(['model','k']).cost_capture.mean().reset_index().sort_values(['cost_capture','k','model'],ascending=[False,True,True])
            winner=dev[dev.k.gt(1)].iloc[0];control=dev[dev.k.eq(1)].iloc[0]
            a.dump(HERE/'locked_before_2026_evaluation.json',dict(model=winner.model,k=float(winner.k),best_symmetric_model=control.model,selection_years=[2023,2024,2025],code_sha256=a.sha(Path(__file__))))
        q=pd.read_parquet(HERE/'features'/f'{year}.parquet');risk=probability(p,year);state=initial(p,h,year)
        for name in a.MODELS:
            for k in a.RATIOS:
                pred=np.load(HERE/'fits'/str(year)/f'{name}_k{k:.1f}.npy');cost,missing=fullcost(p,h,year,q.pipe_id,pred)
                seq=order(g,rev,state,risk,cost)
                row=dict(model=name,k=k,missing_prediction_ids=','.join(missing),**metric(p,h,g,year,seq,cost));rows.append(row)
        if year>=2024:
            cost,missing=cms(p,h,year);seq=order(g,rev,state,risk,cost)
            rows.append(dict(model='CMS001',k=1.,missing_prediction_ids=','.join(missing),**metric(p,h,g,year,seq,cost)))
        pd.DataFrame(rows).to_csv(HERE/'annual_metrics.csv',index=False)
        print('EVALUATED',year,flush=True)
    frame=pd.DataFrame(rows)
    stats=['cost_capture','prevented_dollars','leak_recall','rmse','mae','top10_event_cost_capture','underestimate_fraction']
    development=frame[frame.year.le(2025)&frame.model.ne('CMS001')];eligible=development.groupby(['model','k']).complete.min();development=development[development.set_index(['model','k']).index.map(eligible).to_numpy()]
    dev=development.groupby(['model','k'])[stats].mean().reset_index()
    dev=dev.sort_values(['cost_capture','k','model'],ascending=[False,True,True]);dev.to_csv(HERE/'development_2023_2025.csv',index=False)
    winner=dev[dev.k.gt(1)].iloc[0];control=dev[dev.k.eq(1)].iloc[0]
    selection=dict(model=winner.model,k=float(winner.k),selection_years=[2023,2024,2025],confirmation_year=2026,objective='maximum mean first-year realized repair-dollar capture under $10m',best_symmetric_model=control.model,selected_development_capture=float(winner.cost_capture),best_symmetric_development_capture=float(control.cost_capture))
    locked=json.loads((HERE/'locked_before_2026_evaluation.json').read_text());assert locked['model']==selection['model'] and locked['k']==selection['k']
    a.dump(HERE/'locked_selection.json',selection)
    forwards=[]
    for year in [2024,2025,2026]:
        previous=frame[frame.year.lt(year)&frame.model.ne('CMS001')];eligible=previous.groupby(['model','k']).complete.min();previous=previous[previous.set_index(['model','k']).index.map(eligible).to_numpy()]
        past=previous.groupby(['model','k']).cost_capture.mean().reset_index().sort_values(['cost_capture','k','model'],ascending=[False,True,True])
        for label,allowed in [('asymmetric',past.k.gt(1)),('symmetric',past.k.eq(1))]:
            chosen=past[allowed].iloc[0];now=frame[(frame.year.eq(year))&(frame.model.eq(chosen.model))&(frame.k.eq(chosen.k))].iloc[0].to_dict();forwards.append(dict(selection_max_year=year-1,selector=label,**now))
    pd.DataFrame(forwards).to_csv(HERE/'forward_selection.csv',index=False)
    confirm=frame[frame.year.eq(2026)]
    targets=[(selection['model'],selection['k']),(selection['model'],1.),(selection['best_symmetric_model'],1.),('CMS001',1.)]
    selected=pd.DataFrame([confirm[(confirm.model.eq(n))&(confirm.k.eq(k))].iloc[0] for n,k in dict.fromkeys(targets)])
    selected.to_csv(HERE/'confirmation_2026.csv',index=False)
    new=selected.iloc[0];others=selected.iloc[1:];selection['confirmation_beats_all_controls']=bool((new.prevented_dollars>others.prevented_dollars).all());selection['confirmation_prevented_dollars']=float(new.prevented_dollars)
    a.dump(HERE/'selection_result.json',selection)
    print(json.dumps(selection),flush=True)

def simulations(g,p,h,year,orders,prefix=''):
    records=[]
    for world in [0,1]:
        state,first,repair=b.worlds(p,h,year,draws=128,observed=year<2027,world=world,seed=20261109440)
        values={}
        for label,sequence in orders.items():
            z=b.evaluate(sequence,g[8],g[9],g[10],g[2],state,g[3],first,repair);assert np.isfinite(z).all() and (z[:,5]==1).all()
            values[label]=z[:,0]
            for s in range(len(z)):records.append(dict(year=year,world=world,draw=s,model=label,total_cost=float(z[s,0]),construction=float(z[s,1]),repair=float(z[s,2]),completion_year=int(year+z[s,3])))
    frame=pd.DataFrame(records);frame.to_csv(HERE/f'{prefix}simulated_{year}.csv',index=False)
    pivot=frame.pivot(index=['world','draw'],columns='model',values='total_cost');reference=pivot['selected']
    result=[]
    for label in pivot:
        delta=reference-pivot[label]
        result.append(dict(model=label,mean_total_cost=float(pivot[label].mean()),selected_minus_model=float(delta.mean()),paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),selected_wins_fraction=float((delta<0).mean()),draws=len(delta),interpretation='train-only hypothetical future; first year observed only for historical origins'))
    pd.DataFrame(result).to_csv(HERE/f'{prefix}simulated_summary_{year}.csv',index=False)

def deploy():
    selection=json.loads((HERE/'selection_result.json').read_text());name=selection['model'];k=selection['k'];best=selection['best_symmetric_model']
    for n,kk in dict.fromkeys([(name,k),(name,1.),(best,1.)]):a.train(n,[2027],[kk])
    p,h,g,rev=load();orders={};costs={}
    for year in [2026,2027]:
        q=pd.read_parquet(HERE/'features'/f'{year}.parquet');risk=probability(p,year);state=initial(p,h,year);orders={};costs={}
        for label,n,kk in [('selected',name,k),('same_model_k1',name,1.),('best_symmetric',best,1.)]:
            pred=np.load(HERE/'fits'/str(year)/f'{n}_k{kk:.1f}.npy');cost,missing=fullcost(p,h,year,q.pipe_id,pred);costs[label]=cost;orders[label]=order(g,rev,state,risk,cost)
        cost,missing=cms(p,h,year);costs['CMS001']=cost;orders['CMS001']=order(g,rev,state,risk,cost)
        simulations(g,p,h,year,orders)
        if year==2026: continue
        forecasts=pd.DataFrame(dict(pipe_id=p['Pipe ID'],active=~state,leak_probability=risk,**costs));forecasts.to_csv(HERE/'forecast_costs_2027.csv',index=False)
        folder=HERE/'submission_files';folder.mkdir(exist_ok=True)
        for label,seq in orders.items():
            path=folder/(label+'_common_circles.csv')
            lines=[f'"({g[5][j,0]:.9f}, {g[5][j,1]:.9f})",{g[6][j]:.9f}\n' for j in seq];path.write_text(''.join(lines),encoding='utf8')
            report=b.validate(path,p,h.drop(columns='year'),g[0],g[1],(g[0]+g[1])/2,g[2],g[3]);assert report['completed_by_2051']
        a.dump(HERE/'deployment.json',dict(selection=selection,official_submission=False,public_score=None,csv_path=str(folder/'selected_common_circles.csv'),csv_sha256=a.sha(folder/'selected_common_circles.csv'),model_artifact=str(HERE/'fits/2027'/f'{name}_k{k:.1f}.joblib')))
    print('DEPLOYMENT COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','deploy']);args=parser.parse_args()
    if args.stage=='evaluate':evaluate()
    else:deploy()
