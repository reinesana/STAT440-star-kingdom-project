"""Budget-aware comparison with fixed timing and geometry controls."""
from pathlib import Path
import sys,json,csv,heapq,importlib.util,time,argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score,brier_score_loss
from numba import njit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import tail_model as tm
ROOT=tm.ROOT;OLD=tm.OLD
sys.path.insert(0,str(OLD));import evaluate as legacy
sys.path.insert(0,str(OLD));import g17 as oldg17
b=legacy.b

class AssignmentPlan:
    """Cache original assignments and dependencies for the fixed G17 circles."""
    def __init__(self,g,state):
        self.g=g;self.state=state;self.owner=np.full(len(state),-1,int);owned=[];cost=[]
        for j,ids in enumerate(g[7]):
            new=ids[(~state[ids])&(self.owner[ids]<0)];self.owner[new]=j;owned.append(new)
            cost.append(int(np.floor((g[10][j]+200*g[2][new].sum())*100+.5)) if len(new) else 0)
        self.active=np.flatnonzero(np.array(cost)>0);self.free=np.flatnonzero(np.array(cost)==0);self.cost=np.array(cost,dtype=float)/100
        off=np.r_[0,np.cumsum([len(x) for x in owned])].astype(np.int64);flat=np.concatenate(owned).astype(np.int64)
        self.owned=csr_matrix((np.ones(len(flat)),flat,off),shape=(len(g[4]),len(state)))
        self.children=[[] for _ in g[4]];self.pending=np.zeros(len(g[4]),int)
        for j in self.active:
            deps=set(self.owner[g[7][j]]);deps={int(k) for k in deps if k>=0 and k!=j};assert all(k<j for k in deps)
            self.pending[j]=len(deps)
            for k in deps:self.children[k].append(int(j))
        self.total=float(sum(int(round(x*100)) for x in self.cost))/100
    def order(self,values):
        priority=np.asarray(self.owned@values).ravel()/np.maximum(self.cost,1)
        pending=self.pending.copy();heap=[(-priority[j],int(j)) for j in self.active if pending[j]==0];heapq.heapify(heap);out=[]
        while heap:
            _,j=heapq.heappop(heap);out.append(j)
            for child in self.children[j]:
                pending[child]-=1
                if pending[child]==0:heapq.heappush(heap,(-priority[child],child))
        assert len(out)==len(self.active)
        return np.r_[out,self.free].astype(np.int64)

@njit(cache=False)
def first_year(sequence,off,flat,base,length,state):
    live=state.copy();budget=1_000_000_000;spent=0.;count=0
    for j in sequence:
        ll=0.;n=0
        for k in range(off[j],off[j+1]):
            i=flat[k]
            if not live[i]:ll+=length[i];n+=1
        if n==0:continue
        cents=int(np.floor((base[j]+200*ll)*100+.5))
        if cents>budget:break
        budget-=cents;spent+=cents/100.;count+=1
        for k in range(off[j],off[j+1]):live[flat[k]]=True
    return live,spent,count

def data():
    p=pd.read_csv(ROOT/'data/pipes.csv');h=pd.read_csv(ROOT/'data/train.csv');h['year']=pd.to_datetime(h.Date).dt.year
    common=b.geometry(p);g17,source=oldg17.geometry(p)
    incidence=csr_matrix((np.ones(len(common[9])),common[9],common[8]),shape=(len(common[4]),len(p)));reverse=incidence.T.tocsr()
    return p,h,{'common':common,'G17':g17},reverse

def cms(p,h,year):
    # Preserve prior experiment files: reconstruct without calling its audit writer.
    if year==2027:
        c=pd.read_csv(ROOT/'output/2026-11-02_robustness_paper/inputs/CMS001_predictions_2027.csv');return legacy.fullcost(p,h,year,c.pipe_id,c.predicted_repair_cost)[0]
    folder=ROOT/'github-main/notebooks/Takito_Kobayashi/2026-10-28/ThreeModelStacking'
    q=pd.read_csv(ROOT/f'output/2026-10-28_cost_model_comparison/combination_policies/cost_inventory_{year}.csv')
    pipe=p.set_index('Pipe ID');q['material_surface']=q.pipe_id.map(pipe.Material).fillna('__unknown__')+' | '+q.pipe_id.map(pipe.Surface).fillna('__unknown__')
    config=json.loads((folder/f'config_{year}.json').read_text())['models']['conditional_material_surface_rmse'];global_w=np.array(config['global_weights']);weights=np.tile(global_w,(len(q),1))
    for key,z in config['groups'].items():
        factor=z['n']/(z['n']+config['tau']);weights[q.material_surface.eq(key)]=factor*np.array(z['local_weights'])+(1-factor)*global_w
    pred=(q[['takito_catboost','anthony_rf_log_mean','rion_lgb_log_mean_refit']].to_numpy()*weights).sum(1)
    return legacy.fullcost(p,h,year,q.pipe_id,pred)[0]

def forecasts(p,h,year):
    result={};probability={}
    for label in tm.labels():
        with np.load(HERE/'predictions'/str(year)/(label+'.npz')) as z:
            cost,missing=legacy.fullcost(p,h,year,z['pipe_id'],z['predicted_cost']);result[label]=cost
            probability[label]={x:p['Pipe ID'].map(pd.Series(z[x],index=z['pipe_id'])).to_numpy() for x in ['p_tail95','p_tail99']}
    q=pd.read_parquet(OLD/'features'/f'{year}.parquet')
    for label,k in [('standard_LGBM',1.),('asymmetric_LGBM_2p4',2.4)]:result[label]=legacy.fullcost(p,h,year,q.pipe_id,np.load(OLD/'fits'/str(year)/f'lightgbm_k{k:.1f}.npy'))[0]
    if year>=2024:result['CMS001']=cms(p,h,year)
    # The old conditional top-10% test is now evaluated as a full-pipe policy.
    ref='lightgbm_basic_bins3_raw';prob=probability[ref]['p_tail99'];result['tail_probability_only']=np.nan_to_num(prob)
    result['F168_only']=np.ones(len(p));result['F168_only'][legacy.initial(p,h,year)]=0
    return result,probability

def metric(p,h,g,year,sequence,cost):
    state=legacy.initial(p,h,year);event=p['Pipe ID'].isin(h.loc[h.year.eq(year),'Pipe ID']).to_numpy()
    actual=p['Pipe ID'].map(h.loc[h.year.eq(year)].set_index('Pipe ID').Cost).fillna(0).to_numpy()
    live,spent,count=first_year(sequence,g[8],g[9],g[10],g[2],state);prevented=event&live&~state
    threshold=float(h.loc[h.year.lt(year),'Cost'].quantile(.99));high=event&(actual>=threshold);q95=float(h.loc[h.year.lt(year),'Cost'].quantile(.95));high95=event&(actual>=q95)
    top1=event&(actual>=np.quantile(actual[event],.99))
    no=legacy.old.replay(sequence,g[8],g[9],g[10],g[2],state,g[3],np.zeros(len(p),bool),np.zeros(len(p)))
    replay=legacy.old.replay(sequence,g[8],g[9],g[10],g[2],state,g[3],event,actual)
    amount=float(actual[prevented].sum());np.testing.assert_allclose(amount,actual.sum()-replay[1],atol=1e-5)
    e=cost[event]-actual[event]
    return dict(year=year,high_threshold_train99=threshold,high_count=int(high.sum()),high_prevented=int((high&prevented).sum()),high_recall=float((high&prevented).sum()/high.sum()),
        high_dollar_capture=float(actual[high&prevented].sum()/actual[high].sum()),high95_recall=float((high95&prevented).sum()/high95.sum()),
        observed_top1_prevented=int((top1&prevented).sum()),observed_top1_count=int(top1.sum()),prevented_dollars=amount,cost_capture=amount/actual.sum(),
        rmse=float(np.sqrt(np.mean(e**2))),mae=float(abs(e).mean()),total_bias_pct=float(100*e.sum()/actual.sum()),
        spent_first_year=float(spent),executed_first_year=int(count),leak_recall=float(prevented.sum()/event.sum()),no_future_construction=float(no[0]),
        no_future_complete=bool(no[5]),observed_year_only_complete=bool(replay[5]),completion_offset=int(no[2])),prevented,high,actual

def evaluate():
    p,h,geometries,reverse=data();rows=[];diagnostics=[]
    for year in [2023,2024,2025,2026]:
        if year==2026:
            frame=pd.DataFrame(rows);dev=frame[frame.geometry.eq('common')&frame.year.isin([2024,2025])&frame.model.isin(tm.labels())]
            summary=dev.groupby('model').agg(high_recall=('high_recall','mean'),cost_capture=('cost_capture','mean'),worst_high_recall=('high_recall','min'),feasible=('no_future_complete','all')).reset_index()
            summary=summary[summary.feasible].sort_values(['high_recall','cost_capture','model'],ascending=[False,False,True]);assert len(summary)
            summary.to_csv(HERE/'development_selection_2024_2025.csv',index=False);winner=summary.iloc[0]
            tm.dump(HERE/'locked_before_2026.json',dict(model=winner.model,selection_years=[2024,2025],selection_geometry='common',primary='mean recall of observed leak costs above train-only 99th percentile under $10m',development_high_recall=float(winner.high_recall),development_cost_capture=float(winner.cost_capture),source_code_sha256=tm.sha(Path(__file__))))
            print('LOCKED',winner.model,flush=True)
        costs,probs=forecasts(p,h,year);risk=legacy.probability(p,year);state=legacy.initial(p,h,year)
        for geometry,g in geometries.items():
            if geometry=='G17' and year==2023:continue
            plan=AssignmentPlan(g,state) if geometry=='G17' else None
            for name,cost in costs.items():
                sequence=plan.order(risk*cost) if geometry=='G17' else legacy.order(g,reverse,state,risk,cost)
                row,prevented,high,actual=metric(p,h,g,year,sequence,cost)
                if name in ['tail_probability_only','F168_only']:row.update(rmse=None,mae=None,total_bias_pct=None)
                rows.append(dict(model=name,geometry=geometry,**row));folder=HERE/'decisions'/str(year);folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{geometry}_{name}.npz',prevented=prevented,high=high,actual_cost=actual,order=sequence)
            if geometry=='G17':
                sequence=np.arange(len(g[4]),dtype=np.int64);row,prevented,high,actual=metric(p,h,g,year,sequence,np.zeros(len(p)));row.update(rmse=None,mae=None,total_bias_pct=None)
                rows.append(dict(model='G17_original',geometry=geometry,**row));np.savez_compressed(HERE/'decisions'/str(year)/'G17_G17_original.npz',prevented=prevented,high=high,actual_cost=actual,order=sequence)
        for name,prob in probs.items():
            live_event=p['Pipe ID'].isin(h.loc[h.year.eq(year),'Pipe ID']).to_numpy();actual=p['Pipe ID'].map(h.loc[h.year.eq(year)].set_index('Pipe ID').Cost).to_numpy()
            cfg=json.loads((HERE/'predictions'/str(year)/('_'.join(name.split('_')[:2])+'_fit.json')).read_text())
            for head,cutoff in zip(['p_tail95','p_tail99'],cfg['thresholds']):
                pp=prob[head][live_event];y=actual[live_event]>=cutoff
                if not np.isfinite(pp).all():continue
                top=np.argsort(-pp,kind='stable')[:int(np.ceil(.1*len(pp)))]
                diagnostics.append(dict(model=name,year=year,head=head,n=len(pp),positive_count=int(y.sum()),AP=float(average_precision_score(y,pp)),Brier=float(brier_score_loss(y,pp)),conditional_top10_recall=float(y[top].sum()/y.sum())))
        pd.DataFrame(rows).to_csv(HERE/'annual_metrics.csv',index=False);pd.DataFrame(diagnostics).to_csv(HERE/'conditional_classifier_metrics.csv',index=False)
        print('EVALUATED',year,flush=True)
    frame=pd.DataFrame(rows);lock=json.loads((HERE/'locked_before_2026.json').read_text());controls=[lock['model'],'CMS001','standard_LGBM','asymmetric_LGBM_2p4','tail_probability_only','F168_only','G17_original']
    frame[frame.year.eq(2026)&frame.model.isin(controls)].to_csv(HERE/'confirmation_2026.csv',index=False)
    forward=[]
    for year in [2025,2026]:
        past=frame[frame.geometry.eq('common')&frame.year.between(2024,year-1)&frame.model.isin(tm.labels())]
        ranked=past.groupby('model').agg(high_recall=('high_recall','mean'),cost_capture=('cost_capture','mean'),feasible=('no_future_complete','all')).reset_index().query('feasible').sort_values(['high_recall','cost_capture','model'],ascending=[False,False,True])
        name=ranked.iloc[0].model
        for geo in ['common','G17']:
            row=frame[frame.year.eq(year)&frame.geometry.eq(geo)&frame.model.eq(name)].iloc[0].to_dict();forward.append(dict(selection_max_year=year-1,**row))
    pd.DataFrame(forward).to_csv(HERE/'forward_selection.csv',index=False)
    print('EVALUATION COMPLETE',flush=True)

def simulate(g,p,h,orders,prefix):
    rows=[]
    for world in [0,1]:
        state,first,repair=b.worlds(p,h,2027,draws=128,observed=False,world=world,seed=202611091117)
        for label,sequence in orders.items():
            z=b.evaluate(sequence,g[8],g[9],g[10],g[2],state,g[3],first,repair);assert np.isfinite(z).all() and (z[:,5]==1).all()
            for j in range(len(z)):rows.append(dict(model=label,world=world,draw=j,total_cost=float(z[j,0]),construction=float(z[j,1]),repair=float(z[j,2])))
    frame=pd.DataFrame(rows);frame.to_csv(HERE/(prefix+'_simulations.csv'),index=False);pivot=frame.pivot(index=['world','draw'],columns='model',values='total_cost');rows=[]
    for name in pivot:
        delta=pivot['selected']-pivot[name]
        rows.append(dict(model=name,mean_total_cost=float(pivot[name].mean()),selected_minus_model=float(delta.mean()),paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),selected_win_fraction=float((delta<0).mean()),scenarios=len(delta)))
    pd.DataFrame(rows).to_csv(HERE/(prefix+'_simulation_summary.csv'),index=False)

def deploy():
    lock=json.loads((HERE/'locked_before_2026.json').read_text());name=lock['model'];p,h,geometries,reverse=data();costs,probs=forecasts(p,h,2027);risk=legacy.probability(p,2027);state=legacy.initial(p,h,2027)
    folder=HERE/'submission_files';folder.mkdir(exist_ok=True);validations=[]
    pd.DataFrame(dict(pipe_id=p['Pipe ID'],active=~state,F168_next_year=risk,selected_cost=costs[name],selected_p95=probs[name]['p_tail95'],selected_p99=probs[name]['p_tail99'],CMS001=costs['CMS001'])).to_csv(HERE/'forecast_2027.csv',index=False)
    for geometry,g in geometries.items():
        plan=AssignmentPlan(g,state) if geometry=='G17' else None;orders={}
        for label,candidate in [('selected',name),('CMS001','CMS001'),('standard_LGBM','standard_LGBM')]:
            cost=costs[candidate];sequence=plan.order(risk*cost) if plan else legacy.order(g,reverse,state,risk,cost);orders[label]=sequence
            path=folder/f'{label}_{geometry}.csv';path.write_text(''.join(f'"({g[5][j,0]:.9f}, {g[5][j,1]:.9f})",{g[6][j]:.9f}\n' for j in sequence),encoding='utf8')
            report=b.validate(path,p,h.drop(columns='year'),g[0],g[1],(g[0]+g[1])/2,g[2],g[3]);assert report['completed_by_2051']
            if plan:np.testing.assert_allclose(report['construction_cost'],plan.total,atol=.01)
            validations.append(dict(model=candidate,label=label,geometry=geometry,sha256=tm.sha(path),validation=report))
        if geometry=='G17':orders['G17_original']=np.arange(len(g[4]),dtype=np.int64)
        simulate(g,p,h,orders,geometry)
        print('DEPLOYED',geometry,flush=True)
    tm.dump(HERE/'deployment.json',dict(model=name,official_submission=False,public_score=None,validations=validations))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','deploy']);args=parser.parse_args()
    if args.stage=='evaluate':evaluate()
    else:deploy()
