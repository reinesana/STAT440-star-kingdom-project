"""Three-year risk-aware adaptive grouping; preserves all previous outputs."""
from collections import defaultdict
from pathlib import Path
import argparse, csv, heapq, json, math, time, hashlib
import numpy as np
import pandas as pd
from optimize import ROOT, DATA, Problem, YEARS, RATES, candidates, choose, risk_model, execute, cents

OUT=ROOT/'outputs_early3'

def fit_fine(pr,cutoff,width=1.,decay=.85):
    """Kernel-smoothed half-year age hazards; recent years receive more weight."""
    mats=sorted(set(pr.mat)-{'polyurethane'});surfaces=sorted(RATES)
    centers=np.arange(0,121,.5);K=np.exp(-.5*((centers[:,None]-centers)/width)**2)
    counts={};global_k=global_e=0.
    for m in mats:
        for s in surfaces:
            ev=np.zeros(len(centers));ex=np.zeros(len(centers))
            for year in range(2019,cutoff+1):
                mask=((pr.mat==m)&(pr.surface==s)&(np.isnan(pr.event)|(pr.event>=year)))
                ix=np.flatnonzero(mask);bins=np.clip(np.rint((pr.age[ix]+year-2027)/.5).astype(int),0,len(centers)-1)
                w=decay**(cutoff-year)
                ex+=w*np.bincount(bins,minlength=len(centers))
                ev+=w*np.bincount(bins,weights=(pr.event[ix]==year).astype(float),minlength=len(centers))
            counts[m,s]=(ev,ex);global_k+=ev.sum();global_e+=ex.sum()
    global_rate=global_k/max(global_e,1);tables={}
    for m in mats:
        ev=sum(counts[m,s][0] for s in surfaces);ex=sum(counts[m,s][1] for s in surfaces)
        rate=(ev.sum()+100*global_rate)/(ex.sum()+100)
        # Age curve shrinks only lightly, preserving narrow wear-out peaks.
        prior=(K@ev+30*rate)/(K@ex+30)
        for s in surfaces:
            se,sx=counts[m,s];tables[m,s]=(K@se+80*prior)/(K@sx+80)
    def predict(year):
        age=pr.age+year-2027;pred=np.zeros(len(pr.p))
        for (m,s),table in tables.items():
            ix=np.flatnonzero((pr.mat==m)&(pr.surface==s))
            pred[ix]=np.interp(age[ix],centers,table)
        return np.clip(pred,0,.95)
    _,severity=risk_model(pr,cutoff)
    return predict,severity

def metrics(pr,predict,year):
    mask=(np.isnan(pr.event)|(pr.event>=year))&(pr.mat!='polyurethane')
    y=(pr.event[mask]==year).astype(float);p=np.clip(predict(year)[mask],1e-12,1-1e-12)
    top=np.argsort(-p,kind='stable')[:math.ceil(len(p)*.1)]
    return dict(year=year,rows=len(y),observed=float(y.mean()),predicted=float(p.mean()),
        log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),brier=float(np.mean((y-p)**2)),
        top_decile_event_recall=float(y[top].sum()/max(y.sum(),1)))

def select_model(pr):
    records=[];options=[(.5,.85),(1.,.85),(2.,.85),(.5,1.),(1.,1.),(2.,1.)]
    for width,decay in options:
        scores=[]
        for year in [2024,2025]:
            pred,_=fit_fine(pr,year-1,width,decay);r=metrics(pr,pred,year)
            scores.append(r['log_loss']);records.append(dict(model='fine',width=width,decay=decay,**r))
        print('risk candidate',width,decay,'loss',np.mean(scores),flush=True)
    best=min(options,key=lambda c:np.mean([r['log_loss'] for r in records if r['width']==c[0] and r['decay']==c[1]]))
    holdout=[]
    for name in ['old','fine']:
        pred,_=risk_model(pr,2025) if name=='old' else fit_fine(pr,2025,*best)
        holdout.append(dict(model=name,**metrics(pr,pred,2026)))
    return best,records,holdout

def benefits(predict,severity,year,end):
    survival=np.ones(len(severity));value=np.zeros(len(severity))
    for y in range(year,end+1):
        h=predict(y);value+=survival*h*severity;survival*=1-h
    return value

def early_schedule(pr,singles,cands,predict,severity):
    """Exact marginal updates: no duplicated risk credit for already replaced pipes."""
    catalog=list(singles.values())+[u for _,_,u in cands]
    incidence=defaultdict(list)
    for j,u in enumerate(catalog):
        for i in u[2]:incidence[int(i)].append(j)
    state=pr.start.copy();selected=[];audit=[];balance=0
    versions=np.zeros(len(catalog),int)
    for year in range(2027,2030):
        balance+=1_000_000_000
        value=benefits(predict,severity,year,2029);value[pr.start]=0
        heap=[]
        def push(j):
            u=catalog[j];ids=u[2][~state[u[2]]]
            if not len(ids) or not np.any(pr.iron[ids]):return
            cost=pr.cost(u,state);benefit=float(value[ids].sum())
            heapq.heappush(heap,(-benefit/max(cost/100,1),j,int(versions[j]),cost,benefit))
        for j in range(len(catalog)):push(j)
        count=0
        while heap:
            neg,j,version,cost,benefit=heapq.heappop(heap)
            if version!=versions[j]:continue
            if cost>balance:continue
            if benefit<=0:break
            u=catalog[j];ids=u[2][~state[u[2]]]
            selected.append(u);balance-=cost;state[ids]=True;count+=1
            audit.append(dict(planned_year=year,radius=u[1],new_pipes=len(ids),new_iron=int(pr.iron[ids].sum()),
                cost=cost/100,remaining_three_year_avoided_cost=benefit,benefit_per_dollar=-neg))
            affected=set(jj for i in ids for jj in incidence[int(i)])
            for jj in affected:versions[jj]+=1;push(jj)
        print('early plan',year,'circles',count,'balance',balance/100,flush=True)
    # Smaller design groups after the initial three years; only positive savings.
    used=set(np.flatnonzero(state));tail=[]
    for saving,ids,u in sorted(cands,key=lambda x:(-x[0]/max(x[2][1],1)**.7,x[1])):
        active=[i for i in ids if not state[i]]
        if len(active)<2 or len(ids)>3 or any(i in used for i in ids):continue
        used.update(ids);tail.append(u)
    for i,u in singles.items():
        if i not in used:tail.append(u)
    value=benefits(predict,severity,2030,2032);value[state]=0
    tail.sort(key=lambda u:(-float(value[u[2]].sum())/max(pr.cost(u,state)/100,1),u[1]))
    return selected+tail,audit,len(selected)

def write_read(pr,units,path):
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        for c,r,*_ in units:writer.writerow([f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}'])
    return read(pr,path)

def read(pr,path):
    out=[];covered=np.zeros(len(pr.p),bool)
    with path.open(newline='') as f:
        for row in csv.reader(f):
            assert len(row)==2
            c=np.array(list(map(float,row[0].strip('()').split(','))));r=float(row[1])
            assert np.isfinite(c).all() and math.isfinite(r) and r>=0
            ids=pr.grid.query(c,r)
            ids=ids[(np.linalg.norm(pr.a[ids]-c,axis=1)<=r)&(np.linalg.norm(pr.b[ids]-c,axis=1)<=r)]
            covered[ids]=True
            out.append((c,r,ids,100+sum(RATES[s] for s in set(pr.surface[ids]))*(math.pi*r*r)**.85))
    assert covered[pr.target].all()
    return out

def schedule_audit(pr,units,predict,severity):
    state=pr.start.copy();cursor=balance=0;out=[];value=benefits(predict,severity,2027,2029)
    protected=np.zeros(len(pr.p),bool);survival=np.ones(len(pr.p));expected_leaks=expected_loss=0.
    for year in YEARS:
        balance+=1_000_000_000;radii=[];count=repcount=0;spent=0
        while cursor<len(units):
            u=units[cursor];rep=u[2][~state[u[2]]]
            if not len(rep):cursor+=1;continue
            cost=pr.cost(u,state)
            if cost>balance:break
            balance-=cost;spent+=cost;state[rep]=True;protected[rep]=True
            repcount+=int(pr.iron[rep].sum());radii.append(u[1]);count+=1;cursor+=1
        if year<=2029:
            h=predict(year);alive=~pr.start&~protected
            expected_leaks+=float((survival*h*alive).sum())
            expected_loss+=float((survival*h*severity*alive).sum())
            survival*=1-h
        out.append(dict(year=year,circles=count,new_iron=repcount,cost=spent/100,
            median_radius=float(np.median(radii)) if radii else None,mean_radius=float(np.mean(radii)) if radii else None,
            p90_radius=float(np.quantile(radii,.9)) if radii else None,balance=balance/100))
    return out,dict(expected_first3_leaks=expected_leaks,expected_first3_repair=expected_loss)

def compare(pr,arms,models,severity,sims):
    hist=pr.hist.merge(pr.p[['Pipe ID','Surface']],on='Pipe ID')
    pools=hist.groupby('Surface').Cost.apply(lambda x:np.array([cents(v) for v in x])).to_dict()
    records=[]
    for model_name,predict in models.items():
        for scale in [.5,1.,2.]:
            rng=np.random.default_rng(441)
            h=np.column_stack([np.clip(predict(y)*scale,0,.95) for y in YEARS]);h[pr.start]=0
            for sim in range(sims):
                hit=rng.random(h.shape)<h;first=np.where(hit.any(1),hit.argmax(1),25)
                costs=np.zeros(len(pr.p),np.int64)
                for s,pool in pools.items():
                    ids=np.flatnonzero(pr.surface==s);costs[ids]=rng.choice(pool,len(ids))
                for name,units in arms.items():
                    r=execute(pr,units,first,costs);r.pop('annual')
                    # A second execution using the same event times but only early costs
                    # isolates first-three-year repair without modifying long-run geometry.
                    earlycost=costs.copy();earlycost[first>=3]=0
                    early=execute(pr,units,first,earlycost)
                    records.append(dict(model=model_name,scale=scale,simulation=sim,arm=name,
                        first3_repair=early['repair'],**r))
            print('simulation',model_name,scale,'done',flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'simulations.csv',index=False);summary=[]
    for (model,scale,arm),g in df.groupby(['model','scale','arm']):
        base=df[(df.model==model)&(df.scale==scale)&(df.arm=='previous_71_40')].set_index('simulation')
        d=g.set_index('simulation').score-base.score
        early=base.first3_repair-g.set_index('simulation').first3_repair
        se=float(d.std(ddof=1)/math.sqrt(len(d)))
        summary.append(dict(model=model,scale=scale,arm=arm,mean_score=float(g.score.mean()),
            paired_gain=float(d.mean()),paired_mc95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
            first3_repair=float(g.first3_repair.mean()),first3_repair_saved=float(early.mean()),
            feasible_fraction=float(g.feasible.mean())))
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sims',type=int,default=30);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);start=time.time();pr=Problem()
    best,tuning,holdout=select_model(pr)
    print('selected',best,'holdout',holdout,flush=True)
    predict,severity=fit_fine(pr,2026,*best);old_predict,_=risk_model(pr)
    singles,cands=candidates(pr,6)
    units,selection,early_count=early_schedule(pr,singles,cands,predict,severity)
    units=write_read(pr,units,OUT/'early3_adaptive.csv')
    previous=read(pr,ROOT/'outputs/multi_pipe.csv')
    # Isolates the model/horizon change with unchanged prior circle geometry.
    three=benefits(predict,severity,2027,2029);three[pr.start]=0
    reorder=sorted(previous,key=lambda u:(-float(three[u[2]].sum())/max(pr.cost(u,pr.start)/100,1),u[1]))
    reorder=write_read(pr,reorder,OUT/'early3_reorder_only.csv')
    arms={'previous_71_40':previous,'early3_adaptive':units,'early3_reorder_only':reorder};audits={}
    for name,arm in arms.items():
        annual,early=schedule_audit(pr,arm,predict,severity)
        pd.DataFrame(annual).to_csv(OUT/f'{name}_annual.csv',index=False)
        deterministic=execute(pr,arm);deterministic.pop('annual')
        assert deterministic['feasible'] and deterministic['remaining_iron']==0
        audits[name]=dict(rows=len(arm),no_future_leaks=deterministic,**early)
        print(name,audits[name],flush=True)
    pd.DataFrame(selection).to_csv(OUT/'early_selection.csv',index=False)
    pd.DataFrame(tuning).to_csv(OUT/'model_selection.csv',index=False)
    pd.DataFrame(holdout).to_csv(OUT/'holdout_2026.csv',index=False)
    result=compare(pr,arms,{'old':old_predict,'fine':predict},severity,args.sims)
    report=dict(selected_model=dict(width=best[0],decay=best[1]),holdout_2026=holdout,
        early_planned_circles=early_count,arms=audits,comparison=result,sims_per_scenario=args.sims,
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')},
        previous_sha256=hashlib.sha256((ROOT/'outputs/multi_pipe.csv').read_bytes()).hexdigest(),
        submissions={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('early3_*.csv') if 'annual' not in p.name},
        elapsed_seconds=time.time()-start)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
