"""Cost-only same-planned-year mergers of the best submitted circle plan."""
from collections import Counter
from itertools import combinations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/'submissions/2026-10-28_ryota_early3'
sys.path.insert(0,str(BASE))
from optimize import ROOT,DATA,Problem,execute,risk_model,cents
from early_three_years import read,fit_fine

OUT=HERE/'outputs'
SOURCE=ROOT/'outputs_early3/early3_adaptive.csv'

def trace(pr,units,first=None):
    n=len(pr.p);first=np.full(n,25) if first is None else first
    state=pr.start.copy();cursor=balance=0;unit_year=np.full(len(units),-1,int)
    pipe_year=np.full(n,2052,int);pipe_year[pr.start]=2026
    active=[np.array([],int) for _ in units];total=0
    for yi,year in enumerate(range(2027,2052)):
        state|=first<yi;balance+=1_000_000_000
        while cursor<len(units):
            u=units[cursor];ids=u[2][~state[u[2]]]
            if not len(ids):cursor+=1;continue
            cost=pr.cost(u,state)
            if cost>balance:break
            balance-=cost;total+=cost;unit_year[cursor]=year;pipe_year[ids]=year
            active[cursor]=ids;state[ids]=True;cursor+=1
    return unit_year,pipe_year,active,total

def generate(pr,units,predict,max_gap=160):
    uy,py,active,_=trace(pr,units)
    leakage={};survival=np.ones(len(pr.p))
    for year in range(2027,2052):
        leakage[year]=1-survival.copy();survival*=1-predict(year)
    proposals=[];seen=set();evaluated=0
    for year in sorted(set(uy)-{-1}):
        indices=np.flatnonzero(uy==year);xy=np.array([units[i][0] for i in indices]);tree=cKDTree(xy)
        _,neighbors=tree.query(xy,k=min(12,len(indices)),distance_upper_bound=80)
        if neighbors.ndim==1:neighbors=neighbors[:,None]
        for row,near in enumerate(neighbors):
            for col in near:
                if col>=len(indices):continue
                i,j=sorted((int(indices[row]),int(indices[col])))
                if i==j or j-i>max_gap or (i,j) in seen:continue
                seen.add((i,j));a,b=units[i],units[j]
                if np.linalg.norm(a[0]-b[0])>1.7*(a[1]+b[1])+2:continue
                original=np.union1d(a[2],b[2])
                if len(original)>12:continue
                merged=pr.unit(original);evaluated+=1
                extras=np.setdiff1d(merged[2],original)
                # No newly included live pipe assigned to this or a later year.
                if np.any((~pr.start[extras])&(py[extras]>=year)):continue
                fixed_saving=a[3]+b[3]-merged[3]
                if fixed_saving<25:continue
                leak=leakage[year]
                empty_a=float(np.prod(leak[active[i]]));empty_b=float(np.prod(leak[active[j]]))
                pending=merged[2][(~pr.start[merged[2]])&(py[merged[2]]>=year)]
                empty_m=float(np.prod(leak[pending]))
                expected=(1-empty_a)*a[3]+(1-empty_b)*b[3]-(1-empty_m)*merged[3]
                if expected<10:continue
                proposals.append((expected,fixed_saving,i,j,merged))
        print('merge search year',year,'evaluated',evaluated,'accepted candidates',len(proposals),flush=True)
    used=set();selected={};removed=set();audit=[]
    for expected,saving,i,j,merged in sorted(proposals,key=lambda x:(-x[0],x[2],x[3])):
        if i in used or j in used:continue
        used.update((i,j));selected[i]=merged;removed.add(j)
        audit.append(dict(first_row=i+1,second_row=j+1,planned_year=int(uy[i]),row_gap=j-i,
            fixed_cost_saving=saving,estimated_future_leak_adjusted_saving=expected,
            old_radius_a=units[i][1],old_radius_b=units[j][1],merged_radius=merged[1]))
    output=[selected.get(i,u) for i,u in enumerate(units) if i not in removed]
    return output,audit,dict(evaluated=evaluated,positive_proposals=len(proposals),selected_pairs=len(audit),
        max_row_gap=max_gap,max_new_design_pipes=12,nominal_fixed_savings=sum(a['fixed_cost_saving'] for a in audit))

def compare(pr,arms,models,sims):
    hist=pr.hist.merge(pr.p[['Pipe ID','Surface']],on='Pipe ID')
    pools=hist.groupby('Surface').Cost.apply(lambda x:np.array([cents(v) for v in x])).to_dict();records=[]
    for model,predict in models.items():
        for scale in [.5,1.,2.]:
            rng=np.random.default_rng(446)
            h=np.column_stack([np.clip(predict(y)*scale,0,.95) for y in range(2027,2052)]);h[pr.start]=0
            for sim in range(sims):
                hit=rng.random(h.shape)<h;first=np.where(hit.any(1),hit.argmax(1),25)
                cost=np.zeros(len(pr.p),np.int64)
                for s,pool in pools.items():
                    ids=np.flatnonzero(pr.surface==s);cost[ids]=rng.choice(pool,len(ids))
                for name,units in arms.items():
                    result=execute(pr,units,first,cost);result.pop('annual')
                    records.append(dict(model=model,scale=scale,simulation=sim,arm=name,**result))
            print('simulated',model,scale,flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'simulations.csv',index=False);summary=[]
    for (model,scale,arm),g in df.groupby(['model','scale','arm']):
        b=df[(df.model==model)&(df.scale==scale)&(df.arm=='best_76_91')].set_index('simulation')
        gg=g.set_index('simulation');d=gg.score-b.score;se=float(d.std(ddof=1)/math.sqrt(len(d)))
        summary.append(dict(model=model,scale=scale,arm=arm,mean_score=float(g.score.mean()),
            score_gain=float(d.mean()),gain_mc95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
            construction_saved=float((b.construction-gg.construction).mean()),
            repair_saved=float((b.repair-gg.repair).mean()),feasible_fraction=float(g.feasible.mean())))
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sims',type=int,default=30);args=ap.parse_args();assert args.sims>=2
    OUT.mkdir(exist_ok=True);pr=Problem();baseline=read(pr,SOURCE);fine,_=fit_fine(pr,2026,.5,.85)
    candidate,audit,search=generate(pr,baseline,fine)
    path=OUT/'submission_construction_savings.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        for c,r,*_ in candidate:w.writerow([f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}'])
    candidate=read(pr,path)
    # Independent exported geometry check for every original unit's contained pipes.
    old_cover=np.zeros(len(pr.p),bool);new_cover=old_cover.copy()
    for u in baseline:old_cover[u[2]]=True
    for u in candidate:new_cover[u[2]]=True
    assert new_cover[old_cover].all()
    _,old_year,_,old_cost=trace(pr,baseline);_,new_year,_,new_cost=trace(pr,candidate)
    protected=old_year<2052;delayed=protected&(new_year>old_year)
    assert not delayed.any(),'A baseline pipe replacement is delayed in no-future-leak schedule'
    assert new_cost<old_cost,'No actual construction saving'
    pd.DataFrame(audit).to_csv(OUT/'merges.csv',index=False)
    audits={}
    for name,units in [('best_76_91',baseline),('construction_savings',candidate)]:
        r=execute(pr,units);annual=r.pop('annual');assert r['feasible'] and r['remaining_iron']==0
        pd.DataFrame(annual).to_csv(OUT/f'{name}_no_leak_annual.csv',index=False);audits[name]=r
    comparisons=compare(pr,{'best_76_91':baseline,'construction_savings':candidate},{'fine':fine,'old':risk_model(pr)[0]},args.sims)
    report=dict(cost_model_version='2026-10-26-area-exponent',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_units=len(baseline),new_units=len(candidate),search=search,
        forecast_changed=False,coverage_preserved=True,no_future_leak_delayed_pipes=int(delayed.sum()),
        no_future_leak_advanced_pipes=int((protected&(new_year<old_year)).sum()),
        exact_no_future_leak_construction_saving=(old_cost-new_cost)/100,arms=audits,comparison=comparisons,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')},
        timing_limitation='Same-year mergers refer to baseline no-future-leak schedule, not guaranteed identical timing under every unknown future.',
        original_iron_coverage_note='Preserves all baseline covered pipes; historical original iron coverage limitation inherited, not supplemented.',official_score=None)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',search,'actual no-leak savings',(old_cost-new_cost)/100,flush=True)

if __name__=='__main__':main()
