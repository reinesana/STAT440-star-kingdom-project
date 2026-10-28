"""Shrink redundant circle coverage after earlier work; preserve each row assignment."""
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
SOURCE=REPO/'submissions/2026-10-28_ryota_construction_expanded/outputs/submission_construction_expanded.csv'

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

def generate(pr,units,predict=None):
    state=pr.start.copy();covered=np.zeros(len(pr.p),bool);output=[];audit=[]
    for j,old in enumerate(units):
        ids=old[2];rep=ids[~state[ids]];required=ids[~covered[ids]];chosen=old
        if len(rep) and 0<len(required)<len(ids):
            proposed=pr.unit(required)
            proposed_rep=proposed[2][~state[proposed[2]]]
            saving=pr.cost(old,state)-pr.cost(proposed,state)
            if (proposed[1]<old[1]-1e-5 and saving>=100
                and np.array_equal(np.sort(rep),np.sort(proposed_rep))):
                chosen=proposed
                audit.append(dict(original_row=j+1,old_radius=old[1],new_radius=chosen[1],
                    original_contained=len(ids),required_not_previously_covered=len(required),
                    retained_replacement_pipes=len(rep),construction_saving=saving/100))
        # Preserve the entire original prefix coverage, including historical pipes.
        assert np.all(covered[ids] | np.isin(ids,chosen[2]))
        covered[chosen[2]]=True;state[rep]=True;output.append(chosen)
    return output,audit,dict(shrunk_circles=len(audit),removed_circles=0,
        exact_replacement_set_per_row_preserved=True,all_original_prefix_coverage_preserved=True,
        nominal_fixed_savings=sum(a['construction_saving'] for a in audit))

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
        b=df[(df.model==model)&(df.scale==scale)&(df.arm=='best_77_22')].set_index('simulation')
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
    path=OUT/'submission_shrink_after_work.csv'
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
    pd.DataFrame(audit).to_csv(OUT/'shrinks.csv',index=False)
    audits={}
    for name,units in [('best_77_22',baseline),('construction_savings',candidate)]:
        r=execute(pr,units);annual=r.pop('annual');assert r['feasible'] and r['remaining_iron']==0
        pd.DataFrame(annual).to_csv(OUT/f'{name}_no_leak_annual.csv',index=False);audits[name]=r
    comparisons=compare(pr,{'best_77_22':baseline,'construction_savings':candidate},{'fine':fine,'old':risk_model(pr)[0]},args.sims)
    report=dict(cost_model_version='2026-10-26-area-exponent',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_units=len(baseline),new_units=len(candidate),search=search,
        forecast_changed=False,coverage_preserved=True,no_future_leak_delayed_pipes=int(delayed.sum()),
        no_future_leak_advanced_pipes=int((protected&(new_year<old_year)).sum()),
        exact_no_future_leak_construction_saving=(old_cost-new_cost)/100,arms=audits,comparison=comparisons,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')},
        timing_limitation='Replacement assignments per row are preserved; future leaks may change timing in both plans. No universal timing guarantee.',
        original_iron_coverage_note='Preserves all baseline covered pipes; historical original iron coverage limitation inherited, not supplemented.',official_score=None)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',search,'actual no-leak savings',(old_cost-new_cost)/100,flush=True)

if __name__=='__main__':main()
