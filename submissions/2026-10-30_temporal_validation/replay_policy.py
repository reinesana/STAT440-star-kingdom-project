"""Six controlled one-year historical policy replays, not competition scores."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
REPOSITORY=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
R=REPOSITORY
OUT=args.output_dir.resolve()
sys.path[:0]=[str(R/'src'),str(R/'submissions/2026-10-28_ryota_early3')]
from optimize import Problem,risk_model,cents
import simulate_score as sim
import search_orders as so

def main():
    assert (OUT/'leak_annual_metrics.csv').exists()
    protocol={'years':[2024,2025,2026],'models':['ryota_fine','team_ml_equal'],
      'geometry':'Fixed inventory midpoint circle for each original iron pipe; radius half length plus 1e-6m; no optimized submitted geometry',
      'budget':10000000,'ordering':'Shared order_dynamic, expected net repair benefit per current construction cost',
      'cost_forecast':'Same past-only material/surface means for both arms',
      'evaluation':'Actual held-year recorded repair costs avoided under start-of-year replacement assumption',
      'limitations':['One-year retrospective counterfactual, not official or 25-year score.','Unknown/missing ML inputs use the same Ryota fine forecast fallback.','Does not compare the manually overridden p17b procedure or compare alternative cost models in planning.']}
    (OUT/'policy_protocol.json').write_text(json.dumps(protocol,indent=2))
    pr=Problem();pipes,a,b=sim.load();length=pr.length
    xy=pr.mid[pr.iron];rad=length[pr.iron]/2+1e-6
    inside=sim.containment(xy,rad,a,b,length)
    fixed=np.array([u+100 for u in [sum(sim.RATES[s] for s in set(pr.surface[ii]))*(np.pi*rr**2)**.85 for ii,rr in zip(inside,rad)]])
    rows=[]
    for year in protocol['years']:
        _,sev=risk_model(pr,year-1)
        # Restore the known replacement state at this historical origin.
        pp=pipes.copy();pp['leak_date']=pp.leak_date.where(pp.leak_date.dt.year<year)
        start=(pp.mat.eq('polyurethane')|pp.leak_date.notna()).to_numpy()
        pred=pd.read_csv(OUT/f'leak_predictions_{year}.csv',dtype={'pipe_id':str}).set_index('pipe_id')
        # Fill only excluded inventory rows identically across arms.
        from early_three_years import fit_fine
        ff,_=fit_fine(pr,year-1,.5,.85);fallback=ff(year)
        for model in protocol['models']:
            p=pr.p['Pipe ID'].map(pred[model]).to_numpy();p=np.where(np.isfinite(p),p,fallback);p[start]=0
            val=np.maximum(p*(sev-200*length),0)
            order=so.order_dynamic(inside,rad,pp,length,np.zeros((len(pp),1)),static_value=val,blend=1.)
            assert len(order)==len(inside) and len(set(order))==len(order)
            state=start.copy();spent=units=0
            for j in order:
                ids=inside[j];rep=ids[~state[ids]]
                if not len(rep):continue
                cost=cents(fixed[j]+200*length[rep].sum())
                if spent+cost>1000000000:break
                spent+=cost;units+=1;state[rep]=True
            hit=(pr.event==year)&~start;prevented=hit&state
            actual=pr.p['Pipe ID'].map(pr.hist.set_index('Pipe ID').Cost).fillna(0).to_numpy()
            repair=float(actual[hit&~state].sum());avoided=float(actual[prevented].sum())
            rows.append({'year':year,'model':model,'construction':spent/100,'remaining_repair':repair,'avoided_repair':avoided,'one_year_total_cost':spent/100+repair,'prevented_leaks':int(prevented.sum()),'replaced_pipes':int((state&~start).sum()),'work_units':units})
            print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'controlled_policy_replays.csv',index=False)

if __name__=='__main__':main()
