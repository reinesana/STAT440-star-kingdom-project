"""Past-only comparison; preserve repository and existing submissions."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPOSITORY=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
REPO=REPOSITORY
OUT=args.output_dir.resolve()
sys.path.insert(0,str(REPO/'submissions/2026-10-28_ryota_early3'))
from optimize import Problem,risk_model
from early_three_years import fit_fine

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12);y=np.asarray(y,float)
    f=pd.DataFrame({'p':p,'y':y}).groupby('p',sort=True).y.agg(['sum','size']).iloc[::-1]
    ap=float(((f['sum'].cumsum()/f['size'].cumsum())*f['sum']).sum()/max(y.sum(),1))
    return dict(n=len(y),events=int(y.sum()),brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),average_precision=ap,predicted_events=float(p.sum()),count_bias_pct=float(100*(p.sum()/max(y.sum(),1)-1)))

def main():
    OUT.mkdir(exist_ok=False,parents=True)
    protocol={'years':[2024,2025,2026],'development_years':[2024,2025],'confirmation_year':2026,'untouched_holdout':False,
      'leak_primary':'Equal-year Brier, supported by log loss and calibration; no public-score selection',
      'cost_primary':'Equal-year RMSE with MAE and total-cost bias diagnostics',
      'candidate_gate':'Change existing submission only with consistent temporal evidence and controlled planning comparison',
      'scope':'Reuse saved past-only ML forecasts; refit Ryota baseline per cutoff; no hyperparameter search',
      'limitations':['These historical years have previously been inspected.','p17b manually staged 2027 cohort overrides have no frozen historical backtest and must not be retroactively applied.','One-year validation cannot certify 25-year forecasts.'],
      'cost_model_version':'2026-10-26-area-exponent'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    pr=Problem();used=[REPO/'data/pipes.csv',REPO/'data/train.csv',REPO/'data/leak_prediction_master.csv',REPO/'submissions/2026-10-28_ryota_early3/optimize.py']
    master=pd.read_csv(used[2],dtype={'pipe_id':str});ref=master[master.year.between(2023,2026)].set_index(['pipe_id','year']).sort_index()
    names={'xgb':REPO/'notebooks/Amy_Sim/predictions/xgboost_oof_predictions.csv','cat':REPO/'notebooks/Ryota_Sato/catboost_predictions.csv','lgb':REPO/'notebooks/Mar_Nacianceno/lightgbm_baseline_oof_predictions.csv'}
    cols=[]
    for name,path in names.items():
        d=pd.read_csv(path,dtype={'pipe_id':str}).set_index(['pipe_id','year']).sort_index()
        assert d.index.is_unique and d.index.equals(ref.index)
        np.testing.assert_array_equal(d.y_true,ref.Y);cols.append(d.predicted_probability.to_numpy());used.append(path)
    ml=pd.Series(np.mean(cols,axis=0),index=ref.index)
    year_event=pd.Series(pr.event,index=pr.p['Pipe ID']);idx=pd.Series(np.arange(len(pr.p)),index=pr.p['Pipe ID'])
    records=[];cal=[];sub=[];forecasts={};invariance=[]
    basepath=REPO/'submissions/2026-10-26_best20a_catboost_logrf/base_oof_2023_2026.csv';used.append(basepath)
    costs=pd.read_csv(basepath,dtype={'pipe_id':str});cr=[];weight_records=[]
    for year in [2024,2025,2026]:
        print('Temporal audit',year,flush=True)
        predict,sev=risk_model(pr,year-1);rp=predict(year)
        fine,_=fit_fine(pr,year-1,.5,.85);fp=fine(year)
        # Behavioral check: unknown future labels and costs cannot change fitted forecasts.
        oldevent=pr.event.copy();oldhist=pr.hist.copy();pr.event=np.where(pr.event>=year,np.nan,pr.event)
        pr.hist=pr.hist.copy();pr.hist.loc[pr.dates.dt.year>=year,'Cost']*=37
        changed,changedsev=risk_model(pr,year-1)
        np.testing.assert_array_equal(rp,changed(year));np.testing.assert_array_equal(sev,changedsev)
        changedfine,_=fit_fine(pr,year-1,.5,.85);np.testing.assert_array_equal(fp,changedfine(year))
        pr.event=oldevent;pr.hist=oldhist;invariance.append({'year':year,'future_label_cost_invariance':True})
        d=ref.xs(year,level='year').copy();ii=idx.loc[d.index].to_numpy();y=d.Y.to_numpy()
        expected=(pr.event[ii]==year).astype(int);np.testing.assert_array_equal(y,expected)
        assert (np.isnan(pr.event[ii])|(pr.event[ii]>=year)).all()
        # Original PU pipes are deterministic zero-risk; compare only eligible metals.
        d=d.loc[pr.mat[ii]!='polyurethane'];ii=idx.loc[d.index].to_numpy();y=d.Y.to_numpy()
        preds={'ryota_age':rp[ii],'ryota_fine':fp[ii],'team_ml_equal':ml.xs(year,level='year').loc[d.index].to_numpy()}
        # Fixed 50/50 blend is diagnostic only, not a searched ensemble.
        preds['fixed_half_blend']=(preds['ryota_fine']+preds['team_ml_equal'])/2
        for name,p in preds.items():
            records.append(dict(year=year,model=name,**metrics(y,p)))
            tmp=pd.DataFrame({'p':p,'y':y,'material':pr.mat[ii],'surface':pr.surface[ii],'age_bin':np.floor((pr.age[ii]+year-2027)/10)*10})
            tmp['bin']=pd.cut(p,[-1e-12,.01,.025,.05,.1,.2,.4,.6,.8,1.])
            for bn,g in tmp.groupby('bin',observed=True):cal.append(dict(year=year,model=name,bin=str(bn),n=len(g),predicted=float(g.p.mean()),actual=float(g.y.mean())))
            for key in ['material','surface','age_bin']:
                for group,g in tmp.groupby(key):sub.append(dict(year=year,model=name,feature=key,group=str(group),**metrics(g.y,g.p)))
        prior=costs[costs.event_year<year];test=costs[costs.event_year==year].copy()
        # Two base procedures fixed in advance; coefficients use prior years only.
        xx=prior[['takito_catboost','anthony_rf_log_mean']].to_numpy();yy=prior.cost.to_numpy()
        fit=minimize(lambda w:np.mean(((xx@w-yy)/100000)**2),[.5,.5],bounds=[(0,1)]*2,constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
        assert fit.success;w=fit.x;weight_records.append({'year':year,'weights':w.tolist(),'training_years':sorted(prior.event_year.unique().tolist())})
        ids=idx.loc[test.pipe_id].to_numpy();raw=pr.hist.set_index('Pipe ID').loc[test.pipe_id]
        np.testing.assert_allclose(test.cost,raw.Cost)
        cp={'ryota_material_surface':sev[ids],'catboost':test.takito_catboost.to_numpy(),'corrected_log_rf':test.anthony_rf_log_mean.to_numpy(),'past_only_two_model_stack':test[['takito_catboost','anthony_rf_log_mean']].to_numpy()@w}
        for name,p in cp.items():
            e=p-test.cost.to_numpy();cr.append(dict(year=year,model=name,n=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),cost_bias_pct=float(100*(p.sum()/test.cost.sum()-1))))
        forecasts[year]=pd.DataFrame({'pipe_id':d.index,'y':y,**preds})
    pd.DataFrame(records).to_csv(OUT/'leak_annual_metrics.csv',index=False)
    pd.DataFrame(cal).to_csv(OUT/'leak_calibration.csv',index=False);pd.DataFrame(sub).to_csv(OUT/'leak_subgroups.csv',index=False)
    pd.DataFrame(cr).to_csv(OUT/'cost_annual_metrics.csv',index=False)
    (OUT/'past_only_cost_weights.json').write_text(json.dumps(weight_records,indent=2))
    (OUT/'implementation_checks.json').write_text(json.dumps(invariance,indent=2))
    for year,d in forecasts.items():d.to_csv(OUT/f'leak_predictions_{year}.csv',index=False)
    for kind,frame,metric in [('leak',pd.DataFrame(records),'brier'),('cost',pd.DataFrame(cr),'rmse')]:
        dev=frame[frame.year<2026].groupby('model')[metric].mean().sort_values();selection={'selected_on_2024_2025':dev.index[0],'development_metrics':dev.to_dict(),'selection_does_not_use_2026':True}
        (OUT/f'{kind}_selection.json').write_text(json.dumps(selection,indent=2))
        print(kind,frame.to_string(index=False),flush=True)
    (OUT/'input_hashes.json').write_text(json.dumps({str(p):digest(p) for p in used},indent=2))

if __name__=='__main__':main()
