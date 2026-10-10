"""Run independent member searches in five bounded local worker processes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import pandas as pd
import optuna
from optimize import ROOT, OUT, MODELS, CORE
from features import HERE
from train import dump

def run(name):
    with (ROOT/f'run_{name}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(HERE/'optimize.py'),'--trials','30','--member',name],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see {ROOT}/run_{name}.log')
    print('Completed all temporal searches and final fit:',name,flush=True)

def main():
    # The prior serial worker was stopped explicitly. Mark its abandoned trials.
    storage='sqlite:///'+(ROOT/'studies.sqlite3').as_posix()
    for meta in optuna.get_all_study_summaries(storage=storage):
        study=optuna.load_study(study_name=meta.study_name,storage=storage)
        for t in study.trials:
            if t.state==optuna.trial.TrialState.RUNNING:study.tell(t.number,state=optuna.trial.TrialState.FAIL)
    # Reuse only completed full folds whose model definitions are current.
    for year in range(2020,2028):
        cache=OUT/f'fold_{year}.csv'
        if not cache.exists():continue
        p=pd.read_csv(cache)
        for name in CORE:
            target=OUT/f'fold_{year}_{name}.csv'
            if target.exists():continue
            meta=['event_id','pipe_id','event_year','cost'] if year<2027 else ['pipe_id','scenario_year']
            cols=meta+[c for c in [name,name+'_plain','anthony_rf_log_plain' if name=='anthony_rf_log_mean' else name] if c in p]
            p[list(dict.fromkeys(cols))].to_csv(target,index=False)
    # Start the expensive members first; no agent delegation is involved.
    ordering=['takito_catboost','anthony_rf_log_mean','rion_rf_raw_refit','takito_xgboost','rion_lgb_log_mean_refit',
              'takito_lightgbm','anthony_gamma','rion_gamma_refit','anthony_linear_raw','prior_surface_mean']
    with ThreadPoolExecutor(max_workers=5) as pool:
        for task in as_completed([pool.submit(run,n) for n in ordering]):task.result()
    oof=None;future=None
    for name in CORE:
        p=pd.read_csv(OUT/f'oof_{name}.csv');f=pd.read_csv(OUT/f'forecast_{name}.csv')
        if oof is None:oof=p;future=f
        else:
            assert p.event_id.tolist()==oof.event_id.tolist() and f.pipe_id.tolist()==future.pipe_id.tolist()
            for c in p:
                if c not in oof:oof[c]=p[c]
            for c in f:
                if c not in future:future[c]=f[c]
    assert not oof[CORE].isna().any().any()
    oof.to_csv(OUT/'out_of_time_predictions.csv',index=False);future.to_csv(OUT/'forecast_costs_2027.csv',index=False)
    d=pd.read_csv(HERE/'data/training_rich.csv')
    features=pd.read_json(HERE/'data/feature_contract.json',typ='series')['features']
    dump(MODELS/'training_contract.json',{'all_training_rows':len(d),'training_years':[2019,2026],
        'observed_history_cutoff':'2027-01-01 exclusive','features':features,'core_models':CORE,
        'future_support_rows':len(future),'default_ensemble':'ensemble_equal','default_is_provisional':True,
        'supported_materials':sorted(d.material.unique().tolist()),'unsupported_policy':'No PU/unknown severity estimate; never assign zero.',
        'optuna_trials':30,'historical_results_already_inspected':True})
    subprocess.run([sys.executable,str(HERE/'finish_optuna.py')],check=True)

if __name__=='__main__':main()
