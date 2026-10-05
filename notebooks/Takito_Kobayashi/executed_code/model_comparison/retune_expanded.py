"""Optimize actual shared rounds and parameters on the expanded fixed feature set."""
import argparse,json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import optuna
import run_experiment as e

def main(model,year,threads,lead=False):
    e.THREADS=threads;folder=e.OUT/model/str(year);cfg=json.loads((folder/'selected.json').read_text())
    d=pd.read_csv(e.OUT/'features.csv');splits=[(d[d.event_year<y],d[d.event_year==y]) for y in cfg['inner_years']]
    study_name='expanded_actual_common_rounds'+('_remaining_spatial' if cfg.get('supplemental_second_stage') else '')
    if lead:study_name='pre_feature_actual_common_rounds'
    if cfg.get('expanded_tuning_round',0):study_name+=f'_round{cfg["expanded_tuning_round"]}'
    study=optuna.create_study(study_name=study_name,storage='sqlite:///'+str(folder/'studies.sqlite'),direction='minimize',load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=e.SEED+123))
    if not study.trials:
        study.enqueue_trial(dict(cfg['params'],shared_iterations=cfg['iterations']))
    round_limit=min(6000,max(100,cfg['iterations']*4))
    def objective(trial):
        params=e.parameters(trial,model)
        n=trial.suggest_int('shared_iterations',1,round_limit,log=True)
        def one_fold(item):
            (a,b),year_inner=item
            _,pred,_,_=e.fit(model,params,a,b,cfg['features'],n)
            error=b.cost.to_numpy()-pred
            return {'year':year_inner,'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error))),'iterations':n,'n_train':len(a),'n_valid':len(b)}
        if model=='catboost':
            with ThreadPoolExecutor(max_workers=2) as workers:folds=list(workers.map(one_fold,zip(splits,cfg['inner_years'])))
        else:folds=[one_fold(item) for item in zip(splits,cfg['inner_years'])]
        trial.set_user_attr('features',cfg['features']);trial.set_user_attr('model_params',params);trial.set_user_attr('inner_folds',folds)
        return float(np.mean([f['rmse'] for f in folds]))
    complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,60-complete))
    expansions=[]
    while study.best_trial.params['shared_iterations']>=.95*round_limit and round_limit<6000:
        previous=round_limit;round_limit=min(6000,round_limit*4)
        study.optimize(objective,n_trials=20)
        expansions.append({'from':previous,'to':round_limit,'additional_trials':20})
    baseline=cfg['inner_mean_rmse'];best=study.best_trial
    if best.value<baseline:
        cfg.update(params=best.user_attrs['model_params'],inner_folds=best.user_attrs['inner_folds'],iterations=best.params['shared_iterations'],inner_mean_rmse=best.value)
    cfg['pre_feature_parameter_search' if lead else 'expanded_parameter_search']={'study_name':study_name,'features':cfg['features'],'completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),'before_rmse':baseline,'after_rmse':cfg['inner_mean_rmse'],'true_shared_round_fits':True,'round_limit':round_limit,'round_limit_expansions':expansions}
    e.dump(folder/'selected.json',cfg);study.trials_dataframe().to_csv(folder/'expanded_parameter_trials.csv',index=False)
    print('RETUNED',model,year,cfg['inner_mean_rmse'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--threads',type=int,default=3);p.add_argument('--lead',action='store_true');a=p.parse_args();main(a.model,a.year,a.threads,a.lead)
