"""Optuna nested calendar-year tuning. No outer-year labels enter a study.

Outputs are isolated from the previous fixed-grid results until verified.
2020 is a cold start: one training year cannot support temporal tuning.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, GammaRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import train as previous
from features import HERE, BASE

ROOT=HERE/'optuna'
OUT=ROOT/'results'
MODELS=ROOT/'models'
CORE=previous.CORE
THREADS=2
LOG={'anthony_rf_log_mean','rion_lgb_log_mean_refit'}
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore',message='Skipping features without any observed values')
warnings.filterwarnings('ignore',message='Found unknown categories')

def model_columns(name,columns):
    if name in ['anthony_linear_raw','anthony_gamma']:
        return [c for c in columns if c not in ['pipe_length_m','age_at_year_start','midpoint_x_m','midpoint_y_m','material_surface']]
    if name=='rion_gamma_refit':
        return [c for c in columns if c not in ['log1p_pipe_length_m','log1p_age_at_year_start','material_surface']]
    return columns

def space(t,name):
    """Conditional predictive search parameters; runtime controls stay fixed."""
    if name=='prior_surface_mean':
        return {'smoothing':t.suggest_float('smoothing',1,300,log=True)}
    p={'feature_pack':t.suggest_categorical('feature_pack',['base','group_history','local_history','inventory','topology','all'])}
    if 'linear' in name:
        p.update(fit_intercept=t.suggest_categorical('fit_intercept',[True,False]),positive=t.suggest_categorical('positive',[True,False]))
        p['regularizer']=t.suggest_categorical('regularizer',['none','ridge','lasso','elasticnet'])
        if p['regularizer']!='none':p['alpha']=t.suggest_float('alpha',1e-4,1e5 if p['regularizer']=='ridge' else 10,log=True)
        if p['regularizer']=='elasticnet':p['l1_ratio']=t.suggest_float('l1_ratio',.01,.99)
    elif 'gamma' in name:
        p.update(alpha=t.suggest_float('alpha',1e-8,100,log=True),fit_intercept=t.suggest_categorical('fit_intercept',[True,False]))
    elif '_rf_' in name:
        p.update(n_estimators=t.suggest_int('n_estimators',100,600,step=100),
            max_depth=t.suggest_categorical('max_depth',[None,4,8,12,20]),
            min_samples_split=t.suggest_int('min_samples_split',2,40),
            min_samples_leaf=t.suggest_int('min_samples_leaf',1,80),
            max_features=t.suggest_float('max_features',.2,1),
            max_leaf_nodes=t.suggest_categorical('max_leaf_nodes',[None,32,128,512]),
            bootstrap=t.suggest_categorical('bootstrap',[True,False]),
            criterion=t.suggest_categorical('criterion',['squared_error','friedman_mse']))
        if p['bootstrap']:p['max_samples']=t.suggest_float('max_samples',.5,1)
    elif 'xgboost' in name:
        p.update(n_estimators=t.suggest_int('n_estimators',50,500,step=50),max_depth=t.suggest_int('max_depth',2,8),
            learning_rate=t.suggest_float('learning_rate',.01,.2,log=True),min_child_weight=t.suggest_float('min_child_weight',1,100,log=True),
            subsample=t.suggest_float('subsample',.5,1),colsample_bytree=t.suggest_float('colsample_bytree',.5,1),
            colsample_bylevel=t.suggest_float('colsample_bylevel',.5,1),
            reg_alpha=t.suggest_float('reg_alpha',1e-8,100,log=True),reg_lambda=t.suggest_float('reg_lambda',1e-3,300,log=True),
            gamma=t.suggest_float('gamma',1e-8,100,log=True),max_bin=t.suggest_categorical('max_bin',[64,128,256]),
            grow_policy=t.suggest_categorical('grow_policy',['depthwise','lossguide']))
        if p['grow_policy']=='lossguide':p['max_leaves']=t.suggest_int('max_leaves',8,64)
    elif 'catboost' in name:
        p.update(iterations=t.suggest_int('iterations',50,500,step=50),depth=t.suggest_int('depth',2,7),
            learning_rate=t.suggest_float('learning_rate',.01,.2,log=True),l2_leaf_reg=t.suggest_float('l2_leaf_reg',1,300,log=True),
            random_strength=t.suggest_float('random_strength',1e-3,10,log=True),rsm=t.suggest_float('rsm',.5,1),
            border_count=t.suggest_categorical('border_count',[32,64,128]),one_hot_max_size=t.suggest_int('one_hot_max_size',2,10),
            leaf_estimation_iterations=t.suggest_int('leaf_estimation_iterations',1,10),
            bootstrap_type=t.suggest_categorical('bootstrap_type',['Bayesian','Bernoulli','MVS']))
        if p['bootstrap_type']=='Bayesian':p['bagging_temperature']=t.suggest_float('bagging_temperature',0,5)
        else:p['subsample']=t.suggest_float('subsample',.5,1)
    else:
        p.update(n_estimators=t.suggest_int('n_estimators',50,500,step=50),max_depth=t.suggest_int('max_depth',2,8),
            num_leaves=t.suggest_int('num_leaves',4,64),learning_rate=t.suggest_float('learning_rate',.01,.2,log=True),
            min_child_samples=t.suggest_int('min_child_samples',10,150),min_child_weight=t.suggest_float('min_child_weight',1e-3,10,log=True),
            subsample=t.suggest_float('subsample',.5,1),subsample_freq=t.suggest_int('subsample_freq',1,7),
            colsample_bytree=t.suggest_float('colsample_bytree',.5,1),reg_alpha=t.suggest_float('reg_alpha',1e-8,100,log=True),
            reg_lambda=t.suggest_float('reg_lambda',1e-3,300,log=True),min_split_gain=t.suggest_float('min_split_gain',1e-8,100,log=True),
            max_bin=t.suggest_categorical('max_bin',[63,127,255]),extra_trees=t.suggest_categorical('extra_trees',[True,False]))
    if name in LOG:p['correction']=t.suggest_categorical('correction',['none','training_residual','past_oof'])
    return p

def fit(name,a,b,params,groups,past):
    p=params.copy();pack=p.pop('feature_pack','base');cols=model_columns(name,groups[pack])
    correction=p.pop('correction','none')
    if name=='prior_surface_mean':
        alpha=p['smoothing'];mean=a.cost.mean();s=a.groupby('surface').cost.agg(['sum','count'])
        values=(s['sum']+alpha*mean)/(s['count']+alpha)
        bundle={'family':'optuna_surface','means':values.to_dict(),'fallback':mean,'params':params}
        return predict(bundle,b),bundle
    y=np.log(a.cost.to_numpy()) if name=='anthony_rf_log_mean' else np.log1p(a.cost.to_numpy()) if name in LOG else a.cost.to_numpy()
    if 'gamma' in name or 'linear' in name or '_rf_' in name:
        num=[c for c in cols if c not in previous.CATEGORICAL];cat=[c for c in cols if c in previous.CATEGORICAL]
        pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'),cat)])
        if 'gamma' in name:
            estimator=GammaRegressor(**p,max_iter=3000,tol=1e-6);scale=float(np.mean(y))
        elif 'linear' in name:
            regularizer=p.pop('regularizer','none');scale=float(np.mean(y))
            if regularizer=='none':estimator=LinearRegression(**p)
            elif regularizer=='ridge':estimator=Ridge(**p,max_iter=5000,tol=1e-5)
            elif regularizer=='lasso':estimator=Lasso(**p,max_iter=5000,tol=1e-5)
            else:estimator=ElasticNet(**p,max_iter=5000,tol=1e-5)
        else:estimator=RandomForestRegressor(**p,n_jobs=THREADS,random_state=440);scale=1.
        m=make_pipeline(pre,estimator)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m.fit(a,y/scale)
        if any('converge' in str(w.message).lower() for w in ws):raise ValueError('Failed convergence')
        z=b;za=a
        bundle={'family':'optuna_pipeline','model':m,'scale':scale}
    else:
        family='xgboost' if 'xgboost' in name else 'catboost' if 'catboost' in name else 'lightgbm'
        za,z,levels=previous.encode(a,b,cols,family)
        if family=='xgboost':m=XGBRegressor(**p,objective='reg:squarederror',tree_method='hist',enable_categorical=True,n_jobs=THREADS,random_state=440)
        elif family=='catboost':m=CatBoostRegressor(**p,loss_function='RMSE',verbose=False,allow_writing_files=False,thread_count=THREADS,random_seed=440)
        else:m=LGBMRegressor(**p,objective='regression',verbosity=-1,n_jobs=THREADS,random_state=440,deterministic=True,force_col_wise=True)
        m.fit(za,y,**({'cat_features':[c for c in cols if c in previous.CATEGORICAL]} if family=='catboost' else {}))
        scale=1.
        bundle={'family':family,'model':m,'features':cols,'levels':levels,'scale':scale}
    bundle.update(params=params,log_target='log' if name=='anthony_rf_log_mean' else 'log1p' if name in LOG else None,smearing_factor=1.)
    if name in LOG:
        if correction=='training_residual':
            predlog=m.predict(za)*scale
            bundle['smearing_factor']=float(np.mean(np.exp(y-predlog)))
        elif correction=='past_oof' and len(past):
            bundle['smearing_factor']=float(np.mean((past.cost+(name!='anthony_rf_log_mean'))/past[name+'_plain']))
    values=predict(bundle,b)
    if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
    return values,bundle

def predict(bundle,b,plain=False):
    if bundle['family']=='optuna_surface':return b.surface.map(bundle['means']).fillna(bundle['fallback']).to_numpy(float)
    if bundle['family']=='optuna_pipeline':z=b
    else:
        z=b[bundle['features']].copy()
        for c,levels in bundle['levels'].items():
            z[c]=z[c].fillna('__unknown__').astype(str) if bundle['family']=='catboost' else pd.Categorical(z[c].where(z[c].isin(levels)),categories=levels)
    p=np.asarray(bundle['model'].predict(z),float)*bundle['scale']
    if bundle['log_target']:
        p=np.exp(p)*(1 if plain else bundle['smearing_factor'])
        if not plain and bundle['log_target']=='log1p':p-=1
    return np.maximum(p,0)

def main():
    global THREADS
    parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=30)
    parser.add_argument('--member',choices=CORE);parser.add_argument('--origin',type=int,choices=range(2020,2028));parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--tune-only',action='store_true');args=parser.parse_args()
    THREADS=args.threads
    if args.origin and not args.member:parser.error('--origin requires --member')
    if args.tune_only and not args.origin:parser.error('--tune-only requires --origin')
    members=[args.member] if args.member else CORE
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    path=HERE/'data/training_rich.csv';d=pd.read_csv(path);future=pd.read_csv(HERE/'data/forecast_features_2027.csv')
    features=json.loads((HERE/'data/feature_contract.json').read_text())['features'];groups=previous.packs(features)
    # Preserve completed studies whose search/model definitions are unchanged.
    policy=ROOT/'search_policy.json'
    fingerprint=json.loads(policy.read_text())['fingerprint'] if policy.exists() else hashlib.sha256(path.read_bytes()+Path(__file__).read_bytes()).hexdigest()[:16]
    previous.dump(ROOT/'search_policy.json',{'requested_trials_per_model_origin':args.trials,'fingerprint':fingerprint,
        'objective':'mean annual RMSE on the two most recent available inner years; one year where only one exists',
        'outer_years':list(range(2020,2028)),'cold_start_2020':'No temporal validation exists; seeded midpoint configuration, not optimized',
        'fixed_controls':['random seed 440','worker thread count is a runtime control, not optimized','raw-cost RMSE objective; log families retain log target','CPU histogram/gbdt/symmetric-tree implementations','convergence tolerances and iterations'],
        'current_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_revision':'Anthony linear/Gamma use log length/age; Rion Gamma uses raw length/age, avoiding duplicate Gamma specifications.',
        'scope_limitation':'Substantive listed predictive parameters, not every library option. RF absolute-error/Poisson criteria and alternate booster/loss families are not searched. No guarantee of global optimum.'})
    parts=[];configs=[]
    with threadpool_limits(THREADS):
        for origin in ([args.origin] if args.origin else range(2020,2028)):
            a=d[d.event_year<origin];b=d[d.event_year==origin] if origin<2027 else future
            cache=OUT/(f'fold_{origin}_{args.member}.csv' if args.member else f'fold_{origin}.csv')
            if cache.exists():
                if origin<2027:parts.append(pd.read_csv(cache))
                continue
            past=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
            if args.origin and args.member in LOG:
                # Inner validation at origin-1 needs calibration OOFs only up
                # to origin-2. Final fitting additionally needs origin-1.
                end=origin-2 if args.tune_only else origin-1
                paths=[OUT/f'fold_{y}_{args.member}.csv' for y in range(2020,end+1)]
                if not all(p.exists() for p in paths):raise RuntimeError(f'Missing preceding calibration OOFs for {origin}')
                past=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True) if paths else pd.DataFrame()
            result=b[['event_id','pipe_id','event_year','cost']].copy() if origin<2027 else b[['pipe_id']].assign(scenario_year=2027)
            inner=sorted(a.event_year.unique())[1:][-2:]
            for name in members:
                study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=440),
                    storage='sqlite:///'+(ROOT/'studies.sqlite3').as_posix(),study_name=f'{fingerprint}_{name}_{origin}'+('_regularized_linear_v3' if name=='anthony_linear_raw' else '_original_numeric_v2' if name in ['anthony_gamma','rion_gamma_refit'] else ''),load_if_exists=True)
                def objective(t):
                    params=space(t,name);scores=[]
                    for year in inner:
                        tr=a[a.event_year<year];va=a[a.event_year==year]
                        h=past[past.event_year<year] if len(past) else past
                        values,_=fit(name,tr,va,params,groups,h)
                        scores.append(previous.rmse(va.cost,values))
                    t.set_user_attr('annual_RMSE',scores)
                    return float(np.mean(scores))
                if inner:
                    def progress(study,t):
                        if (t.number+1)%5==0:print(f'  {origin} {name}: {t.number+1}/{args.trials} trials',flush=True)
                    study.optimize(objective,n_trials=max(0,args.trials-len(study.trials)),catch=(ValueError,RuntimeError,np.linalg.LinAlgError),callbacks=[progress])
                    params=space(optuna.trial.FixedTrial(study.best_params),name)
                    score=study.best_value
                else:
                    trial=optuna.trial.FixedTrial({'smoothing':20}) if name=='prior_surface_mean' else optuna.trial.FixedTrial(defaults(name))
                    params=space(trial,name);score=None
                if args.tune_only:
                    print(f'Completed tuning only: {name} {origin}, {len(study.trials)} trials',flush=True)
                    return
                rejected=[];selected_trial=None
                if inner:
                    # A candidate that fails to converge on the full past-only
                    # training set cannot be deployed. Try ranked completed
                    # trials, without consulting the outer-year true costs.
                    candidates=sorted([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE],key=lambda t:t.value)
                    for candidate in candidates:
                        params=space(optuna.trial.FixedTrial(candidate.params),name)
                        try:values,bundle=fit(name,a,b,params,groups,past)
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                            rejected.append({'trial':candidate.number,'reason':str(error)});continue
                        score=candidate.value;selected_trial=candidate.number;break
                    else:raise RuntimeError(f'No completed trial could be fit on all past training rows: {name} {origin}')
                else:values,bundle=fit(name,a,b,params,groups,past)
                result[name]=values
                if name in LOG:result[name+'_plain']=predict(bundle,b,plain=True)
                if name=='anthony_rf_log_mean':result['anthony_rf_log_plain']=result[name+'_plain']
                config={'model':name,'origin':origin,'training_end':origin-1,'inner_years':[int(y) for y in inner],
                    'params':params,'features':model_columns(name,groups[params.get('feature_pack','base')]),'inner_mean_RMSE':score,
                    'completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),'fingerprint':fingerprint,
                    'training_rows':len(a),'smearing_factor':bundle.get('smearing_factor'),
                    'selected_trial':selected_trial,'full_training_rejected_candidates':rejected}
                previous.dump(OUT/'configs'/name/f'{origin}.json',config)
                study.trials_dataframe().to_csv(OUT/f'trials_{name}_{origin}.csv',index=False)
                if origin==2027:joblib.dump(bundle,MODELS/f'{name}.joblib')
                print(f'Optuna {origin} {name}: trials={len(study.trials)}, inner_RMSE={score}',flush=True)
            result.to_csv(cache,index=False)
            if origin<2027:parts.append(result)
        if args.origin:return
        oof=pd.concat(parts,ignore_index=True)
        if args.member:
            oof.to_csv(OUT/f'oof_{args.member}.csv',index=False)
            pd.read_csv(OUT/f'fold_2027_{args.member}.csv').to_csv(OUT/f'forecast_{args.member}.csv',index=False)
            return
        oof.to_csv(OUT/'out_of_time_predictions.csv',index=False)
        forecast=pd.read_csv(OUT/'fold_2027.csv');forecast.to_csv(OUT/'forecast_costs_2027.csv',index=False)
        previous.dump(MODELS/'training_contract.json',{'all_training_rows':len(d),'training_years':[2019,2026],
            'observed_history_cutoff':'2027-01-01 exclusive','features':features,'core_models':CORE,
            'future_support_rows':len(forecast),'default_ensemble':'ensemble_equal','default_is_provisional':True,
            'supported_materials':sorted(d.material.unique().tolist()),'unsupported_policy':'No PU/unknown severity estimate; never assign zero.',
            'optuna_trials':args.trials,'fingerprint':fingerprint,'historical_results_already_inspected':True})

def defaults(name):
    # Deterministic seeded samples define a cold-start model without evaluating 2020.
    sampler=optuna.samplers.RandomSampler(seed=440)
    s=optuna.create_study(sampler=sampler);t=s.ask();space(t,name)
    result=t.params;result['feature_pack']='base'
    return result

if __name__=='__main__':main()
