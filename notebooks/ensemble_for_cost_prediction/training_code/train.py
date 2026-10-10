"""Nested temporal Takito refits, adapted Anthony models and stable convex blends."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from features import BASE, HERE, FeatureBuilder

os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
OUT=HERE/'results'
MODELS=HERE/'models'
BASE_CORE=['prior_surface_mean','anthony_linear_raw','anthony_gamma','anthony_rf_log_mean',
           'takito_xgboost','takito_catboost','takito_lightgbm']
CORE=BASE_CORE+['rion_gamma_refit','rion_rf_raw_refit','rion_lgb_log_mean_refit']
CATEGORICAL=['material','surface','material_surface']
SEED=440


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def packs(features):
    groups={'base':BASE}
    groups['group_history']=BASE+[c for c in features if c.startswith('hist_') and not c.startswith(('hist_radius','hist_nearest'))]
    groups['local_history']=BASE+[c for c in features if c.startswith(('hist_radius','hist_nearest'))]
    groups['inventory']=BASE+[c for c in features if c.startswith(('inventory_','state_')) or '_x_' in c or c.startswith(('span_','orientation_'))]
    groups['topology']=BASE+[c for c in features if c.startswith('topology_') or c.startswith(('hist_all_route_','hist_last2_route_'))]
    groups['all']=features
    return {k:list(dict.fromkeys(v)) for k,v in groups.items()}


def encode(train,test,features,model):
    a,b=train[features].copy(),test[features].copy()
    levels={}
    for c in set(features)&set(CATEGORICAL):
        levels[c]=sorted(a[c].dropna().unique())
        if model=='catboost':
            a[c]=a[c].fillna('__unknown__').astype(str)
            b[c]=b[c].fillna('__unknown__').astype(str)
        else:
            a[c]=pd.Categorical(a[c],categories=levels[c])
            b[c]=pd.Categorical(b[c].where(b[c].isin(levels[c])),categories=levels[c])
    return a,b,levels


def fit_booster(model,train,test,features,regime):
    a,b,levels=encode(train,test,features,model)
    conservative=regime=='conservative'
    trees=50 if conservative else 150
    depth=3 if conservative else 4
    if model=='xgboost':
        m=XGBRegressor(n_estimators=trees,max_depth=depth,learning_rate=.05,
            min_child_weight=40 if conservative else 15,reg_lambda=100 if conservative else 20,
            subsample=.8,colsample_bytree=.8,tree_method='hist',enable_categorical=True,
            objective='reg:squarederror',random_state=SEED,n_jobs=2)
        m.fit(a,train.cost)
    elif model=='lightgbm':
        m=LGBMRegressor(n_estimators=trees,max_depth=depth,num_leaves=8 if conservative else 16,
            learning_rate=.05,min_child_samples=80 if conservative else 40,
            reg_lambda=100 if conservative else 20,objective='regression',
            random_state=SEED,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
        m.fit(a,train.cost)
    else:
        m=CatBoostRegressor(iterations=trees,depth=depth,learning_rate=.05,
            l2_leaf_reg=100 if conservative else 20,loss_function='RMSE',
            random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False)
        m.fit(a,train.cost,cat_features=[c for c in features if c in CATEGORICAL])
    p=np.asarray(np.maximum(m.predict(b),0),dtype=float)
    assert np.isfinite(p).all()
    return p,{'model':m,'family':model,'features':features,'levels':levels,'regime':regime}


def select_and_fit(model,d,test,origin,features):
    training=d[d.event_year<origin]
    inner_years=sorted(training.event_year.unique())[1:][-2:]
    groups=packs(features)
    trials=[]
    if not inner_years:
        best={'feature_pack':'base','regime':'conservative','features':BASE,'inner_mean_RMSE':None}
    else:
        best=None
        for pack,cols in groups.items():
            for regime in ['conservative','flexible']:
                errors=[]
                for inner in inner_years:
                    a=training[training.event_year<inner]; b=training[training.event_year==inner]
                    assert a.event_year.max()<inner<origin
                    p,_=fit_booster(model,a,b,cols,regime)
                    errors.append(rmse(b.cost,p))
                row={'model':model,'origin':origin,'feature_pack':pack,'regime':regime,
                     'inner_mean_RMSE':float(np.mean(errors)),'inner_years':[int(y) for y in inner_years],
                     'n_features':len(cols)}
                trials.append(row)
                if best is None or row['inner_mean_RMSE']<best['inner_mean_RMSE']:
                    best={**row,'features':cols}
        assert best is not None
    p,bundle=fit_booster(model,training,test,best['features'],best['regime'])
    config={k:v for k,v in best.items() if k!='features'}
    config['features']=best['features']
    config['training_end']=origin-1
    dump(OUT/'configs'/model/f'{origin}.json',config)
    if trials:pd.DataFrame(trials).to_csv(OUT/f'trials_{model}_{origin}.csv',index=False)
    print(f'{model} origin={origin} selected={best["feature_pack"]}/{best["regime"]} features={len(best["features"])}',flush=True)
    return p,bundle


def anthony_pipe(estimator,rf=False):
    columns=['log1p_pipe_length_m','log1p_age_at_year_start']
    if not rf: columns+=['scenario_year']
    # Original notebook used material cast iron and surface grassland as reference.
    # For portable annual adaptation, training-only one-hot encoding drops one
    # level per category and handles future unseen categories explicitly.
    num=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
    cat=OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False)
    pre=ColumnTransformer([('num',num,columns),('cat',cat,['material','surface'])])
    return make_pipeline(pre,estimator)


def anthony_fit(train,test,past_oof):
    models={};predictions={}
    specs=[('anthony_linear_raw',LinearRegression()),('anthony_gamma',GammaRegressor(alpha=0,max_iter=2000,tol=1e-7))]
    for name,estimator in specs:
        pipe=anthony_pipe(estimator)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            pipe.fit(train,train.cost)
        important=[str(r.message) for r in records if 'converge' in str(r.message).lower()]
        if important:raise RuntimeError(f'{name} did not converge: {important}')
        predictions[name]=np.maximum(pipe.predict(test),0)
        models[name]={'model':pipe,'family':'anthony_pipeline'}
    rf=anthony_pipe(RandomForestRegressor(n_estimators=500,min_samples_leaf=5,
                                          n_jobs=2,random_state=0),rf=True)
    rf.fit(train,np.log(train.cost))
    plain=np.exp(rf.predict(test))
    # Preserve Anthony's original log RF as a diagnostic. For expected repair
    # cost, mean retransformation uses ONLY preceding out-of-time residuals.
    smear=1.0 if past_oof.empty else float(np.mean(past_oof.cost/past_oof.anthony_rf_log_plain))
    predictions['anthony_rf_log_plain']=plain
    predictions['anthony_rf_log_mean']=plain*smear
    models['anthony_rf_log_mean']={'model':rf,'family':'anthony_log_rf','smearing_factor':smear}
    for name,p in predictions.items():assert np.isfinite(p).all() and (p>=0).all(),name
    return predictions,models


def model_prediction(bundle,x):
    family=bundle['family']
    if family=='anthony_log_rf':return np.exp(bundle['model'].predict(x))*bundle['smearing_factor']
    if family=='anthony_pipeline':return np.maximum(bundle['model'].predict(x),0)
    z=x[bundle['features']].copy()
    for c,levels in bundle['levels'].items():
        if family=='catboost':z[c]=z[c].fillna('__unknown__').astype(str)
        else:z[c]=pd.Categorical(z[c].where(z[c].isin(levels)),categories=levels)
    prediction=bundle['model'].predict(z)
    if bundle.get('target_transform')=='log1p':
        prediction=(np.maximum(np.expm1(prediction),0)+1)*bundle['smearing_factor']-1
    return np.asarray(np.maximum(prediction,0),dtype=float)


def fit_weights(past,regularization):
    # Equal annual weighting matches the primary mean-annual RMSE convention.
    x=past[CORE].to_numpy();y=past.cost.to_numpy();years=past.event_year.to_numpy()
    scale=max(float(np.sqrt(np.mean(y*y))),1)
    x,y=x/scale,y/scale
    target=np.full(len(CORE),1/len(CORE))
    def loss(w):
        e=x@w-y
        return np.mean([np.sqrt(np.mean(e[years==year]**2)) for year in np.unique(years)])+regularization*np.sum((w-target)**2)
    result=minimize(loss,target,method='SLSQP',bounds=[(0,1)]*len(CORE),
                    constraints={'type':'eq','fun':lambda w:w.sum()-1},
                    options={'ftol':1e-10,'maxiter':1500})
    assert result.success,result.message
    assert np.all(result.x>=-1e-10) and np.isclose(result.x.sum(),1)
    return np.maximum(result.x,0)/result.x.sum()


def ensemble_methods(oof):
    rows=[]
    for year in range(2023,2027):
        current=oof.event_year==year;past=oof[oof.event_year<year]
        assert past.event_year.max()<year
        oof.loc[current,'ensemble_equal']=oof.loc[current,CORE].mean(axis=1)
        w=fit_weights(past,0)
        ws=fit_weights(past,.02)
        # Fixed 50% shrinkage is evaluated prospectively; it is not chosen from
        # the held-out year. It retains every model for diversification.
        ws=.5*ws+.5/len(CORE)
        oof.loc[current,'ensemble_rmse_weighted']=oof.loc[current,CORE].to_numpy()@w
        oof.loc[current,'ensemble_stable']=oof.loc[current,CORE].to_numpy()@ws
        for name,values in [('ensemble_rmse_weighted',w),('ensemble_stable',ws)]:
            for candidate,weight in zip(CORE,values):rows.append({'test_year':year,'method':name,'candidate':candidate,
                         'weight':float(weight),'weight_training_end':year-1})
    return oof,pd.DataFrame(rows)


def diagnostics(oof,d):
    methods=CORE+['anthony_rf_log_plain','ensemble_equal','ensemble_rmse_weighted','ensemble_stable']
    rows=[]
    for year,q in oof[oof.event_year>=2023].groupby('event_year'):
        threshold=d.loc[d.event_year<year,'cost'].quantile(.99)
        for name in methods:
            err=q[name]-q.cost
            rows.append({'model':name,'year':int(year),'n_test':len(q),'RMSE':rmse(q.cost,q[name]),
                'MAE':float(np.mean(abs(err))),'actual_total':float(q.cost.sum()),'predicted_total':float(q[name].sum()),
                'total_bias_pct':float(100*(q[name].sum()/q.cost.sum()-1)),
                'high_cost_threshold_prior_only':float(threshold),
                'high_cost_RMSE':rmse(q.loc[q.cost>=threshold,'cost'],q.loc[q.cost>=threshold,name])})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'annual_metrics.csv',index=False)
    summary=metrics.groupby('model').agg(mean_annual_RMSE=('RMSE','mean'),
        annual_RMSE_sd=('RMSE','std'),worst_annual_RMSE=('RMSE','max'),
        mean_absolute_annual_total_bias_pct=('total_bias_pct',lambda x:abs(x).mean()),
        worst_absolute_annual_total_bias_pct=('total_bias_pct',lambda x:abs(x).max()))
    q=oof[oof.event_year>=2023]
    summary['pooled_RMSE']=[rmse(q.cost,q[name]) for name in summary.index]
    summary['pooled_total_bias_pct']=[float(100*(q[name].sum()/q.cost.sum()-1)) for name in summary.index]
    # Compare annual RMSE relative to each year's simple baseline: raw annual
    # standard deviation alone also reflects genuine changes in test difficulty.
    baseline=metrics[metrics.model=='prior_surface_mean'].set_index('year').RMSE
    relative=metrics.assign(relative_RMSE=metrics.RMSE/metrics.year.map(baseline))
    summary['annual_relative_RMSE_sd']=relative.groupby('model').relative_RMSE.std()
    summary.sort_values('mean_annual_RMSE').to_csv(OUT/'summary_metrics.csv')
    pd.DataFrame({name:q[name]-q.cost for name in CORE}).corr().to_csv(OUT/'residual_correlations.csv')
    # Paired month-cluster bootstrap; measures sampling sensitivity, not proof
    # of future-year stability. No confidence-based candidate search follows.
    events=pd.read_csv(HERE.parent/'ensemble_preparation/prepared/events_merged_all.csv')
    dates=q.pipe_id.map(events.set_index('pipe_id').date)
    blocks=pd.to_datetime(dates).dt.to_period('M').astype(str).to_numpy()
    unique=np.unique(blocks); rng=np.random.default_rng(SEED)
    sums={name:pd.DataFrame({'block':blocks,'e2':(q[name]-q.cost).to_numpy()**2}).groupby('block').e2.sum().reindex(unique).to_numpy() for name in methods}
    counts=pd.Series(blocks).value_counts().reindex(unique).to_numpy()
    b=[]
    for _ in range(500):
        ix=rng.integers(0,len(unique),len(unique));n=counts[ix].sum()
        base=np.sqrt(sums['prior_surface_mean'][ix].sum()/n)
        for name in ['ensemble_equal','ensemble_rmse_weighted','ensemble_stable']:
            b.append({'model':name,'RMSE_difference_vs_surface_mean':float(np.sqrt(sums[name][ix].sum()/n)-base)})
    boot=pd.DataFrame(b).groupby('model').RMSE_difference_vs_surface_mean.agg(
        median='median',lower95=lambda x:x.quantile(.025),upper95=lambda x:x.quantile(.975))
    boot.to_csv(OUT/'paired_month_bootstrap.csv')
    print(summary.sort_values('mean_annual_RMSE').round(2).to_string(),flush=True)


def main():
    global CORE
    # Base fitting is one pipeline stage; refit_rion.py and reweight.py bring
    # the independently refitted Rion families into the final 10-model blend.
    CORE=BASE_CORE
    parser=argparse.ArgumentParser();parser.add_argument('--skip-final',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(exist_ok=True)
    d=pd.read_csv(HERE/'data/training_rich.csv')
    contract=json.loads((HERE/'data/feature_contract.json').read_text())
    features=contract['features']; assert len(d)==14113
    parts=[]
    with threadpool_limits(limits=2):
        for year in range(2020,2027):
            cache=OUT/f'oof_{year}.csv'
            if cache.exists():
                p=pd.read_csv(cache)
                assert p.event_year.eq(year).all() and set(p.pipe_id)==set(d.loc[d.event_year==year,'pipe_id'])
                parts.append(p);print(f'Using completed fold {year}',flush=True);continue
            train=d[d.event_year<year];test=d[d.event_year==year]
            past=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
            pred,_=anthony_fit(train,test,past)
            p=test[['event_id','pipe_id','event_year','cost']].copy()
            p['prior_surface_mean']=test.hist_all_surface_mean.to_numpy()
            for name,values in pred.items():p[name]=values
            for model in ['xgboost','catboost','lightgbm']:
                values,_=select_and_fit(model,d,test,year,features)
                p['takito_'+model]=values
            p.to_csv(cache,index=False)
            parts.append(p)
            print(f'Completed annual out-of-time fold {year}, n={len(p)}',flush=True)
        oof=pd.concat(parts,ignore_index=True)
        oof,w=ensemble_methods(oof)
        oof.to_csv(OUT/'out_of_time_predictions.csv',index=False)
        w.to_csv(OUT/'past_only_weights.csv',index=False)
        diagnostics(oof,d)
        if not args.skip_final:
            future=pd.read_csv(HERE/'data/forecast_features_2027.csv')
            pred,models=anthony_fit(d,future,oof)
            forecast=future[['pipe_id']].copy()
            forecast['scenario_year']=2027
            forecast['prior_surface_mean']=future.hist_all_surface_mean
            for name,values in pred.items():forecast[name]=values
            for model in ['xgboost','catboost','lightgbm']:
                values,bundle=select_and_fit(model,d,future,2027,features)
                name='takito_'+model;models[name]=bundle;forecast[name]=values
            ordinary=fit_weights(oof,0)
            stable=.5*fit_weights(oof,.02)+.5/len(CORE)
            final_weights={'ensemble_equal':np.full(len(CORE),1/len(CORE)),
                           'ensemble_rmse_weighted':ordinary,'ensemble_stable':stable}
            for name,values in final_weights.items():forecast[name]=forecast[CORE].to_numpy()@values
            forecast.to_csv(OUT/'forecast_costs_2027.csv',index=False)
            for name,bundle in models.items():joblib.dump(bundle,MODELS/(name+'.joblib'))
            dump(MODELS/'ensemble_weights.json',{name:dict(zip(CORE,values.tolist())) for name,values in final_weights.items()})
            dump(MODELS/'training_contract.json',{'all_training_rows':len(d),'training_years':[2019,2026],
                'observed_history_cutoff':'2027-01-01 exclusive','core_models':CORE,'features':features,
                'default_ensemble':'ensemble_stable','default_is_provisional':True,
                'stable_rule':'50% regularized past-OOF RMSE weights + 50% uniform; regularization=.02 fixed before this comparison',
                'supported_materials':sorted(d.material.unique().tolist()),'future_support_rows':len(forecast),
                'unsupported_policy':'PU/unknown material not assigned severity or zero cost.',
                'anthony_adaptation':'Remove realized season/time. Scenario year and January-1 age. Missing-age imputation training only. Original log RF 500 trees, leaf5, seed0; preceding-OOF smearing for expected cost.',
                'takito_adaptation':'Raw cost boosters; rich deployable families rebuilt; 6 feature packs x 2 regimes selected only on two most recent inner training years.',
                'no_guarantee_of_future_stability':True,'historical_results_already_inspected':True})
    print('Training and ensemble diagnostics complete.',flush=True)


if __name__=='__main__':main()
