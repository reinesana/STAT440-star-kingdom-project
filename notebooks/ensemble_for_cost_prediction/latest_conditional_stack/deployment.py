"""Fit and apply the three fixed severity bases; prepared year-start features only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

CATEGORICAL=['material','surface','material_surface']

def base_predict(bundle,frame):
    columns=bundle['features']
    missing=set(columns)-set(frame)
    if missing: raise ValueError('Missing prepared features: '+str(sorted(missing)))
    x=frame[columns].copy()
    if bundle['family']!='optuna_pipeline':
        for col,levels in bundle['levels'].items():
            if bundle['family']=='catboost':x[col]=x[col].fillna('__unknown__').astype(str)
            else:x[col]=pd.Categorical(x[col].where(x[col].isin(levels)),categories=levels)
    p=np.asarray(bundle['model'].predict(x),dtype=float)
    if bundle['log_target']:
        p=np.exp(p)*bundle['smearing_factor']
        if bundle['log_target']=='log1p':p-=1
    p=np.maximum(p,0)
    if not np.isfinite(p).all():raise ValueError('Nonfinite base prediction')
    return p

def fit_base(name,training,config,past_oof,threads=2):
    assert training.event_year.max()==2026 and config['training_end']==2026
    cols=config['features'];params=config['params'].copy()
    params.pop('feature_pack',None);correction=params.pop('correction','none')
    assert not set(cols)&{'event_id','pipe_id','event_year','cost'}
    target='log' if name=='anthony_rf_log_mean' else 'log1p' if name=='rion_lgb_log_mean_refit' else None
    y=np.log(training.cost.to_numpy()) if target=='log' else np.log1p(training.cost.to_numpy()) if target=='log1p' else training.cost.to_numpy()
    x=training[cols].copy();levels={}
    if name=='anthony_rf_log_mean':
        numeric=[c for c in cols if c not in CATEGORICAL];categorical=[c for c in cols if c in CATEGORICAL]
        pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),numeric),
                               ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'),categorical)])
        model=make_pipeline(pre,RandomForestRegressor(**params,n_jobs=threads,random_state=440))
        family='optuna_pipeline'
    else:
        family='catboost' if name=='takito_catboost' else 'lightgbm'
        for col in set(cols)&set(CATEGORICAL):
            levels[col]=sorted(x[col].dropna().astype(str).unique().tolist())
            x[col]=x[col].fillna('__unknown__').astype(str) if family=='catboost' else pd.Categorical(x[col],categories=levels[col])
        if family=='catboost':model=CatBoostRegressor(**params,loss_function='RMSE',verbose=False,allow_writing_files=False,thread_count=threads,random_seed=440)
        else:model=LGBMRegressor(**params,objective='regression',verbosity=-1,n_jobs=threads,random_state=440,deterministic=True,force_col_wise=True)
    model.fit(x,y,**({'cat_features':[c for c in cols if c in CATEGORICAL]} if family=='catboost' else {}))
    factor=1.
    if correction=='training_residual':factor=float(np.mean(np.exp(y-model.predict(x))))
    elif correction=='past_oof':
        assert past_oof.event_year.max()<=2026
        factor=float(np.mean((past_oof.cost+(target=='log1p'))/past_oof[name+'_plain']))
    return dict(family=family,model=model,features=cols,levels=levels,log_target=target,
                smearing_factor=factor,training_end=2026,training_rows=len(training),params=config['params'])

def model_predict(model,features):
    from predict import predict
    frame=features[['pipe_id','material','surface']].copy()
    for name,bundle in model['bases'].items():frame[name]=base_predict(bundle,features)
    return predict(frame,model['stack_config']),frame
