"""Nested annual, single-booster cost comparison. No outer outcomes used in tuning."""
from pathlib import Path
import argparse, hashlib, json, os, time, warnings, sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SEED = 440
THREADS = 5
CAP = 1600
PATIENCE = 80
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def dump(path, obj):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(5):
        try:temporary.replace(path);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05)

def prepare(source):
    source = Path(source).resolve()
    t = pd.read_csv(source)
    assert t.pipe_id.is_unique
    required = ['pipe_id','date','time','cost','lay_date','material','surface','gps_x1','gps_x2','gps_y1','gps_y2']
    assert set(required)<=set(t)
    # The requested GitHub CSV is the sole source of both costs and pipe attributes.
    # Retain supplied candidate encodings and derive extra geometry/history only here.
    d = t.copy().rename(columns={'gps_x1':'x1','gps_x2':'x2','gps_y1':'y1','gps_y2':'y2'})
    event = pd.to_datetime(d.date+' '+d.time)
    lay = pd.to_datetime(d.lay_date)
    dx, dy = d.x2-d.x1, d.y2-d.y1
    d['pipe_length'] = np.hypot(dx, dy)
    d['midpoint_x'] = (d.x1+d.x2)/2
    d['midpoint_y'] = (d.y1+d.y2)/2
    d['years_used'] = (event.dt.normalize()-lay).dt.days/365.25
    d['lay_year'] = lay.dt.year
    d['lay_month'] = lay.dt.month
    d['event_year'] = event.dt.year
    d['event_month'] = event.dt.month
    d['event_weekday'] = event.dt.dayofweek
    d['event_hour'] = event.dt.hour
    d['event_weekend'] = (event.dt.dayofweek>=5).astype(int)
    d['span_x'], d['span_y'] = dx.abs(), dy.abs()
    d['orientation_sin2'] = 2*dx*dy/d.pipe_length**2
    d['orientation_cos2'] = (dx**2-dy**2)/d.pipe_length**2
    d['length_age'] = d.pipe_length*d.years_used
    d['log_length'] = np.log1p(d.pipe_length)
    d['log_age'] = np.log1p(d.years_used)
    for name, phase in [('month',(event.dt.month-1)/12),('hour',event.dt.hour/24),('weekday',event.dt.dayofweek/7)]:
        d[name+'_sin'], d[name+'_cos'] = np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)
    d['material_surface'] = d.material+' | '+d.surface
    groups = ['material','surface','material_surface']
    for size in [250,500,1000]:
        name = f'grid{size}'
        d[name] = np.floor(d.midpoint_x/size).astype(int).astype(str)+'_'+np.floor(d.midpoint_y/size).astype(int).astype(str)
        groups.append(name)
    d['age_band'] = pd.cut(d.years_used,[-np.inf,20,40,60,80,np.inf],labels=False).fillna(-1).astype(str)
    d['length_band'] = pd.cut(d.pipe_length,[-np.inf,10,25,50,100,np.inf],labels=False).astype(str)
    groups += ['age_band','length_band']
    history_cols = []
    def add(ix, name, value):
        if name not in history_cols:
            history_cols.append(name)
            d[name] = np.nan
        d.loc[ix,name] = value
    for year, ix in d.groupby('event_year').groups.items():
        h = d[d.event_year<year]
        q = d.loc[ix]
        hmean = h.cost.mean()
        hthreshold = h.cost.quantile(.99)
        hrate = (h.cost>=hthreshold).mean() if len(h) else np.nan
        hcost = h.cost.to_numpy()
        for window in ['all','last2','last1']:
            hh = h if window=='all' else h[h.event_year>=year-(2 if window=='last2' else 1)]
            prior = hh.cost.mean()
            threshold = h.cost.quantile(.99)
            rate = (hh.cost>=threshold).mean() if len(hh) else np.nan
            for key in groups:
                stats = hh.assign(tail=(hh.cost>=threshold).astype(int)).groupby(key).agg(n=('cost','size'),total=('cost','sum'),median=('cost','median'),maximum=('cost','max'),std=('cost','std'),tail=('tail','sum'))
                n = q[key].map(stats.n).fillna(0)
                total = q[key].map(stats.total).fillna(0)
                vals = {'n':n,'mean':(total+20*prior)/(n+20),'median':q[key].map(stats['median']),'max':q[key].map(stats.maximum),'std':q[key].map(stats['std']),'tail_rate':(q[key].map(stats['tail']).fillna(0)+50*rate)/(n+50)}
                for suffix, v in vals.items():
                    add(ix,f'hist_{key}_{window}_{suffix}',v)
        for radius in [50,100,250,500]:
            tree = cKDTree(h[['midpoint_x','midpoint_y']]) if len(h) else None
            neighbors = tree.query_ball_point(q[['midpoint_x','midpoint_y']],radius) if tree else [[] for _ in ix]
            rows = []
            for neighbors_ix in neighbors:
                z = hcost[neighbors_ix]
                n = len(z)
                rows.append([n,(z.sum()+20*hmean)/(n+20),np.median(z) if n else np.nan,z.max() if n else np.nan,((z>=hthreshold).sum()+50*hrate)/(n+50)])
            for j, suffix in enumerate(['n','mean','median','max','tail_rate']):
                add(ix,f'hist_radius{radius}_{suffix}',np.array(rows)[:,j])
        if len(h):
            tree = cKDTree(h[['midpoint_x','midpoint_y']])
            _, nearest = tree.query(q[['midpoint_x','midpoint_y']],k=min(25,len(h)))
            for k in [5,10,25]:
                arr = h.cost.to_numpy()[nearest[:,:k]]
                for suffix, v in [('mean',arr.mean(axis=1)),('median',np.median(arr,axis=1)),('max',arr.max(axis=1))]:
                    add(ix,f'hist_nearest{k}_{suffix}',v)
        else:
            for k in [5,10,25]:
                for suffix in ['mean','median','max']:
                    add(ix,f'hist_nearest{k}_{suffix}',np.nan)
    cats = ['material','surface','material_surface']
    excluded = ['pipe_id','date','time','cost','lay_date','grid250','grid500','grid1000','age_band','length_band']
    features = [c for c in d if c not in excluded]
    assert len(d)==len(t) and np.allclose(d.cost,t.set_index('pipe_id').cost.reindex(d.pipe_id))
    assert (d.cost>0).all() and (d.pipe_length>0).all()
    assert d.years_used.dropna().ge(0).all()
    d.to_csv(OUT/'features.csv',index=False)
    dump(OUT/'feature_manifest.json',{'features':features,'categorical':cats,'history_features':history_cols,'source_rows':len(t),'missing_age_rows':int(d.years_used.isna().sum()),'annual_counts':{str(k):int(v) for k,v in d.groupby('event_year').size().items()},'history_rule':'For every row, only costs from strictly earlier calendar years; all annual test histories frozen at previous December 31. No within-year test outcomes. Prior-year costs assumed available at year start.','coordinates':'meters; endpoint Euclidean lengths, not excavation lengths','source_path':str(source),'sha256':{source.name:hashlib.sha256(source.read_bytes()).hexdigest()}})
    print(f'PREPARED {len(d)} rows, {len(features)} candidate features',flush=True)

def matrices(a,b,features,model):
    x,z = a[features].copy(),b[features].copy()
    cats = [c for c in features if c in ['material','surface','material_surface']]
    levels = {}
    for c in cats:
        levels[c] = sorted(a[c].dropna().unique().tolist())
        if model=='catboost':
            x[c],z[c] = x[c].fillna('__missing__').astype(str),z[c].fillna('__missing__').astype(str)
        else:
            x[c] = pd.Categorical(x[c],categories=levels[c])
            z[c] = pd.Categorical(z[c].where(z[c].isin(levels[c])),categories=levels[c])
    return x,z,cats,levels

def fit(model,params,a,b,features,iterations=None,collect_curve=False):
    x,z,cats,levels = matrices(a,b,features,model)
    n = iterations or CAP
    if model=='xgboost':
        m = xgb.XGBRegressor(**params,n_estimators=n,objective='reg:squarederror',tree_method='hist',enable_categorical=True,n_jobs=THREADS,random_state=SEED,eval_metric='rmse',early_stopping_rounds=PATIENCE if iterations is None else None)
        m.fit(x,a.cost,eval_set=[(z,b.cost)] if iterations is None or collect_curve else None,verbose=False)
        best = m.best_iteration+1 if iterations is None else n
    elif model=='lightgbm':
        m = lgb.LGBMRegressor(**params,n_estimators=n,objective='regression',n_jobs=THREADS,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
        m.fit(x,a.cost,eval_set=[(z,b.cost)] if iterations is None or collect_curve else None,eval_metric='rmse' if collect_curve else None,callbacks=[lgb.early_stopping(PATIENCE,verbose=False)] if iterations is None else None)
        best = m.best_iteration_ if iterations is None else n
    else:
        m = cb.CatBoostRegressor(**params,iterations=n,loss_function='RMSE',eval_metric='RMSE',thread_count=THREADS,random_seed=SEED,allow_writing_files=False,verbose=False)
        m.fit(x,a.cost,cat_features=cats,eval_set=(z,b.cost) if iterations is None or collect_curve else None,early_stopping_rounds=PATIENCE if iterations is None else None,use_best_model=iterations is None)
        best = m.tree_count_
    prediction = np.maximum(m.predict(z),0)
    assert np.isfinite(prediction).all()
    return m,prediction,int(best),levels

def parameters(trial,model):
    p = {'learning_rate':trial.suggest_float('learning_rate',.012,.18,log=True)}
    if model=='xgboost':
        p.update(max_depth=trial.suggest_int('max_depth',2,8),min_child_weight=trial.suggest_float('min_child_weight',1,150,log=True),subsample=trial.suggest_float('subsample',.6,1),colsample_bytree=trial.suggest_float('colsample_bytree',.55,1),reg_lambda=trial.suggest_float('reg_lambda',.001,200,log=True),reg_alpha=trial.suggest_float('reg_alpha',.001,100,log=True),gamma=trial.suggest_float('gamma',.001,1e8,log=True),max_bin=trial.suggest_categorical('max_bin',[64,128,256]))
    elif model=='lightgbm':
        p.update(num_leaves=trial.suggest_int('num_leaves',4,64,log=True),max_depth=trial.suggest_int('max_depth',3,10),min_child_samples=trial.suggest_int('min_child_samples',5,160,log=True),subsample=trial.suggest_float('subsample',.6,1),subsample_freq=1,colsample_bytree=trial.suggest_float('colsample_bytree',.55,1),reg_lambda=trial.suggest_float('reg_lambda',.001,200,log=True),reg_alpha=trial.suggest_float('reg_alpha',.001,100,log=True),max_bin=trial.suggest_categorical('max_bin',[63,127,255]),min_split_gain=trial.suggest_float('min_split_gain',.001,1e8,log=True))
    else:
        p.update(depth=trial.suggest_int('depth',3,8),l2_leaf_reg=trial.suggest_float('l2_leaf_reg',.1,200,log=True),random_strength=trial.suggest_float('random_strength',.001,20,log=True),bootstrap_type='Bernoulli',subsample=trial.suggest_float('subsample',.6,1),rsm=trial.suggest_float('rsm',.55,1),border_count=trial.suggest_categorical('border_count',[64,128,254]),one_hot_max_size=trial.suggest_categorical('one_hot_max_size',[2,32]))
    return p

def feature_groups(features):
    groups = {'basic':[],'geometry':[],'age':[],'calendar':[],'history_groups':[],'history_recent':[],'history_spatial':[]}
    for c in features:
        if c.startswith('hist_'):
            key = 'history_spatial' if 'radius' in c or 'nearest' in c else ('history_recent' if 'last' in c else 'history_groups')
        elif c in ['material','surface','material_surface','midpoint_x','midpoint_y'] or c.startswith(('material_','surface_')):
            key = 'basic'
        elif 'age' in c or c.startswith('lay_') or c=='years_used':
            key = 'age'
        elif c.startswith('event_') or c.startswith(('month_','hour_','weekday_','season_')):
            key = 'calendar'
        else:
            key = 'geometry'
        groups[key].append(c)
    return groups

def score_config(model,params,features,train,inner_years):
    metrics,importance = [],np.zeros(len(features))
    for year in inner_years:
        a,b = train[train.event_year<year],train[train.event_year==year]
        m,p,n,_ = fit(model,params,a,b,features)
        err = b.cost.to_numpy()-p
        metrics.append({'year':year,'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))),'iterations':n,'n_train':len(a),'n_valid':len(b)})
        v = np.asarray(m.feature_importances_,dtype=float)
        importance += v/(v.sum() or 1)
    return float(np.mean([z['rmse'] for z in metrics])),metrics,importance

def run(model,initial,selection,polish,outer_years=None,revisit=False):
    out = OUT/model
    out.mkdir(exist_ok=True)
    d = pd.read_csv(OUT/'features.csv')
    manifest = json.loads((OUT/'feature_manifest.json').read_text())
    all_features = manifest['features']
    groups = feature_groups(all_features)
    for outer in (outer_years or [2023,2024,2025,2026]):
        target = out/str(outer)
        target.mkdir(exist_ok=True)
        incumbent = None
        if (target/'selected.json').exists() and revisit:
            incumbent = json.loads((target/'selected.json').read_text())
        elif (target/'selected.json').exists():
            print(f'{model} {outer} already selected; skipping',flush=True)
            continue
        start = time.time()
        train = d[d.event_year<outer].copy()
        candidates = [c for c in all_features if train[c].nunique(dropna=True)>1]
        inner_years = [outer-2,outer-1]
        cache = {}
        def evaluate(params,fs):
            fs = sorted(set(fs))
            assert any(c.startswith('hist_') for c in fs)
            key = json.dumps([params,fs],sort_keys=True)
            if key not in cache:
                val,details,importance = score_config(model,params,fs,train,inner_years)
                cache[key] = (val,details,dict(zip(fs,importance.tolist())))
            return cache[key]
        storage = f'sqlite:///{(target/"studies.sqlite").as_posix()}'
        study = optuna.create_study(study_name='joint',storage=storage,load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED+outer,multivariate=True,n_startup_trials=24))
        for unfinished in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(unfinished.number,state=optuna.trial.TrialState.FAIL)
        def objective(trial):
            params = parameters(trial,model)
            chosen = []
            for name,cols in groups.items():
                use = trial.suggest_categorical('use_'+name,[True,False])
                if use:
                    chosen += [c for c in cols if c in candidates]
            if not any(c.startswith('hist_') for c in chosen):
                chosen += [c for c in groups['history_groups'] if c in candidates]
            if not chosen:
                raise RuntimeError('Empty feature set')
            val,details,importance = evaluate(params,chosen)
            trial.set_user_attr('features',sorted(set(chosen)))
            trial.set_user_attr('model_params',params)
            trial.set_user_attr('inner_folds',details)
            trial.set_user_attr('importance',importance)
            return val
        def progress(study,trial):
            if (trial.number+1)%10==0:
                print(f'{model} outer {outer} {study.study_name} {trial.number+1} best_inner_RMSE={study.best_value:.2f} elapsed={time.time()-start:.0f}s',flush=True)
        # Deterministic broad feature configurations seed the adaptive search.
        if len(study.trials)==0:
            for enabled in [list(groups),['basic','history_groups'],['basic','geometry','age','calendar','history_groups'],['basic','history_spatial'],['basic','age','history_groups','history_recent']]:
                seed_config = {'use_'+g:g in enabled for g in groups}
                if model=='catboost': seed_config['one_hot_max_size']=32
                study.enqueue_trial(seed_config)
        elif model=='catboost' and not any('one_hot_max_size' in t.params for t in study.trials):
            study.enqueue_trial({**study.best_trial.params,'one_hot_max_size':32})
        study.optimize(objective,n_trials=max(0,initial-sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)),callbacks=[progress])
        best = study.best_trial
        anchor = best.user_attrs['model_params']
        ranked = sorted(best.user_attrs['importance'],key=best.user_attrs['importance'].get,reverse=True)
        # Training-only importance screen, then individual inclusion decisions.
        # Add strong features from all groups by fitting a full-feature candidate.
        full_val,full_details,full_imp = evaluate(anchor,candidates)
        pool = sorted(candidates,key=full_imp.get,reverse=True)[:45]
        pool = list(dict.fromkeys(ranked[:25]+pool))[:60]
        fs_study = optuna.create_study(study_name='individual_features',storage=storage,load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED+outer+1,n_startup_trials=12))
        def fs_objective(trial):
            fs = [c for c in pool if trial.suggest_categorical('include_'+c,[True,False])]
            if not any(c.startswith('hist_') for c in fs):
                fs += [next(c for c in pool if c.startswith('hist_'))]
            val,details,importance = evaluate(anchor,fs)
            trial.set_user_attr('features',sorted(set(fs)))
            trial.set_user_attr('model_params',anchor)
            trial.set_user_attr('inner_folds',details)
            trial.set_user_attr('importance',importance)
            return val
        if len(fs_study.trials)==0:
            for k in [5,10,15,20,30,45,60]:
                fs_study.enqueue_trial({'include_'+c:c in pool[:k] for c in pool})
            fs_study.enqueue_trial({'include_'+c:c in ranked for c in pool})
        fs_study.optimize(fs_objective,n_trials=max(0,selection-sum(t.state==optuna.trial.TrialState.COMPLETE for t in fs_study.trials)),callbacks=[progress])
        sources = [(best.value,best.user_attrs),(fs_study.best_value,fs_study.best_trial.user_attrs),(full_val,{'features':candidates,'model_params':anchor,'inner_folds':full_details,'importance':full_imp})]
        if incumbent:
            incumbent_value,incumbent_details,incumbent_importance = evaluate(incumbent['params'],incumbent['features'])
            sources.append((incumbent_value,{'features':incumbent['features'],'model_params':incumbent['params'],'inner_folds':incumbent_details,'importance':incumbent_importance}))
        value,chosen = min(sources,key=lambda v:v[0])
        # Greedy additions/removals retain only improvements in inner validation.
        fs = chosen['features'][:]
        params = chosen['model_params']
        toggles = sorted(chosen['importance'],key=chosen['importance'].get)[:10]+[c for c in pool[:15] if c not in fs]
        changes = []
        for c in dict.fromkeys(toggles):
            new = sorted(set(fs)^{c})
            if not any(v.startswith('hist_') for v in new):
                continue
            val,details,importance = evaluate(params,new)
            if val<value:
                changes.append({'feature':c,'action':'remove' if c in fs else 'add','rmse_before':value,'rmse_after':val})
                value,fs = val,new
                chosen = {'features':fs,'model_params':params,'inner_folds':details,'importance':importance}
        tune = optuna.create_study(study_name='polish',storage=storage,load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED+outer+2,multivariate=True,n_startup_trials=10))
        if not tune.trials:
            tune.enqueue_trial({k:v for k,v in params.items() if k not in ['bootstrap_type','subsample_freq']})
        def polish_objective(trial):
            p = parameters(trial,model)
            val,details,importance = evaluate(p,fs)
            trial.set_user_attr('features',fs)
            trial.set_user_attr('model_params',p)
            trial.set_user_attr('inner_folds',details)
            trial.set_user_attr('importance',importance)
            return val
        tune.optimize(polish_objective,n_trials=max(0,polish-sum(t.state==optuna.trial.TrialState.COMPLETE for t in tune.trials)),callbacks=[progress])
        if tune.best_value<value:
            value,chosen = tune.best_value,tune.best_trial.user_attrs
        # Do not inspect outer test costs or metrics here.
        selected = dict(model=model,outer_test_year=outer,train_years=[2019,outer-1],inner_years=inner_years,inner_mean_rmse=value,features=chosen['features'],params=chosen['model_params'],inner_folds=chosen['inner_folds'],iterations=int(np.median([v['iterations'] for v in chosen['inner_folds']])),greedy_changes=changes,search_trials=len(study.trials)+len(fs_study.trials)+len(tune.trials),completed_search_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for s in [study,fs_study,tune] for t in s.trials),unique_configurations_this_process=len(cache),elapsed_seconds=time.time()-start,seed=SEED,cap=CAP,patience=PATIENCE,threads=THREADS)
        dump(target/'selected.json',selected)
        pd.DataFrame([{'features':c,'inner_importance':v} for c,v in chosen['importance'].items()]).sort_values('inner_importance',ascending=False).to_csv(target/'selected_importance.csv',index=False)
        for s in [study,fs_study,tune]:
            s.trials_dataframe().to_csv(target/(s.study_name+'_trials.csv'),index=False)
        print(f'SELECTED {model} {outer}: inner RMSE={value:.2f}, {len(chosen["features"])} features, {selected["iterations"]} trees',flush=True)

def extend_selected(model,outer_years,polish):
    """Extend the final parameter search without repeating completed feature work."""
    d = pd.read_csv(OUT/'features.csv')
    for outer in (outer_years or [2023,2024,2025,2026]):
        folder = OUT/model/str(outer)
        config = json.loads((folder/'selected.json').read_text())
        train = d[d.event_year<outer]
        fs = config['features']
        start = time.time()
        study = optuna.create_study(study_name='polish',storage=f'sqlite:///{(folder/"studies.sqlite").as_posix()}',load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED+outer+3,multivariate=True,n_startup_trials=10))
        assert not study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
        def objective(trial):
            p = parameters(trial,model)
            value,metrics,importance = score_config(model,p,fs,train,config['inner_years'])
            trial.set_user_attr('features',fs)
            trial.set_user_attr('model_params',p)
            trial.set_user_attr('inner_folds',metrics)
            trial.set_user_attr('importance',dict(zip(fs,importance.tolist())))
            return value
        def progress(study,trial):
            if (trial.number+1)%10==0:
                print(f'{model} {outer} extended polish {trial.number+1} best_inner_RMSE={study.best_value:.2f} elapsed={time.time()-start:.0f}s',flush=True)
        completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,polish-completed),callbacks=[progress])
        if study.best_value<config['inner_mean_rmse']:
            best = study.best_trial.user_attrs
            config.update(features=best['features'],params=best['model_params'],inner_folds=best['inner_folds'],inner_mean_rmse=study.best_value,iterations=int(np.median([v['iterations'] for v in best['inner_folds']])))
            pd.DataFrame([{'features':c,'inner_importance':v} for c,v in best['importance'].items()]).sort_values('inner_importance',ascending=False).to_csv(folder/'selected_importance.csv',index=False)
        with sqlite3.connect(folder/'studies.sqlite') as db:
            completed_all=db.execute('SELECT COUNT(*) FROM trials WHERE state=?',('COMPLETE',)).fetchone()[0]
        config.update(completed_search_trials=completed_all,threads=THREADS,extension_seconds=time.time()-start)
        dump(folder/'selected.json',config)
        study.trials_dataframe().to_csv(folder/'polish_trials.csv',index=False)
        print(f'EXTENDED SELECTED {model} {outer}: inner RMSE={config["inner_mean_rmse"]:.2f}',flush=True)

def finalize():
    global THREADS
    d = pd.read_csv(OUT/'features.csv')
    selected_paths = [OUT/model/str(year)/'selected.json' for model in ['xgboost','catboost','lightgbm'] for year in [2023,2024,2025,2026]]
    assert all(p.exists() for p in selected_paths)
    for selected_path in selected_paths:
        chosen_config=json.loads(selected_path.read_text())
        assert 'joint_final_selection' in chosen_config
        if 'supplemental_inventory' in json.loads((OUT/'feature_manifest.json').read_text()):
            assert chosen_config.get('expanded_post_parameter_feature_check'), f'Expanded search unfinished: {selected_path}'
            assert json.loads((selected_path.parent/'exhaustive_addition_state.json').read_text())['complete']
            assert chosen_config['expanded_parameter_search']['completed_trials']>=60
            assert chosen_config['expanded_parameter_search']['features']==chosen_config['features']
            assert chosen_config.get('expanded_inner_refit_verified')
            assert chosen_config['exhaustive_expansion']['candidate_count']==len(json.loads((OUT/'feature_manifest.json').read_text())['features'])
            assert 'families_checked' in chosen_config['exhaustive_expansion']
        with sqlite3.connect(selected_path.parent/'studies.sqlite') as db:
            counts=dict(db.execute('SELECT study_name,COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id WHERE state=? GROUP BY study_name',('COMPLETE',)).fetchall())
            running=db.execute('SELECT COUNT(*) FROM trials WHERE state=?',('RUNNING',)).fetchone()[0]
        assert not running, f'Search still running: {selected_path.parent}'
        assert counts['joint']>=80 and counts['individual_features']>=40 and counts['polish']>=100
    dump(OUT/'frozen_selection_manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'selection_hashes':{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in selected_paths},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},'rule':'All configurations frozen before any outer-test prediction or metric is generated.'})
    frozen_inputs=json.loads((OUT/'frozen_selection_manifest.json').read_text())
    frozen_inputs['feature_matrix_sha256']=hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest()
    frozen_inputs['feature_manifest_sha256']=hashlib.sha256((OUT/'feature_manifest.json').read_bytes()).hexdigest()
    frozen_inputs['geometry_helper_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'lightgbm_cost_model/build_local_geometry.py',ROOT/'lightgbm_cost_model/spatial_experiment.py']}
    dump(OUT/'frozen_selection_manifest.json',frozen_inputs)
    metrics,predictions,training_predictions = [],[],[]
    for model in ['xgboost','catboost','lightgbm']:
        for fold,year in enumerate([2023,2024,2025,2026],1):
            folder = OUT/model/str(year)
            config = json.loads((folder/'selected.json').read_text())
            a,b = d[d.event_year<year],d[d.event_year==year]
            THREADS = config.get('threads',3 if model=='catboost' else 5)
            m,p,n,levels = fit(model,config['params'],a,b,config['features'],config['iterations'])
            error = b.cost.to_numpy()-p
            train_matrix,_,_,_ = matrices(a,b,config['features'],model)
            train_prediction = np.maximum(m.predict(train_matrix),0)
            train_error = a.cost.to_numpy()-train_prediction
            tp = a[['pipe_id','date','cost']].copy().rename(columns={'cost':'actual_cost'})
            tp['model'],tp['fold'],tp['test_year'],tp['predicted_cost'] = model,fold,year,train_prediction
            training_predictions.append(tp)
            metrics.append(dict(model=model,fold=fold,train_start=2019,train_end=year-1,test_year=year,n_train=len(a),n_test=len(b),MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),train_MAE=float(np.abs(train_error).mean()),train_RMSE=float(np.sqrt(np.mean(train_error**2))),features=len(config['features']),iterations=n,inner_RMSE=config['inner_mean_rmse']))
            metrics[-1]['actual_trees']=int(m.get_booster().num_boosted_rounds() if model=='xgboost' else m.booster_.num_trees() if model=='lightgbm' else m.tree_count_)
            pp = b[['pipe_id','date','cost']].copy().rename(columns={'cost':'actual_cost'})
            pp['model'],pp['fold'],pp['test_year'],pp['predicted_cost'] = model,fold,year,p
            predictions.append(pp)
            if model=='lightgbm':
                m.booster_.save_model(str(folder/'model.txt'))
            else:
                m.save_model(str(folder/('model.json' if model=='xgboost' else 'model.cbm')))
            dump(folder/'category_levels.json',levels)
            print('TEST '+json.dumps(metrics[-1]),flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv(OUT/'fold_metrics.csv',index=False)
    pp = pd.concat(predictions,ignore_index=True)
    pp.to_csv(OUT/'predictions.csv',index=False)
    pd.concat(training_predictions,ignore_index=True).to_csv(OUT/'training_predictions.csv',index=False)
    average = table.groupby('model')[['MAE','RMSE']].mean().sort_values('RMSE')
    average.to_csv(OUT/'average_metrics.csv')
    table.groupby('model')[['train_MAE','train_RMSE']].mean().to_csv(OUT/'average_training_metrics.csv')
    pooled = []
    for model,z in pp.groupby('model'):
        e = z.actual_cost-z.predicted_cost
        pooled.append({'model':model,'n':len(z),'pooled_MAE':float(e.abs().mean()),'pooled_RMSE':float(np.sqrt((e**2).mean()))})
    pd.DataFrame(pooled).to_csv(OUT/'pooled_metrics.csv',index=False)
    dump(OUT/'versions.json',{'xgboost':xgb.__version__,'catboost':cb.__version__,'lightgbm':lgb.__version__,'optuna':optuna.__version__,'pandas':pd.__version__,'threads_for_final_fits':{model:json.loads((OUT/model/'2023/selected.json').read_text())['threads'] for model in ['xgboost','catboost','lightgbm']},'seed':SEED,'source_fingerprint':json.loads((OUT/'feature_manifest.json').read_text())['sha256']})
    print('AVERAGES\n'+average.to_string(),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--source',default=str(ROOT/'github-main/data/train_predict_cost.csv'))
    parser.add_argument('--model',choices=['xgboost','catboost','lightgbm'])
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--initial',type=int,default=80)
    parser.add_argument('--selection',type=int,default=40)
    parser.add_argument('--polish',type=int,default=40)
    parser.add_argument('--outer-years',type=int,nargs='+',choices=[2023,2024,2025,2026])
    parser.add_argument('--threads',type=int,default=THREADS)
    parser.add_argument('--revisit',action='store_true')
    parser.add_argument('--extend-only',action='store_true')
    args = parser.parse_args()
    THREADS = args.threads
    OUT.mkdir(exist_ok=True)
    if args.prepare: prepare(args.source)
    if args.model:
        if args.extend_only: extend_selected(args.model,args.outer_years,args.polish)
        else: run(args.model,args.initial,args.selection,args.polish,args.outer_years,args.revisit)
    if args.finalize: finalize()
