"""Nested forward validation. Selection never accepts outer-year labels.

Run: python experiment.py --trials 40 --deployment-trials 60
Resumes completed origins and Optuna studies; do not edit code during a study.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits
from data import BASE, REPO, SEED, Dataset, prepare
from models import MODELS, Estimator, ranking, select, propose

optuna.logging.set_verbosity(optuna.logging.WARNING)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf8')
    temporary.replace(path)


def metrics(y, p):
    p=np.clip(p,1e-7,1-1e-7)
    return dict(brier=float(brier_score_loss(y,p)), log_loss=float(log_loss(y,p,labels=[0,1])),
        average_precision=float(average_precision_score(y,p)) if np.sum(y) else float('nan'),
        f1_at_05=float(f1_score(y,p>=.5,zero_division=0)),
        precision_at_05=float(precision_score(y,p>=.5,zero_division=0)),
        recall_at_05=float(recall_score(y,p>=.5,zero_division=0)),
        actual_events=int(y.sum()), expected_events=float(p.sum()), rows=len(y))


def optimize(data, name, origin, budget, out):
    """Only origins < outer origin enter both tuning and feature ranking."""
    inner_years=[origin-2, origin-1]
    inner=[]
    for year in inner_years:
        train,duration,event=data.train(year-1)
        val=data.forecast(year)
        rank_path=BASE/'cache'/f'ranking_{year-1}.json'
        if rank_path.exists(): ranked=json.loads(rank_path.read_text())
        else:
            ranked=ranking(train,duration,event,data.groups)
            write_json(rank_path,ranked)
        inner.append((year,train,duration,event,val,data.labels(val,year),ranked))
    storage=f'sqlite:///{(out / "study_nested.sqlite3").as_posix()}'
    study=optuna.create_study(study_name=f'{name}_{origin}', storage=storage, load_if_exists=True,
        direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED+origin,n_startup_trials=12),
        pruner=optuna.pruners.NopPruner())
    if not study.trials:
        for initial in [{'features':'basic'}, {'features':'all'},
                        {'features':'ranked','top_k':8}, {'features':'ranked','top_k':24},
                        {'features':'ranked','top_k':64}]:
            study.enqueue_trial(initial)
    # Fixed budgets, no outer-label-based pruning or early stopping.
    def objective(trial):
        started=time.time()
        c=propose(trial,name)
        trial.set_user_attr('full_config',c)
        scores=[]; selected=[]
        try:
            with threadpool_limits(limits=4):
                for year,train,duration,event,val,y,ranked in inner:
                    cols=select(train,c,data.groups,ranked)
                    estimator=Estimator(name,c,cols,year-1).fit(train,duration,event)
                    horizons=list(range(1,min(2,origin-year)+1))
                    assert max(year+h-1 for h in horizons)<origin
                    s=estimator.survival(val,horizons)
                    if not np.isfinite(s).all(): raise ValueError('Nonfinite inner predictions')
                    horizon_scores=[dict(horizon=h,**metrics(data.labels(val,year,h),1-s[:,i]))
                                    for i,h in enumerate(horizons)]
                    scores.append(dict(validation_year=year,horizons=horizon_scores,
                                       brier=float(np.mean([m['brier'] for m in horizon_scores]))))
                    selected.append(cols)
            trial.set_user_attr('inner_metrics',scores)
            trial.set_user_attr('selected_features',selected)
            return float(np.mean([m['brier'] for m in scores]))
        finally:
            trial.set_user_attr('elapsed_seconds',time.time()-started)
    # Enqueued WAITING candidates are not attempted trials; optimize consumes
    # them within n_trials. Counting them here would silently subtract five.
    remaining=max(0,budget-sum(t.state.name!='WAITING' for t in study.trials))
    study.optimize(objective,n_trials=remaining,catch=(Exception,),gc_after_trial=True)
    trials=[]
    for t in study.trials:
        trials.append(dict(number=t.number,state=t.state.name,value=t.value,params=t.params,**t.user_attrs))
    write_json(out/'trials.json',trials)
    study.trials_dataframe().to_csv(out/'trials.csv',index=False)
    complete=[t for t in study.trials if t.state.name=='COMPLETE' and np.isfinite(t.value)]
    if not complete: raise RuntimeError(f'No valid trial for {name} {origin}')
    best=study.best_trial
    config=best.user_attrs['full_config']
    train,duration,event=data.train(origin-1)
    rank_path=BASE/'cache'/f'ranking_{origin-1}.json'
    if rank_path.exists(): ranked=json.loads(rank_path.read_text())
    else:
        ranked=ranking(train,duration,event,data.groups);write_json(rank_path,ranked)
    columns=select(train,config,data.groups,ranked)
    selection=dict(model=name,origin=origin,train_years=[2019,origin-1],inner_validation_years=inner_years,
        best_trial=best.number,inner_mean_brier=best.value,config=config,selected_features=columns,
        requested_trials=budget,completed_trials=len(complete),failed_trials=sum(t.state.name=='FAIL' for t in study.trials),
        training_landmarks=len(train),training_event_landmarks=int(event.sum()),
        unique_training_pipes=int(train.pipe_id.nunique()))
    write_json(out/'selection.json',selection)
    return selection,train,duration,event


def export_prediction(model,frame,origin,out):
    s=model.survival(frame,np.arange(1,26))
    supported=np.isfinite(s)
    assert np.isfinite(s[:,0]).all()
    assert ((s[supported]>=0)&(s[supported]<=1)).all()
    assert (np.diff(np.nan_to_num(s,nan=-1),axis=1)<=1e-5).all()
    pred=frame[['pipe_id']].copy()
    pred['origin_year']=origin
    pred['p_failure_next_year']=1-s[:,0]
    pred['support_horizon_years']=model.support
    pred['missing_install_date']=frame.lay_missing.astype(int)
    pred['unknown_material']=frame.material.eq('__unknown__').astype(int)
    for q in (.1,.5,.9):
        crossed=(s<=1-q)&supported
        has=crossed.any(axis=1)
        year=np.where(has,origin+crossed.argmax(axis=1),np.nan)
        pred[f'failure_year_q{int(q*100):02d}']=year
    pred['median_status']=np.where(pred.failure_year_q50.notna(),'within_supported_horizon',
        'beyond_2051_or_not_identified' if model.support==25 else 'not_identified_beyond_empirical_support')
    for i in range(25): pred[f'survival_{i+1:02d}y']=s[:,i]
    pred.to_csv(out/'predictions.csv.gz',index=False,float_format='%.8g',compression={'method':'gzip','mtime':0})
    return pred


def run_model(data,name,budget,deployment_budget,origins):
    for origin in origins:
        out=BASE/name/str(origin);out.mkdir(parents=True,exist_ok=True)
        required=deployment_budget if origin==2027 else budget
        attempted=len(json.loads((out/'trials.json').read_text())) if (out/'trials.json').exists() else 0
        if (out/'finished.json').exists() and attempted>=required:
            print(f'SKIP {name} {origin}: already complete',flush=True);continue
        started=time.time(); print(f'START {name} {origin}',flush=True)
        selection,train,duration,event=optimize(data,name,origin,deployment_budget if origin==2027 else budget,out)
        with threadpool_limits(limits=4):
            estimator=Estimator(name,selection['config'],selection['selected_features'],origin-1).fit(train,duration,event)
            frame=data.forecast(origin)
            pred=export_prediction(estimator,frame,origin,out)
        model_dir=BASE/'models'/name;model_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(estimator,model_dir/f'{origin}.joblib',compress=3)
        write_json(out/'finished.json',dict(model=name,origin=origin,rows=len(pred),support_years=estimator.support,
            elapsed_seconds=time.time()-started,prediction_sha256=hashlib.sha256((out/'predictions.csv.gz').read_bytes()).hexdigest()))
        print(f'FINISH {name} {origin} inner_Brier={selection["inner_mean_brier"]:.6f} seconds={time.time()-started:.1f}',flush=True)


def report(data):
    # Called only after every requested configuration and prediction is frozen.
    rows=[]; oof=[]; multiyear=[]
    for name in MODELS:
        for origin in range(2023,2027):
            out=BASE/name/str(origin)
            if not (out/'finished.json').exists(): continue
            p=pd.read_csv(out/'predictions.csv.gz')
            frame=data.forecast(origin)
            assert list(p.pipe_id)==list(frame.pipe_id)
            y=data.labels(frame,origin)
            rows.append(dict(model=name,year=origin,**metrics(y,p.p_failure_next_year.to_numpy())))
            oof.append(pd.DataFrame(dict(model=name,pipe_id=p.pipe_id,year=origin,y_true=y,predicted_probability=p.p_failure_next_year)))
            for horizon in range(2,min(5,2028-origin)):
                prob=1-p[f'survival_{horizon:02d}y'].to_numpy()
                if np.isfinite(prob).all():
                    multiyear.append(dict(model=name,origin=origin,horizon=horizon,
                        **metrics(data.labels(frame,origin,horizon),prob)))
    results=BASE/'results';results.mkdir(exist_ok=True)
    if not rows:return
    scores=pd.DataFrame(rows)
    scores.to_csv(results/'annual_metrics.csv',index=False)
    summary=scores.groupby('model')[['brier','log_loss','average_precision','f1_at_05','precision_at_05','recall_at_05']].mean()
    summary.sort_values('brier').to_csv(results/'mean_annual_metrics.csv')
    pd.DataFrame(multiyear).to_csv(results/'multi_year_metrics.csv',index=False)
    oof=pd.concat(oof,ignore_index=True)
    oof.to_csv(results/'stacking_oof.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pivot=oof.pivot(index=['pipe_id','year','y_true'],columns='model',values='predicted_probability')
    if not pivot.isna().any().any():
        residuals=pivot.subtract(pivot.index.get_level_values('y_true').to_numpy(),axis=0)
        residuals.corr().to_csv(results/'oof_residual_correlations.csv')
        for attribute in ['material','lay_decade']:
            groups=[]
            for origin in range(2023,2027):
                f=data.forecast(origin)
                groups.append(pd.DataFrame(dict(pipe_id=f.pipe_id,year=origin,group=f[attribute])))
            joined=oof.merge(pd.concat(groups),on=['pipe_id','year'],validate='many_to_one')
            output=[]
            for (name,year,group),g in joined.groupby(['model','year','group']):
                output.append(dict(model=name,year=year,group=group,**metrics(g.y_true.to_numpy(),g.predicted_probability.to_numpy())))
            pd.DataFrame(output).to_csv(results/f'metrics_by_{attribute}.csv',index=False)
    for metric in ['brier','log_loss','average_precision','f1_at_05']:
        table=scores.pivot(index='model',columns='year',values=metric)
        table['Mean']=table.mean(axis=1)
        table.to_csv(results/f'shared_doc_{metric}.tsv',sep='\t')
    print(summary.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=40)
    parser.add_argument('--deployment-trials',type=int,default=60)
    parser.add_argument('--models',nargs='+',default=MODELS,choices=MODELS)
    parser.add_argument('--origins',nargs='+',type=int,default=list(range(2023,2028)))
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    prepare();data=Dataset()
    if not args.report_only:
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
        path=BASE/'experiment_contract.json'
        contract=dict(seed=SEED,outer_years=[2023,2024,2025,2026],deployment_origin=2027,
            objective='mean across two prior origins of their mean 1/2-year Brier, truncated before outer year',
            features='joint semantic-group / supervised-ranked subset and hyperparameter search',
            requested_outer_trials=args.trials,requested_deployment_trials=args.deployment_trials,
            code_sha256=code_hashes,shared_record='https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit',
            censoring='administrative at training year end; annual event timing uses year midpoint',
            landmarking='all still-at-risk pipes at each historical Jan 1; correlated within pipe',
            forecast='first leak without future planned replacement; observed history frozen at forecast origin',
            asymmetric_penalty=False,source_commit='668a598ba8efd478d1c5bf569b39fc7fb93162e2')
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['code_sha256']!=code_hashes: raise RuntimeError('Code changed after experiment freeze; archive old studies before rerun')
        else:write_json(path,contract)
        versions={v:importlib.metadata.version(v) for v in ['numpy','pandas','scipy','scikit-learn','scikit-survival','xgboost','catboost','torch','optuna','joblib']}
        write_json(BASE/'environment.json',dict(python=sys.version,platform=platform.platform(),versions=versions))
        for name in args.models:run_model(data,name,args.trials,args.deployment_trials,args.origins)
    report(data)


if __name__=='__main__':main()
