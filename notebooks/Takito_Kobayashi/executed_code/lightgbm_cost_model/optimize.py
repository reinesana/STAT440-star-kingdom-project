"""Reproducible bounded LightGBM search and high-cost error diagnostics."""
from pathlib import Path
import itertools
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from train_baseline import SOURCE, FEATURES, PARAMS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'optimization'


def diagnose(p, label):
    e = p.prediction - p.actual
    rows = []
    for name, mask in [('all', p.actual >= 0), ('top_1pct', p.actual >= p.actual.quantile(.99)),
                       ('top_5pct', p.actual >= p.actual.quantile(.95)),
                       ('bottom_95pct', p.actual < p.actual.quantile(.95))]:
        g, errors = p[mask], e[mask]
        rows.append(dict(model=label, segment=name, n=len(g), min_actual=g.actual.min(),
                         actual_mean=g.actual.mean(), prediction_mean=g.prediction.mean(),
                         bias=errors.mean(), mae=errors.abs().mean(), rmse=np.sqrt((errors**2).mean()),
                         underprediction_rate=float((errors < 0).mean()),
                         squared_error_share=float((errors**2).sum()/(e**2).sum())))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(SOURCE)
    old = json.loads((ROOT/'selected_config.json').read_text())
    # Freeze original selected features from the historical search, not mutable default.
    selected = json.loads((ROOT/'feature_search/selection.json').read_text())['selected']['features']
    old_predictions = pd.read_csv(ROOT/'feature_search/selected_predictions.csv')
    before = diagnose(old_predictions, 'previous_selected')
    pd.DataFrame(before).to_csv(OUT/'diagnostics_before.csv', index=False)
    print('BEFORE '+json.dumps(before), flush=True)
    joined = old_predictions.merge(d[['pipe_id','material','surface','pipe_length','years_used_at_leak']], on='pipe_id', validate='one_to_one')
    joined['error'] = joined.prediction-joined.actual
    joined['squared_error'] = joined.error**2
    joined.nlargest(30, 'squared_error').to_csv(OUT/'largest_errors_before.csv', index=False)
    joined.groupby(['material','surface']).agg(n=('actual','size'), actual_mean=('actual','mean'),
        prediction_mean=('prediction','mean'), mean_error=('error','mean'), squared_error=('squared_error','sum')).to_csv(OUT/'group_errors_before.csv')
    cache_path=OUT/'trials.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    fingerprint=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    def run(features, overrides, save=False):
        features=sorted(features)
        params={**PARAMS, **overrides}
        key=json.dumps([fingerprint,features,params],sort_keys=True)
        if key in cache and not save: return cache[key]
        metrics, predictions=[],[]
        for year in range(2023,2027):
            tr,te=d[d.event_year<year],d[d.event_year==year]
            inner,valid=tr[tr.event_year<year-1],tr[tr.event_year==year-1]
            def matrix(a,b):
                x,v=a[features].copy(),b[features].copy()
                for c in set(features)&{'material','surface','material_code','surface_code'}:
                    levels=sorted(x[c].unique())
                    x[c]=pd.Categorical(x[c],categories=levels)
                    v[c]=pd.Categorical(v[c],categories=levels)
                return x,v
            xi,xv=matrix(inner,valid)
            def metric(y,p): return 'dollar_rmse',float(np.sqrt(np.mean((y-np.maximum(p,0))**2))),False
            model=lgb.LGBMRegressor(**params,n_estimators=3000,metric='None')
            model.fit(xi,inner.cost,eval_set=[(xv,valid.cost)],eval_metric=metric,
                      callbacks=[lgb.early_stopping(100,verbose=False)])
            iterations=model.best_iteration_
            x,z=matrix(tr,te)
            model=lgb.LGBMRegressor(**params,n_estimators=iterations)
            model.fit(x,tr.cost)
            p=np.maximum(model.predict(z),0)
            assert np.isfinite(p).all()
            err=p-te.cost.to_numpy()
            metrics.append(dict(year=year,rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.abs(err).mean()),iterations=iterations))
            if save:
                model.booster_.save_model(str(OUT/f'model_{year}.txt'))
                predictions.append(pd.DataFrame(dict(pipe_id=te.pipe_id,year=year,actual=te.cost,prediction=p)))
        result=dict(features=features,params=params,rmse=float(np.mean([m['rmse'] for m in metrics])),
                    mae=float(np.mean([m['mae'] for m in metrics])),folds=metrics)
        cache[key]=result
        cache_path.write_text(json.dumps(cache,indent=2))
        if save:
            p=pd.concat(predictions,ignore_index=True)
            p.to_csv(OUT/'predictions.csv',index=False)
            pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
            pd.DataFrame(before+diagnose(p,'optimized')).to_csv(OUT/'diagnostics_comparison.csv',index=False)
        return result
    best=run(selected,{})
    for features in [selected,FEATURES]:
        for leaves,minimum,regularization in itertools.product([15,31,63],[10,20,50],[0.,1.,10.]):
            r=run(features,dict(num_leaves=leaves,min_child_samples=minimum,reg_lambda=regularization))
            if r['rmse']<best['rmse']:
                best=r
                print(f"NEW BEST trial {len(cache)} RMSE {best['rmse']:.2f} leaves={leaves} min_samples={minimum} lambda={regularization} features={len(features)}",flush=True)
            elif len(cache)%10==0: print(f"Completed {len(cache)} configurations; best RMSE {best['rmse']:.2f}",flush=True)
    # Refine regularization / sampling / learning rate around the grid winner.
    anchor=best
    for extra in [dict(learning_rate=.01),dict(learning_rate=.05),dict(reg_alpha=1.),
                  dict(reg_alpha=10.),dict(subsample=.8,subsample_freq=1),dict(colsample_bytree=.8),
                  dict(min_child_samples=5),dict(min_child_samples=100)]:
        r=run(anchor['features'],{**anchor['params'],**extra})
        if r['rmse']<best['rmse']: best=r
    # Reassess all single additions/removals at tuned settings until convergence.
    candidates=[c for c in d if c not in ['pipe_id','cost','date','time','lay_date']]
    history=[]
    while True:
        winner=best
        action=None
        for c in candidates:
            if c=='event_year':continue
            fs=set(best['features']); fs.symmetric_difference_update([c])
            r=run(fs,best['params'])
            if r['rmse']<winner['rmse']-1e-6:
                winner=r; action=('remove ' if c in best['features'] else 'add ')+c
        if action is None:break
        best=winner
        history.append(dict(action=action,rmse=best['rmse']))
        print(f"FEATURE {action}: RMSE {best['rmse']:.2f}",flush=True)
    final=run(best['features'],best['params'],save=True)
    assert np.isclose(final['rmse'],best['rmse'])
    result=dict(best=final,configurations=len(cache),feature_changes=history,source_sha256=fingerprint,
                prior_default=old,lightgbm_version=lgb.__version__,
                caveat='All four outer years used for model selection; no untouched final test. Bounded search, not global optimum.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame([dict(rmse=r['rmse'],mae=r['mae'],features=','.join(r['features']),params=json.dumps(r['params'])) for r in cache.values()]).sort_values('rmse').to_csv(OUT/'leaderboard.csv',index=False)
    (ROOT/'selected_config.json').write_text(json.dumps(dict(features=final['features'],target='raw',params=final['params'],validation_rmse=final['rmse'],note=result['caveat']),indent=2))
    print('FINAL '+json.dumps(result),flush=True)


if __name__=='__main__':main()
