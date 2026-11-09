"""Year-start tail probabilities and hierarchically pooled conditional amounts."""
from pathlib import Path
import sys,os,json,hashlib,time,itertools,warnings,argparse
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score,brier_score_loss
import joblib
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'output/2026-11-09_asymmetric_cost'
sys.path.insert(0,str(ROOT/'ensemble_training'))
import train as recipes
BASE=['scenario_year','material','surface','material_surface','pipe_length_m','log1p_pipe_length_m','midpoint_x_m','midpoint_y_m','age_at_year_start','log1p_age_at_year_start','age_missing']
CONTRACT=json.loads((ROOT/'ensemble_training/data/feature_contract.json').read_text())
PACKS={'basic':BASE,'local':recipes.packs(CONTRACT['features'])['local_history']}
FAMILIES=['lightgbm','catboost'];BINS=[2,3];CALS=['raw','platt']
YEARS=[2023,2024,2025,2026,2027]

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def label(family,pack,bins,cal):return f'{family}_{pack}_bins{bins}_{cal}'
def labels():return [label(*x) for x in itertools.product(FAMILIES,PACKS,BINS,CALS)]

def audit():
    previous=json.loads((OLD/'input_audit.json').read_text())
    for key in ['data\\pipes.csv','data\\train.csv','ensemble_training\\data\\training_rich.csv','ensemble_training\\data\\feature_contract.json']:
        assert sha(ROOT/key)==previous['source_sha256'][key],key
    assert all(x['future_mutation_invariant'] for x in previous['checks'])
    assert all(x['future_cost_location_material_mutation']=='passed' and x['future_event_removal']=='passed' for x in CONTRACT['future_invariance_checks'])
    d=pd.read_csv(ROOT/'ensemble_training/data/training_rich.csv');p=pd.read_csv(ROOT/'data/pipes.csv');h=pd.read_csv(ROOT/'data/train.csv')
    assert len(d)==len(h)==14113 and d.pipe_id.is_unique and (d.cost>0).all()
    np.testing.assert_allclose(d.cost,h.set_index('Pipe ID').loc[d.pipe_id,'Cost'])
    lay=pd.to_datetime(p.set_index('Pipe ID')['Lay date']);checks=[]
    for year in YEARS:
        q=pd.read_parquet(OLD/'features'/f'{year}.parquet')
        expected=(pd.Timestamp(f'{year}-01-01')-q.pipe_id.map(lay)).dt.days/365.25
        np.testing.assert_allclose(q.age_at_year_start,expected,equal_nan=True,rtol=0,atol=1e-10)
        assert q.scenario_year.eq(year).all() and q.pipe_id.is_unique
        assert not set(['cost','date','event_month','event_datetime','years_used_at_leak'])&set(sum(PACKS.values(),[]))
        if year<2027:
            e=d[d.event_year.eq(year)].set_index('pipe_id');x=q.set_index('pipe_id').loc[e.index]
            for cols in PACKS.values():pd.testing.assert_frame_equal(x[cols],e[cols],check_dtype=False,atol=1e-7,rtol=1e-10)
        checks.append(dict(year=year,n=len(q),query_sha256=sha(OLD/'features'/f'{year}.parquet'),age_at_jan1_verified=True,event_features_match_past_only_cache=True))
    dump(HERE/'input_audit.json',dict(training_rows=len(d),query_checks=checks,previous_invariance_audit_sha256=sha(OLD/'input_audit.json'),feature_contract_sha256=sha(ROOT/'ensemble_training/data/feature_contract.json'),protocol_sha256=sha(HERE/'PROTOCOL.md'),coordinate_units='metres',future_ages_not_used=True,source_sha256={f:sha(ROOT/f) for f in ['data/pipes.csv','data/train.csv','ensemble_training/data/training_rich.csv','ensemble_training/features.py']},code_sha256=sha(Path(__file__))))
    print('INPUT AUDIT COMPLETE',flush=True)

def calibrate(prob,y):
    if min(int(y.sum()),int(len(y)-y.sum()))<5:return dict(slope=1.,intercept=0.,fallback='fewer than 5 examples in one class')
    x=logit(np.clip(prob,1e-6,1-1e-6));y=np.asarray(y,float)
    def obj(beta):
        z=beta[0]*x+beta[1];err=expit(z)-y
        value=np.logaddexp(0,z).sum()-y@z+2*((beta[0]-1)**2+beta[1]**2)
        grad=np.array([err@x+4*(beta[0]-1),err.sum()+4*beta[1]])
        return value,grad
    fit=minimize(obj,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0.,3.),(-4.,4.)],options=dict(ftol=1e-12,maxiter=1000))
    assert fit.success,fit.message
    return dict(slope=float(fit.x[0]),intercept=float(fit.x[1]),positive_count=int(y.sum()),n=len(y),optimizer_success=True)

def calibrated(prob,params):return expit(params['slope']*logit(np.clip(prob,1e-6,1-1e-6))+params['intercept'])

def coherent(raw):
    p=np.clip(np.asarray(raw,float),0,1).copy()
    if p.shape[1]==2:
        bad=p[:,1]>p[:,0];mean=p[bad].mean(axis=1);p[bad,0]=mean;p[bad,1]=mean
    return np.column_stack([1-p[:,0],p[:,0]]) if p.shape[1]==1 else np.column_stack([1-p[:,0],p[:,0]-p[:,1],p[:,1]])

def encode(a,b,cols,family):return recipes.encode(a,b,cols,family)

def classifier(a,b,y,cols,family):
    x,z,levels=encode(a,b,cols,family)
    if family=='lightgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=100,learning_rate=.04,num_leaves=7,min_child_samples=100,reg_lambda=10,verbosity=-1,n_jobs=2,random_state=440,deterministic=True,force_col_wise=True)
        model.fit(x,y)
    else:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=150,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=440,thread_count=2,verbose=False,allow_writing_files=False)
        model.fit(x,y,cat_features=[c for c in cols if c in recipes.CATEGORICAL])
    return model.predict_proba(z)[:,1],dict(model=model,features=cols,levels=levels,family=family)

def predict_classifier(bundle,b):
    z=b[bundle['features']].copy()
    for c,levels in bundle['levels'].items():z[c]=z[c].fillna('__unknown__').astype(str) if bundle['family']=='catboost' else pd.Categorical(z[c].where(z[c].isin(levels)),categories=levels)
    return bundle['model'].predict_proba(z)[:,1]

def severity(a,thresholds,b):
    group=np.searchsorted(thresholds,a.cost.to_numpy(),side='right');lookup=[];values=[]
    for k in range(len(thresholds)+1):
        d=a[group==k];global_mean=float(d.cost.mean());assert len(d)>0
        ss=d.groupby('surface').cost.agg(['sum','count']);sm=(ss['sum']+50*global_mean)/(ss['count']+50)
        gg=d.groupby('material_surface').cost.agg(['sum','count']);prior=d[['material_surface','surface']].drop_duplicates().set_index('material_surface').surface.map(sm)
        means=(gg['sum']+20*prior.reindex(gg.index))/(gg['count']+20)
        pred=b.material_surface.map(means).fillna(b.surface.map(sm)).fillna(global_mean).to_numpy()
        assert (pred>0).all()
        values.append(pred);lookup.append(dict(n=len(d),global_mean=global_mean,surface=sm.to_dict(),material_surface=means.to_dict()))
    return np.column_stack(values),lookup

def severity_predict(lookups,b):
    return np.column_stack([b.material_surface.map(r['material_surface']).fillna(b.surface.map(r['surface'])).fillna(r['global_mean']).to_numpy() for r in lookups])

def fit_year(year):
    d=pd.read_csv(ROOT/'ensemble_training/data/training_rich.csv');a=d[d.event_year.lt(year)];inner=a[a.event_year.lt(year-1)];val=a[a.event_year.eq(year-1)]
    assert len(inner) and len(val) and inner.event_year.max()<val.event_year.min()<year
    q=pd.read_parquet(OLD/'features'/f'{year}.parquet');folder=HERE/'predictions'/str(year);folder.mkdir(parents=True,exist_ok=True)
    thresholds=[float(inner.cost.quantile(.95)),float(inner.cost.quantile(.99))]
    with warnings.catch_warnings(),threadpool_limits(limits=2):
        warnings.filterwarnings('ignore',message='.*valid feature names.*')
        for family,pack in itertools.product(FAMILIES,PACKS):
            source=folder/f'{family}_{pack}_fit.json'
            if source.exists() and all((folder/(label(family,pack,bins,cal)+'.npz')).exists() for bins,cal in itertools.product(BINS,CALS)):continue
            t=time.time();raw=[];calibrations=[];models=[]
            for cutoff in thresholds:
                y=inner.cost.ge(cutoff).to_numpy();v=val.cost.ge(cutoff).to_numpy()
                vprob,_=classifier(inner,val,y,PACKS[pack],family);cal=calibrate(vprob,v)
                prob,bundle=classifier(a,q,a.cost.ge(cutoff).to_numpy(),PACKS[pack],family)
                raw.append(prob);calibrations.append(cal);models.append(bundle)
            raw=np.column_stack(raw)
            for bins in BINS:
                th=thresholds[:bins-1];means,lookups=severity(a,th,q)
                for mode in CALS:
                    tail=raw[:,:bins-1].copy()
                    if mode=='platt':tail=np.column_stack([calibrated(tail[:,j],calibrations[j]) for j in range(tail.shape[1])])
                    probabilities=coherent(tail);prediction=(probabilities*means).sum(axis=1)
                    assert np.isfinite(prediction).all() and (prediction>0).all()
                    tag=label(family,pack,bins,mode)
                    np.savez_compressed(folder/(tag+'.npz'),pipe_id=q.pipe_id.to_numpy(str),predicted_cost=prediction,p_tail95=probabilities[:,1:].sum(1),p_tail99=probabilities[:,-1] if bins==3 else np.full(len(q),np.nan))
                    if year==2027:
                        artifact=dict(classifiers=models[:bins-1],calibrations=calibrations[:bins-1],calibration_mode=mode,severity=lookups,thresholds=th,train_end=year-1)
                        joblib.dump(artifact,folder/(tag+'.joblib'),compress=3)
                        # Inference uses selected features only, including after query-target mutation.
                        poisoned=q.head(32).copy();poisoned['cost']=1e15;poisoned['event_month']=12
                        for bb in models[:bins-1]:np.testing.assert_array_equal(predict_classifier(bb,q.head(32)),predict_classifier(bb,poisoned))
            dump(source,dict(year=year,family=family,pack=pack,train_end=int(a.event_year.max()),inner_train_end=int(inner.event_year.max()),calibration_year=year-1,thresholds=thresholds,train_rows=len(a),calibrations=calibrations,seconds=time.time()-t,features=PACKS[pack],class_weights_used=False))
            print('FIT',year,family,pack,f'{time.time()-t:.1f}s',flush=True)

def verify_models():
    q=pd.read_parquet(OLD/'features/2027.parquet');records=[]
    for path in sorted((HERE/'predictions/2027').glob('*.joblib')):
        bundle=joblib.load(path);probs=[]
        for m,c in zip(bundle['classifiers'],bundle['calibrations']):
            z=predict_classifier(m,q);probs.append(calibrated(z,c) if bundle['calibration_mode']=='platt' else z)
        expected=(coherent(np.column_stack(probs))*severity_predict(bundle['severity'],q)).sum(1)
        with np.load(path.with_suffix('.npz')) as saved:np.testing.assert_allclose(saved['predicted_cost'],expected,rtol=1e-10,atol=1e-6)
        records.append(dict(model=path.stem,n=len(q),reloaded_prediction_match=True))
    dump(HERE/'model_reload_verification.json',records)

def checks():
    out=coherent(np.array([[.1,.3],[.7,.2],[.0,.0],[1.,1.]]));assert (out>=0).all();np.testing.assert_allclose(out.sum(1),1)
    np.testing.assert_allclose(out[0],[.8,0,.2]);np.testing.assert_allclose((out*np.array([100,1000,10000])).sum(1)[0],2080)
    dump(HERE/'probability_checks.json',dict(probabilities_nonnegative=True,sum_to_one=True,threshold_order_projection=True,mixture_amount_arithmetic=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','train','verify','checks']);parser.add_argument('--year',type=int);args=parser.parse_args()
    if args.stage=='audit':audit()
    elif args.stage=='checks':checks()
    elif args.stage=='verify':verify_models()
    else:fit_year(args.year)
