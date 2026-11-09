from pathlib import Path
import sys, os, json, hashlib, time, warnings, argparse, importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ensemble_training'))
import optimize as op
from features import FeatureBuilder
sys.path.insert(0,str(ROOT/'output/2026-10-27_order_benchmark'))
import run as bench
MODELS=['xgboost','catboost','lightgbm','random_forest','extra_trees','ridge']
RATIOS=[1.]+[round(1.2+.2*i,1) for i in range(10)]
YEARS=[2023,2024,2025,2026]
SCALE=10000.

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda z:z.item() if isinstance(z,np.generic) else str(z),allow_nan=False),encoding='utf8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def loss(y,p,k): return float(np.mean(.5*np.where(p<y,k,1.)*(p-y)**2))

def objective(k):
    def f(y,p):
        w=np.where(p<y,k,1.)
        return w*(p-y),w
    return f

class CatObjective:
    def __init__(self,k): self.k=k
    def calc_ders_range(self,approxes,targets,weights):
        out=[]
        for i in range(len(targets)):
            w=self.k if approxes[i]<targets[i] else 1.
            if weights is not None: w*=weights[i]
            out.append(((targets[i]-approxes[i])*w,-w))
        return out

def cat_objective(k):
    # CatBoost can JIT a closure with a fixed ratio; using self.k prevents JIT.
    class FixedObjective:
        def calc_ders_range(self,approxes,targets,weights):
            out=[]
            for i in range(len(targets)):
                w=k if approxes[i]<targets[i] else 1.
                if weights is not None: w*=weights[i]
                out.append(((targets[i]-approxes[i])*w,-w))
            return out
    return FixedObjective()

def expectile(y,k):
    value=float(np.mean(y))
    for _ in range(100):
        w=np.where(y>value,k,1.);new=float(w@y/w.sum())
        if abs(new-value)<1e-10: break
        value=new
    return value

def prepare():
    d=pd.read_csv(ROOT/'ensemble_training/data/training_rich.csv')
    p=pd.read_csv(ROOT/'data/pipes.csv');h=pd.read_csv(ROOT/'data/train.csv')
    h['year']=pd.to_datetime(h.Date).dt.year
    assert len(d)==len(h)==14113 and d.pipe_id.is_unique
    raw=h.set_index('Pipe ID');np.testing.assert_allclose(d.cost,raw.loc[d.pipe_id,'Cost'])
    builder=FeatureBuilder()
    future=pd.read_csv(ROOT/'ensemble_preparation/prepared/forecast_features_2027_supported.csv')
    source=pd.concat([builder.raw.drop(columns=['cost'],errors='ignore'),future],ignore_index=True).drop_duplicates('pipe_id')
    lay=pd.to_datetime(p.set_index('Pipe ID')['Lay date']);checks=[]
    for year in YEARS+[2027]:
        path=HERE/'features'/f'{year}.parquet'
        if path.exists(): continue
        initial=p.Material.eq('polyurethane')|p['Pipe ID'].isin(h.loc[h.year.lt(year),'Pipe ID'])
        q=source[source.pipe_id.isin(p.loc[~initial,'Pipe ID'])].copy()
        q['scenario_year']=year
        q['age_at_year_start']=(pd.Timestamp(f'{year}-01-01')-q.pipe_id.map(lay)).dt.days/365.25
        q['log1p_age_at_year_start']=np.log1p(q.age_at_year_start)
        x=builder.build(q,year)
        # Query target mutation and all future-history removal must leave features identical.
        sample=q.head(12).copy();sample['cost']=1e14
        events=builder.events.copy();events.loc[events.event_year.ge(year),'cost']=1e14
        pd.testing.assert_frame_equal(builder.build(sample,year,events),builder.build(q.head(12),year,events[events.event_year.lt(year)]))
        # Matched event features must equal the independently cached training features.
        if year<2027:
            event=d[d.event_year.eq(year)].copy();ix=pd.Index(q.pipe_id).get_indexer(event.pipe_id);assert (ix>=0).all()
            pd.testing.assert_frame_equal(x.iloc[ix].reset_index(drop=True)[event.columns.intersection(x.columns)],event.reset_index(drop=True)[event.columns.intersection(x.columns)],check_dtype=False,atol=1e-7,rtol=1e-10)
        x.insert(0,'pipe_id',q.pipe_id.to_numpy());path.parent.mkdir(exist_ok=True);x.to_parquet(path,index=False)
        checks.append(dict(year=year,query_rows=len(x),active_pipes=int((~initial).sum()),missing=int((~initial).sum()-len(x)),future_mutation_invariant=True))
        print('PREPARE',year,len(x),flush=True)
    dump(HERE/'input_audit.json',dict(checks=checks,training_rows=len(d),source_sha256={str(z.relative_to(ROOT)):sha(z) for z in [ROOT/'data/pipes.csv',ROOT/'data/train.csv',ROOT/'ensemble_training/data/training_rich.csv',ROOT/'ensemble_training/data/feature_contract.json',Path(__file__),HERE/'PROTOCOL.md']},shared_doc_sha256=sha(HERE/'shared_document_snapshot.txt')))

def columns_params(name,year):
    if name in ['xgboost','catboost','lightgbm']:
        path=ROOT/'ensemble_training/optuna/results/configs'/('takito_'+name)/f'{year}.json'
        cfg=json.loads(path.read_text());assert cfg['training_end']<year and max(cfg['inner_years'])<year
        params=cfg['params'].copy();params.pop('feature_pack',None)
        return cfg['features'],params,dict(source=str(path.relative_to(ROOT)),sha256=sha(path),training_end=cfg['training_end'])
    cols=op.previous.packs(json.loads((ROOT/'ensemble_training/data/feature_contract.json').read_text())['features'])['local_history']
    params=dict(n_estimators=128,max_depth=8,min_samples_leaf=30,max_features=.7,n_jobs=2,random_state=440)
    return cols,params,dict(source='fixed local-history recipe; no per-year tuning',training_end=year-1)

def transform(a,b,cols):
    cat=[c for c in cols if c in op.previous.CATEGORICAL];num=[c for c in cols if c not in cat]
    pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
    return pre.fit_transform(a),pre.transform(b),pre

def fit(name,a,b,year,k):
    cols,params,source=columns_params(name,year);y=a.cost.to_numpy(float)
    log=[];bundle=dict(name=name,year=year,k=k,features=cols,params=params,source=source)
    if name in ['xgboost','catboost','lightgbm']:
        x,z,levels=op.previous.encode(a,b,cols,name);center=expectile(y,k);yt=y-center
        if name=='xgboost':
            from xgboost import XGBRegressor
            m=XGBRegressor(**params,objective=objective(k),base_score=0.,tree_method='hist',enable_categorical=True,n_jobs=2,random_state=440)
        elif name=='lightgbm':
            from lightgbm import LGBMRegressor
            m=LGBMRegressor(**params,objective=objective(k),verbosity=-1,n_jobs=2,random_state=440,deterministic=True,force_col_wise=True)
        else:
            from catboost import CatBoostRegressor
            m=CatBoostRegressor(**params,loss_function=cat_objective(k),eval_metric='RMSE',boost_from_average=False,verbose=False,allow_writing_files=False,thread_count=2,random_seed=440)
        m.fit(x,yt,**({'cat_features':[c for c in cols if c in op.previous.CATEGORICAL]} if name=='catboost' else {}))
        pred=m.predict(z)+center;bundle.update(model=m,levels=levels,center=center)
        log.append(dict(iteration=0,loss=loss(y,m.predict(x)+center,k)))
    else:
        x,z,pre=transform(a,b,cols);bundle['transform']=pre
        if name=='ridge':
            xx=np.column_stack([np.ones(len(x)),x]);zz=np.column_stack([np.ones(len(z)),z]);yt=y/SCALE
            alpha=100.
            def fun(beta):
                err=xx@beta-yt;w=np.where(err<0,k,1.)
                return .5*np.sum(w*err**2)+.5*alpha*np.sum(beta[1:]**2),xx.T@(w*err)+alpha*np.r_[0,beta[1:]]
            initial=np.r_[expectile(yt,k),np.zeros(x.shape[1])]
            result=minimize(fun,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=3000,ftol=1e-12,gtol=1e-7))
            assert result.success,result.message
            pred=zz@result.x*SCALE;bundle.update(coef=result.x,scale=SCALE,alpha=alpha)
            log.append(dict(iteration=int(result.nit),loss=loss(y,xx@result.x*SCALE,k),converged=True))
        else:
            weights=np.ones(len(y));best=np.inf;bestmodel=None;bestpred=None
            for iteration in range(1 if k==1 else 4):
                cls=RandomForestRegressor if name=='random_forest' else ExtraTreesRegressor
                m=cls(**params);m.fit(x,y,sample_weight=weights)
                tr=m.predict(x);value=loss(y,tr,k);log.append(dict(iteration=iteration,loss=value,underestimate_fraction=float(np.mean(tr<y))))
                if value<best: best=value;bestmodel=m;bestpred=m.predict(z)
                new=np.where(tr<y,k,1.)
                if np.array_equal(new,weights): break
                weights=new
            bundle['model']=bestmodel;pred=bestpred;bundle['irls_max_fits']=4
    pred=np.maximum(pred,0);assert np.isfinite(pred).all()
    bundle['training_trace']=log
    return pred,bundle

def train(name,years,ratios=RATIOS):
    d=pd.read_csv(ROOT/'ensemble_training/data/training_rich.csv')
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*valid feature names.*')
        warnings.filterwarnings('ignore',message='.*optimize method.*')
        for year in years:
            a=d[d.event_year.lt(year)];b=pd.read_parquet(HERE/'features'/f'{year}.parquet')
            for k in ratios:
                folder=HERE/'fits'/str(year);folder.mkdir(parents=True,exist_ok=True);tag=f'{name}_k{k:.1f}';dest=folder/(tag+'.npy')
                if dest.exists(): continue
                t=time.time();pred,bundle=fit(name,a,b,year,k)
                np.save(dest,pred)
                metadata={z:val for z,val in bundle.items() if z not in ['model','transform','coef','levels']}
                metadata.update(seconds=time.time()-t,train_rows=len(a),forecast_rows=len(b),train_end=int(a.event_year.max()))
                dump(folder/(tag+'.json'),metadata)
                if year==2027:
                    # Custom objective closure is not required for inference.
                    if name in ['xgboost','lightgbm']:bundle['model'].set_params(objective='reg:squarederror' if name=='xgboost' else 'regression')
                    if name=='catboost': bundle['model'].save_model(str(folder/(tag+'.cbm')));bundle['model_path']=str(folder/(tag+'.cbm'));bundle.pop('model')
                    joblib.dump(bundle,folder/(tag+'.joblib'),compress=3)
                print('FIT',year,name,k,f'{time.time()-t:.1f}s',flush=True)

def checks():
    y=np.array([2.,5.,8.]);p=np.array([1.,7.,9.]);grad,hess=objective(3)(y,p)
    np.testing.assert_allclose(grad,[-3,2,1]);np.testing.assert_allclose(hess,[3,1,1])
    # Finite difference checks away from the kink.
    for j in range(3):
        plus=p.copy();minus=p.copy();plus[j]+=1e-5;minus[j]-=1e-5
        numerical=(loss(y,plus,3)-loss(y,minus,3))*len(y)/(2e-5)
        np.testing.assert_allclose(numerical,grad[j],rtol=1e-6)
    np.testing.assert_allclose(np.asarray(CatObjective(3).calc_ders_range(p,y,None)),np.column_stack([-grad,-hess]))
    for ratio in RATIOS:
        gg,hh=objective(ratio)(y,p)
        np.testing.assert_allclose(np.asarray(cat_objective(ratio).calc_ders_range(p,y,None)),np.column_stack([-gg,-hh]))
    assert expectile(np.array([0.,10.]),3)==7.5
    dump(HERE/'loss_verification.json',dict(asymmetry_ratio_correct=True,finite_difference_gradient=True,catboost_negative_derivative_sign=True,constant_expectile_example=7.5))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','checks']);parser.add_argument('--model',choices=MODELS);parser.add_argument('--years',nargs='+',type=int,default=YEARS);parser.add_argument('--ratios',nargs='+',type=float,default=RATIOS);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='checks':checks()
    else:train(args.model,args.years,args.ratios)
