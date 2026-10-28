"""Rebuild deployment from prepared training features and frozen historical OOF.

Features/OOF are not bundled; preserve the original past-only feature contract.
No hyperparameter search or future repair labels are used here.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from deployment import fit_base,model_predict

HERE=Path(__file__).resolve().parent

def weights(data,columns):
    x=data[columns].to_numpy()/100000.;y=data.cost.to_numpy()/100000.
    h=x.T@x/len(y);b=x.T@y/len(y)
    def objective(w):return float(w@h@w-2*b@w),2*(h@w-b)
    fits=[minimize(objective,start,jac=True,method='SLSQP',bounds=[(0,1)]*3,
          constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(3))],
          options=dict(ftol=1e-13,maxiter=2000)) for start in [np.ones(3)/3,*np.eye(3)]]
    fit=min((f for f in fits if f.success),key=lambda f:f.fun)
    w=np.maximum(fit.x,0);w[w<1e-10]=0;w/=w.sum()
    return w

def fit_meta(data,template):
    # Template defines the already selected recipe: tau=50 and minimum group n=50.
    # Refit learned weights; retain its historical-selection provenance separately.
    config=json.loads(json.dumps(template))
    columns=config['base_columns'];data=data[data.event_year.between(2023,2026)].copy()
    assert len(data)==8592 and data.event_year.max()==2026
    data['material_surface']=data.material+' | '+data.surface
    global_weights=weights(data,columns)
    params=config['parameters'];params['global_weights']=global_weights.tolist();params['groups']={}
    for key,q in data.groupby('material_surface'):
        if len(q)>=params['minimum_group_rows']:
            params['groups'][str(key)]=dict(n=len(q),local_weights=weights(q,columns).tolist())
    config['training_rows']=len(data)
    return config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-features',type=Path,required=True)
    p.add_argument('--past-oof',type=Path,required=True,help='2020–2026 OOF, including cost and *_plain columns for RF mean correction')
    p.add_argument('--meta-oof',type=Path,required=True,help='2023–2026 OOF with cost, the three base columns, material and surface')
    p.add_argument('--forecast-features',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):p.error('Choose a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    training=pd.read_csv(args.training_features);training=training[training.event_year.between(2019,2026)]
    past=pd.read_csv(args.past_oof);past=past[past.event_year.between(2020,2026)]
    future=pd.read_csv(args.forecast_features)
    assert len(training)==14113 and training.event_year.max()==2026
    assert future.scenario_year.eq(2027).all()
    template=json.loads((HERE/'model_2027.json').read_text())
    stack=fit_meta(pd.read_csv(args.meta_oof),template)
    model=dict(format_version=1,training_cutoff_exclusive='2027-01-01',base_training_years=[2019,2026],
               base_training_rows=len(training),stack_config=stack,bases={})
    for name in stack['base_columns']:
        config=json.loads((HERE/'base_configs'/f'{name}.json').read_text())
        model['bases'][name]=fit_base(name,training,config,past,threads=2)
        print('Fitted '+name,flush=True)
    joblib.dump(model,out/'fitted_model_2027.joblib',compress=3)
    reloaded=joblib.load(out/'fitted_model_2027.joblib')
    predictions,_=model_predict(reloaded,future)
    predictions.to_csv(out/'predictions_2027.csv',index=False)
    (out/'model_2027.json').write_text(json.dumps(stack,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Saved 3 fitted bases, refitted conditional weights and {len(predictions)} forecasts to {out}')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
