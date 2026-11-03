"""Data leakage and survival interface checks; separate from model selection."""
import json
import tempfile
import numpy as np
import pandas as pd
import optuna
from data import BASE, Dataset, prepare, Features, raw
from models import MODELS, Estimator, propose, select


def smoke():
    prepare();d=Dataset()
    f,t,e=d.train(2020)
    rng=np.random.default_rng(440)
    ix=rng.choice(len(f),3000,replace=False)
    f=f.iloc[ix].reset_index(drop=True);t=t[ix];e=e[ix]
    checks=[]
    for name in MODELS:
        study=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=440))
        c=propose(study.ask(),name)
        c['features']='basic'
        if 'trees' in c:c['trees']=20
        if 'epochs' in c:c['epochs']=2
        cols=select(f,c,d.groups,sum(d.groups.values(),[]))
        model=Estimator(name,c,cols,2020).fit(f,t,e)
        s=model.survival(f.iloc[:50],[1,2,3])
        assert np.isfinite(s[:,:2]).all(),name
        assert ((s[:,:2]>=0)&(s[:,:2]<=1)).all(),name
        assert (np.diff(s[:,:2],axis=1)<=1e-5).all(),name
        single=model.survival(f.iloc[:50],[1])
        assert single.shape==(50,1),name
        np.testing.assert_allclose(single[:,0],s[:,0],atol=1e-6)
        checks.append(dict(model=name,finite_supported_curves=True,monotone=True,
                           support=model.support,rows=3000,smoke_only_not_evaluation=True))
        print('SMOKE PASS',name,flush=True)
    # Changing labels after cutoff must leave every training target unchanged.
    before=d.train(2022)
    d.events=d.events.loc[d.events.dt.year<=2022]
    after=d.train(2022)
    pd.testing.assert_frame_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1],after[1]);np.testing.assert_array_equal(before[2],after[2])
    (BASE/'verification.json').write_text(json.dumps(dict(smoke=checks,training_future_label_invariant=True),indent=2))


def verify_exports():
    d=Dataset();records=[]
    for name in MODELS:
        for origin in range(2023,2028):
            out=BASE/name/str(origin)
            selection=json.loads((out/'selection.json').read_text())
            assert max(selection['inner_validation_years'])<origin
            assert selection['train_years'][1]==origin-1
            p=pd.read_csv(out/'predictions.csv.gz')
            f=d.forecast(origin)
            assert list(p.pipe_id)==list(f.pipe_id)
            assert not p.pipe_id.duplicated().any()
            a=p[[f'survival_{i:02d}y' for i in range(1,26)]].to_numpy()
            assert np.isfinite(a[:,0]).all()
            finite=a[np.isfinite(a)];assert ((finite>=0)&(finite<=1)).all()
            assert (np.diff(np.nan_to_num(a,nan=-1),axis=1)<=1e-5).all()
            np.testing.assert_allclose(1-a[:,0],p.p_failure_next_year,atol=1e-7)
            records.append(dict(model=name,origin=origin,rows=len(p),passed=True))
    (BASE/'results/export_verification.json').write_text(json.dumps(records,indent=2))
    print('Verified all 50 exports')


if __name__=='__main__':
    import sys
    if '--exports' in sys.argv:verify_exports()
    else:smoke()
