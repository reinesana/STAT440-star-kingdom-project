"""Independently compare custom censored AFT MLEs to lifelines on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from models import Estimator
from data import BASE

rng=np.random.default_rng(441)
x=rng.normal(size=1800)
records=[]
for name,kind in [('02_weibull_aft','weibull'),('03_lognormal_aft','normal')]:
    noise=np.log(rng.exponential(size=len(x))) if kind=='weibull' else rng.normal(size=len(x))
    true=np.exp(1+.3*x+.7*noise)
    censor=rng.uniform(1,7,size=len(x))
    t=np.minimum(true,censor);e=true<=censor
    f=pd.DataFrame({'age':x})
    own=Estimator(name,{'penalty':0},['age'],2026).fit(f,t,e)
    lf=(WeibullAFTFitter() if kind=='weibull' else LogNormalAFTFitter()).fit(pd.DataFrame({'x':x,'t':t,'e':e}),duration_col='t',event_col='e')
    times=np.array([1.,2.,4.])
    a=own.survival(f.iloc[:100],times)
    b=lf.predict_survival_function(pd.DataFrame({'x':x[:100]}),times=times).to_numpy().T
    error=float(np.max(np.abs(a-b)))
    assert error<2e-4,(name,error)
    records.append({'model':name,'max_abs_survival_difference':error,'reference':'lifelines unpenalized AFT','passed':True})
(BASE/'checks/aft_reference_results.json').write_text(json.dumps(records,indent=2))
print(records)
