"""Reproduce the frozen 21a plan from checked-in inputs (no model retraining)."""
from pathlib import Path
import sys, hashlib, json
from collections import Counter
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import simulate_score as sim
import search_orders as so
from build_cost_submissions import validate
from build_baseline import IRON
pipes,a,b=sim.load()
lengths=pipes.len.to_numpy()
audit=pd.read_csv(HERE/'severity_join.audit.csv',dtype={'pipe_id':str}).set_index('pipe_id').loc[pipes['Pipe ID']]
weights=json.loads((HERE/'weights_2027.json').read_text())['weights']
base=pd.read_csv(HERE/'base_predictions_2027.csv',dtype={'pipe_id':str}).set_index('pipe_id')
forecast=pd.read_csv(HERE/'predictions_2027.csv',dtype={'pipe_id':str}).set_index('pipe_id')
recomputed=sum(w*base.loc[forecast.index,k] for k,w in weights.items() if w)
np.testing.assert_allclose(recomputed,forecast.predicted_repair_cost,rtol=1e-12,atol=1e-8)
active=~(pipes.mat.eq('polyurethane')|pipes.leak_date.notna()).to_numpy()
severity=pipes['Pipe ID'].map(forecast.predicted_repair_cost).to_numpy(float)
missing=active & ~np.isfinite(severity)
severity[missing]=audit.old_severity.to_numpy()[missing]
severity[~active]=0
np.testing.assert_allclose(severity,audit.new_severity,rtol=1e-12,atol=1e-8)
prob=audit.leak_probability.to_numpy()
assert np.isfinite(severity).all() and np.isfinite(prob).all()
source=HERE/'20a_original_control.csv'
lines=source.read_text().splitlines()
xy,r=sim.read_units(source)
inside=sim.containment(xy,r,a,b,lengths)
value=np.clip(prob*(severity-200*lengths),0,None)
order=so.order_dynamic(inside,r,pipes,lengths,np.zeros((len(pipes),len(sim.YEARS))),static_value=value,blend=1.)
content='\n'.join(lines[k] for k in order)+'\n'
expected=(HERE/'21a_20a_catboost6863_logrf3137.csv').read_text()
assert content==expected, 'Reproduced order differs'
assert Counter(content.splitlines())==Counter(lines)
assert content.encode()==(HERE/'21a_20a_catboost6863_logrf3137.csv').read_bytes().replace(b'\r\n',b'\n')
output=HERE/'21a.reproduced.csv';output.write_text(content,encoding='utf-8',newline='\n')
history=pd.read_csv(ROOT/'data/train.csv',dtype={'Pipe ID':str})
rep=validate(output,pipes,history,a,b,(a+b)/2,lengths,pipes.Material.isin(IRON).to_numpy())
assert rep['uncovered_iron_count']==0 and rep['completed_by_2051']
assert abs(rep['construction_cost']-148028948.98)<.01
print(json.dumps(rep,indent=2))
print('Frozen 21a plan reproduced exactly. Cost check uses the historical local formula, not the revised evaluator formula.')
