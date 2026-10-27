from pathlib import Path
import csv,json,math,hashlib,itertools
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
S=ROOT/'source'
p=pd.read_csv(S/'pipes.csv'); h=pd.read_csv(S/'train.csv')
a=p[['GPS x1','GPS y1']].to_numpy(float); b=p[['GPS x2','GPS y2']].to_numpy(float)
m=(a+b)/2; lengths=np.linalg.norm(a-b,axis=1)
iron=p.Material.isin(['cast iron','wrought iron','gray iron']).to_numpy()
initial=(p.Material.eq('polyurethane')|p['Pipe ID'].isin(h['Pipe ID'])).to_numpy()
rates={'grassland':3,'farmland':7,'swamp':12,'road':17,'structure':50,'water':32}
surf=p.Surface.to_numpy(); grid={}; CELL=50
for i,xy in enumerate(m):grid.setdefault(tuple(np.floor(xy/CELL).astype(int)),[]).append(i)
def inside(c,r):
 lo=np.floor((c-r)/CELL).astype(int);hi=np.floor((c+r)/CELL).astype(int)
 ids=np.array([i for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1) for i in grid.get((x,y),[])],int)
 return ids[(np.sum((a[ids]-c)**2,axis=1)<=r*r)&(np.sum((b[ids]-c)**2,axis=1)<=r*r)]
def read(path):
 out=[]
 for xy,r in csv.reader(path.open()):out.append((np.array(list(map(float,xy.strip('()').split(',')))),float(r)))
 return out
def cents(c,r,ids,state):
 active=ids[~state[ids]]
 if not len(active):return 0
 rate=sum(rates[s] for s in set(surf[ids]))
 raw=100+rate*(math.pi*r*r)**.85+200*lengths[active].sum()
 return int(Decimal(str(raw)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)*100)
def validate(path):
 units=read(path);contained=[inside(*u) for u in units];covered=np.zeros(len(p),bool)
 for ids in contained:covered[ids]=True
 assert covered[iron].all()
 state=initial.copy();budget=total=cursor=skipped=0;annual=[]
 for year in range(2027,2052):
  budget+=1000000000;spent=executed=0
  while cursor<len(units):
   ids=contained[cursor];cost=cents(*units[cursor],ids,state)
   if cost>budget:break
   if cost:state[ids]=True;executed+=1
   else:skipped+=1
   total+=cost;spent+=cost;budget-=cost;cursor+=1
  annual.append(dict(year=year,spent=spent/100,balance=budget/100,remaining=len(units)-cursor))
  if cursor==len(units):break
 assert cursor==len(units) and state[iron].all()
 report=dict(file=path.name,units=len(units),original_iron=int(iron.sum()),uncovered_iron=int((iron&~covered).sum()),remaining_iron=int((iron&~state).sum()),construction_cost=total/100,completion_year=year,completed_by_2051=True,skipped_units=skipped,scenario='No future leaks; corrected surface_rate * area**0.85; not an official score prediction',annual=annual,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2));return report



from collections import Counter
import lightgbm as lgb
meta=json.loads((S/'metadata.json').read_text());model=lgb.Booster(model_file=str(S/'risk.txt'))
frame=pd.DataFrame(dict(material=p.Material,surface=p.Surface,pipe_length=lengths,midpoint_x=m[:,0],midpoint_y=m[:,1]))
for col,cats in meta['categories'].items():frame[col]=pd.Categorical(frame[col],categories=cats)
survive=np.ones(len(p))
for year in [2027,2028]:
 frame['pipe_age']=(pd.Timestamp(year,1,1)-pd.to_datetime(p['Lay date'])).dt.days/365.25
 survive*=1-model.predict(frame[meta['features']],num_threads=4)
prob=1-survive;prob[initial]=0
severity=p['Pipe ID'].map(pd.read_csv(S/'ensemble_stable.csv',dtype={'id':str}).set_index('id').predicted_cost).to_numpy(copy=True);severity[initial]=0
assert np.isfinite(severity).all() and np.isfinite(prob).all()
lookup={pid:i for i,pid in enumerate(p['Pipe ID'])}
def raw_cost(u,ids,corrected):
 active=ids[~initial[ids]]
 if not len(active):return 0.
 rate=sum(rates[s] for s in set(surf[ids]));area=math.pi*u[1]**2
 return 100+(rate*area**.85 if corrected else (rate*area)**.85)+200*lengths[active].sum()
results=[]
for label,original in [('06','06_stable_risk_per_work_cost'),('07a','07a_full_ensemble_gross'),('07b','07b_catboost_lightgbm_gross')]:
 source=S/(original+'.csv');rows=list(csv.reader(source.open()));units=read(source);contained=[inside(*u) for u in units]
 assert len(rows)==38455
 if label=='06':
  ties=[]
  # The original circle geometry identifies its source pipe uniquely.
  geom={tuple(np.round(m[i],9))+(float(f'{lengths[i]/2+1e-6:.9f}'),):str(p['Pipe ID'].iloc[i]) for i in np.flatnonzero(iron)}
  for c,radius in units:ties.append(geom[tuple(c)+(radius,)])
  ties=np.array(ties);benefit=np.array([(severity*prob)[ids].sum() for ids in contained])
 else:
  audit=pd.read_csv(S/(original+'.audit.csv'),dtype={'pipe_id':str});ties=audit.pipe_id.to_numpy();benefit=audit.circle_benefit.to_numpy()
  assert audit['rank'].tolist()==list(range(1,len(rows)+1))
  for u,pid in zip(units,ties):
   i=lookup[pid];assert np.linalg.norm(u[0]-m[i])<1e-7 and abs(u[1]-lengths[i]/2-1e-6)<1e-8
 old=np.array([raw_cost(u,ids,False) for u,ids in zip(units,contained)])
 original_order=np.lexsort((ties,-benefit/np.maximum(old,1)))
 assert np.array_equal(original_order,np.arange(len(units))), label+' original order reproduction failed'
 corrected=np.array([raw_cost(u,ids,True) for u,ids in zip(units,contained)])
 order=np.lexsort((ties,-benefit/np.maximum(corrected,1)))
 name={'06':'06_corrected_stable_risk_per_work_cost','07a':'07a_corrected_full_ensemble_gross','07b':'07b_corrected_catboost_lightgbm_gross'}[label]
 path=ROOT/(name+'.csv')
 with path.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows[j] for j in order)
 assert Counter(map(tuple,csv.reader(path.open())))==Counter(map(tuple,rows))
 rep=validate(path);rep.update(original_order_reproduced=True,same_circle_multiset_as_original=True,surface_formula='100 + surface_rate * area**0.85 + 200 * unreplaced_length',original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),official_public_score='Pending')
 path.with_suffix('.validation.json').write_text(json.dumps(rep,indent=2))
 pd.DataFrame({'rank':np.arange(1,len(order)+1),'pipe_id':ties[order],'original_rank':order+1,'gross_benefit':benefit[order],'corrected_initial_cost':corrected[order],'priority':(benefit/np.maximum(corrected,1))[order]}).to_csv(ROOT/(name+'.audit.csv'),index=False)
 results.append({k:rep[k] for k in ['file','units','construction_cost','completion_year','uncovered_iron','remaining_iron','sha256']});print(json.dumps(results[-1]),flush=True)
(ROOT/'comparison.json').write_text(json.dumps(results,indent=2))
