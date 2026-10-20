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
 raw=100+(rate*math.pi*r*r)**.85+200*lengths[active].sum()
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
 report=dict(file=path.name,units=len(units),original_iron=int(iron.sum()),uncovered_iron=int((iron&~covered).sum()),remaining_iron=int((iron&~state).sum()),construction_cost=total/100,completion_year=year,completed_by_2051=True,skipped_units=skipped,scenario='No future leaks; not an official score prediction',annual=annual,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2));return report


from collections import Counter
import heapq
source=S/'09a_full_ensemble_net.csv'; rows=list(csv.reader(source.open())); units=read(source)
audit=pd.read_csv(S/'09a_full_ensemble_net.audit.csv'); predictions=pd.read_csv(S/'probabilities.csv').set_index('pipe_id'); costs=pd.read_csv(S/'ensemble_stable.csv').set_index('id')
prob=p['Pipe ID'].map(predictions.leak_probability).to_numpy(copy=True)
lookup=dict(zip(audit.pipe_id,audit.pipe_leak_probability_2027))
for i in np.flatnonzero(np.isnan(prob)):
 if p['Pipe ID'].iloc[i] in lookup:prob[i]=lookup[p['Pipe ID'].iloc[i]]
severity=p['Pipe ID'].map(costs.predicted_cost).to_numpy();value=prob*(severity-200*lengths);value[initial]=0
contained=[inside(*u) for u in units]
missing=np.flatnonzero(~np.isfinite(value));value[missing]=0
# Recover only missing-input contributions from published 09a aggregate benefits.
if len(missing):
 matrix=np.zeros((len(units),len(missing)));positions={v:k for k,v in enumerate(missing)}
 for j,ids in enumerate(contained):
  for i in ids:
   if i in positions:matrix[j,positions[i]]=1
 residual=audit.circle_benefit.to_numpy()-np.array([value[x].sum() for x in contained])
 assert np.linalg.matrix_rank(matrix)==int((matrix!=0).any(axis=0).sum()), 'Ambiguous missing-pipe benefits'
 recovered=np.linalg.lstsq(matrix,residual,rcond=None)[0];value[missing]=recovered
aggregate=np.array([value[x].sum() for x in contained]);err=float(np.max(np.abs(aggregate-audit.circle_benefit.to_numpy())))
assert np.allclose(aggregate,audit.circle_benefit,rtol=1e-9,atol=1e-5),err
assert hashlib.sha256(source.read_bytes()).hexdigest()=='3c871b165901564653cd1acb56cbfbd0df7f943eaf1ef733e55aa9ca20d7c1c2'
ref=validate(source);assert abs(ref['construction_cost']-152944811.51)<.011
reverse=[[] for _ in range(len(p))]
for j,ids in enumerate(contained):
 for i in ids:reverse[i].append(j)
state=initial.copy();done=np.zeros(len(units),bool);versions=np.zeros(len(units),int);heap=[]
def push(j):
 ids=contained[j];active=ids[~state[ids]]
 if len(active):
  raw=100+(sum(rates[s] for s in set(surf[ids]))*math.pi*units[j][1]**2)**.85+200*lengths[active].sum()
  score=value[active].sum()/max(raw,1)
 else:score=float('inf')
 heapq.heappush(heap,(-score,str(audit.pipe_id.iloc[j]),j,int(versions[j])))
for j in range(len(units)):push(j)
order=[]
while heap:
 _,_,j,v=heapq.heappop(heap)
 if done[j] or v!=versions[j]:continue
 done[j]=True;order.append(j)
 active=contained[j][~state[contained[j]]];state[active]=True
 affected=set(k for i in active for k in reverse[i] if not done[k])
 for k in affected:versions[k]+=1;push(k)
assert len(order)==len(units)
path=ROOT/'12a_dynamic_net_benefit_per_cost.csv'
with path.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows[j] for j in order)
assert Counter(map(tuple,csv.reader(path.open())))==Counter(map(tuple,rows))
report=validate(path)
report.update(same_circle_multiset_as_09a=True,max_original_benefit_reconstruction_error=err,missing_input_benefits_recovered_from_09a=len(missing),reference_construction_cost=ref['construction_cost'])
path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
pd.DataFrame({'rank':np.arange(1,len(order)+1),'original_09a_rank':np.array(order)+1,'pipe_id':audit.pipe_id.iloc[order].to_numpy()}).to_csv(ROOT/'12a_order.audit.csv',index=False)
print(json.dumps({k:v for k,v in report.items() if k!='annual'},indent=2))
