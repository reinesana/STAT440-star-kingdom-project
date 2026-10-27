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
source=S/'09b_original.csv';rows=list(csv.reader(source.open()));units=read(source)
audit=pd.read_csv(S/'09b_original.audit.csv',dtype={'pipe_id':str})
assert len(rows)==len(audit)==38455
assert audit['rank'].tolist()==list(range(1,38456))
lookup={pid:i for i,pid in enumerate(p['Pipe ID'])}
for u,pid in zip(units,audit.pipe_id):
 i=lookup[pid];assert np.linalg.norm(u[0]-m[i])<1e-7 and abs(u[1]-lengths[i]/2-1e-6)<1e-8
benefit=audit.circle_benefit.to_numpy();old_cost=audit.initial_circle_work_cost.to_numpy();tie=audit.pipe_id.to_numpy()
assert np.isfinite(benefit).all()
assert np.array_equal(np.lexsort((tie,-benefit/np.maximum(old_cost,1))),np.arange(len(rows)))
contained=[inside(*u) for u in units]
def initial_cost(u,ids):
 active=ids[~initial[ids]]
 if not len(active):return 0.
 return 100+sum(rates[s] for s in set(surf[ids]))*(math.pi*u[1]**2)**.85+200*lengths[active].sum()
cost=np.array([initial_cost(u,ids) for u,ids in zip(units,contained)])
order=np.lexsort((tie,-benefit/np.maximum(cost,1)))
path=ROOT/'09b_corrected_catboost_lightgbm_net.csv'
with path.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows[j] for j in order)
assert Counter(map(tuple,csv.reader(path.open())))==Counter(map(tuple,rows))
rep=validate(path)
rep.update(surface_formula='100 + surface_rate * area**0.85 + 200 * unreplaced_length',original_09b_order_reproduced=True,identical_circle_multiset_as_original_09b=True,original_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),benefit_source='Published original 09b audit; unchanged aggregate benefits and fallback contributions',official_public_score='Pending')
path.with_suffix('.validation.json').write_text(json.dumps(rep,indent=2))
pd.DataFrame({'rank':np.arange(1,len(order)+1),'pipe_id':tie[order],'source_09b_rank':order+1,'circle_benefit':benefit[order],'corrected_initial_cost':cost[order],'priority':(benefit/np.maximum(cost,1))[order]}).to_csv(ROOT/'09b_corrected.audit.csv',index=False)
print(json.dumps({k:v for k,v in rep.items() if k!='annual'},indent=2))
