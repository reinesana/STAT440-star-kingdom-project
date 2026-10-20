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
units=read(S/'09a_full_ensemble_net.csv'); ids=[inside(*u) for u in units]
ref=validate(S/'09a_full_ensemble_net.csv');assert abs(ref['construction_cost']-152944811.51)<.011
print('REFERENCE',ref['construction_cost'],flush=True)
def mec(points):
 best=None
 candidates=[(x,0.) for x in points]
 for x,y in itertools.combinations(points,2):c=(x+y)/2;candidates.append((c,float(np.linalg.norm(x-c))))
 for x,y,z in itertools.combinations(points,3):
  mat=2*np.array([y-x,z-x]);det=np.linalg.det(mat)
  if abs(det)<1e-9:continue
  c=x+np.linalg.solve(mat,np.array([np.sum((y-x)**2),np.sum((z-x)**2)]));candidates.append((c,float(np.linalg.norm(x-c))))
 for c,r in candidates:
  if (best is None or r<best[1]) and (np.linalg.norm(points-c,axis=1)<=r+1e-7).all():best=(c,r)
 c,r=best;return np.round(c,9),float(f'{r+0.000002:.9f}')
# Compare adjacent-in-priority units within 100 positions; preserve all pipes covered by both originals.
used=set();replacements={};audit=[]
for i,(c,r) in enumerate(units):
 if i in used:continue
 if i%5000==0:print('SEARCH',i,flush=True)
 if not (~initial[ids[i]]).any():continue
 best=None
 for j in range(i+1,min(i+101,len(units))):
  if j in used or not (~initial[ids[j]]).any():continue
  d=np.linalg.norm(c-units[j][0])
  if d>r+units[j][1]+15:continue
  required=np.union1d(ids[i],ids[j]); active=required[~initial[required]]
  if len(required)>12:continue
  # No free savings from charging overlapping replacement length twice.
  state=initial.copy();old=cents(*units[i],ids[i],state);state[ids[i]]=True;old+=cents(*units[j],ids[j],state)
  nc,nr=mec(np.vstack([a[required],b[required]]));ni=inside(nc,nr)
  new=cents(nc,nr,ni,initial)
  saving=old-new
  if saving>10000 and (best is None or saving>best[0]):best=(saving,j,nc,nr,ni)
 if best:
  saving,j,nc,nr,ni=best;used.add(i);used.add(j);replacements[i]=(nc,nr)
  audit.append(dict(first_rank=i+1,second_rank=j+1,initial_pair_saving=saving/100,radius=nr))
out=ROOT/'10a_09a_nearby_pairs.csv'
with out.open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n')
 for i,u in enumerate(units):
  if i in replacements:u=replacements[i]
  elif i in used:continue
  c,r=u;w.writerow([f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}'])
result=validate(out)
assert result['construction_cost']<ref['construction_cost'],'Candidate did not reduce full simulated cost'
pd.DataFrame(audit).to_csv(ROOT/'pair_merges.csv',index=False)
print(json.dumps(dict(merges=len(audit),reference=ref['construction_cost'],candidate=result['construction_cost'],saving=ref['construction_cost']-result['construction_cost'],completion_year=result['completion_year']),indent=2),flush=True)
