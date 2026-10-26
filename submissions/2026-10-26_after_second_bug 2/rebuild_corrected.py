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

def raw_cost(u,ids,state):
 active=ids[~state[ids]]
 if not len(active):return 0.
 return 100+sum(rates[s] for s in set(surf[ids]))*(math.pi*u[1]**2)**.85+200*lengths[active].sum()
def export(name,us,order=None):
 path=ROOT/(name+'.csv')
 if order is None:order=range(len(us))
 with path.open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n')
  for j in order:
   c,r=us[j];w.writerow([f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}'])
 return path
results=[]
def check(path,method):
 rep=validate(path);rep['method']=method;rep['surface_formula']='surface_rate * (pi * radius**2)**0.85'
 path.with_suffix('.validation.json').write_text(json.dumps(rep,indent=2))
 results.append({k:rep[k] for k in ['file','units','construction_cost','completion_year','uncovered_iron','remaining_iron','sha256','method']})
 print(json.dumps(results[-1]),flush=True);return rep
# Reference checks: leave all original submitted files unchanged; use the new evaluator formula.
for file in ['09a_full_ensemble_net.csv','original_10a_09a_nearby_pairs.csv','original_12a_dynamic_net_benefit_per_cost.csv','original_13a_grouped_dynamic_net_benefit_per_cost.csv']:
 rep=validate(S/file)
 print('UNCHANGED REFERENCE',file,rep['construction_cost'],rep['completion_year'],flush=True)
base=units.copy();base_inside=contained;tie=audit.pipe_id.astype(str).to_numpy()
initial_cost=np.array([raw_cost(u,x,initial) for u,x in zip(base,base_inside)])
benefit=np.array([value[x].sum() for x in base_inside])
# 11b has no cost denominator, so its order should remain unchanged.
for name,pri in [('09a_corrected_reference',benefit/np.maximum(initial_cost,1)),('11b_corrected_net_benefit_only',benefit),('11c_corrected_net_benefit_sqrt_cost',benefit/np.sqrt(np.maximum(initial_cost,1)))]:
 order=np.lexsort((tie,-pri));path=export(name,base,order);check(path,name)
 if name.startswith('09a'):base_order=order
# Regenerate grouping from the corrected static 09a ordering; keep the original geometric search settings.
us=[base[j] for j in base_order];ids=[base_inside[j] for j in base_order]
used=set();replacements={};merge_audit=[]
# Do not introduce previously unconstrained missing-input pipes into a merged circle.
unsupported=set(missing[np.all(matrix==0,axis=0)]) if len(missing) else set()
for i,(c,r) in enumerate(us):
 if i in used:continue
 if i%5000==0:print('GROUP SEARCH',i,flush=True)
 if not (~initial[ids[i]]).any():continue
 best=None
 for j in range(i+1,min(i+101,len(us))):
  if j in used or not (~initial[ids[j]]).any():continue
  if np.linalg.norm(c-us[j][0])>r+us[j][1]+15:continue
  required=np.union1d(ids[i],ids[j])
  if len(required)>12:continue
  st=initial.copy();old=cents(*us[i],ids[i],st);st[ids[i]]=True;old+=cents(*us[j],ids[j],st)
  nc,nr=mec(np.vstack([a[required],b[required]]));ni=inside(nc,nr)
  if unsupported.intersection(ni):continue
  saving=old-cents(nc,nr,ni,initial)
  if saving>10000 and (best is None or saving>best[0]):best=(saving,j,nc,nr)
 if best:
  saving,j,nc,nr=best;used.update([i,j]);replacements[i]=(nc,nr)
  merge_audit.append({'first_rank':i+1,'second_rank':j+1,'initial_pair_saving':saving/100})
grouped=[];group_ties=[]
for i,u in enumerate(us):
 if i in replacements:u=replacements[i]
 elif i in used:continue
 grouped.append(u)
 if i in replacements:group_ties.append(min(p['Pipe ID'].iloc[k] for k in inside(*u) if iron[k]))
 else:group_ties.append(tie[base_order[i]])
group_path=export('10a_corrected_nearby_pairs',grouped);check(group_path,'Regroup corrected 09a; same pair-search settings')
pd.DataFrame(merge_audit).to_csv(ROOT/'pair_merges.audit.csv',index=False)
# Independent all-pipe checks on merged circles; unchanged circles retain exact geometry.
def brute(u):
 c,r=u;return (np.linalg.norm(a-c,axis=1)<=r)&(np.linalg.norm(b-c,axis=1)<=r)
k=0;seconds={x['first_rank']-1:x['second_rank']-1 for x in merge_audit}
for i,u in enumerate(us):
 if i in replacements:
  required=brute(u)|brute(us[seconds[i]])
  assert brute(grouped[k])[required].all()
 elif i in used:continue
 else:assert np.array_equal(u[0],grouped[k][0]) and u[1]==grouped[k][1]
 k+=1
assert k==len(grouped)

def dynamic(us,ties,name):
 ins=[inside(*u) for u in us];reverse=[[] for _ in range(len(p))]
 for j,x in enumerate(ins):
  for i in x:reverse[i].append(j)
 state=initial.copy();done=np.zeros(len(us),bool);versions=np.zeros(len(us),int);heap=[]
 def push(j):
  active=ins[j][~state[ins[j]]];cost=raw_cost(us[j],ins[j],state)
  pri=value[active].sum()/max(cost,1) if len(active) else float('inf')
  heapq.heappush(heap,(-pri,str(ties[j]),j,int(versions[j])))
 for j in range(len(us)):push(j)
 order=[]
 while heap:
  _,_,j,v=heapq.heappop(heap)
  if done[j] or v!=versions[j]:continue
  done[j]=True;order.append(j);active=ins[j][~state[ins[j]]];state[active]=True
  for k in {k for i in active for k in reverse[i] if not done[k]}:versions[k]+=1;push(k)
 assert len(order)==len(us)
 path=export(name,us,order)
 assert Counter(map(tuple,csv.reader(path.open())))==Counter((f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}') for c,r in us)
 check(path,'Dynamic remaining net benefit / corrected current construction cost')
 pd.DataFrame({'rank':np.arange(1,len(order)+1),'source_rank':np.array(order)+1,'tie_break_id':np.asarray(ties)[order]}).to_csv(ROOT/(name+'.audit.csv'),index=False)
dynamic(base,tie,'12a_corrected_dynamic')
dynamic(grouped,group_ties,'13a_corrected_grouped_dynamic')
pd.DataFrame(results).to_csv(ROOT/'comparison.csv',index=False)
(ROOT/'rebuild_checks.json').write_text(json.dumps({'corrected_formula':'100 + surface_rate * area**0.85 + 200 * unreplaced_length','merged_pairs':len(merge_audit),'independent_merged_coverage_check':True,'original_iron_count':int(iron.sum()),'max_09a_benefit_reconstruction_error':err,'prediction_models_retrained':False,'official_results':'Pending','source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.iterdir() if f.is_file() and f.suffix=='.csv'}},indent=2))
