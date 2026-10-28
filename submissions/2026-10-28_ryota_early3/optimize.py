"""Independent, deterministic circle search. Requires only numpy and pandas."""
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from decimal import Decimal, ROUND_HALF_UP
import argparse, csv, hashlib, json, math, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data' if (ROOT/'data'/'pipes.csv').exists() else ROOT.parents[1]/'data'
RATES = {'grassland':3,'farmland':7,'swamp':12,'road':17,'structure':50,'water':32}
IRON = {'cast iron','wrought iron','gray iron'}
YEARS = range(2027,2052)

def cents(value):
    return int(Decimal(str(float(value))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)*100)

def enclosing(points):
    """Incremental minimum enclosing disk; all 2/3-point boundary disks considered."""
    points = np.asarray(points)
    c, r = points[0].copy(), 0.
    for i,p in enumerate(points):
        if np.linalg.norm(p-c) <= r+1e-9: continue
        c,r=p.copy(),0.
        for j,q in enumerate(points[:i]):
            if np.linalg.norm(q-c) <= r+1e-9: continue
            c,r=(p+q)/2,np.linalg.norm(p-q)/2
            for s in points[:j]:
                if np.linalg.norm(s-c) <= r+1e-9: continue
                # A robust small-set fallback enumerates every possible boundary.
                prefix = np.vstack((points[:j+1],p,s))
                best=(None,float('inf'))
                for u,v in combinations(prefix,2):
                    cc=(u+v)/2; rr=np.linalg.norm(u-v)/2
                    if rr<best[1] and np.all(np.linalg.norm(prefix-cc,axis=1)<=rr+1e-8): best=cc,rr
                for u,v,w in combinations(prefix,3):
                    vv,ww=v-u,w-u
                    det=2*(vv[0]*ww[1]-vv[1]*ww[0])
                    if abs(det)<1e-10: continue
                    v2,w2=vv@vv,ww@ww
                    cc=u+np.array([ww[1]*v2-vv[1]*w2,vv[0]*w2-ww[0]*v2])/det
                    rr=np.linalg.norm(cc-u)
                    if rr<best[1] and np.all(np.linalg.norm(prefix-cc,axis=1)<=rr+1e-8): best=cc,rr
                c,r=best
    return c, float(np.max(np.linalg.norm(points-c,axis=1)))+1e-6

class Grid:
    def __init__(self,xy,cell=100):
        self.xy=xy; self.cell=cell; self.cells=defaultdict(list)
        for i,k in enumerate(np.floor(xy/cell).astype(int)): self.cells[tuple(k)].append(i)
    def query(self,c,r):
        lo=np.floor((c-r)/self.cell).astype(int); hi=np.floor((c+r)/self.cell).astype(int)
        ids=[i for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1) for i in self.cells.get((x,y),())]
        ids=np.asarray(ids,dtype=int)
        return ids[np.sum((self.xy[ids]-c)**2,axis=1)<=r*r]

class Problem:
    def __init__(self):
        self.p=pd.read_csv(DATA/'pipes.csv',dtype={'Pipe ID':str})
        self.hist=pd.read_csv(DATA/'train.csv',dtype={'Pipe ID':str})
        p,h=self.p,self.hist
        assert p['Pipe ID'].is_unique and h['Pipe ID'].is_unique
        assert set(h['Pipe ID'])<=set(p['Pipe ID']) and set(p.Surface)<=set(RATES)
        assert set(p.Material.dropna())<=IRON|{'copper','brass','polyurethane'}
        self.lay=pd.to_datetime(p['Lay date'],errors='raise')
        self.dates=pd.to_datetime(h.Date,errors='raise')
        assert self.dates.between('2019-01-01','2026-12-31').all()
        assert np.isfinite(h.Cost).all() and (h.Cost>=0).all()
        self.a=p[['GPS x1','GPS y1']].to_numpy(float);self.b=p[['GPS x2','GPS y2']].to_numpy(float)
        assert np.isfinite(self.a).all() and np.isfinite(self.b).all()
        self.mid=(self.a+self.b)/2;self.length=np.linalg.norm(self.b-self.a,axis=1)
        self.grid=Grid(self.mid);self.mat=p.Material.fillna('unknown').to_numpy()
        self.iron=p.Material.isin(IRON).to_numpy()
        self.start=(p.Material.eq('polyurethane')|p['Pipe ID'].isin(h['Pipe ID'])).to_numpy()
        self.target=np.flatnonzero(self.iron & ~self.start)
        self.surface=p.Surface.to_numpy()
        self.event=pd.to_datetime(p['Pipe ID'].map(h.set_index('Pipe ID').Date)).dt.year.to_numpy(float)
        self.age=2027-(self.lay.dt.year+(self.lay.dt.dayofyear-1)/365.25).to_numpy()
        self.age=np.where(np.isnan(self.age),np.nanmedian(self.age),self.age)
        self.bins=np.floor(self.age/5).astype(int)
    def unit(self,ids):
        c,r=enclosing(np.vstack((self.a[list(ids)],self.b[list(ids)])))
        # Round exactly as submission before computing containment/cost.
        c=np.round(c,9);r=float(f'{r:.9f}')
        near=self.grid.query(c,r)
        inside=near[(np.linalg.norm(self.a[near]-c,axis=1)<=r)&(np.linalg.norm(self.b[near]-c,axis=1)<=r)]
        assert set(ids)<=set(inside)
        surface=sum(RATES[s] for s in set(self.surface[inside]))*(math.pi*r*r)**.85
        return (c,r,inside,100+surface)
    def cost(self,u,state): return cents(u[3]+200*self.length[u[2][~state[u[2]]]].sum())

def risk_model(pr,cutoff=2026):
    """Shrunk material/5-year-age/surface annual hazards, with at-risk exposures."""
    n=len(pr.p); keys=list(zip(pr.mat,pr.bins,pr.surface))
    counts=defaultdict(lambda:[0,0]);mcounts=defaultdict(lambda:[0,0]);total=[0,0]
    # Bin age is recalculated at each historical year; no future outcomes used.
    for y in range(2019,cutoff+1):
        alive=np.isnan(pr.event)|(pr.event>=y)
        agebin=np.floor((pr.age+y-2027)/5).astype(int)
        for i in np.flatnonzero(alive & (pr.mat!='polyurethane')):
            hit=int(pr.event[i]==y); k=(pr.mat[i],int(agebin[i]),pr.surface[i])
            counts[k][0]+=hit;counts[k][1]+=1
            mcounts[pr.mat[i]][0]+=hit;mcounts[pr.mat[i]][1]+=1
            total[0]+=hit;total[1]+=1
    global_rate=total[0]/max(total[1],1)
    rates={m:(v[0]+100*global_rate)/(v[1]+100) for m,v in mcounts.items()}
    def predict(year):
        bins=np.floor((pr.age+year-2027)/5).astype(int)
        values=[]
        for m,b,s in zip(pr.mat,bins,pr.surface):
            k,e=counts.get((m,int(b),s),(0,0));values.append((k+100*rates.get(m,global_rate))/(e+100))
        out=np.asarray(values);out[pr.mat=='polyurethane']=0
        return out
    h=pr.hist[pr.dates.dt.year<=cutoff].copy()
    joined=h.merge(pr.p[['Pipe ID','Material','Surface']],on='Pipe ID',validate='one_to_one')
    means=joined.groupby(['Material','Surface'],dropna=False).Cost.mean().to_dict()
    surfmeans=joined.groupby('Surface').Cost.mean().to_dict()
    severity=np.array([means.get((m,s),surfmeans.get(s,float(h.Cost.mean()))) for m,s in zip(pr.mat,pr.surface)])
    return predict,severity

def validate_model(pr):
    records=[]
    for year in [2025,2026]:
        predict,_=risk_model(pr,year-1);prob=predict(year)
        mask=(np.isnan(pr.event)|(pr.event>=year))&(pr.mat!='polyurethane')
        y=(pr.event[mask]==year).astype(float);p=np.clip(prob[mask],1e-12,1-1e-12)
        records.append(dict(year=year,n=len(y),events=int(y.sum()),observed=float(y.mean()),predicted=float(p.mean()),
            log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),brier=float(np.mean((y-p)**2))))
    return records

def candidates(pr,max_size=6):
    """Local pairs, all anchor triples among four neighbors, and 4..6 prefixes."""
    singles={int(i):pr.unit([i]) for i in pr.target}
    base={i:pr.cost(u,pr.start) for i,u in singles.items()}
    targetset=set(singles);seen=set();out=[]
    for pos,i in enumerate(pr.target):
        near=pr.grid.query(pr.mid[i],125)
        near=np.array([j for j in near if j!=i and j in targetset],dtype=int)
        near=near[np.argsort(np.sum((pr.mid[near]-pr.mid[i])**2,axis=1),kind='stable')[:5]]
        groups=[(int(i),int(j)) for j in near]
        if max_size>=3:groups += [(int(i),int(j),int(k)) for j,k in combinations(near[:4],2)]
        groups += [tuple([int(i)]+list(map(int,near[:k-1]))) for k in range(4,min(max_size,len(near)+1)+1)]
        for group in groups:
            group=tuple(sorted(group))
            if group in seen:continue
            seen.add(group);u=pr.unit(group)
            covered=tuple(sorted(j for j in u[2] if j in targetset))
            # Pairwise limit is a limit on active iron pipes actually contained.
            if len(covered)>max_size:continue
            saving=sum(base[j] for j in covered)-pr.cost(u,pr.start)
            if saving>0:out.append((saving,covered,u))
        if pos%5000==0:print('candidate search',pos,'/',len(pr.target),flush=True)
    return singles,out

def choose(pr,singles,cands,max_size,predict,severity):
    used=set();units=[];sizes=[]
    for saving,ids,u in sorted(cands,key=lambda x:(-x[0],x[1])):
        if len(ids)>max_size or any(i in used for i in ids):continue
        used.update(ids);units.append(u);sizes.append(len(ids))
    for i,u in singles.items():
        if i not in used:units.append(u);sizes.append(1)
    risk=(1-(1-predict(2027))*(1-predict(2028)))*severity
    risk[pr.start]=0
    # Identical ordering rule for each geometry arm. Per-unit heuristic may overlap.
    order=sorted(range(len(units)),key=lambda j:(-risk[units[j][2]].sum()/max(pr.cost(units[j],pr.start)/100,1),
                     -float(np.max(pr.age[units[j][2]])),float(units[j][0][0]),float(units[j][0][1])))
    return [units[j] for j in order],dict(Counter(sizes))

def export(pr,units,name):
    path=ROOT/'outputs'/f'{name}.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        for c,r,*_ in units:w.writerow([f'({c[0]:.9f}, {c[1]:.9f})',f'{r:.9f}'])
    # Independent re-read, endpoint geometry and initial coverage audit.
    reread=[];covered=np.zeros(len(pr.p),bool)
    with path.open(newline='') as f:
        for row in csv.reader(f):
            assert len(row)==2
            c=np.array(list(map(float,row[0].strip('()').split(','))));r=float(row[1])
            assert np.isfinite(c).all() and math.isfinite(r) and r>=0
            ids=pr.grid.query(c,r)
            ids=ids[(np.linalg.norm(pr.a[ids]-c,axis=1)<=r)&(np.linalg.norm(pr.b[ids]-c,axis=1)<=r)]
            covered[ids]=True
            fixed=100+sum(RATES[s] for s in set(pr.surface[ids]))*(math.pi*r*r)**.85
            reread.append((c,r,ids,fixed))
    assert covered[pr.target].all()
    return reread,dict(rows=len(reread),uncovered_active_iron=int((~covered[pr.target]).sum()),
        uncovered_original_iron=int((pr.iron&~covered).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def execute(pr,units,first=None,leakcost=None):
    """Construction at start of year, leaks afterward; leaks become polyurethane."""
    n=len(pr.p);first=np.full(n,25) if first is None else first
    state=pr.start.copy();replacement=np.full(n,25);balance=construction=cursor=0;annual=[]
    for yi,year in enumerate(YEARS):
        balance+=1_000_000_000;state|=first<yi;spent=executed=0
        while cursor<len(units):
            u=units[cursor];ids=u[2];rep=ids[~state[ids]]
            if not len(rep):cursor+=1;continue
            cost=pr.cost(u,state)
            if cost>balance:break
            balance-=cost;spent+=cost;construction+=cost;state[rep]=True;replacement[rep]=yi
            cursor+=1;executed+=1
        annual.append(dict(year=year,units=executed,cost=spent/100,balance=balance/100,remaining=len(units)-cursor))
        # Continue all 25 years to count leaks in non-target pipes after completion.
    leaks=(first<replacement)&(first<25)&~pr.start
    repair=0 if leakcost is None else int(np.sum(leakcost[leaks]))
    complete=cursor==len(units)
    finish=next((a['year'] for a in annual if a['remaining']==0),None)
    return dict(feasible=complete,finish=finish,construction=construction/100,repair=repair/100,
        score=(25_000_000_000-construction-repair)/100 if complete else None,
        leaks=int(leaks.sum()),remaining_iron=int((pr.iron&~state).sum()),annual=annual)

def simulate(pr,arms,predict,severity,sims):
    # Held fixed across arms: common random numbers and severity bootstrap samples.
    hist=pr.hist.merge(pr.p[['Pipe ID','Material','Surface']],on='Pipe ID')
    pools=hist.groupby('Surface').Cost.apply(lambda x:np.array([cents(v) for v in x])).to_dict()
    out=[]
    for scale in [.5,1.,2.]:
        rng=np.random.default_rng(440)
        hazard=np.column_stack([np.clip(predict(y)*scale,0,.95) for y in YEARS])
        hazard[pr.start]=0
        for sim in range(sims):
            hit=rng.random(hazard.shape)<hazard
            first=np.where(hit.any(axis=1),hit.argmax(axis=1),25)
            costs=np.zeros(len(pr.p),dtype=np.int64)
            for s,pool in pools.items():
                ids=np.flatnonzero(pr.surface==s);costs[ids]=rng.choice(pool,len(ids))
            for name,units in arms.items():
                result=execute(pr,units,first,costs);result.pop('annual')
                out.append(dict(scenario=f'hazard_x{scale:g}',simulation=sim,arm=name,**result))
        print('simulation scenario complete',scale,flush=True)
    df=pd.DataFrame(out);df.to_csv(ROOT/'outputs/simulations.csv',index=False)
    summaries=[]
    for (scenario,arm),g in df.groupby(['scenario','arm']):
        base=df[(df.scenario==scenario)&(df.arm=='pairwise')].set_index('simulation')
        delta=g.set_index('simulation').score-base.score
        se=float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else None
        summaries.append(dict(scenario=scenario,arm=arm,simulations=len(g),feasible_fraction=float(g.feasible.mean()),
            mean_score=float(g.score.mean()),mean_construction=float(g.construction.mean()),mean_repair=float(g.repair.mean()),
            mean_leaks=float(g.leaks.mean()),latest_finish=int(g.finish.max()) if g.finish.notna().any() else None,
            paired_gain_vs_pairwise=float(delta.mean()),paired_standard_error=se,
            paired_mc_95_interval=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)] if se is not None else None))
    return summaries

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sims',type=int,default=30);args=ap.parse_args()
    assert args.sims>=2
    start=time.time();pr=Problem();predict,severity=risk_model(pr)
    print('data verified; active iron',len(pr.target),flush=True)
    validation=validate_model(pr)
    singles,cands=candidates(pr,6);arms={};audits={}
    for name,size in [('pairwise',2),('multi_pipe',6),('single_pipe',1)]:
        units,counts=choose(pr,singles,cands,size,predict,severity)
        arms[name],audit=export(pr,units,name)
        noleak=execute(pr,arms[name]);noleak.pop('annual')
        pd.DataFrame(execute(pr,arms[name])['annual']).to_csv(ROOT/'outputs'/f'{name}_annual.csv',index=False)
        audits[name]=dict(audit,selected_group_sizes=counts,no_future_leaks=noleak)
        print(name,audits[name],flush=True)
    results=simulate(pr,arms,predict,severity,args.sims)
    report=dict(data=dict(pipes=len(pr.p),events=len(pr.hist),active_iron=len(pr.target),original_iron=int(pr.iron.sum()),
        missing_material=int(pr.p.Material.isna().sum()),missing_lay_date=int(pr.lay.isna().sum()),
        history_start=str(pr.dates.min().date()),history_end=str(pr.dates.max().date())),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')},
        validation=validation,arms=audits,simulation_summary=results,seconds=time.time()-start)
    (ROOT/'outputs/report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(results).to_csv(ROOT/'outputs/comparison.csv',index=False)
    print('FINISHED',time.time()-start,flush=True)

if __name__=='__main__':main()
