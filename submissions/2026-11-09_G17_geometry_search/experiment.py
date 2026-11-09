"""Prespecified G17 geometry experiment; no official future outcomes used."""
from pathlib import Path
import sys, json, time, argparse, itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'output/2026-11-02_regional_optimization/round4'))
import assess as e

BASE = e.HERE/'G17/G17_single_merge.csv'

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if isinstance(x,np.generic) else str(x)), encoding='utf8')

def protocol():
    save('protocol.json', dict(comparator='G17', source_sha256=e.sha(BASE),
        methods=['C01_convex_endpoint_exclusion', 'C02_new_bipartitions', 'C03_combined'],
        center_search='Convex minimum-radius subproblems with endpoint-exclusion halfplanes; beam width 4, depth 3, up to 8 offending pipes per state. Numerical local solver; global geometry optimum not claimed.',
        split_search='All unique bipartitions for 2..7 targets; 12 angular midpoint cuts for 8..25. Minimum circles and inherited center search per subset. Require lower nominal construction and positive expected fixed-cost saving under BOTH forecast families.',
        adoption='Every one of 24 setting mean gains positive, both forecast-family paired MC95 lower bounds positive, and all scenarios feasible in development AND fresh confirmation.',
        development_seed=202611091701, confirmation_seed=202611091702,
        development_draws=48, confirmation_draws=128, hazard_scales=[.5,1.,1.5],
        spatial_rho=[0.,.35], repair_cost_cv=[0.,1.5],
        changes_after_confirmation=False, official_submission=False,
        note='Synthetic forecast-conditional evaluation; Monte Carlo confidence is not evidence of real-world forecast accuracy.',
        algorithm_sha256=e.sha(Path(__file__))))

def families(pr,s,m):
    f,fm=e.fit_fine(pr,2026,.5,.85)
    fs=np.cumprod(1-np.column_stack([f(y) for y in range(2027,2052)]),axis=1)
    fs[pr.start]=0; fm[pr.start]=0
    return {'A_CMS001_F168':(s,m),'B_age_surface_historical_cost':(fs,fm)}

def exclusion_center(pr,required,excluded,old):
    """For each excluded endpoint v and target endpoint q:
    ||c-v||² > ||c-q||² iff 2(v-q).c < ||v||²-||q||².
    With fixed endpoint exclusions this is a convex epigraph problem.
    Coordinates are centered/scaled to avoid cancellation at map coordinates.
    """
    pts=np.unique(np.vstack([pr.a[required],pr.b[required]]),axis=0)
    origin=pts.mean(0); scale=max(np.ptp(pts,axis=0).max(),1.)
    q=(pts-origin)/scale; v=(np.asarray(excluded)-origin)/scale
    A=(2*(v[:,None,:]-q[None,:,:])).reshape(-1,2)
    margin=(.001+4*old[1]*1e-6)/scale**2
    b=(np.sum(v*v,axis=1)[:,None]-np.sum(q*q,axis=1)[None,:]-margin).ravel()
    c=(old[0]-origin)/scale
    x=np.r_[c,np.max(np.sum((q-c)**2,axis=1))]
    def constraints(z):
        return np.r_[z[2]-np.sum((z[:2]-q)**2,axis=1), b-A@z[:2]]
    def jac(z):
        return np.vstack([np.column_stack([-2*(z[:2]-q),np.ones(len(q))]),np.column_stack([-A,np.zeros(len(A))])])
    result=minimize(lambda z:z[2],x,jac=lambda z:np.array([0.,0.,1.]),
        constraints={'type':'ineq','fun':constraints,'jac':jac},bounds=[(None,None),(None,None),(0,None)],
        method='SLSQP',options={'maxiter':100,'ftol':1e-11})
    if not result.success or constraints(result.x).min() < -1e-7:return None
    return origin+scale*result.x[:2]

def improve_center(pr,sr,required,state,old):
    minimal=pr.unit(required)
    mandatory=sum(e.RATES[s] for s in set(pr.surface[required]))
    lower=100+mandatory*(np.pi*minimal[1]**2)**.85
    if old[3]-lower<.02:return old,0
    best=old; tested=0
    raw=sr.at([minimal[0]],required,state,required,old[3])
    if raw is not None:best=raw
    # Retain invalid minimum circles as intermediate search states. Excluding
    # one blocking endpoint can reveal the next blocking pipe.
    beams=[((),minimal),((),old)]
    seen=set()
    for depth in range(3):
        next_beams=[]
        for excludes,current in beams:
            bad=[int(i) for i in current[2] if i not in required and
                 (not state[i] or pr.surface[i] not in set(pr.surface[required]))]
            bad.sort(key=lambda i:(not state[i],e.RATES[pr.surface[i]]),reverse=True)
            for i in bad[:8]:
                for endpoint in (2*i,2*i+1):
                    key=tuple(sorted(set(excludes+(endpoint,))))
                    if key in seen:continue
                    seen.add(key)
                    points=[pr.a[k//2] if k%2==0 else pr.b[k//2] for k in key]
                    c=exclusion_center(pr,required,points,old);tested+=1
                    if c is None:continue
                    # Get geometry even when a different unintended pipe remains.
                    near=pr.grid.query(c,np.max(np.linalg.norm(np.vstack([pr.a[required],pr.b[required]])-c,axis=1))+2e-6)
                    loose=sr.at([c],required,np.ones(len(state),bool),near,old[3])
                    if loose is None:continue
                    next_beams.append((key,loose))
                    valid=sr.at([c],required,state,required,best[3])
                    if valid is not None:best=valid
        beams=sorted(next_beams,key=lambda x:x[1][3])[:4]
        if not beams:break
    return best,tested

def partitions(pr,rep):
    if len(rep)<=7:
        for mask in range(1,2**(len(rep)-1)):
            take=np.array([True]+[bool(mask & (1<<(i-1))) for i in range(1,len(rep))])
            # mask=0 is valid too; all-true is invalid. Yield below separately.
            if take.all():continue
            yield rep[take],rep[~take]
        yield rep[:1],rep[1:]
    else:
        seen=set()
        for angle in np.arange(12)*np.pi/12:
            order=rep[np.argsort(pr.mid[rep]@np.array([np.cos(angle),np.sin(angle)]),kind='stable')]
            for cut in range(1,len(rep)):
                key=tuple(sorted(order[:cut]))
                if key in seen:continue
                seen.add(key);yield order[:cut],order[cut:]

def split_search(pr,sr,rep,state,old,year,fs):
    if not 2<=len(rep)<=25:return [old],0
    ps=[np.column_stack([(~pr.start).astype(float),sv[:,:-1]])[:,year-2027] for sv,_ in fs.values()]
    expected_old=[(1-np.prod(1-p[rep]))*old[3] for p in ps]
    best=[old];best_fixed=old[3];tested=0;cache={}
    def unit(part):
        key=tuple(sorted(part))
        if key not in cache:
            if len(part)==1:
                # Every two-work split pays at least $100 for the other work.
                # Avoid the inherited $1e9 fallback bound for infeasible singles.
                cap=old[3]-100
                small=pr.unit(part)
                u=sr.at([small[0],old[0]],part,state,part,cap)
                v=sr.single(int(part[0]),state,part,u[3] if u is not None else cap)
                cache[key]=v if v is not None else u
            else:cache[key]=sr.group(part,state,part,deep=True)
        return cache[key]
    for a,b in partitions(pr,rep):
        # Rigorous geometry/surface lower bound before expensive search.
        lb=0.
        for part in (a,b):
            u=pr.unit(part)
            lb+=100+sum(e.RATES[s] for s in set(pr.surface[part]))*(np.pi*u[1]**2)**.85
        if lb>=best_fixed-1:continue
        ua,ub=unit(a),unit(b);tested+=1
        if ua is None or ub is None:continue
        cost=ua[3]+ub[3]
        if cost>=best_fixed-1:continue
        gains=[eo-((1-np.prod(1-p[a]))*ua[3]+(1-np.prod(1-p[b]))*ub[3]) for eo,p in zip(expected_old,ps)]
        if min(gains)<=1:continue
        best=[ua,ub];best_fixed=cost
    return best,tested

def generate():
    protocol()
    pr0,_,_,_,_,_,before,s,m=e.load();pr=e.FastProblem()
    fs=families(pr,s,m);units=e.read(pr,BASE)
    uy,py,active,total=e.shrink.trace(pr,units)
    sr=e.Search(pr);state=pr.start.copy();out={k:[] for k in ['C01','C02','C03']};moves=[]
    start=time.time();center_tested=split_tested=0
    for j,old in enumerate(units):
        rep=active[j]
        if not len(rep):
            for v in out.values():v.append(old)
            continue
        center,n=improve_center(pr,sr,rep,state,old);center_tested+=n
        split,n=split_search(pr,sr,rep,state,old,int(uy[j]),fs);split_tested+=n
        combined=split if sum(x[3] for x in split)<center[3] else [center]
        alternatives={'C01':[center],'C02':split,'C03':combined}
        for tag,group in alternatives.items():
            local=state.copy();cost=0
            for u in group:
                new=u[2][~local[u[2]]]
                assert set(new)<=set(rep)
                cost+=pr.cost(u,local);local[u[2]]=True
            assert local[rep].all()
            assert cost<=pr.cost(old,state)
            out[tag].extend(group)
            if cost<pr.cost(old,state):
                moves.append(dict(arm=tag,row=j,planned_year=int(uy[j]),targets=len(rep),
                    saving=(pr.cost(old,state)-cost)/100,old_radius=old[1],new_radii=[x[1] for x in group],
                    old_surfaces=sorted(set(pr.surface[old[2]])),new_surfaces=[sorted(set(pr.surface[x[2]])) for x in group]))
        state[rep]=True
        if j%2000==0:print('GENERATE',j,'moves',len(moves),'solves',center_tested,'splits',split_tested,'seconds',round(time.time()-start),flush=True)
    checks={}
    for tag,arm in out.items():
        path=OUT/(tag+'.csv');arm=e.write_read(pr,arm,path)
        ny,npy,na,nt=e.shrink.trace(pr,arm)
        assert nt<=total and np.all(npy<=py)
        checks[tag]=dict(sha256=e.sha(path),rows=len(arm),nominal_saving=(total-nt)/100,
            delayed_pipes=int(((py<2052)&(npy>py)).sum()),
            validation=e.validate(path,pr.p,pr.hist,pr.a,pr.b,pr.mid,pr.length,pr.iron))
    save('generation.json',dict(baseline_sha256=e.sha(BASE),center_solves=center_tested,split_tests=split_tested,
        elapsed_seconds=time.time()-start,candidates=checks))
    pd.DataFrame(moves).to_csv(OUT/'moves.csv',index=False)
    print(json.dumps(checks,default=str),flush=True)

def evaluate():
    e.OUT=OUT
    pr,_,_,_,_,_,_,s,m=e.load();fs=families(pr,s,m)
    paths={'G05':BASE,**{tag:OUT/(tag+'.csv') for tag in ['C01','C02','C03']}}
    arms={};audits={};rng=np.random.default_rng(202611091703)
    for tag,path in paths.items():
        units=e.read(pr,path);arms[tag]=e.packed(units)
        covered=np.zeros(len(pr.p),bool)
        for u in units:covered[u[2]]=True
        assert not (pr.iron&~covered).any()
        for random_future in [False,True]:
            first=rng.integers(0,26,len(pr.p)).astype(np.int64) if random_future else np.full(len(pr.p),25,np.int64)
            costs=rng.integers(0,10000000,len(pr.p)).astype(np.int64) if random_future else np.zeros(len(pr.p),np.int64)
            a=e.run(*arms[tag],pr.length,pr.start,pr.iron,first,costs);b=e.execute(pr,units,first,costs)
            assert abs(a[0]/100-b['construction'])<.01 and abs(a[1]/100-b['repair'])<.01
            assert a[4]==0 and a[3]>=0
        audits[tag]=dict(sha256=e.sha(path),uncovered_iron=0,independent_replay_agrees=True)
        print('AUDITED',tag,flush=True)
    save('replay_audit.json',audits)
    _,region=np.unique(np.floor(pr.mid/250).astype(int),axis=0,return_inverse=True)
    dev=e.bank(pr,arms,fs,region,202611091701,48,'development')
    selected=dev.loc[dev.gate_pass,'arm'].tolist()
    if selected:
        conf=e.bank(pr,{k:arms[k] for k in ['G05']+selected},fs,region,202611091702,128,'confirmation')
        accepted=conf.loc[conf.gate_pass,'arm'].tolist()
    else:accepted=[]
    save('decision.json',dict(comparator='G17',passed_development=selected,passed_confirmation=accepted,
        official_submission=False,public_scores=None))
    for tag,path in paths.items():assert e.sha(path)==audits[tag]['sha256']

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    action=argparse.ArgumentParser();action.add_argument('stage',choices=['generate','evaluate']);args=action.parse_args()
    {'generate':generate,'evaluate':evaluate}[args.stage]()
