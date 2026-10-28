"""Reproduce the fixed-circle CMS001/F168 official73.71M order. No training runs."""
from pathlib import Path
import json,hashlib,heapq,argparse
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from numba import njit
HERE=Path(__file__).resolve().parent
RATES={'grassland':3,'farmland':7,'swamp':12,'road':17,'structure':50,'water':32}
IRON=['cast iron','wrought iron','gray iron']

def geometry(p):
    a=p[['GPS x1','GPS y1']].to_numpy(float);bb=p[['GPS x2','GPS y2']].to_numpy(float)
    mid=(a+bb)/2;length=np.linalg.norm(bb-a,axis=1);iron=p.Material.isin(IRON).to_numpy();target=np.flatnonzero(iron)
    xy=np.round(mid[target],9);r=np.round(length[target]/2+1e-6,9);tree=cKDTree(mid)
    inside=[];base=[]
    for center,rad in zip(xy,r):
        ii=np.array(tree.query_ball_point(center,rad),int)
        ii=ii[(np.linalg.norm(a[ii]-center,axis=1)<=rad)&(np.linalg.norm(bb[ii]-center,axis=1)<=rad)]
        inside.append(ii);base.append(100+sum(RATES[s] for s in set(p.Surface.iloc[ii]))*(np.pi*rad**2)**.85)
    off=np.r_[0,np.cumsum([len(ii) for ii in inside])].astype(np.int64);flat=np.concatenate(inside).astype(np.int64)
    covered=np.zeros(len(p),bool);covered[flat]=True;assert covered[iron].all()
    lines=[]
    for center,rad in zip(xy,r):lines.append(f'"({center[0]:.9f}, {center[1]:.9f})",{rad:.9f}\n'.encode('utf-8'))
    return dict(a=a,b=bb,length=length,iron=iron,target=target,xy=xy,r=r,off=off,flat=flat,base=np.array(base),lines=lines)

def pipe_values(cfg,inputs,g):
    c=inputs['cost'];s=inputs['survival'];live=~inputs['initial'];H=cfg['horizon'];Fshort=1-s[:,H-1];Flong=1-s[:,24]
    risk=(1-cfg['long_weight'])*Fshort+cfg['long_weight']*Flong
    adjusted=c/(np.maximum(g['length'],.1)/np.median(g['length'][live]))**cfg['length_power']
    net=np.maximum(adjusted-cfg['net']*200*g['length'],0)
    value=np.maximum(risk,0)**cfg['risk_power']*net**cfg['cost_power'];value[~live]=0
    assert np.isfinite(value).all() and (value>=0).all()
    return value

@njit(cache=True)
def greedy_order(off,flat,rev_off,rev_flat,base,length,initial,value,gamma):
    m=len(base);poly=initial.copy();live=np.zeros(m,np.int64);cost=np.zeros(m);benefit=np.zeros(m)
    done=np.zeros(m,np.bool_);version=np.zeros(m,np.int64);heap=[(0.,0,0)];heap.pop();out=np.empty(m,np.int64);cursor=0
    for j in range(m):
        cost[j]=base[j]
        for k in range(off[j],off[j+1]):
            i=flat[k]
            if not poly[i]:live[j]+=1;cost[j]+=200*length[i];benefit[j]+=value[i]
        priority=max(benefit[j],0)/max(cost[j],1)**gamma if live[j] else 1e100
        heapq.heappush(heap,(-priority,j,version[j]))
    while heap:
        _,j,v=heapq.heappop(heap)
        if done[j] or v!=version[j]:continue
        done[j]=True;out[cursor]=j;cursor+=1
        for k in range(off[j],off[j+1]):
            i=flat[k]
            if poly[i]:continue
            poly[i]=True
            for t in range(rev_off[i],rev_off[i+1]):
                other=rev_flat[t]
                if done[other]:continue
                live[other]-=1;cost[other]-=200*length[i];benefit[other]-=value[i];version[other]+=1
                priority=max(benefit[other],0)/max(cost[other],1)**gamma if live[other] else 1e100
                heapq.heappush(heap,(-priority,other,version[other]))
    return out

def generate(cfg,inputs,g,rev):
    return greedy_order(g['off'],g['flat'],rev.indptr.astype(np.int64),rev.indices.astype(np.int64),g['base'],g['length'],inputs['initial'],pipe_values(cfg,inputs,g),cfg['gamma'])

def apply_local(order,moves,check_units=True):
    out=order.copy()
    for move in moves:
        if not move['accepted']:continue
        j=int(out[move['from_position']])
        if check_units:assert j==move['unit']
        out=np.insert(np.delete(out,move['from_position']),move['to_position'],j).astype(np.int64)
    return out

def apply_blocks(order,moves):
    out=order.copy()
    for t in moves:
        if not t['accepted']:continue
        start,size,dest=t['start'],t['size'],t['dest']
        if t['kind']=='reverse':out[start:start+size]=out[start:start+size][::-1]
        else:
            block=out[start:start+size];rest=np.r_[out[:start],out[start+size:]];dest=dest-size if dest>start else dest
            out=np.r_[rest[:dest],block,rest[dest:]].astype(np.int64)
    return out

def make_seed(kind,inputs,g,rev,recipe,check_units=True):
    if kind in ['joint','blocks']:
        order=generate(recipe['joint_config'],inputs,g,rev)
        if kind=='blocks':order=apply_blocks(apply_local(order,recipe['local_moves'],check_units),recipe['block_moves'])
        return order
    value=inputs['cost'].copy() if kind=='cost' else 1-inputs['survival'][:,0]
    value[inputs['initial']]=0
    return greedy_order(g['off'],g['flat'],rev.indptr.astype(np.int64),rev.indices.astype(np.int64),g['base'],g['length'],inputs['initial'],value,1.)

@njit(cache=True)
def own_calendar(order,off,flat,base,length,initial):
    years=np.full(len(order),25,np.int64);poly=initial.copy();budget=0;cursor=0
    for yr in range(25):
        budget+=1000000000
        while cursor<len(order):
            j=order[cursor];ll=0.;n=0
            for k in range(off[j],off[j+1]):
                i=flat[k]
                if not poly[i]:ll+=length[i];n+=1
            cents=int(np.floor((base[j]+200*ll)*100+.5)) if n else 0
            if cents>budget:break
            budget-=cents;years[j]=yr
            for k in range(off[j],off[j+1]):poly[flat[k]]=True
            cursor+=1
    return years

@njit(cache=True)
def refine(off,flat,rev_off,rev_flat,base,length,initial,values,gamma,deadline):
    m=len(base);poly=initial.copy();done=np.zeros(m,np.bool_);out=np.empty(m,np.int64);version=np.zeros(m,np.int64)
    live=np.zeros(m,np.int64);cost=np.zeros(m);benefit=np.zeros(m);cursor=0;budget=0;pending=-1
    for yr in range(25):
        budget+=1000000000;heap=[(0,0.,0,0)];heap.pop()
        for j in range(m):
            if done[j]:continue
            live[j]=0;cost[j]=base[j];benefit[j]=0.
            for k in range(off[j],off[j+1]):
                i=flat[k]
                if not poly[i]:live[j]+=1;cost[j]+=200*length[i];benefit[j]+=values[i,yr]
            key=max(deadline[j],yr) if live[j] else -2
            priority=max(benefit[j],0)/max(cost[j],1)**gamma if live[j] else 1e100
            heapq.heappush(heap,(key,-priority,j,version[j]))
        if pending>=0:heapq.heappush(heap,(-3,-1e200,pending,version[pending]))
        pending=-1
        while heap:
            _,_,j,v=heapq.heappop(heap)
            if done[j] or v!=version[j]:continue
            cents=int(np.floor(max(cost[j],0)*100+.5)) if live[j] else 0
            if cents>budget:pending=j;break
            budget-=cents;done[j]=True;out[cursor]=j;cursor+=1
            for k in range(off[j],off[j+1]):
                i=flat[k]
                if poly[i]:continue
                poly[i]=True
                for t in range(rev_off[i],rev_off[i+1]):
                    other=rev_flat[t]
                    if done[other]:continue
                    live[other]-=1;cost[other]-=200*length[i];benefit[other]-=values[i,yr];version[other]+=1
                    key=max(deadline[other],yr) if live[other] else -2
                    priority=max(benefit[other],0)/max(cost[other],1)**gamma if live[other] else 1e100
                    heapq.heappush(heap,(key,-priority,other,version[other]))
        if cursor==m:break
    for j in range(m):
        if not done[j]:out[cursor]=j;cursor+=1
    return out

def own_values(cfg,inputs,g):
    s=inputs['survival'];n=len(s);mass=np.maximum(0,-np.diff(np.column_stack([np.ones(n),s]),axis=1));live=~inputs['initial']
    c=inputs['cost']/(np.maximum(g['length'],.1)/np.median(g['length'][live]))**cfg.get('length_power',0.)
    net=np.maximum(c-cfg['net']*200*g['length'],0)**cfg['cost_power'];out=np.zeros((n,25))
    for yr in range(25):
        start=0 if cfg['signal']=='static' else yr
        short=mass[:,start:min(25,start+cfg['horizon'])].sum(1);long=mass[:,start:].sum(1)
        risk=(1-cfg.get('long_weight',0.))*short+cfg.get('long_weight',0.)*long
        out[:,yr]=np.maximum(risk,0)**cfg['risk_power']*net
    out[~live]=0;return out

def generate_self(cfg,inputs,g,rev,recipe,seed_cache=None,check_units=True):
    seed=make_seed(cfg['seed'],inputs,g,rev,recipe,check_units) if seed_cache is None else seed_cache[cfg['seed']]
    years=own_calendar(seed,g['off'],g['flat'],g['base'],g['length'],inputs['initial']);deadline=years//cfg['width']*cfg['width']
    return refine(g['off'],g['flat'],rev.indptr.astype(np.int64),rev.indices.astype(np.int64),g['base'],g['length'],inputs['initial'],own_values(cfg,inputs,g),cfg['gamma'],deadline)

def seed_order(inputs,g,rev,recipe,check=True):
    order=generate_self(recipe['self_config'],inputs,g,rev,recipe['self_seed_recipe'],check_units=check)
    return apply_blocks(apply_blocks(order,recipe['self_blocks']),recipe['final_blocks'])

def make_context(inputs,g,rev,recipe,check=True):
    seed=seed_order(inputs,g,rev,recipe,check);years=own_calendar(seed,g['off'],g['flat'],g['base'],g['length'],inputs['initial']);adj=(rev.T@rev).tocsr();adj.sort_indices()
    return years,adj

@njit(cache=True)
def terms(j,off,flat,base,length,poly,done,value,adj_off,adj_flat):
    length_sum=0.;benefit=0.;live=0
    for k in range(off[j],off[j+1]):
        i=flat[k]
        if not poly[i]:length_sum+=length[i];benefit+=value[i];live+=1
    if live==0:return 0.,0.,0.
    credit=0.
    for z in range(adj_off[j],adj_off[j+1]):
        other=adj_flat[z]
        if other==j or done[other]:continue
        any_active=False;covered=True
        for k in range(off[other],off[other+1]):
            i=flat[k]
            if poly[i]:continue
            any_active=True;found=False
            for q in range(off[j],off[j+1]):
                if flat[q]==i:found=True;break
            if not found:covered=False;break
        if any_active and covered:credit+=base[other]
    return base[j]+200*length_sum,benefit,credit

@njit(cache=True)
def credit_order(off,flat,rev_off,rev_flat,base,length,initial,value,gamma,deadline,adj_off,adj_flat,weight):
    m=len(base);poly=initial.copy();done=np.zeros(m,np.bool_);version=np.zeros(m,np.int64);marks=np.zeros(m,np.int64);affected=np.empty(m,np.int64);tick=0;out=np.empty(m,np.int64);cursor=0;budget=0;pending=-1
    for yr in range(25):
        budget+=1000000000;heap=[(0,0.,0,0)];heap.pop()
        for j in range(m):
            if done[j]:continue
            cost,benefit,credit=terms(j,off,flat,base,length,poly,done,value,adj_off,adj_flat);key=max(deadline[j],yr) if cost else -2;priority=(benefit+weight*credit)/max(cost,1.)**gamma if cost else np.inf
            heapq.heappush(heap,(key,-priority,j,version[j]))
        if pending>=0:heapq.heappush(heap,(-3,-np.inf,pending,version[pending]));pending=-1
        while heap:
            _,_,j,v=heapq.heappop(heap)
            if done[j] or v!=version[j]:continue
            cost,_,_=terms(j,off,flat,base,length,poly,done,value,adj_off,adj_flat);cents=int(np.floor(cost*100+.5))
            if cents>budget:pending=j;break
            budget-=cents;done[j]=True;out[cursor]=j;cursor+=1;tick+=1;n=0
            # Credit can change two hops away: another circle loses an outside
            # pipe and becomes redundant under a circle sharing its other pipes.
            for k in range(off[j],off[j+1]):
                i=flat[k]
                if poly[i]:continue
                poly[i]=True
                for z in range(rev_off[i],rev_off[i+1]):
                    other=rev_flat[z]
                    for q in range(adj_off[other],adj_off[other+1]):
                        target=adj_flat[q]
                        if not done[target] and marks[target]!=tick:marks[target]=tick;affected[n]=target;n+=1
            for q in range(n):
                other=affected[q];version[other]+=1;cost,benefit,credit=terms(other,off,flat,base,length,poly,done,value,adj_off,adj_flat);key=max(deadline[other],yr) if cost else -2;priority=(benefit+weight*credit)/max(cost,1.)**gamma if cost else np.inf
                heapq.heappush(heap,(key,-priority,other,version[other]))
        if cursor==m:break
    for j in range(m):
        if not done[j]:out[cursor]=j;cursor+=1
    return out

def generate_credit(cfg,inputs,g,rev,recipe,cached=None,check=True):
    years,adj=make_context(inputs,g,rev,recipe,check) if cached is None else cached
    if cfg['base']=='joint':value=own_values(recipe['self_config'],inputs,g)[:,0];gamma=recipe['self_config']['gamma']
    else:value=(1-inputs['survival'][:,0])*(inputs['cost']-200*g['length']);gamma=1.
    value[inputs['initial']]=0;deadline=years//cfg['width']*cfg['width']
    return credit_order(g['off'],g['flat'],rev.indptr.astype(np.int64),rev.indices.astype(np.int64),g['base'],g['length'],inputs['initial'],value,gamma,deadline,adj.indptr.astype(np.int64),adj.indices.astype(np.int64),cfg['weight'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=HERE.parents[2]/'data');parser.add_argument('--output-dir',type=Path,default=HERE);args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.data_dir/'pipes.csv');h=pd.read_csv(args.data_dir/'train.csv');dates=pd.to_datetime(h.Date);assert dates.max()<pd.Timestamp('2027-01-01') and dates.min()>=pd.Timestamp('2019-01-01');assert p['Pipe ID'].is_unique and h['Pipe ID'].is_unique
    g=geometry(p);initial=(p.Material.eq('polyurethane')|p['Pipe ID'].isin(h['Pipe ID'])).to_numpy();cost=pd.read_csv(HERE/'inputs/CMS001_predictions_2027.csv');raw=p['Pipe ID'].map(cost.set_index('pipe_id').predicted_repair_cost);missing=raw.isna().to_numpy()&~initial;median=float(cost.predicted_repair_cost.median());c=raw.fillna(median).to_numpy();c[initial]=0
    with np.load(HERE/'inputs/F168_curves_2027.npz') as z:ids=z['pipe_id'];small=z['survival'].astype(float)
    positions=pd.Index(p['Pipe ID']).get_indexer(ids);assert (positions>=0).all() and set(ids)==set(p.loc[~initial,'Pipe ID']);s=np.ones((len(p),25));s[positions]=small;assert np.isfinite(s).all() and s.min()>=0 and s.max()<=1 and (np.diff(s,axis=1)<=1e-7).all()
    inputs=dict(cost=c,survival=s,initial=initial);rev=csr_matrix((np.ones(len(g['flat'])),g['flat'],g['off']),shape=(len(g['base']),len(p))).T.tocsr();recipe=json.loads((HERE/'recipe.json').read_text(encoding='utf-8'));cfg=dict(base='joint',width=1,weight=.01);order=generate_credit(cfg,inputs,g,rev,recipe);assert np.array_equal(np.sort(order),np.arange(38455));payload=b''.join(g['lines'][j] for j in order);sha=hashlib.sha256(payload).hexdigest();assert sha=='5c092bfa45a3872bad57830e8be16f28a36bfb430f2021f2fb4d04ecad47d7e4';(out/'submission_2027.csv').write_bytes(payload)
    state=initial.copy();first_rank=np.full(len(p),-1,np.int64);pipeyear=np.full(len(p),-1,np.int64);first_unit=np.full(len(p),-1,np.int64);rows=[];budget=0;total=0;cursor=0
    for year in range(2027,2052):
        budget+=1000000000
        while cursor<len(order):
            j=int(order[cursor]);contained=g['flat'][g['off'][j]:g['off'][j+1]];active=contained[~state[contained]];work=int(np.floor((g['base'][j]+200*sum(float(g['length'][i]) for i in active))*100+.5)) if len(active) else 0
            if work>budget:break
            budget-=work;total+=work;first_rank[active]=cursor+1;pipeyear[active]=year;first_unit[active]=j;state[active]=True
            rows.append(dict(rank=cursor+1,circle_pipe_id=p['Pipe ID'].iloc[g['target'][j]],x=g['xy'][j,0],y=g['xy'][j,1],radius_m=g['r'][j],planned_year_no_future_leaks=year,newly_replaced_pipe_count=len(active),newly_replaced_iron_count=int(g['iron'][active].sum()),construction_cost_no_future_leaks=work/100,skipped_no_future_leaks=not len(active)));cursor+=1
        if cursor==len(order):break
    assert cursor==38455 and state[g['iron']].all()
    pd.DataFrame(rows).to_csv(out/'work_units_ranked.csv',index=False)
    target=np.flatnonzero(g['iron']);table=pd.DataFrame(dict(pipe_id=p['Pipe ID'].iloc[target].to_numpy(),already_replaced_by_2026=initial[target],first_cover_work_rank=first_rank[target],planned_replacement_year_no_future_leaks=pipeyear[target],CMS001_predicted_repair_cost=c[target],F168_failure_probability_1y=1-s[target,0],F168_failure_probability_2y=1-s[target,1],F168_failure_probability_25y=1-s[target,24],CMS001_median_fallback=missing[target]))
    active=table[~table.already_replaced_by_2026].sort_values(['first_cover_work_rank','pipe_id']).copy();active.insert(0,'repair_rank',np.arange(1,len(active)+1));active.to_csv(out/'active_iron_pipes_ranked.csv',index=False);table.sort_values(['already_replaced_by_2026','first_cover_work_rank','pipe_id']).to_csv(out/'all_original_iron_pipes.csv',index=False)
    manifest=dict(verification_pass=True,model_training_cutoff='2026-12-31',forecast_origin=2027,raw_training_records=len(h),pipe_inventory=len(p),original_iron=len(target),active_iron=len(active),already_replaced_original_iron=int(initial[target].sum()),work_units=38455,completion_year_no_future_leaks=year,construction_cost_no_future_leaks=total/100,all_iron_covered=True,completed_by_2051=True,source_only=['CMS001','F168'],inherited_12a_order_or_years=False,submission_sha256=sha,cost_fallback_ids=p.loc[missing,'Pipe ID'].tolist(),cost_fallback_value=median,official_public_M=73.71,official_private_M=None,official_result_source='user confirmed submission order plus screenshot2026-10-28; sourceCSV exact SHA256 matched',training_reused_not_rerun=True,input_sha256={str(path.relative_to(HERE)) if path.is_relative_to(HERE) else path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.data_dir/'pipes.csv',args.data_dir/'train.csv',HERE/'inputs/CMS001_predictions_2027.csv',HERE/'inputs/F168_curves_2027.npz']})
    (out/'verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
