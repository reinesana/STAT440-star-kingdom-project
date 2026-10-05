"""Repeat every individual addition, deletion and high-correlation replacement."""
import argparse,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import run_experiment as e

def main(model,year,threads):
    e.THREADS=threads;folder=e.OUT/model/str(year)
    cfg=json.loads((folder/'selected.json').read_text())
    manifest=json.loads((e.OUT/'feature_manifest.json').read_text());pool=manifest['features']
    path=folder/'exhaustive_addition_state.json'
    state=json.loads(path.read_text()) if path.exists() else {'baseline':cfg['features'],'steps':[],'cache':{},'complete':False}
    fingerprint=hashlib.sha256(json.dumps([cfg['params'],cfg['iterations'],pool,threads],sort_keys=True).encode()).hexdigest()
    inputs={'cost_source':manifest['sha256'],'inventory':manifest.get('supplemental_inventory',{}).get('sha256')}
    assert state.get('input_fingerprints',inputs)==inputs
    state['input_fingerprints']=inputs
    if state['complete'] and cfg.get('exhaustive_expansion') and sorted(cfg['features'])==sorted(state['baseline']) and state['fingerprint']==fingerprint:return
    assert state.get('fingerprint',fingerprint)==fingerprint;state['fingerprint']=fingerprint
    d=pd.read_csv(e.OUT/'features.csv');d=d[d.event_year<year]
    numeric=d[pool].select_dtypes('number');corr=numeric.corr(method='spearman')
    pairs=[]
    for i,c in enumerate(corr):
        for other in corr.columns[i+1:]:
            value=corr.loc[c,other]
            if abs(value)>=.98:pairs.append({'feature':c,'other':other,'spearman':float(value)})
    for category in [c for c in ['material','surface','material_surface'] if c in pool]:
        labels=d[category];levels=labels.nunique()
        for col in numeric:
            values=d[col]
            if values.isna().any() or values.nunique()!=levels:continue
            relation=pd.DataFrame({'category':labels,'value':values})
            if relation.groupby('category').value.nunique().max()==1 and relation.groupby('value').category.nunique().max()==1:
                pairs.append({'feature':category,'other':col,'spearman':None,'relation':'bijective categorical encoding; identical information'})
    pd.DataFrame(pairs).to_csv(folder/'expanded_correlations.csv',index=False)
    splits=[(d[d.event_year<y],d[d.event_year==y]) for y in cfg['inner_years']]
    def score(fs):
        fs=sorted(fs);key='|'.join(fs)
        if key in state['cache']:return state['cache'][key]
        def one_fold(pair):
            a,b=pair
            _,pred,_,_=e.fit(model,cfg['params'],a,b,fs,cfg['iterations'])
            return float(np.sqrt(np.mean((b.cost.to_numpy()-pred)**2)))
        if model=='catboost':
            with ThreadPoolExecutor(max_workers=2) as workers:values=list(workers.map(one_fold,splits))
        else:values=[one_fold(pair) for pair in splits]
        result={'rmse':float(np.mean(values)),'fold_rmse':values};state['cache'][key]=result
        if len(state['cache'])%10==0:e.dump(path,state)
        return result
    def eligible(fs):return bool(fs) and any(c.startswith('hist_') and not c.endswith('_n') for c in fs)
    while True:
        fs=sorted(state['baseline']);base=score(fs);best=(base['rmse'],fs,None);tested=0
        proposals=[('add',c,sorted(fs+[c])) for c in pool if c not in fs]
        proposals += [('remove',c,[x for x in fs if x!=c]) for c in fs]
        # Test related families too: complementary columns can help together
        # even when neither improves alone. Every adoption still resets baseline.
        families={f'inventory_{radius}m':[c for c in pool if c.startswith(f'inventory_{radius}m_')] for radius in [5,10,25,50,100,250]}
        families['inventory_distances']=[c for c in pool if c.startswith('inventory_') and 'distance' in c]
        families['planar_intersections']=[c for c in pool if c in ['inventory_intersection_count','inventory_proper_crossing_count','inventory_collinear_overlap_count','inventory_shared_endpoint_count']]
        families['log_historic_cost']=[c for c in pool if c.startswith('hist_') and c.endswith('_log1p')]
        families['categorical_identifiers']=[c for c in ['material','surface','material_surface'] if c in pool]
        families['material_encodings']=[c for c in pool if c.startswith('material_') and c!='material_surface']
        families['surface_encodings']=[c for c in pool if c.startswith('surface_')]
        families['segment_cost_history']=[c for c in pool if c.startswith('hist_segment')]
        for label,group in families.items():
            new=set(group)-set(fs)
            if len(new)>1:proposals.append(('add_family',label,sorted(set(fs)|set(group))))
            present=set(group)&set(fs)
            if len(present)>1:proposals.append(('remove_family',label,sorted(set(fs)-present)))
        for pair in pairs:
            x,y=pair['feature'],pair['other']
            if (x in fs)!=(y in fs):
                old,new=(x,y) if x in fs else (y,x)
                proposals.append(('replace',old+' -> '+new,sorted([c for c in fs if c!=old]+[new])))
        for action,label,candidate in proposals:
            if not eligible(candidate):continue
            result=score(candidate);tested+=1
            if result['rmse']<best[0]-1e-7:best=(result['rmse'],candidate,{'action':action,'feature':label,'before':base,'after':result})
            if tested%25==0:print(model,year,'step',len(state['steps']),'tested',tested,'/',len(proposals),'best',best[0],flush=True)
        if best[2] is None:break
        state['baseline']=best[1];state['steps'].append(best[2]);e.dump(path,state)
        print('ADOPT',model,year,best[2],flush=True)
    state['complete']=True;e.dump(path,state)
    cfg['before_exhaustive_expansion']={'features':cfg['features'],'inner_mean_rmse':cfg['inner_mean_rmse'],'iterations':cfg['iterations']}
    cfg['features']=state['baseline'];cfg['inner_mean_rmse']=score(cfg['features'])['rmse'];cfg['threads']=threads
    importance=np.zeros(len(cfg['features']))
    for details,(a,b),value in zip(cfg['inner_folds'],splits,score(cfg['features'])['fold_rmse']):
        fitted,pred,_,_=e.fit(model,cfg['params'],a,b,cfg['features'],cfg['iterations'])
        error=b.cost.to_numpy()-pred
        actual=float(np.sqrt(np.mean(error**2)))
        assert np.isclose(actual,value,rtol=1e-9,atol=1e-5)
        details.update(rmse=actual,mae=float(np.mean(abs(error))),iterations=cfg['iterations'])
        imp=np.asarray(fitted.feature_importances_,dtype=float);importance+=imp/(imp.sum() or 1)
    pd.DataFrame({'features':cfg['features'],'inner_importance':importance}).sort_values('inner_importance',ascending=False).to_csv(folder/'selected_importance.csv',index=False)
    cfg['exhaustive_expansion']={'candidate_count':len(pool),'steps':state['steps'],'configurations':len(state['cache']),'stopping_rule':'No single addition/deletion, >=.98 Spearman replacement, or predefined related-family addition/deletion improves fixed-parameter common-round inner mean RMSE.','families_checked':list(families)}
    e.dump(folder/'selected.json',cfg);print('COMPLETE',model,year,len(cfg['features']),cfg['inner_mean_rmse'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();main(a.model,a.year,a.threads)
