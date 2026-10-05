"""Re-rank promising feature/parameter profiles by actual common-round CV."""
import argparse,json,sqlite3,time
import numpy as np
import pandas as pd
import run_experiment as experiment

OUT=experiment.OUT

def signature(profile):
    return json.dumps([sorted(profile['features']),profile['params']],sort_keys=True)

def profiles(folder,incumbent):
    current={'features':incumbent['features'],'params':incumbent['params'],'inner_folds':incumbent['inner_folds'],'screening_rmse':incumbent.get('selection_inner_mean_rmse',incumbent['inner_mean_rmse']),'origin':'incumbent','limit':incumbent.get('common_tree_count_refinement',{}).get('curve_limit',0)}
    found=[]
    with sqlite3.connect(folder/'studies.sqlite') as db:
        trials=db.execute('SELECT trials.trial_id,value,study_name FROM trials JOIN trial_values ON trials.trial_id=trial_values.trial_id JOIN studies ON trials.study_id=studies.study_id WHERE state=? ORDER BY value',('COMPLETE',)).fetchall()
        assert db.execute('SELECT COUNT(*) FROM trials WHERE state=?',('RUNNING',)).fetchone()[0]==0
        for ident,value,stage in trials:
            attrs={k:json.loads(v) for k,v in db.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?',(ident,)).fetchall()}
            if not {'features','model_params','inner_folds'}<=set(attrs): continue
            if not any(c.startswith('hist_') and not c.endswith('_n') for c in attrs['features']): continue
            found.append({'features':attrs['features'],'params':attrs['model_params'],'inner_folds':attrs['inner_folds'],'screening_rmse':value,'origin':f'{stage}:trial_id={ident}'})
    chosen=[current]
    seen={signature(current)}
    # Add two strong profiles, then one distinct feature set and one joint-search profile.
    for p in found:
        if signature(p) not in seen:
            chosen.append(p);seen.add(signature(p))
        if len(chosen)>=3: break
    for p in found:
        if sorted(p['features']) not in [sorted(c['features']) for c in chosen]:
            chosen.append(p);seen.add(signature(p));break
    for p in found:
        if p['origin'].startswith('joint:') and signature(p) not in seen:
            chosen.append(p);break
    return chosen[:5]

def evaluate(profile,model,inner_years,d):
    fs,p=profile['features'],profile['params']
    median=int(np.median([r['iterations'] for r in profile['inner_folds']]))
    limit=max(median,max(r['iterations'] for r in profile['inner_folds']))+80
    limit=max(limit,profile.get('limit',0))
    curves,folds=[],[]
    for year in inner_years:
        a,b=d[d.event_year<year],d[d.event_year==year]
        m,_,_,_=experiment.fit(model,p,a,b,fs,iterations=limit,collect_curve=model!='catboost')
        _,z,_,_=experiment.matrices(a,b,fs,model)
        if model=='catboost':
            curve=[float(np.sqrt(np.mean((b.cost.to_numpy()-np.maximum(pred,0))**2))) for pred in m.staged_predict(z,eval_period=1,thread_count=experiment.THREADS)]
        elif model=='xgboost':curve=m.evals_result()['validation_0']['rmse']
        else:curve=m.evals_result_['valid_0']['rmse']
        curves.append(np.asarray(curve));folds.append((a,b,m,z))
    average=np.mean(curves,axis=0)
    approximate=int(np.argmin(average))+1
    if model=='catboost':
        # CatBoost's prediction prefix is only a screen: CTR representation can differ.
        candidates={median,approximate}
        candidates.update(r['iterations'] for r in profile['inner_folds'])
        candidates.update(max(1,min(limit,int(round(approximate*f)))) for f in [.5,.8,1.2,1.5])
    else:
        candidates=set((np.argsort(average)[:30]+1).tolist())|{median}
        candidates.update(np.geomspace(1,limit,35).astype(int).tolist())
    rows=[]
    for n in sorted(candidates):
        metrics,importance=[],np.zeros(len(fs))
        for (a,b,m,z),year in zip(folds,inner_years):
            if model=='catboost':
                fitted,pred,_,_=experiment.fit(model,p,a,b,fs,iterations=n)
                imp=np.asarray(fitted.feature_importances_,dtype=float)
                importance+=imp/(imp.sum() or 1)
            elif model=='xgboost':pred=m.predict(z,iteration_range=(0,n))
            else:pred=m.predict(z,num_iteration=n)
            e=b.cost.to_numpy()-np.maximum(pred,0)
            metrics.append({'year':year,'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.abs(e).mean()),'iterations':n,'n_train':len(a),'n_valid':len(b)})
        value=float(np.mean([v['rmse'] for v in metrics]))
        rows.append({'iterations':n,'RMSE':value,'MAE':float(np.mean([v['mae'] for v in metrics])),'inner_folds':metrics,'importance':dict(zip(fs,importance.tolist()))})
    best=min(rows,key=lambda r:r['RMSE'])
    # For XGBoost/LightGBM, verify the selected prediction prefix by actual retraining.
    if model!='catboost':
        checked,importance=[],np.zeros(len(fs))
        for year in inner_years:
            a,b=d[d.event_year<year],d[d.event_year==year]
            m,pred,_,_=experiment.fit(model,p,a,b,fs,iterations=best['iterations'])
            e=b.cost.to_numpy()-pred
            imp=np.asarray(m.feature_importances_,dtype=float);importance+=imp/(imp.sum() or 1)
            checked.append({'year':year,'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.abs(e).mean()),'iterations':best['iterations'],'n_train':len(a),'n_valid':len(b)})
        checked_value=float(np.mean([v['rmse'] for v in checked]))
        assert np.isclose(checked_value,best['RMSE'],rtol=1e-7,atol=1e-4)
        best.update(RMSE=checked_value,inner_folds=checked,importance=dict(zip(fs,importance.tolist())))
    return best,rows,limit

def run(model,years):
    d=pd.read_csv(OUT/'features.csv')
    for outer in years:
        folder=OUT/model/str(outer)
        c=json.loads((folder/'selected.json').read_text())
        if 'joint_final_selection' in c:
            print(f'Already final: {model} {outer}',flush=True);continue
        start=time.time()
        candidates=profiles(folder,c)
        summaries,details=[],[]
        winner=None
        for index,profile in enumerate(candidates):
            best,rows,limit=evaluate(profile,model,c['inner_years'],d)
            summary={'profile':index,'origin':profile['origin'],'screening_RMSE':profile['screening_rmse'],'common_RMSE':best['RMSE'],'trees':best['iterations'],'feature_count':len(profile['features']),'features':profile['features'],'params':profile['params'],'tested_tree_counts':len(rows),'limit':limit}
            summaries.append(summary)
            details.extend({'profile':index,**{k:v for k,v in r.items() if k not in ['inner_folds','importance']}} for r in rows)
            if winner is None or best['RMSE']<winner[0]['RMSE']:winner=(best,profile,index)
            print(f'JOINT {model} {outer} profile {index+1}/{len(candidates)} actual_common_RMSE={best["RMSE"]:.2f} trees={best["iterations"]} elapsed={time.time()-start:.0f}s',flush=True)
        best,profile,index=winner
        result={'model':model,'test_year':outer,'profiles_evaluated':len(candidates),'winning_profile':index,'actual_common_inner_RMSE':best['RMSE'],'trees':best['iterations'],'actual_refits_for_catboost':model=='catboost','seconds':time.time()-start}
        c.update(selection_inner_mean_rmse=profile['screening_rmse'],features=profile['features'],params=profile['params'],inner_mean_rmse=best['RMSE'],inner_folds=best['inner_folds'],iterations=best['iterations'],threads=experiment.THREADS,joint_final_selection=result)
        experiment.dump(folder/'selected.json',c)
        experiment.dump(folder/'joint_profile_results.json',summaries)
        pd.DataFrame(details).to_csv(folder/'joint_tree_count_candidates.csv',index=False)
        pd.DataFrame([{'features':f,'inner_importance':v} for f,v in best['importance'].items()]).sort_values('inner_importance',ascending=False).to_csv(folder/'selected_importance.csv',index=False)
        print('JOINT FINAL '+json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['xgboost','catboost','lightgbm']);parser.add_argument('--outer-years',type=int,nargs='+',default=[2023,2024,2025,2026]);parser.add_argument('--threads',type=int,default=3)
    args=parser.parse_args();experiment.THREADS=args.threads
    run(args.model,args.outer_years)
