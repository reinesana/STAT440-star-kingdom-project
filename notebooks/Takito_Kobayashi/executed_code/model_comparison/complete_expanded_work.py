"""Finish remaining spatial candidates and evaluate only after all selections converge."""
import sys,subprocess,json,time,hashlib
import pandas as pd
import run_experiment as e
OUT=e.OUT
paths=[OUT/m/str(y)/'selected.json' for m in ['xgboost','lightgbm','catboost'] for y in range(2023,2027)]
def execute(script,*args):
    print('RUN',script,*args,flush=True)
    result=subprocess.run([sys.executable,str(OUT/script),*args])
    if result.returncode:raise RuntimeError(f'{script}: exit {result.returncode}')
if __name__=='__main__':
    if not json.loads((OUT/'feature_manifest.json').read_text()).get('remaining_spatial_stage_added'):
        while not all(json.loads(p.read_text()).get('expanded_post_parameter_feature_check') for p in paths):time.sleep(10)
    m=json.loads((OUT/'feature_manifest.json').read_text())
    if not m.get('remaining_spatial_stage_added'):
        d=pd.read_csv(OUT/'features.csv');extra=pd.read_csv(OUT/'extra_spatial_candidates.csv');assert extra.pipe_id.is_unique
        staged=json.loads((OUT/'staged_feature_details.json').read_text())
        added=[c for c in extra if c!='pipe_id' and c not in d];assert set(added)==set(staged['features'])
        merged=d.merge(extra[['pipe_id']+added],on='pipe_id',validate='one_to_one',sort=False);assert merged.pipe_id.equals(d.pipe_id)
        d.to_csv(OUT/'features_pool446.csv',index=False);merged.to_csv(OUT/'features.csv',index=False)
        m['features']=sorted(set(m['features'])|set(added));m['supplemental_inventory']['features']+=added;m['supplemental_cost_features']=staged['cost_features'];m['remaining_spatial_stage_added']=True;e.dump(OUT/'feature_manifest.json',m)
        m['history_features']=[c for c in m['features'] if c.startswith('hist_')];e.dump(OUT/'feature_manifest.json',m)
        for path in paths:
            cfg=json.loads(path.read_text());e.dump(path.parent/'selected_pool446.json',cfg)
            state_path=path.parent/'exhaustive_addition_state.json';state=json.loads(state_path.read_text());assert state['complete']
            e.dump(path.parent/'exhaustive_addition_state_pool446.json',state)
            initial_state=path.parent/'exhaustive_addition_state_before_retune.json'
            if initial_state.exists():e.dump(path.parent/'exhaustive_addition_state_before_retune_pool446.json',json.loads(initial_state.read_text()))
            state.update(complete=False,steps=[],baseline=cfg['features'],fingerprint=hashlib.sha256(json.dumps([cfg['params'],cfg['iterations'],m['features'],cfg['threads']],sort_keys=True).encode()).hexdigest());e.dump(state_path,state)
            cfg['expanded_parameter_search_pool446']=cfg.pop('expanded_parameter_search');cfg.pop('expanded_post_parameter_feature_check');cfg['supplemental_second_stage']=True;e.dump(path,cfg)
    execute('run_expanded_search.py')
    execute('refresh_final_inner.py')
    execute('feature_catalogue.py')
    execute('export_expanded_search.py')
    execute('verify_inventory.py')
    execute('verify_future_invariance.py')
    execute('run_experiment.py','--finalize')
    execute('verify_experiment.py')
    execute('summarize.py')
    e.dump(OUT/'expanded_work_complete.json',{'complete':True,'candidate_features':len(m['features']),'selection_scope':'training-only; sequential additions/deletions/replacements; shared-round parameter retuning','source':m['source_path']})
    print('ALL EXPANDED WORK COMPLETE',flush=True)
