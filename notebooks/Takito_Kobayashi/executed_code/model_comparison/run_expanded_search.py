"""Run four independent fold jobs concurrently, preserving per-job logs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys
from pathlib import Path
import json,time,hashlib

OUT=Path(__file__).resolve().parent
def job(model,year):
    folder=OUT/model/str(year)
    threads=json.loads((folder/'selected.json').read_text())['threads']
    with (OUT/model/str(year)/'exhaustive_additions.log').open('a',encoding='utf-8') as log:
        def execute(script,*extra_args):
            result=subprocess.run([sys.executable,str(OUT/script),'--model',model,'--year',str(year),'--threads',str(threads),*extra_args],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{model} {year}: exit {result.returncode}')
        cfg=json.loads((folder/'selected.json').read_text())
        if model=='catboost' and year in [2023,2025] and not cfg.get('pre_feature_parameter_search') and not cfg.get('supplemental_second_stage'):
            execute('retune_expanded.py','--lead');cfg=json.loads((folder/'selected.json').read_text())
            path=folder/'exhaustive_addition_state.json'
            if path.exists():
                pool=json.loads((OUT/'feature_manifest.json').read_text())['features'];state=json.loads(path.read_text())
                fingerprint=hashlib.sha256(json.dumps([cfg['params'],cfg['iterations'],pool,threads],sort_keys=True).encode()).hexdigest()
                if fingerprint!=state['fingerprint']:path.replace(folder/'exhaustive_addition_state_before_parameter_lead.json')
        if 'expanded_parameter_search' not in cfg:
            execute('exhaustive_additions.py');execute('retune_expanded.py')
        cfg=json.loads((folder/'selected.json').read_text())
        path=folder/'exhaustive_addition_state.json'
        state=json.loads(path.read_text())
        pool=json.loads((OUT/'feature_manifest.json').read_text())['features']
        fingerprint=hashlib.sha256(json.dumps([cfg['params'],cfg['iterations'],pool,threads],sort_keys=True).encode()).hexdigest()
        if fingerprint!=state['fingerprint']:
            path.replace(folder/'exhaustive_addition_state_before_retune.json')
        execute('exhaustive_additions.py')
        cfg=json.loads((folder/'selected.json').read_text())
        while cfg.get('supplemental_second_stage') and cfg['features']!=cfg['expanded_parameter_search'].get('features'):
            current_round=cfg.get('expanded_tuning_round',0)+1
            for state_name in ['exhaustive_addition_state.json','exhaustive_addition_state_before_retune.json']:
                source=folder/state_name
                if source.exists():(folder/f'{source.stem}_round{current_round}{source.suffix}').write_bytes(source.read_bytes())
            cfg['expanded_tuning_round']=current_round
            (folder/'selected.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
            execute('retune_expanded.py')
            cfg=json.loads((folder/'selected.json').read_text());state=json.loads(path.read_text())
            fingerprint=hashlib.sha256(json.dumps([cfg['params'],cfg['iterations'],pool,threads],sort_keys=True).encode()).hexdigest()
            if fingerprint!=state['fingerprint']:path.replace(folder/'exhaustive_addition_state_before_retune.json')
            execute('exhaustive_additions.py');cfg=json.loads((folder/'selected.json').read_text())
        cfg['expanded_post_parameter_feature_check']=True
        (folder/'selected.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    return model,year

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(job,m,y) for m in ['xgboost','lightgbm','catboost'] for y in range(2023,2027)]
        for result in as_completed(pending):print('FINISHED',result.result(),flush=True)
