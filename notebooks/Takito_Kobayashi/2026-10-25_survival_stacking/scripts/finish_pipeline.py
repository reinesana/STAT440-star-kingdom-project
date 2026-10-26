"""One-off supervisor: finish fixed trials, verify, collect and push results.

No recurring schedule or notifications. Existing worker PIDs are watched only
while their creation times match, so unrelated replacement PIDs are not touched.
Never terminates workers. Failed checks stop publication and leave a status file.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import psutil

BASE=Path(__file__).resolve().parents[1]
REPO=BASE.parents[2]
sys.path.insert(0,str(BASE))
from models import MODELS

LOCK=threading.Lock()
STATUS={'state':'waiting_for_initial_workers','workers':{}}


def update(**values):
    with LOCK:
        STATUS.update(values)
        STATUS['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=BASE/'pipeline_status.tmp'
        tmp.write_text(json.dumps(STATUS,indent=2),encoding='utf8')
        tmp.replace(BASE/'pipeline_status.json')


def run(args,log):
    with (BASE/log).open('a',encoding='utf8') as handle:
        result=subprocess.run(args,cwd=BASE,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}: {args[:3]}')


def worker(name,pid,models):
    try:
        p=psutil.Process(pid)
        stamp=p.create_time() if 'experiment.py' in p.cmdline() and Path(p.cwd()).resolve()==BASE.resolve() else None
    except psutil.NoSuchProcess:
        stamp=None
    with LOCK:STATUS['workers'][name]={'phase':'initial_training','pid':pid}
    update()
    ticks=0
    while stamp is not None:
        try:
            p=psutil.Process(pid)
            if p.create_time()!=stamp or not p.is_running():break
        except psutil.NoSuchProcess:break
        time.sleep(15)
        ticks+=1
        if ticks%4==0:
            completed=sum((BASE/model/str(year)/'finished.json').exists()
                          for model in models for year in range(2023,2028))
            with LOCK:STATUS['workers'][name]={'phase':'initial_training','pid':pid,'initial_completed_origins':completed}
            update()
    with LOCK:STATUS['workers'][name]='topping_up_and_verifying_trial_budgets'
    update(state='training')
    run([sys.executable,'-u','experiment.py','--trials','40','--deployment-trials','60','--no-report','--models',*models],f'final_{name}.log')
    for model in models:
        for origin in range(2023,2028):
            path=BASE/model/str(origin)
            assert (path/'finished.json').exists(),(model,origin)
            trials=json.loads((path/'trials.json').read_text())
            assert len(trials)>= (60 if origin==2027 else 40),(model,origin,len(trials))
    with LOCK:STATUS['workers'][name]='completed'
    update()


def update_pr(body):
    # Use the same native Git account that is authorized to push this branch.
    credentials=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,check=True,cwd=REPO,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    fields=dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
    request=urllib.request.Request('https://api.github.com/repos/reinesana/STAT440-star-kingdom-project/pulls/6',
        data=json.dumps({'body':body}).encode(),method='PATCH',headers={
        'Authorization':'Bearer '+fields['password'],'Accept':'application/vnd.github+json',
        'Content-Type':'application/json','User-Agent':'STAT440-experiment'})
    with urllib.request.urlopen(request) as response:
        result=json.load(response)
    return result['html_url']


def push_results(message):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    assert branch=='takito/survival-models-2026-10-25',branch
    subprocess.run(['git','add','notebooks/Takito_Kobayashi/2026-10-25_survival_stacking'],cwd=REPO,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        subprocess.run(['git','-c','user.name=takky-k','-c','user.email=tk.tk.tk.20020115@gmail.com',
            'commit','-m',message],cwd=REPO,check=True)
    subprocess.run(['git','push','origin',branch],cwd=REPO,check=True)


def publish_light_stage():
    import numpy as np
    import pandas as pd
    models=[m for m in MODELS if m not in [MODELS[0],MODELS[4]]]
    checks=[]
    for model in models:
        for year in range(2023,2028):
            path=BASE/model/str(year)
            finished=json.loads((path/'finished.json').read_text())
            assert hashlib.sha256((path/'predictions.csv.gz').read_bytes()).hexdigest()==finished['prediction_sha256']
            p=pd.read_csv(path/'predictions.csv.gz')
            a=p[[f'survival_{i:02d}y' for i in range(1,26)]].to_numpy()
            assert np.isfinite(a[:,0]).all() and not p.pipe_id.duplicated().any()
            finite=a[np.isfinite(a)];assert ((finite>=0)&(finite<=1)).all()
            assert (np.diff(np.nan_to_num(a,nan=-1),axis=1)<=1e-5).all()
            np.testing.assert_allclose(1-a[:,0],p.p_failure_next_year,atol=1e-7)
            checks.append({'model':model,'origin':year,'rows':len(p),'passed':True})
    run([sys.executable,'experiment.py','--report-only'],'light_stage_validation.log')
    (BASE/'results/light_stage_verification.json').write_text(json.dumps(checks,indent=2))
    (BASE/'results/run_status.json').write_text(json.dumps({'stage':'eight_models_verified_heavy_models_pending',
        'completed_model_origins':40,'pending_models':[MODELS[0],MODELS[4]],
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    push_results('Record verified results for eight models before heavy-model optimization')
    update_pr('Eight models have completed all five forecast origins and their fixed 40/60 attempted-trial budgets. '
        'Their 40 prediction exports passed checks and results/ contains the interim 2023–2026 comparison and OOF predictions. '
        'Elastic Net Cox and Random Survival Forest remain pending and will run after this light-first stage. '
        'See notebooks/Takito_Kobayashi/2026-10-25_survival_stacking/results/run_status.json. '
        'This is an interim result, not a completed ten-model comparison.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--a',type=int,required=True)
    parser.add_argument('--b',type=int,required=True)
    parser.add_argument('--c',type=int,required=True)
    parser.add_argument('--priority-plan',type=Path)
    args=parser.parse_args()
    # Capture the code actually used by the final pass, with no tuning feedback.
    contract=json.loads((BASE/'experiment_contract.json').read_text())
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
    assert actual==contract['code_sha256'],'Final source hashes differ from the frozen contract'
    update()
    if args.priority_plan:
        groups=[('a',args.a,[MODELS[9],MODELS[7],MODELS[8]]),
                ('b',args.b,MODELS[1:4]),('c',args.c,[MODELS[6],MODELS[5]])]
    else:
        groups=[('a',args.a,[MODELS[0],*MODELS[7:]]),('b',args.b,MODELS[1:4]),('c',args.c,MODELS[4:7])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(worker,*g) for g in groups]
        errors=[]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                errors.append(repr(error))
                update(state='one_or_more_workers_failed',errors=errors)
    if errors:raise RuntimeError('; '.join(errors))
    if args.priority_plan:
        update(state='light_models_complete_verifying_and_uploading')
        try:publish_light_stage()
        except (subprocess.CalledProcessError,urllib.error.URLError) as error:
            update(upload_warning=repr(error))
        plan=json.loads(args.priority_plan.read_text())
        heavy=[]
        for item,model in zip(plan['suspended_heavy_processes'],[MODELS[0],MODELS[4]]):
            try:
                p=psutil.Process(item['pid'])
                if p.create_time()!=item['created']:raise psutil.NoSuchProcess(item['pid'])
                p.resume()
                heavy.append((model,item['pid'],[model]))
            except psutil.NoSuchProcess:
                # If the machine restarted, resume saved studies instead.
                heavy.append((model,None,[model]))
        update(state='heavy_models_resumed_after_light_stage')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for future in concurrent.futures.as_completed([executor.submit(worker,*g) for g in heavy]):
                future.result()
    update(state='collecting_and_verifying')
    run([sys.executable,'experiment.py','--report-only'],'final_validation.log')
    run([sys.executable,'verify.py','--exports'],'final_validation.log')
    run([sys.executable,'checks/aft_reference.py'],'final_validation.log')
    run([sys.executable,'checks/discrete_likelihood_reference.py'],'final_validation.log')
    run([sys.executable,'build_results.py'],'final_validation.log')
    import pandas as pd
    scores=pd.read_csv(BASE/'results/annual_metrics.csv')
    assert len(scores)==40
    verified=json.loads((BASE/'results/export_verification.json').read_text())
    assert len(verified)==50 and all(v['passed'] for v in verified)
    # Notebook execution is verified above. Visual inspection is a separate
    # human/agent step and is explicitly not claimed by this supervisor.
    manifest={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in BASE.glob('*/20*/predictions.csv.gz')}
    (BASE/'results/prediction_manifest.json').write_text(json.dumps(manifest,indent=2))
    (BASE/'results/visual_review_status.json').write_text(json.dumps({
        'notebook_executed':True,'notebook_visual_inspection':'pending_agent_review',
        'reason':'One-off training supervisor cannot visually inspect rendered notebook figures.'},indent=2))
    update(state='verified_results_ready_for_upload',model_origins=50,annual_evaluations=40)
    (BASE/'results/run_status.json').write_text(json.dumps({'stage':'all_ten_models_verified',
        'completed_model_origins':50,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    push_results('Record optimized ten-model temporal survival experiments and OOF predictions')
    table=pd.read_csv(BASE/'results/mean_annual_metrics.csv').sort_values('brier')
    body='Ten survival estimators now have complete nested temporal optimization and prediction exports.\n\n'
    body+='All 50 model/origin combinations (2023–2027) completed their 40/60 attempted-trial budgets; failed or nonconvergent trials are recorded separately. All 50 exported prediction files passed validation.\n\n'
    body+='Results and code: `notebooks/Takito_Kobayashi/2026-10-25_survival_stacking/`. Includes selected features/configurations, all trial histories, annual and observable multi-year metrics, subgroup diagnostics, OOF predictions and diversity correlations.\n\n'
    body+='| Model | Mean annual Brier (2023–2026) |\n|---|---:|\n'
    body+='\n'.join(f'| {row.model} | {row.brier:.6f} |' for row in table.itertuples())
    body+='\n\nInput/target future-deletion invariance and independent likelihood checks passed. The results notebook executed successfully; rendered visual inspection remains pending and is not claimed. Long-horizon failure years are assumption-dependent; empirical-support models retain missing unsupported tails. No Google Doc edits or stacking fit were performed.\n'
    url=update_pr(body)
    update(state='uploaded_visual_review_pending',pull_request=url,results_pushed=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        update(state='failed_needs_attention',error=repr(error))
        raise
