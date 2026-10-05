"""Expand training budget when a selected inner fit approaches the tree cap."""
import json,time
import numpy as np
import pandas as pd
import run_experiment as experiment

OUT=experiment.OUT
d=pd.read_csv(OUT/'features.csv')
results=json.loads((OUT/'iteration_cap_check.json').read_text()) if (OUT/'iteration_cap_check.json').exists() else []
for model in ['xgboost','catboost','lightgbm']:
    for year in range(2023,2027):
        path=OUT/model/str(year)/'selected.json'
        if not path.exists(): continue
        config=json.loads(path.read_text())
        old_cap=config.get('cap',1600)
        if max(v['iterations'] for v in config['inner_folds'])<.9*old_cap:
            continue
        experiment.THREADS=config.get('threads',3 if model=='catboost' else 5)
        experiment.CAP=6000
        start=time.time()
        value,metrics,importance=experiment.score_config(model,config['params'],config['features'],d[d.event_year<year],config['inner_years'])
        row={'model':model,'test_year':year,'old_cap':old_cap,'expanded_cap':6000,'before_RMSE':config['inner_mean_rmse'],'after_RMSE':value,'before_inner_folds':config['inner_folds'],'after_inner_folds':metrics,'seconds':time.time()-start}
        if value<config['inner_mean_rmse']:
            config.update(inner_mean_rmse=value,inner_folds=metrics,iterations=int(np.median([v['iterations'] for v in metrics])),cap=6000,cap_expansion=row)
            experiment.dump(path,config)
        results=[r for r in results if (r['model'],r['test_year'])!=(model,year)]
        results.append(row)
experiment.dump(OUT/'iteration_cap_check.json',results)
print(json.dumps(results,indent=2),flush=True)
