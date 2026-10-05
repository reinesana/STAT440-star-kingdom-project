"""Independently refit final configurations to verify cached RMSE and refresh MAE/importance."""
import json
import numpy as np
import pandas as pd
import run_experiment as e
if __name__=='__main__':
    d=pd.read_csv(e.OUT/'features.csv')
    for model in ['xgboost','lightgbm','catboost']:
        for year in range(2023,2027):
            folder=e.OUT/model/str(year);cfg=json.loads((folder/'selected.json').read_text())
            assert cfg.get('expanded_post_parameter_feature_check')
            e.THREADS=cfg['threads'];importance=np.zeros(len(cfg['features']));scores=[]
            for details,y in zip(cfg['inner_folds'],cfg['inner_years']):
                a,b=d[d.event_year<y],d[d.event_year==y]
                fitted,pred,_,_=e.fit(model,cfg['params'],a,b,cfg['features'],cfg['iterations'])
                error=b.cost.to_numpy()-pred;value=float(np.sqrt(np.mean(error**2)));scores.append(value)
                assert np.isclose(value,details['rmse'],rtol=1e-8,atol=1e-4)
                details.update(rmse=value,mae=float(np.mean(abs(error))),iterations=cfg['iterations'])
                imp=np.asarray(fitted.feature_importances_,dtype=float);importance+=imp/(imp.sum() or 1)
            assert np.isclose(np.mean(scores),cfg['inner_mean_rmse'],rtol=1e-8,atol=1e-4)
            cfg['expanded_inner_refit_verified']=True;e.dump(folder/'selected.json',cfg)
            pd.DataFrame({'features':cfg['features'],'inner_importance':importance}).sort_values('inner_importance',ascending=False).to_csv(folder/'selected_importance.csv',index=False)
            print('VERIFIED',model,year,flush=True)
