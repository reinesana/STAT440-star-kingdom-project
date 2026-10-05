"""Exercise installed library APIs before the expensive searches."""
import json,time
import pandas as pd
from run_experiment import OUT,fit,score_config

d=pd.read_csv(OUT/'features.csv')
fs=['material','surface','midpoint_x','midpoint_y','pipe_length','years_used','event_year','hist_surface_all_mean','hist_radius250_mean']
params={
    'xgboost':{'learning_rate':.05,'max_depth':3,'min_child_weight':10,'reg_lambda':10},
    'catboost':{'learning_rate':.05,'depth':4,'l2_leaf_reg':10},
    'lightgbm':{'learning_rate':.05,'num_leaves':15,'min_child_samples':30,'reg_lambda':10},
}
for model,p in params.items():
    start=time.time()
    rmse,details,_=score_config(model,p,fs,d[d.event_year<2023],[2021,2022])
    _,prediction,n,_=fit(model,p,d[d.event_year<2022],d[d.event_year==2022],fs,50)
    print(json.dumps({'model':model,'smoke_inner_RMSE':rmse,'inner_folds':details,'fixed_fit_trees':n,'seconds':time.time()-start}),flush=True)
