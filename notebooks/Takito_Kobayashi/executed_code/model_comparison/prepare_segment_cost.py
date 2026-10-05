import pandas as pd
import run_experiment as e
from segment_cost_features import build
if __name__=='__main__':
    d=pd.read_csv(e.OUT/'features.csv');cost=build(d);extra=pd.read_csv(e.OUT/'extra_spatial_candidates.csv')
    columns=[c for c in cost if c!='pipe_id']
    extra=extra.drop(columns=[c for c in columns if c in extra]).merge(cost,on='pipe_id',validate='one_to_one',sort=False)
    extra.to_csv(e.OUT/'extra_spatial_candidates.csv',index=False)
    e.dump(e.OUT/'staged_feature_details.json',{'features':[c for c in extra if c!='pipe_id'],'cost_features':columns,'history_rule':'Strictly prior calendar years; chosen cost CSV only; segment geometry is not observed repair location.'})
    print('Staged 6 geometry and',len(columns),'strict prior-year segment-cost features')
