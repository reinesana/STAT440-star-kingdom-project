"""Export inspectable candidate comparisons and sequential adoption paths."""
import json
import pandas as pd
from pathlib import Path
OUT=Path(__file__).resolve().parent
if __name__=='__main__':
    final_pool=len(json.loads((OUT/'feature_manifest.json').read_text())['features'])
    rows,steps,summary=[],[],[]
    for model in ['xgboost','lightgbm','catboost']:
        for year in range(2023,2027):
            folder=OUT/model/str(year);cfg=json.loads((folder/'selected.json').read_text())
            assert cfg.get('expanded_post_parameter_feature_check')
            phases=[('pool446_before_parameter_retune','exhaustive_addition_state_before_retune_pool446.json'),('pool446_final_parameters','exhaustive_addition_state_pool446.json'),(f'pool{final_pool}_before_parameter_retune','exhaustive_addition_state_before_retune.json'),(f'pool{final_pool}_final_parameters','exhaustive_addition_state.json')]
            phases.extend((f'pool{final_pool}_{p.stem}',p.name) for p in sorted(folder.glob('exhaustive_addition_state*round*.json')))
            for phase,name in phases:
                path=folder/name
                if not path.exists():continue
                state=json.loads(path.read_text());assert state['complete']
                for features,score in state['cache'].items():rows.append({'model':model,'test_year':year,'phase':phase,'features':features,'inner_mean_RMSE':score['rmse'],'first_inner_RMSE':score['fold_rmse'][0],'second_inner_RMSE':score['fold_rmse'][1]})
                for idx,step in enumerate(state['steps'],1):steps.append({'model':model,'test_year':year,'phase':phase,'step':idx,'action':step['action'],'feature':step['feature'],'before_RMSE':step['before']['rmse'],'after_RMSE':step['after']['rmse'],'improvement':step['before']['rmse']-step['after']['rmse']})
            summary.append({'model':model,'test_year':year,'feature_count':len(cfg['features']),'inventory_features':sum(c.startswith('inventory_') for c in cfg['features']),'historic_cost_features':sum(c.startswith('hist_') and not c.endswith('_n') for c in cfg['features']),'inner_RMSE':cfg['inner_mean_rmse'],'iterations':cfg['iterations'],'threads':cfg['threads']})
    pd.DataFrame(rows).to_csv(OUT/'expanded_candidate_comparisons.csv',index=False)
    pd.DataFrame(steps).to_csv(OUT/'expanded_adoption_paths.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'expanded_feature_summary.csv',index=False)
    print('Exported',len(rows),'feature configurations;',len(steps),'adopted changes')
