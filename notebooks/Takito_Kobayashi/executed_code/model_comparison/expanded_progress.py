import json,sqlite3
from pathlib import Path
OUT=Path(__file__).resolve().parent
for model in ['xgboost','lightgbm','catboost']:
    for year in range(2023,2027):
        folder=OUT/model/str(year);cfg=json.loads((folder/'selected.json').read_text())
        path=folder/'exhaustive_addition_state.json'
        if not path.exists():print(model,year,'queued');continue
        state=json.loads(path.read_text())
        print(model,year,'complete' if cfg.get('expanded_post_parameter_feature_check') else 'parameter retune/final features' if cfg.get('expanded_parameter_search') else 'initial features','adoptions',len(state['steps']),'configs',len(state['cache']),'columns',len(state['baseline']),'feature_converged',state['complete'])
        if model=='catboost' and year in [2023,2025] and not cfg.get('pre_feature_parameter_search'):
            with sqlite3.connect(folder/'studies.sqlite') as db:
                counts=db.execute('SELECT state,COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id WHERE study_name=? GROUP BY state',('pre_feature_actual_common_rounds',)).fetchall()
            if counts:print('  pre-feature parameter trials:',counts)
