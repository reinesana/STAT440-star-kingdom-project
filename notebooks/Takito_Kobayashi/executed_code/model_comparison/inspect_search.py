import sqlite3,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
for folder in sorted((OUT/'catboost').glob('20*')):
    with sqlite3.connect(folder/'studies.sqlite') as db:
        best=db.execute('SELECT trials.trial_id,value FROM trials JOIN trial_values ON trials.trial_id=trial_values.trial_id WHERE state=? ORDER BY value LIMIT 1',('COMPLETE',)).fetchone()
        attrs=dict(db.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?',(best[0],)).fetchall())
    print(folder.name, 'best inner RMSE',best[1],'parameters',json.loads(attrs['model_params']),'iterations',[d['iterations'] for d in json.loads(attrs['inner_folds'])],flush=True)
