"""Compact progress view of training-only searches (never reads test scores)."""
from pathlib import Path
import sqlite3,json
OUT=Path(__file__).resolve().parent
for model in ['xgboost','catboost','lightgbm']:
    records=[]
    for year in range(2023,2027):
        folder=OUT/model/str(year)
        if (folder/'selected.json').exists():
            c=json.loads((folder/'selected.json').read_text())
            with sqlite3.connect(folder/'studies.sqlite') as db:
                rows=db.execute('SELECT study_name, state, COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id GROUP BY study_name,state').fetchall()
            complete=sum(n for stage,state,n in rows if stage=='polish' and state=='COMPLETE')
            records.append(f'{year}: selected ({len(c["features"])} features; inner RMSE {c["inner_mean_rmse"]:,.0f}; final search {complete}/100)')
        elif (folder/'studies.sqlite').exists():
            with sqlite3.connect(folder/'studies.sqlite') as db:
                rows=db.execute('SELECT study_name, state, COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id GROUP BY study_name,state').fetchall()
            records.append(f'{year}: {rows}')
    print(model+': '+'; '.join(records),flush=True)
