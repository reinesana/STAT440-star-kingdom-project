"""Independent source, history, split, prediction and metric audit."""
from pathlib import Path
import json,hashlib,math,sqlite3
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
manifest=json.loads((OUT/'feature_manifest.json').read_text())
source=Path(manifest['source_path'])
s=pd.read_csv(source)
d=pd.read_csv(OUT/'features.csv')
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['sha256'][source.name]
assert set(s.pipe_id)==set(d.pipe_id) and len(s)==len(d)==14107
assert np.allclose(d.cost,s.set_index('pipe_id').cost.reindex(d.pipe_id))
assert np.array_equal(pd.to_datetime(d.date).dt.year,d.event_year)
assert np.allclose(d.pipe_length,np.sqrt((s.gps_x1-s.gps_x2)**2+(s.gps_y1-s.gps_y2)**2))
assert not {'cost','pipe_id','date','time','lay_date'} & set(manifest['features'])
assert not any('actual' in c or 'predicted' in c for c in manifest['features'])
if 'supplemental_inventory' in manifest:
    inventory=manifest['supplemental_inventory']
    assert hashlib.sha256(Path(inventory['path']).read_bytes()).hexdigest()==inventory['sha256']
    assert json.loads((OUT/'inventory_audit.json').read_text())['source_hash_verified']
checks=0
# Independently reconstruct group statistics on one record per year/known surface.
for year in range(2020,2027):
    for surface in d.loc[d.event_year==year,'surface'].unique():
        row=d[(d.event_year==year)&(d.surface==surface)].iloc[0]
        history=s[pd.to_datetime(s.date).dt.year<year]
        group=history[history.surface==surface].cost
        n=len(group)
        mean=(math.fsum(group)+20*history.cost.mean())/(n+20)
        assert row.hist_surface_all_n==n
        assert np.isclose(row.hist_surface_all_mean,mean,equal_nan=True)
        threshold=history.cost.quantile(.99)
        rate=((group>=threshold).sum()+50*(history.cost>=threshold).mean())/(n+50)
        assert np.isclose(row.hist_surface_all_tail_rate,rate,equal_nan=True)
        # Independently use direct distances, rather than the generation KDTree.
        midx=(history.gps_x1+history.gps_x2)/2
        midy=(history.gps_y1+history.gps_y2)/2
        near=history[np.hypot(midx-row.midpoint_x,midy-row.midpoint_y)<=250].cost
        assert row.hist_radius250_n==len(near)
        mean=(math.fsum(near)+20*history.cost.mean())/(len(near)+20)
        assert np.isclose(row.hist_radius250_mean,mean,equal_nan=True)
        checks+=1
assert d.loc[d.event_year==2019,'hist_surface_all_n'].eq(0).all()
assert d.loc[d.event_year==2019,'hist_surface_all_mean'].isna().all()
results={'source_rows':len(s),'source_hash_verified':True,'history_samples_verified':checks,'history_sample_scope':'one row per surface and year 2020-2026; grouped cost and direct-distance 250m statistics','no_current_cost_predictor':True,'annual_counts':manifest['annual_counts']}
if (OUT/'future_invariance.json').exists():
    invariance=json.loads((OUT/'future_invariance.json').read_text())
    assert {r['outer_test_year'] for r in invariance}=={2023,2024,2025,2026}
    assert all(r['all_features_invariant'] and r['verified_feature_columns']==len(manifest['features']) for r in invariance)
    results['all_four_current_future_cost_mutation_checks_passed']=True
if (OUT/'predictions.csv').exists():
    p=pd.read_csv(OUT/'predictions.csv')
    tp=pd.read_csv(OUT/'training_predictions.csv')
    m=pd.read_csv(OUT/'fold_metrics.csv')
    a=pd.read_csv(OUT/'average_metrics.csv').set_index('model')
    frozen=json.loads((OUT/'frozen_selection_manifest.json').read_text())
    assert hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest()==frozen['feature_matrix_sha256']
    assert hashlib.sha256((OUT/'feature_manifest.json').read_bytes()).hexdigest()==frozen['feature_manifest_sha256']
    for relative,digest in frozen['geometry_helper_hashes'].items():assert hashlib.sha256((OUT.parent/relative).read_bytes()).hexdigest()==digest
    for relative,digest in frozen['selection_hashes'].items():
        assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==digest
    for name,digest in frozen['code_hashes'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    assert not p[['model','pipe_id']].duplicated().any()
    for model in ['xgboost','catboost','lightgbm']:
        rows=[]
        for fold,year in enumerate(range(2023,2027),1):
            z=p[(p.model==model)&(p.test_year==year)]
            expected=s[pd.to_datetime(s.date).dt.year==year]
            assert set(z.pipe_id)==set(expected.pipe_id)
            assert np.allclose(z.actual_cost,s.set_index('pipe_id').cost.reindex(z.pipe_id))
            assert np.isfinite(z.predicted_cost).all() and z.predicted_cost.ge(0).all()
            errors=(z.actual_cost-z.predicted_cost).tolist()
            mae=math.fsum(abs(e) for e in errors)/len(errors)
            rmse=math.sqrt(math.fsum(e*e for e in errors)/len(errors))
            r=m[(m.model==model)&(m.test_year==year)].iloc[0]
            assert np.isclose(r.MAE,mae) and np.isclose(r.RMSE,rmse)
            assert r.fold==fold and r.n_test==len(expected)
            assert r.n_train==int((pd.to_datetime(s.date).dt.year<year).sum())
            train_p=tp[(tp.model==model)&(tp.test_year==year)]
            expected_train=s[pd.to_datetime(s.date).dt.year<year]
            assert set(train_p.pipe_id)==set(expected_train.pipe_id) and len(train_p)==len(expected_train)
            assert np.allclose(train_p.actual_cost,s.set_index('pipe_id').cost.reindex(train_p.pipe_id))
            train_e=(train_p.actual_cost-train_p.predicted_cost).tolist()
            train_mae=math.fsum(abs(e) for e in train_e)/len(train_e)
            train_rmse=math.sqrt(math.fsum(e*e for e in train_e)/len(train_e))
            assert np.isclose(r.train_MAE,train_mae) and np.isclose(r.train_RMSE,train_rmse)
            config=json.loads((OUT/model/str(year)/'selected.json').read_text())
            assert max(config['inner_years'])<year
            assert any(c.startswith('hist_') and not c.endswith('_n') for c in config['features'])
            assert all(v['year']<year for v in config['inner_folds'])
            assert 'joint_final_selection' in config
            assert all(v['iterations']==config['iterations'] for v in config['inner_folds'])
            with sqlite3.connect(OUT/model/str(year)/'studies.sqlite') as db:
                counts=dict(db.execute('SELECT study_name,COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id WHERE state=? GROUP BY study_name',('COMPLETE',)).fetchall())
            assert counts['joint']>=80 and counts['individual_features']>=40 and counts['polish']>=100
            rows.append((mae,rmse))
        assert np.isclose(a.loc[model,'MAE'],sum(v[0] for v in rows)/4)
        assert np.isclose(a.loc[model,'RMSE'],sum(v[1] for v in rows)/4)
    results.update(prediction_rows=len(p),training_prediction_rows=len(tp),all_12_fold_metrics_recomputed=True,all_12_training_metrics_recomputed=True,macro_averages_verified=True,outer_test_sets_identical=True,all_selection_hashes_frozen_before_test=True,minimum_220_completed_trials_per_model_fold_verified=True)
(OUT/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
