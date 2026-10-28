"""Verify the published model, comparisons and frozen forecasts without refitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from predict import predict

HERE=Path(__file__).resolve().parent

def canonical_digest(path):
    data=path.read_bytes()
    if path.suffix not in {'.gz','.joblib'}: data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()

def main():
    c=json.loads((HERE/'model_2027.json').read_text(encoding='utf-8'))
    inputs=pd.read_csv(HERE/'evidence/base_forecasts_2027.csv.gz')
    expected=pd.read_csv(HERE/'predictions_2027.csv').set_index('pipe_id')
    assert len(inputs)==len(expected)==27240 and expected.index.is_unique
    for query in [inputs,inputs.iloc[::-1].copy()]:
        result=predict(query,c)
        assert result.pipe_id.tolist()==query.pipe_id.tolist()
        np.testing.assert_allclose(result.predicted_repair_cost,expected.loc[query.pipe_id,'predicted_repair_cost'],rtol=1e-12,atol=1e-7)
        x=query[c['base_columns']].to_numpy()
        assert (result.predicted_repair_cost.to_numpy()>=x.min(axis=1)-1e-7).all()
        assert (result.predicted_repair_cost.to_numpy()<=x.max(axis=1)+1e-7).all()
    unknown=inputs.iloc[:3].copy(); unknown['material']=['__new__',None,'__new__']
    result=predict(unknown,c)
    np.testing.assert_allclose(result.predicted_repair_cost,unknown[c['base_columns']].to_numpy()@c['parameters']['global_weights'],rtol=1e-12,atol=1e-7)
    for kind in ['duplicate','nonfinite','negative']:
        bad=inputs.iloc[:2].copy()
        if kind=='duplicate': bad.iloc[1,bad.columns.get_loc('pipe_id')]=bad.iloc[0].pipe_id
        elif kind=='nonfinite': bad.loc[bad.index[0],c['base_columns'][0]]=np.nan
        else: bad.loc[bad.index[0],c['base_columns'][0]]=-1.
        try: predict(bad,c)
        except ValueError: pass
        else: raise AssertionError('Invalid input was accepted: '+kind)
    annual=pd.read_csv(HERE/'evidence/annual_metrics_all_recipes.csv')
    mean=pd.read_csv(HERE/'evidence/mean_metrics_all_recipes.csv').set_index('model')
    for name,q in annual.groupby('model'):
        for col in ['RMSE','MAE','total_bias_pct']:
            assert abs(q[col].mean()-mean.loc[name,col])<1e-7
    selected=pd.read_csv(HERE/'evidence/selected_annual_metrics.csv')
    assert len(selected)==3 and selected.n.sum()==6766
    cat=annual[annual.model=='catboost'].set_index('year')
    for row in selected.itertuples():
        assert abs(row.RMSE_change_vs_catboost_pct-100*(row.RMSE/cat.loc[row.year,'RMSE']-1))<1e-10
    assert abs(selected.RMSE.mean()-112943.12407343672)<1e-7
    assert abs(selected.MAE.mean()-23375.49526099346)<1e-7
    priors=pd.read_csv(HERE/'evidence/prior_model_comparison.csv')
    assert len(priors)==24 and (priors.mean_rmse>selected.RMSE.mean()).all()
    np.testing.assert_allclose(priors.new_RMSE_change_pct,100*(selected.RMSE.mean()/priors.mean_rmse-1),rtol=1e-10,atol=1e-10)
    # Old and new benchmark tables agree for every shared model identity.
    aliases={'takito_catboost':'catboost','anthony_rf_log_mean':'log_rf',
             'rion_lgb_log_mean_refit':'log_lgbm','all_optimized':'previous_stack'}
    for old,new in aliases.items():
        ref=priors.set_index('model').loc[old]
        assert abs(ref.mean_rmse-mean.loc[new,'RMSE'])<1e-7
        assert abs(ref.mean_mae-mean.loc[new,'MAE'])<1e-7
    timing=pd.read_csv(HERE/'evidence/shared_doc_timing_ensemble_comparison.csv')
    timing_report=json.loads((HERE/'evidence/shared_doc_timing_comparison.json').read_text())
    timing_means=timing.groupby('model').Brier.mean()
    assert timing_report['common_pipe_years']==132121
    for name,q in timing.groupby('model'):
        assert q.n.sum()==132121 and q.year.tolist()==[2023,2024,2025,2026]
        assert abs(q.Brier.mean()-timing_report['mean_annual_metrics'][name]['Brier'])<1e-12
        assert abs(abs(q.count_bias_pct).mean()-timing_report['mean_absolute_annual_count_bias_pct'][name])<1e-10
    change=100*(timing_means['F168']/timing_means['shared_doc_equal_xgb_cat_lgb']-1)
    assert abs(change-timing_report['relative_mean_Brier_change_pct'])<1e-10 and round(change,2)==-9.47
    history=json.loads((HERE/'evidence/historical_models.json').read_text())
    for record in [*history,dict(origin=2027,training_years=c['training_years'],parameters=c['parameters'])]:
        assert max(record['training_years'])<record['origin']
        for trial in record['parameters']['selection']['trials']:
            assert all(y<record['origin'] for y in trial['validation_years'])
        for g in record['parameters']['groups'].values(): assert g['n']>=50
    metadata=json.loads((HERE.parent/'latest_model.json').read_text())
    assert metadata['model_id']==c['model_id'] and metadata['official_planning_score'] is None
    import joblib
    fitted=joblib.load(HERE/'fitted_model_2027.joblib')
    assert fitted['training_cutoff_exclusive']=='2027-01-01' and fitted['base_training_rows']==14113
    assert set(fitted['bases'])==set(c['base_columns'])
    for bundle in fitted['bases'].values():
        assert bundle['training_end']==2026 and bundle['training_rows']==14113
        assert not set(bundle['features'])&{'event_id','pipe_id','event_year','cost'}
    np.testing.assert_allclose(fitted['stack_config']['parameters']['global_weights'],c['parameters']['global_weights'],rtol=0,atol=1e-12)
    manifest=json.loads((HERE/'SHA256.json').read_text())
    for rel,sha in manifest['files'].items(): assert canonical_digest(HERE/rel)==sha,rel
    print('PASS: 3 actual fitted bases through 2026, 27,240 forecasts, reverse order, convex bounds, unseen/missing groups, invalid-input rejection, aggregate metrics, 24 prior recipes, chronological selection and all package hashes.')

if __name__=='__main__': main()
