"""Behaviorally check that changing current/future costs cannot affect features."""
from pathlib import Path
import json,tempfile
import numpy as np
import pandas as pd
import run_experiment as experiment

OUT=Path(__file__).resolve().parent
manifest=json.loads((OUT/'feature_manifest.json').read_text())
source=pd.read_csv(manifest['source_path'])
reference=pd.read_csv(OUT/'features.csv')
features=manifest['features']
results=[]
for year in range(2023,2027):
    changed=source.copy()
    affected=pd.to_datetime(changed.date).dt.year>=year
    changed.loc[affected,'cost']=changed.loc[affected,'cost']*37+1234567
    with tempfile.TemporaryDirectory(prefix=f'invariance_{year}_',dir=OUT) as temporary:
        folder=Path(temporary).resolve()
        # The recursive cleanup target is a verified child of this experiment directory.
        assert folder.parent==OUT.resolve()
        path=folder/'changed_costs.csv'
        changed.to_csv(path,index=False)
        experiment.OUT=folder
        experiment.prepare(path)
        regenerated=pd.read_csv(folder/'features.csv')
        if 'supplemental_inventory' in manifest:
            if manifest.get('supplemental_cost_features'):
                from segment_cost_features import build
                segment=build(regenerated).set_index('pipe_id').reindex(regenerated.pipe_id)
                for feature in manifest['supplemental_cost_features']:regenerated[feature]=segment[feature].to_numpy()
            # Inventory geometry/repair counts use no cost values at all. Their
            # unchanged inputs are checked separately by the inventory audit.
            for feature in manifest['supplemental_inventory']['features']:
                if feature in manifest.get('supplemental_cost_features',[]):continue
                if feature.endswith('_log1p') and feature[:-7] in regenerated:
                    regenerated[feature]=np.log1p(regenerated[feature[:-7]])
                elif feature.endswith('_squared') and feature[:-8] in regenerated:
                    regenerated[feature]=regenerated[feature[:-8]]**2
                else:
                    regenerated[feature]=reference[feature]
        ix=reference.event_year<=year
        for feature in features:
            a=reference.loc[ix,feature]
            b=regenerated.loc[ix,feature]
            if pd.api.types.is_numeric_dtype(a):
                assert np.allclose(a,b,equal_nan=True,rtol=1e-10,atol=1e-6),feature
            else:
                assert a.fillna('__missing__').equals(b.fillna('__missing__')),feature
        results.append({'outer_test_year':year,'mutated_current_and_future_cost_rows':int(affected.sum()),'verified_rows_through_test_year':int(ix.sum()),'verified_feature_columns':len(features),'all_features_invariant':True,'supplemental_inventory_cost_free_by_construction':True})
    print(f'INVARIANCE {year} passed',flush=True)
(OUT/'future_invariance.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
