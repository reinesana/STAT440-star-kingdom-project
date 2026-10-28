"""Portable inference from frozen base forecasts; no base-model fitting.

python notebooks/ensemble_for_cost_prediction/latest_conditional_stack/predict.py
python .../predict.py --input own_forecasts.csv --output my_predictions.csv
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def predict(frame, config):
    columns = config['base_columns']
    required = ['pipe_id', 'material', 'surface', *columns]
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.pipe_id.isna().any() or not frame.pipe_id.is_unique:
        raise ValueError('pipe_id must be non-null and unique')
    x = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Base forecasts must be finite, nonnegative dollar amounts')
    model = config['parameters']
    global_weights = np.asarray(model['global_weights'])
    weights = np.tile(global_weights,(len(frame),1))
    # This separator is part of the prepared feature contract. Missing categories
    # have no matching group and therefore retain the global weights.
    categories = frame.material.astype('string')+' | '+frame.surface.astype('string')
    for key,group in model['groups'].items():
        blend = group['n']/(group['n']+model['tau'])
        weights[categories.eq(key).fillna(False).to_numpy(dtype=bool)] = (
            blend*np.asarray(group['local_weights'])+(1-blend)*global_weights)
    if np.max(abs(weights.sum(axis=1)-1),initial=0)>1e-12 or weights.min(initial=0)<0:
        raise ValueError('Invalid convex model weights')
    return pd.DataFrame({'pipe_id':frame.pipe_id.to_numpy(),
                         'predicted_repair_cost':np.sum(x*weights,axis=1)})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'evidence/base_forecasts_2027.csv.gz')
    parser.add_argument('--config',type=Path,default=HERE/'model_2027.json')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    frame = pd.read_csv(args.input)
    result = predict(frame,config)
    if args.output:
        result.to_csv(args.output,index=False)
        print(f'Wrote {len(result)} conditional repair-cost forecasts to {args.output}')
    else:
        expected = pd.read_csv(HERE/'predictions_2027.csv').set_index('pipe_id')
        assert set(result.pipe_id)==set(expected.index)
        np.testing.assert_allclose(result.predicted_repair_cost,
            expected.loc[result.pipe_id,'predicted_repair_cost'],rtol=1e-12,atol=1e-7)
        print(f'Reproduced all {len(result)} frozen 2027 forecasts; no files modified.')

if __name__=='__main__': main()
