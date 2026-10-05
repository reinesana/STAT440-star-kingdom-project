"""Rebuild all features and refit the 12 saved single-model configurations."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('takito_reproduction'))
    parser.add_argument('--features', type=Path, help='Optional existing generated features.csv')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    root = args.output.resolve()
    if root.exists():
        raise SystemExit('Use a new output directory to avoid overwriting previous results.')
    root.mkdir(parents=True)
    shutil.copytree(package/'executed_code/model_comparison', root/'model_comparison')
    shutil.copytree(package/'executed_code/lightgbm_cost_model', root/'lightgbm_cost_model')
    data = root/'github-main/data'
    data.mkdir(parents=True)
    hashes = json.loads((package/'source_hashes.json').read_text())
    for name, expected in hashes.items():
        source = repo/'data'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f'Source changed: {name}'
        shutil.copy2(source, data/name)
    out = root/'model_comparison'
    def execute(name, *options):
        subprocess.run([sys.executable, str(out/name), *options], check=True)
    if args.features:
        shutil.copy2(args.features.resolve(), out/'features.csv')
    else:
        execute('run_experiment.py', '--prepare', '--source', str(data/'train_predict_cost.csv'))
        execute('expand_features.py')
        execute('prepare_extra_spatial.py')
        execute('prepare_segment_cost.py')
    sys.path.insert(0, str(out))
    import numpy as np
    import pandas as pd
    import run_experiment as e
    d = pd.read_csv(out/'features.csv')
    if not args.features:
        extra = pd.read_csv(out/'extra_spatial_candidates.csv')
        d = d.merge(extra, on='pipe_id', validate='one_to_one', sort=False)
    assert d.pipe_id.is_unique and len(d) == 14107
    records, metrics = [], []
    for model in ['xgboost', 'catboost', 'lightgbm']:
        for year in range(2023, 2027):
            cfg = json.loads((package/'selected_configs'/model/f'{year}.json').read_text())
            e.THREADS = cfg['threads']
            train = d[(d.event_year >= 2019) & (d.event_year < year)]
            test = d[d.event_year == year]
            _, pred, _, _ = e.fit(model, cfg['params'], train, test, cfg['features'], cfg['iterations'])
            errors = test.cost.to_numpy() - pred
            metrics.append(dict(model=model, test_year=year, MAE=float(np.mean(abs(errors))), RMSE=float(np.sqrt(np.mean(errors**2)))))
            records.append(pd.DataFrame(dict(pipe_id=test.pipe_id.to_numpy(), predicted_cost=pred, model=model, test_year=year)))
            print(model, year, metrics[-1], flush=True)
    predictions = pd.concat(records, ignore_index=True)
    for model, rows in predictions.groupby('model', sort=False):
        rows[['pipe_id', 'predicted_cost']].to_csv(root/f'output_{model}.csv', index=False)
        reference = pd.read_csv(package/f'output_{model}.csv')
        assert rows.pipe_id.tolist() == reference.pipe_id.tolist()
        assert np.allclose(rows.predicted_cost, reference.predicted_cost, rtol=1e-8, atol=1e-5), f'Prediction mismatch: {model}'
    pd.DataFrame(metrics).to_csv(root/'fold_metrics.csv', index=False)
    print('All saved predictions reproduced:', root)


if __name__ == '__main__':
    main()
