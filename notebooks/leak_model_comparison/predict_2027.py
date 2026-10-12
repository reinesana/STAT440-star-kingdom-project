"""Train the agreed raw XGBoost/CatBoost/baseline-LightGBM ensemble for 2027.

Run from any directory: python predict_2027.py
Install: pip install numpy pandas scikit-learn xgboost catboost lightgbm joblib
Outputs go to deployment_2027/ beside this file. Historical first leaks remove
pipes from this model's risk set; excluded pipes get no probability, not zero.
Missing required inventory attributes are reported, never silently imputed.
All three model definitions are included here; no contributor scripts are imported.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
if (PROJECT / '.runtime_deps').exists():
    sys.path.insert(0, str(PROJECT / '.runtime_deps'))

import joblib
import numpy as np
import pandas as pd
import catboost
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = ROOT / 'deployment_2027'
DEFAULT_INPUT = PROJECT / 'data/leak_prediction_master.csv'
FEATURES = ['material', 'surface', 'pipe_length', 'pipe_age']
KEYS = ['pipe_id', 'year']
XGB_PARAMS = dict(objective='binary:logistic', eval_metric='logloss',
                  scale_pos_weight=10, max_depth=4, learning_rate=.10,
                  n_estimators=100, random_state=440, n_jobs=4)
CAT_PARAMS = dict(iterations=500, depth=6, learning_rate=.05,
                  loss_function='Logloss', random_seed=440, verbose=False,
                  allow_writing_files=False, thread_count=4)
LGB_PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=31,
                  max_depth=-1, min_child_samples=20, colsample_bytree=1.,
                  subsample=1., n_estimators=500, scale_pos_weight=1,
                  random_state=440, verbosity=-1, n_jobs=4)


def load_data(path):
    data = pd.read_csv(path, dtype={"pipe_id": "string"})
    required = [*KEYS, "Y", *FEATURES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Missing required values; use the shared cleaned master.")
    if data.pipe_id.str.strip().eq("").any() or data.duplicated(KEYS).any():
        raise ValueError("Blank identifiers or duplicate pipe-year records.")
    for column in ["year", "Y", "pipe_length", "pipe_age"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Non-finite values in {column}")
    if (data.year % 1 != 0).any() or not data.year.between(2019, 2026).all():
        raise ValueError("Expected integer historical years 2019-2026.")
    data["year"] = data.year.astype(int)
    if not data.Y.isin([0, 1]).all():
        raise ValueError("Y must be binary.")
    if (data.pipe_length < 0).any() or (data.pipe_age < 0).any():
        raise ValueError("Negative lengths or ages.")
    data["Y"] = data.Y.astype(int)
    for column in ["material", "surface"]:
        data[column] = data[column].astype(str)
    # First-leak data must not retain a pipe after its observed first event.
    events = data.loc[data.Y.eq(1), KEYS]
    if events.pipe_id.duplicated().any():
        raise ValueError("Multiple first-leak outcomes for a pipe.")
    last_year = data.groupby("pipe_id").year.max()
    if (events.pipe_id.map(last_year).to_numpy() > events.year.to_numpy()).any():
        raise ValueError("Risk set includes observations after first leak.")
    return data

def make_model():
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["material", "surface"]),
        ("numeric", "passthrough", ["pipe_length", "pipe_age"]),
    ])
    return Pipeline([("preprocessing", preprocessing),
                     ("model", XGBClassifier(**XGB_PARAMS))])

def fit_model(data):
    if data.empty or data.Y.nunique() != 2:
        raise ValueError("Training data must contain both outcome classes.")
    return make_model().fit(data[FEATURES], data.Y)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data(DEFAULT_INPUT)
    if set(data.year.unique()) != set(range(2019,2027)):
        raise ValueError('Expected complete 2019-2026 historical training years.')
    pipes_path, history_path = PROJECT/'data/pipes.csv', PROJECT/'data/train.csv'
    pipes = pd.read_csv(pipes_path, dtype={'Pipe ID':'string'})
    history = pd.read_csv(history_path, dtype={'Pipe ID':'string'})
    if pipes['Pipe ID'].isna().any() or pipes['Pipe ID'].duplicated().any():
        raise ValueError('Invalid inventory IDs.')
    history['event_date'] = pd.to_datetime(history['Date'], errors='coerce')
    if history['Pipe ID'].isna().any() or history.event_date.isna().any():
        raise ValueError('Missing historical IDs/dates: resolve before predicting.')
    if not history.event_date.dt.year.between(2019,2026).all():
        raise ValueError('Leak history outside the supported 2019-2026 window.')
    leaked = set(history['Pipe ID'])
    if not leaked.issubset(set(pipes['Pipe ID'])):
        raise ValueError('Leak history contains IDs missing from inventory.')
    inventory = pipes.rename(columns={'Pipe ID':'pipe_id','Material':'material','Surface':'surface'}).copy()
    inventory['lay_date'] = pd.to_datetime(inventory['Lay date'], errors='coerce')
    inventory['year'] = 2027
    inventory['pipe_age'] = (pd.Timestamp('2027-01-01')-inventory.lay_date).dt.days/365.25
    coords = inventory[['GPS x1','GPS y1','GPS x2','GPS y2']].apply(pd.to_numeric,errors='coerce')
    inventory['pipe_length'] = np.hypot(coords['GPS x2']-coords['GPS x1'],coords['GPS y2']-coords['GPS y1'])
    reasons = []
    for row in inventory.itertuples(index=False):
        reason = []
        if row.pipe_id in leaked:
            reason.append('historical_leak_replaced_outside_first_leak_scope')
        if pd.isna(row.lay_date):
            reason.append('missing_or_invalid_lay_date')
        elif row.lay_date > pd.Timestamp('2027-01-01'):
            reason.append('not_installed_at_2027_start')
        if pd.isna(row.material):
            reason.append('missing_material')
        if pd.isna(row.surface):
            reason.append('missing_surface')
        if not np.isfinite(row.pipe_length):
            reason.append('invalid_geometry')
        reasons.append(';'.join(reason))
    inventory['exclusion_reason'] = reasons
    eligible = inventory.loc[inventory.exclusion_reason.eq('')].copy().sort_values('pipe_id')
    for col in ['material','surface']:
        eligible[col] = eligible[col].astype(str)
        unknown = set(eligible[col])-set(data[col])
        if unknown:
            raise ValueError(f'Unseen {col} labels: {unknown}')
    # Verify reconstructed inventory features against the established master.
    last = data.loc[data.year.eq(2026)].set_index('pipe_id')
    common = eligible.set_index('pipe_id').index.intersection(last.index)
    check = eligible.set_index('pipe_id').loc[common]
    for col in ['material','surface']:
        np.testing.assert_array_equal(check[col],last.loc[common,col])
    np.testing.assert_allclose(check.pipe_length,last.loc[common,'pipe_length'],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(check.pipe_age,last.loc[common,'pipe_age']+365/365.25,rtol=1e-9,atol=1e-8)
    absent = set(eligible.pipe_id)-set(data.pipe_id)
    if absent:
        raise ValueError(f'{len(absent)} eligible pipes absent from training population; review scope.')
    print(f'Training on {len(data):,} pipe-years; predicting {len(eligible):,} eligible pipes.',flush=True)
    xgb = fit_model(data)
    cat = CatBoostClassifier(**CAT_PARAMS).fit(data[FEATURES],data.Y,
                                              cat_features=['material','surface'])
    train_encoded = pd.get_dummies(data[FEATURES],columns=['material','surface'],dtype=float)
    future_encoded = pd.get_dummies(eligible[FEATURES],columns=['material','surface'],dtype=float).reindex(
        columns=train_encoded.columns,fill_value=0.)
    lgb = LGBMClassifier(**LGB_PARAMS).fit(train_encoded,data.Y)
    result = eligible[['pipe_id','year']].reset_index(drop=True)
    result['xgboost_probability'] = xgb.predict_proba(eligible[FEATURES])[:,1].astype(float)
    result['catboost_probability'] = cat.predict_proba(eligible[FEATURES])[:,1]
    result['lightgbm_probability'] = lgb.predict_proba(future_encoded)[:,1]
    components = ['xgboost_probability','catboost_probability','lightgbm_probability']
    result['leak_probability'] = result[components].mean(axis=1)
    if result.pipe_id.duplicated().any() or not result[components+['leak_probability']].apply(
        lambda s:np.isfinite(s)&s.between(0,1)).all().all():
        raise ValueError('Invalid output IDs/probabilities.')
    result.to_csv(OUT/'ensemble_2027_probabilities.csv',index=False)
    result[['pipe_id','year','leak_probability']].to_csv(OUT/'ensemble_2027_for_cost_join.csv',index=False)
    saved = pd.read_csv(OUT/'ensemble_2027_probabilities.csv',dtype={'pipe_id':'string'})
    pd.testing.assert_frame_equal(saved,result,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(saved.leak_probability,saved[components].mean(axis=1),rtol=1e-12)
    eligible[['pipe_id','year',*FEATURES]].to_csv(OUT/'prediction_features_2027.csv',index=False)
    audit = inventory[['pipe_id','material','surface','exclusion_reason']].copy()
    audit['eligible'] = audit.exclusion_reason.eq('')
    audit.to_csv(OUT/'inventory_scope_audit.csv',index=False)
    joblib.dump(dict(xgboost=xgb,catboost=cat,lightgbm=lgb,
                     lightgbm_columns=list(train_encoded.columns),weights=[1/3]*3),OUT/'ensemble_models.joblib')
    totals = result[components+['leak_probability']].sum().to_dict()
    manifest = dict(prediction_year=2027,training_years=[2019,2026],training_rows=len(data),
        inventory_rows=len(inventory),eligible_rows=len(eligible),excluded_rows=len(inventory)-len(eligible),
        historically_leaked_inventory=len(leaked),feature_checks_against_2026=len(common),expected_leaks=totals,
        weights=dict(xgboost=1/3,catboost=1/3,lightgbm=1/3),
        parameters=dict(xgboost=XGB_PARAMS,catboost=CAT_PARAMS,lightgbm=LGB_PARAMS),
        versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
                      sklearn=sklearn.__version__,xgboost=xgboost.__version__,
                      catboost=catboost.__version__,lightgbm=lightgbm.__version__),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DEFAULT_INPUT,pipes_path,history_path]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Annual first-recorded-leak probabilities for eligible inventory without historical leaks; before any 2027 planned replacement.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report = ['# 2027 ensemble leak probabilities','',
        f'Predictions generated for **{len(eligible):,}** pipes from **{len(inventory):,}** inventory pipes.',
        f'Training uses **{len(data):,}** pipe-year observations from 2019–2026.',
        '',f'Ensemble expected first leaks in 2027: **{totals["leak_probability"]:,.1f}**, before planned replacements.',
        '', 'Each pipe probability is the equal-weight mean of raw XGBoost, CatBoost, and unweighted baseline LightGBM. Configurations are frozen; no calibration or tuning is added.',
        '', '## Scope and validation','',
        'Historical leaks imply replacement and exclusion from this first-leak model. Missing required features also cause exclusion. See inventory_scope_audit.csv for every inventory ID and reason. Excluded pipes have unknown/out-of-scope probabilities, not zero risk.',
        f'{len(common):,} eligible feature records were checked against the 2026 master: material, surface, length, and age advanced by 365/365.25 years.',
        'IDs, probability bounds, ensemble arithmetic, and CSV round-trip contents were validated.',
        '', '## Files','',
        '- ensemble_2027_probabilities.csv: three component probabilities and final leak_probability.',
        '- ensemble_2027_for_cost_join.csv: pipe_id, year, leak_probability.',
        '- prediction_features_2027.csv: inputs for eligible pipes.',
        '- inventory_scope_audit.csv: full inventory eligibility and exclusion reasons.',
        '- ensemble_models.joblib: fitted models, encoding columns, and weights; load only trusted model files.',
        '- manifest.json: training settings, versions, input hashes, scope, and expected counts.',
        '', '## Interpretation','',
        f'Component expected leak counts: XGBoost {totals["xgboost_probability"]:,.1f}; CatBoost {totals["catboost_probability"]:,.1f}; LightGBM {totals["lightgbm_probability"]:,.1f}. Substantial disagreement is an uncertainty signal. The agreed weights were not changed.',
        'These are forecasts with no known 2027 labels, not verified future accuracy. Different package versions may affect results. The prior ensemble underestimated 2026 leak counts by 19.12%; this forecast has not been mechanically adjusted to match 2026.',
        'For repair-cost integration join pipe_id to the cost model ID and check year, population overlap, and conditional-cost meaning. Probability times conditional repair cost is an expected-loss estimate, not scheduled construction cost.',
        'Previously leaked polyurethane pipes and inventory with missing attributes still need separate handling in full-network planning. Probabilities for original polyurethane pipes with no historical leak remain in this first-recorded-leak population.',
        'Long-horizon 2027–2051 predictions and replacement scheduling are not included. Planned replacements alter the risk set; do not apply this pre-work forecast unchanged afterward.']
    (OUT/'README.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['inventory_rows','eligible_rows','excluded_rows','expected_leaks']},indent=2))


if __name__ == '__main__':
    run()
