"""Train the agreed raw XGBoost/CatBoost/baseline-LightGBM ensemble for 2027.

Run from any directory: python predict_2027.py
Install: pip install numpy pandas scikit-learn xgboost catboost lightgbm joblib
Outputs go to deployment_2027_zero_leak_applied/ beside this file.
Confirmed project rule: original polyurethane and historically leaked/replaced
pipes have zero future leak probability. Historical training is unchanged. Historical first leaks remove
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

OUT = ROOT / 'deployment_2027_zero_leak_applied'
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
    raw_result = result.copy()
    polyurethane = eligible.material.str.strip().str.lower().eq('polyurethane').to_numpy()
    result.loc[polyurethane, components] = 0.0
    result['leak_probability'] = result[components].mean(axis=1)
    result['probability_status'] = np.where(polyurethane, 'zero_original_polyurethane', 'ensemble_prediction')
    np.testing.assert_array_equal(result.loc[~polyurethane, components], raw_result.loc[~polyurethane, components])
    assert (result.loc[polyurethane, components + ['leak_probability']] == 0).all().all()

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
    full = inventory[['pipe_id', 'year', 'material', 'exclusion_reason']].merge(
        result[['pipe_id', *components, 'leak_probability', 'probability_status']],
        on='pipe_id', how='left', validate='one_to_one')
    historical_zero = full.pipe_id.isin(leaked)
    original_zero = full.material.str.strip().str.lower().eq('polyurethane').fillna(False)
    known_zero = historical_zero | original_zero
    full.loc[known_zero, components + ['leak_probability']] = 0.0
    full.loc[original_zero, 'probability_status'] = 'zero_original_polyurethane'
    full.loc[historical_zero, 'probability_status'] = 'zero_historical_leak_replaced'
    full['probability_status'] = full.probability_status.fillna('unknown_missing_required_input')
    unknown_count = int(full.leak_probability.isna().sum())
    assert unknown_count == 17, 'Prediction scope changed: review missing-input policy.'
    assert (full.loc[known_zero, components + ['leak_probability']] == 0).all().all()
    full.to_csv(OUT/'full_inventory_2027_probabilities.csv',index=False)
    full.loc[full.leak_probability.isna()].to_csv(OUT/'missing_input_17_pipes.csv',index=False)

    joblib.dump(dict(xgboost=xgb,catboost=cat,lightgbm=lgb,
                     lightgbm_columns=list(train_encoded.columns),weights=[1/3]*3),OUT/'ensemble_models.joblib')
    totals = result[components+['leak_probability']].sum().to_dict()
    manifest = dict(prediction_year=2027,training_years=[2019,2026],training_rows=len(data),
        inventory_rows=len(inventory),eligible_rows=len(eligible),excluded_rows=len(inventory)-len(eligible),
        historically_leaked_inventory=len(leaked),feature_checks_against_2026=len(common),expected_leaks=totals,
        zero_original_polyurethane_rows=int(polyurethane.sum()),
        full_inventory_zero_rows=int(known_zero.sum()),unknown_probability_rows=unknown_count,
        zero_leak_rule='Originally polyurethane and all historical leak replacements have exactly zero risk.',
        weights=dict(xgboost=1/3,catboost=1/3,lightgbm=1/3),
        parameters=dict(xgboost=XGB_PARAMS,catboost=CAT_PARAMS,lightgbm=LGB_PARAMS),
        versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
                      sklearn=sklearn.__version__,xgboost=xgboost.__version__,
                      catboost=catboost.__version__,lightgbm=lightgbm.__version__),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DEFAULT_INPUT,pipes_path,history_path]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='2027 pre-work ensemble probabilities with confirmed zero-risk rules; missing-input risks remain unknown. Training and historical comparison unchanged.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report = ['# 2027 ensemble forecast with confirmed zero-leak rules', '',
        'Run from the project repository:', '',
        '```powershell', 'python notebooks/leak_model_comparison/predict_2027_zero_leak_applied.py', '```', '',
        'Dependencies: numpy, pandas, scikit-learn, xgboost, catboost, lightgbm, joblib.', '',
        '## What changed compared with the previous forecast', '',
        '- **Purpose:** apply the confirmed zero-leak rules to the 2027 forecast; this is a rule-based correction, not a new model-selection experiment.',
        '- **Originally polyurethane pipes:** 1,678 pipes previously had model-generated probabilities; all three component probabilities and the ensemble probability are now exactly zero.',
        '- **Previously leaked/replaced pipes:** 14,113 pipes were excluded from the earlier prediction population. They are now explicitly included with zero probability in the new full-inventory file; the cost-join file still excludes them.',
        '- **Other predicted pipes:** probabilities are exactly unchanged from the previous export.',
        '- **Missing-input pipes:** the same 17 pipes remain unknown, not zero. They still need a separate missing-data policy.',
        '- **Expected leak count:** decreased from 3,113.3 to 3,080.4, a reduction of approximately 32.9 expected leaks (1.06%). These totals exclude the unknown contribution of the 17 missing-input pipes.',
        '- **Unchanged modeling:** the three model configurations, equal one-third weights, historical training data, and historical selection/evaluation results are unchanged. The script refits the same models and then applies the zero-risk override.',
        '- **Preserved files:** the original predict_2027.py and deployment_2027/ outputs remain unchanged; corrected outputs are saved separately in deployment_2027_zero_leak_applied/.',
        '- **Historical metrics:** the reported 2023–2025 and 2026 metrics belong to the earlier workflow and have not been recalculated with the zero-risk override.',
        '',
        '## Counts and expected leaks', '',
        f'- Historical training: {len(data):,} pipe-year rows, 2019–2026.',
        f'- Full inventory: {len(inventory):,} pipes.',
        f'- Cost-join forecast: {len(result):,} pipes without historical leaks and with complete inputs.',
        f'- Originally polyurethane pipes zeroed in that forecast: {int(polyurethane.sum()):,}.',
        f'- Historically leaked/replaced inventory pipes assigned zero: {len(leaked):,}.',
        f'- Full-inventory zero-risk pipes, counted once: {int(known_zero.sum()):,}.',
        f'- Missing-input pipes with unknown probability: {unknown_count}.',
        f'- Revised expected leaks across known probabilities: {totals["leak_probability"]:,.4f}. This excludes the unknown contribution of the 17 missing-input pipes.',
        f'- Components after the rule: XGBoost {totals["xgboost_probability"]:,.4f}; CatBoost {totals["catboost_probability"]:,.4f}; LightGBM {totals["lightgbm_probability"]:,.4f}.', '',
        '## Files', '',
        '| File | Purpose |', '|---|---|',
        '| ensemble_2027_probabilities.csv | Component and ensemble probabilities for the 28,909 complete-input, no-history pipes; includes zero-risk original polyurethane and probability_status. |',
        '| ensemble_2027_for_cost_join.csv | pipe_id, year, leak_probability for joining to the cost team. |',
        '| full_inventory_2027_probabilities.csv | All inventory IDs, explicit zero risks, missing-input unknowns, and reasons/status. |',
        '| missing_input_17_pipes.csv | Pipes whose risk remains unknown. |',
        '| prediction_features_2027.csv | Features used for the complete-input population. |',
        '| inventory_scope_audit.csv | Input eligibility; eligible means complete input/no history, not necessarily nonzero risk. |',
        '| ensemble_models.joblib | Fitted estimators and encoding; the zero rule must still be applied when using them directly. |',
        '| manifest.json | Settings, hashes, versions, counts, confirmed rule and totals. |', '',
        '## Difference from cost-team pipe count', '',
        'The cost team has 28,926 IDs; the cost-join export has 28,909. All exported IDs match the cost population. The 17 missing probabilities are not zero: nine wrought-iron pipes lack installation dates (P10869, P12275–P12282); eight lack material (P09986, P09987, P42036–P42041). None has a recorded historical leak.',
        'Left-join from the cost table, preserve these 17, and flag missing probabilities rather than filling them with zero. Expected leak loss equals annual leak probability times conditional repair cost, not construction cost.', '',
        '## Validation and limitations', '',
        'Validated IDs, bounds, exact zero-rule application, unchanged non-polyurethane component values before/after the override, ensemble arithmetic, reconstructed historical features, and CSV round-trip contents.',
        'The original approximately 3,113 expected-leak total is superseded by this rule-adjusted forecast. Historical selection and 2026 metrics were not recalculated; they describe the earlier workflow without this explicit override.',
        'No 2027 labels exist. Substantial model disagreement remains. The unknown contribution of missing-input pipes is not included in the total. This is not a full 2027–2051 replacement simulation.',
        'When planned replacement occurs, newly polyurethane pipes must also receive zero future leak risk.']
    (OUT/'README.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['inventory_rows','eligible_rows','excluded_rows','expected_leaks']},indent=2))


if __name__ == '__main__':
    run()
