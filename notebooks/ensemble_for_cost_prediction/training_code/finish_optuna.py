"""Assemble, independently verify and document the optimized members."""
import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
from contextlib import redirect_stdout
import io
import ast
import joblib
import numpy as np
import pandas as pd
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train
from features import HERE, FeatureBuilder
from optimize import ROOT, OUT, MODELS, CORE, predict

def forecast_year(year):
    builder=FeatureBuilder();q,x=builder.future(year)
    c=json.loads((MODELS/'training_contract.json').read_text())
    assert list(x)==c['features']
    f=q[['pipe_id']].assign(scenario_year=year)
    for name in CORE:f[name]=predict(joblib.load(MODELS/f'{name}.joblib'),x)
    weights=json.loads((MODELS/'ensemble_weights.json').read_text())
    for name,w in weights.items():f[name]=f[CORE].to_numpy()@np.array([w[n] for n in CORE])
    f['predicted_cost']=f[c['default_ensemble']]
    assert np.isfinite(f[CORE+['predicted_cost']].to_numpy()).all()
    f.to_csv(OUT/f'forecast_costs_{year}.csv',index=False)
    f[['pipe_id','scenario_year','predicted_cost']].to_csv(OUT/f'forecast_costs_{year}_adopted.csv',index=False)
    return f

def main():
    train.OUT=OUT;train.MODELS=MODELS
    # Import only after redirecting module constants to the isolated new run.
    import stabilize
    d=pd.read_csv(HERE/'data/training_rich.csv');oof=pd.read_csv(OUT/'out_of_time_predictions.csv')
    oof,weights=train.ensemble_methods(oof)
    oof.to_csv(OUT/'out_of_time_predictions.csv',index=False);weights.to_csv(OUT/'past_only_weights.csv',index=False)
    train.diagnostics(oof,d)
    final={'ensemble_equal':np.full(len(CORE),1/len(CORE)),
        'ensemble_rmse_weighted':train.fit_weights(oof,0),
        'ensemble_stable':.5*train.fit_weights(oof,.02)+.5/len(CORE)}
    train.dump(MODELS/'ensemble_weights.json',{n:dict(zip(CORE,w.tolist())) for n,w in final.items()})
    f=pd.read_csv(OUT/'forecast_costs_2027.csv')
    for name,w in final.items():f[name]=f[CORE].to_numpy()@w
    f.to_csv(OUT/'forecast_costs_2027.csv',index=False)
    stabilize.main()
    f=pd.read_csv(OUT/'forecast_costs_2027.csv')
    summary=pd.read_csv(OUT/'summary_metrics.csv').set_index('model')
    # Retrospective deployment choice, clearly distinct from temporal fitting.
    # Preserve every retrained family in production, as requested for
    # stabilization. Unconstrained weights remain a comparison benchmark.
    choices=summary.loc[['ensemble_equal','ensemble_stable','ensemble_balanced']]
    eligible=choices[choices.mean_annual_RMSE<=choices.mean_annual_RMSE.min()*1.01]
    selected=eligible.sort_values(['worst_absolute_annual_total_bias_pct','mean_absolute_annual_total_bias_pct','annual_relative_RMSE_sd']).index[0]
    contract=json.loads((MODELS/'training_contract.json').read_text())
    contract.update(default_ensemble=selected,default_is_provisional=False,
        adoption_reason='Among diversified ensemble methods (positive weight for every member) within 1% of their best mean annual RMSE, prefer lower worst annual total bias, then mean absolute bias, then relative annual RMSE variability.',
        selection_caveat='Deployment method chosen after inspecting retrospective 2023-2026 results; not an untouched holdout. No future labels used.')
    train.dump(MODELS/'training_contract.json',contract)
    f['predicted_cost']=f[selected];f.to_csv(OUT/'forecast_costs_2027.csv',index=False)
    f[['pipe_id','scenario_year','predicted_cost']].to_csv(OUT/'forecast_costs_2027_adopted.csv',index=False)
    validate(d,oof=pd.read_csv(OUT/'out_of_time_predictions.csv'),forecast=f,contract=contract)
    document(summary,contract)
    print('Optuna models and verified ensemble complete:',selected,flush=True)

def validate(d,oof,forecast,contract):
    assert len(d)==14113 and len(oof)==13035 and len(forecast)==27240
    assert d.event_id.is_unique and oof.event_id.is_unique and forecast.pipe_id.is_unique
    assert set(oof.event_id)==set(d.loc[d.event_year>=2020,'event_id'])
    assert np.allclose(oof.cost,d.set_index('event_id').cost.reindex(oof.event_id))
    assert np.isclose(d.cost.sum(),352997703.33)
    assert np.isfinite(oof[CORE].to_numpy()).all() and (oof[CORE].to_numpy()>=0).all()
    x=pd.read_csv(HERE/'data/forecast_features_2027.csv');sample=x.iloc[np.linspace(0,len(x)-1,100,dtype=int)]
    assert forecast.pipe_id.tolist()==x.pipe_id.tolist()
    configs=[]
    for name in CORE:
        bundle=joblib.load(MODELS/f'{name}.joblib')
        assert np.allclose(predict(bundle,sample),forecast[name].iloc[sample.index],rtol=1e-9,atol=1e-5),name
        for year in range(2020,2028):
            cfg=json.loads((OUT/'configs'/name/f'{year}.json').read_text())
            if name=='anthony_linear_raw':assert 'regularizer' in cfg['params']
            assert cfg['training_end']<year and all(y<year for y in cfg['inner_years'])
            assert cfg['training_rows']==int((d.event_year<year).sum())
            assert set(cfg['features'])<=set(contract['features'])
            trials=pd.read_csv(OUT/f'trials_{name}_{year}.csv') if year>2020 else pd.DataFrame()
            if year>2020:
                assert len(trials)==contract['optuna_trials'] and cfg['completed_trials']>0
                assert np.isclose(trials.loc[trials.state=='COMPLETE','value'],cfg['inner_mean_RMSE']).any()
            configs.append({'model':name,'prediction_year':year,'trials':len(trials),'completed_trials':cfg['completed_trials'],
                'feature_pack':cfg['params'].get('feature_pack','surface'),'inner_mean_RMSE':cfg['inner_mean_RMSE'],
                'training_end':cfg['training_end'],'training_rows':cfg['training_rows']})
        oof[['event_id','pipe_id','event_year','cost',name]].rename(columns={name:'predicted_cost'}).to_csv(OUT/f'output_{name}_out_of_time.csv',index=False)
        forecast[['pipe_id','scenario_year',name]].rename(columns={name:'predicted_cost'}).to_csv(OUT/f'output_{name}_2027.csv',index=False)
    for filename in ['past_only_weights.csv','balanced_past_only_weights.csv']:
        w=pd.read_csv(OUT/filename);assert (w.weight_training_end<w.test_year).all()
        by=['test_year']+(['method'] if 'method' in w else [])
        assert np.allclose(w.groupby(by).weight.sum(),1) and w.weight.ge(-1e-9).all()
    weights=json.loads((MODELS/'ensemble_weights.json').read_text())
    for method,w in weights.items():
        assert set(w)==set(CORE) and min(w.values())>=0 and np.isclose(sum(w.values()),1)
        assert np.allclose(forecast[CORE].to_numpy()@np.array([w[n] for n in CORE]),forecast[method],rtol=1e-10,atol=1e-6)
    for _,row in pd.read_csv(OUT/'annual_metrics.csv').iterrows():
        q=oof[oof.event_year==row.year]
        assert np.isclose(train.rmse(q.cost,q[row.model]),row.RMSE)
        assert np.isclose(100*(q[row.model].sum()/q.cost.sum()-1),row.total_bias_pct)
    builder=FeatureBuilder();checks=[]
    for year in range(2023,2027):
        q=builder.events[builder.events.event_year==year].head(8);before=builder.build(q,year)
        saved=builder.repair_date.copy();builder.repair_date[builder.repair_date>=np.datetime64(f'{year}-01-01')]=np.datetime64('2099-01-01')
        pd.testing.assert_frame_equal(before,builder.build(q,year));builder.repair_date=saved
        expected=d.loc[d.event_id.isin(q.event_id),contract['features']].reset_index(drop=True)
        pd.testing.assert_frame_equal(before,expected,check_dtype=False,rtol=1e-9,atol=1e-9)
        checks.append(year)
    q,_=builder.future(2027);q=q.head(8);clean=builder.build(q,2027)
    injected=pd.concat([builder.events,builder.events.head(1).assign(event_year=2030,cost=1e12)],ignore_index=True)
    pd.testing.assert_frame_equal(clean,builder.build(q,2027,injected))
    manifest=json.loads((HERE.parent/'ensemble_preparation/sources/manifest.json').read_text())
    for meta in manifest['local_inputs'].values():assert hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest()==meta['sha256']
    pd.DataFrame(configs).to_csv(OUT/'selected_configs_summary.csv',index=False)
    train.dump(OUT/'validation.json',{'status':'passed','training_rows':len(d),'out_of_time_rows':len(oof),
        'evaluation_rows':int((oof.event_year>=2023).sum()),'future_supported_pipes':len(forecast),
        'features':len(contract['features']),'temporal_config_checks':len(configs),
        'saved_models_replayed':len(CORE),'all_metrics_and_blends_independently_recomputed':True,
        'future_date_mutation_invariance_years':checks,'future_cost_injection_invariance':True,
        'all_2026_and_earlier_events_retained':True,'source_commit':manifest['commit'],
        'versions':{name:importlib.metadata.version(name) for name in ['optuna','numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','catboost','joblib']},
        'sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [HERE/'data/training_rich.csv',HERE/'data/forecast_features_2027.csv',HERE/'features.py',HERE/'optimize.py',
             HERE/'finish_optuna.py',OUT/'out_of_time_predictions.csv',OUT/'forecast_costs_2027.csv']}})

def document(summary,c):
    oof=pd.read_csv(OUT/'out_of_time_predictions.csv');metrics=pd.read_csv(OUT/'annual_metrics.csv')
    methods=['prior_surface_mean','ensemble_equal','ensemble_rmse_weighted','ensemble_stable','ensemble_balanced']
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for name in methods:
        q=metrics[metrics.model==name];ax[0].plot(q.year,q.RMSE,marker='o',label=name);ax[1].plot(q.year,q.total_bias_pct,marker='o',label=name)
    ax[0].set(ylabel='RMSE in original cost units',xlabel='Validation year');ax[1].set(ylabel='Annual total bias (%)',xlabel='Validation year');ax[1].axhline(0,color='grey',lw=.7)
    for a in ax:a.set_xticks(range(2023,2027));a.grid(alpha=.2)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=(0,.15,1,1));fig.savefig(ROOT/'stability.png',dpi=180);plt.close(fig)
    table=summary.loc[methods,['mean_annual_RMSE','annual_RMSE_sd','mean_absolute_annual_total_bias_pct','worst_absolute_annual_total_bias_pct']].round(2).to_markdown()
    text=f'''# Optuna最適化後のアンサンブル

各モデル・各学習期間で30trial。2020年のみ、学習年が2019年しかないため探索せず固定設定です。2021–2027の10モデルで計2,100trialの探索枠です（収束失敗・処理方式切替で中断したtrialも含み、完了数はCSVで確認）。正常な試行を内側RMSEの順で選び、全学習データでも収束・有限予測を確認して採用します。2027用の全モデルは2019–2026年の14,113件を再学習しています。

採用候補: `{c['default_ensemble']}`。安定化のため全モデルに正の重みを残す等重み・縮小重み・制約付き重みを採用候補とします。RMSEを主軸とし、その候補内で最良の年平均RMSEから1%以内にある方法を、年間総額偏りの最大値、平均値、相対年間RMSEのばらつきの順に比較しました。制約のないRMSE重みは比較用です。選択は過去結果を確認した後なので、未使用ホールドアウトでの評価とは呼びません。

{table}

## 漏洩の防止

各外側予測年YではYより前の年のみで学習し、その内側の最新2年を順方向検証します。特徴量の選択、前処理、探索、対数補正、重み学習は外側Yの正解を使いません。内側年の予測時も、その内側年より前のラベルのみ使用します。空間特徴量の距離単位はmです。歴史的な修理状態や費用の集約は各年1月1日より前だけを使います。

重み学習には2020–2026年の正当な順方向OOF予測13,035件を使用します。2019年には、それより前の学習用ラベルがないためOOF予測を作りません。2019年の1,078件も含め、最終的なベースモデルには全14,113件を使っています。

Shanaの既存予測、将来に利用できない故障時刻・曜日を使った既存予測は投入しません。Anthony/Rionのモデル系統とTakitoの3モデルを新たに最適化・再学習した予測だけを使います。

## 探索範囲

`../optimize.py`の`space()`に全範囲を記載しています。特徴量6群、線形回帰のOLS/Ridge/Lasso/ElasticNet・正則化強度・L1比率・切片・非負係数制約、GammaのL2正則化・切片、RFの木数・深さ・葉/分岐の最小件数・特徴量比率・最大葉数・bootstrap/標本比率・2種類の二乗誤差criterion、boostersの学習率・木数・複雑度・正則化・標本/特徴量比率・bin数などを探索します。XGBoost/LightGBMはL1・L2、CatBoostはL2を探索します。対数モデルは補正なし、学習残差補正、過去OOF補正も探索します。

線形回帰は数値特徴量を学習期間内だけで標準化し、学習期間の平均費用で目的変数を正規化してから推定・元単位に戻します。Ridgeのalphaは1e-4–1e5、Lasso/ElasticNetは正規化した目的変数に対するalpha 1e-4–10、ElasticNetのL1比率は0.01–0.99です。初期のOLSだけの探索は`previous_ols/`に保存し、採用する線形回帰は別studyで改めて30trialを実行しています。

すべてのライブラリ引数を探索したわけではありません。乱数、スレッド数、目的変数、目的関数、CPU実装、収束制御は固定です。RFのabsolute_error/Poissonや別のbooster/loss系統は範囲外です。30trialで探索空間全体や大域最適値を保証しません。Gamma/線形などに豊富な特徴量を追加する候補もあるため、元ノートブックの数値的な再現ではなく、同じモデル系統の安全な改良です。

## 予算への適用

2027用CSVは故障した場合の条件付き修理費です。将来予算には故障確率・期待故障件数と組み合わせる必要があります。歴史年の実際の故障群に対する総額偏りは、その条件付き費用の検証です。将来の全パイプ費用を単純合計すると予算にはなりません。

予測対象は修理前の既知材料27,240本です。PU15,791本と不明材料8本の計15,799本には故障費用の教師データがなく、ゼロ円扱いせず未推定として別途管理します。2027以降の履歴は2026年末で固定し、将来の未知修理・費用は追加しません。

## 再実行と成果物

`python ensemble_training/run_optuna.py`で各モデル30trialの探索・全件再学習・アンサンブル・検証を行います（5個の独立したローカル処理）。探索はSQLiteに保存され、完了済みの年別予測を再利用します。2027以降は`python ensemble_training/finish_optuna.py --year 2030`で再予測できます。

`results/selected_configs_summary.csv`、`results/trials_*.csv`、`results/configs/`で選択設定・全trialを確認できます。`results/validation.json`は独立再計算・保存モデル再生・将来ラベル/日付の変更検査の結果です。`results/forecast_costs_2027_adopted.csv`が採用予測、`results/out_of_time_predictions.csv`が過去の順方向予測です。元の固定グリッド成果物は上書きしていません。

入力の出典: GitHub `reinesana/STAT440-star-kingdom-project` commit `{json.loads((OUT/'validation.json').read_text())['source_commit']}`。ファイルハッシュと取得経路は`../../ensemble_preparation/sources/manifest.json`。
'''
    (ROOT/'README_ja.md').write_text(text,encoding='utf-8')
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\ncandidates=[Path.cwd(),Path.cwd()/'optuna',Path.cwd()/'ensemble_training'/'optuna']\nroot=next(p for p in candidates if (p/'search_policy.json').exists())\npd.read_csv(root/'results'/'summary_metrics.csv')"),nbformat.v4.new_code_cell("pd.read_csv(root/'results'/'selected_configs_summary.csv')")])
    # Actually execute cells in one Python namespace; no Jupyter kernel installed.
    namespace={};counter=0
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        counter+=1;cell.execution_count=counter;tree=ast.parse(cell.source)
        tail=tree.body.pop() if isinstance(tree.body[-1],ast.Expr) else None
        stream=io.StringIO()
        with redirect_stdout(stream):
            exec(compile(tree,'review.ipynb','exec'),namespace)
            result=eval(compile(ast.Expression(tail.value),'review.ipynb','eval'),namespace) if tail else None
        cell.outputs=[]
        if stream.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
        if result is not None:
            data={'text/plain':str(result)}
            if isinstance(result,pd.DataFrame):data['text/html']=result.to_html(index=False)
            cell.outputs.append(nbformat.v4.new_output('execute_result',execution_count=counter,data=data))
    nbformat.write(notebook,ROOT/'review.ipynb')

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--year',type=int);args=p.parse_args()
    if args.year:
        if args.year<2027:p.error('Year must be >=2027')
        forecast_year(args.year)
    else:main()
