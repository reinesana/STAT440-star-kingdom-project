"""Publish bilingual, evidence-backed candidate files into the repository checkout."""
from pathlib import Path
import sys, json, shutil, hashlib
import numpy as np
import pandas as pd
import joblib
from features import FeatureBuilder, HERE, BASE
from optimize import predict, CORE, MODELS, OUT

DEST=HERE.parent/'github-main/notebooks/ensemble_for_cost_prediction'
METHODS=['takito_catboost','ensemble_equal','ensemble_stable','ensemble_balanced','ensemble_rmse_weighted','prior_surface_mean']
DESCRIPTIONS={
'takito_catboost':('Best single model: CatBoost','最良単一モデル：CatBoost','One raw-cost CatBoost model. Lowest mean annual and pooled RMSE among evaluated single models; higher budget bias.','元単位の費用を予測するCatBoost単体。単一モデル中で年平均・プールRMSEが最小。ただし総額の偏りは大きい。'),
'ensemble_equal':('Equal-weight ensemble','等重みアンサンブル','Ten components, each weighted 10%. Lowest mean annual RMSE among the ensembles.','10構成モデルを各10%で平均。アンサンブル中では年平均RMSEが最小。'),
'ensemble_stable':('Shrinkage ensemble (budget-focused default)','縮小重みアンサンブル（予算重視の既定候補）','50% regularized past-OOF RMSE weights + 50% equal weights. Every component has at least 5% weight. Lowest mean and worst absolute annual total bias among compared candidates.','過去OOFのRMSE重みに正則化を加え、その50%と等重み50%を混合。各構成モデルに最低5%の重みを残す。比較候補中で平均・最悪年の総額絶対偏りが最小。'),
'ensemble_balanced':('Constrained balance ensemble','制約付きバランスアンサンブル','Past-only optimization considers annual total bias, relative annual RMSE variation and a uniform-weight penalty, with 1–70% component weights and an RMSE constraint. Its observed bias is worse than shrinkage.','過去データだけで年間総額偏り・相対年間RMSEのばらつき・等重みからの乖離を考慮。各重み1〜70%、RMSE制約付き。今回の実測偏りは縮小重みより大きい。'),
'ensemble_rmse_weighted':('RMSE-optimized ensemble','RMSE最適化アンサンブル','Nonnegative weights summing to one minimize past mean annual RMSE. Zero weights are allowed; concentration risk remains.','過去の年平均RMSEを最小化する非負・合計1の重み。ゼロ重みを許すため一部モデルへの集中が残る。'),
'prior_surface_mean':('Surface mean (reference)','表面種別の過去平均（比較基準）','All-history surface-specific mean, shrunk toward the global mean using an Optuna-selected smoothing strength.','全学習履歴の表面種別平均を全体平均へ縮小。縮小強度はOptunaで選択。')}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for folder in ['predictions','evidence','configs','training_code']: (DEST/folder).mkdir(exist_ok=True)
    pipes=pd.read_csv(HERE.parent/'github-main/data/pipes.csv');train=pd.read_csv(HERE.parent/'github-main/data/train.csv')
    target=pipes.loc[~pipes['Pipe ID'].isin(train['Pipe ID'])].copy()
    saved=pd.read_csv(OUT/'forecast_costs_2027.csv')
    extra=target.loc[~target['Pipe ID'].isin(saved.pipe_id)].copy()
    assert len(train)==14113 and train['Pipe ID'].is_unique and len(target)==28926
    builder=FeatureBuilder()
    q=pd.DataFrame({'pipe_id':extra['Pipe ID'].to_numpy(),'scenario_year':2027,
       'material':extra.Material.fillna('__unknown__').to_numpy(),'surface':extra.Surface.fillna('__unknown__').to_numpy()})
    q['pipe_length_m']=np.linalg.norm(extra[['GPS x2','GPS y2']].to_numpy()-extra[['GPS x1','GPS y1']].to_numpy(),axis=1)
    q['log1p_pipe_length_m']=np.log1p(q.pipe_length_m)
    q['midpoint_x_m']=(extra['GPS x1'].to_numpy()+extra['GPS x2'].to_numpy())/2
    q['midpoint_y_m']=(extra['GPS y1'].to_numpy()+extra['GPS y2'].to_numpy())/2
    q['age_at_year_start']=(pd.Timestamp('2027-01-01')-pd.to_datetime(extra['Lay date'])).dt.days.to_numpy()/365.25
    q['age_missing']=q.age_at_year_start.isna().astype(int);q['log1p_age_at_year_start']=np.log1p(q.age_at_year_start)
    q['material_surface']=q.material+' | '+q.surface
    x=builder.build(q,origin_year=2027)
    ext=q[['pipe_id']].copy();weights=json.loads((MODELS/'ensemble_weights.json').read_text())
    for name in CORE:
        cfg=json.loads((OUT/'configs'/name/'2027.json').read_text());assert cfg['training_rows']==len(train) and cfg['training_end']==2026
        bundle=joblib.load(MODELS/(name+'.joblib'));ext[name]=predict(bundle,x)
        assert np.isfinite(ext[name]).all()
        shutil.copy2(OUT/'configs'/name/'2027.json',DEST/'configs'/f'{name}_2027.json')
        print('Replayed full-data model:',name,flush=True)
    for method,w in weights.items():ext[method]=ext[CORE].to_numpy()@np.array([w[n] for n in CORE])
    joined=pd.concat([saved[['pipe_id']+METHODS],ext[['pipe_id']+METHODS]],ignore_index=True)
    assert joined.pipe_id.is_unique and set(joined.pipe_id)==set(target['Pipe ID'])
    for method in METHODS:
        result=joined[['pipe_id',method]].rename(columns={'pipe_id':'id',method:'predicted_cost'})
        result=result.sort_values(['predicted_cost','id'],ascending=[False,True]).reset_index(drop=True)
        assert len(result)==len(target) and np.isfinite(result.predicted_cost).all() and result.predicted_cost.ge(0).all()
        result.to_csv(DEST/'predictions'/f'{method}.csv',index=False,float_format='%.10f')
        read=pd.read_csv(DEST/'predictions'/f'{method}.csv')
        assert read.columns.tolist()==['id','predicted_cost'] and read.predicted_cost.is_monotonic_decreasing
    scope=target[['Pipe ID','Material']].rename(columns={'Pipe ID':'id','Material':'material'})
    scope['prediction_support']=np.where(scope.id.isin(extra['Pipe ID']),'unvalidated_unseen_material_extrapolation','supported_material')
    scope.to_csv(DEST/'evidence/prediction_scope.csv',index=False)
    summary=pd.read_csv(OUT/'summary_metrics.csv').set_index('model')
    annual=pd.read_csv(OUT/'annual_metrics.csv')
    for filename in ['summary_metrics.csv','annual_metrics.csv','selected_configs_summary.csv','validation.json','out_of_time_weights.csv']:
        source=OUT/filename
        if source.exists():shutil.copy2(source,DEST/'evidence'/filename)
    shutil.copy2(MODELS/'ensemble_weights.json',DEST/'evidence/ensemble_weights.json')
    shutil.copy2(MODELS/'training_contract.json',DEST/'evidence/training_contract.json')
    shutil.copy2(HERE/'optuna/search_policy.json',DEST/'evidence/search_policy.json')
    for name in ['features.py','optimize.py','train.py','stabilize.py','finish_optuna.py','run_optuna.py','package_candidates.py']:
        shutil.copy2(HERE/name,DEST/'training_code'/name)
    shutil.copy2(HERE/'optuna/sources.html',DEST/'evidence/sources.html')
    shutil.copy2(HERE/'optuna/sources.json',DEST/'evidence/sources.json')
    model_table=[]
    for n in CORE:
        c=json.loads((DEST/'configs'/f'{n}_2027.json').read_text())
        model_table.append([n,c['params'].get('feature_pack','surface'),len(c.get('features',[])),json.dumps(c['params'],ensure_ascii=False)])
    pd.DataFrame(model_table,columns=['model','feature_pack','n_features','parameters']).to_csv(DEST/'evidence/final_model_settings.csv',index=False)
    weights_table=pd.DataFrame({m:([1. if n=='takito_catboost' else 0. for n in CORE] if m=='takito_catboost' else [1. if n=='prior_surface_mean' else 0. for n in CORE] if m=='prior_surface_mean' else [weights[m][n] for n in CORE]) for m in METHODS},index=CORE)
    weights_table.index.name='component';weights_table.to_csv(DEST/'evidence/candidate_weights.csv')
    for lang in ['en','ja']:write_report(lang,summary,annual,weights_table,model_table,len(target),len(extra))
    manifest={'status':'passed','scenario_year':2027,'training_rows':14113,'inventory_rows':len(pipes),'skipped_training_ids':len(train),
      'prediction_rows_per_candidate':len(target),'supported_material_rows':len(saved),'unvalidated_material_rows':len(extra),
      'unvalidated_material_breakdown':{'polyurethane':1678,'missing':8},'candidates':METHODS,'sort':'predicted_cost descending, id ascending ties',
      'columns':['id','predicted_cost'],'seed':440,'finite_nonnegative_predictions':True,'exact_inventory_minus_train_id_coverage':True,
      'source_sha256':{str(p.relative_to(HERE.parent)):digest(p) for p in [HERE.parent/'github-main/data/train.csv',HERE.parent/'github-main/data/pipes.csv',OUT/'forecast_costs_2027.csv']},
      'prediction_sha256':{p.name:digest(p) for p in (DEST/'predictions').glob('*.csv')}}
    (DEST/'evidence/export_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text('# Repair cost model candidates for 2027\n\n[English report](report_en.md) · [日本語レポート](report_ja.md)\n\nSix CSVs are in `predictions/`: five model candidates plus one reference. Each contains all 28,926 inventory IDs absent from train.csv, with exactly `id,predicted_cost`, sorted by decreasing cost. See the reports before using the 1,686 extrapolated rows.\n\n[Export validation](evidence/export_validation.json) · [Model composition](evidence/candidate_weights.csv) · [Final settings](evidence/final_model_settings.csv) · [Sources](evidence/sources.html)\n\nTraining-code snapshots are provided for inspection. They retain their original workspace paths and are not a standalone execution package; the CSV exporter was run in the original reviewed workspace. Fitted model binaries and large intermediate feature tables are not included.\n',encoding='utf-8')
    print('Validated package:',DEST,flush=True)

def write_report(lang,s,a,w,settings,n,extra):
    ja=lang=='ja';title='修理コスト予測：単一モデルとアンサンブル候補の比較' if ja else 'Repair cost prediction: single-model and ensemble candidates'
    text=[f'# {title}', '']
    text.append(('最良単一モデルのCatBoostを正式な候補に追加しました。RMSEを最優先するならCatBoost、年間総額の偏りを抑えるなら縮小重みアンサンブルが有力です。従来の既定候補は縮小重みのまま保持し、単一モデルを含む候補集合を拡張しました。' if ja else 'CatBoost is now an explicit candidate: it has the best single-model RMSE. The shrinkage ensemble remains the budget-focused default because it has smaller annual total bias. The candidate set has been expanded without automatically replacing that default.'))
    text+=['', '## '+('評価結果' if ja else 'Observed evaluation results'),'']
    metric_cols=['mean_annual_RMSE','pooled_RMSE','annual_RMSE_sd','mean_absolute_annual_total_bias_pct','worst_absolute_annual_total_bias_pct']
    table=s.loc[METHODS,metric_cols].copy();table.index.name='candidate'
    text.append(table.to_markdown(floatfmt='.2f'))
    text.append('')
    text.append(('評価は2023〜2026年の8,592件。主指標は元単位でのRMSEです。年平均RMSEは各年のRMSEの単純平均、pooled RMSEは全件をまとめたRMSE、annual_RMSE_sdは4年のRMSEの標本標準偏差です。総額偏りは100×(予測総額/実績総額−1)で、表はその年別絶対値の平均・最大を示します。' if ja else 'Evaluation covers 8,592 observed repair events in 2023–2026. RMSE in original cost units is primary. Mean annual RMSE gives each year equal weight; pooled RMSE combines all events; annual_RMSE_sd is the sample standard deviation across four years. Annual total bias is 100 × (predicted total / observed total − 1); the table reports the mean and maximum absolute annual bias.'))
    text+=['',('CatBoostの年平均RMSEは124,569.85、縮小重みは125,230.52（約0.53%高い）。一方、最悪年の総額偏りはCatBoost34.45%に対し縮小重み10.02%です。縮小重みの年別RMSE標準偏差は基準モデルより約2.34%大きく、RMSEの年別安定化まで達成したとはいえません。総額の偏り改善とRMSEのばらつきは分けて判断してください。' if ja else 'CatBoost mean annual RMSE is 124,569.85 versus 125,230.52 for shrinkage (about 0.53% higher). Worst absolute annual total bias is 34.45% versus 10.02%. Shrinkage annual RMSE standard deviation is about 2.34% higher than the reference: annual RMSE stabilization has not been established. Budget-bias improvement and RMSE variation should be assessed separately.'),'']
    text+=['## '+('各候補の説明' if ja else 'Candidate explanations'),'']
    for method in METHODS:
        d=DESCRIPTIONS[method];text +=['### '+d[1 if ja else 0],d[3 if ja else 2],'',f'CSV: [predictions/{method}.csv](predictions/{method}.csv)','']
    text+=['## '+('モデルの内訳と最終重み' if ja else 'Components and final deployment weights'),'']
    text.append((w*100).to_markdown(floatfmt='.2f'));text+=['',('重みは%。各アンサンブルは同じ10構成モデルを使います。表面平均、Anthonyの線形回帰・Gamma・対数Random Forest、TakitoのXGBoost・CatBoost・LightGBM、RionのGamma・Random Forest・対数LightGBMです。対数系は元単位の期待費用に戻す補正も過去データだけで選択しました。' if ja else 'Weights are percentages. Ensembles combine the same ten components: surface mean; Anthony linear/Gamma/log-RF; Takito XGBoost/CatBoost/LightGBM; Rion Gamma/raw-RF/log-LightGBM. Log-model back-transform corrections were also selected using past data only.'),'']
    text+=['## '+('実施した処理とリーケージ対策' if ja else 'Work performed and leakage controls'),'']
    text.append(('1. train.csvの2019〜2026年全14,113件を保持し、pipes.csvとIDで結合。欠損のある6件も復元し、高額費用を削除・上限処理しませんでした。\n2. 各予測年より前のラベルだけで、年初時点の年齢、材料・表面、距離、近傍在庫、端点経路、過去費用集約を作成。距離はm。平面交差を物理接続と見なしません。\n3. 各モデル・各学習期間2021〜2027で30trial。内側は直前の最大2年で順方向検証し、特徴量6群も探索。2020は過去年が1年だけのため固定設定。全2,100試行枠のうち正常完了1,978、失敗・中断122です。\n4. 線形回帰はOLS/Ridge/Lasso/ElasticNetを比較。XGBoostとLightGBMはL1/L2、GammaとCatBoostはL2、RFは木の深さ・葉の件数・抽出率等で過学習を抑制。最終Anthony線形回帰では正則化なしが選択されました。GammaのL1は未比較です。\n5. 重みは過去年のOOFだけで学習。2027モデルは全14,113件で学習済み。seed=440。既存のShana予測や将来利用不能な故障時刻・曜日を使う予測は除外。\n6. 保存10モデルの再生、80期間の設定、全指標・重みの独立再計算、将来費用・修理日の変更不変性を検証済み。' if ja else '1. Retained all 14,113 train.csv events from 2019–2026 and joined inventory by ID. Restored six missing-date records; no expensive events were removed or capped.\n2. Built January-1 age, material/surface, metric distances, inventory neighborhoods, endpoint routes and past-cost aggregates using labels strictly before each prediction year. Coordinates are meters; planar crossings do not imply connectivity.\n3. Ran 30 Optuna trials per model/origin in 2021–2027, with up to two latest preceding inner validation years and six feature packs. The 2020 cold start uses fixed settings. Of 2,100 trial slots, 1,978 completed and 122 failed/were interrupted.\n4. Compared OLS/Ridge/Lasso/ElasticNet; searched L1/L2 for XGBoost/LightGBM, L2 for Gamma/CatBoost, and structural controls for RF. Final Anthony linear regression selected no regularization. Gamma L1 was not evaluated.\n5. Learned ensemble weights only from preceding OOF years. Final 2027 components were trained on all 14,113 events with seed 440. Excluded existing Shana predictions and models requiring unavailable realized failure time/weekday.\n6. Verified ten saved models, 80 temporal configurations, independent metric/blend recomputation and invariance to future cost/date changes.'))
    text+=['','## '+('最良単一モデルの設定' if ja else 'Best single-model settings'),'']
    cfg=json.loads((DEST/'configs/takito_catboost_2027.json').read_text());text+=['```json',json.dumps(cfg['params'],indent=2),'```','']
    text.append(('最終CatBoostはtopology特徴量49列、深さ2、150反復を選択。これは2027用の設定で、各評価年にはその年の過去だけで選択した別の設定を使っています。全10モデルの設定はconfigs/とevidence/final_model_settings.csvに保存しました。' if ja else 'Final CatBoost selected 49 topology-pack features, depth 2 and 150 iterations. These are deployment settings for 2027; each historical fold used separately tuned past-only settings. All ten final configurations are in configs/ and evidence/final_model_settings.csv.'))
    text+=['','## '+('CSVの対象と解釈' if ja else 'Prediction population and interpretation'),'']
    text.append((f'実ファイル名はpipes.csvです。全43,039本からtrain.csvの14,113個のIDを除外し、残る{n:,}本を全候補で予測しました。各CSVはid,predicted_costの2列で、コスト降順（同額はID昇順）です。予測年は2027、履歴は2026年末で固定。既知材料27,240本の予測を保存済み全件学習モデルから使用し、追加の{extra:,}本も同じ保存モデルで推定しました。追加内訳はPU1,678本、材料不明8本で、学習に該当材料の費用がありません。未知カテゴリ処理を通した外挿であり、正当なOOF評価に含まれず、精度・校正を確認できません。これらを0円と置かず数値を出していますが、supported_materialの推定と同等の信頼度で予算に使うべきではありません。evidence/prediction_scope.csvで識別できます。' if ja else f'The actual inventory filename is pipes.csv. All 43,039 IDs were considered and the 14,113 IDs in train.csv skipped, leaving {n:,} predictions per candidate. CSVs contain exactly id,predicted_cost, sorted by decreasing cost and then ascending ID. Scenario year is 2027 with history frozen at end-2026. Predictions for 27,240 supported-material pipes come from saved full-data fits; the same models additionally predict {extra:,} pipes (1,678 PU and eight missing-material rows). No cost labels exist for these materials. These predictions use native unknown-category handling and are unvalidated extrapolations, outside the OOF evaluation. They are numeric estimates, not zero assignments, but should not be budgeted with the same confidence as supported-material estimates. Identify them in evidence/prediction_scope.csv.'))
    text+=['',('全値は故障した場合の条件付き費用で、必要修理量や故障確率ではありません。CSVの単純合計は年間予算を意味しません。修理予算には故障確率・件数との組合せが必要です。金額単位は元データの単位で、円と断定しません。評価結果を参照した後の候補選択であり、未使用の最終ホールドアウトによる性能保証ではありません。' if ja else 'Costs are conditional on a failure, not repair necessity or failure probability. Summing every CSV row does not produce an annual budget: failure probabilities/counts are needed. Monetary units follow the source data and are not assumed to be yen. Candidate selection follows inspection of historical results; this is not an untouched final holdout guarantee.'),'','## '+('年別結果' if ja else 'Annual results'),'']
    text.append(a[a.model.isin(METHODS)][['model','year','n_test','RMSE','total_bias_pct']].to_markdown(index=False,floatfmt='.2f'))
    text+=['','## '+('出典・検証' if ja else 'Sources and validation'),'','Source snapshot: `a09355d835d9acbb5746f11edcddd0cfd5e8ec3e`. [Training labels](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/train.csv), [Inventory](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/pipes.csv).','[Sources receipt](evidence/sources.html) · [Temporal validation](evidence/validation.json) · [Export checks](evidence/export_validation.json) · [Per-pipe support](evidence/prediction_scope.csv)','']
    (DEST/f'report_{lang}.md').write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
