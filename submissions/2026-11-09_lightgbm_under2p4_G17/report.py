from pathlib import Path
import sys,json,zipfile,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate as ev
a=ev.a;ROOT=a.ROOT

def verify_models():
    import joblib
    q=pd.read_parquet(HERE/'features/2027.parquet');records=[]
    for path in sorted((HERE/'fits/2027').glob('*.joblib')):
        bundle=joblib.load(path);name=bundle['name'];b=q
        if name in ['xgboost','catboost','lightgbm']:
            z=b[bundle['features']].copy()
            for c,levels in bundle['levels'].items():
                z[c]=z[c].fillna('__unknown__').astype(str) if name=='catboost' else pd.Categorical(z[c].where(z[c].isin(levels)),categories=levels)
            if name=='catboost':
                from catboost import CatBoostRegressor
                model=CatBoostRegressor();model.load_model(bundle['model_path'])
            else:model=bundle['model']
            pred=model.predict(z)+bundle['center']
        else:
            z=bundle['transform'].transform(b)
            pred=np.column_stack([np.ones(len(z)),z])@bundle['coef']*bundle['scale'] if name=='ridge' else bundle['model'].predict(z)
        saved=np.load(path.with_suffix('.npy'))[:len(b)]
        np.testing.assert_allclose(np.maximum(pred,0),saved,rtol=1e-10,atol=1e-5)
        records.append(dict(path=path.name,reloaded_prediction_match=True,n=len(b)))
    a.dump(HERE/'model_reload_verification.json',records)

def verify_inventory():
    records=[]
    for year in a.YEARS:
        q=pd.read_parquet(HERE/'features'/f'{year}.parquet')
        for name in a.MODELS:
            for k in a.RATIOS:
                path=HERE/'fits'/str(year)/f'{name}_k{k:.1f}.npy';pred=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
                assert len(pred)==len(q) and np.isfinite(pred).all() and (pred>=0).all()
                assert meta['train_end']<year and meta['k']==k and meta['name']==name
                records.append(dict(year=year,model=name,k=k,n=len(pred),sha256=a.sha(path),train_end=meta['train_end']))
    assert len(records)==264
    pd.DataFrame(records).to_csv(HERE/'verified_prediction_inventory.csv',index=False)

def event_sensitivity():
    # Spatial block resampling is descriptive uncertainty, not a new test set.
    p,h,g,rev=ev.load();sel=json.loads((HERE/'selection_result.json').read_text());year=2026
    q=pd.read_parquet(HERE/'features/2026.parquet');risk=ev.probability(p,year);state=ev.initial(p,h,year)
    observed=p['Pipe ID'].map(h.loc[h.year.eq(year)].set_index('Pipe ID').Cost).fillna(0).to_numpy()
    event=observed>0;amounts={};plans={}
    for label,name,k in [('selected',sel['model'],sel['k']),('same_model_k1',sel['model'],1.),('best_symmetric',sel['best_symmetric_model'],1.),('CMS001','CMS001',1.)]:
        if name=='CMS001':cost,_=ev.cms(p,h,year)
        else:cost,_=ev.fullcost(p,h,year,q.pipe_id,np.load(HERE/'fits/2026'/f'{name}_k{k:.1f}.npy'))
        seq=ev.order(g,rev,state,risk,cost);live=state.copy();budget=1_000_000_000;replace_year=np.full(len(p),2051,int)
        for j in seq:
            ids=g[9][g[8][j]:g[8][j+1]];new=ids[~live[ids]]
            if not len(new):continue
            cents=int(np.floor((g[10][j]+200*g[2][new].sum())*100+.5))
            if cents>budget:break
            budget-=cents;live[new]=True;replace_year[new]=2026
        amounts[label]=observed*live*~state;plans[label]=live&~state
    frame=pd.DataFrame(dict(pipe_id=p['Pipe ID'][event].to_numpy(),actual_cost=observed[event],x=((g[0]+g[1])/2)[event,0],y=((g[0]+g[1])/2)[event,1],**{label:v[event] for label,v in amounts.items()}))
    frame.to_csv(HERE/'confirmation_2026_event_decisions.csv',index=False)
    frame['block']=(np.floor(frame.x/1000).astype(int).astype(str)+'_'+np.floor(frame.y/1000).astype(int).astype(str))
    grouped=frame.groupby('block')[list(amounts)].sum();rng=np.random.default_rng(20261109);weights=rng.multinomial(len(grouped),np.full(len(grouped),1/len(grouped)),size=2000)
    boots=weights@grouped.to_numpy();columns=list(grouped.columns);rows=[]
    top=frame.actual_cost.nlargest(5).index
    for label in amounts:
        delta=boots[:,columns.index('selected')]-boots[:,columns.index(label)]
        diff=frame['selected']-frame[label]
        rows.append(dict(control=label,selected_minus_control=float(diff.sum()),spatial_bootstrap_low=float(np.quantile(delta,.025)),spatial_bootstrap_high=float(np.quantile(delta,.975)),excluding_largest_one_event=float(diff.drop(frame.actual_cost.idxmax()).sum()),excluding_largest_five_events=float(diff.drop(top).sum()),spatial_blocks=len(grouped)))
    pd.DataFrame(rows).to_csv(HERE/'confirmation_2026_sensitivity.csv',index=False)

def main():
    verify_inventory();verify_models();event_sensitivity()
    selection=json.loads((HERE/'selection_result.json').read_text());metrics=pd.read_csv(HERE/'annual_metrics.csv');dev=pd.read_csv(HERE/'development_2023_2025.csv');confirm=pd.read_csv(HERE/'confirmation_2026.csv');name=selection['model'];k=selection['k']
    candidate=confirm[confirm.model.eq(name)&confirm.k.eq(k)].iloc[0];symmetric=confirm[confirm.model.eq(name)&confirm.k.eq(1)].iloc[0]
    forward=pd.read_csv(HERE/'forward_selection.csv').groupby('selector').cost_capture.mean()
    future=pd.read_csv(HERE/'simulated_summary_2027.csv').set_index('model')
    decision=dict(candidate_model=name,candidate_penalty=k,adoption_status='hold; consistent benefit for the most expensive cases not established',
        confirmation_capture_delta_percentage_points=float(100*(candidate.cost_capture-symmetric.cost_capture)),confirmation_prevented_dollars_delta=float(candidate.prevented_dollars-symmetric.prevented_dollars),
        confirmation_top1pct_prevented_candidate=int(candidate.tail_prevented_count),confirmation_top1pct_prevented_symmetric=int(symmetric.tail_prevented_count),confirmation_top1pct_total=int(candidate.tail_actual_count),
        forward_asymmetric_capture=float(forward['asymmetric']),forward_symmetric_capture=float(forward['symmetric']),
        simulated_2027_candidate_minus_CMS001=float(future.loc['CMS001','selected_minus_model']),official_submission=False)
    a.dump(HERE/'ADOPTION_DECISION.json',decision)
    best=dev[dev.k.gt(1)].sort_values(['cost_capture','k'],ascending=[False,True]).drop_duplicates('model');best.to_csv(HERE/'best_penalty_by_model.csv',index=False)
    # Common-year aggregate is always distinguished from development selection.
    metrics.groupby(['model','k']).agg(year_count=('year','nunique'),mean_cost_capture=('cost_capture','mean'),mean_rmse=('rmse','mean'),mean_mae=('mae','mean')).reset_index().to_csv(HERE/'all_year_average.csv',index=False)
    lines=['# 金額予測：過小評価ペナルティの比較','',f'**試験候補は{name}・{k:.1f}倍。ただし、現行モデルからの採用変更は保留。** 2023〜2025年の同一予算での実測修理費回避率により選択し、2026年を候補選択から除外した。','',
        f'2026年の通常設定に対する改善は${decision["confirmation_prevented_dollars_delta"]:,.2f}（+{decision["confirmation_capture_delta_percentage_points"]:.3f}ポイント）。高額上位1％の先回り本数は両方とも{int(candidate.tail_prevented_count)}/{int(candidate.tail_actual_count)}本で増えなかった。',
        f'各年の前年までの結果だけで選ぶforward比較の平均回避率は、非対称損失{forward["asymmetric"]:.2%}、通常損失{forward["symmetric"]:.2%}。倍率を重くすれば一貫して良くなる証拠は得られなかった。',
        f'2027年起点の共通円・25年参考シミュレーションでは、候補の総費用はCMS001より${decision["simulated_2027_candidate_minus_CMS001"]:+,.2f}。提出ファイルは検討用の候補として作成し、良いモデルが確立したとの主張や公式送信は行わない。','',
        '学習時の非対称二乗損失を変更した。金額予測の事後一律乗算ではない。過小評価側に重みを付けた点予測は上方に寄るため、通常の条件付き平均金額とは異なる。漏水確率を掛けた値は優先順位用の指標であり、不偏な将来費用の見積りではない。','',
        '6モデル×11倍率×4年＝264構成。k=1.0を対照、1.2〜3.0は0.2刻み。XGBoost/CatBoost/LightGBMの年別設定を再利用し、Random Forest/Extra Trees/Ridgeは今回固定の保守的設定。森林は最大4回の重み更新による近似で、全モデルの厳密最適化ではない。','',
        '## 各モデルで開発期間に選ばれた倍率','', '| モデル | 倍率 | 2023〜2025平均回避率 | 同モデル1倍 | 差（pt） |','|---|---:|---:|---:|---:|']
    for row in best.itertuples():
        base=dev[(dev.model.eq(row.model))&dev.k.eq(1)].iloc[0]
        lines.append(f'| {row.model} | {row.k:.1f} | {row.cost_capture:.2%} | {base.cost_capture:.2%} | {100*(row.cost_capture-base.cost_capture):+.2f} |')
    lines+=['','## 2026年の確認','', '| モデル | 倍率 | 実測修理費の回避額 | 回避率 | RMSE |','|---|---:|---:|---:|---:|']
    for row in confirm.itertuples(): lines.append(f'| {row.model} | {row.k:.1f} | ${row.prevented_dollars:,.2f} | {row.cost_capture:.2%} | ${row.rmse:,.2f} |')
    lines+=['',('候補は2026年の実測回避額で全対照を上回った。' if selection['confirmation_beats_all_controls'] else '候補は2026年の実測回避額で全対照を上回らなかった。最良の倍率を選んだことと、翌年の優位が確認できたことは異なる。'),'',
        '全管を順位付けし、年間1,000万ドルの交換予算に収まる工事を先に実行した場合に、その年のどの漏水を防げたかを集計した。漏水しない管への支出と、円内の付随交換費も含む。初年度の施工が同年の漏水より先に行われる年間モデルの評価。交換円は全original iron一本ごとの共通円。最適化済みG17円の順位変更は別の参考実験。','',
        '## 25年の参考シミュレーション','', '過去だけから推定する材質・管齢別漏水率と材質・地表面別の修理金額分布を用い、2条件×128乱数でペア比較した。2026年のシミュレーションは初年度のみ実測、2027年以降は全て仮想将来。分布・漏水率の推定誤差やPrivateを保証しない。数値は総費用で、低い方が良い。','', '| 円配置 | モデル | 平均25年総費用 | 候補−モデル |','|---|---|---:|---:|']
    for prefix,label in [('', '共通円'),('G17_','G17円')]:
        for row in pd.read_csv(HERE/f'{prefix}simulated_summary_2027.csv').itertuples():lines.append(f'| {label} | {row.model} | ${row.mean_total_cost:,.2f} | ${row.selected_minus_model:+,.2f} |')
    lines+=['','## 提出用ファイル','',
        '- `submission_files/selected_common_circles.csv`：比較実験と同じ共通円。',
        '- `submission_files/selected_G17_circles.csv`：選んだ金額モデルで既存G17円を並べ替えた候補。重複する円の前後依存を保ち、各工事の交換対象を変えない。',
        '- 各CSVを再読込し、全original ironの幾何カバー、年間1,000万ドル＋繰越、修正済み工事費、2051年までの完了を検証した。',
        '- 公式送信はしていない。Public/Privateスコアは未取得。既存G17/G25の優位を上回ったと主張しない。','',
        '## 限界と検証記録','',
        '- 過去研究で2023〜2026年を利用済み。今回の2026留保は今回の倍率選択からの留保であり、完全未使用の最終テストではない。F168も既存の開発済み予測を共通利用した。',
        '- 実際の失敗日時を25年先まで観測した評価ではない。金額モデルだけでは漏水時期の誤りを直せない。',
        '- `forward_selection.csv`：その年より前の結果だけでモデル・倍率を選択した比較。',
        '- `confirmation_2026_sensitivity.csv`：1km空間ブロックbootstrapと最大1件/5件を除く感度分析。これは再標本化による記述で、新しいテストではない。',
        '- `CMS001_reconstruction_audit.json`：過去年の全管向けCMS001再計算と元OOFとの差。保存済みの年別設定を利用した再計算対照であり、元のイベント向けOOFとの完全一致は主張しない。',
        '- 予測が用意できない各年8本には、当該年より前の修理実績に基づく共通の平滑化地表面平均を利用。ゼロで埋めない。',
        '- `input_audit.json`、`loss_verification.json`、`model_reload_verification.json`、各`.validation.json`に検証を保存。',
        '- GitHub mainのデータと施工検証コードがローカルと一致することを確認した。一部のローカル年別設定は現在のmainの同じパスでは取得できず、ローカル成果物のハッシュで記録した。',
        '- 共有Google Docの現行内容と年別MAE/RMSE表形式を確認。冒頭の旧工事費式より、ユーザー指定の修正式 `100 + rate * area**0.85 + 200 * unreplaced_length` を優先した。共有Docは編集していない。','',
        '## 再実行','', '```powershell', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/run.py checks', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/run.py prepare', '# 各modelについて run.py train --model MODEL', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/evaluate.py evaluate', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/evaluate.py deploy', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/g17.py', './.venv-eda/Scripts/python.exe output/2026-11-09_asymmetric_cost/report.py', '```']
    (HERE/'REPORT.ja.md').write_text('\n'.join(lines),encoding='utf8')
    # Shared document format, prepared locally only.
    doc=['Cost prediction under asymmetric underestimation penalties; Takito; 2026-11-09.','Selected ratio using 2023–2025 prevention only; 2026 reserved for confirmation within this experiment. Historical years were used in previous research.','']
    for metric in ['mae','rmse']:
        doc += [f'| Group | member | Model | 2023 {metric.upper()}($) | 2024 | 2025 | 2026 | Mean |','|---|---|---|---:|---:|---:|---:|---:|']
        for row in best.itertuples():
            q=metrics[metrics.model.eq(row.model)&metrics.k.eq(row.k)].sort_values('year');doc.append(f'| Shana | Takito | {row.model} k={row.k:.1f} | '+' | '.join(f'{v:,.2f}' for v in q[metric])+f' | {q[metric].mean():,.2f} |')
        doc.append('')
    (HERE/'RESULTS_FOR_DOC.md').write_text('\n'.join(doc),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for model in a.MODELS:
        q=dev[dev.model.eq(model)].sort_values('k');axes[0].plot(q.k,q.cost_capture*100,marker='o',ms=3,label=model)
        q=metrics[metrics.model.eq(model)&metrics.year.eq(2026)].sort_values('k');axes[1].plot(q.k,q.cost_capture*100,marker='o',ms=3,label=model)
    for axis,title in zip(axes,['Development: mean 2023–2025','Confirmation: 2026 (selection frozen)']):
        axis.set_title(title);axis.set_xlabel('Underestimation penalty multiplier');axis.grid(alpha=.2)
    axes[0].set_ylabel('Observed repair dollars prevented in year 1 (%)');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'penalty_comparison.png',dpi=180);plt.close(fig)
    files=sorted((HERE/'submission_files').glob('*'));manifest={str(path.relative_to(HERE)):a.sha(path) for path in files}
    manifest.update({str(path.relative_to(HERE)):a.sha(path) for path in HERE.glob('*.py')})
    a.dump(HERE/'SHA256.json',manifest)
    with zipfile.ZipFile(HERE/'submission_package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[HERE/'REPORT.ja.md',HERE/'SHA256.json']:z.write(path,str(path.relative_to(HERE)))
    a.dump(HERE/'completion.json',dict(complete=True,candidates=264,submission_created=True,official_submission=False,selection=selection,report='REPORT.ja.md',validated_csv_count=len(list((HERE/'submission_files').glob('*.csv')))))
    print('REPORT AND VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
