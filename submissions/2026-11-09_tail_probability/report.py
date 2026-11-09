from pathlib import Path
import sys,json,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import tail_model as tm
ROOT=tm.ROOT

def sensitivities(selected):
    p=pd.read_csv(ROOT/'data/pipes.csv');mid=(p[['GPS x1','GPS y1']].to_numpy()+p[['GPS x2','GPS y2']].to_numpy())/2
    blocks=np.floor(mid/1000).astype(int);block=[f'{x}_{y}' for x,y in blocks];rows=[]
    for geo in ['common','G17']:
        with np.load(HERE/'decisions/2026'/f'{geo}_{selected}.npz') as z:actual=z['actual_cost'];high=z['high'];saved=z['prevented']
        top=np.argsort(-actual);event=actual>0
        decisions=pd.DataFrame(dict(pipe_id=p['Pipe ID'][event].to_numpy(),actual_cost=actual[event],high=high[event],selected_prevented=saved[event],x=mid[event,0],y=mid[event,1]))
        for control in ['CMS001','standard_LGBM','tail_probability_only']:
            with np.load(HERE/'decisions/2026'/f'{geo}_{control}.npz') as z:baseline=z['prevented']
            diff=saved.astype(float)-baseline.astype(float)
            frame=pd.DataFrame(dict(block=block,delta_dollars=actual*diff,delta_high=high*diff,high=high.astype(float)))
            grouped=frame.groupby('block')[['delta_dollars','delta_high','high']].sum();rng=np.random.default_rng(202611091139)
            count=rng.multinomial(len(grouped),np.full(len(grouped),1/len(grouped)),size=2000);boot=count@grouped.to_numpy();valid=boot[:,2]>0;ratio=boot[valid,1]/boot[valid,2]
            row=dict(geometry=geo,control=control,selected_minus_control_high_count=int((high*diff).sum()),selected_minus_control_high_recall=float((high*diff).sum()/high.sum()),
                high_recall_bootstrap_low=float(np.quantile(ratio,.025)),high_recall_bootstrap_high=float(np.quantile(ratio,.975)),
                selected_minus_control_dollars=float((actual*diff).sum()),dollars_bootstrap_low=float(np.quantile(boot[:,0],.025)),dollars_bootstrap_high=float(np.quantile(boot[:,0],.975)),
                excluding_largest1_delta_dollars=float((actual*diff).sum()-(actual*diff)[top[:1]].sum()),excluding_largest5_delta_dollars=float((actual*diff).sum()-(actual*diff)[top[:5]].sum()),spatial_blocks=len(grouped),bootstrap_draws=2000)
            rows.append(row);decisions[control+'_prevented']=baseline[event]
        decisions.to_csv(HERE/f'2026_{geo}_event_decisions.csv',index=False)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'confirmation_2026_sensitivity.csv',index=False);return frame

def checks():
    rows=[]
    for year in tm.YEARS:
        for name in tm.labels():
            path=HERE/'predictions'/str(year)/(name+'.npz')
            with np.load(path) as z:
                assert len(z['pipe_id'])==len(np.unique(z['pipe_id']))
                assert (z['predicted_cost']>0).all() and np.isfinite(z['predicted_cost']).all()
                assert ((z['p_tail95']>=0)&(z['p_tail95']<=1)).all()
                if 'bins3' in name:assert ((z['p_tail99']>=0)&(z['p_tail99']<=z['p_tail95']+1e-12)).all()
                rows.append(dict(year=year,model=name,n=len(z['pipe_id']),sha256=tm.sha(path)))
    pd.DataFrame(rows).to_csv(HERE/'verified_prediction_inventory.csv',index=False);assert len(rows)==80
    assert len(json.loads((HERE/'model_reload_verification.json').read_text()))==16

def main():
    checks();lock=json.loads((HERE/'locked_before_2026.json').read_text());selected=lock['model'];sens=sensitivities(selected)
    metrics=pd.read_csv(HERE/'annual_metrics.csv');confirm=pd.read_csv(HERE/'confirmation_2026.csv');dep=json.loads((HERE/'deployment.json').read_text())
    common=confirm[confirm.geometry.eq('common')].set_index('model');g17=confirm[confirm.geometry.eq('G17')].set_index('model')
    scenario=pd.read_csv(HERE/'G17_simulation_summary.csv').set_index('model')
    good=bool(common.loc[selected,'high_recall']>common.loc['CMS001','high_recall'] and common.loc[selected,'cost_capture']>=common.loc['CMS001','cost_capture'] and g17.loc[selected,'high_recall']>g17.loc['CMS001','high_recall'] and scenario.loc['CMS001','selected_minus_model']<0)
    decision=dict(candidate=selected,adoption_recommended=good,reason='No consistent confirmed improvement over CMS001' if not good else 'Improved internal measures; official performance remains unverified',
        selection_years=[2024,2025],confirmation_year=2026,common_high_prevented_selected=int(common.loc[selected,'high_prevented']),common_high_prevented_CMS001=int(common.loc['CMS001','high_prevented']),high_total=int(common.loc[selected,'high_count']),
        G17_high_prevented_selected=int(g17.loc[selected,'high_prevented']),G17_high_prevented_CMS001=int(g17.loc['CMS001','high_prevented']),
        G17_selected_minus_CMS001_simulated_total=float(scenario.loc['CMS001','selected_minus_model']),G17_selected_minus_original_simulated_total=float(scenario.loc['G17_original','selected_minus_model']),
        tail_only_is_exploratory_control_not_selected=True,official_submission=False,public_score=None)
    tm.dump(HERE/'ADOPTION_DECISION.json',decision)
    lines=['# 高額確率＋条件別金額：年初情報による比較','',
        '**結論：今回の条件別金額モデルをCMS001へ置き換える根拠は得られなかった。**','',
        f'2024〜2025年の共通円・同じ1,000万ドル予算で選んだ候補は `{selected}`。CatBoost、基本属性、通常／上位5%／上位1%の3群、確率の事後校正なし。各群の材質×地表面別金額平均を縮約して予測した。2026年を見て候補を変更していない。','',
        f'2026年の共通円では高額案件を候補が{decision["common_high_prevented_selected"]}/{decision["high_total"]}件、CMS001が{decision["common_high_prevented_CMS001"]}/{decision["high_total"]}件先回りできた。G17円の参考比較でも候補{decision["G17_high_prevented_selected"]}件、CMS001 {decision["G17_high_prevented_CMS001"]}件となり、候補の優位は確認できなかった。','',
        '## 2026年の同一予算比較','',
        '高額案件は、2019〜2025年の実費99%分位点以上だった2026年の漏水36件。今回の判定閾値は約40.76万ドル。前の非対称損失実験で示した「2026年の実費上位1%」31件とは定義が異なる。下表の件数同士は同じ36件で比較している。','',
        '| 円 | 方法 | 高額先回り | 高額再現率 | 全修理費の回避率 | 実測修理費の回避額 |','|---|---|---:|---:|---:|---:|']
    controls=[selected,'CMS001','standard_LGBM','asymmetric_LGBM_2p4','tail_probability_only','F168_only']
    for geo in ['common','G17']:
        for name in controls:
            row=confirm[confirm.geometry.eq(geo)&confirm.model.eq(name)].iloc[0]
            lines.append(f'| {geo} | {name} | {int(row.high_prevented)}/{int(row.high_count)} | {row.high_recall:.2%} | {row.cost_capture:.2%} | ${row.prevented_dollars:,.2f} |')
    lines+=['','G17円は2026年までの情報で作られた既存円を過去へ遡及した固定条件の参考評価である。2027年時点のPU状態も過去年とは異なり、元G17順の過去年成績から実際の元G17の性能を評価してはいけない。候補選択にはこの円を使わず、全original iron一本ずつの共通円を使った。','',
        '## 高額確率だけの順位付けには手掛かりが残る','',
        '対照の `tail_probability_only` は、年初情報で学習したLightGBMの上位1%金額確率に、F168の翌年漏水確率を掛けて工事費で割る。条件別金額の推定値を掛けない。共通円では21/36件でCMS001の19/36件を上回った。G17円では19/36件でCMS001の18/36件を上回ったが、回避額はCMS001より約14.4万ドル少なかった。','',
        'この対照は事前に比較対象へ入れたが、候補選択の対象16モデルには含めていない。2026年を見て有利な対照へ乗り換え、「独立検証で改善した最終モデル」と主張しない。次の検討材料として、対照のCSVと結果も保存した。','',
        '## 開発期間と選択','', '| 年 | 方法 | 高額先回り | 全修理費回避率 |','|---|---|---:|---:|']
    for year in [2024,2025]:
        for name in [selected,'CMS001','standard_LGBM']:
            row=metrics[metrics.geometry.eq('common')&metrics.year.eq(year)&metrics.model.eq(name)].iloc[0]
            lines.append(f'| {year} | {name} | {int(row.high_prevented)}/{int(row.high_count)} | {row.cost_capture:.2%} |')
    lines+=['','主指標は高額再現率の年平均、同点なら実測修理費回避率。開発年の高額案件は8件・16件と少ない。2026年に性能が維持されなかった。各年の前年までだけで選ぶ評価は `forward_selection.csv`、16候補全ては `annual_metrics.csv` に保存した。','',
        '## 25年の参考シミュレーション','', '| 円 | 比較先 | 候補−比較先の総費用 |','|---|---|---:|']
    for geo in ['common','G17']:
        for r in pd.read_csv(HERE/f'{geo}_simulation_summary.csv').itertuples():lines.append(f'| {geo} | {r.model} | ${r.selected_minus_model:+,.2f} |')
    lines+=['','2種類の材質・管齢別漏水率推定×128共通乱数＝256仮想将来でペア比較。修理金額分布も過去から推定。高い費用ほど悪い。実測でも公式Publicでもなく、分布の推定誤差を保証しない。G17円では候補がCMS001より約290万ドル、元G17より約575万ドル高い結果で、採用判断を支持しなかった。','',
        '## 実装と検証','',
        '- 基本属性／局所履歴、LightGBM／CatBoost、2群／3群、未校正／前年の時間外予測によるPlatt校正の16候補。2023〜2027年の計80組の予測を保存。',
        '- 年初管齢・前年末までの履歴だけを使用。旧解析の漏水時点管齢、漏水月・時刻・日付は投入しない。`input_audit.json` に生データハッシュ、年初管齢、既存の未来改変不変性の検証結果との照合を保存。',
        '- 群の閾値は校正年より前のデータだけから計算し、前年での校正と全過去再学習で同じ閾値を維持。p(99%)<=p(95%)を整合させ、群確率の和を1にする。高額実費は削除・上限処理していない。',
        '- 16学習済みモデルを再読込し、全27,240本の2027年予測の一致を確認。`model_reload_verification.json`。',
        '- 2026年の地域1kmブロックbootstrapと最大1件／5件除外の感度分析を `confirmation_2026_sensitivity.csv` に保存。新しいテストではなく記述的な不確実性評価。',
        '- 2019〜2026年は過去研究でも利用済み。2026年は今回の候補選択から留保した年で、完全未使用の最終テストではない。F168も既存の選択済みモデルを共通利用した。',
        '- 旧96.35%は漏水案件内の上位10%選択の結果であり、今回の全管・施工予算内の捕捉率とは異なる。年初情報へ修正後の条件付き分類指標は `conditional_classifier_metrics.csv` に保存。','',
        '## CSVと利用状態','',
        '- `submission_files/selected_G17.csv`：開発期間で選ばれた条件別金額モデル。採用非推奨の検討用。',
        '- `submission_files/tail_probability_only_G17.csv`：高額確率だけで順位付けする対照。選択済みの勝者ではない。',
        '- 各共通円版、CMS001対照、通常LightGBM対照も保存。書き出し後に全original ironカバー、年間1,000万ドルと繰越、修正式工事費、2051年までの完了を再検査した。','',
        '**公式提出・GitHub公開・共有Google Doc編集は行っていない。Publicスコアは未取得。**','',
        '## 再現','', '```powershell', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/tail_model.py audit', '# year=2023,2024,2025,2026,2027', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/tail_model.py train --year 2027', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/tail_model.py verify', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/study.py evaluate', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/study.py deploy', './.venv-eda/Scripts/python.exe output/2026-11-09_tail_probability/report.py', '```']
    (HERE/'REPORT.ja.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    doc=['Conditional tail-probability repair-cost models; Takito; 2026-11-09.','Selected on 2024–2025 high-cost prevention; 2026 confirmation. No consistent benefit over CMS001.','']
    for met in ['mae','rmse']:
        doc += [f'| Group | member | Model | 2023 {met.upper()}($) | 2024 | 2025 | 2026 | Mean |','|---|---|---|---:|---:|---:|---:|---:|']
        for name in [selected,'CMS001','standard_LGBM']:
            q=metrics[metrics.geometry.eq('common')&metrics.model.eq(name)].set_index('year');vals=[f'{q.loc[y,met]:,.2f}' if y in q.index else 'N/A' for y in [2023,2024,2025,2026]]
            doc.append(f'| Shana | Takito | {name} | '+' | '.join(vals)+f' | {q[met].mean():,.2f} |')
        doc.append('')
    (HERE/'RESULTS_FOR_DOC.md').write_text('\n'.join(doc),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[selected,'CMS001','standard_LGBM','tail_probability_only'];titles=['Selected mixture','CMS001','Standard LightGBM','Tail probability only'];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for axis,geo in zip(axes,['common','G17']):
        subset=confirm[confirm.geometry.eq(geo)].set_index('model').loc[chosen];axis.bar(titles,subset.high_recall*100,color=['#c95b45','#386cb0','#5b8e7d','#c7a44a']);axis.set_ylim(0,100);axis.set_title(f'2026: {geo} circles');axis.set_ylabel('High-cost cases prevented (%)');axis.tick_params(axis='x',labelrotation=20)
        for i,row in enumerate(subset.itertuples()):axis.text(i,row.high_recall*100+2,f'{row.high_prevented}/{row.high_count}',ha='center')
        axis.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'confirmation_2026.png',dpi=180);plt.close(fig)
    files=sorted((HERE/'submission_files').glob('*'));manifest={str(p.relative_to(HERE)):tm.sha(p) for p in files};manifest.update({p.name:tm.sha(p) for p in HERE.glob('*.py')});tm.dump(HERE/'SHA256.json',manifest)
    with zipfile.ZipFile(HERE/'candidate_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[HERE/'REPORT.ja.md',HERE/'ADOPTION_DECISION.json',HERE/'SHA256.json']:archive.write(p,str(p.relative_to(HERE)))
    tm.dump(HERE/'completion.json',dict(complete=True,candidate_models=16,forecast_years=5,candidate=selected,adoption_recommended=good,official_submission=False,validated_csvs=len(list((HERE/'submission_files').glob('*.csv'))),report='REPORT.ja.md'))
    print('REPORT COMPLETE',json.dumps(decision),flush=True)

if __name__=='__main__':main()
