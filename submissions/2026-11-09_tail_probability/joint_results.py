from pathlib import Path
from decimal import Decimal as D
import json,csv,hashlib,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    records=json.loads((HERE/'official_submissions.json').read_text());assert all(r['status']=='complete' for r in records)
    baseline_path=ROOT/'output/2026-11-02_regional_optimization/G17/evidence/official_G17.csv'
    with baseline_path.open() as f:annual=list(csv.DictReader(f))
    cap=sum(D(r['CAPEX']) for r in annual);op=sum(D(r['OPEX']) for r in annual);assert D(250000000)-cap-op==D('79168928.88')
    (HERE/'official_G17_baseline.csv').write_bytes(baseline_path.read_bytes())
    confirm=pd.read_csv(HERE/'confirmation_2026.csv');sim=pd.read_csv(HERE/'G17_simulations.csv');pivot=sim.pivot(index=['world','draw'],columns='model',values='total_cost')
    rows=[]
    for record,name,simname in zip(records,['catboost_basic_bins3_raw','tail_probability_only'],['selected','tail_probability_only']):
        internal=confirm[confirm.geometry.eq('G17')&confirm.model.eq(name)].iloc[0];delta=pivot[simname]-pivot['G17_original']
        rows.append(dict(candidate=record['candidate'],model=name,high_prevented=int(internal.high_prevented),high_total=int(internal.high_count),high_recall=float(internal.high_recall),cost_capture=float(internal.cost_capture),
            public_score=float(record['public_score']),public_delta_vs_G17=float(record['delta_vs_G17']),official_capex=float(record['CAPEX']),official_opex=float(record['OPEX']),
            capex_delta_vs_G17=float(D(record['CAPEX'])-cap),opex_delta_vs_G17=float(D(record['OPEX'])-op),simulated_total_delta_vs_original_G17=float(delta.mean()),paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),adoption='not recommended'))
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'joint_internal_public_comparison.csv',index=False)
    # Uncertainty for the previously specified tail-probability control vs CMS001.
    p=pd.read_csv(ROOT/'data/pipes.csv');mid=(p[['GPS x1','GPS y1']].to_numpy()+p[['GPS x2','GPS y2']].to_numpy())/2;blocks=np.floor(mid/1000).astype(int)
    with np.load(HERE/'decisions/2026/G17_tail_probability_only.npz') as z:tail=z['prevented'];high=z['high'];actual=z['actual_cost']
    with np.load(HERE/'decisions/2026/G17_CMS001.npz') as z:cms=z['prevented']
    diff=tail.astype(int)-cms.astype(int);stats=pd.DataFrame(dict(block=[f'{x}_{y}' for x,y in blocks],high_delta=high*diff,high=high.astype(int),dollar_delta=actual*diff)).groupby('block').sum()
    rng=np.random.default_rng(202611091143);counts=rng.multinomial(len(stats),np.ones(len(stats))/len(stats),size=2000);draw=counts@stats.to_numpy();valid=draw[:,1]>0;ratio=draw[valid,0]/draw[valid,1]
    uncertainty=dict(high_count_delta=int((high*diff).sum()),high_recall_delta=float((high*diff).sum()/high.sum()),high_recall_spatial_bootstrap_95=[float(np.quantile(ratio,.025)),float(np.quantile(ratio,.975))],dollar_delta=float((actual*diff).sum()),dollar_spatial_bootstrap_95=[float(np.quantile(draw[:,2],.025)),float(np.quantile(draw[:,2],.975))])
    (HERE/'tail_control_uncertainty.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf8')
    names={'TP01':'高額確率＋条件別金額','TP02':'高額確率だけで順位付け'}
    lines=['# 内部評価とPublicスコアによる総合判断','', '**結論：今回の2案はどちらも元G17への置き換えを推奨しない。**','',
        '高額先回り件数の内部評価と、実際に固定CSVを提出したPublic総費用評価の両方を確認した。結果を見て分類器・閾値・倍率・工事順を再調整していない。','',
        '## 内部指標と公式結果','',
        '内部の基準はCMS001を同じG17円・同じ依存制約付き順序規則で並べ替えた対照。2026年の高額漏水36件のうち18件を初年度の1,000万ドルで先回りした。Publicの基準は実際に提出済みの元G17で79,168,928.88ドル。内部のCMS001並べ替え対照と、元G17の施工順は同一ではない。','',
        '| 方法 | 内部：高額先回り | 内部：修理費回避率 | Public | 元G17との差 | 総合判断 |','|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f'| {names[r["candidate"]]} | {r["high_prevented"]}/{r["high_total"]} | {r["cost_capture"]:.2%} | ${r["public_score"]:,.2f} | ${r["public_delta_vs_G17"]:+,.2f} | 採用しない |')
    lines+=['',
        '条件別金額モデル（TP01）は内部で16/36件となりCMS001の18/36件を下回り、Publicも約384万ドル悪化した。両方の評価が採用を支持しなかった。','',
        '高額確率だけの順位付け（TP02）は内部で19/36件と1件多く拾えたが、全修理費の回避額はCMS001並べ替え対照より144,004.01ドル少なかった。Publicも約425万ドル悪化した。高額案件の本数だけが少し増えても、工事費と残る漏水修理費を含めた全体の改善にはならなかった。','',
        f'この1件増の内部高額再現率差の1km空間ブロックbootstrap 95%区間は {uncertainty["high_recall_spatial_bootstrap_95"][0]:+.1%}〜{uncertainty["high_recall_spatial_bootstrap_95"][1]:+.1%}。少数案件・地域差の影響が大きく、確実な改善とは扱わない。','',
        '## Publicの費用内訳','', '| 方法 | 工事費 CAPEX | 漏水修理費 OPEX | 元G17からの工事費増 | 元G17からの修理費増 |','|---|---:|---:|---:|---:|',
        f'| 元G17 | ${cap:,.2f} | ${op:,.2f} | — | — |']
    for r in rows:lines.append(f'| {names[r["candidate"]]} | ${r["official_capex"]:,.2f} | ${r["official_opex"]:,.2f} | ${r["capex_delta_vs_G17"]:+,.2f} | ${r["opex_delta_vs_G17"]:+,.2f} |')
    lines+=['','公式スコアは `250,000,000 − CAPEX − OPEX`。25年分の公式原表からDecimalでセント単位まで計算した。円の形状が同じでも、施工順と自然漏水後のPU状態が変わるため、実際の工事費も変化する。','',
        '## 内部25年シミュレーション','', '| 方法 | 元G17に対する仮想総費用の差 |','|---|---:|']
    for r in rows:lines.append(f'| {names[r["candidate"]]} | ${r["simulated_total_delta_vs_original_G17"]:+,.2f} |')
    lines+=['','こちらも2案とも総費用が増え、元G17への変更を支持しなかった。2種類×128乱数の仮想将来であり、実測・公式Public・Privateの保証ではない。','',
        '## 過剰評価を避けるための条件','',
        '- TP01は2024〜2025年の共通円による内部結果だけで選び、2026年評価前に固定した。TP02は事前に比較へ入れた高額確率のみの対照で、2026年の改善を見て今回Publicも確認した探索案。選択から独立した最終検証済みモデルとは呼ばない。',
        '- 2026年は以前の研究で参照済み。G17円の過去適用も既存の2027年用円を固定した参考比較。モデルの特徴量自体は年初情報だけを使用した。',
        '- Publicは今回の実際の提出結果。繰り返し試すほどPublicへの選択過剰適合が起き得るため、この結果から新しい設定へ微調整して再提出していない。Privateの優位は未確認。',
        '- 高額案件の内部基準は学習全期間の99%分位点以上。修理費の上位1%という記述分析と区別する。','',
        '## 提出物と記録','']
    for r in records:
        rel=r['csv'].replace('\\','/');lines += [f'- {r["candidate"]}：[{names[r["candidate"]]}]({rel})、提出ID `{r["submission_id"]}`、Public **${D(r["public_score"]):,.2f}**。',f'  - SHA256：`{r["sha256"]}`',f'  - [公式25年内訳]({r["official_annual_breakdown"]}) ／ [サイト原表]({r["result_url"]})']
    lines+=['','[学習・検証レポート](REPORT.ja.md) ／ [事前プロトコル](PROTOCOL.md) ／ [Public確認前の固定記録](locked_before_public.json) ／ [公式提出記録](official_submissions.json)。','',
        '共有Google Docは編集していない。以前の実験・提出は削除も変更もしていない。2案の提出CSV、SHA256、Public、公式年間内訳、内部評価を同じGitHub記録に保存する。']
    (HERE/'JOINT_DECISION.ja.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    report=HERE/'REPORT.ja.md';text=report.read_text(encoding='utf8')
    notice='> 公式提出の追記（2026-11-09）：後続のユーザー依頼により2案を提出し、内部指標とPublicを総合評価した。[最新の総合判断](JOINT_DECISION.ja.md)。以下は提出前に完了した学習・内部評価の記録。\n\n'
    if '> 公式提出の追記' not in text:report.write_text(notice+text,encoding='utf8')
    for f in ['completion.json','ADOPTION_DECISION.json']:
        path=HERE/f;obj=json.loads(path.read_text());obj.update(official_submission=True,official_public_scores={r['candidate']:r['public_score'] for r in records},joint_adoption_recommended=False,joint_decision='JOINT_DECISION.ja.md');path.write_text(json.dumps(obj,indent=2),encoding='utf8')
    print(frame.to_string(index=False));print(json.dumps(uncertainty))

if __name__=='__main__':main()
