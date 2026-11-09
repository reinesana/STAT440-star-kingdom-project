"""Record official annual results and combine with the frozen internal checks."""
from pathlib import Path
from decimal import Decimal
import csv,json,sys,hashlib
import pandas as pd

HERE=Path(__file__).resolve().parent
REFERENCE=Decimal('79168928.88')
LABELS={'C01':'中心移動','C02':'分割','C03':'中心移動＋分割'}

def record(tag,url):
    path=HERE/f'{tag}_official_annual.txt'
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==25 and [int(r['YEAR']) for r in rows]==list(range(1,26))
    cap=sum(Decimal(r['CAPEX']) for r in rows);op=sum(Decimal(r['OPEX']) for r in rows)
    score=Decimal(250000000)-cap-op
    records=json.loads((HERE/'official_submissions.json').read_text())
    item=next(r for r in records if r['candidate']==tag)
    assert hashlib.sha256((HERE/item['csv']).read_bytes()).hexdigest()==item['sha256']
    item.update(status='complete',public_score=str(score),CAPEX=str(cap),OPEX=str(op),
                delta_vs_G17=str(score-REFERENCE),result_url=url,
                annual_breakdown=path.name,score_source='Decimal sum of official 25-year annual CAPEX and OPEX')
    (HERE/'official_submissions.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(json.dumps(item))

def report():
    records=json.loads((HERE/'official_submissions.json').read_text())
    assert all(r['status']=='complete' for r in records)
    gen=json.loads((HERE/'generation.json').read_text())
    internal=pd.read_csv(HERE/'confirmation_summary.csv').set_index('arm')
    rows=[]
    for r in records:
        v=internal.loc[r['candidate']]
        rows.append(dict(candidate=r['candidate'],method=LABELS[r['candidate']],public_score=r['public_score'],
            public_gain=r['delta_vs_G17'],internal_mean_gain=v.mean_gain,
            internal_worst_setting_gain=v.worst_setting_gain,positive_settings=int(v.positive_settings),
            internal_gate_pass=bool(v.gate_pass),all_feasible=bool(v.all_feasible),
            nominal_no_future_leak_saving=gen['candidates'][r['candidate']]['nominal_saving'],
            sha256=r['sha256'],submission_id=r['submission_id']))
    pd.DataFrame(rows).to_csv(HERE/'comparison.csv',index=False)
    best=max(rows,key=lambda r:Decimal(r['public_score']))
    decision=json.loads((HERE/'decision.json').read_text())
    decision.update(official_submission=True,public_scores={r['candidate']:r['public_score'] for r in records},
        recommended=best['candidate'] if best['internal_gate_pass'] and Decimal(best['public_gain'])>0 else 'G17',
        reason='Both official Public gain and fresh paired internal validation support this candidate; Private superiority remains unverified.')
    (HERE/'decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['G17の円の中心・半径・分割探索：公式提出と内部評価（2026-11-09）','',
        f'推奨：{decision["recommended"]}。公式Publicと内部評価を別々に確認して判断した。Privateでの優位は未確認。',
        f'基準：G17 Public ${REFERENCE:,.2f}。提出済みCSVのGitHub版とSHA256の一致を確認。','',
        '候補 / Public score / G17との差 / 内部平均改善 / 内部の最悪条件平均改善 / 改善条件数']
    for r in rows:
        lines.append(f'{r["candidate"]} {r["method"]} / ${Decimal(r["public_score"]):,.2f} / ${Decimal(r["public_gain"]):+,.2f} / ${r["internal_mean_gain"]:+,.2f} / ${r["internal_worst_setting_gain"]:+,.2f} / {r["positive_settings"]}/24')
    lines+=['','方法：',
        'C01：高い地表または予定外の未交換管の端点を円から外す制約を半平面として表し、各制約集合で最小半径を数値最適化。幅4・深さ3の探索。全体最適の保証はない。',
        'C02：対象2～7管では全二分割、8～25管では12方向で並べた管の各区切りを比較。漏水なしの工事費と2種類の予測系統の期待固定費が改善する分割を採用。',
        'C03：各元工事でC01とC02のうち工事費が安い形を選んだ併用案。モデル予測と元の対象集合の順は固定。',
        '全候補CSVを再読込し、対象集合、全鉄管被覆、修正済み工事費、年次予算と繰越、2051年までの完了を検査。漏水なしでは全案2045年完了・遅延管ゼロ。',
        '通常実装と高速実装を漏水なし・ランダム漏水条件で照合し、CAPEX/OPEXがセント単位で一致。','',
        '内部評価：',
        '2予測系統（CMS001/F168、過去の年齢・地表別予測）×漏水率3段階×修理費ばらつき2段階×地域内漏水相関2段階＝24条件。',
        '開発48乱数×24条件、別乱数の確認128×24＝3,072シナリオ。候補間で乱数を共通化。3案すべてが事前の内部採用基準を通過。',
        '表の内部差は仮想将来でのG17比スコア差。G05という内部参照キーは実際にはSHA256を固定したG17を指す。',
        '確認結果やPublicを見た後の候補変更はない。内部評価は予測に依存し、Monte Carlo区間は予測の正しさを保証しない。','',
        '公式提出：ユーザーの明示依頼により3案を各1回提出。結果は公式25年内訳をDecimalで合計し、250,000,000−CAPEX−OPEXで算出。']
    for r in records:
        lines += [f'{r["candidate"]}: ID={r["submission_id"]}, SHA256={r["sha256"]}',f'公式年間内訳：{r["result_url"]}']
    lines+=['','再実行：.venv-eda/Scripts/python.exe output/2026-11-09_geometry_search/experiment.py generate',
        '.venv-eda/Scripts/python.exe output/2026-11-09_geometry_search/experiment.py evaluate',
        '依存する既存実装と予測入力のハッシュはsource_audit.json、設定はprotocol.jsonとsubmission_policy.json。',
        '共有Google Docの現在の関連内容と結果形式を参照した。共有Docの編集はしていない。']
    (HERE/'REPORT.ja.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (HERE/'RESULTS_FOR_DOC.txt').write_text('\n'.join(lines[:9])+'\n\nMethod: Geometry changes only, with fixed model forecasts. All three candidates pass 24-condition paired internal confirmation; official Public results are reported separately. Private performance remains unknown.\n',encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    if sys.argv[1]=='record':record(sys.argv[2],sys.argv[3])
    else:report()
