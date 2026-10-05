"""Write a compact Japanese result with inspectable per-fold settings."""
from pathlib import Path
import json,sqlite3
import pandas as pd

OUT=Path(__file__).resolve().parent
m=pd.read_csv(OUT/'fold_metrics.csv')
a=pd.read_csv(OUT/'average_metrics.csv')
manifest=json.loads((OUT/'feature_manifest.json').read_text())
verify=json.loads((OUT/'verification.json').read_text())
def table(df):
    head='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    return head+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
text=['# 単体ブースティング3モデルの工事費予測比較\n',
    '指定の `github-main/data/train_predict_cost.csv` の14,107件を使用。元の金額単位でRMSEを優先して特徴量とパラメータを選択し、指定の4Foldでテストしました。\n',
    '## 4Fold単純平均\n']
av=a.copy()
for c in ['MAE','RMSE']: av[c]=av[c].map(lambda v:f'{v:,.2f}')
text.append(table(av))
text.append('各年を同じ重みで平均した値です。全テスト案件をまとめた参考値は `pooled_metrics.csv` に保存し、依頼の単純平均と区別しています。\n')
text.append('## 各Foldのテスト値\n')
cols=['model','fold','train_end','test_year','n_train','n_test','MAE','RMSE']
mm=m[cols].copy()
for c in ['MAE','RMSE']: mm[c]=mm[c].map(lambda v:f'{v:,.2f}')
text.append(table(mm))
text.append('## 学習データ上の値（参考）\n')
tr=m[['model','fold','train_MAE','train_RMSE']].copy()
for c in ['train_MAE','train_RMSE']: tr[c]=tr[c].map(lambda v:f'{v:,.2f}')
text.append(table(tr))
text.append('学習データ上の誤差は、同じデータで学習したモデルの適合度です。将来の予測性能は上のテスト値で判断します。学習期間はFold間で重複します。\n')
text.append('## 選択した特徴量・設定\n')
summary,details,search=[] ,[],[]
for model in ['xgboost','catboost','lightgbm']:
    for year in range(2023,2027):
        folder=OUT/model/str(year)
        c=json.loads((folder/'selected.json').read_text())
        summary.append({'model':model,'test_year':year,'feature_count':len(c['features']),'trees':c['iterations'],'inner_RMSE':f'{c["inner_mean_rmse"]:,.2f}'})
        details.append({'model':model,'test_year':year,'features':json.dumps(c['features'],ensure_ascii=False),'parameters':json.dumps(c['params'],sort_keys=True),'trees':c['iterations']})
        with sqlite3.connect(folder/'studies.sqlite') as db:
            rows=db.execute('SELECT study_name,state,COUNT(*) FROM trials JOIN studies ON trials.study_id=studies.study_id GROUP BY study_name,state').fetchall()
        for stage,state,n in rows:
            search.append({'model':model,'test_year':year,'stage':stage,'state':state,'trials':n})
        text.append(f'### {model} — テスト{year}年\n')
        text.append('特徴量：`'+'`, `'.join(c['features'])+'`。\n')
        text.append('パラメータ：\n```json\n'+json.dumps(c['params'],indent=2)+'\n```\n')
        actual=int(m.loc[(m.model==model)&(m.test_year==year),'actual_trees'].iloc[0])
        text.append(f'反復数の設定上限：{c["iterations"]}。最終学習で生成された木の数：{actual}。内側の平均RMSE：{c["inner_mean_rmse"]:,.2f}。\n')
pd.DataFrame(summary).to_csv(OUT/'selected_config_summary.csv',index=False)
pd.DataFrame(details).to_csv(OUT/'selected_features_and_parameters.csv',index=False)
search=pd.DataFrame(search)
search.to_csv(OUT/'search_counts.csv',index=False)
complete=int(search.loc[search.state=='COMPLETE','trials'].sum())
text.insert(2,f'完了したOptuna試行は合計{complete:,}件です。このほか、内側評価で特徴量の追加・削除を検討しました。有限の探索で得た最良候補であり、絶対最適の保証はありません。\n')
text.append('## 方法と検証\n')
text.append('各Foldの学習期間内の直近2年を時系列検証に使用し、その2年のRMSE単純平均を最小化しました。反復ごとのearly stoppingによる探索後、有力な最大5設定を、両年で同じ木の数を使った場合の平均RMSEで再順位付けしました。CatBoostでは候補の木の数ごとに最初から再学習して比較しました。全12設定を確定してから外側テストを実行しました。履歴特徴量は全て前年末までの工事費から作り、同年の工事費は使用していません。前年実費が翌年初に確定している仮定があります。\n')
text.append('工事費の上限カット、外れ値除外、対数目的変数、モデルのアンサンブルは使用していません。負の予測値は0に丸めました。\n')
text.append(f'元CSVのSHA256と実費・対象IDの一致を確認し、履歴統計を{verify["history_samples_verified"]}件で独立照合しました。12FoldのMAE/RMSEと4Fold平均を案件別予測から別の計算方法で再計算しています。履歴照合は抽出検証であり、全行の独立再計算を意味しません。\n')
text.append(f'各テスト年以降の工事費を37倍＋1,234,567に改変してコスト依存の前処理をやり直し、当該テスト年までの特徴量が変わらないことを4Fold全てで確認しました。候補は{len(manifest["features"])}列です。コストを使わない補助インベントリ特徴量は改変の影響がない構造と、24件×6半径の独立計算を確認しました。学習データ上のMAE/RMSEも、保存した学習案件別予測から独立再計算しています。\n')
text.append('追加探索では、pipes.csvの周辺管情報と過去の実験コードから候補を追加しました。各モデル・Foldで、全候補の単独追加・採用列の削除・学習内Spearman相関の絶対値0.98以上の置換・カテゴリと数値コードの同情報の置換・関連する列群の同時追加/削除を比較し、改善を採用するたびに基準を更新して全比較を繰り返しました。その後にパラメータと共通反復数を60試行ずつ再調整し、特徴量比較を収束まで繰り返しました。最終的に、パラメータを探索した特徴量構成と、再確認後の構成が一致するまで確認しました。高相関だけで除外せず、検証RMSEの改善で判断しました。線分距離で近い管の過去コストも試しましたが、実際の修理地点や接続を示すものではありません。採用経路と全比較結果はexpanded_adoption_paths.csv、expanded_candidate_comparisons.csv、相関は各Foldのexpanded_correlations.csvに保存しています。拡張前のテスト出力は退避し、その誤差は追加探索に使用していません。\n')
text.append('入力ファイルの2026年末までのデータを使用しました。実行日時点で全実績が確定しているという意味ではありません。過去の作業で2026年も確認済みのため、この比較は今回の探索でテスト費用を未使用にした評価であり、プロジェクト全体で完全に未閲覧の最終テストではありません。\n')
text.append('周辺管の情報には、案件日より後に敷設された管と敷設日不明の管を除き、pipes.csvの属性が予測時点でも有効という仮定があります。撤去履歴や材質・地表面の変更履歴はありません。幾何的な交差や共通端点は確認済みの物理接続を表さず、端点間の直線長は実際の曲線長や掘削長を表しません。\n')
(OUT/'results_ja.md').write_text('\n'.join(text),encoding='utf-8')
print(a.to_string(index=False))
print('Completed Optuna trials:',complete)
