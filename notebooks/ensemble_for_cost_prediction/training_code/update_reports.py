"""Update bilingual report from committed predictions; no fitting or forecast changes.

Run: python notebooks/ensemble_for_cost_prediction/training_code/update_reports.py
from the repository root. Requires pandas, numpy and tabulate.
"""
from pathlib import Path
from decimal import Decimal
import json,hashlib
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]
METHODS=['takito_catboost','ensemble_equal','ensemble_stable','ensemble_balanced','ensemble_rmse_weighted','prior_surface_mean']

def main():
    p=pd.read_csv(REPO/'data/pipes.csv');t=pd.read_csv(REPO/'data/train.csv')
    ids=set(p['Pipe ID'])-set(t['Pipe ID']);q=p.loc[p['Pipe ID'].isin(ids)].copy()
    scope=pd.read_csv(PACKAGE/'evidence/prediction_scope.csv')
    assert set(scope.id)==ids and len(ids)==28926
    material=p.set_index('Pipe ID').Material
    issue=q[['Pipe ID','Material','Lay date']].rename(columns={'Pipe ID':'id','Material':'material','Lay date':'lay_date'})
    issue['material_unseen_in_training']=issue.material.eq('polyurethane')
    issue['material_missing']=issue.material.isna()
    issue['lay_date_missing']=issue.lay_date.isna()
    issue.to_csv(PACKAGE/'evidence/input_quality_by_pipe.csv',index=False)
    coords=p[['GPS x1','GPS y1','GPS x2','GPS y2']].apply(pd.to_numeric,errors='coerce')
    lengths=np.linalg.norm(coords[['GPS x2','GPS y2']].to_numpy()-coords[['GPS x1','GPS y1']].to_numpy(),axis=1)
    audit={'inventory_rows':len(p),'training_rows':len(t),'target_rows':len(q),
      'inventory_missing':{k:int(v) for k,v in p.isna().sum().items()},
      'target_missing':{k:int(v) for k,v in q.isna().sum().items()},
      'training_missing':{k:int(v) for k,v in t.isna().sum().items()},
      'invalid_or_nonfinite_coordinate_rows':int((~np.isfinite(coords.to_numpy())).any(axis=1).sum()),
      'zero_length_rows':int((lengths==0).sum()),'nonpositive_training_cost_rows':int(t.Cost.le(0).sum()),
      'maximum_observed_training_cost':float(t.Cost.max()),
      'maximum_endpoint_length_m':float(lengths.max()),
      'source_sha256':{str(z.relative_to(REPO)):hashlib.sha256(z.read_bytes()).hexdigest() for z in [REPO/'data/train.csv',REPO/'data/pipes.csv']}}
    totals=[]
    for name in METHODS:
        path=PACKAGE/'predictions'/f'{name}.csv';df=pd.read_csv(path,dtype={'predicted_cost':str})
        assert set(df.id)==ids and len(df)==len(q)
        values=pd.to_numeric(df.predicted_cost);assert np.isfinite(values).all() and values.ge(0).all()
        mats=df.id.map(material)
        def amount(mask):return float(sum((Decimal(s) for s in df.loc[mask,'predicted_cost']),Decimal(0)))
        pu=mats.eq('polyurethane');missing=mats.isna();supported=~(pu|missing)
        totals.append({'candidate':name,'n_pipes':len(df),'supported_material_total':amount(supported),
           'pu_extrapolation_total':amount(pu),'missing_material_extrapolation_total':amount(missing),
           'all_target_total':amount(pd.Series(True,index=df.index)),'zero_predictions':int(values.eq(0).sum())})
    total=pd.DataFrame(totals);total.to_csv(PACKAGE/'evidence/repair_all_2027_totals.csv',index=False,float_format='%.2f')
    audit['zero_prediction_counts']=dict(zip(total.candidate,total.zero_predictions.astype(int)))
    (PACKAGE/'evidence/input_quality_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    configs={}
    for f in (PACKAGE/'configs').glob('*.json'):
        c=json.loads(f.read_text());configs[c['model']]=c
    (PACKAGE/'evidence/model_input_contract.json').write_text(json.dumps({
      'scenario_year':2027,'history_cutoff_exclusive':'2027-01-01',
      'inventory_columns':p.columns.tolist(),'label_columns':t.columns.tolist(),
      'target_policy':'all inventory IDs not present in train.csv',
      'output_columns':['id','predicted_cost'],'output_meaning':'conditional repair cost per pipe under 2027 scenario, not failure time or probability',
      'per_model_selected_features':{n:c['features'] for n,c in configs.items()},
      'unknown_categories':'CatBoost strings; XGBoost/LightGBM missing category; sklearn OHE ignores unseen (all-zero reference collision)',
      'invalid_new_inputs':'Duplicate IDs, invalid dates, nonfinite coordinates or zero endpoint length require correction and revalidation; no universal sanitization implemented.'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    for language in ['ja','en']:update(language,total,configs)
    print(total.to_string(index=False))

def update(lang,total,configs):
    ja=lang=='ja';path=PACKAGE/f'report_{lang}.md';text=path.read_text(encoding='utf-8')
    old_start='## 年別結果' if ja else '## Annual results'
    new_start='## 2027年に対象パイプをすべて修理する場合の総額' if ja else '## Total cost if every target pipe is repaired in 2027'
    if old_start in text:text=text[:text.index(old_start)]+text[text.index('## 出典・検証' if ja else '## Sources and validation',text.index(old_start)):]
    if new_start in text:text=text[:text.index(new_start)]+text[text.index('## 出典・検証' if ja else '## Sources and validation',text.index(new_start)):]
    # Remove prior sections before regenerating; preserve all other narrative and results.
    headers=['## 欠損・未知材料・不正値の処理詳細','## モデルの入力・出力と故障時期チームとの連携'] if ja else ['## Detailed handling of missing, unseen and invalid values','## Model inputs, outputs and coordination with failure-timing models']
    for header in headers:
        if header in text:
            start=text.index(header);end=text.find('\n## ',start+len(header));text=text[:start]+(text[end+1:] if end!=-1 else '')
    note=('\n\n**Shanaの既存提出物の除外理由（時間的リーケージ）:** 2023〜2025年の評価年の正解を特徴量・モデル選択に使用していたため、その既存予測を安全なOOFとしてアンサンブルに投入できませんでした。この時間的な検証汚染と、実際の未来の故障時刻・曜日が予測時には不明である問題は別です。Shanaの手法自体が使用不能なのではなく、過去データだけの内側選択で再学習すれば候補にできます。\n' if ja else '\n\n**Why existing Shana submissions were excluded (temporal leakage):** Outcomes from evaluation years 2023–2025 were used for feature/model selection, so those predictions cannot serve as clean OOF ensemble inputs. This temporal validation contamination is distinct from using realized future failure hour/weekday, which is unavailable at forecast time. The architectures themselves can be candidates if refitted with selection entirely inside past-only folds.\n')
    section='## 実施した処理とリーケージ対策' if ja else '## Work performed and leakage controls'
    if '**Shanaの既存提出物の除外理由' not in text and '**Why existing Shana submissions' not in text:
        after=text.find('\n## ',text.index(section)+len(section));text=text[:after]+note+text[after:]
    details=details_ja() if ja else details_en()
    interface=interface_ja(configs) if ja else interface_en(configs)
    cols=['candidate','supported_material_total','pu_extrapolation_total','missing_material_extrapolation_total','all_target_total']
    table=total[cols].copy()
    if ja:table.columns=['候補','既知材料27,240本','PU 1,678本（外挿）','材料不明8本（外挿）','全28,926本の合計']
    else:table.columns=['Candidate','Supported: 27,240 pipes','PU: 1,678 (extrapolated)','Missing material: 8 (extrapolated)','All 28,926 targets']
    scenario='\n'+new_start+'\n\n'+table.to_markdown(index=False,floatfmt=',.2f')+'\n\n'
    scenario+=('この表は各CSVの予測費用をそのまま合計した「2027年に全対象を1本につき1回修理する」仮定の総額です。全43,039本ではなく、指定どおりtrain.csvの14,113個のIDを除外した28,926本が対象です。費用単位は元データの単位であり、円・ドルのどちらとも断定しません。PU・材料不明分を分け、数値が未検証の外挿に依存する範囲を見えるようにしました。\n\nこの合計に故障確率は掛けていません。通常の2027年予算ではなく、全修理シナリオの費用です。またモデルは故障時の修理費で学習しているため、予防交換や同時施工による割引・固定費共有、インフレを別途モデル化していません。全件一斉の計画工事の見積額として校正された値ではありません。元のRMSE・総額偏りの説明は、過去に実際に故障した対象での精度検証として保持しています。\n\n集計はCSVの保存値を十進数で加算しました。[機械可読の集計CSV](evidence/repair_all_2027_totals.csv)に同じ数値を保存しています。\n' if ja else 'This table sums saved CSV predictions under the explicit assumption that every target pipe is repaired once in 2027. The population is 28,926 pipes after excluding the 14,113 train.csv IDs as requested, not all 43,039 inventory pipes. Units are those of the source costs; neither yen nor dollars is asserted. PU and missing-material contributions are separate to expose reliance on unvalidated extrapolation.\n\nNo failure probabilities are applied. This is a repair-all scenario, not the usual annual failure budget. Models were trained on costs incurred at failures, not planned preventive replacement; shared mobilization costs, coordinated-work discounts and inflation have not been modeled separately. These totals are not calibrated quotes for a coordinated replacement project. The earlier RMSE and total-bias explanations remain as historical validation on actually failed pipes.\n\nTotals were calculated with decimal arithmetic from the saved CSV values. The same figures are available in the [machine-readable totals CSV](evidence/repair_all_2027_totals.csv).\n')
    source='## 出典・検証' if ja else '## Sources and validation';text=text.replace(source,details+'\n'+interface+'\n'+scenario+'\n'+source)
    path.write_text(text,encoding='utf-8')

def details_ja():return '''## 欠損・未知材料・不正値の処理詳細

### 実データで何が見つかったか

| 項目 | 元のpipes.csv：43,039本 | trainのIDを除いた予測対象：28,926本 | 対応 |
|:---|---:|---:|:---|
| PU（polyurethane） | 1,678 | 1,678 | 材料は既知だが、その材料の費用ラベルがない。各モデルの未知カテゴリ処理で推定 |
| 材料欄の空欄 | 8 | 8 | `__unknown__`という文字列に置換。鉄やPUと断定しない |
| 敷設日の空欄 | 15 | 9 | 年齢をNaNのまま保持し、`age_missing=1`。学習側の6件も削除しない |
| 表面欄の空欄 | 0 | 0 | 実データでは処理不要。空欄が来れば`__unknown__`に置換 |
| 座標の空欄・非数値・非有限値 | 0 | 0 | 実データでは補完せず使用。新規入力の不正座標は自動補完できない |
| 座標からの長さ0 | 0 | 0 | 実データでは問題なし。新規の長さ0は方向特徴量の計算前に修正・再検証が必要 |

train.csvのID・Date・Time・Costには空欄がなく、費用の0以下もありません。欠損敷設日の6件を落とした既存の拡張表だけに頼らず、生のtrain.csvから復元して全14,113件を保持しました。教師データの材料はwrought iron、cast iron、gray iron、brass、copperの5種類です。PUと材料欠損の修理費は学習にありません。最大実績費用7,257,576.36を含む高額例を、異常値と決めつけて削除・上限処理していません。

### PUと材料空欄を各モデルがどう計算したか

PUは文字列`polyurethane`をそのまま保ち、材料空欄は`__unknown__`にしました。材料と表面を連結したカテゴリにもこの値を反映します。どちらも既知材料へ付け替えてはいません。ただしエンコーダの振る舞いに次の違いがあります。

| モデル系統 | 未知カテゴリの実際の処理 | 計算の根拠と限界 |
|:---|:---|:---|
| CatBoost | PU／`__unknown__`を文字列として入力し、CatBoostのカテゴリ処理で予測 | 学習済みの木と選択済みの年齢・表面・長さ・経路等で評価。PU特有の価格関係は学習していない |
| XGBoost／LightGBM | 学習時のカテゴリ一覧にない値をカテゴリ欠損に変換 | 学習済みの欠損カテゴリ処理と他の選択特徴量で評価。欠損経路がPUの実態を表す保証はない |
| 線形回帰／Gamma／Random Forest | `OneHotEncoder(handle_unknown='ignore', drop='first')`で未知カテゴリのダミー変数をすべて0にする | 符号化が学習済みの基準カテゴリと同じになる。その他の特徴量は残るが、PUに適切な材料効果を推定したとはいえない |
| 表面種別の過去平均 | 材料を使わず、同じ表面種別の費用平均を全体平均へ縮小 | `(表面別費用合計 + α×全体平均)/(表面別件数 + α)`。αはOptunaで選択。表面も未知なら全体平均 |
| アンサンブル | 上記10構成モデルの予測を保存済みの重みで加重平均 | 外挿の不確実性は加重平均しても消えない。PU専用モデルの代わりにはならない |

### 年齢・過去実績などの欠損をどう処理したか

年齢は`(2027-01-01 − Lay dateの日数)/365.25`で計算します。敷設日がない場合は年齢と対数年齢をNaNにし、`age_missing=1`で明示しました。年齢帯の集計キーも欠損用の`-1`に分けています。敷設日・年齢を0年と仮定してはいません。

数値欠損は、線形回帰・Gamma・RFでは学習期間だけで求めた中央値を`SimpleImputer(strategy='median', add_indicator=True)`で補完し、学習時に欠損のあった列に欠損フラグを追加してから標準化しました。追加フラグは「将来のすべての欠損列」に自動で付くわけではありません。CatBoost・XGBoost・LightGBMは数値NaNを残し、各学習済みモデルの欠損値処理で予測しました。

過去実績のない材料・材料×表面・経路等では、件数と費用合計の集計上の空欄を0にします。これは「修理費0」の代入ではありません。平均費用特徴量は`(過去合計 + 20×過去全体平均)/(過去件数 + 20)`とし、件数0なら過去全体平均です。中央値・最大値・標準偏差は実績がなければNaNを保持します。高額費用比率は`(過去高額件数 + 50×過去全体比率)/(過去件数 + 50)`で縮小します。空間半径集計も同様で、近傍の他材料の過去修理費は利用できますが、PU自身の実績にはなりません。

近傍在庫・端点経路は同じpipes.csvから作成し、過去の修理状態は予測年より前のDateだけで更新します。近傍在庫の候補には敷設日が既知のパイプを使うため、敷設日欠損のパイプは近傍側の候補から外れます。端点接続と平面交差は区別し、経路や長さは幾何学的な代理特徴量です。

### 「変な値」と新規入力に対する境界

今回の座標は有限で、長さ0はなく、既知の敷設日は2019年より前でした。データをこの条件で検査して使用したので、任意の不正値を自動修復する仕組みではありません。未知の非空カテゴリは上記未知カテゴリ処理となりますが、綴り間違いも本当の新カテゴリも区別して自動修正していません。重複ID、解析できない日付、非有限座標、長さ0、存在しないIDは入力を修正して再検証する必要があります。欠損座標を0や別地点へ置き換える処理はしていません。

モデルが負の費用を出した場合は最後に`max(予測, 0)`で0へ切り上げます。対数モデルは`exp`／`expm1`と学習済みの補正係数で元の費用単位へ戻してから同じ処理をします。非有限な出力は検査失敗とし保存しません。高い予測値は自動で上限処理していません。保存CSVのCatBoostには0予測が35件あり、その他5候補は0件でした。これらは未知材料に一律0を割り当てた結果ではなく、予測と非負制約による出力です。0は修理が無料という実証ではありません。ゼロ件数は[evidence/input_quality_audit.json](evidence/input_quality_audit.json)に記録しました。

行単位の欠損・未知材料は[evidence/input_quality_by_pipe.csv](evidence/input_quality_by_pipe.csv)、入力監査集計は[evidence/input_quality_audit.json](evidence/input_quality_audit.json)で確認できます。欠損敷設日の有無と材料の支持範囲は別のフラグであり、`supported_material`でも年齢が欠損する行があります。
'''

def details_en():return '''## Detailed handling of missing, unseen and invalid values

### What was actually found

| Item | Inventory: 43,039 | Prediction targets: 28,926 | Handling |
|:---|---:|---:|:---|
| PU (`polyurethane`) | 1,678 | 1,678 | Material is known, but has no cost labels; use each model's unseen-category handling |
| Missing material | 8 | 8 | Replace with `__unknown__`, without asserting iron or PU |
| Missing lay date | 15 | 9 | Keep age NaN, set `age_missing=1`; retain the six training rows too |
| Missing surface | 0 | 0 | No observed replacements required; categorical blanks become `__unknown__` |
| Missing/non-numeric/nonfinite coordinates | 0 | 0 | Use observed coordinates; no automatic repair of invalid new coordinates |
| Zero endpoint length | 0 | 0 | None observed; new zero-length input requires correction before directional features |

train.csv has no missing ID, Date, Time or Cost and no nonpositive costs. The six training records with missing inventory lay date were restored from raw train.csv rather than dropped with an incomplete enriched table. All 14,113 labels were retained. Training materials are wrought iron, cast iron, gray iron, brass and copper; neither PU nor missing-material costs were observed. Expensive events, including the maximum observed cost of 7,257,576.36, were not deleted or capped as presumed outliers.

### How each model calculated PU and missing-material costs

PU retains the string `polyurethane`; missing material becomes `__unknown__`. The material–surface composite category carries the same values. Neither is relabeled as a supported material in the source features, although the encoder introduces the limitations below.

| Family | Actual unseen-category handling | Basis and limitation |
|:---|:---|:---|
| CatBoost | Pass PU/`__unknown__` as strings through CatBoost categorical processing | Learned trees and selected age, surface, length, route and other inputs; no PU-specific price relationship was learned |
| XGBoost/LightGBM | Convert categories outside fitted category levels to category missing | Learned missing-category behavior plus remaining features; that path is not validated for PU |
| Linear/Gamma/Random Forest | `OneHotEncoder(handle_unknown='ignore', drop='first')` makes every dummy in the unseen category block zero | Encoding collides with the dropped reference category; other inputs remain, but this does not establish an appropriate PU material effect |
| Surface mean | Ignore material; shrink surface-specific average toward global mean | `(surface cost sum + α × global mean)/(surface count + α)`, with Optuna-selected α; unseen surface falls back to global mean |
| Ensembles | Weighted average of the ten saved component predictions | Averaging does not remove unsupported-material uncertainty or substitute for a PU-specific fit |

### Missing age and missing historical statistics

Age is `(2027-01-01 − Lay date in days)/365.25`. Missing lay date produces NaN age/log-age and `age_missing=1`; the age-band aggregation key is the separate missing group `-1`. Age was not assumed to be zero.

Linear, Gamma and RF pipelines use `SimpleImputer(strategy='median', add_indicator=True)` with medians fitted only on their training period, then standardize. Indicators are added for columns missing during fit, not automatically for every column that might become missing later. CatBoost, XGBoost and LightGBM retain numeric NaN and use their learned native missing-value behavior.

For material, material–surface, route and similar groups with no historical repairs, aggregation count and sum are set to zero. This does not assign zero repair cost. Mean features are `(past sum + 20 × past global mean)/(past count + 20)`; zero count gives the global past mean. Median, maximum and standard deviation remain NaN if unsupported. High-cost rates are shrunk as `(past high-cost count + 50 × past global rate)/(past count + 50)`. Radius aggregates follow the same approach: nearby costs from other materials may be available, but they are not PU observations.

Inventory neighborhoods and endpoint routes use the same pipes.csv; repair-state features use only dates preceding the forecast origin. Neighborhood inventory candidates require known lay dates, so missing-lay-date pipes are excluded as neighbors. Endpoint connections are distinguished from planar crossings; geometry is a proxy, not excavated length or verified physical connectivity.

### Unusual values and limits on new inputs

Observed coordinates are finite, endpoint lengths are nonzero, and known lay dates precede 2019. These were checked before using the data; there is no universal invalid-value sanitization. Nonblank unseen strings follow category rules above; typos are not automatically distinguished from genuinely new categories. Duplicate IDs, unparseable dates, nonfinite coordinates, zero-length geometry or missing inventory IDs require correction and revalidation. Missing coordinates were not replaced with zero or another location.

Negative model predictions are clipped at zero with `max(prediction, 0)`. Log models are first transformed back with `exp`/`expm1` and their fitted correction factor, then clipped. Nonfinite predictions fail validation and are not saved; large predictions are not capped. CatBoost's saved CSV has 35 zero predictions; the other five candidates have none. These arise from model inference and nonnegative constraints, not blanket zero assignment to unseen materials; zero is not evidence of free repair. Counts are recorded in [input_quality_audit.json](evidence/input_quality_audit.json).

See [input_quality_by_pipe.csv](evidence/input_quality_by_pipe.csv) for per-row flags and [input_quality_audit.json](evidence/input_quality_audit.json) for counts. Material support and missing lay date are separate dimensions: a supported-material row can still have missing age.
'''

def feature_table(configs,ja):
    rows=[]
    for n,c in sorted(configs.items()):
        pack=c['params'].get('feature_pack','surface only');features=c['features']
        rows.append({'model':n,'feature_pack':pack,'selected_columns':1 if n=='prior_surface_mean' else len(features),
          'output':('費用平均（表面別）' if ja else 'Surface mean cost') if n=='prior_surface_mean' else ('対数予測→元費用' if ja else 'Log prediction → cost') if n in ['anthony_rf_log_mean','rion_lgb_log_mean_refit'] else ('元単位の費用' if ja else 'Original-unit cost')})
    return pd.DataFrame(rows).to_markdown(index=False)

def interface_ja(c):return '''## モデルの入力・出力と故障時期チームとの連携

### 共通の入力

- パイプ在庫：`Pipe ID`, `Lay date`, `Material`, `Surface`, `GPS x1`, `GPS y1`, `GPS x2`, `GPS y2`。IDは結合・出力用で、ID自体を説明変数にしません。
- 学習・履歴：`train.csv`の`Pipe ID`, `Date`, `Cost`。`Date`は過去の年と履歴カットオフの判定用。`Time`はファイルに存在しますが今回のモデル入力ではありません。
- シナリオ：2027年、年齢基準日2027-01-01、観測履歴カットオフ2027-01-01未満。周辺・経路情報を作るため、対象1本だけでなく在庫全体と過去履歴が必要です。
- 実際のモデル入力：前処理で作った選択済み特徴量だけ。各モデルの完全な列名は[model_input_contract.json](evidence/model_input_contract.json)と`configs/*_2027.json`の`features`に記載。アンサンブルは同じ10構成モデルの予測を入力として加重平均します。

### 各構成モデルが使う特徴量群と出力

'''+feature_table(c,True)+'''

baseは年・材料・表面・年初年齢・直線長等、inventoryは周辺在庫・過去修理状態・方向と交互作用、group_historyは材料・表面・地域・年齢帯・経路別の過去費用、local_historyは半径・最近傍の過去費用、topologyは端点経路とその過去費用、allは全群です。個々の列はモデル系統ごとに削除・選択されるため、群名だけでは完全な入力仕様になりません。表面平均モデルは構成設定に共通列が記録されていますが、実際の推論で読む説明変数はsurfaceのみです。

### 何を出力できるか

各候補は1本ごとに、2027年シナリオで故障した場合の非負の修理費点予測を元費用単位で出力します。提出CSVは`id,predicted_cost`のみ。故障する時期、故障確率、予測区間、PUの信頼度、工事優先度は出力しません。入力監査・材料支持範囲は別ファイルです。

### 故障時期予測チームと合わせる仕様

同じ`id`、在庫スナップショット、観測カットオフ、除外ID、欠損材料・敷設日フラグを共有してください。修理済みIDを故障時期チームが残す場合は、再故障・材料変更を含む別の状態定義が必要です。2027年予算を組むなら、同一対象・同一期間の故障確率`p_i`を受け取り、`Σ p_i × predicted_cost_i`で期待費用を組みます。複数故障を許すなら、費用モデルが各故障に適用できる前提を確認したうえで期待故障件数を使います。これは両モデルを合わせた予算精度の検証を済ませたことを意味しません。

現在の費用モデルは実際の故障月・曜日・時刻を必要とせず、予測された故障日も今回のCSVに入力していません。故障時期チームが日付を出しても、その日付で年齢を置き換えるだけでは今回と同じ仕様ではありません。現状は年初年齢と年シナリオです。2028年以降へ展開する場合は年・年初年齢を更新して再推論し、観測していない将来の修理履歴を追加しないようにします。未知材料の費用予測と故障時期予測の支持範囲を別々に管理してください。
'''

def interface_en(c):return '''## Model inputs, outputs and coordination with failure-timing models

### Shared inputs

- Inventory: `Pipe ID`, `Lay date`, `Material`, `Surface`, `GPS x1`, `GPS y1`, `GPS x2`, `GPS y2`. ID is a join/output key, not a predictor.
- Training/history: train.csv `Pipe ID`, `Date`, `Cost`. Date establishes historical years and cutoff. `Time` exists in the file but is not an input to these models.
- Scenario: year 2027, age reference 2027-01-01, observed history strictly before 2027-01-01. Full inventory and past history, not just one pipe row, are needed for neighborhood and route features.
- Estimator inputs: only selected derived columns, explicitly listed in [model_input_contract.json](evidence/model_input_contract.json) and each configs JSON `features`. Ensembles consume the same ten component predictions and apply their saved weights.

### Features and outputs by component

'''+feature_table(c,False)+'''

base covers year, material, surface, year-start age and endpoint length; inventory adds neighborhood inventory, past repair state, orientation and interactions; group_history adds material/surface/grid/age-band/route historical costs; local_history adds radius/nearest-neighbor historical costs; topology adds endpoint routes and their history; all combines every group. Exact columns are selected/removed by family, so pack names alone are not a full schema. The surface-mean configuration contains common columns, but actual inference only reads surface.

### Outputs

Each candidate returns one nonnegative point estimate of repair cost conditional on a failure in the 2027 scenario, in original cost units. Submitted CSVs contain only `id,predicted_cost`. They do not predict failure time, probability, uncertainty intervals, PU reliability or intervention priority. Input-quality and material-support metadata are separate files.

### Alignment with the failure-timing team

Share ID keys, inventory snapshot, observation cutoff, excluded IDs and material/lay-date missing flags. If that team retains repaired training IDs, repeat failures and material replacement need a separate state definition. For an annual budget, obtain failure probabilities `p_i` for the same population and interval and form `Σ p_i × predicted_cost_i`. If multiple failures are allowed, use expected event counts only after checking that this severity estimate applies to each event. The combined budget model has not thereby been independently validated.

The current severity models do not require realized failure month, weekday or hour; predicted failure dates were not inputs to these CSVs either. Substituting a predicted incident-date age changes the year-start specification and requires validation. For later years, update scenario year/year-start age and re-run inference without adding unobserved future repair history. Track unsupported-material coverage separately in the timing and cost models.
'''

if __name__=='__main__':main()
