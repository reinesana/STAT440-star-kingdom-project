# CMS001＋F168：予測モデルと修理オーダーの統合手法（2026年までで学習）

## 採用案と今回の成果物

**金額予測CMS001、修理時期予測F168、共同指標と工事重複による再順位付けを使う02_OVERLAP_CREDIT。公式Publicは73.71M。** 2026年までの履歴で既に学習済みの2027年用モデルを使用し、今回その凍結予測から全修理対象の順序を再生成した。基底モデル全体を今回再学習したとは主張しない。未来の2027年以降の正解は学習に含めない。

「修理時期」は故障・漏水確率の予測。実際の施工予定年とは別である。円の形状は元の38,455個のまま、中心・半径を変更していない。旧12aの順序・予定年・旧予測は順位生成に使用しない。初期交換状態と物理的な工事計算には元データを使用する。

このフォルダの `submission_2027.csv` は公式提出用（ヘッダーなし）。`work_units_ranked.csv` は円38,455件の順序、`active_iron_pipes_ranked.csv` は2026年末に未交換の修理対象鉄管を最初に含む施工円の順に並べた表、`all_original_iron_pipes.csv` は交換済みも含む元の全鉄管一覧。既に漏水修理済みの管を再修理対象とせず、ただし提出円は元の全鉄管をカバーする。同一円で同時交換する管の表内同点順はPipe IDであり、個別施工順という意味ではない。予定年・工事費は将来漏水なしの計画シナリオで、公式の実際の交換年を保証しない。

## 公式比較の結論

|案|Public M|解釈|
|---|---:|---|
|01 FINAL_BLOCKS|73.54|共同指標＋自作計画年と順序入替え|
|**02 OVERLAP_CREDIT**|**73.71**|上記＋重複工事費の削減額|
|03 PROFILE|73.67|予測タイプ別重み|
|04 SAME12A_INDEX|73.68|12a同式へ新2予測を置換|
|05 S092|73.46|自作計画年で再構成|
|06 fine blocks|72.85|小さい順序入替え|
|07 inverse time|73.38|初回漏水時期の逆数で重み付け|
|旧12a・現行サイトへ再提出|69.89|同一円集合の比較対象|

01〜04と05〜07の対応は、本人の提出順の説明とスクリーンショットに基づく。12a再提出はブラウザーで69.89Mを確認。表示値による02の改善は+3.82M。02と04の差は0.03Mと小さい。今回比較した固定円7案のPublic最高であり、全手法の大域最適性、Private最良、予測器単独の全評価指標優位は未証明。内部の合成未来では順位が逆転し、全3仮定の安定改善基準は未達だった。公式Publicを優先するというユーザー指定で02を採用候補にした。Publicへの反復選択による過適合可能性があり、Privateは未知。異なる円形状を使った他のチーム案より最高とは主張しない。

## 2モデルをどのように混ぜるか

各管iについて、CMS001の条件付き修理金額をC_i、F168のk年生存確率をS_i(k)、F_i(k)=1-S_i(k)、直線長をL_iとする。CMS001未対応の未交換管8本はCMS001予測の中央値のみで補完（具体IDと値はverification.json）。別の金額モデルや正解平均を追加しない。

最終段階で使う管の便益は以下。係数は内部の有限探索で選んだもの。予測値自体の設置年別上書きや追加学習はせず、順位指標を変換する。

```
r_i = (1 - 0.32273279945607375) * F_i(2)
      + 0.32273279945607375 * F_i(25)
C_adjusted_i = C_i / (max(L_i,0.1)/median(L_live))**0.0030795309070914337
v_i = r_i**0.8300158667969959
      * max(C_adjusted_i - 0.9780439932142714*200*L_i,0)**0.878016813874991
```

円jの未交換管だけのv_iを合計する。Q_jは「jを選ぶと、残る未交換管がすべて交換済みになって無料スキップできる他の未施工円」の固定・地表工事費の合計。重複した長さ課金はQへ加算しない。Qは別の将来円でも不要になる費用を数え過ぎる可能性がある近似で、厳密な全体節約額ではない。

```
priority_j = (sum(v_i for remaining pipes inside j) + 0.01*Q_j)
             / max(current_construction_cost_j,1)**0.8490093178605346
current_construction_cost_j = 100 + sum(unique contained surface rates)*area_j**0.85
                              +200*sum(currently non-PU contained lengths)
```

実課金式はareaだけを0.85乗する現行式。優先度用の指数は課金式を変更しない。円を選ぶたびに交換状態を更新し、便益・実工事費・他円の不要化を再計算。Qの変化は隣接の二段階先にも及ぶため更新対象に含める。無料円はすぐスキップする。

## 初期順序・自作予定年・再構成の全工程

1. CF150：2年/25年F168確率とCMS001を組み合わせる初期共同指標で動的順位を作る。具体係数はrecipe.jsonのself_seed_recipe.joint_config。
2. 単件挿入・ブロック挿入/反転を比較し、内部費用が改善する変更を保存。このレシピは旧12aの順序から作らない。
3. 自作順序を年間10M、繰越、先頭の円が入らなければ翌年というルールで実行し、自作予定年を計算。S092では3年帯を優先枠にして上記v_iで再構成する。
4. self_blocks、final_blocksの採用済み変更を順に適用してFINAL_BLOCKSを作る。
5. FINAL_BLOCKSから自作予定年を再計算。最終02では1年帯を使用し、`max(自作予定年,現在計画年)` が小さい円から、同じ優先年内ではpriorityが高い円を選ぶ。予算不足の先頭円は翌年最優先。予算があれば翌年予定円の前倒しもある。予定年は厳密な締切でない。
6. recipe.jsonには全係数・採用済み入替えを収録し、reproduce.pyが予測だけから順序を再生成。最後にPublic73.71MのCSV SHA256と完全一致をassertする。レシピは内部合成未来で選択した順序ポリシーのパラメータであり、金額・時間予測の第三モデルではない。新データへ自動で同じ有効性を保証せず、この2027配置の凍結再現用。

## 再現と学習期間

リポジトリ直下から：

```
python -m pip install -r docs/reports/2026-10-28_CMS001_F168_order/requirements.txt
python docs/reports/2026-10-28_CMS001_F168_order/reproduce.py --data-dir data
```

raw dataはリポジトリ既存のdataを読み、このフォルダへ複製しない。inputsには学習済み2モデルの凍結予測、evidenceには学習設定。環境・ハッシュ・対象数・完了年はverification.json。全基底の再学習には元のモデル別学習手順と計算時間が必要であり、このコマンドは順序の再生成のみ。

CMS001：基底モデルは2019〜2026の14,113漏水レコードで学習済み。メタ重みは2023〜2026の過去年外予測8,592件で学習。F168：基底モデルは2026年末以前のみ、メタは2020〜2026の250,776管年OOF。2027〜2051の曲線は外挿を含み、25年先の実測精度を保証しない。

以下に2予測器そのものの詳細を一つの文書として統合する。下の時間モデル説明中R029は過去の「時期単独の順序」であり、今回の共同オーダー02ではない。関連文書内の旧相対リンクは末尾の参照元を参照。

---
## 付録A：CMS001の学習・選択・検証の詳細

# 最新の修理費モデル：仕組み・従来モデルとの比較

**Model: CMS001 — Material-Surface Cost Stack / 材質・地表面別コストスタック (v1.0.0).** [Current selection](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_model.json) · [Model registry](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/model_registry.json) · [How to switch](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/MODEL_MANAGEMENT.md)
[English](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/METHOD.en.md) · [モデルと根拠ファイル](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/README.md)

## 共有Google Docの「皆で作ったアンサンブル」との直接比較

[共有Google Doc](https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit)の「Summary of ensemble」を2026年10月28日に再確認した。ここで採用されたモデルは、raw XGBoost・CatBoost・baseline raw LightGBMを各1/3で平均する**漏水確率モデル**である。[選定記録](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/leak_model_comparison/comparison_results/selection.json)と[保存済みOOF予測](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/leak_model_comparison/comparison_results/selected_oof_predictions.csv)で対応を確認した。

**最新の時間予測F168は、このチームのアンサンブルに対して、同じ管・同じ年で評価した4年平均Brier、平均PR-AUC、年間漏水件数の予測の偏りで良好だった。** 2023〜2026年の共通132,121管年を使い、ラベル一致と一対一結合を確認して再計算した。Brierは **0.050400 → 0.045625（約9.47%改善）**、PR-AUCは **0.343890 → 0.405169**、年間件数の平均絶対偏りは **19.06% → 4.88%** となった。

| 共通対象でのBrier（低いほど良い） | チームの3モデル等重み | 最新時間モデルF168 |
|---|---:|---:|
| 2023年：35,817管 | **0.035815** | 0.038870 |
| 2024年：33,993管 | 0.035003 | **0.033550** |
| 2025年：32,042管 | 0.048990 | **0.044183** |
| 2026年：30,269管 | 0.081792 | **0.065896** |
| 年平均 | 0.050400 | **0.045625** |

F168はBrierで4年中3年を上回った。一方、2023年のBrierと4年平均log lossは従来モデルが良好で、log lossは従来0.189145、F168は0.230070だった（両者とも確率を1e-7〜1−1e-7にそろえて計算）。**すべての年・すべての指標で優れているという意味ではない。** 対象管の範囲が違うため、共有Docの元の全対象平均やF168の全対象平均とは直接混ぜない。[年別比較](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/shared_doc_timing_ensemble_comparison.csv)と[対象数・出典・検証](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/shared_doc_timing_comparison.json)を添付した。過去年の再利用比較であり、公式工事計画スコアの優位や特徴量追加だけの因果効果を証明するものではない。

この比較は**時間予測F168**の話であり、以下の**修理費stacking**のRMSE比較とは対象が異なる。また、共有Docの「元からポリウレタン・漏水後交換済みの管はゼロ確率」という課題条件に基づく扱いは、旧16b／17bの設置年別の手動確率上書きとは別物である。

## 最新の推奨と、改善した点

最新の推奨は **材質×地表面別・RMSE重視の条件付きstacking**（`conditional_material_surface_rmse`）である。従来の安定型アンサンブル・全体共通の最適化stackに代わる修理費予測の推奨として、年平均精度と年別の悪化幅を確認したユーザーが採用した。予測対象は**漏水した場合の修理金額**であり、故障確率や最終的な工事計画スコアではない。

同じ2024〜2026年、同じ6,766件の比較で、年平均RMSEは **$112,943.12**。[保存済みの従来比較24構成](https://github.com/reinesana/STAT440-star-kingdom-project/blob/8359301/docs/reports/2026-10-28_timing_winner/evidence/cost_internal_comparison.csv)のすべてより低かった。従来の安定型より0.77%、15入力の最適化stackより0.58%、CatBoost単体より0.21%改善した。ただし、これは再利用した過去検証上の観測結果で、有意差や未使用テストでの優位を示すものではない。

| 従来の修理費モデル・構成 | 年平均RMSE | 年平均MAE | 新モデルのRMSE増減 |
|---|---:|---:|---:|
| 従来の安定型アンサンブル | $113,822.99 | $22,667.17 | −0.77% |
| 従来の6入力最適化stack | $113,346.55 | $23,471.65 | −0.36% |
| 従来の15入力最適化stack | $113,605.17 | $23,507.16 | −0.58% |
| CatBoost単体 | $113,185.40 | $25,246.75 | −0.21% |
| 平均補正あり対数RF | $113,599.40 | $21,169.98 | −0.58% |
| 平均補正あり対数LightGBM（C002仕様） | $114,965.39 | $20,823.18 | −1.76% |
| 補正なし対数LightGBM（C024仕様） | $120,454.82 | $17,507.43 | −6.24% |
| **今回の推奨：条件付きstacking** | **$112,943.12** | **$23,375.50** | — |

Shanaらが利用した従来のチーム計画との関係もコードで確認した。[16bの実装](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_16.py)は、修理金額として `ensemble_stable.csv` を読み込み、17bもこの入力処理を再利用する。したがって、**従来のチーム計画で利用・比較された修理費モデルの保存済み共通年評価より、新モデルの平均RMSEは良好**と記載できる。一方、Shana本人の過去の学習済みモデルすべて、またはすべての交換計画より優れているとまでは確認していない。旧記事の4年平均を新モデルの3年平均と直接比べてもいない。

## 平均RMSE最良のRidgeではなく、こちらを選んだ理由

今回試した3入力Ridgeの平均RMSEは **$112,795.93** で、今回の推奨モデルより低い。採用理由は平均誤差だけではなく、CatBoostに対する年別の悪化幅も小さかったことである。

| 構成 | 平均RMSE | 2024年のCatBoost比 | 2025年 | 2026年 |
|---|---:|---:|---:|---:|
| **材質×地表面別** | **$112,943.12** | **−0.22%** | **−1.02%** | **＋0.53%** |
| 全体共通の最適加重平均 | $113,047.50 | −0.01% | −1.18% | ＋0.76% |
| 3モデル等重み | $112,973.85 | ＋0.24% | −2.41% | ＋1.52% |
| 3入力Ridge | $112,795.93 | −2.51% | −0.89% | ＋1.83% |

年ごとの修理費の難しさ自体も変わるため、RMSEの標準偏差だけではなく、同じ年のCatBoostに対する相対誤差で比較した。ただし、3年で将来の安定性が確立したとはいえない。モデルの種類を選んだのは結果確認後であり、探索的な選定である。

年間総額の偏りは **＋11.30%、＋25.26%、＋1.89%**。3モデル等重みは **＋2.69%、＋4.54%、−5.95%** で、総額の偏りでは等重みが優れる。今回の推奨は総額校正の最良モデルではなく、MAEも従来の安定型より大きい。すべての指標で優れたモデルとは記載しない。

## モデルの仕組みと時系列検証

入力は、TakitoのCatBoost、Anthonyの平均補正あり対数RF、Rionの平均補正あり対数LightGBM。基底モデルの設定と年初の特徴量は変更していない。

過去OOF予測の二乗誤差を小さくするよう、非負・合計1の全体共通重みを学習する。その後、「材質×地表面」の条件ごとに、過去OOFが50件以上あれば同様に局所重みを学習する。

```text
最終重み = n/(n+tau) × 条件別の重み
         + tau/(n+tau) × 全体共通の重み
予測修理費 = 3モデルの予測値 × それぞれの最終重み、の合計
```

50件未満、未知・欠損カテゴリでは全体共通重みを使う。3つのモデルを候補として持つが、条件によって重み0のモデルがあってよい。カテゴリの文字列は `material + " | " + surface` とする。

2027年用の全体重みは **CatBoost 68.6270%＋対数RF 31.3730%＋対数LightGBM 0%**。2023〜2026年の8,592件で学習し、18条件の局所重みを使い、tau=50で全体へ寄せる。実際の条件別重みはローカルの検証済み条件別重み表に保存しており、全体の68.6%／31.4%だけでは新モデルを表せない。

2024年は2023年OOFのみ、2025年は2023〜24年、2026年は2023〜25年で重みを学習する。tau候補は0、50、200、1000、1e9。過去のforward検証年の平均RMSEだけで選び、内部検証年がない2024年は固定tau=200とした。選択値は2024年200、2025年0、2026年50、2027年50。2027年の設定を過去へ適用しない。2027年のtau探索表の最小値は選択用の統計であり、年ごとに過去だけで選び直したモデルの正式な内部比較値$112,943.12とは異なる。

## Shanaの旧手動補正を外し、関連情報を特徴量で学習する方針

ユーザーは、Shanaの旧計画にあった設置年別の確率の手動上書きを、現在のモデリング方針には不適切と判断し、採用をやめる方針を指定した。旧[16bコード](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_16.py)には1963年0.12、1964年0.50、1965年0.60、1966年0.30への上書きがあり、[17b](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_17.py)はさらに1964年0.55、1965年0.70へ上書きする。これは学習済みの故障確率を後から手動で置き換える処理である。過去の再現用コードは履歴として残す。

**最新の時間予測F168では、この手動上書きを適用していない。** 代わりに、設置年・設置年代・管齢・材質×設置年のコホート・過去時点までのコホート故障率などを特徴量として扱い、関係を学習する。[保存済みの特徴量一覧](https://github.com/reinesana/STAT440-star-kingdom-project/blob/8359301/docs/reports/2026-10-28_timing_winner/evidence/base_models_2027.json)では、該当する基底モデルに `lay_year`、`lay_decade`、`material_cohort`、`cohort_prior_rate`、`age` などが含まれる。全モデルが全特徴量を使うわけではなく、旧補正の確率値をそのまま特徴量として入れたわけでもない。

**以前の手動補正付き計画より表示スコアが低くても、現在は、予測値を手で上書きせず過去情報から学習する方法の方が妥当で再現しやすい、という方針を採用する。** ただし、補正を外したことだけがスコア低下の原因かは未確認である。円、順序、予測入力、採点器の版も異なるため、旧スコアとの差を補正削除の効果と断定しない。旧publicが高いことだけでは補正の妥当性は証明できず、補正を外しただけで新しい確率がすべて正しく校正されたともいえない。

ここは今回の**修理金額モデル**と区別する。新しい修理費stackingが直接使うのは3モデルの金額予測と材質・地表面である。基底の金額モデルには年初の管齢が入り、一部には過去の年齢帯別修理費要約も入るが、F168のコホート時間予測そのものではない。また、対数RF／LightGBMをドル単位の平均へ戻す補正は残している。これは削除対象の「設置年ごとに故障確率を手動で上書きする処理」とは別物である。

## 検証・公開範囲・公式評価

元実験では、保存予測からの指標の独立再計算、基底モデルの学習・調整期間、各originの再学習、未来の正解値改変への不変性、行順変更、未知条件のフォールバックを確認した。追加の依頼を受け、2019〜2026年の14,113件で基底3モデルを採用済み設定のまま作り直し、2023〜2026年の時系列OOF 8,592件で条件別重みも再学習した。2027年以降の正解値は使っていない。

**実際の3基底モデル・前処理・対数平均補正・条件別重みを含む[学習済みモデル](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/fitted_model_2027.joblib)と、[2027年予測CSV](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/predictions_2027.csv)を公開する。** 保存・再読込後のモデルから27,240本の予測を再現し、逆順の入力でも確認した。元予測との差は数値誤差の範囲であり、過去の内部評価を新しく測り直したという意味ではない。[学習・配布検証](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/deployment_verification.json)、[重み](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/model_2027.json)、[比較表](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/mean_metrics_all_recipes.csv)、[従来24構成との比較](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/prior_model_comparison.csv)、[実行方法](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/README.md#実行方法--use)も添付した。

生の修理費ラベルと学習用・予測用の特徴量行列は公開しない。付属の基底予測から最終CSVを再現でき、実際の基底モデルによる推論には所定の特徴量CSVを与える。`train.py`は互換性のあるローカル特徴量・時系列OOFから、ハイパーパラメータ探索をやり直さず最終モデルを作り直すためのコードである。

上記リンクの旧比較表と時間モデルの特徴量一覧は、既に公開済みのcommit 8359301を参照する。過去年は繰り返し確認しており、未使用テストではない。今回の推奨は結果確認後のユーザーの採用判断である。旧記事の4年平均と新モデルの3年平均は直接比較しない。

新モデルの予測対象は27,240本。既存公開CSVの28,926行とは外挿方針が異なり、旧CSVを置き換えたり、残る1,686本の予測をここで作ったりしていない。

**原記事公開時点では修理費stackingの公式計画スコアは未評価だった。今回のCMS001＋F168＋02共同オーダーはPublic73.71Mで、これはCMS001単独の精度評価ではない。** C024、C002、21aなどの旧スコアを新モデルへ付け替えない。時間予測との組合せ、現行工事費式での順番の再生成、同じ採点器の版での公式評価は別途必要である。


---
## 付録B：F168の学習・選択・検証の詳細

# 首位の時間予測モデルは、どう動くのか

[English](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/METHOD.en.md) · [結果・根拠ファイル](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/README.md)

## 1. 採用したもの

今回採用したのは、時間予測 **F168** と、それを修理順序に変える **R029** の組合せである。

- F168：10種類の生存モデルを、標準化したRidge stackingでまとめる。
- R029：累積故障確率10%への到達時期、各年の故障確率、年間予算から順序を作る。
- 比較した591予測方式の中で、F168の4年平均Brierは最小の **0.04575249**。
- 公式に提出した20候補の中で、R029の総費用は最小の **$219,523,846.12**。

予測モデルと順番の作り方は別の工程である。確率予測がよいだけで、最も安い修理順序まで自動的に決まるわけではない。今回は修理金額の予測と円の大きさの最適化は行っていない。

## 2. 何を学習したのか

入力は `pipes.csv` と2019–2026年の `train.csv`。座標・直線距離・近傍半径の単位はmである。各年の初めに、まだ故障する可能性のある管を1観測とする。

元からポリウレタンの管と、その年より前に漏水した管は対象外とする。過去に漏水した管はポリウレタンに交換済みだからである。故障前の同じ管が複数年の観測として入ることはある。行をランダムに分割する方法は使わない。

特徴量に入れるのは、管の属性と年初までの履歴だけである。材質が不明な管も独立カテゴリとして残し、設置日が欠損する管も欠損フラグ付きで残す。生存学習では、学習終了年までに見えた故障だけを使い、故障は記録年の中点に置く。それ以外は学習終了時点で右打切りとする。

特徴量には、材質・地表・年齢・長さ・対数長さ・設置日欠損、設置年代、平面上の位置や向き、端点一致による隣接の代理変数、50/100/250 mの近傍構成、過去の近傍漏水などがある。全部のモデルが同じ特徴量を使うわけではなく、選択済みの組合せを使う。端点一致は幾何学的な代理指標で、実際の配管接続を保証しない。

欠損補完・標準化・カテゴリ変換・教師あり特徴量順位は、それぞれの学習区間だけで求める。CatBoostはカテゴリ特徴量を扱う仕組みを使う。最終モデルの具体的な特徴量は [base_models_2027.json](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/evidence/base_models_2027.json) に保存した。

## 3. 基底モデル10種類

| モデル | 今回の実装 |
|---|---|
| Elastic-net Cox | 正則化付き比例ハザードモデルと基準生存関数 |
| Weibull AFT | Weibull分布による故障時間の打切り尤度 |
| Lognormal AFT | 対数正規分布による故障時間の打切り尤度 |
| Spline survival | 年齢の3次スプラインと材質別の年齢効果を使う年次ロジスティックハザード |
| Random Survival Forest | 打切り時間・故障を扱う生存木のアンサンブル |
| XGBoost AFT | `survival:aft` に打切り区間を渡す勾配ブースティング |
| CatBoost hazard | 年次故障の分類モデルから将来ハザードを作る |
| Neural MTLR | 累積logitを使う離散時間の打切り尤度モデル |
| DeepHit系ネットワーク | 離散故障時期の確率、打切り尤度、順位損失を使う |
| Hierarchical hazard | グループ効果を持つ正則化MAP推定。MCMCによる事後分布推定ではない |

共有済みの通常のXGBoost・CatBoost・LightGBM漏水モデルも比較対象には含めた。ただし、**F168の入力は上記10種類であり、通常のLightGBMなどを追加で混ぜた17モデル版ではない**。上記CatBoost hazardと共有済みの通常のCatBoost分類モデルも異なる。

既存の検証済みnested Optuna設定と予測を使えるものは再利用した。未完成だったCox・生存森林の出力は、過去だけを使う12試行の探索で補った。2020–2022年の準備用OOFは事前設定のデフォルトを使い、2027年に最適化した設定を過去へ流用していない。モデル間で探索回数が異なるため、同一計算量による比較や全パラメータの大域最適化ではない。

## 4. 各モデルの予測を年次ハザードにそろえる

各モデルの生存確率を \(S_{im}(k)\) とする。これは管 \(i\) が、将来 \(k\) 年目の終わりまで故障しない確率で、\(S_{im}(0)=1\) と置く。

\[
h_{im}(k)=\operatorname{clip}\left(1-\frac{S_{im}(k)}{\max(S_{im}(k-1),10^{-12})},0,1\right)
\]

この \(h\) は「それまで故障していないとき、その年に故障する確率」である。

経験的な生存モデルの一部は25年分の推定を直接持たない。最後に推定できた年のハザードを0.95以下に制限し、その先へ延長した。2027年版のCox・生存森林・MTLR・DeepHitは8年の範囲の後でこの延長を使う。これは外挿の仮定で、25年先まで精度が検証できたという意味ではない。

## 5. Ridge stackingの学習

各過去年について、その年より前だけで基底モデルを学習し、その年の翌1年故障確率 \(1-S(1)\) を予測する。この10個の予測を入力とし、その年に実際に故障したかを目的変数にしてRidgeを学習する。基底モデルが自分の学習ラベルに当てはめた予測を、そのままstackingの学習データにはしない。

2027年版では、2020–2026年の **250,776件の管・年観測**を使った。処理は `StandardScaler → Ridge` である。

\[
z_m=\frac{x_m-\mu_m}{s_m},\qquad
\min_{b,\beta}\sum_i(y_i-b-\beta^\top z_i)^2+\alpha\|\beta\|_2^2
\]

切片は罰しない。係数には負の値もあり、合計1という制約もない。したがって、これは**非負の重み付き平均ではない**。相関のあるモデルの過大な寄与を、別のモデルの負の係数で補正できる。負の係数があることは、故障確率が負でよいという意味ではない。また、係数をそのまま因果的な重要度とは解釈しない。

Ridgeの正則化係数 \(\alpha\) は、OptunaのTPEで0.01〜10,000を対数スケールで12試行した。最も新しい過去年を内側の検証年にする。2027年版なら、2020–2025年のOOFで学習し、2026年のOOFに対する、0〜1へ制限した予測のBrierで選ぶ。その後2020–2026年のOOF全体で再学習する。各候補の標準化も、その候補の学習側だけで求める。

2027年版の選択値は **alpha = 7308.877234445037**。将来の各年にも、同じ学習済み写像を適用する。

\[
\hat h_i(k)=\operatorname{clip}\left(b+\sum_m\beta_m\frac{h_{im}(k)-\mu_m}{s_m},0,1\right)
\]

\[
\hat S_i(k)=\prod_{j=1}^{k}(1-\hat h_i(j))
\]

範囲を制限し、生存確率が増加しない形にする。保存済みのOOFとalphaからRidgeを再構成したところ、2027年の保存済みfloat32生存曲線と**最大差0で一致**した。係数・切片・標準化の平均と尺度は [stacking_2027.json](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/evidence/stacking_2027.json) にある。この確認はstackingの再構成であり、全基底モデルを独立に再学習した検証ではない。

## 6. 内部評価の方法と結果

| 分割 | 基底モデルの学習年 | test年 |
|---|---|---|
| 1 | 2019–2022 | 2023 |
| 2 | 2019–2023 | 2024 |
| 3 | 2019–2024 | 2025 |
| 4 | 2019–2025 | 2026 |

stackingの学習・内側の検証にも各test年より前だけを使う。たとえば2023年testなら、2020–2021年のOOFで候補を学習し、2022年OOFでalphaを選び、2020–2022年OOFで再学習する。

| test年 | Brier | PR-AUC |
|---|---:|---:|
| 2023 | 0.038892 | 0.326699 |
| 2024 | 0.033629 | 0.537127 |
| 2025 | 0.044469 | 0.324140 |
| 2026 | 0.066020 | 0.420636 |
| 年平均 | **0.045752** | — |

Brierは翌年の故障確率の誤差で、小さい方がよい。一方、R029の年間工事予算内での故障先回り割合は4年平均 **0.548824**。DeepHit系の順番は約 **0.619952**で上だった。したがって、R029がすべての内部指標で首位だったわけではない。

## 7. 確率を実際の修理順序に変える

### 同じ円を使う

元の鉄管38,455本それぞれに対応する円を使う。中心は両端点の中点、半径は直線長の半分＋ \(10^{-6}\) m、出力は小数点以下9桁に丸める。両端点が円内に入ることを確認する。別の未交換管も円内に完全に入る場合は同時に交換されるため、実際の交換が必ず1本だけになるわけではない。

### 累積10%と各年の故障確率

累積故障確率 \(F_i(k)=1-\hat S_i(k)\) が **10%** に最初に達する年を求める。

\[
a_i=\min\{k:F_i(k)\ge0.10\}
\]

25年間で到達しない場合は期間外として扱う。これは「故障する／しない」を判定する分類閾値ではない。

各計画年では、到達期限が近いものを優先する。すでに期限に到達したものは同じ当年グループとし、その中を、その年に初めて故障する確率 \(\hat S_i(k-1)-\hat S_i(k)\) の大きい順にする。同順位は元の管データの行順を保つ。

その年ごとの優先順位から、故障なしの工事予算リプレイを使って1本の固定CSV順序を作る。年間予算は1,000万ドル、未使用分は繰り越す。選んだ先頭の工事が予算不足なら翌年も先頭に保持し、安い後続工事を先に通さない。交換済みだけの円は無料で飛ばす。

公式評価では、この固定CSVを実際の評価イベントに対して実行する。未知の公式故障を見てオンラインで順番を作り直す方法ではない。

修正式は

\[
100+R A^{0.85}+200L_{\mathrm{live}}
\]

で、セント単位に丸める。べき乗するのは面積だけ。\(R\) は円内の異なる地表単価の和で、交換済み管の地表も含む。\(L_{\mathrm{live}}\) は円内の現在非ポリウレタンの管長。未交換管がない円は0ドルである。

### なぜ10%を使ったのか

一定閾値の方式では、各モデル・各年について10%、20%、…、90%を比較した。予測曲線に基づく「計画交換より前の期待故障数」を最小化する。可変閾値では、閾値の切替回数にも固定の小さな罰則を加える。2027年版R029は、test年の実績やpublicを見ずに、予測と管の情報から10%を選んだ。

過去の各分割では、違う一定閾値を選ぶ場合がある。R029はこの**選択手順**の名前であり、過去4年すべてで10%を固定した実験という意味ではない。

固定閾値の比較は95,742条件。その後、ユーザーの了承により広い探索を止め、9予測方式×4区間方式（一定・2区間・3等区間・自由区間）の36条件に絞った。以前の広い探索の途中結果は保存し、この36条件の選定には混ぜていない。固定比較から12件、絞り込んだ比較から8件を内部指標で選び、それぞれの公式評価前に候補を確定した。

## 8. 公式費用と、言えること・言えないこと

| 項目 | 金額 |
|---|---:|
| 公式工事費 CAPEX | $158,410,940.32 |
| 公式故障修理費 OPEX | $61,112,905.80 |
| 合計 | **$219,523,846.12** |
| Public score：2.5億ドル − 合計 | **$30,476,153.88** |
| 提出候補の次点の総費用 | $226,735,113.14 |
| 差 | **$7,211,267.02安い** |

故障なしの計画上の工事費は約 **$194.81m**、完了年は **2046年**。これは全交換が可能かを確認する対照条件で、公式総費用ではない。公式評価では故障修理で先に交換される管があり、その後の工事対象長や無料スキップが変わる。内部の「test年以外は故障なし」の総額も、25年総費用の推定値としては扱わない。

現時点の結果は、次の金額モデルや円の最適化の基準としてF168/R029を使う根拠になる。ただし「10%が本質的に最適」「可変閾値は常に悪い」「Ridgeだけが改善の原因」とまでは言えない。過去年はすでに多くの候補の比較に使っており、public首位も20提出からの選択である。privateでの優位や、過学習していないことは未確認である。


## 原資料

- [CMS001詳細](../../../notebooks/ensemble_for_cost_prediction/latest_conditional_stack/METHOD.ja.md)
- [F168詳細（既存公開ブランチ）](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/METHOD.ja.md)
