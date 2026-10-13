# Experiments on October 13 by Takito

10/13 By Takito · 2026

## English

### Purpose and main finding

We compared six ways to order pipe replacement work in the Star Kingdom project. Candidate 06 was best: −$15.61 million, about $9.97 million better than the oldest-first baseline. All candidates retained the same circles; only their order changed.

The project replaces original iron pipes during 2027–2051, with $10 million added to the construction budget each year. Score = $250 million − construction costs − leak repair costs over the project period. Higher scores are better. A negative score means combined costs exceeded the allocation used in the score; it does not alone imply a violation of construction-budget constraints.

Each CSV row gives a circular work area’s center and radius; row order determines construction order. A pipe is replaced only when both endpoints are inside the circle. We kept one circle per original iron pipe, centered at its midpoint with half its endpoint length plus a 0.000001 m radius margin. Coordinates and radii are metres.

## What we compared and the results

01–04 compare four repair-cost forecasts using the same leak probabilities. Expected repair cost means predicted repair cost if a leak occurs multiplied by the probability of a leak in 2027–2028. Circle priorities sum the contributions of fully contained, unreplaced pipes. 05 uses repair cost alone. 06 divides the expected repair cost by each circle’s initial construction cost.

| Candidate | Priority rule | Score M$ | Change vs baseline M$ |
| --- | --- | --- | --- |
| Baseline | Oldest first | -25.58 | — |
| 01 | CatBoost × leak probability | -27.54 | -1.96 |
| 02 | Shrinkage ensemble × leak probability | -25.70 | -0.12 |
| 03 | Equal-weight ensemble × leak probability | -25.58 | +0.00 |
| 04 | RMSE-weighted ensemble × leak probability | -27.16 | -1.58 |
| 05 | CatBoost repair cost only | -35.45 | -9.87 |
| 06 | Shrinkage expected repair cost ÷ construction cost | -15.61 | +9.97 |

M$ means million dollars. Positive changes mean improvement. Scores and differences are approximate because the evaluator displays rounded values. Candidate 03 tied with the baseline at the displayed precision.

## What we learned

01–04 did not beat the baseline; 03 tied at the displayed precision. Cost alone performed worst: 05 scored −$35.45 million. Adding leak probability improved the same CatBoost approach by $7.91 million (01 versus 05).

Including construction cost gave the largest gain: 06 beat 02 by $10.09 million and the baseline by $9.97 million. This supports choosing work by expected benefit per construction dollar. Total scores alone cannot separate construction savings from avoided leak repairs.


## Candidate 06 explained simply

Candidate 06 is a way to decide **which circle to work on first**. It does not learn or optimize circle sizes. The cost ensemble predicts repair costs; a separate model predicts leak probabilities; a final rule turns these predictions into a construction order.

### Step 1 Draw a circle around each original iron pipe

The center is halfway between the pipe’s two endpoints. The radius is half the distance between those endpoints, plus a tiny 0.000001 m margin to prevent rounding from leaving an endpoint outside. For example, endpoints 40 m apart give a radius of approximately 20 m. This is endpoint geometry, not a measured curved pipe length.

We kept all 38,455 baseline circles. Their centers and radii are the same in all six files. A circle can also contain other pipes; any fully contained, currently unreplaced pipe is included in its work cost. **Candidate 06 changes the order, not the size, of the circles.**

### Step 2 Predict repair cost using the 2027 scenario

Yes: **every repair-cost forecast used here assumes the year is 2027**. The model inputs use scenario year 2027, pipe age on 1 January 2027, and observed repair history through the end of 2026. No actual future leak month or day is supplied.

The forecast asks, “If this pipe leaks in the 2027 scenario, how much might the repair cost?” It does not predict the preventive construction bill. Even pipes whose work is delayed until 2028 or later keep their 2027 repair-cost estimate in this experiment; costs are not re-predicted for their eventual work year.

### Step 3 Multiply by the chance of a leak

The separate risk model estimates leak probabilities for 2027 and 2028, updating pipe age for each year. These are combined into a two-year probability. **Repair cost uses 2027; leak probability covers 2027–2028.**

For example, a $100,000 predicted repair cost and a 10% two-year leak probability give $10,000 of expected repair cost. We sum these amounts for all fully contained, unreplaced pipes in each circle.

### Step 4 Put the circles with more expected benefit per construction dollar first

**Priority = expected repair cost inside the circle ÷ initial construction cost of the circle.**

| Illustrative circle | Expected repair cost | Construction cost | Priority |
| --- | --- | --- | --- |
| A | $100,000 | $50,000 | 2 |
| B | $150,000 | $150,000 | 1 |

These are teaching examples, not actual circles from the submissions. Candidate 06 puts A first: although B has a larger expected repair cost, A offers more expected benefit for each construction dollar.

The result is one fixed ordered list. Annual budgets determine when the evaluator can execute each circle. **The plan does not update predictions or reorder circles each year, and it does not optimize the full 25-year schedule.** The initial construction-cost estimate also stays fixed during ranking; actual costs can change when preceding work or leaks replace pipes.


## How Candidate 06 Was Built

Candidate 06 combined three quantities: the cost of a leak, the probability of a leak, and the cost of preventive replacement. The repair-cost estimate came from the team’s optimized ensemble. A separate model supplied leak probabilities. We then used these predictions to order the same construction circles as the baseline.

### 1. Training data and features

We joined train.csv with pipes.csv by Pipe ID and retained all 14,113 repair records from 2019–2026. Records with missing installation dates were retained, and unusually expensive repairs were not removed or capped.

The cost models used information available before the prediction year, including pipe material, surface type, age, endpoint length, location, nearby pipes, endpoint-route features, and historical repair-cost summaries. Different models selected different feature groups. Actual future leak dates, hours, and weekdays were not used.

The target was repair cost conditional on a leak occurring. This is different from preventive construction cost and does not, by itself, predict whether a pipe will leak.

### 2. Component models and model selection

The ensemble contained nine models developed from Takito’s, Anthony’s, and Rion’s work, plus a surface-based reference forecast.

| Component | Main approach | Final weight |
| --- | --- | --- |
| Surface reference | Historical surface-specific mean, smoothed toward the overall mean | 19.61% |
| Anthony linear | Linear regression predicting cost directly | 9.86% |
| Anthony Gamma | Gamma regression for positive costs | 11.08% |
| Anthony Random Forest | Predict log cost, then convert back to cost | 10.39% |
| Takito XGBoost | Boosted trees predicting cost directly | 6.05% |
| Takito CatBoost | Boosted trees predicting cost directly | 9.97% |
| Takito LightGBM | Boosted trees predicting cost directly | 5.00% |
| Rion Gamma | Gamma regression with a different feature specification | 5.32% |
| Rion Random Forest | Predict cost directly | 7.36% |
| Rion LightGBM | Predict log cost, then convert back to cost | 15.35% |

Weights are rounded; the calculation used the saved full-precision values.

Model selection followed time order. For each prediction year, models were trained and tuned using only earlier years. Optuna searched 30 configurations per component and prediction year for 2021–2027, comparing feature groups and model settings. The objective was average annual RMSE on the two most recent available earlier validation years, or one year when only one was available. RMSE measures prediction error and gives larger errors more weight.

The search included regularization for regression and boosting models, and settings such as tree count, depth, leaf size, learning rate, and sampling. For log-cost models, the method used to convert predictions back to expected cost was also selected using past data. For example, the final CatBoost used 49 features, depth 2, and 150 iterations.

Existing Shana prediction files were excluded because evaluation-year outcomes had already influenced their feature or model selection. Models requiring actual future failure times were also excluded. Therefore, the ensemble used the ten eligible components, rather than every existing team prediction file.

### 3. How the shrinkage ensemble was constructed

We compared equal weights, weights optimized for RMSE, shrinkage weights, and a constrained balance ensemble.

Candidate 06 used the shrinkage ensemble. First, we fitted nonnegative weights that summed to one using historical predictions made without training on their target year. These are called out-of-fold, or OOF, predictions. The optimization minimized average annual RMSE and included a penalty discouraging large departures from equal weights.

We then mixed those optimized weights with equal weights:

Final weight = 50% × regularized optimized weight + 50% × 10%.

This gives every component at least a 5% weight. “Shrinkage” here means moving the weights toward equal weighting, reducing dependence on a small number of models.

For each historical evaluation year, weights were learned only from earlier OOF years. The final 2027 weights used 13,035 OOF predictions from 2020–2026. The 2019 records could not have past-only OOF predictions because no earlier training history was available, but they were included in the final component-model training.

We selected the deployment ensemble among diversified candidates whose average annual RMSE was within 1% of the best diversified ensemble. We then preferred smaller worst-year total-cost bias, followed by smaller average absolute bias and lower relative variation in annual RMSE.

The shrinkage ensemble was selected because it offered a useful balance between individual prediction error and total-cost bias:

| Historical metric, evaluated on 2023–2026 repairs | CatBoost alone | Shrinkage ensemble |
| --- | --- | --- |
| Average annual RMSE | 124,569.85 | 125,230.52 |
| Mean absolute annual total-cost bias | 20.50% | 4.61% |
| Worst absolute annual total-cost bias | 34.45% | 10.02% |

Its average annual RMSE was about 0.53% higher than CatBoost’s, but its annual predicted totals were closer to the observed totals. This does not establish better performance in every year. The deployment choice was made after reviewing historical results, so those results are not an untouched final test.

### 4. Full-data refitting and prediction

After selecting the settings, each component was refitted on all 14,113 records through the end of 2026. The saved ensemble weights were then applied to the component predictions.

We generated 2027 cost predictions for 28,926 pipes in pipes.csv that had no recorded historical leak. Historical-leak pipes were treated as already replaced. Of the prediction targets, 27,240 had materials represented in the cost training data; 1,678 polyurethane pipes and eight pipes with missing material required extrapolation.

For construction priorities, polyurethane and historically replaced pipes received zero contribution. Predictions for missing-material pipes remained uncertain. The same full-data ensemble predictions were used by candidates 02 and 06.

### 5. Leak probability and construction priority

Leak probability came from a separate LightGBM classifier, not the repair-cost ensemble. It used material, surface, length, age, and midpoint coordinates. Two configurations were compared using 2025 and 2026 validation log loss, a measure of probability-prediction error. The selected model was refitted on the historical pipe-year observations and used to estimate annual leak probabilities.

The two-year probability was calculated as:

Probability of a leak in 2027–2028 = 1 − (1 − p2027) × (1 − p2028).

We multiplied this probability by the ensemble’s 2027 repair-cost prediction. For each circle, we summed these expected repair costs across all fully contained, unreplaced pipes, including copper and brass.

Candidate 06 then calculated:

Circle priority = summed expected repair cost ÷ initial circle construction cost.

The construction-cost estimate included the fixed work-unit charge, surface excavation cost, and replacement length. Surface types were counted once each, including those associated with already replaced pipes. Only unreplaced pipes contributed replacement length.

We ordered circles from highest to lowest priority. The baseline’s 38,455 circles were retained, with no changes to their centers or radii. Thus, candidate 06 tested a different ordering rule rather than a new circle design.

### 6. Validation and interpretation

The exported CSV was checked for complete iron-pipe coverage and construction-budget feasibility. Under a no-future-leak simulation, candidate 06 finished in 2042 with construction costs of approximately $152.12 million. These are local validation results, not the evaluator’s realized construction costs or completion year.

The submitted score was −$15.61 million, approximately $9.97 million better than the baseline. Compared with candidate 02, which used the same ensemble and leak probabilities without dividing by construction cost, candidate 06 improved by $10.09 million.

This supports the construction-cost denominator in this experiment. However, the priority is a heuristic: it uses a fixed two-year risk window, overlapping circles can share benefits, and the initial construction-cost denominator is not updated for preceding work or future leaks. It does not represent a full optimization of the 25-year project.

### Scope of this experiment

These six files use the separate LightGBM risk model described above. They do not use the newer leak-probability ensemble now present in notebooks/leak_model_comparison. Cost-ensemble selection and leak-risk-model selection are separate processes.

Each candidate has one submitted evaluation, with no uncertainty estimate. All scores remain negative, and candidate 06 is not proven optimal. Its two-year expected repair cost is a ranking proxy rather than a validated estimate of total avoided cost over 25 years.

All six exported CSVs passed full original-iron coverage and no-future-leak budget checks. Their simulated completion year was 2042 and construction costs ranged from $151.39 million to $152.12 million. Validation includes full endpoint containment, surface-type costs, historical replacements, overlaps, zero-cost skips, annual carryover, and stopping at the first unaffordable unit.

Use 06 as the reference for future tests of circle grouping and longer-term replacement scheduling.

## Sources

Submitted scores come from the user-supplied [submission screenshot](assets/experiments_10_13_takito_scores.png). Rows are mapped bottom to top as baseline, then 01–06. The CSV files contain construction circles, not evaluation scores.

The baseline CSV is [2026-10-05_baseline_v1.csv](../submissions/2026-10-05_baseline_v1.csv). The six original submission files are:

- [01_catboost_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/01_catboost_risk.csv)
- [02_stable_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/02_stable_risk.csv)
- [03_equal_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/03_equal_risk.csv)
- [04_rmse_weighted_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/04_rmse_weighted_risk.csv)
- [05_catboost_cost_only.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/05_catboost_cost_only.csv)
- [06_stable_risk_per_work_cost.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/06_stable_risk_per_work_cost.csv)

Model and implementation records: [cost-model report](../notebooks/ensemble_for_cost_prediction/report_en.md), [training contract](../notebooks/ensemble_for_cost_prediction/evidence/training_contract.json), [ensemble weights](../notebooks/ensemble_for_cost_prediction/evidence/ensemble_weights.json), [final model settings](../notebooks/ensemble_for_cost_prediction/evidence/final_model_settings.csv), [submission generator](../src/build_cost_submissions.py), [risk-model training](../src/risk_priority.py), and [local validation and provenance](../submissions/2026-10-13_cost_candidates/README.md).

# Takitoが10月13日にやった実験

2026年10月13日 · Takito

## 日本語

### 目的と主な結果

Star Kingdom課題で、管の交換工事をどの順に行うか、6種類の方法を比較した。最良は06の−15.61百万ドルで、古い管から交換するbaselineより約9.97百万ドル改善した。円の形は揃え、順序だけを変更した。

対象期間は2027〜2051年で、毎年1,000万ドルが工事予算に追加される。スコア＝250百万ドル−工事費−期間中の漏れ修理費で、高いほど良い。マイナスは合計費用がスコア計算上の配分額を超えたことを意味し、それだけで工事予算制約への違反を意味するものではない。

CSVの各行は円の中心と半径を表し、行の順番が工事順になる。管の両端が円に入る場合だけ交換される。元の鉄管1本につき1つの円を使い、中心は中点、半径は両端間の長さの半分＋0.000001 mとした。座標と半径の単位はmである。

## 比較した方法と結果

01〜04は同じ漏れ確率を使い、修理費予測モデルを比較した。期待修理費は「漏れた場合の予測修理費×2027〜2028年の漏れ確率」で、円内に完全に含まれる未交換管の寄与を合計した。05は修理費だけを使い、06は円の期待修理費を初期工事費で割った。

| 候補 | 優先順位の決め方 | スコア 百万ドル | baselineからの差 百万ドル |
| --- | --- | --- | --- |
| Baseline | 古い管から交換 | -25.58 | — |
| 01 | CatBoost × 漏れ確率 | -27.54 | -1.96 |
| 02 | 縮小重みアンサンブル × 漏れ確率 | -25.70 | -0.12 |
| 03 | 等重みアンサンブル × 漏れ確率 | -25.58 | +0.00 |
| 04 | RMSE重みアンサンブル × 漏れ確率 | -27.16 | -1.58 |
| 05 | CatBoostの修理費のみ | -35.45 | -9.87 |
| 06 | 縮小重みの期待修理費 ÷ 工事費 | -15.61 | +9.97 |

単位は百万ドル。差がプラスなら改善。提出画面の丸め済み表示による概数であり、03とbaselineは表示精度で同点だった。

## 分かったこと

01〜04はbaselineを上回らなかった。修理費だけの05は−35.45百万ドルで最も悪かった。同じCatBoostに漏れ確率を加えた01は、05より7.91百万ドル良かった。

工事費の考慮が最大の改善につながった。06は02より10.09百万ドル、baselineより9.97百万ドル良かった。工事費1ドルあたりの期待効果で工事を選ぶ方法が有力である。ただし、総スコアだけでは工事費削減と漏れ修理費削減を分けて評価できない。


## 候補06を簡単に説明すると

06は、**どの円から工事するかを決める方法**である。円の大きさを学習・最適化しているわけではない。コストアンサンブルで修理費、別のモデルで漏れ確率を予測し、最後にルールを使って工事順序を決める。

### 手順1 元の鉄管を囲む円を作る

中心は管の両端のちょうど真ん中。半径は両端間の距離の半分に、丸め誤差で端点が外に出ないよう0.000001 mの余裕を加える。例えば両端間が40 mなら半径は約20 mである。これは端点の座標から決める形で、曲がった管の実測長ではない。

baselineの38,455個の円をそのまま使い、6ファイルで中心・半径を揃えた。円には他の管も入ることがあり、完全に含まれる未交換管は工事費の計算に含める。**06で変えたのは円の大きさではなく、工事する順番である。**

### 手順2 2027年という条件で修理費を予測する

**今回の修理コスト予測はすべて2027年を仮定している。** 入力は予測年2027年、2027年1月1日時点の管の年齢、2026年末までの修理履歴である。実際に将来漏れる月・日は入力していない。

予測するのは「2027年という条件でこの管が漏れたら、修理費はいくらか」で、予防交換の工事費ではない。工事が2028年以降に遅れる管にも、今回は2027年の修理費予測を使い続ける。実際の工事年に合わせた再予測はしていない。

### 手順3 漏れる可能性を掛ける

別のリスクモデルで2027年と2028年の漏れ確率を求め、各年の年齢を反映した上で2年間の漏れ確率にまとめる。**コスト予測は2027年、漏れ確率は2027〜2028年という組み合わせである。**

例えば、漏れた場合の予測修理費が10万ドルで、2年間の漏れ確率が10%なら、期待修理費は1万ドルになる。円に完全に含まれる未交換管について、この金額を合計する。

### 手順4 工事費に対して期待効果が大きい円から並べる

**優先度＝円内の期待修理費合計÷その円の初期工事費。**

| 説明用の円 | 期待修理費 | 工事費 | 優先度 |
| --- | --- | --- | --- |
| A | 10万ドル | 5万ドル | 2 |
| B | 15万ドル | 15万ドル | 1 |

これは説明用の例で、提出ファイル内の実際の円ではない。06はAを先に工事する。Bの方が期待修理費は大きいが、Aは工事費1ドルあたりの期待効果が大きいためである。

こうして1つの固定された順序を作る。各円を実際に何年に工事できるかは、評価側の年間予算で決まる。**毎年予測を更新したり、円の順位を決め直したりする計画ではなく、25年間全体を最適化したものでもない。** 順位計算時の初期工事費も固定であり、先行工事や漏れによって交換済み管が増えると、実際の工事費は変わり得る。


## 候補06をどのように作ったか

候補06は、漏れた場合の修理費、漏れる確率、予防交換の工事費を組み合わせた。修理費にはチームの最適化済みアンサンブルを使い、漏れ確率は別のモデルから求めた。その予測を使って、baselineと同じ円の工事順序を変更した。

### 1 学習データと特徴量

train.csvとpipes.csvをPipe IDで結合し、2019〜2026年の修理履歴14,113件をすべて使用した。敷設日が欠損している記録も残し、極端に高い修理費も削除したり上限を設けたりしなかった。

特徴量には、予測年より前に利用できる材質、地表、年齢、両端間の長さ、位置、近隣管、端点から推定した経路、過去の修理費集計などを使った。モデルごとに特徴量群を選択し、実際の将来の漏れ日・時刻・曜日は使用しなかった。

予測対象は「漏れが起きた場合の修理費」である。予防交換の工事費とは異なり、この予測だけでは漏れるかどうかは分からない。

### 2 構成モデルとモデル選択

Takito・Anthony・Rionのモデル9種類と、地表別の平均修理費を使う基準予測を組み合わせた。

| 構成 | 方法 | 最終重み |
| --- | --- | --- |
| 地表基準 | 地表別平均を全体平均へ縮小 | 19.61% |
| Anthony 線形 | 修理費を直接予測 | 9.86% |
| Anthony Gamma | 正の費用のGamma回帰 | 11.08% |
| Anthony Random Forest | 対数予測を金額に戻す | 10.39% |
| Takito XGBoost | 修理費を直接予測 | 6.05% |
| Takito CatBoost | 修理費を直接予測 | 9.97% |
| Takito LightGBM | 修理費を直接予測 | 5.00% |
| Rion Gamma | 別の特徴量設定のGamma回帰 | 5.32% |
| Rion Random Forest | 修理費を直接予測 | 7.36% |
| Rion LightGBM | 対数予測を金額に戻す | 15.35% |

表の重みは丸めた値で、実際には保存した丸め前の値を使った。

モデル選択は時間順に行い、各予測年より前のデータだけで学習・調整した。2021〜2027年について、各構成・各予測年でOptunaを使って30通りの設定を探索した。利用できる直近の過去2年、1年しかない場合はその1年の検証で、年平均RMSEを小さくする設定を選んだ。RMSEは予測誤差の指標で、大きな誤差を強く評価する。

探索には、回帰・勾配ブースティングの正則化、木の本数・深さ、葉の最小件数、学習率、データ・特徴量の抽出率を含めた。対数修理費を予測するモデルでは、金額に戻す補正も過去データだけで選択した。例えば最終CatBoostは49特徴量、木の深さ2、150反復を使った。

既存のShana予測は、評価年の結果が特徴量やモデルの選択に使われていたため除外した。また、実際の将来の故障時刻を必要とするモデルも除外した。したがって、既存のチーム予測をすべて混ぜたのではなく、条件を満たす10構成を使用した。

### 3 縮小重みアンサンブルの作り方

等重み、RMSEを最小化する重み、縮小重み、制約付きバランス重みを比較した。

06には縮小重みアンサンブルを使った。まず、対象年を学習に使わずに作った過去の予測から、非負で合計1になる重みを求めた。この予測をOOF予測と呼ぶ。重みの学習では、年平均RMSEを小さくすると同時に、等重みから大きく離れることにペナルティを加えた。

その後、最適化した重みと等重みを半分ずつ混ぜた。

**最終重み＝正則化付き最適化重み×50%＋10%×50%。**

これにより、すべての構成に最低5%の重みが残る。「縮小」は重みを等重みに近づけ、少数モデルへの依存を抑えることを意味する。

過去の各評価年では、その年より前のOOF予測だけで重みを学習した。最終2027年用の重みには2020〜2026年のOOF予測13,035件を使用した。2019年はそれより前の学習履歴がなくOOF予測を作れないが、その記録も最終的な構成モデルの学習には含めた。

採用するアンサンブルは、全構成に重みを残す候補のうち、年平均RMSEが最良候補から1%以内のものを対象に選んだ。その中で、年間総額の最悪年の偏り、平均絶対偏り、年別RMSEの相対的なばらつきの順に比較した。

縮小重みを選んだ理由は、個別予測の誤差と総額の偏りのバランスが良かったためである。

| 2023〜2026年の履歴による指標 | CatBoost単体 | 縮小重み |
| --- | --- | --- |
| 年平均RMSE | 124,569.85 | 125,230.52 |
| 年間総額の絶対偏りの平均 | 20.50% | 4.61% |
| 年間総額の絶対偏りの最大 | 34.45% | 10.02% |

年平均RMSEはCatBoostより約0.53%高かったが、年間の予測総額は実績に近かった。すべての年で性能が良いことを意味するわけではない。採用方法は過去の結果を確認して選んだため、この評価は完全に未使用の最終テストではない。

### 4 全データでの再学習と予測

設定選択後、各構成を2026年末までの全14,113件で再学習し、保存した重みで予測を組み合わせた。

2027年の予測対象は、pipes.csvのうち過去に漏れが記録されていない28,926本である。履歴にある管は交換済みとして扱った。27,240本は学習データに存在する材質で、ポリウレタン1,678本と材質不明8本は外挿による予測だった。

工事順位では、ポリウレタンと過去に交換済みの管の寄与を0にした。材質不明管の予測には不確実性が残る。02と06には同じ全データ再学習済みのアンサンブル予測を使った。

### 5 漏れ確率と工事の優先順位

漏れ確率は修理費アンサンブルとは別のLightGBM分類モデルから求めた。特徴量は材質、地表、長さ、年齢、中点座標である。2通りの設定を2025年・2026年の検証log lossで比較した。log lossは確率予測の誤りを評価する指標である。選んだモデルを過去の管・年単位のデータで再学習し、年間の漏れ確率を推定した。

**2027〜2028年に漏れる確率＝1−（1−2027年の確率）×（1−2028年の確率）。**

この確率にアンサンブルの2027年修理費予測を掛けた。円ごとに、完全に含まれる未交換管の期待修理費を合計し、銅管・真鍮管も含めた。

**円の優先度＝円内の期待修理費合計÷円の初期工事費。**

初期工事費には固定費、地表工事費、交換長さを含めた。地表は各種類を1回ずつ数え、交換済み管の地表も含めた。交換長さには未交換管だけを含めた。

優先度が高い円から順に並べた。baselineの38,455個の円を維持し、中心・半径は変えなかった。06は円の設計ではなく、工事順序の比較である。

### 6 検証と結果の意味

出力したCSVについて、鉄管の完全被覆と工事予算内での実行可能性を確認した。将来漏れなしのシミュレーションでは、06は2042年に完了し、工事費は約152.12百万ドルだった。これはローカル検証であり、提出評価の実際の工事費・完了年ではない。

提出スコアは−15.61百万ドルで、baselineから約9.97百万ドル改善した。同じアンサンブルと漏れ確率を使い、工事費で割らなかった02に対しても10.09百万ドル改善した。

この比較は、今回の実験で工事費を分母に加えることの有効性を示している。ただし優先順位は近似的な方法である。固定の2年間リスクを使い、重複する円は同じ期待効果を含み得る。初期工事費も先行工事や将来の漏れに合わせて更新していないため、25年間全体の最適化ではない。


### 今回の実験の範囲

この6ファイルは上記の別途学習したLightGBMリスクモデルを使用した。現在notebooks/leak_model_comparisonにある新しい漏れ確率アンサンブルは使用していない。コストアンサンブルと漏れリスクモデルの選択は別の処理である。

各候補は1回の提出評価で、不確実性の推定はない。すべてのスコアはまだマイナスで、06が最適解とは示されていない。2年間の期待修理費は順位を決める近似指標で、25年間の削減額として検証された推定ではない。

全6CSVで元の鉄管の完全被覆と将来漏れなしの予算検証を通過した。シミュレーションの完了年は2042年、工事費は151.39〜152.12百万ドルだった。両端の完全包含、地表費、履歴による交換、円の重複、費用0でのスキップ、予算繰越、予算不足の先頭工事での停止を検証した。

次は06を比較基準にして、円のまとめ方と長期の工事順序を改善する。

## 出典

提出スコアは共有された[提出画面](assets/experiments_10_13_takito_scores.png)に基づく。下からbaseline、01〜06の順に対応付けた。提出CSVに入っているのは工事用の円で、評価スコアではない。

baselineは[2026-10-05_baseline_v1.csv](../submissions/2026-10-05_baseline_v1.csv)。今回の6提出ファイルは以下のとおり。

- [01_catboost_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/01_catboost_risk.csv)
- [02_stable_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/02_stable_risk.csv)
- [03_equal_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/03_equal_risk.csv)
- [04_rmse_weighted_risk.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/04_rmse_weighted_risk.csv)
- [05_catboost_cost_only.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/05_catboost_cost_only.csv)
- [06_stable_risk_per_work_cost.csv](https://github.com/reinesana/STAT440-star-kingdom-project/blob/e9650d54e2f89c1724442154e6504184f8d62f59/submissions/2026-10-13_cost_candidates/06_stable_risk_per_work_cost.csv)

モデル・実装記録：[コストモデル説明](../notebooks/ensemble_for_cost_prediction/report_ja.md)、[学習記録](../notebooks/ensemble_for_cost_prediction/evidence/training_contract.json)、[重み](../notebooks/ensemble_for_cost_prediction/evidence/ensemble_weights.json)、[最終設定](../notebooks/ensemble_for_cost_prediction/evidence/final_model_settings.csv)、[提出生成コード](../src/build_cost_submissions.py)、[リスクモデル学習](../src/risk_priority.py)、[ローカル検証と予測の出所](../submissions/2026-10-13_cost_candidates/README.md)。
