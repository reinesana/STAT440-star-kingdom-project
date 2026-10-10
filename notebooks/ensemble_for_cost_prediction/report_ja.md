# 修理コスト予測：単一モデルとアンサンブル候補の比較

最良単一モデルのCatBoostを正式な候補に追加しました。RMSEを最優先するならCatBoost、年間総額の偏りを抑えるなら縮小重みアンサンブルが有力です。従来の既定候補は縮小重みのまま保持し、単一モデルを含む候補集合を拡張しました。

## 評価結果

| candidate              |   mean_annual_RMSE |   pooled_RMSE |   annual_RMSE_sd |   mean_absolute_annual_total_bias_pct |   worst_absolute_annual_total_bias_pct |
|:-----------------------|-------------------:|--------------:|-----------------:|--------------------------------------:|---------------------------------------:|
| takito_catboost        |          124569.85 |     126180.72 |         25624.46 |                                 20.50 |                                  34.45 |
| ensemble_equal         |          125151.59 |     127044.21 |         25702.59 |                                  6.47 |                                  12.39 |
| ensemble_stable        |          125230.52 |     127113.11 |         25777.62 |                                  4.61 |                                  10.02 |
| ensemble_balanced      |          125255.35 |     127124.82 |         25856.33 |                                  6.38 |                                  13.33 |
| ensemble_rmse_weighted |          125329.07 |     127093.90 |         25167.36 |                                  7.67 |                                  21.30 |
| prior_surface_mean     |          125255.32 |     127022.98 |         25188.50 |                                  8.63 |                                  21.22 |

評価は2023〜2026年の8,592件。主指標は元単位でのRMSEです。年平均RMSEは各年のRMSEの単純平均、pooled RMSEは全件をまとめたRMSE、annual_RMSE_sdは4年のRMSEの標本標準偏差です。総額偏りは100×(予測総額/実績総額−1)で、表はその年別絶対値の平均・最大を示します。

CatBoostの年平均RMSEは124,569.85、縮小重みは125,230.52（約0.53%高い）。一方、最悪年の総額偏りはCatBoost34.45%に対し縮小重み10.02%です。縮小重みの年別RMSE標準偏差は基準モデルより約2.34%大きく、RMSEの年別安定化まで達成したとはいえません。総額の偏り改善とRMSEのばらつきは分けて判断してください。

## 各候補の説明

### 最良単一モデル：CatBoost
元単位の費用を予測するCatBoost単体。単一モデル中で年平均・プールRMSEが最小。ただし総額の偏りは大きい。

CSV: [predictions/takito_catboost.csv](predictions/takito_catboost.csv)

### 等重みアンサンブル
10構成モデルを各10%で平均。アンサンブル中では年平均RMSEが最小。

CSV: [predictions/ensemble_equal.csv](predictions/ensemble_equal.csv)

### 縮小重みアンサンブル（予算重視の既定候補）
過去OOFのRMSE重みに正則化を加え、その50%と等重み50%を混合。各構成モデルに最低5%の重みを残す。比較候補中で平均・最悪年の総額絶対偏りが最小。

CSV: [predictions/ensemble_stable.csv](predictions/ensemble_stable.csv)

### 制約付きバランスアンサンブル
過去データだけで年間総額偏り・相対年間RMSEのばらつき・等重みからの乖離を考慮。各重み1〜70%、RMSE制約付き。今回の実測偏りは縮小重みより大きい。

CSV: [predictions/ensemble_balanced.csv](predictions/ensemble_balanced.csv)

### RMSE最適化アンサンブル
過去の年平均RMSEを最小化する非負・合計1の重み。ゼロ重みを許すため一部モデルへの集中が残る。

CSV: [predictions/ensemble_rmse_weighted.csv](predictions/ensemble_rmse_weighted.csv)

### 表面種別の過去平均（比較基準）
全学習履歴の表面種別平均を全体平均へ縮小。縮小強度はOptunaで選択。

CSV: [predictions/prior_surface_mean.csv](predictions/prior_surface_mean.csv)

## モデルの内訳と最終重み

| component               |   takito_catboost |   ensemble_equal |   ensemble_stable |   ensemble_balanced |   ensemble_rmse_weighted |   prior_surface_mean |
|:------------------------|------------------:|-----------------:|------------------:|--------------------:|-------------------------:|---------------------:|
| prior_surface_mean      |              0.00 |            10.00 |             19.61 |               61.56 |                    74.44 |               100.00 |
| anthony_linear_raw      |              0.00 |            10.00 |              9.86 |                6.58 |                     0.00 |                 0.00 |
| anthony_gamma           |              0.00 |            10.00 |             11.08 |                1.00 |                     1.34 |                 0.00 |
| anthony_rf_log_mean     |              0.00 |            10.00 |             10.39 |                1.00 |                     5.68 |                 0.00 |
| takito_xgboost          |              0.00 |            10.00 |              6.05 |                1.00 |                     4.55 |                 0.00 |
| takito_catboost         |            100.00 |            10.00 |              9.97 |                1.00 |                     4.50 |                 0.00 |
| takito_lightgbm         |              0.00 |            10.00 |              5.00 |                1.00 |                     0.00 |                 0.00 |
| rion_gamma_refit        |              0.00 |            10.00 |              5.32 |                1.00 |                     0.00 |                 0.00 |
| rion_rf_raw_refit       |              0.00 |            10.00 |              7.36 |                1.00 |                     0.00 |                 0.00 |
| rion_lgb_log_mean_refit |              0.00 |            10.00 |             15.35 |               24.86 |                     9.48 |                 0.00 |

重みは%。各アンサンブルは同じ10構成モデルを使います。表面平均、Anthonyの線形回帰・Gamma・対数Random Forest、TakitoのXGBoost・CatBoost・LightGBM、RionのGamma・Random Forest・対数LightGBMです。対数系は元単位の期待費用に戻す補正も過去データだけで選択しました。

## 実施した処理とリーケージ対策

1. train.csvの2019〜2026年全14,113件を保持し、pipes.csvとIDで結合。欠損のある6件も復元し、高額費用を削除・上限処理しませんでした。
2. 各予測年より前のラベルだけで、年初時点の年齢、材料・表面、距離、近傍在庫、端点経路、過去費用集約を作成。距離はm。平面交差を物理接続と見なしません。
3. 各モデル・各学習期間2021〜2027で30trial。内側は直前の最大2年で順方向検証し、特徴量6群も探索。2020は過去年が1年だけのため固定設定。全2,100試行枠のうち正常完了1,978、失敗・中断122です。
4. 線形回帰はOLS/Ridge/Lasso/ElasticNetを比較。XGBoostとLightGBMはL1/L2、GammaとCatBoostはL2、RFは木の深さ・葉の件数・抽出率等で過学習を抑制。最終Anthony線形回帰では正則化なしが選択されました。GammaのL1は未比較です。
5. 重みは過去年のOOFだけで学習。2027モデルは全14,113件で学習済み。seed=440。既存のShana予測や将来利用不能な故障時刻・曜日を使う予測は除外。
6. 保存10モデルの再生、80期間の設定、全指標・重みの独立再計算、将来費用・修理日の変更不変性を検証済み。

## 最良単一モデルの設定

```json
{
  "feature_pack": "topology",
  "iterations": 150,
  "depth": 2,
  "learning_rate": 0.02932217209384241,
  "l2_leaf_reg": 1.0143268892104178,
  "random_strength": 0.00153216083646107,
  "rsm": 0.7357430693448022,
  "border_count": 128,
  "one_hot_max_size": 6,
  "leaf_estimation_iterations": 4,
  "bootstrap_type": "MVS",
  "subsample": 0.99792074633296
}
```

最終CatBoostはtopology特徴量49列、深さ2、150反復を選択。これは2027用の設定で、各評価年にはその年の過去だけで選択した別の設定を使っています。全10モデルの設定はconfigs/とevidence/final_model_settings.csvに保存しました。

## CSVの対象と解釈

実ファイル名はpipes.csvです。全43,039本からtrain.csvの14,113個のIDを除外し、残る28,926本を全候補で予測しました。各CSVはid,predicted_costの2列で、コスト降順（同額はID昇順）です。予測年は2027、履歴は2026年末で固定。既知材料27,240本の予測を保存済み全件学習モデルから使用し、追加の1,686本も同じ保存モデルで推定しました。追加内訳はPU1,678本、材料不明8本で、学習に該当材料の費用がありません。未知カテゴリ処理を通した外挿であり、正当なOOF評価に含まれず、精度・校正を確認できません。これらを0円と置かず数値を出していますが、supported_materialの推定と同等の信頼度で予算に使うべきではありません。evidence/prediction_scope.csvで識別できます。

全値は故障した場合の条件付き費用で、必要修理量や故障確率ではありません。CSVの単純合計は年間予算を意味しません。修理予算には故障確率・件数との組合せが必要です。金額単位は元データの単位で、円と断定しません。評価結果を参照した後の候補選択であり、未使用の最終ホールドアウトによる性能保証ではありません。

## 年別結果

| model                  |   year |   n_test |      RMSE |   total_bias_pct |
|:-----------------------|-------:|---------:|----------:|-----------------:|
| prior_surface_mean     |   2023 |     1826 | 158531.63 |             4.93 |
| takito_catboost        |   2023 |     1826 | 158723.20 |            10.34 |
| ensemble_equal         |   2023 |     1826 | 159167.41 |             3.47 |
| ensemble_rmse_weighted |   2023 |     1826 | 158626.76 |             5.33 |
| ensemble_stable        |   2023 |     1826 | 159453.14 |            -1.95 |
| prior_surface_mean     |   2024 |     1955 |  99232.67 |            -1.01 |
| takito_catboost        |   2024 |     1955 |  97780.40 |            28.78 |
| ensemble_equal         |   2024 |     1955 |  99065.40 |             5.96 |
| ensemble_rmse_weighted |   2024 |     1955 |  99458.85 |            -2.62 |
| ensemble_stable        |   2024 |     1955 |  99245.82 |             2.59 |
| prior_surface_mean     |   2025 |     1773 | 114824.73 |            21.22 |
| takito_catboost        |   2025 |     1773 | 115475.00 |            34.45 |
| ensemble_equal         |   2025 |     1773 | 113674.62 |            12.39 |
| ensemble_rmse_weighted |   2025 |     1773 | 114737.81 |            21.30 |
| ensemble_stable        |   2025 |     1773 | 113590.50 |            10.02 |
| prior_surface_mean     |   2026 |     3038 | 128432.27 |             7.36 |
| takito_catboost        |   2026 |     3038 | 126300.80 |             8.43 |
| ensemble_equal         |   2026 |     3038 | 128698.93 |            -4.06 |
| ensemble_rmse_weighted |   2026 |     3038 | 128492.84 |             1.43 |
| ensemble_stable        |   2026 |     3038 | 128632.64 |            -3.90 |
| ensemble_balanced      |   2023 |     1826 | 159543.95 |            -9.96 |
| ensemble_balanced      |   2024 |     1955 |  98941.25 |             1.75 |
| ensemble_balanced      |   2025 |     1773 | 113997.86 |            13.33 |
| ensemble_balanced      |   2026 |     3038 | 128538.33 |            -0.50 |

## 出典・検証

Source snapshot: `a09355d835d9acbb5746f11edcddd0cfd5e8ec3e`. [Training labels](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/train.csv), [Inventory](https://github.com/reinesana/STAT440-star-kingdom-project/blob/a09355d835d9acbb5746f11edcddd0cfd5e8ec3e/data/pipes.csv).
[Sources receipt](evidence/sources.html) · [Temporal validation](evidence/validation.json) · [Export checks](evidence/export_validation.json) · [Per-pipe support](evidence/prediction_scope.csv)
