# 最新の推奨修理費モデル — 2026年10月28日

**Model: CMS001 — Material-Surface Cost Stack / 材質・地表面別コストスタック (v1.0.0).** [Current selection](../latest_model.json) · [Model registry](../model_registry.json) · [How to switch](../MODEL_MANAGEMENT.md)
[日本語：仕組み・従来比較・旧手動補正の扱い](METHOD.ja.md) · [English methods](METHOD.en.md)

**最新の推奨は、材質×地表面別・RMSE重視の条件付きstacking。** CatBoost・平均補正あり対数RF・平均補正あり対数LightGBMを組み合わせ、漏水した場合の修理金額を予測する。

**共有Google Docのチームアンサンブルとの直接比較も実施した。** DocのアンサンブルはXGBoost・CatBoost・LightGBM各1/3の漏水確率モデルであり、比較対象は最新の時間予測F168である。同一132,121管年（2023〜2026年）では、F168の年平均Brierは0.050400から0.045625へ**9.47%改善**し、平均PR-AUCと年間漏水件数の平均絶対偏りも良好だった。2023年のBrierと平均log lossは従来モデルが優れるため、優位な指標と例外を[英日記事](METHOD.ja.md#共有google-docの皆で作ったアンサンブルとの直接比較)に明記した。以下の修理費RMSE比較とは別の検証である。

同じ2024〜2026年、6,766件の内部比較で平均RMSEは **$112,943.12**、MAEは **$23,375.50**。保存済みの従来24構成すべてより平均RMSEが低く、従来の安定型比0.77%、15入力最適化stack比0.58%改善した。今回の新しいRidgeの方が平均RMSEは低いが、条件付きstackingは年別の悪化幅も考慮したユーザー承認済みの推奨である。等重み平均が総額の偏りでは優れ、最新年では本モデルもCatBoost比0.53%悪化している。

**時間予測F168では、旧16b／17bの設置年別確率の手動上書きを採用せず、設置年・管齢・過去のコホート故障率などを特徴量として学習する。** 旧補正は現在の方針には不適切とユーザーが判断し、旧計画より表示スコアが低くても手続きの妥当性を優先する。スコア差を補正削除だけの結果とは断定しない。修理費モデルの対数平均補正は別処理として残している。

This is the current **stability-oriented recommendation for conditional repair amount**, not a claim of lowest error on every metric or higher official planning score than every earlier team plan. The English/Japanese articles document the common-year comparison, feature-based timing approach and limits.

## 学習済みモデルと2027年予測

依頼に基づき、2019〜2026年の14,113件で3基底モデルを作り直し、2023〜2026年の時系列OOF 8,592件で条件別の重みを再学習した。ハイパーパラメータとtau=50は採用済みの設定に固定し、2027年以降の正解値は使っていない。再学習・保存・再読込後も2027年の27,240本の予測を再現し、元の予測との差は数値誤差の範囲だった。

- [実際の学習済みモデル](fitted_model_2027.joblib)：3基底モデル、前処理、対数平均補正、条件別重みを含む。
- [2027年予測CSV](predictions_2027.csv)：`pipe_id,predicted_repair_cost`、27,240本。
- [条件別重み](model_2027.json)・[各条件の最終重み表](evidence/effective_weights_2027.csv)・[基底モデル設定](base_configs/)。
- [学習・再現の検証](evidence/deployment_verification.json)・[環境](evidence/deployment_environment.json)・[比較結果](evidence/mean_metrics_all_recipes.csv)・[従来24構成との比較](evidence/prior_model_comparison.csv)・[SHA256](SHA256.json)。

The fitted artifact contains the actual three freshly trained estimators and preprocessing, not only meta weights. Prediction CSVs and fitted model publication were explicitly requested by the user. Raw repair targets and prepared training/forecast feature matrices are not bundled. The new model's official planning score remains **not evaluated**; the CSV is a severity forecast, not an evaluator work-unit submission.

## 実行方法 / Use

```powershell
pip install -r notebooks/ensemble_for_cost_prediction/latest_conditional_stack/requirements.txt
python notebooks/ensemble_for_cost_prediction/latest_conditional_stack/predict.py
python notebooks/ensemble_for_cost_prediction/latest_conditional_stack/verify.py
```

The first prediction command reproduces the committed CSV from bundled frozen base forecasts. To apply the **actual fitted bases** to a prepared year-start feature CSV:

```powershell
python notebooks/ensemble_for_cost_prediction/latest_conditional_stack/predict_fitted.py --features forecast_features_2027.csv --output regenerated_2027.csv
```

The input requires `pipe_id`, `material`, `surface` and the exact numeric/categorical features listed in `base_configs/`; it is not a raw `pipes.csv` reader. Feature history must stop at the end of 2026. Full fitting from compatible prepared features and historical OOF is available through `train.py --help`; those matrices remain local. Existing submissions and historical scores are preserved.
