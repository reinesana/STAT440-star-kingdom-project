# 単体ブースティング3モデルの工事費予測比較

指定の `github-main/data/train_predict_cost.csv` の14,107件を使用。元の金額単位でRMSEを優先して特徴量とパラメータを選択し、指定の4Foldでテストしました。

完了したOptuna試行は合計4,200件です。このほか、内側評価で特徴量の追加・削除を検討しました。有限の探索で得た最良候補であり、絶対最適の保証はありません。

## 4Fold単純平均

| model | MAE | RMSE |
| --- | --- | --- |
| xgboost | 29,878.19 | 130,462.53 |
| catboost | 29,803.85 | 130,560.76 |
| lightgbm | 28,170.72 | 130,766.84 |

各年を同じ重みで平均した値です。全テスト案件をまとめた参考値は `pooled_metrics.csv` に保存し、依頼の単純平均と区別しています。

## 各Foldのテスト値

| model | fold | train_end | test_year | n_train | n_test | MAE | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 1 | 2022 | 2023 | 5521 | 1824 | 36,022.35 | 163,663.83 |
| xgboost | 2 | 2023 | 2024 | 7345 | 1951 | 27,719.82 | 104,873.69 |
| xgboost | 3 | 2024 | 2025 | 9296 | 1773 | 28,803.12 | 120,522.83 |
| xgboost | 4 | 2025 | 2026 | 11069 | 3038 | 26,967.45 | 132,789.79 |
| catboost | 1 | 2022 | 2023 | 5521 | 1824 | 35,284.02 | 159,966.02 |
| catboost | 2 | 2023 | 2024 | 7345 | 1951 | 22,154.64 | 100,359.90 |
| catboost | 3 | 2024 | 2025 | 9296 | 1773 | 34,884.38 | 131,720.14 |
| catboost | 4 | 2025 | 2026 | 11069 | 3038 | 26,892.36 | 130,196.96 |
| lightgbm | 1 | 2022 | 2023 | 5521 | 1824 | 33,623.78 | 160,902.81 |
| lightgbm | 2 | 2023 | 2024 | 7345 | 1951 | 19,819.78 | 100,661.47 |
| lightgbm | 3 | 2024 | 2025 | 9296 | 1773 | 35,306.79 | 129,597.41 |
| lightgbm | 4 | 2025 | 2026 | 11069 | 3038 | 23,932.55 | 131,905.69 |

## 学習データ上の値（参考）

| model | fold | train_MAE | train_RMSE |
| --- | --- | --- | --- |
| xgboost | 1 | 34,537.68 | 168,196.05 |
| xgboost | 2 | 34,122.66 | 151,500.63 |
| xgboost | 3 | 27,667.30 | 143,171.31 |
| xgboost | 4 | 28,312.27 | 141,946.53 |
| catboost | 1 | 34,291.66 | 172,261.53 |
| catboost | 2 | 33,251.91 | 165,284.38 |
| catboost | 3 | 27,234.46 | 134,828.03 |
| catboost | 4 | 26,724.62 | 132,826.39 |
| lightgbm | 1 | 35,192.01 | 171,329.81 |
| lightgbm | 2 | 32,781.40 | 167,163.00 |
| lightgbm | 3 | 27,379.14 | 141,565.14 |
| lightgbm | 4 | 24,274.35 | 134,270.34 |

学習データ上の誤差は、同じデータで学習したモデルの適合度です。将来の予測性能は上のテスト値で判断します。学習期間はFold間で重複します。

## 選択した特徴量・設定

### xgboost — テスト2023年

特徴量：`event_decimal_year`, `event_hour`, `event_month`, `event_weekday`, `event_weekend`, `event_year`, `hist_age_band_all_max`, `hist_age_band_last1_max`, `hist_age_band_last1_mean`, `hist_age_band_last1_median`, `hist_age_band_last1_n`, `hist_age_band_last1_std`, `hist_age_band_last1_tail_rate`, `hist_age_band_last2_max`, `hist_age_band_last2_mean`, `hist_age_band_last2_median`, `hist_age_band_last2_n`, `hist_age_band_last2_tail_rate`, `hist_grid1000_last1_max`, `hist_grid1000_last1_mean`, `hist_grid1000_last1_median`, `hist_grid1000_last1_n`, `hist_grid1000_last1_std`, `hist_grid1000_last1_tail_rate`, `hist_grid1000_last2_max`, `hist_grid1000_last2_mean`, `hist_grid1000_last2_median`, `hist_grid1000_last2_n`, `hist_grid1000_last2_std`, `hist_grid1000_last2_tail_rate`, `hist_grid250_last1_max_log1p`, `hist_grid250_last1_mean`, `hist_grid250_last1_median`, `hist_grid250_last1_n`, `hist_grid250_last1_std`, `hist_grid250_last1_tail_rate`, `hist_grid250_last2_max`, `hist_grid250_last2_max_log1p`, `hist_grid250_last2_mean`, `hist_grid250_last2_median`, `hist_grid250_last2_n`, `hist_grid250_last2_tail_rate`, `hist_grid500_last1_max`, `hist_grid500_last1_mean`, `hist_grid500_last1_median`, `hist_grid500_last1_n`, `hist_grid500_last1_std`, `hist_grid500_last1_tail_rate`, `hist_grid500_last2_max`, `hist_grid500_last2_mean`, `hist_grid500_last2_median`, `hist_grid500_last2_n`, `hist_grid500_last2_std`, `hist_grid500_last2_tail_rate`, `hist_length_band_last1_max`, `hist_length_band_last1_mean`, `hist_length_band_last1_median`, `hist_length_band_last1_n`, `hist_length_band_last1_std`, `hist_length_band_last1_tail_rate`, `hist_length_band_last2_max`, `hist_length_band_last2_mean`, `hist_length_band_last2_median`, `hist_length_band_last2_n`, `hist_length_band_last2_std`, `hist_length_band_last2_tail_rate`, `hist_material_last1_max`, `hist_material_last1_mean`, `hist_material_last1_median`, `hist_material_last1_n`, `hist_material_last1_std`, `hist_material_last1_tail_rate`, `hist_material_last2_max`, `hist_material_last2_mean`, `hist_material_last2_median`, `hist_material_last2_n`, `hist_material_last2_std`, `hist_material_last2_tail_rate`, `hist_material_surface_all_mean`, `hist_material_surface_last1_max`, `hist_material_surface_last1_mean`, `hist_material_surface_last1_median`, `hist_material_surface_last1_n`, `hist_material_surface_last1_std`, `hist_material_surface_last1_tail_rate`, `hist_material_surface_last2_max`, `hist_material_surface_last2_median`, `hist_material_surface_last2_n`, `hist_material_surface_last2_std`, `hist_material_surface_last2_tail_rate`, `hist_surface_last1_max`, `hist_surface_last1_mean`, `hist_surface_last1_median`, `hist_surface_last1_n`, `hist_surface_last1_std`, `hist_surface_last1_tail_rate`, `hist_surface_last2_max`, `hist_surface_last2_mean`, `hist_surface_last2_median`, `hist_surface_last2_n`, `hist_surface_last2_std`, `hist_surface_last2_tail_rate`, `hour_cos`, `hour_sin`, `inventory_50m_road_count`, `lay_month`, `lay_year`, `length_age`, `log1p_pipe_length`, `log_age`, `log_length`, `material`, `material_brass`, `material_cast_iron`, `material_code`, `material_copper`, `material_gray_iron`, `material_surface`, `material_wrought_iron`, `midpoint_x`, `midpoint_y`, `month_cos`, `month_sin`, `orientation_cos2`, `orientation_sin2`, `pipe_length`, `season_cos`, `season_sin`, `span_x`, `span_y`, `surface`, `surface_code`, `surface_farmland`, `surface_grassland`, `surface_road`, `surface_structure`, `surface_swamp`, `surface_water`, `weekday_cos`, `weekday_sin`, `x1`, `x2`, `y1`, `y2`, `years_used`, `years_used_at_leak`, `years_used_squared`。

パラメータ：
```json
{
  "learning_rate": 0.13905014990269526,
  "max_depth": 6,
  "min_child_weight": 34.98462044400472,
  "subsample": 0.6264151106000365,
  "colsample_bytree": 0.6879771575181293,
  "reg_lambda": 0.022033633109930038,
  "reg_alpha": 0.05582770810800726,
  "gamma": 331.48938572459264,
  "max_bin": 256
}
```

反復数の設定上限：5。最終学習で生成された木の数：5。内側の平均RMSE：228,164.92。

### xgboost — テスト2024年

特徴量：`age_x_wet_proximity`, `event_decimal_year`, `hist_age_band_all_mean`, `hist_age_band_all_std`, `hist_grid1000_all_max`, `hist_grid1000_last1_std`, `hist_grid250_all_mean`, `hist_grid250_all_median`, `hist_grid250_last2_median`, `hist_material_surface_all_max`, `hist_material_surface_all_median`, `hist_material_surface_all_n`, `hist_material_surface_last1_mean`, `hist_material_surface_last1_median`, `hist_material_surface_last1_n`, `hist_nearest25_mean`, `hist_nearest5_mean`, `hist_radius100_max`, `hist_radius250_mean`, `hist_radius250_tail_rate`, `hist_radius500_median`, `hist_radius500_n`, `hist_radius50_max`, `hist_radius50_mean`, `hist_surface_all_mean`, `hist_surface_all_std`, `hist_surface_all_tail_rate`, `inventory_100m_road_count`, `inventory_10m_road_count`, `log1p_years_used`, `midpoint_y`, `orientation_cos2`, `surface`, `surface_road`, `x1`。

パラメータ：
```json
{
  "learning_rate": 0.012647139215234684,
  "max_depth": 7,
  "min_child_weight": 1.6319104134943172,
  "subsample": 0.8251104371470617,
  "colsample_bytree": 0.9783371999781912,
  "reg_lambda": 0.0033111601738007807,
  "reg_alpha": 0.0037604097745400585,
  "gamma": 73.59419681731733,
  "max_bin": 64
}
```

反復数の設定上限：29。最終学習で生成された木の数：29。内側の平均RMSE：159,419.88。

### xgboost — テスト2025年

特徴量：`hist_grid1000_all_max`, `hist_grid1000_all_n`, `hist_grid1000_all_tail_rate`, `hist_grid250_all_max`, `hist_grid250_all_median`, `hist_grid250_all_n`, `hist_grid250_all_tail_rate`, `hist_grid250_last2_std`, `hist_grid500_all_mean`, `hist_grid500_all_std`, `hist_grid500_last1_median`, `hist_grid500_last1_std`, `hist_grid500_last2_n`, `hist_length_band_all_max`, `hist_length_band_all_std`, `hist_material_surface_all_median`, `hist_material_surface_all_std`, `hist_material_surface_all_tail_rate`, `hist_nearest10_median`, `hist_nearest25_max`, `hist_nearest25_median`, `hist_nearest5_max`, `hist_radius100_max`, `hist_radius250_max`, `hist_radius250_median`, `hist_radius250_n`, `hist_radius250_tail_rate`, `hist_radius500_median`, `hist_radius500_n`, `hist_surface_all_mean`, `hist_surface_all_median`, `hist_surface_all_tail_rate`, `hour_sin`, `inventory_250m_count`, `inventory_5m_length_sum`, `season_cos`。

パラメータ：
```json
{
  "learning_rate": 0.08409193184055333,
  "max_depth": 3,
  "min_child_weight": 15.526510189285549,
  "subsample": 0.8357745931507727,
  "colsample_bytree": 0.9026079909263425,
  "reg_lambda": 0.14655058813806465,
  "reg_alpha": 18.210142788209243,
  "gamma": 0.018350106046247292,
  "max_bin": 256
}
```

反復数の設定上限：31。最終学習で生成された木の数：31。内側の平均RMSE：121,440.48。

### xgboost — テスト2026年

特徴量：`event_decimal_year`, `event_month`, `hist_age_band_last1_std`, `hist_age_band_last2_mean`, `hist_grid1000_last1_std`, `hist_grid250_all_max`, `hist_grid250_last1_std`, `hist_grid250_last2_n`, `hist_grid500_last1_max`, `hist_grid500_last1_tail_rate`, `hist_grid500_last2_std`, `hist_length_band_all_max`, `hist_length_band_all_mean`, `hist_length_band_last1_mean`, `hist_length_band_last1_median`, `hist_length_band_last2_n`, `hist_material_last1_median`, `hist_material_surface_all_median`, `hist_material_surface_all_std`, `hist_material_surface_all_tail_rate`, `hist_material_surface_last1_median`, `hist_material_surface_last2_max`, `hist_nearest10_max`, `hist_radius250_median`, `hist_radius250_n`, `hist_radius250_tail_rate`, `hist_radius500_max`, `hist_radius500_median`, `hist_radius50_max`, `hist_surface_last1_max`, `hist_surface_last1_median`, `month_cos`, `orientation_cos2`, `season_sin`, `surface`, `x1`, `x2`。

パラメータ：
```json
{
  "learning_rate": 0.0545530453904611,
  "max_depth": 3,
  "min_child_weight": 1.0133149800238779,
  "subsample": 0.960673751576149,
  "colsample_bytree": 0.5865653353486538,
  "reg_lambda": 2.449123018250732,
  "reg_alpha": 4.154093515050563,
  "gamma": 0.48726716844443607,
  "max_bin": 256
}
```

反復数の設定上限：19。最終学習で生成された木の数：19。内側の平均RMSE：102,999.22。

### catboost — テスト2023年

特徴量：`hist_grid250_all_median`, `hist_material_surface_last2_median`, `hist_surface_all_mean`, `hist_surface_all_median_log1p`, `hist_surface_all_tail_rate`, `material`, `material_surface`, `surface`, `surface_road`, `y2`。

パラメータ：
```json
{
  "learning_rate": 0.01951409473634728,
  "depth": 3,
  "l2_leaf_reg": 7.115991090600124,
  "random_strength": 18.658496689170093,
  "bootstrap_type": "Bernoulli",
  "subsample": 0.6128979677188174,
  "rsm": 0.9808188438425882,
  "border_count": 64
}
```

反復数の設定上限：107。最終学習で生成された木の数：107。内側の平均RMSE：227,146.15。

### catboost — テスト2024年

特徴量：`hist_grid250_all_median`, `hist_grid500_all_mean`, `hist_material_all_mean`, `hist_material_surface_all_mean`, `hist_material_surface_all_median`, `hist_material_surface_all_std`, `hist_radius250_tail_rate`, `hist_radius500_median`, `hist_surface_all_median`, `hist_surface_all_n`, `hist_surface_all_std`, `hist_surface_all_tail_rate`, `hist_surface_last2_mean`, `surface_road`, `surface_structure`。

パラメータ：
```json
{
  "learning_rate": 0.1086561515060856,
  "depth": 3,
  "l2_leaf_reg": 0.8820734944000431,
  "random_strength": 0.2883723562711601,
  "bootstrap_type": "Bernoulli",
  "subsample": 0.9029163001449033,
  "rsm": 0.6340304014592798,
  "border_count": 64,
  "one_hot_max_size": 2
}
```

反復数の設定上限：11。最終学習で生成された木の数：11。内側の平均RMSE：160,252.83。

### catboost — テスト2025年

特徴量：`hist_age_band_all_max`, `hist_age_band_all_mean`, `hist_age_band_all_median`, `hist_age_band_all_n`, `hist_age_band_all_std`, `hist_age_band_all_tail_rate`, `hist_age_band_last1_max`, `hist_age_band_last1_mean`, `hist_age_band_last1_median`, `hist_age_band_last1_n`, `hist_age_band_last1_std`, `hist_age_band_last1_tail_rate`, `hist_age_band_last2_max`, `hist_age_band_last2_mean`, `hist_age_band_last2_median`, `hist_age_band_last2_n`, `hist_age_band_last2_std`, `hist_age_band_last2_tail_rate`, `hist_grid1000_all_max`, `hist_grid1000_all_mean`, `hist_grid1000_all_median`, `hist_grid1000_all_n`, `hist_grid1000_all_std`, `hist_grid1000_all_tail_rate`, `hist_grid1000_last1_max`, `hist_grid1000_last1_mean`, `hist_grid1000_last1_median`, `hist_grid1000_last1_n`, `hist_grid1000_last1_std`, `hist_grid1000_last1_tail_rate`, `hist_grid1000_last2_max`, `hist_grid1000_last2_mean`, `hist_grid1000_last2_median`, `hist_grid1000_last2_n`, `hist_grid1000_last2_std`, `hist_grid1000_last2_tail_rate`, `hist_grid250_all_max`, `hist_grid250_all_mean`, `hist_grid250_all_median`, `hist_grid250_all_n`, `hist_grid250_all_std`, `hist_grid250_all_tail_rate`, `hist_grid250_last1_max`, `hist_grid250_last1_mean`, `hist_grid250_last1_median`, `hist_grid250_last1_n`, `hist_grid250_last1_std`, `hist_grid250_last1_tail_rate`, `hist_grid250_last2_max`, `hist_grid250_last2_mean`, `hist_grid250_last2_median`, `hist_grid250_last2_n`, `hist_grid250_last2_std`, `hist_grid250_last2_tail_rate`, `hist_grid500_all_max`, `hist_grid500_all_mean`, `hist_grid500_all_median`, `hist_grid500_all_n`, `hist_grid500_all_std`, `hist_grid500_all_tail_rate`, `hist_grid500_last1_mean`, `hist_grid500_last1_median`, `hist_grid500_last1_n`, `hist_grid500_last1_std`, `hist_grid500_last1_std_log1p`, `hist_grid500_last1_tail_rate`, `hist_grid500_last2_max`, `hist_grid500_last2_mean`, `hist_grid500_last2_median`, `hist_grid500_last2_n`, `hist_grid500_last2_std`, `hist_grid500_last2_tail_rate`, `hist_length_band_all_max`, `hist_length_band_all_mean`, `hist_length_band_all_median`, `hist_length_band_all_n`, `hist_length_band_all_std`, `hist_length_band_all_tail_rate`, `hist_length_band_last1_max`, `hist_length_band_last1_mean`, `hist_length_band_last1_median`, `hist_length_band_last1_n`, `hist_length_band_last1_std`, `hist_length_band_last1_tail_rate`, `hist_length_band_last2_max`, `hist_length_band_last2_mean`, `hist_length_band_last2_median`, `hist_length_band_last2_n`, `hist_length_band_last2_std`, `hist_length_band_last2_tail_rate`, `hist_material_all_max`, `hist_material_all_mean`, `hist_material_all_median`, `hist_material_all_n`, `hist_material_all_std`, `hist_material_all_tail_rate`, `hist_material_last1_max`, `hist_material_last1_mean`, `hist_material_last1_median`, `hist_material_last1_n`, `hist_material_last1_std`, `hist_material_last1_tail_rate`, `hist_material_last2_max`, `hist_material_last2_mean`, `hist_material_last2_median`, `hist_material_last2_n`, `hist_material_last2_std`, `hist_material_last2_tail_rate`, `hist_material_surface_all_max`, `hist_material_surface_all_mean`, `hist_material_surface_all_median`, `hist_material_surface_all_n`, `hist_material_surface_all_std`, `hist_material_surface_all_tail_rate`, `hist_material_surface_last1_max`, `hist_material_surface_last1_mean`, `hist_material_surface_last1_median`, `hist_material_surface_last1_n`, `hist_material_surface_last1_std`, `hist_material_surface_last1_tail_rate`, `hist_material_surface_last2_max`, `hist_material_surface_last2_mean`, `hist_material_surface_last2_median`, `hist_material_surface_last2_n`, `hist_material_surface_last2_std`, `hist_material_surface_last2_tail_rate`, `hist_nearest10_max`, `hist_nearest10_mean`, `hist_nearest10_median`, `hist_nearest25_max`, `hist_nearest25_mean`, `hist_nearest25_median`, `hist_nearest5_max`, `hist_nearest5_mean`, `hist_nearest5_median`, `hist_radius100_max`, `hist_radius100_mean`, `hist_radius100_median`, `hist_radius100_n`, `hist_radius100_tail_rate`, `hist_radius250_max`, `hist_radius250_mean`, `hist_radius250_median`, `hist_radius250_n`, `hist_radius250_tail_rate`, `hist_radius500_max`, `hist_radius500_mean`, `hist_radius500_median`, `hist_radius500_n`, `hist_radius500_tail_rate`, `hist_radius50_max`, `hist_radius50_mean`, `hist_radius50_median`, `hist_radius50_n`, `hist_radius50_tail_rate`, `hist_surface_all_max`, `hist_surface_all_max_log1p`, `hist_surface_all_mean`, `hist_surface_all_median`, `hist_surface_all_n`, `hist_surface_all_std`, `hist_surface_all_tail_rate`, `hist_surface_last1_max`, `hist_surface_last1_mean`, `hist_surface_last1_median`, `hist_surface_last1_n`, `hist_surface_last1_std`, `hist_surface_last1_tail_rate`, `hist_surface_last2_mean`, `hist_surface_last2_median`, `hist_surface_last2_n`, `hist_surface_last2_std`, `hist_surface_last2_tail_rate`, `log1p_pipe_length`, `log1p_years_used`, `log_length`, `material`, `material_brass`, `material_cast_iron`, `material_code`, `material_copper`, `material_gray_iron`, `material_surface`, `material_wrought_iron`, `midpoint_x`, `midpoint_y`, `orientation_cos2`, `orientation_sin2`, `pipe_length`, `span_x`, `span_y`, `surface`, `surface_code`, `surface_farmland`, `surface_grassland`, `surface_road`, `surface_structure`, `surface_swamp`, `surface_water`, `weekday_cos`, `x1`, `x2`, `y1`, `y2`, `years_used_at_leak`。

パラメータ：
```json
{
  "learning_rate": 0.03036174903365222,
  "depth": 3,
  "l2_leaf_reg": 0.6201847397621251,
  "random_strength": 0.0036169292499532543,
  "bootstrap_type": "Bernoulli",
  "subsample": 0.7277701133394161,
  "rsm": 0.914138704034685,
  "border_count": 128,
  "one_hot_max_size": 2
}
```

反復数の設定上限：105。最終学習で生成された木の数：105。内側の平均RMSE：123,346.64。

### catboost — テスト2026年

特徴量：`event_hour`, `event_month`, `event_year`, `hist_grid1000_all_mean`, `hist_grid1000_all_n`, `hist_grid250_all_median`, `hist_grid250_all_tail_rate`, `hist_grid500_all_n`, `hist_grid500_all_std`, `hist_length_band_all_median`, `hist_length_band_all_tail_rate`, `hist_material_surface_all_median`, `hist_material_surface_last1_median`, `hist_radius500_median`, `hist_surface_all_max`, `hist_surface_all_mean`, `hist_surface_all_tail_rate`, `hist_surface_last1_std`, `hour_sin`, `inventory_10m_length_sum`, `inventory_50m_surface_diversity`, `midpoint_x`, `midpoint_y`, `orientation_cos2`, `orientation_sin2`, `season_cos`, `season_sin`, `surface`, `weekday_cos`, `weekday_sin`, `y2`。

パラメータ：
```json
{
  "learning_rate": 0.07749586710478815,
  "depth": 6,
  "l2_leaf_reg": 5.692617395759746,
  "random_strength": 0.0010867950457761608,
  "bootstrap_type": "Bernoulli",
  "subsample": 0.8296272982086644,
  "rsm": 0.7847747510664507,
  "border_count": 254,
  "one_hot_max_size": 32
}
```

反復数の設定上限：43。最終学習で生成された木の数：43。内側の平均RMSE：102,950.88。

### lightgbm — テスト2023年

特徴量：`event_month`, `event_weekday`, `hist_grid250_all_max`, `hist_grid250_all_median`, `hist_grid250_last1_mean`, `hist_grid250_last1_median`, `hist_grid500_all_mean`, `hist_grid500_all_median`, `hist_grid500_last1_median`, `hist_material_all_max`, `hist_material_surface_all_mean`, `hist_material_surface_last1_median`, `hist_nearest25_max`, `hist_nearest5_mean`, `hist_radius100_mean`, `hist_radius500_tail_rate`, `hist_radius50_median`, `hour_sin`, `inventory_250m_count`, `inventory_distance_to_road`, `log1p_years_used`, `material_code`, `month_sin`, `orientation_cos2`, `surface_code`, `surface_road`, `surface_structure`, `weekday_cos`, `weekday_sin`, `x1`。

パラメータ：
```json
{
  "learning_rate": 0.14776118540829655,
  "num_leaves": 61,
  "max_depth": 3,
  "min_child_samples": 43,
  "subsample": 0.683537514633538,
  "subsample_freq": 1,
  "colsample_bytree": 0.8297113874949713,
  "reg_lambda": 0.0029761217317507476,
  "reg_alpha": 1.3126489179268885,
  "max_bin": 127,
  "min_split_gain": 19700778.906826887
}
```

反復数の設定上限：5。最終学習で生成された木の数：5。内側の平均RMSE：226,593.03。

### lightgbm — テスト2024年

特徴量：`event_month`, `hist_age_band_all_max`, `hist_age_band_all_std`, `hist_grid1000_all_max`, `hist_grid1000_all_n`, `hist_grid1000_all_tail_rate`, `hist_grid250_all_max`, `hist_grid250_all_median`, `hist_grid500_all_max`, `hist_grid500_all_mean`, `hist_grid500_all_median`, `hist_length_band_all_max`, `hist_material_all_n`, `hist_material_surface_all_n`, `hist_material_surface_all_tail_rate`, `hist_material_surface_last1_median`, `hist_material_surface_last2_mean`, `hist_nearest25_median`, `hist_radius250_tail_rate`, `hist_surface_last1_std`, `hist_surface_last2_mean`, `hour_cos`, `lay_month`, `log1p_years_used`, `log_age`, `material_copper`, `midpoint_y`, `orientation_cos2`, `road_x_structure_exposure`, `season_cos`, `season_sin`, `surface_road`。

パラメータ：
```json
{
  "learning_rate": 0.16801299589049618,
  "num_leaves": 10,
  "max_depth": 7,
  "min_child_samples": 63,
  "subsample": 0.7216937914576833,
  "subsample_freq": 1,
  "colsample_bytree": 0.5939078561191307,
  "reg_lambda": 2.50738968174609,
  "reg_alpha": 0.03439365605378452,
  "max_bin": 127,
  "min_split_gain": 64327.857575275346
}
```

反復数の設定上限：6。最終学習で生成された木の数：6。内側の平均RMSE：162,772.31。

### lightgbm — テスト2025年

特徴量：`hist_age_band_all_mean`, `hist_age_band_all_std`, `hist_grid1000_all_max`, `hist_grid1000_all_mean`, `hist_grid1000_all_median`, `hist_grid1000_all_n`, `hist_grid1000_all_std`, `hist_grid250_all_max`, `hist_grid250_all_mean`, `hist_grid250_all_median`, `hist_grid250_all_std`, `hist_grid250_all_tail_rate`, `hist_grid500_all_max`, `hist_grid500_all_mean`, `hist_grid500_all_median`, `hist_grid500_all_std`, `hist_length_band_all_std`, `hist_length_band_last2_std`, `hist_material_surface_all_median`, `hist_material_surface_all_tail_rate`, `hist_surface_all_max`, `hist_surface_all_mean`, `hist_surface_all_median`, `hist_surface_all_tail_rate`, `inventory_100m_parallel_count`。

パラメータ：
```json
{
  "learning_rate": 0.0998750061997041,
  "num_leaves": 7,
  "max_depth": 6,
  "min_child_samples": 33,
  "subsample": 0.9003466355212323,
  "subsample_freq": 1,
  "colsample_bytree": 0.8847699781203096,
  "reg_lambda": 0.0036068778622544108,
  "reg_alpha": 2.2717342059203127,
  "max_bin": 127,
  "min_split_gain": 1929.5705562742053
}
```

反復数の設定上限：55。最終学習で生成された木の数：55。内側の平均RMSE：124,011.21。

### lightgbm — テスト2026年

特徴量：`hist_age_band_all_max`, `hist_age_band_all_mean`, `hist_age_band_all_median`, `hist_age_band_all_n`, `hist_age_band_all_std`, `hist_age_band_all_tail_rate`, `hist_grid1000_all_max`, `hist_grid1000_all_mean`, `hist_grid1000_all_median`, `hist_grid1000_all_std`, `hist_grid1000_all_tail_rate`, `hist_grid1000_last2_mean`, `hist_grid250_all_max`, `hist_grid250_all_mean`, `hist_grid250_all_n`, `hist_grid250_all_std`, `hist_grid250_all_tail_rate`, `hist_grid500_all_max`, `hist_grid500_all_mean`, `hist_grid500_all_median`, `hist_grid500_all_n`, `hist_grid500_all_std`, `hist_grid500_all_tail_rate`, `hist_length_band_all_max`, `hist_length_band_all_mean`, `hist_length_band_all_median`, `hist_length_band_all_n`, `hist_length_band_all_std`, `hist_length_band_all_tail_rate`, `hist_material_all_max`, `hist_material_all_mean`, `hist_material_all_median`, `hist_material_all_n`, `hist_material_all_std`, `hist_material_all_tail_rate`, `hist_material_surface_all_max`, `hist_material_surface_all_mean`, `hist_material_surface_all_median_log1p`, `hist_material_surface_all_n`, `hist_material_surface_all_tail_rate`, `hist_material_surface_last2_std`, `hist_nearest10_max`, `hist_nearest10_mean`, `hist_nearest10_median`, `hist_nearest25_max`, `hist_nearest25_mean`, `hist_nearest25_median`, `hist_nearest5_max`, `hist_nearest5_mean`, `hist_nearest5_median`, `hist_radius100_mean`, `hist_radius100_median`, `hist_radius100_tail_rate`, `hist_radius250_max`, `hist_radius250_mean`, `hist_radius250_median`, `hist_radius250_n`, `hist_radius250_tail_rate`, `hist_radius500_max`, `hist_radius500_mean`, `hist_radius500_median`, `hist_radius500_n`, `hist_radius500_tail_rate`, `hist_radius50_max`, `hist_radius50_mean`, `hist_radius50_median`, `hist_radius50_n`, `hist_radius50_tail_rate`, `hist_surface_all_max`, `hist_surface_all_mean`, `hist_surface_all_median`, `hist_surface_all_n`, `hist_surface_all_std`, `hist_surface_all_tail_rate`, `inventory_100m_prior_repairs`, `inventory_10m_road_count`, `log1p_pipe_length`, `log1p_years_used`, `log_length`, `material`, `material_brass`, `material_cast_iron`, `material_code`, `material_copper`, `material_gray_iron`, `material_surface`, `material_wrought_iron`, `midpoint_x`, `midpoint_y`, `orientation_cos2`, `orientation_sin2`, `pipe_length`, `span_x`, `span_y`, `surface`, `surface_code`, `surface_farmland`, `surface_grassland`, `surface_road`, `surface_structure`, `surface_swamp`, `surface_water`, `x1`, `x2`, `y1`, `y2`, `years_used_at_leak`。

パラメータ：
```json
{
  "learning_rate": 0.12907214194382882,
  "num_leaves": 12,
  "max_depth": 7,
  "min_child_samples": 16,
  "subsample": 0.7830482459013572,
  "subsample_freq": 1,
  "colsample_bytree": 0.9439828601929375,
  "reg_lambda": 47.79663467074333,
  "reg_alpha": 9.167377750356485,
  "max_bin": 255,
  "min_split_gain": 1261.5920384304368
}
```

反復数の設定上限：28。最終学習で生成された木の数：28。内側の平均RMSE：101,491.09。

## 方法と検証

各Foldの学習期間内の直近2年を時系列検証に使用し、その2年のRMSE単純平均を最小化しました。反復ごとのearly stoppingによる探索後、有力な最大5設定を、両年で同じ木の数を使った場合の平均RMSEで再順位付けしました。CatBoostでは候補の木の数ごとに最初から再学習して比較しました。全12設定を確定してから外側テストを実行しました。履歴特徴量は全て前年末までの工事費から作り、同年の工事費は使用していません。前年実費が翌年初に確定している仮定があります。

工事費の上限カット、外れ値除外、対数目的変数、モデルのアンサンブルは使用していません。負の予測値は0に丸めました。

元CSVのSHA256と実費・対象IDの一致を確認し、履歴統計を42件で独立照合しました。12FoldのMAE/RMSEと4Fold平均を案件別予測から別の計算方法で再計算しています。履歴照合は抽出検証であり、全行の独立再計算を意味しません。

各テスト年以降の工事費を37倍＋1,234,567に改変してコスト依存の前処理をやり直し、当該テスト年までの特徴量が変わらないことを4Fold全てで確認しました。候補は470列です。コストを使わない補助インベントリ特徴量は改変の影響がない構造と、24件×6半径の独立計算を確認しました。学習データ上のMAE/RMSEも、保存した学習案件別予測から独立再計算しています。

追加探索では、pipes.csvの周辺管情報と過去の実験コードから候補を追加しました。各モデル・Foldで、全候補の単独追加・採用列の削除・学習内Spearman相関の絶対値0.98以上の置換・カテゴリと数値コードの同情報の置換・関連する列群の同時追加/削除を比較し、改善を採用するたびに基準を更新して全比較を繰り返しました。その後にパラメータと共通反復数を60試行ずつ再調整し、特徴量比較を収束まで繰り返しました。最終的に、パラメータを探索した特徴量構成と、再確認後の構成が一致するまで確認しました。高相関だけで除外せず、検証RMSEの改善で判断しました。線分距離で近い管の過去コストも試しましたが、実際の修理地点や接続を示すものではありません。採用経路と全比較結果はexpanded_adoption_paths.csv、expanded_candidate_comparisons.csv、相関は各Foldのexpanded_correlations.csvに保存しています。拡張前のテスト出力は退避し、その誤差は追加探索に使用していません。

入力ファイルの2026年末までのデータを使用しました。実行日時点で全実績が確定しているという意味ではありません。過去の作業で2026年も確認済みのため、この比較は今回の探索でテスト費用を未使用にした評価であり、プロジェクト全体で完全に未閲覧の最終テストではありません。

周辺管の情報には、案件日より後に敷設された管と敷設日不明の管を除き、pipes.csvの属性が予測時点でも有効という仮定があります。撤去履歴や材質・地表面の変更履歴はありません。幾何的な交差や共通端点は確認済みの物理接続を表さず、端点間の直線長は実際の曲線長や掘削長を表しません。
