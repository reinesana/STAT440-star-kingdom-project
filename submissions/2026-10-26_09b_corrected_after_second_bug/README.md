# 修正版09b（10月26日・第2バグ修正後）

既存09bの円を保持し、CatBoost + LightGBMの漏水確率による期待純便益を修正版工事費で割って固定順位を再計算します。全XGBoost入りの09aと区別します。元のaudit・CSV・pipes・trainはsource/に保存されています。

再現：このフォルダで `python build_09b_corrected.py`（numpy、pandasが必要）。提出用は `09b_corrected_catboost_lightgbm_net.csv` のみです。audit・validation・sourceは補助資料です。

費用式は `100 + surface_rate * area**0.85 + 200 * unreplaced_length`。保存validationは将来漏水なしの建設検証であり、公式publicスコアではありません。現行条件での提出・スコアは今回の参照記録から確認できません。共有Docの59.32Mは旧評価条件の記録です。

[統合提出台帳](../../reports/2026-10-27_submitted_models.md)
