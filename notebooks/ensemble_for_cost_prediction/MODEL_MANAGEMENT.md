# 金額予測モデルの名前と切替 / Cost model management

現在の採用モデルは **CMS002 — Fixed Four-Predictor Cost Model（仕様固定・基本4項目コストモデル）、v1.0.0**。

2026-10-30、ユーザー指定により公開スコアを見た調整と、繰り返しの過去検証による候補選定を外す方針へ変更した。[固定仕様と再現手順](fixed_protocol/README.md)を参照。特徴量は基本4項目と欠損フラグ2列。CMS001は以前の選定経緯を含めて履歴に残す。CMS002への切替は精度比較の勝者を選んだ結果ではない。時間モデルもF168からHFX001へ変更する。新しい公式工事計画スコアはない。

| 項目 | 役割 |
|---|---|
| `CMS002` | 現在の金額予測モデルの固定ID。CMS001は履歴として登録を維持 |
| `fixed_four_predictor_raw_catboost` | 現在の手法名。設定は実行前に固定 |
| [model_registry.json](model_registry.json) | モデル一覧。新モデルを追加しても過去モデルを削除しない |
| [latest_model.json](latest_model.json) | 現在の採用モデルを示す入口。GitHubでも参照できる |

## 採用モデルの切替

リポジトリ直下で次を実行する。`--select`なしなら表示だけで、ファイルを変更しない。

```powershell
python notebooks/ensemble_for_cost_prediction/select_model.py
python notebooks/ensemble_for_cost_prediction/select_model.py --select CMS002
```

切替コマンドは登録済みモデルの設定・学習済みモデル・予測CSVの存在を確認し、`latest_model.json`だけを更新する。重みの学習、予測の再生成、GitHubへのpushは行わない。推論コマンドは対象のモデルファイルを明示して使うため、入口を切り替えた後は表示されるモデルの推論手順に従う。

GitHubの画面から変更する場合も、`model_registry.json`の該当モデルの内容を`latest_model.json`に反映し、PRで確認する。パスはこのフォルダからの相対パス。名前だけを変えて予測やモデルを置き換えない。

## 次のバージョンを追加する場合

1. 新しいモデル・予測・検証を別フォルダに保存する。
2. 次の固定ID `CMS002` などでレジストリへ追加し、手法名、バージョン、学習終了年、予測対象、指標、パスを記録する。
3. 今回は実行前に仕様を固定し、データ完全性と再現性の確認で切り替える。記述的な過去検証の成績や公開スコアを見て切替候補を選び直さない。

CMS001の学習済みモデル・予測・公式スコア未評価という履歴は維持。F168／R029も旧版として区別する。CMS002とHFX001は2027年の予測であり、新たな工事順序・25年先の検証済み性能を意味しない。
