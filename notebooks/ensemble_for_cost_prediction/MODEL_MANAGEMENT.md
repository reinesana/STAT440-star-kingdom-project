# 金額予測モデルの名前と切替 / Cost model management

現在の採用モデルは **CMS001 — Material-Surface Cost Stack（材質・地表面別コストスタック）、v1.0.0**。

| 項目 | 役割 |
|---|---|
| `CMS001` | この金額予測モデルの固定ID。既存C025等の実験・提出番号とは別体系 |
| `conditional_material_surface_rmse` | 検証表・内部コードで使う手法名。既存記録との対応のため保持 |
| [model_registry.json](model_registry.json) | モデル一覧。新モデルを追加しても過去モデルを削除しない |
| [latest_model.json](latest_model.json) | 現在の採用モデルを示す入口。GitHubでも参照できる |

## 採用モデルの切替

リポジトリ直下で次を実行する。`--select`なしなら表示だけで、ファイルを変更しない。

```powershell
python notebooks/ensemble_for_cost_prediction/select_model.py
python notebooks/ensemble_for_cost_prediction/select_model.py --select CMS001
```

切替コマンドは登録済みモデルの設定・学習済みモデル・予測CSVの存在を確認し、`latest_model.json`だけを更新する。重みの学習、予測の再生成、GitHubへのpushは行わない。推論コマンドは対象のモデルファイルを明示して使うため、入口を切り替えた後は表示されるモデルの推論手順に従う。

GitHubの画面から変更する場合も、`model_registry.json`の該当モデルの内容を`latest_model.json`に反映し、PRで確認する。パスはこのフォルダからの相対パス。名前だけを変えて予測やモデルを置き換えない。

## 次のバージョンを追加する場合

1. 新しいモデル・予測・検証を別フォルダに保存する。
2. 次の固定ID `CMS002` などでレジストリへ追加し、手法名、バージョン、学習終了年、予測対象、指標、パスを記録する。
3. 比較を確認してから採用モデルを切り替え、記事の推奨も更新する。

CMS001の学習済みモデル・予測・公式スコア未評価という状態は、今回の命名で変わらない。時間予測のIDはF168、工事順序はR029で、別の管理対象である。
