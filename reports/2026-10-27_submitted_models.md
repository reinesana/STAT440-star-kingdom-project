# 提出モデルの資料一覧（2026-10-26提出セッション）

2026-10-27 UTCに整理。会話中の「今日」は提出画面・既存記録に残る10月26日の提出セッションとして扱い、日付を一律に書き換えていません。共有Google Docの現行内容、GitHub main、本人確認済みのチャット記録を照合しました。

## 提出と結果を確認できるモデル

| ID | 手法・変更点 | Public (M$) | 提出CSV |
| --- | --- | ---: | --- |
| 09a | Fixed initial net benefit / corrected cost | 65.40 | [CSV](../submissions/2026-10-26_after_second_bug%202/09a_corrected_reference.csv) |
| 10a | Nearby grouping rebuilt with corrected costs | 65.78 | [CSV](../submissions/2026-10-26_after_second_bug%202/10a_corrected_nearby_pairs.csv) |
| 11b | Fixed net benefit only; unchanged order | 51.99 | [CSV](../submissions/2026-10-26_after_second_bug%202/11b_corrected_net_benefit_only.csv) |
| 11c | Fixed net benefit / sqrt(corrected cost) | 61.70 | [CSV](../submissions/2026-10-26_after_second_bug%202/11c_corrected_net_benefit_sqrt_cost.csv) |
| 12a | Dynamic overlap updates, original circles | 69.89 | [CSV](../submissions/2026-10-26_after_second_bug%202/12a_corrected_dynamic.csv) |
| 13a | Rebuilt grouping plus dynamic overlap updates | 70.24 | [CSV](../submissions/2026-10-26_after_second_bug%202/13a_corrected_grouped_dynamic.csv) |
| 21a | 20a circles + p17b + CatBoost 68.627% / corrected log RF 31.373%; dynamic net benefit/cost | 74.78 | [CSV](../submissions/2026-10-26_best20a_catboost_logrf/21a_20a_catboost6863_logrf3137.csv) |

09a～13aはRionの修正再構築パッケージです。6件の対応は本人が述べた提出順に基づきます。21aはTakitoがファイル名と$74.78Mの対応を確認済みです。11b・11cをTakito作として扱っていません。

- [09a～13a：手法、旧版との違い、予算、再現手順](../submissions/2026-10-26_after_second_bug%202/README.md)。再現はそのフォルダで `python rebuild_corrected.py`。
- [21a：モデル構成、年次検証、重み、チームの貢献、限界](2026-10-26_takito_latest_model.md)。凍結提出の再現はリポジトリ直下で `python submissions/2026-10-26_best20a_catboost_logrf/reproduce.py`。このコマンドだけは過去再現用のlegacy計算を明示的に使用します。
- [機械可読の提出台帳](2026-10-27_submission_register.json)：提出CSVのSHA-256、結果の出典区分を収録。

## 提出画面にあるがファイル対応が未確定の結果

| 画面上の時刻（元記録のまま） | Public (M$) | 状態 |
| --- | ---: | --- |
| 15:25 | 74.16 | ファイル名未確認 |
| 15:34 | 74.25 | ファイル名未確認 |
| 15:34 | 74.00 | ファイル名未確認 |

出典は[21aの既存報告](2026-10-26_takito_latest_model.md)に記載されたユーザー提供画面です。スコアだけから18a、20a、stacking比較などへ割り当てていません。これらの正確なファイル対応が確定するまでは、全提出のモデル名まで確定したとは扱えません。

## 再構築・準備済みだが新規提出結果を確認できないもの

- [修正版09b](../submissions/2026-10-26_09b_corrected_after_second_bug/README.md)：CatBoost + LightGBMのみの漏水確率を使用。生成コード・CSVはmainに存在。現行評価での新規publicスコア未確認。共有Docの59.32Mは10/21の旧条件なので転記しません。
- 06・07a・07b：共有Docには「10/27に修正」とありますが、今回照合したmainには対応する新しい提出パッケージ・確定スコアがありません。旧スコアを現行結果として扱いません。
- [修正版21a](../submissions/2026-10-27_current_rules/README.md)：新費用式で順序を再構築。公式publicスコア未評価。$74.78Mはこの別CSVの結果ではありません。
- ローカルの `2026-10-26_cost_stacking_comparison` は18a対照とstacking差し替えの比較準備。manifestの両publicスコアはnullで、提出確認の根拠になりません。

## 費用式と比較の注意

現在の式は `100 + surface_rate * area**0.85 + 200 * unreplaced_length`。面積のみを0.85乗し、座標と直線管長はmです。[現行ルール](../docs/current_rules.md)を新規実験に使用します。

09a～13aは修正式で再構築された同一評価条件の比較です。21aの提出生成・既存ローカル検証は旧費用式で、公式スコアとローカル工事費は別の指標です。旧20aと21aの差をモデルだけの改善とは断定しません。全てpublic結果でprivate結果は未確定です。

共有記録：[STAT440 Google Doc](https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit)（2026-10-27読み取り）。共有Docは変更していません。
