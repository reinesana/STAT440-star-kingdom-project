# Ryota：予測を固定した工事費削減で$76.91M → $77.07M

ユーザー提供の提出画面で **public表示$77.07M** を確認。
前の最高点$76.91Mとの差は **+$0.16M（表示値で16万ドル）**。
表示は0.01M単位であり、厳密な元スコアの差ではない。

## 前の最高点から何を変えたか

**漏水予測・修理費予測は固定し、同じ施工予定年の近隣円260組を統合した。**

- 元の19,838円を19,578円へ削減。
- 統合円は元の2円の対象管を全て覆い、含まれる全管の地表と交換費用を再計算。
- 統合円は元の早い方の位置に置き、遅い方を削除。それ以外の円の相対順序は保持。
- リスクの重み変更、予測モデルの混合、対象管の追加は行っていない。
- 新たに含まれる未交換管が同年以降の予定なら、その候補は採用しない。

「同じ年」は最高点CSVを未来漏水なしで実行した予定年を指す。
実際の未来で常に同じ交換年になるという保証ではない。

| 比較 | 前の最高点 | 工事費削減版 |
|---|---:|---:|
| Public表示スコア | $76.91M | **$77.07M** |
| 円数 | 19,838 | **19,578** |
| 工事費・未来漏水なし | $180,664,606.53 | **$180,425,969.20** |
| 完了年・同条件 | 2045 | 2045 |

同条件の工事費削減は **$238,637.33**。元の予定より交換年が遅れる管は0本、
早まる管は189本。これはpublic評価の工事費差を測定したものではない。
Publicスコアには修理費と実際の施工時期も影響する。

## 保存した成果物

- [今回の提出CSV](../submissions/2026-10-28_ryota_construction_savings/outputs/submission_construction_savings.csv)
- [前の最高点CSV](../submissions/2026-10-28_ryota_early3/outputs_early3/early3_adaptive.csv)
- [方法・制約・再実行手順](../submissions/2026-10-28_ryota_construction_savings/README.md)
- [統合した260組の一覧](../submissions/2026-10-28_ryota_construction_savings/outputs/merges.csv)
- [入力・出力ハッシュと検証結果](../submissions/2026-10-28_ryota_construction_savings/outputs/report.json)
- [public表示の出典・対応CSVのハッシュ](../submissions/2026-10-28_ryota_construction_savings/official_results.json)
- [提出画面](../submissions/2026-10-28_ryota_construction_savings/outputs/official_score_77_07m.png)

2つの漏水モデル・ハザード0.5/1/2倍・各30回の共通乱数比較で、全6条件の平均工事費と
平均スコアが改善。平均スコア差は約+$0.131M〜+$0.299Mだった。
予測モデルを条件とした試算と、今回報告されたpublic結果を混同しない。
privateスコアは未確認。

基準CSVから引き継ぐ歴史的交換済み鉄管の幾何学的カバーの留保は、方法READMEと
共有検証JSONに明記。未交換鉄管のカバーと予算内完了を検証しており、
全original ironの厳密なカバーに合格したとは記録しない。
