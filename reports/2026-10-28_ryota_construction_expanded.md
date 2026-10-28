# Ryota：同年統合の探索拡張で$77.07M → $77.22M

ユーザー提供の画面で **public表示$77.22M** を確認した。
前の最高点$77.07Mから **+$0.15M**、元の$76.91Mから通算 **+$0.31M**。
いずれも0.01M単位の表示値から計算した差であり、privateスコアは未確認。

## 変更点

**漏水予測を固定し、同じ施工予定年の円を統合する探索範囲だけを拡張した。**

- 近隣候補：12円 → **24円**。
- 中心間距離の上限：80m → **120m**。
- 元のCSV行間隔：160行 → **320行**。
- 追加 **226組**を統合。円数19,578 → **19,352**。
- 同年制約、元の対象管のカバー、付随管の確認、正の節約を求める条件は維持。
- リスク予測、修理費予測、費用式、その他の円の相対順序は維持。

未来漏水なしの工事費は$180,425,969.20から **$180,290,222.26** へ、
**$135,746.94削減**。同条件で交換年が遅れる管は0本、早まる管は155本。
「同じ年」「遅延なし」はこの計画上の条件であり、すべての未来での保証ではない。

| 段階 | Public表示 | 円数 | 工事費・未来漏水なし |
|---|---:|---:|---:|
| 初期3年間重視版 | $76.91M | 19,838 | $180,664,606.53 |
| 同年統合260組 | $77.07M | 19,578 | $180,425,969.20 |
| 今回：さらに226組 | **$77.22M** | **19,352** | **$180,290,222.26** |

Publicスコア差をそのまま工事費差とは扱わない。予算の節約が施工時期を早め、
修理費へも影響する可能性がある。今回の年次内訳はユーザー提供画像として保存した。

## 成果物

- [提出CSV](../submissions/2026-10-28_ryota_construction_expanded/outputs/submission_construction_expanded.csv)
- [方法・再実行手順・留保](../submissions/2026-10-28_ryota_construction_expanded/README.md)
- [公式表示の出典とCSVハッシュ](../submissions/2026-10-28_ryota_construction_expanded/official_results.json)
- [スコア画像](../submissions/2026-10-28_ryota_construction_expanded/outputs/official_score_77_22m.png)
- [年次内訳画像](../submissions/2026-10-28_ryota_construction_expanded/outputs/official_annual_table_77_22m.png)
- [統合226組](../submissions/2026-10-28_ryota_construction_expanded/outputs/merges.csv)
- [比較試算](../submissions/2026-10-28_ryota_construction_expanded/outputs/comparison.csv)

2モデル×ハザード0.5/1/2倍、各30回の共通乱数比較で、6条件すべての平均工事費・
平均スコアが改善。モデル上の平均スコア差は約+$0.110M〜+$0.174M。
歴史的交換済み鉄管の幾何学的カバーに関する基準版の留保はREADMEと検証JSONに明記した。
