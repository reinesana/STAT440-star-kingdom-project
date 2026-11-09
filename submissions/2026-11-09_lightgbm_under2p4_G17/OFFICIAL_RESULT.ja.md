# LightGBM 2.4倍 × G17円：公式提出結果

**Public score: $76,362,946.88**。提出ID：`SDtTshmc1ne4`。Takito / UniUni、2026-11-09 13:53（提出サイト表示）。

ユーザーがこのCSVを実際に提出してPublicを確認するよう明示的に依頼したため、固定済み候補を1回提出した。Public結果を見て金額モデル・倍率・施工順を再調整していない。

| 対象 | Public score | 今回との差 |
|---|---:|---:|
| 今回：LightGBM 2.4倍・G17円 | $76,362,946.88 | — |
| 元G17 | $79,168,928.88 | $-2,805,982.00 |

今回はPublicでG17を上回らなかった。内部評価でも安定した改善は確認できておらず、現行計画からの採用変更は保留する。Privateでの優劣は未確認。

公式総工事費（CAPEX）：$142,832,806.52。公式総漏水修理費（OPEX）：$30,804,246.60。`250,000,000 − CAPEX − OPEX` から上記のセント単位スコアを計算し、サイト表示の$76.36mとも整合した。

[公式年間内訳](http://172.16.42.11:51238/static/output/ee8eORjFV8OTo1vKiFmiK2XG.txt)。ローカルに保存した全25年の原表は [official_annual_breakdown.csv](official_annual_breakdown.csv)。

## 提出ファイル

[実際に提出したCSV](submission_files/selected_G17_circles.csv)。59,117行。

SHA256: `df5083ba455057b3d25d30ab099578391380c2533840f7b50bbf080bfe5129e8`

G17の円の位置・半径を保ち、重複する交換対象に関する前後依存を守って並べ替えた。金額予測はLightGBMの学習時に過小評価側の二乗損失を2.4倍にしたもの。漏水予測は既存F168。G17の元の施工順も変更されているため、Public差を金額モデル単独の因果効果とは呼ばない。

## 内部評価と検証

- [実験レポート](REPORT.ja.md)：6モデル×11倍率×4年、各モデルの倍率選択、2026年の確認。
- [事前プロトコル](PROTOCOL.md)と[2026年評価前の選択固定](locked_before_2026_evaluation.json)。プロトコルは最初のファイル作成依頼時のもの。この公式提出は後続のユーザー明示依頼による。
- [年別結果](annual_metrics.csv)、[過去年のみの順次選択](forward_selection.csv)、[2026年の感度分析](confirmation_2026_sensitivity.csv)。
- [G17円での25年参考シミュレーション](G17_simulated_summary_2027.csv)。元G17比で候補の平均総費用は約432万ドル増加。2種類×128乱数の仮想将来であり、実測や公式スコアではない。
- [提出CSVの再読込検証](submission_files/selected_G17_circles.validation.json)：全original ironのカバー、修正済み工事費、年間予算と繰越、2051年までの完了。漏水なしの保守的な完了年は2045年。
- [提出記録](official_submission.json)、[SHA256一覧](PUBLICATION_SHA256.json)、[サイト画面](official_submission_screenshot.png)。

学習・選択に利用した過去年は既存研究でも参照済みであり、完全未使用の最終テストではない。共有Google Docは編集していない。過去の実験・提出記録は変更していない。
