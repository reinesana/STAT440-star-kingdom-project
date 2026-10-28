# 3円同時統合 — public表示 $77.21M

**この版の基準は元の$77.22M探索拡張版。3版を順番に積み重ねたものではない。**

24組の3円統合。円数19,352→19,304。うち5組は2円だけでは固定費を節約できない配置。漏水ゼロ工事費を$31,274.70削減したが、表示スコアは基準を更新しなかった。

[提出CSV](outputs/submission_circle_triples.csv) / [公式表示の出典とCSVハッシュ](official_results.json)

予測モデル、修理費予測、費用式は維持。両端包含、地表種類、付随する管の費用を再計算した。
費用式は `2026-10-26-area-exponent`。入力データはリポジトリのものと一致し、生データは複製しない。

## 保存した証拠と実行結果

- [スコア画像](outputs/official_score.png)
- [年次内訳画像](outputs/official_annual_table.png)
- [元の実験要約](outputs/SUMMARY.md)
- [検証と入力・出力ハッシュ](outputs/report.json)
- [条件別試算](outputs/comparison.csv)
- [統合時の再確認](outputs/integration_validation.json)

Public値はユーザー提供画像と提出ファイルの対応確認に基づく。評価サイトをこちらから
直接照合したものではなく、表示は0.01M単位。privateスコアは未確認。
`report.json`の公式値nullや要約中の未評価の記述は提出前時点の記録として保存し、
提出後の値は`official_results.json`で管理する。

元の対象管カバーと、漏水ゼロ予定での交換遅延0本を確認。将来漏水で工事が前倒しになる
場合があるため、すべての未来で交換年を保証するものではない。
基準CSVが持つ過去交換済みoriginal ironの幾何学的カバーの留保は維持。
共有検証はactive ironを明示しており、全original ironに合格したとは主張しない。

## 再実行

リポジトリのrequirementsを使用。`submissions/2026-10-28_ryota_early3/`の
凍結した予測・幾何ヘルパーと、`data/`を参照する。統合時のコード変更はパス調整のみ。

```sh
python -m unittest discover -s submissions/2026-10-28_ryota_circle_triples -v
python submissions/2026-10-28_ryota_circle_triples/merge_three_circles.py --sims 30
```

再実行はこのフォルダの出力を上書きする。新実験は新しい日付・名前のフォルダへ保存する。
