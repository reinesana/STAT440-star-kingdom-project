# Timing-only baseline on October 14 by Takito

10/14 By Takito · 2026

## English

This baseline replaces iron pipes individually, prioritizing pipes with the highest predicted probability of leaking in 2027.

- We created one circle for each of the **38,455 original iron pipes**, centered at its midpoint, with a radius equal to half its endpoint length plus a **0.000001 m numerical margin**.
- Circles are ordered by **2027 leak probability, highest first**. Ties use Pipe ID, and the **9 iron pipes with missing predictions go last**.
- We used the selected **equal-weight ensemble of XGBoost, CatBoost, and LightGBM**, with the confirmed zero-leak rules applied.
- The model predicts annual leak probability rather than an exact leak date. Higher probability is used as a proxy for earlier leakage.
- Iron pipes already replaced after historical leaks are included to satisfy the coverage requirement, but are not replaced again. Work units containing only polyurethane pipes are skipped at no cost.
- Any unreplaced copper or brass pipes wholly inside a circle are also replaced, as required by the rules.
- **No predicted repair costs, construction-cost weighting, or pipe grouping are used to determine priority.**

## Results

| Item | Result |
| --- | --- |
| Priority rule | Predicted 2027 leak probability, highest first |
| Submitted circles | 38,455 |
| Uncovered original iron pipes | 0 |
| Construction cost, assuming no future leaks | $149.21 million |
| Completion year, assuming no future leaks | 2041 |
| Work units skipped at no cost | 16,368 |
| Remaining allocation, assuming no future leaks | $100.79 million |

The exported CSV was read back and checked for full endpoint coverage and construction-budget feasibility. Exact construction cost was **$149,211,442.76**. **The remaining allocation excludes future leak repair costs and is not directly comparable with the scores in the October 13 experiment table.** No claim of superior realized score is made.

Nine iron pipes have unknown probabilities and are placed last, not assigned zero. Eight pipes with unknown material have no target circles and remain uncovered; their treatment requires clarification if they are deemed iron. Overlapping circles can replace another pipe before its own listed circle, so realized pipe replacement order can differ from the target order.

## Files and reproduction

- [Submission CSV](../submissions/2026-10-14_timing_only_baseline.csv) — no header, center and radius only.
- [Validation results](../submissions/2026-10-14_timing_only_baseline.validation.json).
- [Order audit](../submissions/2026-10-14_timing_only_baseline.order_audit.csv) — pipe IDs, probabilities, and circle geometry; not a submission file.
- [Builder](../src/build_timing_only_baseline.py). From the repository root, run `python src/build_timing_only_baseline.py` with numpy and pandas installed.

Source files are pinned to commit `58fd8a5b4b3f8b5365c525b0156163ece60a8100` for this result. The builder uses the checked-out source files; rerunning after those files change is a new result. Input and submission SHA-256 hashes are recorded in the validation file.

## 日本語の詳細

# 漏れの早さを年次確率で近似した修繕ベースライン

## 結果

38,455本の元の鉄管について、2027年の漏れ確率が高い順に工事円を並べたベースラインCSVを作成した。書き出したCSVを再読込し、全鉄管の両端が少なくとも1つの円に含まれることを確認した。覆い漏れは0本。将来の漏れがない予算確認では工事費149,211,442.76ドル、2041年完了となった。

## 確率だけを使った優先順位

GitHubの127候補比較で選ばれたXGBoost・CatBoost・LightGBMの等重みアンサンブルを使用した。2019〜2026年で学習した2027年予測のうち、既存ポリウレタン管・過去に漏れて交換された管の確率を0にした出力を採用した。

モデルが出力するのは2027年の年次漏れ確率であり、漏れ日付や2027〜2051年の時期予測ではない。今回は「早く漏れそう」の代理として年次確率の降順を用いた。修理費、工事費、敷設日の古さを優先順位に加えていない。同率はPipe IDの昇順。予測不明の鉄管9本は最後にPipe ID順で配置し、0確率とは区別した。確率だけでは長期の正確な発生順を保証できない。

## 提出円の作り方

cast iron・wrought iron・gray ironを鉄管として、元の鉄管1本につき1つの円を作成した。中心は両端の中点、半径は端点間距離の半分に0.000001 mの余裕を加えた。提出ファイルはヘッダーなしの2列で、中心座標と半径を小数点以下9桁で出力した。

元の鉄管38,455本には、過去の漏れで交換済みの12,218本も含まれる。元の鉄管すべての幾何学的カバー要件を満たすため円を残した。重複する円で別の管が先に交換される場合があるため、管ごとの実際の交換順はCSVの対象管順と完全一致しない。現在非ポリウレタンの銅・真鍮も円に全体が入れば交換される。

## 予算とカバーの検証

CSVを再読込し、有限な座標・非負半径・2列形式と鉄管全体のカバーを検証した。工事費は完全に含まれる全管の地表種類を各1回数え、未交換管の長さだけを課金した。過去の漏れ管は2027年開始時から交換済みとし、重複交換費と空工事費を除いた。各工事費をセントに四捨五入し、整数セントで年額1,000万ドル・残額繰越・次の工事で予算不足なら停止のルールを再現した。

将来漏れなし条件：工事費149,211,442.76ドル、完了2041年、スキップ円16,368件、未交換鉄管0本。2億5,000万ドルとの差額は100,788,557.24ドル。これは漏れ修理費0の条件付き値であり、公式評価スコアや将来修理費込みの期待スコアではない。実際の漏れが起きれば工事費・実行時期も変わる。

## 残る制約

予測不明の鉄管：P10869, P12275, P12276, P12277, P12278, P12279, P12280, P12281, P12282。末尾配置は予測による低リスク判定ではなく、明示的な欠損処理である。

材質不明8本は鉄管と断定できないため対象円を作成せず、今回の円ではいずれも覆われていない。課題側で鉄管扱いと決まった場合は対象と予算の再検証が必要。

既存のモデル比較では2023〜2025年の平均Brierが0.038216（最良単体0.039399）、2026年評価が0.077797。候補選択による楽観性、チームによる2026年データへの事前接触、モデル間の確率差が報告されている。これらは本作業で再学習・再評価した数値ではなく、ゼロ確率ルール適用前の比較結果である。今回のベースラインの優位性や全体最適性を示すものではない。

## 出典と再現性

使用コミット：`58fd8a5b4b3f8b5365c525b0156163ece60a8100`。

[提出ルールと予算](https://github.com/reinesana/STAT440-star-kingdom-project/blob/58fd8a5b4b3f8b5365c525b0156163ece60a8100/README.md)

[モデル比較](https://github.com/reinesana/STAT440-star-kingdom-project/blob/58fd8a5b4b3f8b5365c525b0156163ece60a8100/notebooks/leak_model_comparison/comparison_results/report.md)

[2027年予測とゼロ確率ルール](https://github.com/reinesana/STAT440-star-kingdom-project/blob/58fd8a5b4b3f8b5365c525b0156163ece60a8100/notebooks/leak_model_comparison/deployment_2027_zero_leak_applied/README.md)

[全管の予測CSV](https://github.com/reinesana/STAT440-star-kingdom-project/blob/58fd8a5b4b3f8b5365c525b0156163ece60a8100/notebooks/leak_model_comparison/deployment_2027_zero_leak_applied/full_inventory_2027_probabilities.csv)