# STAT 440 – Star Kingdom Pipe Replacement

Replace every iron pipe with polyurethane between **2027 and 2051**, with **$10M added to the budget each year**. We submit an ordered list of circular work units, one per line, in the form `"(x, y)",r`.

**Score = ($250M − work-unit costs) − leak repair costs during 2027–2051**

The assignment specification is authoritative. Strategy suggestions below should be tested using our scorer.

## Layout

```text
data/          pipes.csv, train.csv (leak history 2019–2026). Don't edit these.
notebooks/     exploration, named yourname_topic
src/           shared code, cost function, simulator and validator
submissions/   submission CSVs, named YYYY-MM-DD_description.csv
```

Setup: create a `.venv` and install the packages in `requirements.txt`.

The assignment calls the historical leak file `costs.csv`; our project uses `train.csv`. Confirm that these refer to the same supplied data.

## Submission format

Submit a CSV with **no header**, one work unit per line:

```csv
"(1234.5, 6789.0)",25.0
```

- Coordinates and radii are in **metres**.
- Each work unit describes a **closed circle**, so the boundary counts as inside.
- Every original iron pipe must be **wholly inside at least one submitted circle**.
- When a unit executes, every currently non-polyurethane pipe wholly inside its circle is replaced, including copper and brass.
- A pipe that only intersects the circle is not replaced.
- For straight pipes, check that **both endpoints** are inside. Checking only the midpoint is insufficient.
- Validate coverage again after exporting the CSV, accounting for numerical precision.

## Work-unit cost

| Component | Rate |
| --- | ---: |
| Groundbreaking | $100/work unit |
| Grass | $3/m² |
| Farm | $7/m² |
| Swamp | $12/m² |
| Road | $17/m² |
| Structure | $50/m² |
| Water | $32/m² |
| Replacement length | $200/m |

For a circle with radius `r`:

```text
area = pi * r²
surface_rate = sum of rates for distinct surface types among all wholly contained pipes
replacement_length = total length of currently non-polyurethane wholly contained pipes

cost = round_to_cents(
    100 + (surface_rate * area)^0.85 + 200 * replacement_length
)
```

### Important cost details

- Count each surface type **once**, regardless of the number or length of pipes.
- Apply the combined surface rate to the **whole circle area**.
- For example, grass and road give a rate of `3 + 17 = 20`.
- Surface types are counted from **all wholly contained pipes**, including polyurethane pipes and pipes already replaced by leaks or earlier work units.
- Only currently non-polyurethane pipes contribute the $200/m term.
- Overlapping circles must not charge replacement length twice.
- If a circle contains no pipes or only currently polyurethane pipes, it is **skipped at zero cost**, including no groundbreaking charge.
- Round each executed work unit's final cost to cents. Use consistent rounding and avoid floating-point budget errors.

## Budget and execution

1. At the beginning of 2027, start with $10M.
2. Execute work units in CSV order. Execution is instantaneous.
3. If the next unit would make the budget negative, stop work for that year. **Do not skip ahead to a cheaper unit.**
4. Carry the remaining budget forward and add $10M at the beginning of the next year.
5. Resume from the blocked unit, recalculating its cost using the current pipe materials.
6. Continue through 2051.

A unit costing exactly the available budget can execute. A unit costing more than $10M may execute after budget accumulates.

The specification is rejected if it cannot finish within the 25-year window or if the required iron replacement and coverage conditions are not satisfied.

Surplus uses the full **$250M allocation**, even if work finishes early. Under the stated rules, repair costs are deducted separately from the score rather than from the construction budget.

## Things to know

- **Order affects both repair costs and work-unit costs.** A pipe that leaks before scheduled replacement becomes polyurethane, reducing later replacement length and sometimes causing a unit to be skipped.
- Pipes that leaked in `train.csv` are already polyurethane when the project starts. `pipes.csv` records their original material.
- Keep original material and current material in separate fields.
- During the project, a leak incurs a repair cost and the pipe is then replaced by polyurethane.
- Evaluate leaks throughout **2027–2051**, even if scheduled work finishes early.
- Every currently non-polyurethane pipe fully inside a circle is replaced and charged $200/m. Avoid unnecessary copper and brass replacement when it increases total cost.
- One structure or water pipe can make a large circle expensive because its surface rate applies to the whole area.
- The `0.85` exponent and $100 charge can favour grouping, but larger circles may add area, expensive surface types and extra replacement length. **Fewer, bigger circles are not always cheaper.**
- Prioritize expected **avoidable repair cost**, using both leak probability and repair severity.

Confirm any unspecified details with the instructor or official evaluator, especially the ordering of leaks and work at exactly the start of a year and whether all pipe geometry is represented by straight segments.

## Baseline

Start with one circle per original iron pipe.

For a straight pipe:

```text
centre = midpoint of the two endpoints
radius = half the pipe length
```

Use a small, documented numerical margin if necessary, then recheck containment.

This gives geometric coverage, but **budget feasibility must still be checked**. Baseline circles can contain other pipes, including copper, brass and expensive surface types.

Order the baseline using estimated avoidable repair costs, then evaluate it with the simulator.

## Scorer and validation — first priority

Build the cost function, simulator and validator before optimizing circles.

Check:

- Full containment, boundary cases and overlapping circles.
- Distinct surface counting, including surfaces from polyurethane pipes.
- Historical leaks and material status at the start of 2027.
- Project-window leaks and their effects on later costs.
- Zero-cost skipped units.
- Exact-budget execution and budget carryover.
- Stopping at the first unaffordable work unit.
- Complete geometric coverage of every original iron pipe.
- Completion by 2051.
- Correct parsing and validation of the exported submission CSV.

For each candidate, report:

- Total work-unit costs.
- Estimated repair costs.
- Estimated score.
- Completion year.
- Uncovered iron count.
- Model/data version and random seed.

Unknown future leaks mean our score is an **estimate**, not the official future score. Compare candidates using the same leak scenarios and record uncertainty.

Also run a no-future-leak simulation as a construction-budget check.

## Data and backtesting

- Preserve pipe IDs as strings.
- Check duplicate records, missing values, dates, units and material/surface labels.
- Verify joins between `pipes.csv` and `train.csv`.
- Keep raw data unchanged.
- Fit models on **2019–2023** and evaluate on **2024–2026**.
- Reconstruct pipe status at each cutoff using only information available at that time.
- Do not use future leaks, future repair costs or future replacement status as training features.

Historical backtesting evaluates predictions. It does not directly observe the score of a proposed 2027–2051 replacement plan.

## Roles

| # | Role | Person |
| --- | --- | --- |
| 1 | Data cleaning and exploratory analysis | |
| 2 | Leak probability model (per pipe, per year) | |
| 3 | Leak cost model and backtest (fit on 2019–2023, test on 2024–2026) | |
| 4 | Cost function, scorer and validator (**first priority**) | |
| 5 | Circle design (grouping pipes cheaply) | |
| 6 | Ordering within the budget, final submission | |

## Steps

1. **Week 1:** everyone reads the specification. Build the scorer and validator, clean the data, and make a simple leak-risk estimate.
2. **Week 2:** create a baseline submission, check coverage and budget feasibility, and estimate its score.
3. **Weeks 3–4:** improve the risk model, circles and ordering. Keep changes supported by validation that outperform the best candidate so far.
4. **Week 5:** pick the final submission, have someone else independently recheck the exported CSV, and write the report.

Work on branches and open pull requests into `main`.

Record each candidate's method, model/data version, random seed, costs, completion year and validation result so results can be reproduced.


## Data handling reminders

- Treat both `cast iron` and `wrought iron` as iron replacement targets. Check the actual material labels before filtering.
- Keep missing materials as unknown; do not assume these pipes are polyurethane or exempt from replacement. Confirm how they should be handled.
- For pipes replaced after historical leaks, the original `Lay date` is not the installation date of the replacement polyurethane pipe.
- EDA surface maps show estimated regions, not measured land boundaries. Calculate work-unit surface costs from the labels of wholly contained pipes, not map colours or area proportions.

EDA documentation 
[English](notebooks/EDA/EDA.md) and
[Japanese](notebooks/EDA/EDA_Japanese.md).

















### 1. 課題

2027〜2051年の25年間で、すべての鉄管を polyurethane（ポリウレタン）管に交換します。

毎年1,000万ドル、合計2億5,000万ドルの予算があります。評価は、

**スコア ＝ 2億5,000万ドル − 交換工事費 − 期間中の漏れの修理費**

です。つまり、**工事を安くして、修理費の高い漏れを先回りして防ぐほど高得点**になります。

### 2. 提出するのは「円のリスト」

提出する CSV の各行には、円の中心と半径を書きます。

```csv
"(1234.5, 6789.0)",25.0
```

これは「中心が `(1234.5, 6789.0)`、半径25mの円で工事する」という意味です。座標と半径の単位はメートルです。

**行の順番が工事の順番**になります。

円に管の一部分が入っているだけでは交換されません。管全体が入っている必要があります。直線の管なら、両端が円の内側か境界上にあるかを確認します。

すべての元の鉄管が、少なくとも1つの提出する円に完全に含まれる必要があります。

また、円の中に完全に入っている、現在ポリウレタンではない管は、**鉄だけでなく銅や真鍮も交換されます**。

### 3. 工事費

- 1回の工事につき **100ドル**
- 円の面積と、含まれる地表の種類から計算する **地表工事費**
- 交換する管の長さ1mにつき **200ドル**

```text
円の面積 = π × 半径²

工事費 =
    100
    + (地表単価の合計 × 円の面積)^0.85
    + 200 × 交換する管の合計長さ
```

最後に、工事費をセント単位に丸めます。

地表工事費は少し特殊です。例えば円の中に草地と道路の管があれば、単価を `3 + 17 = 20` として、**円全体の面積**に掛けます。その結果を0.85乗します。

草地の管が何本あっても、草地の単価は1回だけ足します。一方、建物の管が1本でも完全に含まれると、建物の単価50が加わります。

**すでにポリウレタンになった管の地表の種類も計算に含まれます。** ただし、その管の交換費200ドル/mは掛かりません。

前の工事で交換された管が別の円にも含まれている場合も、交換費を二重に払うことはありません。ただし、地表の種類には引き続き含まれます。

円の中に管がない場合、または現在ポリウレタンの管しかない場合は、工事がスキップされ、100ドルの基本料金も含めて費用は掛かりません。

### 4. 年間予算と順番

毎年の初めに1,000万ドルが追加され、CSV の順番どおりに工事を実行します。工事自体は瞬時に完了する設定です。

例えば、ある年の予算残高が200万ドルで、次の工事が300万ドルなら、その年はそこで止まります。

翌年に1,000万ドルを追加するので、残高は1,200万ドルになり、止まっていた工事から再開します。

**残った予算は繰り越せますが、後ろにある安い工事を先に実行することはできません。**

予算残高と工事費が同額なら実行できます。また、1つの工事が1,000万ドルを超えていても、繰越予算が十分にあれば実行できます。

25年間で工事を完了できない計画は受理されません。そのため、円を作るだけでなく、リストの順番も考える必要があります。

なお、早く工事が終わっても、スコアの余剰予算は合計2億5,000万ドルから工事費を引いて計算します。漏れの修理費はスコアから別途引かれます。

### 5. 漏れた管

2019〜2026年の履歴にある漏れた管は、修理時にすでにポリウレタンへ交換されています。

`pipes.csv` の材質は元の材質なので、そのまま現在の状態として使ってはいけません。元の材質と、2027年時点の材質を分けて管理します。

2027年以降も、漏れると修理費が発生し、その管はポリウレタンになります。その後の工事では、その管の交換費が不要になります。

したがって、**工事の順番は、漏れの修理費だけでなく工事費にも影響します。**

ただし、漏れを待つと修理費が発生するので、「交換費が減ったから得」とは限りません。工事費と修理費の両方を合わせて判断します。

予定した工事が早く終わっても、2027〜2051年の全期間に発生する漏れの修理費が評価対象です。

### 6. 最初に何を予測する？

最初の優先事項は、過去のデータを使って、各管について次の2つを推定することです。

- **いつ漏れそうか：** 年ごとの漏れ確率、または次の漏れまでの時間。
- **漏れた場合にいくら掛かりそうか：** 修理費。

将来の漏れの日時や修理費は与えられていないため、**自分たちでモデルを選び、過去データから予測する必要があります。**

ただし、次の漏れ日を正確に当てる必要があるとは限りません。例えば「今後1年以内に漏れる確率」や「漏れるまでの時間の分布」を推定する方法も考えられます。

モデルの種類は課題で指定されているわけではなく、自分たちで選択して比較します。

優先順位を考える際は、漏れやすさだけでなく、漏れた場合の修理費も重要です。

```text
ある期間の期待修理費
≈ その期間に漏れる確率 × 漏れた場合の予想修理費
```

これは優先順位を考えるための基本的な目安です。最終的には、工事費や予算による実行時期も含めて比較します。

過去データでは、2019〜2023年でモデルを作り、2024〜2026年で予測を確認する方針です。

その際、**予測する時点より後の情報を使わない**ようにします。例えば、2024年を予測するときに、2025年に漏れたという情報を使ってはいけません。

### 7. 予測した後　--> 円と工事の順番を決める

漏れのリスクと修理費の予測を使って、次に以下を決めます。

- 円の中心をどこに置くか。
- 円の半径をどれくらいにするか。
- どの管を同じ円にまとめるか。
- どの地域・どの管から先に工事するか。
- その順番で、各工事を何年に実行できるか。

まずは、鉄管1本ずつを囲む円を作る方法を baseline（比較の基準）にします。

直線の管なら、中心は管の中点、半径は管の長さの半分です。数値の丸めによる囲み漏れが起きないよう、出力後にも確認します。

これで鉄管を囲めますが、**25年間の予算内に収まるかは別途確認が必要**です。また、小さい円でも他の管が入ることがあります。

そこから、次のように改善します。

- 漏れそうで、漏れたときの修理費も高い管を早めに交換する。
- 近い鉄管を同じ円にまとめて工事費を減らす。
- 円を広げすぎて、銅・真鍮や高い地表の種類を余分に含めない。

「大きい円にまとめれば必ず安い」「漏れる確率が高い管から交換すれば必ず良い」という単純な話ではなく、計算して比較します。

### 8. 計画を比較するための道具

予測モデルの準備と並行して、**scorer／simulator／validator** を作ります。

- **Scorer：** 計画の工事費、予想修理費、予想スコアを計算する。
- **Simulator：** 年ごとの予算、工事、漏れ、管の状態変化を再現する。
- **Validator：** 鉄管の囲み漏れや、25年間で完了できない問題がないか確認する。

これがないと、モデルや円の作り方を改善しても、本当に良くなったか判断できません。

将来の漏れは未知なので、自分たちが計算するスコアは予測に基づく推定値です。候補を比較するときは、同じ条件・同じ漏れシナリオで評価します。

最終提出前には、作成中のデータだけでなく、**実際に提出する CSV を読み直して検証**します。

### 9. データを扱うときの注意

- 鉄管には `cast iron`（鋳鉄）と `wrought iron`（錬鉄）の両方を含めます。
- 材質不明の管を、勝手にポリウレタンや交換不要と判断しません。扱いを確認します。
- 過去の漏れで交換された管の元の `Lay date` は、新しいポリウレタン管の敷設日ではありません。
- EDA の地表地図は推定図です。工事費は地図の色や面積比ではなく、円に完全に含まれる管の `Surface` ラベルから計算します。
- 元のデータは変更せず、補正や派生変数はコードで再現できるようにします。

### 10. チームの進め方

1. 課題のルールとデータを確認する。
2. 漏れの時期・確率と修理費を予測するモデルを選び、検証する。
3. 並行して scorer／simulator／validator を準備する。
4. baseline の円と工事順序を作り、費用・スコア・実行可能性を確認する。
5. 予測を使って、円の大きさ・位置・組合せ・工事順序を改善する。
6. 最終 CSV を別のメンバーが再確認し、レポートを作成する。
