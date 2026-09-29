# Star Kingdom：配管網の探索的データ分析

ガス配管網、2019～2026年の漏洩履歴、および2027年時点の配管状態を整理した探索的データ分析です。

## 要約

- 元データは配管43,039本、漏洩記録14,113件です。漏洩記録のある配管は、修理後にポリウレタンへ交換されたものとして2027年時点の素材を再構成しています。
- 過去の修理費総額は **$353.0M** で、上位1%の高額漏洩が **45.3%** を占めます。中央値$3,275に対し平均$25,012で、分布は高額側に長い裾を持ちます。
- 素材別漏洩割合は元の素材・敷設年齢・地表・場所が絡む記述値であり、素材の因果効果や年間確率ではありません。敷設年齢の比較には鉄管だけの図も併記します。
- 同じ元素材の中で、年ごとの1m当たり修理費の平均・中央値と、素材自身の全期間平均からの上下を比較します。
- 地図では、train.csvに記録された漏洩配管を交換後のポリウレタンとして扱います。別図で交換箇所だけを強調し、地表図では推定地表領域を塗り、配管線を一色にします。

## 1. データの範囲と品質

提供ファイルは配管1本につき1行、漏洩配管1本につき1件の記録です。漏洩IDはすべて配管表に結合でき、配管IDの重複、長さ0以下、既知の敷設日より前の漏洩はありません。課題文のcosts.csvに相当する履歴として、提供されたtrain.csvを使用します。

### Missing values in source columns

![Missing values in source columns](figures/01_missingness.png)

敷設日が15件、素材が8件欠損しています。座標と漏洩記録の項目には欠損がありません。不明素材を鉄やポリウレタンと決めつけず、未確定として残します。

## 2. 配管網の構成

### Original pipe materials

![Original pipe materials](figures/02_material_inventory.png)

元の配管は錬鉄が最も多くを占めます。ただし過去の漏洩でポリウレタンに交換されるため、元の素材と計画開始時の素材は一致しません。

### Material changes after historical leak repairs

![Material changes after historical leak repairs](figures/03_material_transition.png)

交換計画の前に必要な補正です。2019〜2026年に漏洩した配管は、元の素材の残存本数から除外し、ポリウレタンとして扱います。

### Network inventory by surface type

![Network inventory by surface type](figures/04_surface_inventory.png)

Surfaceは配管に付いた区分であり、土地の境界や標高ではありません。配管本数や延長の割合を土地面積の割合として解釈してはいけません。

### Distribution of individual pipe lengths

![Distribution of individual pipe lengths](figures/05_length_distribution.png)

これは配管網の総延長ではなく、各配管レコード1本ごとの長さの分布です。長さは記録された両端座標から計算し、1m当たり200ドルの交換費に直接影響します。

### Time since original installation at 2027

![Time since original installation at 2027](figures/06_age_distribution.png)

既知の敷設日はすべて2019年の観測開始より前です。これは元の敷設日からの年数で、交換済みポリウレタンの年齢ではありません。欠損日は除外しています。

### Installation age differs by original material

![Installation age differs by original material](figures/07_age_by_material.png)

素材と敷設年は関連しています。単純な年齢と漏洩の関係には素材構成の違いが混ざるため、老朽化の独立した影響とは断定できません。

### Surface mix within each original material

![Surface mix within each original material](figures/08_material_surface_mix.png)

各行の合計は100%です。素材ごとの地表区分の違いは比較の交絡要因になります。Unknownは8本のみです。

## 3. 漏洩の時系列

### Historical leak counts and repair costs

![Historical leak counts and repair costs](figures/09_annual_history.png)

件数と総費用は同じ動きをしません。1件当たりの損失が変わるためです。8年の履歴から将来の安定した傾向を決めつけたり、2051年まで繰り返したりはできません。

### Monthly leak counts, 2019–2026

![Monthly leak counts, 2019–2026](figures/10_monthly_heatmap.png)

年×月で見ることで、一時的な増加と毎年繰り返す季節性を区別できます。ただし素材別の残存数の変化で補正した値ではありません。

### Annual leaks by original material

![Annual leaks by original material](figures/11_material_time.png)

素材別の系列に分け、全体の変動が各素材に共通するか確認します。この図だけで特定年の増加原因は特定できません。

## 4. 漏洩割合・発生率の比較

### Observed leak fraction by original material

![Observed leak fraction by original material](figures/12_material_leak_fraction.png)

8年間に漏洩記録がある割合とWilson法の95%区間です。年間確率でも素材の因果効果でもありません。区間は空間的依存を考慮していません。

### First-leak incidence by original material

![First-leak incidence by original material](figures/13_exposure_adjusted_rates.png)

元の素材としての観測期間は最初の漏洩・交換時、または2026年末で終了します。Poissonの95%区間を表示しています。ポリウレタンの履歴が0件でも将来リスクが0とは限りません。

### Observed leak fraction: material and surface

![Observed leak fraction: material and surface](figures/14_material_surface_risk.png)

素材と地表区分の組合せによる違いを示します。30本未満のセルは図から除き、出力表には全件数を残しています。

### Observed leaks by original-installation age

![Observed leaks by original-installation age](figures/15_age_leak_fraction.png)

左は全素材、右は元の素材が鉄の配管だけに限定した比較です。鉄だけを見ることで素材構成による交絡は一部抑えられますが、場所や漏洩後の交換による影響は残ります。この棒の差は観測群の違いであり、老朽化の因果効果そのものではありません。

## 5. 過去の修理費の分布

### Repair costs have a long upper tail

![Repair costs have a long upper tail](figures/16_cost_distribution.png)

中央値は3,275ドル、平均は25,012ドルです。対数軸で最高額も含む全データを保持しています。累積分布はstatsmodelsのECDFで計算しました。

### A small share of leaks drives much of the cost

![A small share of leaks drives much of the cost](figures/17_cost_concentration.png)

高額な上位1%の漏洩が費用の45.3%、上位10%が81.6%を占めます。事後的に並べた結果であり、予測モデルがこの集中を捉えられることを意味しません。

### Repair severity by original material

![Repair severity by original material](figures/18_cost_by_material.png)

漏洩が起きた場合の費用分布です。発生確率を掛け合わせた期待損失や交換優先度ではありません。外れ値は削除していません。

### Repair severity by surface type

![Repair severity by surface type](figures/19_cost_by_surface.png)

漏洩が起きた場合の費用分布です。発生確率を掛け合わせた期待損失や交換優先度ではありません。外れ値は削除していません。

### Pipe length and observed repair cost

![Pipe length and observed repair cost](figures/20_length_cost.png)

Spearman順位相関は0.171です。これは配管長の順位と修理費の順位がどの程度そろうかを見る指標で、+1は長いほど必ず高額、0は単調な関係なし、-1は長いほど必ず低額を表します。0.171は弱い正の関連です。六角形は重なる漏洩を集計したもので、配管長が修理費の原因だとは示しません。

### Installation age at leak and repair severity

![Installation age at leak and repair severity](figures/21_age_cost.png)

各点は1件の漏洩で、横軸が漏洩時点の元の敷設年齢、縦軸が修理費です。オレンジのLOWESS曲線は直線関係を仮定せず、対数修理費の局所的な傾向を要約します。Spearman順位相関は-0.018です。年齢が高い観測配管ほど修理費が大きい傾向があったかを見る図であり、予測や老朽化の因果効果ではありません。素材・地表・場所の違いも影響し得ます。

### Repair cost per metre over time within original material

![Repair cost per metre over time within original material](figures/21b_material_cost_per_m_time.png)

左は年別平均で、少数の非常に高額な修理に強く影響されます。右の中央値は典型的な記録を表しやすい指標です。同じ元素材の線を年ごとに比べることで素材間の水準差を避けられますが、地表や場所の構成変化は残ります。金額は物価調整前の名目ドルです。

### Annual mean cost per metre relative to each material's own average

![Annual mean cost per metre relative to each material's own average](figures/21c_material_cost_per_m_heatmap.png)

赤はその素材自身の2019～2026年平均より1m当たり平均修理費が高い年、青は低い年です。各セルには漏洩件数nも示します。同一素材内の上下を見やすくしますが、高額な外れ値が平均を大きく動かし、nが小さいセルは不安定です。

## 6. 配管と記録済み交換箇所の位置

### Pipe-network materials at the start of 2027

![Pipe-network materials at the start of 2027](figures/22_material_map.png)

train.csvに記録された漏洩配管は、漏洩後にポリウレタンへ交換されたものとしてすべてポリウレタン色で表示します。それ以外は元の素材を維持し、素材欠損はUnknownのままです。座標はローカルなメートル座標で、背景地図はありません。

### Estimated surface regions with pipe network

![Estimated surface regions with pipe network](figures/23_surface_map.png)

同じ地表区分と推定された範囲を同じ色で塗り、配管はすべて同じ濃色の線で表示します。背景は配管上の記録から60座標単位以内を最近傍で推定したもので、実測の土地境界・標高・地理背景ではありません。

### Pipes replaced after leaks recorded in train.csv

![Pipes replaced after leaks recorded in train.csv](figures/24_replaced_pipe_map.png)

オレンジ線はtrain.csvに含まれ、記録された漏洩後にポリウレタンへ交換されたと扱う配管です。それ以外はすべて灰色で、元の素材による色分けはしていません。

### Map of the 15 longest pipes

![Map of the 15 longest pipes](figures/25_longest_pipe_map.png)

端点間距離が長い上位15本をそれぞれ別の色で示し、地図上に長さを記載します。それ以外の配管は灰色です。高額修理が長さで説明できそうかを目視する補助図ですが、全体の配管長と修理費のSpearman順位相関は弱い正の関連にとどまり、強調した配管の中には漏洩修理費の記録がないものもあります。

### Map of pipes in the highest 1% of repair costs

![Map of pipes in the highest 1% of repair costs](figures/26_top1_cost_pipe_map.png)

記録修理費が上位1%に入る142本だけを色付けし、その他は灰色にしています。色は修理費の対数尺度です。強調された漏洩は過去の修理費総額の45.3%を占めますが、位置だけで高額になった原因を説明するものではありません。

## 方法と読み方

**観測期間。** 2019-01-01から2026-12-31までを漏洩履歴、2027-01-01を計画開始時点とします。

**素材の更新。** train.csvに漏洩記録がある配管は、課題ルールに従い、漏洩後にポリウレタンへ交換されたものとして2027年時点の素材を更新します。素材欠損はUnknownのままです。元の敷設年齢を交換後のポリウレタン管の年齢としては扱いません。

**配管長。** 配管ごとの長さは、記録された2端点間のユークリッド距離です。「配管長の分布」は、この1本ごとの長さの分布を表します。

**漏洩割合と発生率。** 漏洩割合は8年間に漏洩記録のあった配管の割合です。発生率は最初の漏洩・交換または観測終了までのpipe-yearsを分母にします。いずれも因果効果や将来の年間確率ではありません。

**修理費との関連。** Spearman順位相関は、2変数の大小順位がどの程度同じ向きに並ぶかを-1～+1で表します。LOWESSはデータの局所傾向を滑らかに示す記述線です。どちらも予測式や因果効果を示しません。1m当たり修理費は修理費を各配管の端点間距離で割った名目額で、物価調整はしていません。

**地表図。** 同じ地表区分を同じ色で示すため、配管上の地表ラベルから近傍領域を推定しています。実測の土地境界ではありません。

**再現性。** eda/run_eda.pyが図・表・指標・入力ファイルのハッシュを再生成し、eda/build_reports.pyが同じ根拠から日本語版と英語版を構築します。