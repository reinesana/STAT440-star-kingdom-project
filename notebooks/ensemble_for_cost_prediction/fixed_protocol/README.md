# Fixed specification: CMS002 cost and HFX001 annual leak risk

This update changes the selection process. **It does not claim better predictive
accuracy or a new untouched test.** The user's requested policy is to remove both
public-score-driven adjustment and repeated backtest-driven candidate selection.
CMS001 and F168 remain historical models with their original selection provenance.

The four underlying predictors are material, surface, straight-line length (m),
and age at year start. Missing age and length add two flags, giving **six input
columns**. Pipe IDs are join keys only. There are no neighborhood/cost-history
aggregates, installation cohorts, manual probability overrides, stacking,
threshold searches, feature searches, hyperparameter searches or early stopping.

Both targets use one fixed shallow CatBoost specification: 200 iterations,
depth 4, learning rate 0.05 and L2=10. These are conservative engineering choices,
not settings selected for winning any reported year. The hazard target is annual
first observed leak; the cost target is raw conditional event cost. Cost estimates
are clipped at zero by the target's physical support. No upper clipping or
post-hoc calibration is applied. The complete choices are in [PROTOCOL.json](PROTOCOL.json).

## 日本語

**公開スコアを見た調整と、過去検証を繰り返し見て候補を選ぶ工程を外した版**。
修理費をCMS002、翌1年の漏水確率をHFX001として新しく区別する。
F168/CMS001のパラメーターやOOF予測は再利用しない。少ない入力と浅い木を固定し、
検証結果の良し悪しを理由に設定を変更したり、旧版へ選び戻したりしない。

過去年は他の実験で既に見ているので、2023〜2026年を「完全未使用テスト」とは呼ばない。
過去の知識も消せないため、完全な意味での事前登録や、過去の適応的な選定の影響が
ゼロになったことは主張しない。これは**今後の選定手順のリセット**であり、精度改善や
privateスコアの向上を保証するものではない。特に高額修理費の予測を単純化すると
精度が下がる可能性がある。測定結果は記述的な時系列バックテストとして全年度を残す。

仕様とコードを先にGitで固定し、データ完全性テストの後に学習する。
全年度と2027年モデルの予測が終わってから指標を計算し、その指標を学習側へ渡さない。
公式採点サイトへの提出、公開スコアによる候補比較、工事計画の最適化はこの更新には含めない。
25年先の精度を検証したとは扱わず、配布する予測は2027年だけ。

## Reproduce

From the repository root, install this folder's requirements and run:

```powershell
python -m unittest discover -s notebooks/ensemble_for_cost_prediction/fixed_protocol -v
python notebooks/ensemble_for_cost_prediction/fixed_protocol/model.py --data-dir data --output-dir output/fixed_protocol_reproduction
```

Choose a fresh output directory. Existing runs are never overwritten. The run
records protocol/source/input SHA256 hashes before fitting and records environment
versions. Predictors and labels are reconstructed from the two raw supplied CSVs.
Each backtest year uses only earlier training years. Categorical encoding and
numeric borders are learned within each model's own training data.

`artifacts/predictions_2027.csv` contains every inventory pipe, eligibility,
missing-input flags, annual probability, conditional repair cost and their product.
`artifacts/cost_predictions_2027.csv` is the two-column severity join interface.
Inactive pipes get zero planning cost/risk; this is an eligibility rule, not an
estimate of repair severity if those inactive pipes could fail. Unknown material
and missing required attributes are retained with `input_incomplete=true`; their
predictions are model extrapolations, not validated unknown-category forecasts.

Use `hazard.cbm` with `CatBoostClassifier.load_model` and `severity.cbm` with
`CatBoostRegressor.load_model`, then use `features` and `forecast` in `model.py`.
Never load model files from an untrusted source. Data-integrity tests cover the
year boundary, removal after first leak, unknown inputs and future-label mutation.
Deployment checks require exact save/reload and input-row-reversal agreement.

Historical comparisons and public scores belong to their original models. The
new model has no official score and no established superiority over CMS001/F168.
The prior models' reports and scores are preserved unchanged for provenance.
