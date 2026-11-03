"""Build a portable executed results notebook and concise Japanese result log."""
import json
from pathlib import Path
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient
from data import BASE
from models import MODELS


def main():
    results=BASE/'results'
    scores=pd.read_csv(results/'annual_metrics.csv')
    assert len(scores)==40,'All ten models and four outer years must finish first'
    oof=pd.read_csv(results/'stacking_oof.csv.gz')
    wide=oof.pivot(index=['pipe_id','year','y_true'],columns='model',values='predicted_probability')
    assert wide.notna().all().all()
    y=wide.index.get_level_values('y_true').to_numpy()
    wide.corr().to_csv(results/'oof_prediction_correlations.csv')
    wide.subtract(y,axis=0).pow(2).corr().to_csv(results/'oof_squared_error_correlations.csv')
    summary=pd.read_csv(results/'mean_annual_metrics.csv')
    lines=['# 故障年予測：10モデルの独立最適化実験','',
        '2019–2026年の履歴を使用。2023–2026年を各年より前のデータだけで予測し、各学習期間内の2年で特徴量とパラメータを同時選択した。',
        '主指標は4年のBrier scoreの単純平均。小さい方がよい。予測の早め・遅めに非対称な罰則は用いていない。','',
        '| Model | 2023 Brier | 2024 | 2025 | 2026 | Mean |','|---|---:|---:|---:|---:|---:|']
    table=scores.pivot(index='model',columns='year',values='brier')
    for name in MODELS:
        vals=table.loc[name];lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*vals,vals.mean()])+' |')
    lines+=['','## 読み方','',
        'これは故障年の正解率ではなく、観測可能な将来の故障確率を検証した結果である。長期故障年は中央値と10/90%予測分位点として保存した。',
        'Cox・RSF・MTLR・DeepHitは観測追跡期間を超える予測を欠損にしている。他のモデルの長期予測も外挿の仮定を伴う。',
        '特徴量選択・最適化履歴は各モデルの各予測年フォルダに保存した。models/の学習済みバイナリはローカルのみ。',
        '最適化は指定範囲の有限探索であり、全特徴量組合せに対する大域最適性を保証しない。','',
        '## Stacking用データ','',
        '`results/stacking_oof.csv.gz` のキーは pipe_id・year・model。全モデルで同じ対象管を予測している。',
        '誤差相関だけでなく予測相関、二乗誤差相関、材質・敷設年代別誤差も保存した。',
        '次段階のstackingでも時系列分割が必要。例えば2023–2025年OOFでメタモデルを学習し、2026年で評価する。',
        '今回の結果を見てモデルを選択した後の2026年は、新たな未使用テストとは呼べない。','',
        '共有Docへの書込みは行っていない。shared_doc_*.tsv は既存の年別フォーマットに合わせた転記用ファイル。']
    (BASE/'REPORT_ja.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    nb=nbf.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbf.v4.new_markdown_cell('# Failure timing: ten survival models\n\n## tl;dr\nResults are temporal backtests, not validation of exact 2030–2051 failure years. Lower Brier is better. Each model was tuned independently using prior years only.'),
        nbf.v4.new_markdown_cell('## Context & Methods\nSee README.md and experiment_contract.json. Outer years: 2023–2026. Inner selection: the two preceding years. Full prior-year risk sets are used. Repeated landmarks from the same pipe are correlated.\n\n### Key assumptions\nNo preventive replacement during a forecast; past neighborhood history is frozen. Long tails require extrapolation. Empirical-support models retain missing long-horizon predictions.'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nbase=Path.cwd()\nannual=pd.read_csv(base/'results/annual_metrics.csv')\nsummary=pd.read_csv(base/'results/mean_annual_metrics.csv').sort_values('brier')\nplt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})"),
        nbf.v4.new_markdown_cell('## Data\nSource: repository data/pipes.csv and data/train.csv; 2019–2026. Input hashes and future-outcome mutation tests are recorded in input_audit.json.'),
        nbf.v4.new_code_cell("display(annual.groupby('year')[['rows','actual_events']].first())\ndisplay(summary.round(5))"),
        nbf.v4.new_markdown_cell('## Results\n### Annual probability accuracy'),
        nbf.v4.new_code_cell("fig,ax=plt.subplots(figsize=(10,5))\nfor model,g in annual.groupby('model'):\n    ax.plot(g.year,g.brier,marker='o',label=model)\nax.set(xlabel='Forecast year',ylabel='Next-year Brier score',title='Forward validation: 2023–2026')\nax.set_xticks([2023,2024,2025,2026]);ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)\nfig.tight_layout();plt.show()"),
        nbf.v4.new_markdown_cell('### Complementarity for future stacking\nSquared-error correlations compare where models incur loss; prediction correlations distinguish similar forecasts. These are descriptive, not proof that a stacker will improve.'),
        nbf.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(13,6))\nfor ax,file,title in zip(axes,['oof_prediction_correlations.csv','oof_squared_error_correlations.csv'],['Prediction correlation','Squared-error correlation']):\n    c=pd.read_csv(base/'results'/file,index_col=0)\n    im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')\n    ax.set_xticks(range(len(c)),[x[:2] for x in c.columns]);ax.set_yticks(range(len(c)),c.index,fontsize=8)\n    ax.set_title(title)\nfig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6);plt.show()"),
        nbf.v4.new_markdown_cell('## Takeaways\nUse annual and subgroup accuracy together with diversity diagnostics. Small tuning-score differences do not establish statistical superiority. Saved deployment outputs return median failure year where identifiable, predictive quantiles, and an explicit support status. Do not interpret missing long-horizon medians as zero failure risk.')]
    nbf.validate(nb)
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(BASE)}}).execute()
    nbf.write(nb,BASE/'results_notebook.ipynb')
    print('Wrote REPORT_ja.md and executed results_notebook.ipynb')


if __name__=='__main__':main()
