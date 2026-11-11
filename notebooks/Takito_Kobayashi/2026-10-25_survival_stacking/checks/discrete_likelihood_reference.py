"""Check event and right-censor interval indexing against pycox losses."""
from pathlib import Path
import json
import numpy as np
import torch
from pycox.models.loss import nll_pmf, nll_mtlr

torch.manual_seed(440)
n,k=100,5
raw=torch.randn(n,k+1)
event=torch.rand(n)>.5
duration=torch.randint(0,k,(n,)).float()+.5
duration[~event]=torch.randint(1,k+1,(int((~event).sum()),)).float()
idx=duration.floor().long()
reference_idx=torch.where(event,idx,idx-1)
results=[]
for name in ['DeepHit_PMF','MTLR']:
    score=raw if name=='DeepHit_PMF' else torch.flip(torch.cumsum(torch.flip(raw,[1]),1),[1])
    lp=torch.log_softmax(score,dim=1)
    le=lp.gather(1,idx[:,None]).squeeze(1)
    lc=torch.logsumexp(lp.masked_fill(torch.arange(k+1)[None,:]<idx[:,None],-torch.inf),dim=1)
    own=-torch.where(event,le,lc).mean()
    if name=='DeepHit_PMF': reference=nll_pmf(raw[:,:k]-raw[:,k:],reference_idx,event)
    else:reference=nll_mtlr(raw[:,:k],reference_idx,event)
    delta=float(torch.abs(own-reference))
    assert delta<1e-5,(name,delta)
    results.append({'model':name,'absolute_NLL_difference':delta,'reference':'pycox','passed':True})
path=Path(__file__).with_name('discrete_likelihood_reference_results.json')
path.write_text(json.dumps(results,indent=2))
print(results)
