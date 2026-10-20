import csv,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
p=pd.read_csv(R/'source/pipes.csv');a=p[['GPS x1','GPS y1']].to_numpy();b=p[['GPS x2','GPS y2']].to_numpy()
def read(path):
 return [(np.array(list(map(float,x.strip('()').split(',')))),float(r)) for x,r in csv.reader(path.open())]
def brute(u):
 c,r=u;return (np.linalg.norm(a-c,axis=1)<=r)&(np.linalg.norm(b-c,axis=1)<=r)
old=read(R/'source/09a_full_ensemble_net.csv');new=read(R/'10a_09a_nearby_pairs.csv');pairs=pd.read_csv(R/'pair_merges.csv')
starts={int(row.first_rank)-1:int(row.second_rank)-1 for row in pairs.itertuples()};removed=set(starts.values());k=0
for i,u in enumerate(old):
 if i in removed:continue
 if i in starts:
  required=brute(u)|brute(old[starts[i]])
  assert (brute(new[k])[required]).all()
 else:assert np.array_equal(u[0],new[k][0]) and u[1]==new[k][1]
 k+=1
assert k==len(new)==len(old)-len(pairs)
report={'passed':True,'method':'Independent all-pipe endpoint distance checks, without spatial grid, for every merged pair; every untouched circle matches source exactly','merged_pairs':len(pairs),'source_sha256':hashlib.sha256((R/'source/09a_full_ensemble_net.csv').read_bytes()).hexdigest(),'source_matches_published_09a':hashlib.sha256((R/'source/09a_full_ensemble_net.csv').read_bytes()).hexdigest()=='3c871b165901564653cd1acb56cbfbd0df7f943eaf1ef733e55aa9ca20d7c1c2'}
assert report['source_matches_published_09a']
(R/'independent_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
