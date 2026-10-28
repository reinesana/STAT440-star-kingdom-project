"""Brute-force spot audit of serialized early-three-year submissions."""
import csv, hashlib, json
import numpy as np
from optimize import Problem, ROOT
from early_three_years import OUT

def main():
    pr=Problem();rng=np.random.default_rng(442);result={}
    for name in ['early3_adaptive','early3_reorder_only']:
        path=OUT/f'{name}.csv'
        with path.open() as f:rows=list(csv.reader(f))
        for j in rng.choice(len(rows),100,replace=False):
            c=np.array(list(map(float,rows[j][0].strip('()').split(','))));r=float(rows[j][1])
            full=np.flatnonzero((np.linalg.norm(pr.a-c,axis=1)<=r)&(np.linalg.norm(pr.b-c,axis=1)<=r))
            near=pr.grid.query(c,r)
            inside=near[(np.linalg.norm(pr.a[near]-c,axis=1)<=r)&(np.linalg.norm(pr.b[near]-c,axis=1)<=r)]
            assert set(full)==set(inside)
        result[name]=dict(rows=len(rows),full_data_brute_force_checks=100,passed=True,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'spot_audit.json').write_text(json.dumps(result,indent=2))
    assert hashlib.sha256((ROOT/'outputs/multi_pipe.csv').read_bytes()).hexdigest()=='a9f62647860ea96aa59a0a5b48eac0662e57e544e86ee77bec68612c8487daba'
    print('200 exported circles passed brute-force checks; previous submission unchanged')

if __name__=='__main__':main()
