"""Check 100 serialized disks per arm against all pipes, bypassing spatial index."""
import csv, json
import numpy as np
from optimize import ROOT, Problem

def main():
    pr=Problem();rng=np.random.default_rng(440);out={}
    for name in ['single_pipe','pairwise','multi_pipe']:
        with (ROOT/'outputs'/f'{name}.csv').open() as f:rows=list(csv.reader(f))
        tested=[]
        for j in rng.choice(len(rows),100,replace=False):
            row=rows[j];c=np.array(list(map(float,row[0].strip('()').split(','))));r=float(row[1])
            full=np.flatnonzero((np.linalg.norm(pr.a-c,axis=1)<=r)&(np.linalg.norm(pr.b-c,axis=1)<=r))
            near=pr.grid.query(c,r)
            indexed=near[(np.linalg.norm(pr.a[near]-c,axis=1)<=r)&(np.linalg.norm(pr.b[near]-c,axis=1)<=r)]
            assert set(full)==set(indexed)
            tested.append(int(j)+1)
        out[name]={'brute_force_checked_rows':tested,'passed':True}
    out['extra_data_checks']={
        'zero_length_pipes':int((pr.length==0).sum()),
        'min_length':float(pr.length.min()),'max_length':float(pr.length.max()),
        'min_coordinate':float(min(pr.a.min(),pr.b.min())),
        'max_coordinate':float(max(pr.a.max(),pr.b.max())),
        'events_on_initial_polyurethane':int(pr.hist['Pipe ID'].isin(pr.p.loc[pr.p.Material.eq('polyurethane'),'Pipe ID']).sum())}
    (ROOT/'outputs/independent_spot_audit.json').write_text(json.dumps(out,indent=2))
    print('300 exported circles passed full-data brute-force containment checks')

if __name__=='__main__':main()
