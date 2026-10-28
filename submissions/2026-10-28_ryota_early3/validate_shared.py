"""Audit the frozen submitted CSVs with the repository's current shared code.

Strict all-original-iron geometry failure is recorded, never silently weakened.
Active-iron validation and zero-hazard simulation are separate cross-checks.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'src'))
from build_cost_submissions import validate, read_units, containment
from scipy.spatial import cKDTree
import simulate_score as shared_sim

def main():
    pipes=pd.read_csv(REPO/'data/pipes.csv',dtype={'Pipe ID':str})
    history=pd.read_csv(REPO/'data/train.csv',dtype={'Pipe ID':str})
    a=pipes[['GPS x1','GPS y1']].to_numpy(float);b=pipes[['GPS x2','GPS y2']].to_numpy(float)
    mid=(a+b)/2;length=np.linalg.norm(b-a,axis=1)
    original=pipes.Material.isin({'cast iron','wrought iron','gray iron'}).to_numpy()
    start=(pipes.Material.eq('polyurethane')|pipes['Pipe ID'].isin(history['Pipe ID'])).to_numpy()
    active=original&~start
    result={}
    for relative in ['outputs/multi_pipe.csv','outputs_early3/early3_adaptive.csv']:
        path=HERE/relative;inside=containment(read_units(path),cKDTree(mid),a,b)
        covered=np.zeros(len(pipes),bool)
        for ids in inside:covered[ids]=True
        try:
            validate(path,pipes,history,a,b,mid,length,original)
            strict=True;reason=None
        except AssertionError as e:
            strict=False;reason=str(e)
        # The separate report explicitly describes the supplied target mask.
        report=validate(path,pipes,history,a,b,mid,length,active)
        report.update(target_scope='Initially unreplaced iron ONLY',strict_all_original_iron_geometry_passed=strict,
            strict_failure=reason,uncovered_original_iron_count=int((original&~covered).sum()),
            uncovered_active_iron_count=int((active&~covered).sum()),
            note='User-submitted scored artifact preserved byte-for-byte; not fully compliant with the repository all-original geometric coverage check.')
        path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
        sim_pipes=pipes.copy();sim_pipes['mat']=pipes.Material.fillna('unknown')
        sim_pipes['leak_date']=pd.to_datetime(pipes['Pipe ID'].map(history.set_index('Pipe ID').Date))
        radii=np.array([u[1] for u in read_units(path)])
        sim=shared_sim.simulate(inside,radii,sim_pipes,length,np.zeros((len(pipes),25)),[np.array([0.])]*len(pipes),1)
        simulated=float(sim.iloc[0].construction)
        assert abs(simulated-report['construction_cost'])<.01
        result[relative]=dict(strict_original_geometry_passed=strict,strict_failure=reason,
            uncovered_original_iron=int((original&~covered).sum()),uncovered_active_iron=int((active&~covered).sum()),
            active_iron_no_future_leak_validation_passed=True,
            shared_simulator_no_future_leak_construction=simulated,
            cost_model_version='2026-10-26-area-exponent',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPO/'data/pipes.csv',REPO/'data/train.csv']}
    result['shared_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPO/'src/build_cost_submissions.py',REPO/'src/simulate_score.py']}
    result['shared_doc_access']='Read attempted during GitHub integration; Google Doc unavailable via web tool. No retraining/new optimization performed during integration.'
    (HERE/'outputs_early3/shared_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
