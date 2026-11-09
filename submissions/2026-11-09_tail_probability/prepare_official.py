from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=HERE/'study.py';source=p.read_text(encoding='utf8')
    new="[('selected',name),('CMS001','CMS001'),('standard_LGBM','standard_LGBM'),('tail_probability_only','tail_probability_only')]"
    old="[('selected',name),('CMS001','CMS001'),('standard_LGBM','standard_LGBM')]"
    original=source.replace(new,old);lock=json.loads((HERE/'locked_before_2026.json').read_text())
    assert hashlib.sha256(original.encode()).hexdigest()==lock['source_code_sha256']
    (HERE/'selection_code_snapshot.py').write_text(original,encoding='utf8')
    records=[]
    for label,filename,method in [('TP01','selected_G17.csv',lock['model']),('TP02','tail_probability_only_G17.csv','lightgbm_basic_bins3_raw; F168 * p(high cost) / construction cost')]:
        path=HERE/'submission_files'/filename;valid=json.loads(path.with_suffix('.validation.json').read_text())
        assert valid['sha256']==sha(path) and valid['completed_by_2051']
        records.append(dict(candidate=label,method=method,csv=str(path.relative_to(HERE)),sha256=sha(path),rows=valid['work_unit_count'],status='prepared',submission_id=None,public_score=None,official_annual_breakdown='pending',authorized_user_request='Evaluate jointly using internal indicators and actual submitted Public scores',hyperparameters_changed_after_Public=False))
    frozen=dict(candidates=records,internal_metrics_sha256=sha(HERE/'confirmation_2026.csv'),simulation_summary_sha256=sha(HERE/'G17_simulation_summary.csv'),selection_lock_sha256=sha(HERE/'locked_before_2026.json'),official_submissions_authorized=2)
    (HERE/'locked_before_public.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    if not (HERE/'official_submissions.json').exists():(HERE/'official_submissions.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
