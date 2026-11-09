from pathlib import Path
from decimal import Decimal
import json,csv,hashlib
HERE=Path(__file__).resolve().parent
INFO={
 'TP01':dict(submission_id='S1txbe31ipyk',result_url='http://172.16.42.11:51238/static/output/C2VLj4J71iElYOKcwblv8ZET.txt'),
 'TP02':dict(submission_id='SklR6l27kez6',result_url='http://172.16.42.11:51238/static/output/fMIxd_AyXDX9sLyhd3Ky0Jqh.txt')}
def main():
    path=HERE/'official_submissions.json';records=json.loads(path.read_text());locked=json.loads((HERE/'locked_before_public.json').read_text())
    assert hashlib.sha256((HERE/'confirmation_2026.csv').read_bytes()).hexdigest()==locked['internal_metrics_sha256']
    assert hashlib.sha256((HERE/'G17_simulation_summary.csv').read_bytes()).hexdigest()==locked['simulation_summary_sha256']
    for row in records:
        row.update(INFO[row['candidate']]);file=HERE/f'official_{row["candidate"]}.csv'
        assert hashlib.sha256((HERE/row['csv']).read_bytes()).hexdigest()==row['sha256']
        if file.exists():
            with file.open(encoding='utf8') as f:annual=list(csv.DictReader(f))
            assert [int(r['YEAR']) for r in annual]==list(range(1,26))
            cap=sum(Decimal(r['CAPEX']) for r in annual);op=sum(Decimal(r['OPEX']) for r in annual);score=Decimal(250000000)-cap-op
            row.update(status='complete',public_score=str(score),CAPEX=str(cap),OPEX=str(op),delta_vs_G17=str(score-Decimal('79168928.88')),official_annual_breakdown=file.name)
        else:row.update(status='evaluating',official_annual_breakdown='pending')
    path.write_text(json.dumps(records,indent=2),encoding='utf8');print(json.dumps(records,indent=2))
if __name__=='__main__':main()
