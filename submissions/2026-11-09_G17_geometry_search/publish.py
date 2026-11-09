"""Publish this user-authorized submission record, without changing main."""
from pathlib import Path
import importlib.util, json, hashlib, base64

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('publication_helpers',HERE.parent/'2026-11-09_asymmetric_cost/publish_official.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
BRANCH='takito/g17-geometry-search-2026-11-09'
PREFIX='submissions/2026-11-09_G17_geometry_search'

def main():
    records=json.loads((HERE/'official_submissions.json').read_text())
    assert records and all(x['status']=='complete' for x in records)
    for record in records:assert p.sha(HERE/record['csv'])==record['sha256']
    files=[x for x in HERE.iterdir() if x.is_file() and x.suffix in {'.py','.json','.csv','.txt','.jpg'}
           and x.name not in {'publication.json','PUBLICATION_SHA256.json'}]
    manifest={x.name:p.sha(x) for x in files}
    (HERE/'PUBLICATION_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    files.append(HERE/'PUBLICATION_SHA256.json')
    token=p.credential();assert p.api(token,'')['permissions']['push']
    if (HERE/'publication.json').exists():
        previous=json.loads((HERE/'publication.json').read_text())
        assert p.api(token,'git/ref/heads/'+BRANCH)['object']['sha']==previous['commit']
        print(json.dumps(previous));return
    parent=p.api(token,'git/ref/heads/main')['object']['sha']
    base=p.api(token,'git/commits/'+parent)['tree']['sha'];entries=[]
    for path in sorted(files):
        blob=p.api(token,'git/blobs','POST',dict(content=base64.b64encode(path.read_bytes()).decode(),encoding='base64'))
        entries.append(dict(path=PREFIX+'/'+path.name,mode='100644',type='blob',sha=blob['sha']))
    tree=p.api(token,'git/trees','POST',dict(base_tree=base,tree=entries))
    commit=p.api(token,'git/commits','POST',dict(message='Evaluate G17 circle geometry with paired simulations and official Public results',tree=tree['sha'],parents=[parent],author=p.IDENTITY,committer=p.IDENTITY))
    p.api(token,'git/refs','POST',dict(ref='refs/heads/'+BRANCH,sha=commit['sha']))
    assert p.api(token,'git/ref/heads/'+BRANCH)['object']['sha']==commit['sha']
    for record in records:
        entry=next(x for x in entries if x['path']==PREFIX+'/'+record['csv'])
        blob=p.api(token,'git/blobs/'+entry['sha'])
        assert hashlib.sha256(base64.b64decode(blob['content'])).hexdigest()==record['sha256']
    result=dict(branch=BRANCH,commit=commit['sha'],files=len(files),submitted_remote_hashes_verified=True,
                url=f'https://github.com/{p.REPO}/tree/{BRANCH}/{PREFIX}')
    (HERE/'publication.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
