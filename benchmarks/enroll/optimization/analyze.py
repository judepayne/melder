import json,statistics,hashlib,tarfile,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;out=Path('benchmarks/enroll/optimization');out.mkdir(exist_ok=True);r=json.loads((D/'results.json').read_text());assert len(r)==21
stats={};audit=[]
def flat(job):
 a=json.loads((D/job/'responses.json').read_text());return [v for q in a for v in q['body'].get('results',[q['body']])]
for c in ['c1','c10','bulk50']:
 g={b:[x for x in r if x['case']==c and x['binary']==b and x['trial']>0] for b in ['baseline','candidate']};rates={b:statistics.median(x['records_s'] for x in v) for b,v in g.items()};stats[c]={**rates,'speedup':rates['candidate']/rates['baseline']}
 for rep in [1,2,3]:
  a,b=[flat(next(x['job'] for x in g[k] if x['trial']==rep)) for k in g];assert len(a)==len(b)==1000;ids=sets=0;maxdiff=0;unequal=0
  for u,v in zip(a,b):
   assert u['id']==v['id']; ue=u['edges'];ve=v['edges'];ids+= [e['id'] for e in ue]!=[e['id'] for e in ve];sets+=set(e['id'] for e in ue)!=set(e['id'] for e in ve)
   vs={e['id']:e for e in ve}
   for e in ue:
    if e['id'] in vs:maxdiff=max(maxdiff,abs(e['score']-vs[e['id']]['score']))
   unequal+=u!=v
  audit.append(dict(case=c,trial=rep,ordered_edge_differences=ids,edge_set_differences=sets,max_score_difference=maxdiff,response_differences=unequal))
print(json.dumps(stats,indent=2));print(json.dumps(audit,indent=2));(out/'results.json').write_text(json.dumps({'runs':r,'summary':stats,'audit':audit},indent=2))
(out/'measure.py').write_text((D/'measure.py').read_text());(out/'analyze.py').write_text(Path(__file__).read_text());meta={'baseline_rev':'3feb0270dd4f9ffb6611b997c8c47aa2b23d98bd','hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode().strip(),'sha256':{f:hashlib.sha256((D/f).read_bytes()).hexdigest() for f in ['baseline','candidate','seed.csv','records.json']}};(out/'manifest.json').write_text(json.dumps(meta,indent=2))
with tarfile.open(out/'evidence.tar.gz','w:gz') as t:
 for x in D.iterdir():
  if x.is_dir() and (x/'responses.json').exists():
   for f in ['server.log','config.yaml','responses.json']:t.add(x/f,arcname=x.name+'/'+f)
