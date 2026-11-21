import concurrent.futures as cf,csv,datetime,hashlib,json,os,shutil,signal,socket,statistics,subprocess,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=Path.cwd();template=(ROOT/'benchmarks/enroll/10kx10k_enroll3k_usearch/warm/config.yaml').read_text(); prefix='benchmarks/enroll/10kx10k_enroll3k_usearch/warm';rows=[];last=0
mapping={'counterparty_id':'entity_id','counterparty_name':'legal_name','domicile':'country_code','lei_code':'lei'}
with (ROOT/'benchmarks/data/dataset_b_10k.csv').open() as f:records=[{mapping.get(k,k):v for k,v in r.items()} for r in list(csv.DictReader(f))[:1000]]
# Duplicate IDs exercise repeated enrollment in a bulk request; keep corpus identical.
records[49]=dict(records[0]); records[99]=dict(records[50])
(D/'records.json').write_text(json.dumps(records));shutil.copy2(ROOT/'benchmarks/data/dataset_a_10k.csv',D/'seed.csv')
def dump(): (D/'results.json').write_text(json.dumps(rows,indent=2))
def run(binary,case,trial,prepare=False):
 global last
 if last:time.sleep(max(0,120-(time.monotonic()-last)))
 tag=f'{binary}-{case}-{trial}'; job=D/tag;job.mkdir();config=template.replace(prefix,str(job)).replace('benchmarks/data/dataset_a_10k.csv',str(D/'seed.csv'));(job/'config.yaml').write_text(config)
 if not prepare:shutil.copytree(D/'prep/cache',job/'cache')
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();url=f'http://127.0.0.1:{port}'
 with (job/'server.log').open('w') as log:
  start=time.time();p=subprocess.Popen([str(D/binary),'enroll','--config',str(job/'config.yaml'),'--port',str(port)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  try:
   for i in range(1800):
    assert p.poll() is None,'server failed'
    try:
     with urllib.request.urlopen(url+'/api/v1/health',timeout=.5) as r:before=json.load(r)
     break
    except Exception:time.sleep(.1)
   else:raise TimeoutError('startup')
   if prepare:return
   def post(item):
    payload={'records':item} if case=='bulk50' else {'record':item};req=urllib.request.Request(url+'/api/v1/'+('enroll-batch' if case=='bulk50' else 'enroll'),data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});t=time.monotonic()
    with urllib.request.urlopen(req,timeout=120) as response:body=json.load(response);status=response.status
    assert status in (200,201) and 'error' not in body,(status,body)
    for record in (body.get('results',[]) if case=='bulk50' else [body]):assert record.get('enrolled') is True,record
    return {'latency_ms':(time.monotonic()-t)*1000,'body':body}
   t=time.monotonic()
   if case=='bulk50':answers=[post(records[i:i+50]) for i in range(0,len(records),50)]
   else:
    with cf.ThreadPoolExecutor(max_workers=10 if case=='c10' else 1) as executor:answers=list(executor.map(post,records))
   elapsed=time.monotonic()-t
   with urllib.request.urlopen(url+'/api/v1/health') as r:after=json.load(r)
   (job/'responses.json').write_text(json.dumps(answers));row=dict(binary=binary,case=case,trial=trial,seconds=elapsed,records_s=len(records)/elapsed,start=start,end=time.time(),health_before=before,health_after=after,job=tag);rows.append(row);dump();print(tag,round(row['records_s'],1),flush=True)
  finally:
   p.send_signal(signal.SIGTERM)
   try:p.wait(timeout=90)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   assert p.returncode==0,p.returncode
   last=time.monotonic()
try:
 prep=D/'prep';run('baseline','prep',0,True);shutil.copytree(D/'baseline-prep-0',prep)
 for case in ['c1','c10','bulk50']:run('baseline',case,0)
 print('waiting candidate',flush=True)
 while not (D/'candidate').exists():time.sleep(15)
 for trial in [1,2,3]:
  for case in ['c1','c10','bulk50']:
   for binary in (['candidate','baseline'] if trial%2 else ['baseline','candidate']):run(binary,case,trial)
 (D/'done').write_text('ok')
except BaseException:
 import traceback;traceback.print_exc();(D/'failed').write_text('failed');raise
