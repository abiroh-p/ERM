"""Recompute external Celebrity tables from immutable saved fits; no training/text required."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,math,statistics,zipfile
REPO=Path(__file__).resolve().parents[1]
MODELS=['tfidf','distilbert','roberta']
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def rebuild(archive,out):
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  read=lambda n:json.loads(z.read(n))
  c=read('configs/external_celebrity_v1.json');prep='results/external_celebrity/prepared_v1/';target=read(prep+'target_selection.json');index=read(prep+'selection_index.json')
  assert digest(target)==index['target_sha256']
  result_rows=[];deltas=[];summaries=[];conditions=[]
  for model in MODELS:
   base=f'runs/external_celebrity_v1/{model}/';signature=read(base+'run_protocol.json')
   for p,h in signature['code_sha256'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
   assert signature['config_sha256']==hashlib.sha256(z.read('configs/external_celebrity_v1.json')).hexdigest()
   assert signature['inputs_sha256']==hashlib.sha256(z.read(prep+'inputs.json')).hexdigest()
   assert signature['ner_provenance_sha256']==hashlib.sha256(z.read(prep+'ner_provenance.json')).hexdigest()
   assert signature['target_sha256']==digest(target)
   assert signature['source_manifests']==index['source_manifests']
   grid=[dict(index=i,seed=s,cond=k,source=d) for i,(s,k,d) in enumerate(itertools.product(c['seeds'],c['conditions'],c['sources']))]
   assert read(base+'jobs.json')==grid
   records={}
   for job in grid:
    r=read(base+f"jobs/job_{job['index']:04d}.json")
    assert r['job']==job and r['model']==model and r['test']=='Celebrity'
    assert digest({k:v for k,v in r.items() if k!='result_sha256'})==r['result_sha256']
    assert r['signature_sha256']==digest(signature)
    length=job['cond'] in {'L','LE','L_RE','SLE','L_RSE'}
    t=target['secondary' if length else 'primary'];assert digest(t['rows'])==t['sample_sha256']
    m=read(prep+f"source_seed_{job['seed']}.json")
    assert digest({k:v for k,v in m.items() if k!='manifest_sha256'})==m['manifest_sha256']==index['source_manifests'][str(job['seed'])]
    s=next(s for s in m['samples'] if s['dataset']==job['source'] and s['role']=='tr' and s['length_matched']==length)
    assert digest(s['rows'])==s['sample_sha256']==r['train_sample_sha256'] and len(s['rows'])==r['n_train']
    assert r['source_manifest_sha256']==m['manifest_sha256']
    assert r['target_sample_sha256']==t['sample_sha256'] and r['n_test']==len(t['rows'])
    assert len(r['target_word_deletions'])==r['n_test']
    assert all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k in ['auc','f1','acc','ap'])
    records[job['seed'],job['cond'],job['source']]=r
    result_rows.append(dict(model=model,**job,**r['metrics'],n_train=r['n_train'],n_test=r['n_test']))
   for scope,pairs in [('primary',c['primary_contrasts']),('secondary',c['secondary_contrasts'])]:
    for ca,cb in pairs:
     for source in c['sources']:
      values=[]
      for seed in c['seeds']:
       a,b=records[seed,ca,source],records[seed,cb,source]
       assert a['train_sample_sha256']==b['train_sample_sha256'] and a['target_sample_sha256']==b['target_sample_sha256']
       assert a['target_word_deletions']==b['target_word_deletions']
       value=a['metrics']['auc']-b['metrics']['auc'];values.append(value)
       deltas.append(dict(model=model,source=source,seed=seed,scope=scope,contrast=ca+'-'+cb,auc_difference=value))
      summaries.append(dict(model=model,source=source,scope=scope,contrast=ca+'-'+cb,n=len(values),mean=statistics.mean(values),sd=statistics.stdev(values),min=min(values),max=max(values),positive_seeds=sum(v>0 for v in values),negative_seeds=sum(v<0 for v in values)))
   for source,cond in itertools.product(c['sources'],c['conditions']):
    v=[records[s,cond,source]['metrics']['auc'] for s in c['seeds']]
    conditions.append(dict(model=model,source=source,condition=cond,n=len(v),mean=statistics.mean(v),sd=statistics.stdev(v),min=min(v),max=max(v)))
  out=Path(out);out.mkdir(parents=True,exist_ok=True)
  for name,data in [('metrics.csv',result_rows),('per_seed_contrasts.csv',deltas),('contrast_summary.csv',summaries),('condition_summary.csv',conditions)]:
   with open(out/name,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
  report=dict(status='PASS',fits_per_model={m:sum(r['model']==m for r in result_rows) for m in MODELS},fits=len(result_rows),contrasts=len(deltas),summary_rows=len(summaries),source_condition_summaries=len(conditions),input_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),target_sizes={k:len(target[k]['rows']) for k in ['primary','secondary']},limits='Recomputes saved-output aggregation, not training, raw-source overlap, NER extraction or metrics from prediction scores. No new inferential tests or source pooling.')
  (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report,indent=2))
  return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-zip',type=Path,default=REPO/'results/external_celebrity/accounts/ERM_Celebrity_Results.zip');p.add_argument('--out',type=Path,default=REPO/'results/external_celebrity/completed_v1');a=p.parse_args();rebuild(a.results_zip,a.out)
