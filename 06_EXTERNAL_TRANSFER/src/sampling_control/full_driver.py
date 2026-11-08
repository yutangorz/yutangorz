"""Integrated selected-union driver. Import has no scientific side effects.
Production requires fresh SHA-bound author approval; mock never invokes real science.
"""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/sampling_control')))
from pathlib import Path
import json,hashlib,time,os,platform,importlib.util,sys
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=True),encoding='utf8')
def load_ledger_module():
 p=_repo_path('06_EXTERNAL_TRANSFER/workflows/execution_control/executor.py')
 spec=importlib.util.spec_from_file_location('prior_ledger',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def mother_aggregate(rows):
 """Descriptive aggregation only: no bootstrap; fail/missing never removed."""
 import numpy as np
 out=[]
 for ID in sorted({r['ID'] for r in rows}):
  vals={arm:[r for r in rows if r['ID']==ID and r['group']==arm] for arm in ['A','B']}
  complete=all(len(v)==10 and all(r['status']=='ok' and r['covered_n']==48 for r in v) for v in vals.values())
  a=float(np.mean([r['MAE_eV'] for r in vals['A']])) if complete else None
  b=float(np.mean([r['MAE_eV'] for r in vals['B']])) if complete else None
  out.append(dict(ID=ID,complete=complete,A=a,B=b,B_minus_A=b-a if complete else None))
 return out
def resources():
 d=dict(wall=time.perf_counter(),cpu=time.process_time(),pid=os.getpid(),platform=platform.platform(),threads=os.cpu_count())
 try:
  import psutil
  p=psutil.Process();d.update(rss_bytes=p.memory_info().rss,process_threads=p.num_threads())
 except ImportError:d['rss_bytes']=None
 return d
def integrate(output,cfg,plans,cases,anchors,scale,ops,limits,cache_receipts=None):
 """Internal orchestration. Production entry is run_approved; tests inject fixed stubs."""
 import numpy as np
 output=Path(output);output.mkdir(exist_ok=False);lm=load_ledger_module()
 ledger=lm.Ledger(output/'ledger.json',cfg['ET_model_sha256'],cfg['features'],limits)
 if cache_receipts:
  for key,value,receipt in cache_receipts:ledger.cache_import(key,value,receipt)
 start=resources();rows=[];preds=[]
 dump(output/'planned.json',dict(parents=sorted(anchors),fit_plans=plans,cases=[dict(ID=c['ID'],seed=c['seed'],test=c['test']) for c in cases]))
 def cost(kind,before,status):
  end=resources();entry=dict(kind=kind,status=status,before=before,after=end,wall_seconds=end['wall']-before['wall'],cpu_seconds=end['cpu']-before['cpu'])
  with (output/'resources.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
 for ID in sorted(anchors):
  x0=anchors[ID];myplans=[p for p in plans if p['ID']==ID];mycases=[c for c in cases if c['ID']==ID]
  before=resources()
  gx=ops.load_candidates(ID,x0)
  ledger.state['candidate_rows_reused_from_preflight']=ledger.state.get('candidate_rows_reused_from_preflight',0)+len(gx);ledger.save()
  assert gx.shape==(512,9) and np.isfinite(gx).all() and np.array_equal(gx[0],x0),'Invalid Gaussian/anchor'
  pools={}
  for c in mycases:
   x,y,kinds=ops.pool(c);test=np.array(c['test']);pools[c['seed']]=(x,y,kinds)
   assert np.isfinite(x).all() and np.isfinite(y).all()
   # Check ALL nonanchor candidates, not only the queried union.
   for t in x[test]:
    if np.any(np.all(gx==t,axis=1)) or np.any(np.max(abs(gx-t),axis=1)<1e-12):raise RuntimeError('Gaussian-test collision; no filtering/resampling')
  cost('load_and_recheck_frozen_pool',before,'PASS')
  union=sorted({int(i) for p in myplans if p['group']=='A' for i in p['fit_indices']});selected=[0]+union
  request=[(np.asarray(gx[i],dtype='<f8',order='C').tobytes(),'<f8') for i in selected]
  tick=resources()
  try:values=ledger.query(request,ops.oracle)
  except BaseException:cost('response_request',tick,'FAILED');raise
  cost('response_request',tick,'SUCCESS');lookup=dict(zip(selected,values));f0=lookup[0]
  np.savez_compressed(output/(ID+'_selected.npz'),indices=selected,features=gx[selected],absolute_predictions=values,anchor=x0)
  for p in myplans:
   c=next(c for c in mycases if c['seed']==p['seed']);x,y,kinds=pools[p['seed']];test=np.array(c['test']);ii=np.array(p['fit_indices'])
   assert len(ii)==64 and len(set(ii))==64
   assert (set(ii)<=set(range(1,512))) if p['group']=='A' else (set(ii)<=set(c['fit']) and not set(ii)&set(test))
   xx=(gx[ii]-x0)/scale if p['group']=='A' else (x[ii]-x0)/scale
   yy=np.array([lookup[int(i)]-f0 for i in ii]) if p['group']=='A' else y[ii]
   ledger.reserve('fit_'+p['group'],1);ledger.save();tick=resources()
   error=None
   try:
    coef,intercept=ops.fit(xx,yy);pp=((x[test]-x0)/scale)@coef+intercept
    if pp.shape!=(48,) or not np.isfinite(pp).all():raise ValueError('Invalid prediction')
    status='ok'
   except Exception as e:pp=np.full(48,np.nan);status='failed';error=repr(e)
   cost('fit_'+p['group'],tick,status)
   err=abs(pp-y[test]);rows.append(dict(ID=ID,seed=p['seed'],repeat=p['repeat'],group=p['group'],status=status,error=error,covered_n=int(np.isfinite(pp).sum()),MAE_eV=float(err.mean()) if status=='ok' else None))
   for index,truth,pred in zip(test,y[test],pp):preds.append(dict(ID=ID,seed=p['seed'],repeat=p['repeat'],group=p['group'],candidate=int(index),truth=float(truth),prediction=float(pred),status=status))
   dump(output/'fit_results.json',rows)
 parents=mother_aggregate(rows);dump(output/'parents.json',parents);dump(output/'predictions.json',preds)
 # Real analysis callback is only reached by approved production; mock callback never bootstraps.
 stats=ops.statistics(rows,preds,sorted(anchors));dump(output/'statistics.json',stats)
 end=resources();dump(output/'global_resources.json',dict(start=start,end=end,wall_seconds=end['wall']-start['wall'],cpu_seconds=end['cpu']-start['cpu'],ledger_reserved=ledger.state))
 return rows,parents,ledger.state
def run_approved(approval,bindings,output,ops_factory,limits):
 """No callbacks/scientific imports before author+current SHA+environment gate."""
 required=['protocol_approved','statistics_approved','data_manifest_approved','budget_approved','secondary_10percent_rule_approved','batch03_execution_approved']
 if approval.get('authority')!='explicit_human_author' or not all(approval.get(x) is True for x in required):raise PermissionError('No real execution authorization')
 if not all(approval.get(x) for x in ['author','time','message_reference','nonce']):raise PermissionError('Missing author receipt')
 required_bindings={'protocol_sha256','statistics_sha256','data_manifest_sha256','budget_sha256','driver_sha256','factory_sha256','ledger_sha256','candidate_manifest_sha256','candidate_loader_sha256'}
 if not required_bindings<=set(bindings):raise PermissionError('Missing required frozen bindings')
 if Path(bindings['driver_sha256']).resolve()!=_repo_path('06_EXTERNAL_TRANSFER/src/sampling_control/full_driver.py'):raise PermissionError('Wrong driver identity')
 if Path(bindings['factory_sha256']).resolve()!=Path(sys.modules[ops_factory.__module__].__file__).resolve():raise PermissionError('Wrong factory identity')
 if Path(bindings['ledger_sha256']).resolve()!=_repo_path('06_EXTERNAL_TRANSFER/workflows/execution_control/executor.py'):raise PermissionError('Wrong actual ledger path')
 if Path(bindings['candidate_loader_sha256']).resolve()!=_repo_path('06_EXTERNAL_TRANSFER/src/sampling_control/full_driver.py').with_name('candidate_store.py').resolve():raise PermissionError('Wrong actual candidate loader path')
 for k,p in bindings.items():
  if approval.get(k)!=sha(p):raise PermissionError('Binding SHA mismatch')
 if approval.get('limits')!=limits:raise PermissionError('Budget mismatch')
 # Validate the entire actual-input manifest immediately before initialization.
 import csv,importlib.metadata as meta
 with open(bindings['data_manifest_sha256'],encoding='utf-8-sig',newline='') as f:
  for x in csv.DictReader(f):
   if sha(x['path'])!=x['sha256']:raise RuntimeError('Actual input SHA drift')
 for k,v in approval['environment'].items():
  if meta.version(k)!=v:raise RuntimeError('Environment drift')
 nonce=Path(output).parent/('nonce_'+hashlib.sha256(approval['nonce'].encode()).hexdigest()+'.spent')
 with nonce.open('x') as f:f.write(json.dumps(approval))
 cfg,plans,cases,anchors,scale,ops=ops_factory()
 return integrate(output,cfg,plans,cases,anchors,scale,ops,limits)
