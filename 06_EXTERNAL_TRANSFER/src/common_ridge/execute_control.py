"""Phase C runner, NOT executed by A/B preparation. Refuses draft/absent author authorization."""

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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/common_ridge')))
from pathlib import Path
import argparse,json,hashlib,datetime,os,sys,traceback,time
sys.dont_write_bytecode=True
ROOT=_repo_path('.')
BUNDLE=_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def require_approval(record,protocol_sha,manifest_sha):
 if record.get('authority_type')!='explicit_author_user_confirmation':raise PermissionError('Explicit author confirmation is required')
 if not record.get('protocol_approved') or not record.get('experiment_execution_approved'):raise PermissionError('Draft/principle agreement is not execution approval')
 for k in ['author','approval_time','user_message_reference','authorization_nonce']:
  if not record.get(k):raise PermissionError('Missing approval field: '+k)
 if record.get('protocol_sha256')!=protocol_sha or record.get('immutable_manifest_sha256')!=manifest_sha:raise PermissionError('Approval does not bind frozen protocol and inputs')
 if record.get('ET_rows_limit')!=19456 or record.get('ET_batch_limit')!=38 or record.get('Ridge_fit_limit')!=760:raise PermissionError('Budget approval mismatch')
 if record.get('feature_extractions_remaining')!=0:raise PermissionError('38 anchors already prepared; no feature calls remain')
 return True
def execute(approval_path,output_dir):
 # Guard is deliberately before Gaussian imports, model load, directory creation or budget use.
 protocol=_repo_path('CRISP_X_TRANSFER_FINAL_PROTOCOL_v1.md');manifest=_repo_path('06_EXTERNAL_TRANSFER/configs/common_ridge_source_identity_immutable_experiment_manifest.json')
 approval=json.loads(Path(approval_path).read_text(encoding='utf-8'))
 require_approval(approval,sha(protocol),sha(manifest))
 for x in json.loads(manifest.read_text(encoding='utf-8'))['references']:
  if sha(x['path'])!=x['sha256']:raise RuntimeError('Frozen input changed: '+x['path'])
 out=Path(output_dir).resolve();allowed=_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge')
 if not out.is_relative_to(allowed) or not out.name.startswith('run_'):raise RuntimeError('Use a new numbered run under the reserved phase-C stage')
 if out.exists():raise RuntimeError('Output exists; no overwrite/resume/retry with this script')
 # Nonce is single-use globally, even when another output name is supplied.
 nonce=hashlib.sha256(approval['authorization_nonce'].encode()).hexdigest();ad=_repo_path('PROJECT_MEMORY/AUTHORIZATIONS');ad.mkdir(exist_ok=True)
 spent=ad/(nonce+'.spent.json')
 with open(spent,'x',encoding='utf-8') as f:json.dump(dict(output=str(out),approval=str(approval_path),time=datetime.datetime.now().isoformat()),f)
 out.mkdir(parents=True);(out/'Gaussian_caches').mkdir()
 config=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/EXPERIMENT_CONFIG_LOCKED.json')).read_text(encoding='utf-8'))
 ledger=dict(ET_rows_reserved=0,ET_batches_reserved=0,Ridge_fits_reserved=0,ET_rows_successful=0,ET_batches_successful=0,Ridge_fits_successful=0,
  feature_extractions_prepared=38,feature_extractions_this_run=0,new_structures=0,status='STARTED',approval_nonce_hash=nonce)
 def event(kind,data):
  with open(out/'ATTEMPTS.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,time=datetime.datetime.now().isoformat(),**data),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 def reserve(kind,ID,extra=None):
  if kind=='ET':
   if ledger['ET_rows_reserved']+512>19456 or ledger['ET_batches_reserved']+1>38:raise RuntimeError('ET budget exhausted; no retry')
   ledger['ET_rows_reserved']+=512;ledger['ET_batches_reserved']+=1
  else:
   if ledger['Ridge_fits_reserved']+1>760:raise RuntimeError('Local fit budget exhausted')
   ledger['Ridge_fits_reserved']+=1
  save(out/'QUERY_BUDGET_LEDGER.json',ledger);event('RESERVED_'+kind,dict(ID=ID,**(extra or {})))
 save(out/'QUERY_BUDGET_LEDGER.json',ledger)
 try:
  import importlib.metadata
  versions=json.loads((_repo_path('06_EXTERNAL_TRANSFER/configs/common_ridge_environment.json')).read_text(encoding='utf-8'))
  for package,version in versions.items():
   if importlib.metadata.version(package)!=version:raise RuntimeError('Environment drift: '+package)
  import numpy as np,pandas as pd,joblib
  from lime.lime_tabular import LimeTabularExplainer
  sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/common_ridge')));from refined import weighted_local
  sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/common_ridge')));from statistics_locked import scores,analyze
  cfg=config;F=cfg['features'];training=np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/training5714.npy'),allow_pickle=False);scale=np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/training_SD.npy'),allow_pickle=False)
  with np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/EXACT_ANCHORS38.npz'),allow_pickle=False) as a:anchors={str(ID):v.copy() for ID,v in zip(a['IDs'],a['x0'])}
  cases=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/CASE_INPUTS76.json')).read_text(encoding='utf-8'));plans=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/SELECTED_FIT_INDICES760.csv'),dtype={'ID':str})
  # Copies planned denominators before any model query; even interrupted runs retain all38/76/760.
  plans.to_csv(out/'ALL_PLANNED_FITS760.csv',index=False)
  model=joblib.load(cfg['ET_model_path']);assert model.n_features_in_==9
  rows=[];predictions=[];coefficients=[];support=[];start=time.perf_counter()
  for ID in sorted(anchors):
   x0=anchors[ID];seed=int(hashlib.sha256(('run022|'+ID).encode()).hexdigest()[:8],16)%(2**31-1)
   # Official frozen sampler only, not explain_instance (which would add unbudgeted local fits).
   ex=LimeTabularExplainer(training,feature_names=F,mode='regression',discretize_continuous=False,sample_around_instance=True,
    feature_selection='none',kernel_width=2.25,random_state=seed)
   _,gx=ex._LimeTabularExplainer__data_inverse(x0,512)
   assert gx.shape==(512,9) and np.array_equal(gx[0],x0) and np.isfinite(gx).all()
   np.save(out/'Gaussian_caches'/f'{ID}_features_before_query.npy',gx)
   reserve('ET',ID);tick=time.perf_counter()
   gp=np.asarray(model.predict(gx),float)
   assert gp.shape==(512,) and np.isfinite(gp).all()
   ledger['ET_rows_successful']+=512;ledger['ET_batches_successful']+=1
   event('ET_COMPLETED',dict(ID=ID,rows=512,seconds=time.perf_counter()-tick));save(out/'QUERY_BUDGET_LEDGER.json',ledger)
   np.savez_compressed(out/'Gaussian_caches'/f'{ID}.npz',features=gx,ET_prediction=gp,x0=x0,Gaussian_seed=seed,training_SD=scale)
   gz=(gx-x0)/scale;gy=gp-gp[0]
   for c in [v for v in cases if v['ID']==ID]:
    ss=c['seed'];pool=pd.read_csv(c['pool_path']);x=pool[F].to_numpy();z=(x-x0)/scale;y=pool.delta_frozen_ET.to_numpy();test=np.array(c['test'])
    events=json.loads(Path(c['events_path']).read_text(encoding='utf-8'));kind=np.array(['geometry' if events[int(j)]['kind']=='strain_and_internal_displacement' else 'chemical' for j in test])
    # Stop on exact descriptor leakage; don't alter the heldout set or drop offending rows.
    assert not any(np.any(np.max(abs(gx[1:]-x[j]),axis=1)<1e-12) for j in test)
    for q in plans[(plans.ID==ID)&(plans.seed==ss)].itertuples():
     ii=np.array(json.loads(q.fit_indices));xx,yy=(gz[ii],gy[ii]) if q.group=='A' else (z[ii],y[ii]);reserve('RIDGE',ID,dict(seed=ss,repeat=q.repeat,group=q.group));tick=time.perf_counter()
     try:
      m=weighted_local(xx,yy,1,2.25,False);pp=z[test]@m['coefficient']+m['intercept'];assert np.isfinite(pp).all()
      ledger['Ridge_fits_successful']+=1;status='ok';err=''
      w=np.maximum(np.exp(-np.sum(xx**2,axis=1)/(2*2.25**2)),1e-12);mu=np.average(xx,axis=0,weights=w);my=np.average(yy,weights=w);center=xx-mu
      # Residual check only; do not solve/refit a second local model.
      residual=center.T@(w*(yy-my-center@m['coefficient']))-m['coefficient']
      assert np.linalg.norm(residual)<1e-8*(1+np.linalg.norm(center.T@(w*(yy-my))))
      for f,beta in zip(F,m['coefficient']):coefficients.append(dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,feature=f,beta_eV_per_SD=beta,intercept=m['intercept']))
      dist=np.linalg.norm(xx,axis=1);support.append(dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,fit_radius_mean=dist.mean(),fit_radius_median=np.median(dist),fit_radius_max=dist.max(),
       test_radius_mean=np.linalg.norm(z[test],axis=1).mean(),nearest_fit_distance=np.linalg.norm(z[test,None,:]-xx[None,:,:],axis=2).min(axis=1).mean(),response_mean_abs=np.mean(abs(yy)),
       fit_rank=m['fit_rank'],ESS=m['effective_weight_n'],condition_number=np.linalg.cond(xx*np.sqrt(w[:,None]))))
     except Exception:
      status='FAILED_LOCAL_FIT_OR_QA';err=traceback.format_exc();pp=np.full(len(test),np.nan);event('RIDGE_FAILED',dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,error=err))
     event('RIDGE_FINISHED',dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,status=status,seconds=time.perf_counter()-tick));save(out/'QUERY_BUDGET_LEDGER.json',ledger)
     for scope,mask in [('all',np.ones(len(test),bool)),('geometry',kind=='geometry'),('chemical',kind=='chemical')]:
      rows.append(dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,scope=scope,status=status,error=err,**scores(y[test][mask],pp[mask])))
     for j,n in enumerate(test):predictions.append(dict(ID=ID,seed=ss,repeat=q.repeat,group=q.group,candidate=int(n),kind=kind[j],truth=y[n],prediction=pp[j],status=status))
   pd.DataFrame(rows).to_csv(out/'RESULTS_PER_FIT.csv',index=False);pd.DataFrame(predictions).to_csv(out/'PREDICTIONS.csv',index=False)
  result=pd.DataFrame(rows);parent,stats=analyze(result,list(anchors),10000,20261025);parent.to_csv(out/'PARENT_RESULTS.csv',index=False);stats.to_csv(out/'PAIRED_STATISTICS.csv',index=False)
  pd.DataFrame(coefficients).to_csv(out/'COEFFICIENTS.csv',index=False);pd.DataFrame(support).to_csv(out/'SUPPORT_AND_DESIGN.csv',index=False)
  assert ledger['ET_rows_reserved']==19456 and ledger['ET_batches_reserved']==38 and ledger['Ridge_fits_reserved']==760
  ledger['status']='CALCULATION_FINISHED_TECHNICAL_QA_PENDING';ledger['execution_seconds']=time.perf_counter()-start;ledger['scientific_closeout_approved']=False
  save(out/'QUERY_BUDGET_LEDGER.json',ledger)
 except Exception:
  ledger['status']='FAILED_OR_INTERRUPTED_NO_AUTOMATIC_RETRY';save(out/'QUERY_BUDGET_LEDGER.json',ledger)
  save(out/'FAILURE.json',dict(traceback=traceback.format_exc(),planned_mothers=38,planned_fits=760,remaining_cases_not_removed=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--approval-record',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();execute(a.approval_record,a.output_dir)
