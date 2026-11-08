"""Lazy real callbacks. NOT invoked by offline tests. Caller must use run_approved."""

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
def factory():
 from pathlib import Path
 import json,sys,hashlib
 import numpy as np,pandas as pd,joblib
 P=_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge')
 cfg=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/EXPERIMENT_CONFIG_LOCKED.json')).read_text());F=cfg['features']
 sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/common_ridge')));from refined import weighted_local
 sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/common_ridge')));from statistics_locked import scores,analyze
 scale=np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/training_SD.npy'),allow_pickle=False)
 with np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/EXACT_ANCHORS38.npz'),allow_pickle=False) as a:anchors={str(k):v.copy() for k,v in zip(a['IDs'],a['x0'])}
 cases=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/CASE_INPUTS76.json')).read_text());plans=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/SELECTED_FIT_INDICES760.csv')).to_dict('records')
 here=_repo_path('06_EXTERNAL_TRANSFER/workflows/sampling_control');sys.path.insert(0,str(here))
 import candidate_store
 if Path(candidate_store.__file__).resolve()!=_repo_path('06_EXTERNAL_TRANSFER/src/sampling_control/candidate_store.py'):raise RuntimeError('Wrong actual candidate loader module')
 manifest=_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/GAUSSIAN_CANDIDATE_SHA256.csv')
 # Constant receipt is inserted before freezing this new source version.
 store=candidate_store.FrozenCandidateStore(manifest,'47f5db6ff9c673065326893cf2f2fb990fc348c0992cfb5400a6310095fafab5',anchors)
 for p in plans:p['fit_indices']=json.loads(p['fit_indices'])
 model=joblib.load(cfg['ET_model_path'])
 if model.n_features_in_!=9:raise RuntimeError('Model architecture mismatch')
 class Ops:
  def load_candidates(self,ID,x0):return store.load(ID,x0)
  def pool(self,c):
   d=pd.read_csv(c['pool_path']);events=json.loads(Path(c['events_path']).read_text())
   return d[F].to_numpy(),d.delta_frozen_ET.to_numpy(),np.array(['geometry' if e['kind']=='strain_and_internal_displacement' else 'chemical' for e in events])
  def oracle(self,rows):return model.predict(np.array([np.frombuffer(b,dtype=t) for b,t in rows]))
  def fit(self,x,y):
   m=weighted_local(x,y,1,2.25,False);return m['coefficient'],m['intercept']
  def statistics(self,rows,preds,IDs):
   z=pd.DataFrame(preds);out=[]
   for key,d in z.groupby(['ID','seed','repeat','group']):
    c=next(c for c in cases if c['ID']==key[0] and c['seed']==key[1]);_,_,kind=self.pool(c)
    for scope in ['all','geometry','chemical']:
     w=d if scope=='all' else d[np.array([kind[i]==scope for i in d.candidate])]
     out.append(dict(ID=key[0],seed=key[1],repeat=key[2],group=key[3],scope=scope,status='ok' if w.status.eq('ok').all() else 'failed',**scores(w.truth.to_numpy(),w.prediction.to_numpy())))
   parent,stats=analyze(pd.DataFrame(out),IDs,10000,20261025)
   return dict(parents=parent.to_dict('records'),statistics=stats.to_dict('records'),integrity='COMPLETE' if all(r['status']=='ok' for r in rows) else 'INCOMPLETE',effect_is_not_integrity_gate=True,reference_10percent='secondary descriptive, not required to pass experiment')
 return cfg,plans,cases,anchors,scale,Ops()
