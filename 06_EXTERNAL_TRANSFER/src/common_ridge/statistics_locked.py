"""Locked scalar-response statistics; no model, Gaussian or feature operations."""

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
import numpy as np
import pandas as pd
def scores(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p);a=y[ok];b=p[ok]
 den=float(np.sum((a-a.mean())**2)) if len(a) else 0.
 return dict(planned_n=len(y),covered_n=int(ok.sum()),coverage=float(ok.mean()),
  MAE_eV=float(np.mean(abs(a-b))) if len(a) else np.nan,RMSE_eV=float(np.sqrt(np.mean((a-b)**2))) if len(a) else np.nan,
  R2=float(1-np.sum((a-b)**2)/den) if len(a)>1 and den>1e-20 else np.nan,
  signed_error_eV=float(np.mean(b-a)) if len(a) else np.nan,response_mean_abs_eV=float(np.mean(abs(y))),
  normalized_MAE=float(np.mean(abs(a-b))/np.mean(abs(y))) if len(a) and np.mean(abs(y))>1e-12 else np.nan)
def analyze(results,IDs,repeats=10000,seed=20261025):
 """Exactly ten planned fits per parent/group/scope. Any failure invalidates confirmatory CI."""
 IDs=sorted(IDs);out=[];parents=[];rng=np.random.default_rng(seed);ix=rng.integers(0,len(IDs),size=(repeats,len(IDs)))
 for scope in ['all','geometry','chemical']:
  rr=results[results.scope==scope]
  for ID in IDs:
   for arm in ['A','B']:
    a=rr[(rr.ID==ID)&(rr.group==arm)];valid=len(a)==10 and a.status.eq('ok').all() and a.coverage.eq(1).all()
    parents.append(dict(ID=ID,group=arm,scope=scope,planned_fit_n=10,observed_fit_n=len(a),successful_fit_n=int(a.status.eq('ok').sum()),
     complete=valid,MAE_eV=float(a.MAE_eV.mean()) if valid else np.nan,RMSE_eV=float(a.RMSE_eV.mean()) if valid else np.nan))
  pp=pd.DataFrame(parents);a=pp[(pp.scope==scope)&(pp.group=='A')].set_index('ID').loc[IDs];b=pp[(pp.scope==scope)&(pp.group=='B')].set_index('ID').loc[IDs]
  complete=a.complete.all() and b.complete.all();row=dict(scope=scope,primary=scope=='all',planned_parents=len(IDs),complete_parents=int((a.complete&b.complete).sum()),
   status='COMPLETE_FIXED_PROTOCOL' if complete else 'INCOMPLETE_PRIMARY_NO_CONFIRMATORY_CI',
   A_MAE_eV=np.nan,B_MAE_eV=np.nan,B_minus_A_MAE_eV=np.nan,CI95_low=np.nan,CI95_high=np.nan,relative_reduction=np.nan,relative_CI95_low=np.nan,relative_CI95_high=np.nan,
   parents_favor_B=np.nan,parents_favor_A=np.nan,ties=np.nan,unit='mother;2 generator seeds and5 subset repeats nested',
   multiplicity='one prespecified primary95%; others exploratory unadjusted95%')
  if complete:
   aa=a.MAE_eV.to_numpy();bb=b.MAE_eV.to_numpy();d=bb-aa;bs=d[ix].mean(axis=1)
   rel=(aa[ix].mean(axis=1)-bb[ix].mean(axis=1))/aa[ix].mean(axis=1) if aa.mean()>1e-12 else np.full(repeats,np.nan)
   row.update(A_MAE_eV=aa.mean(),B_MAE_eV=bb.mean(),B_minus_A_MAE_eV=d.mean(),CI95_low=np.quantile(bs,.025),CI95_high=np.quantile(bs,.975),
    relative_reduction=1-bb.mean()/aa.mean() if aa.mean()>1e-12 else np.nan,relative_CI95_low=np.quantile(rel,.025),relative_CI95_high=np.quantile(rel,.975),
    parents_favor_B=int((d<0).sum()),parents_favor_A=int((d>0).sum()),ties=int((d==0).sum()))
  out.append(row)
 return pd.DataFrame(parents),pd.DataFrame(out)
