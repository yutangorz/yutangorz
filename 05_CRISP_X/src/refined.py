"""Restricted local regression and identifiable chemical contrasts.

Independent descriptor-property model explanations, never causal/ALIGNN labels.
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
import numpy as np
from sklearn.linear_model import Ridge

def weighted_local(z,y,alpha,bandwidth,anchor,columns=None):
    z,y=np.asarray(z,float),np.asarray(y,float)
    cols=np.arange(z.shape[1]) if columns is None else np.array(columns,int)
    w=np.exp(-np.sum(z[:,cols]**2,axis=1)/(2*bandwidth**2));w=np.maximum(w,1e-12)
    m=Ridge(alpha=alpha,fit_intercept=not anchor).fit(z[:,cols],y,sample_weight=w)
    coef=np.zeros(z.shape[1]);coef[cols]=m.coef_
    return dict(coefficient=coef,intercept=float(m.intercept_),effective_weight_n=float(w.sum()**2/(w@w)),fit_rank=int(np.linalg.matrix_rank(z[:,cols]*np.sqrt(w[:,None]))))

def supported_svd(a,y,cutoff=.03,ridge=1e-6):
    a,y=np.asarray(a,float),np.asarray(y,float)
    if a.ndim!=2 or len(a)==0:return dict(coefficient=np.zeros(7),support=np.zeros(7),rank=0,singular_values=np.array([]),relative_residual=None)
    u,s,vt=np.linalg.svd(a,full_matrices=False);keep=s>max(1e-10,s[0]*cutoff)
    v=vt[keep].T;coef=v@((s[keep]/(s[keep]**2+ridge))*(u[:,keep].T@y));projection=v@v.T
    return dict(coefficient=coef,support=np.diag(projection),rank=int(keep.sum()),singular_values=s,relative_residual=float(np.linalg.norm(y-a@coef)/max(np.linalg.norm(y),1e-12)))

def chemical_contrasts(z,y,events,path_model,seed,bootstrap=200):
    # One coupled elemental contrast per chemical path. Geometry adjusted using
    # that path's fitted slope; a path jump remains a finite joint intervention.
    keys=sorted(k for k in path_model if k!='geometry_only');a=[];jumps=[];counts=[]
    for key in keys:
        old,new=key.split('->');idx=[j for j,e in enumerate(events) if e.get('old')==old and e.get('new')==new]
        a.append(np.mean(z[idx,:7],axis=0));jumps.append(path_model[key]['joint_jump_at_unperturbed_geometry']);counts.append(len(idx))
    a=np.array(a).reshape(-1,7);jumps=np.array(jumps)
    fit=supported_svd(a,jumps);rng=np.random.default_rng(seed);betas=[];supports=[]
    for _ in range(bootstrap):
        ii=rng.integers(0,len(a),len(a)) if len(a) else np.array([],int)
        b=supported_svd(a[ii],jumps[ii]);betas.append(b['coefficient']);supports.append(b['support'])
    betas=np.array(betas);supports=np.array(supports);low,high=np.quantile(betas,[.025,.975],axis=0)
    sign_stability=np.maximum((betas>0).mean(axis=0),(betas<0).mean(axis=0))
    # Hold out whole chemical paths; do not evaluate the same fitted paths only.
    loo=[];loo_support=[]
    for j in range(len(a)):
        mask=np.arange(len(a))!=j;b=supported_svd(a[mask],jumps[mask]);loo.append(float(a[j]@b['coefficient']));loo_support.append(b['rank'])
    loo=np.array(loo);rmse=float(np.sqrt(np.mean((jumps-loo)**2))) if len(a) else None
    scale=float(np.sqrt(np.mean(jumps**2))) if len(a) else None
    r2=float(1-np.sum((jumps-loo)**2)/np.sum((jumps-jumps.mean())**2)) if len(a)>1 and np.var(jumps)>1e-20 else None
    # Numerical support alone is insufficient: also require stable direction and
    # meaningful whole-path transfer. These are operational, noncausal gates.
    transfer=(r2 is not None and r2>=.8 and rmse<=.05)
    qualified=(fit['support']>=.95)&((supports>=.95).mean(axis=0)>=.8)&(sign_stability>=.9)&((low>0)|(high<0))&transfer
    return dict(keys=keys,atomic_design=a,jumps=jumps,counts=counts,fit=fit,bootstrap_coefficients=betas,bootstrap_support=supports,ci_low=low,ci_high=high,sign_stability=sign_stability,loo_prediction=loo,loo_rmse=rmse,loo_r2=r2,transfer_passed=bool(transfer),qualified=qualified)
