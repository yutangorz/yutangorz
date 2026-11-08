"""CRISP-X v0.5 audit candidate; exact origin for geometry, no causal claim."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
import numpy as np
from crispx.path_model import path_key

def validate(z,y,events):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=9 or not np.isfinite(z).all():raise ValueError('Expected finite n x9 standardized concepts')
    if len(events)!=len(z):raise ValueError('Event/coordinate count mismatch')
    for e in events:
        if e.get('kind') not in ['strain_and_internal_displacement','bounded_same_block_species_substitution']:raise ValueError('Unknown event kind')
        if e['kind']!='strain_and_internal_displacement' and (not e.get('old') or not e.get('new')):raise ValueError('Chemical event lacks species')
    if y is None:return z,None
    y=np.asarray(y,float)
    if y.ndim!=1 or len(y)!=len(z) or not np.isfinite(y).all():raise ValueError('Malformed response vector')
    return z,y

def fit_audited(z,y,events,ridge=.001):
    z,y=validate(z,y,events)
    if not np.isfinite(ridge) or ridge<=0:raise ValueError('Positive ridge required')
    result={}
    for key in sorted(set(path_key(e) for e in events)):
        ii=np.array([j for j,e in enumerate(events) if path_key(e)==key]);x=z[ii,7:];target=y[ii]
        if key=='geometry_only':
            center=np.zeros(2);intercept=0.;a=x;b=target
        else:
            center=x.mean(axis=0);intercept=float(target.mean());a=x-center;b=target-intercept
        coef=np.linalg.solve(a.T@a+ridge*np.eye(2),a.T@b);singular=np.linalg.svd(a,compute_uv=False);threshold=max(1e-10,singular[0]*.03) if len(singular) else 1e-10
        result[key]=dict(center=center,mean=intercept,coefficient=coef,n=len(ii),geometry_rank=int((singular>threshold).sum()),train_path_mean=float(target.mean()),geometry_bounds_min=x.min(axis=0),geometry_bounds_max=x.max(axis=0),joint_jump_at_unperturbed_geometry=float(intercept-center@coef))
    return result

def predict_audited(model,z,events):
    z,_=validate(z,None,events);p=np.full(len(z),np.nan);mean=np.full(len(z),np.nan);outside=np.full(len(z),False)
    for j,e in enumerate(events):
        key=path_key(e)
        if key in model:
            m=model[key];p[j]=m['mean']+(z[j,7:]-m['center'])@m['coefficient'];mean[j]=m['train_path_mean'];outside[j]=bool(np.any(z[j,7:]<m['geometry_bounds_min']-1e-12) or np.any(z[j,7:]>m['geometry_bounds_max']+1e-12))
    return p,np.isfinite(p),mean,outside

def svd_audit(a,y,cutoff=.03,ridge=1e-6):
    a,y=np.asarray(a,float),np.asarray(y,float)
    if a.ndim!=2 or y.ndim!=1 or len(a)!=len(y) or not np.isfinite(a).all() or not np.isfinite(y).all():raise ValueError('Bad SVD inputs')
    if len(a)==0:return dict(coefficient=np.zeros(a.shape[1]),support=np.zeros(a.shape[1]),rank=0)
    u,s,vt=np.linalg.svd(a,full_matrices=False);keep=s>max(1e-10,s[0]*cutoff);v=vt[keep].T
    return dict(coefficient=v@((s[keep]/(s[keep]**2+ridge))*(u[:,keep].T@y)),support=np.clip(np.diag(v@v.T),0.,1.),rank=int(keep.sum()))
