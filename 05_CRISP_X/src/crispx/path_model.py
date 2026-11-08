"""Separate finite chemical paths from local geometry responses.

A chemical jump changes coupled elemental concepts together. It is reported
as a joint path response; fitting it does not identify seven independent causes.
Unknown paths trigger abstention instead of silent chemical extrapolation.
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
import numpy as np
from crispx.core import informative_subset


def path_key(event):
    if event['kind']=='strain_and_internal_displacement':
        return 'geometry_only'
    return event['old']+'->'+event['new']


def path_balanced_subset(z, events, budget):
    """Cover known chemical paths before D-optimal geometric diversification."""
    groups={}
    for j,e in enumerate(events):
        groups.setdefault(path_key(e),[]).append(j)
    # Greedy concept information order, not response magnitude.
    order=informative_subset(z,len(z)).tolist()
    first=[]
    seen=set()
    for j in order:
        key=path_key(events[j])
        if key not in seen:
            first.append(j);seen.add(key)
    chosen=(first+[j for j in order if j not in set(first)])[:budget]
    return np.array(chosen,dtype=int)


def fit_paths(z, y, events, ridge=1e-3):
    result={}
    groups={}
    for j,e in enumerate(events):
        groups.setdefault(path_key(e),[]).append(j)
    for key,idx in groups.items():
        zz=z[idx,7:];yy=y[idx]
        # Chemical paths use an intercept for the finite joint change. Geometric
        # response is fitted within a path, so chemical jumps cannot create huge
        # cross-path slopes in near-null atomic directions.
        if key=='geometry_only':
            zz=np.vstack([zz,np.zeros(2)]);yy=np.r_[yy,0.]
        center=zz.mean(axis=0);mean=float(yy.mean())
        a=zz-center;b=yy-mean
        coef=np.linalg.solve(a.T@a+ridge*np.eye(2),a.T@b)
        singular=np.linalg.svd(a,compute_uv=False)
        rank=int((singular>max(1e-10,singular[0]*.03)).sum()) if len(singular) else 0
        result[key]=dict(center=center,mean=mean,coefficient=coef,n=len(idx),geometry_rank=rank,
                         joint_jump_at_unperturbed_geometry=mean-center@coef)
    return result


def predict_paths(model,z,events):
    prediction=np.full(len(z),np.nan)
    supported=np.zeros(len(z),dtype=bool)
    for j,event in enumerate(events):
        key=path_key(event)
        if key in model:
            fit=model[key]
            prediction[j]=fit['mean']+(z[j,7:]-fit['center'])@fit['coefficient']
            supported[j]=True
    return prediction,supported
