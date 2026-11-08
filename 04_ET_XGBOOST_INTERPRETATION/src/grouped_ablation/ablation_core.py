"""Training-fold-only correlation groups and paired prediction error contrasts."""

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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/grouped_ablation')))
import numpy as np


def correlation_groups(x,threshold=.8):
    x=np.asarray(x,float)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all() or not 0<threshold<1:
        raise ValueError('Finite training matrix and valid threshold required')
    corr=np.corrcoef(x,rowvar=False);corr=np.nan_to_num(corr,nan=0.);np.fill_diagonal(corr,1.)
    seen=set();groups=[]
    for j in range(x.shape[1]):
        if j in seen:continue
        stack=[j];component=[];seen.add(j)
        while stack:
            k=stack.pop();component.append(k)
            for m in np.flatnonzero(abs(corr[k])>=threshold):
                if int(m) not in seen:seen.add(int(m));stack.append(int(m))
        if len(component)>1:groups.append(tuple(sorted(component)))
    return groups,corr


def make_variants(p,groups):
    variants={():['full9']}
    for j in range(p):variants[(j,)]=['single_feature']
    for cols,tag in [(tuple(range(7)),'atomic7_group'),((7,8),'structural2_group')]+[(tuple(g),'training_correlation_component') for g in groups]:
        if len(set(cols))!=len(cols) or any(j<0 or j>=p for j in cols):raise ValueError('Bad removal group')
        variants.setdefault(tuple(sorted(cols)),[]).append(tag)
    return variants


def paired_errors(truth,full,dropped):
    y,a,b=map(lambda x:np.asarray(x,float),(truth,full,dropped))
    if y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape or not all(np.isfinite(v).all() for v in [y,a,b]):
        raise ValueError('Matching finite prediction vectors required')
    return abs(y-b)-abs(y-a)


def clustered_interval(effects,groups,seed=20261031,repeats=2000):
    v=np.asarray(effects,float);g=np.asarray(groups)
    if v.ndim!=1 or len(v)!=len(g) or not len(v) or not np.isfinite(v).all():raise ValueError('Invalid clustered effects')
    labels,idx=np.unique(g,return_inverse=True);sums=np.bincount(idx,weights=v);counts=np.bincount(idx)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        ii=rng.integers(0,len(labels),len(labels));values.append(sums[ii].sum()/counts[ii].sum())
    return np.asarray(values)
