"""Resource-bounded exact distance-profile replay and explicit corrected helpers."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/material_selection')))
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.cluster.hierarchy import linkage as scipy_linkage


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def normalize_id(value):
    text=str(value).strip()
    if text.endswith('.vasp'):text=text[:-5]
    if not text or not text.isdigit():raise ValueError('Expected numeric material ID, optionally ending .vasp')
    return text


class DiskDistanceBackend:
    """Same native pdist/profile-pdist/Ward arithmetic as linkage(squareform(pdist(X))).

    Only storage/interception changes. Dense squareform is filled into a memmap,
    and native pdist(D) is passed as a condensed vector to native linkage. This
    bypasses only the dense-input warning/validation branch. No metric correction.
    """
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False);self.x=None;self.raw=None;self.dense=None;self.tree=None;self.calls=[]

    def pdist(self,x,metric='euclidean'):
        x=np.asarray(x,float)
        if x.ndim!=2 or not np.isfinite(x).all() or metric!='euclidean':raise ValueError('Unexpected historical pdist input')
        if self.x is not None:
            if not np.array_equal(x,self.x):raise ValueError('Cannot reuse historical cache for different features')
            self.calls.append('original_pdist_reused_identical_X');return self.raw
        self.x=x.copy();n=len(x)
        self.raw=np.memmap(self.folder/'original_condensed.float64',dtype='float64',mode='w+',shape=(n*(n-1)//2,))
        scipy_pdist(x,metric=metric,out=self.raw);self.raw.flush();self.calls.append('native_original_pdist')
        return self.raw

    def squareform(self,condensed):
        if condensed is not self.raw:raise ValueError('Unexpected condensed distance vector')
        if self.dense is not None:return self.dense
        n=len(self.x);self.dense=np.memmap(self.folder/'original_square.float64',dtype='float64',mode='w+',shape=(n,n))
        self.dense[:]=0.;start=0
        for i in range(n-1):
            size=n-i-1;v=condensed[start:start+size]
            self.dense[i,i+1:]=v;self.dense[i+1:,i]=v;start+=size
        self.dense.flush();self.calls.append('exact_squareform_disk_storage')
        return self.dense

    def linkage(self,dense,method='ward'):
        if dense is not self.dense or method!='ward':raise ValueError('Unexpected historical linkage call')
        if self.tree is not None:
            self.calls.append('same_profile_tree_reused');return self.tree.copy()
        n=len(self.x)
        profile=np.memmap(self.folder/'profile_condensed.float64',dtype='float64',mode='w+',shape=(n*(n-1)//2,))
        print('Computing native pdist of distance profiles',n,'x',n,flush=True)
        scipy_pdist(dense,metric='euclidean',out=profile);profile.flush()
        print('Distance profiles complete; native Ward linkage',flush=True)
        self.tree=scipy_linkage(profile,method='ward');np.save(self.folder/'historical_linkage.npy',self.tree)
        self.calls.append('native_profile_pdist_then_Ward')
        return self.tree.copy()


def select_local_two(z,labels,ids):
    """Per-cluster nearest centroid + local farthest from first, ID tie breaking.

    Independent of arbitrary cluster numbering; no historical-ID constraints.
    """
    z=np.asarray(z,float);labels=np.asarray(labels);ids=np.asarray(ids,str)
    if z.ndim!=2 or len(z)!=len(labels) or len(z)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(z).all():raise ValueError('Invalid selection input')
    selected=[]
    for label in sorted(set(labels)):
        ii=np.flatnonzero(labels==label)
        if len(ii)<2:raise ValueError('Cluster has fewer than two candidates')
        distance=np.linalg.norm(z[ii]-z[ii].mean(axis=0),axis=1)
        first=min(range(len(ii)),key=lambda k:(distance[k],ids[ii[k]]));a=ii[first]
        remainder=ii[ii!=a];d=np.linalg.norm(z[remainder]-z[a],axis=1)
        b=min(range(len(remainder)),key=lambda k:(-d[k],ids[remainder[k]]))
        selected.extend([int(a),int(remainder[b])])
    return np.asarray(selected,int)


def candidate_partition_metrics(ids,labels,expected):
    ids=np.asarray(ids,str);labels=np.asarray(labels);lookup=dict(zip(ids,labels))
    old=[];new=[];pairs=[]
    for row in expected.itertuples():
        if row.material_id not in lookup:raise ValueError('Missing historical candidate')
        old.append(int(row.cluster_id));new.append(int(lookup[row.material_id]))
    for i in range(len(old)):
        for j in range(i+1,len(old)):pairs.append((old[i]==old[j],new[i]==new[j]))
    positives=[b for a,b in pairs if a];negative=[not b for a,b in pairs if not a]
    return dict(candidate_n=len(old),occupied_clusters=len(set(new)),same_pair_recovered=sum(positives),same_pair_total=len(positives),
        different_pair_separated=sum(negative),different_pair_total=len(negative),numeric_labels_equal=bool(np.array_equal(old,new)))
