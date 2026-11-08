# -*- coding: utf-8 -*-
"""Compute descriptor-domain distance of external candidates relative to frozen OQMD training."""

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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/selection')))
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=_repo_path('.')
RUN19=_repo_path('05_CRISP_X/workflows/local_reconstruction')
OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
FEATURES=['period','electronegativity','covalent_radius','valence_electrons',
          'ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']

mi=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str})
fold=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv'),dtype={'ID':str})
mi=mi.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one')
train=mi[mi.outer_fold!=0]; held=mi[mi.outer_fold==0]
mu=train[FEATURES].mean().to_numpy(float); sd=train[FEATURES].std(ddof=1).to_numpy(float); sd[sd==0]=1
Ztr=(train[FEATURES].to_numpy(float)-mu)/sd
Zhe=(held[FEATURES].to_numpy(float)-mu)/sd
cov=np.cov(Ztr,rowvar=False)+np.eye(9)*1e-6; inv=np.linalg.inv(cov)
def mah(Z): return np.sqrt(np.einsum('ij,jk,ik->i',Z,inv,Z))
d_he=mah(Zhe)
d_ext=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_ALL.csv'),dtype={'external_id':str})
Zex=(d_ext[FEATURES].to_numpy(float)-mu)/sd
d_ex=mah(Zex)
# nearest training neighbor distance (Euclidean in z)
# chunk to avoid memory
nn=np.empty(len(Zex))
for i in range(0,len(Zex),256):
    q=Zex[i:i+256]
    dist=np.sqrt(((q[:,None,:]-Ztr[None,:,:])**2).sum(axis=2))
    nn[i:i+256]=dist.min(axis=1)
# empirical percentile vs heldout
pct=np.array([(d_he < v).mean() for v in d_ex])
meta=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_INORGANIC.csv'),dtype={'external_id':str})
out=d_ext.merge(meta,on='external_id',validate='one_to_one')
out['mahalanobis_train']=d_ex
out['mahal_pct_vs_oqmd_heldout']=pct
out['nn_train_z']=nn
out['domain_group']=np.where(pct<0.5,'A_near',np.where(pct<0.9,'B_moderate','C_challenge'))
out['strict_feature_compatible']=(out.max_missing_fraction==0)
out.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_WITH_DOMAIN.csv'),index=False)
print('external',len(out),'strict',out.strict_feature_compatible.sum())
print(out.domain_group.value_counts().to_dict())
print('by strict')
print(out[out.strict_feature_compatible].domain_group.value_counts().to_dict())
print('distance quantiles external',np.quantile(d_ex,[0,.1,.25,.5,.75,.9,.95,.99,1]))
print('distance quantiles oqmd held',np.quantile(d_he,[0,.1,.25,.5,.75,.9,.95,.99,1]))
print('lineage x group')
print(pd.crosstab(out.source_lineage,out.domain_group).to_string())
