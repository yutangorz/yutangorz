# -*- coding: utf-8 -*-
"""Select 40 pilot and 20 challenge subsets from the 300-material master; write features/splits."""

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
import hashlib
import pandas as pd

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
SEED=20261025
FEATURES=['period','electronegativity','covalent_radius','valence_electrons',
          'ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']

master=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MASTER.csv'),dtype={'external_id':str})
feat=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_WITH_DOMAIN.csv'),dtype={'external_id':str})
master=master.merge(feat[['external_id']+FEATURES+['max_missing_fraction','angle_center_coverage','min_shell_distance','n_sites_extracted','volume_extracted']],on='external_id',how='left')
master=master.drop(columns=[c for c in master.columns if c.endswith('_x') or c.endswith('_y')])
master['hash']=[hashlib.sha256(f"{SEED}|{x}".encode()).hexdigest() for x in master.external_id]

# ---- pilot: A/B, tiered low cost, no f, gap 0.3-4 ----
p=master[(master.domain_group.isin(['A','B'])) & (master.number_of_atoms<=25) &
         (master.f_element==0) & (master.band_gap_eV>=0.3) & (master.band_gap_eV<=4.0)].copy()
p['tier']=1
p.loc[(p.number_of_atoms>15)|(p.heavy_soc==1)|(p.magnetic_3d==1),'tier']=2
p.loc[(p.number_of_atoms>20)|((p.heavy_soc==1)&(p.magnetic_3d==1)),'tier']=3
p['source_pri']=p.source_lineage.map({'JARVIS-native/AFLOW':0,'unknown':1,'Materials-Project-derived':2}).fillna(3)
p=p.sort_values(['tier','source_pri','pilot_cost_proxy','hash'])
pilot=[]; used=set()
for grp in ['A','B']:
    target=20
    for _,r in p[p.domain_group==grp].iterrows():
        if len([x for x in pilot if x['domain_group']==grp])>=target: break
        if r.formula_reduced in used: continue
        pilot.append(r.to_dict()); used.add(r.formula_reduced)
pilot=pd.DataFrame(pilot).drop_duplicates('external_id').head(40)
pilot['pilot_reason']='A/B near/moderate; tiered cost (T1 <=15 atoms/no heavy/magnetic; T2 <=20 or one flag; T3 <=25); no f; gap 0.3-4; formula diversity'

# ---- challenge: C, highest descriptor distance, class quotas ----
c=master[master.domain_group=='C'].copy().sort_values('D_mahalanobis_train',ascending=False)
challenge=[]; usedc=set()
class_targets={'ABX3':4,'A2BX4':4,'A2X3':4,'AX2':4,'other':4,'A2BBX6':1}
for cls,n in class_targets.items():
    sub=c[c.formula_class==cls]
    for _,r in sub.iterrows():
        if len([x for x in challenge if x['formula_class']==cls])>=n: break
        if r.formula_reduced in usedc: continue
        challenge.append(r.to_dict()); usedc.add(r.formula_reduced)
# fill to 20
for _,r in c.iterrows():
    if len(challenge)>=20: break
    if r.formula_reduced in usedc: continue
    challenge.append(r.to_dict()); usedc.add(r.formula_reduced)
challenge=pd.DataFrame(challenge).drop_duplicates('external_id').head(20)
challenge['challenge_reason']='C challenge; top descriptor-distance within formula-class diversity; strict 9D'

# ---- outputs ----
master['split_role']='master_only'
master.loc[master.external_id.isin(pilot.external_id),'split_role']='pilot'
master.loc[master.external_id.isin(challenge.external_id),'split_role']='challenge'
master['selection_rank']=range(1,len(master)+1)
master['selection_seed']=SEED
master['selection_rule']='strict 9D; no OQMD/CRISP duplicates; 100 per A/B/C; source-lineage quotas; formula-class/source diversity'
master.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MASTER.csv'),index=False)
pilot.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_PILOT.csv'),index=False)
challenge.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_CHALLENGE.csv'),index=False)

splits=master[['external_id','formula','formula_class','domain_group','source_lineage','D_mahalanobis_train','split_role','selection_rank','selection_seed','selection_rule','inclusion_reason']].copy()
splits['pilot_flag']=splits.split_role.eq('pilot')
splits['challenge_flag']=splits.split_role.eq('challenge')
splits.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_SPLITS.csv'),index=False)

features=feat[feat.external_id.isin(master.external_id)][['external_id']+FEATURES+['max_missing_fraction','angle_center_coverage','min_shell_distance','n_sites_extracted','volume_extracted']].copy()
features['feature_status']='extracted_frozen_run019_strict'
features.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D.csv'),index=False)

# feature scaling from frozen training
root=_repo_path('.')
mi=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str})
fold=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv'),dtype={'ID':str})
tr=mi.merge(fold[['ID','outer_fold']],on='ID').query('outer_fold != 0')
scale=pd.DataFrame({'feature':FEATURES,'train_mean':tr[FEATURES].mean().values,'train_std':tr[FEATURES].std(ddof=1).values})
scale.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURE_SCALING.csv'),index=False)

print('master',len(master),'pilot',len(pilot),'challenge',len(challenge))
print('pilot groups',pilot.domain_group.value_counts().to_dict(),'lineage',pilot.source_lineage.value_counts().to_dict())
print('challenge groups',challenge.domain_group.value_counts().to_dict(),'classes',challenge.formula_class.value_counts().to_dict())
print('splits',master.split_role.value_counts().to_dict())
