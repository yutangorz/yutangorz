# -*- coding: utf-8 -*-
"""Delivery QA: counts, IDs, CIF readback, features, splits, labels, dedupe consistency."""

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
import json
import numpy as np
import pandas as pd
from pymatgen.core import Structure

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
FEATURES=['period','electronegativity','covalent_radius','valence_electrons',
          'ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']

m=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MASTER.csv'),dtype={'external_id':str})
p=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_PILOT.csv'),dtype={'external_id':str})
c=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_CHALLENGE.csv'),dtype={'external_id':str})
f=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D.csv'),dtype={'external_id':str})
s=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_SPLITS.csv'),dtype={'external_id':str})
d=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DEDUPLICATION_REPORT.csv'),dtype={'external_id':str})

checks={
 'master_300':len(m)==300,
 'pilot_40':len(p)==40,
 'challenge_20':len(c)==20,
 'pilot_subset_master':set(p.external_id).issubset(set(m.external_id)),
 'challenge_subset_master':set(c.external_id).issubset(set(m.external_id)),
 'no_duplicate_master_ids':m.external_id.duplicated().sum()==0,
 'no_duplicate_pilot_ids':p.external_id.duplicated().sum()==0,
 'no_duplicate_challenge_ids':c.external_id.duplicated().sum()==0,
 'master_abc_100_each':m.domain_group.value_counts().to_dict()=={'A':100,'B':100,'C':100},
 'features_300':len(f)==300,
 'features_match_master':set(f.external_id)==set(m.external_id),
 'feature_columns_order':list(f.columns)[1:10]==FEATURES,
 'no_nan_features':not f[FEATURES].isna().any().any(),
 'splits_300':len(s)==300,
 'splits_match_master':set(s.external_id)==set(m.external_id),
 'all_structure_paths_exist':all((OUT/x).exists() for x in m.structure_path),
 'band_gap_method_single':m.band_gap_method.nunique()==1 and str(m.band_gap_method.iloc[0]).startswith('OptB88vdW'),
 'dedupe_no_keep_with_match':not ((d.duplicate_status=='keep') & (d.matched_id.fillna('')!='')).any(),
 'dedupe_oqmd_212':int((d.duplicate_status=='duplicate_oqmd').sum())==212,
 'dedupe_external_713':int((d.duplicate_status=='duplicate_external').sum())==713,
 'dedupe_crisp_0':int((d.duplicate_status=='duplicate_crisp').sum())==0,
}
# CIF readback check (all 300)
read_fail=[]; ns_mismatch=[]
for r in m.itertuples(index=False):
    try:
        st=Structure.from_file(OUT/r.structure_path)
        if len(st)!=r.number_of_atoms: ns_mismatch.append(r.external_id)
    except Exception as e:
        read_fail.append((r.external_id,str(e)[:120]))
checks['all_cifs_read']=len(read_fail)==0
checks['all_cif_nsites_match']=len(ns_mismatch)==0
qa=dict(checks=checks, counts={'master':len(m),'pilot':len(p),'challenge':len(c),'features':len(f),
    'dedupe_rows':len(d),'dedupe_status':d.duplicate_status.value_counts().to_dict(),
    'domain':m.domain_group.value_counts().to_dict(),'lineage':m.source_lineage.value_counts().to_dict(),
    'pilot_domain':p.domain_group.value_counts().to_dict(),'challenge_domain':c.domain_group.value_counts().to_dict()},
    read_fail=read_fail, ns_mismatch=ns_mismatch)
qa_json=json.loads(json.dumps(qa,ensure_ascii=False,default=lambda o: o.item() if hasattr(o,'item') else str(o)))
(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/DELIVERY_QA.json')).write_text(json.dumps(qa_json,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(qa_json,ensure_ascii=False,indent=2))
