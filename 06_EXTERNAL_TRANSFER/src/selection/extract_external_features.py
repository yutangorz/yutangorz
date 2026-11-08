# -*- coding: utf-8 -*-
"""Extract frozen nine descriptors from JARVIS structures using run019 code."""

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
import sys, json, traceback
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms

ROOT=_repo_path('.')
RUN19=_repo_path('05_CRISP_X/workflows/local_reconstruction')
sys.path.insert(0,str(_repo_path('05_CRISP_X/src/local_reconstruction')))
from shared_features.extraction import ConceptExtractor, FEATURES

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
d=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_INORGANIC.csv'),dtype={'external_id':str})
extractor=ConceptExtractor(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv'))

rows=[]; fails=[]
for i,r in enumerate(d.itertuples(index=False)):
    try:
        atoms=Atoms.from_dict(json.loads(r.atoms_json))
        s=atoms.pymatgen_converter()
        x,q=extractor.extract(s)
        rows.append(dict(external_id=r.external_id, **dict(zip(FEATURES,[float(v) for v in x])),
                         feature_status='extracted_frozen_run019', n_sites_extracted=len(s),
                         volume_extracted=float(s.volume),
                         min_shell_distance=float(q['min_shell_distance']),
                         mean_coordination=float(q['mean_coordination']),
                         angle_center_coverage=float(q['angle_center_coverage']),
                         max_missing_fraction=float(max(q['missing_fraction_'+f] for f in FEATURES[:7]))))
    except Exception as e:
        fails.append(dict(external_id=r.external_id, error_type=type(e).__name__, error=str(e)[:300]))
    if (i+1)%500==0:
        print('processed',i+1,'ok',len(rows),'fail',len(fails),flush=True)

out=pd.DataFrame(rows); out.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_ALL.csv'),index=False)
pd.DataFrame(fails).to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURE_FAILURES.csv'),index=False)
print('done',len(out),'fail',len(fails))
print(out[FEATURES].describe().to_string())
print('max missing',out.max_missing_fraction.value_counts().head(10).to_dict())
