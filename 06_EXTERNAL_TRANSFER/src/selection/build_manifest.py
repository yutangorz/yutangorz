# -*- coding: utf-8 -*-
"""Build SHA256 manifest and machine-readable delivery metadata."""

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
import hashlib, json
import pandas as pd

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

files={}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='EXTERNAL_DATASET_MANIFEST.json':
        rel=str(p.relative_to(OUT)).replace('\\','/')
        files[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
master=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MASTER.csv'),dtype={'external_id':str})
pilot=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_PILOT.csv'),dtype={'external_id':str})
challenge=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_CHALLENGE.csv'),dtype={'external_id':str})
scale=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURE_SCALING.csv'))
manifest={
 'dataset_name':'CRISP-X external transfer dataset from JARVIS-DFT OptB88vdW',
 'dataset_version':'1.0',
 'generated_date':'2026-10-25',
 'source':{
   'primary_database':'JARVIS-DFT',
   'jarvis_doi':'10.6084/m9.figshare.6815699.v11',
   'jarvis_file':'jdft_3d-9-24-2025.json',
   'jarvis_url':'https://ndownloader.figshare.com/files/64391379',
   'jarvis_zip_sha256':'f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5',
   'jarvis_json_sha256':'9dacb55ac8371c7c2bf0332933b55c98e9cca6201e92eb9d6fcd03aab0975907',
   'jarvis_records':93902,
   'license':'CC BY 4.0',
   'citation':'Choudhary, K., Garrity, K.F., Reid, A.C.E. et al. npj Comput. Mater. 6, 173 (2020). DOI 10.1038/s41524-020-00440-1',
   'band_gap_method':'OptB88vdW only; MBJ/HSE stored separately, not used as main label',
   'soc_status':'not explicitly tagged in JARVIS OptB88vdW records',
   'materials_project':'investigated but not included: official API requires a key; no MP_API_KEY was available; no MP records in this delivery',
   'structural_provenance':'mixed JARVIS-native/AFLOW, unknown, and Materials-Project/ICSD-derived; see source_lineage'
 },
 'reference':{
   'oqmd_model_input':str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv')),
   'oqmd_model_input_sha256':'8fc24c95a8d8150a622ff07020d4c9005b13bcf77288b5cde8f382f1756f4499',
   'et_model_sha256':'1e3c7c3370b5fcc41b25db1eb20ff483c877c190d23d3c83b2035e3fd21784c6',
   'crisp_development_manifest':str(_repo_path('05_CRISP_X/data/perturbation_design/new_parent_manifest.csv')),
   'crisp_confirmation_manifest':str(_repo_path('05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv')),
   'crisp_parents':48
 },
 'counts':{
   'raw_candidates':7308,'inorganic_candidates':6026,'strict_feature_compatible':3630,
   'dedupe_keep_all':5101,'dedupe_keep_strict':3050,
   'master':len(master),'pilot':len(pilot),'challenge':len(challenge),
   'perovskite_core':int((master.formula_class.isin(['ABX3','A2BBX6','A2BX4'])).sum()),
   'strict_perovskite':int((master.formula_class.isin(['ABX3','A2BBX6'])).sum()),
   'pilot_perovskite_core':int(pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_PILOT.csv')).formula_class.isin(['ABX3','A2BBX6','A2BX4']).sum()),
   'master_domain':master.domain_group.value_counts().to_dict(),
   'master_lineage':master.source_lineage.value_counts().to_dict()
 },
 'deduplication':{
   'structure_matcher':'pymatgen StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=False)',
   'prefilter':'same reduced formula; volume ratio <=15%; atom-count ratio >=0.80',
   'internal_duplicates':713,'oqmd_structure_duplicates':212,'crisp_structure_duplicates':0,
   'same_formula_different_phase':'allowed; formula_phase_count/phase_status retained'
 },
 'selection':{
   'seed':20261025,
   'master_rule':'100 A / 100 B / 100 C; strict 9D; target source-lineage quotas with fill; actual A 33/42/25, B 51/31/18, C 60/20/20 (native/unknown/MP); one formula per stratum round-robin',
   'pilot_rule':'40 from A/B; tiered cost; no f; gap 0.3-4; 20 A / 20 B',
   'challenge_rule':'20 from C; top descriptor distance with formula-class quotas'
 },
 'features':{
   'order':['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle'],
   'extractor':'run019 shared_features.extraction.ConceptExtractor',
   'extractor_source':str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features/extraction.py')),
   'shell_ratio':1.20,
   'scaling':scale.to_dict('records'),
   'status':'frozen_nine_descriptor_extraction; verify against the declared extractor'
 },
 'files':files
}
(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MANIFEST.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('manifest files',len(files),'size', (_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MANIFEST.json')).stat().st_size)
print(manifest['counts'])
