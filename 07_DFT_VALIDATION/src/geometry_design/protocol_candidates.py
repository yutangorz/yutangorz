"""Read small accepted-reference inputs; describe candidate compatibility, no DFT writes."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/geometry_design')))
import csv,json,re
from pathlib import Path

ROOT=_repo_path('.')
OUT=_repo_path('07_DFT_VALIDATION/workflows/geometry_design')
with (_repo_path('07_DFT_VALIDATION/data/geometry_design/CURRENT_20_MATERIALS.csv')).open(encoding='utf-8',newline='') as f:materials=list(csv.DictReader(f))
rows=[]
for m in materials:
    p=Path(m['baseline_structure']).parent
    incar=p/'INCAR';kpoints=p/'KPOINTS'
    params={}
    if incar.is_file():
        for line in incar.read_text(encoding='utf-8',errors='replace').splitlines():
            clean=line.split('#',1)[0].split('!',1)[0]
            if '=' in clean:
                k,v=clean.split('=',1);params[k.strip().upper()]=v.strip()
    mesh='UNKNOWN'
    if kpoints.is_file():
        lines=[x.strip() for x in kpoints.read_text(encoding='utf-8',errors='replace').splitlines() if x.strip()]
        if len(lines)>=4:mesh=lines[3]
    oldclass=m['accepted_parent_protocol_class']
    same=oldclass=='NEW_LOW_COST_NON_SOC_FIXED_REFERENCE_GEOMETRY'
    rows.append({'material_id':m['material_id'],'parent_job':m['parent_job'],'parent_INCAR':str(incar) if incar.is_file() else 'MISSING_LOCAL','parent_KPOINTS':str(kpoints) if kpoints.is_file() else 'MISSING_LOCAL','parent_protocol_class':oldclass,'parent_ENCUT':params.get('ENCUT','UNKNOWN'),'parent_EDIFF':params.get('EDIFF','UNKNOWN'),'parent_ISPIN':params.get('ISPIN','UNKNOWN'),'parent_LSORBIT':params.get('LSORBIT','UNKNOWN'),'parent_NBANDS':params.get('NBANDS','UNKNOWN'),'parent_ISMEAR':params.get('ISMEAR','UNKNOWN'),'parent_SIGMA':params.get('SIGMA','UNKNOWN'),'parent_IVDW_explicit':params.get('IVDW','NOT_EXPLICIT_VALIDATE_RUNTIME'),'parent_MAGMOM':params.get('MAGMOM','NOT_EXPLICIT_OR_UNKNOWN'),'parent_mesh':mesh,'PAW_exact_hash':'NEEDS_FROZEN_SOURCE_CROSSCHECK_BEFORE_DFT','NELECT_exact':'NEEDS_FROZEN_SOURCE_CROSSCHECK_BEFORE_DFT','proposed_main_protocol':'PBE_NON_SOC_FIXED_EDIFF1E-6_CANDIDATE','parent_baseline_reuse':'CANDIDATE_SAME_PROTOCOL_NEEDS_FULL_INPUT_SHA_AND_SPECTRUM_CHECK' if same else 'NOT_COMPATIBLE_WITH_PROPOSED_MAIN_PROTOCOL_WITHOUT_NEW_MATCHED_BASELINE','protocol_status':'NOT_FROZEN_BATCH00'})
with (_repo_path('07_DFT_VALIDATION/workflows/geometry_design/PROTOCOL_CANDIDATES.csv')).open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('protocol candidates',len(rows),'possible parent reuse',sum('CANDIDATE' in x['parent_baseline_reuse'] for x in rows))
