# -*- coding: utf-8 -*-
"""Build the raw external candidate table from JARVIS-DFT 3D (2025-09-24)."""

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
import json, re, math
import pandas as pd
from pymatgen.core import Composition

RAW = _repo_path('external_inputs/external_raw/extracted_2025/jdft_3d-9-24-2025.json')
OUT = _repo_path('06_EXTERNAL_TRANSFER/data/external_source')
(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/reports')).mkdir(parents=True, exist_ok=True)
(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/structures')).mkdir(exist_ok=True)

def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

with open(RAW, encoding='utf-8') as f:
    data=json.load(f)

rows=[]
for r in data:
    try:
        dim=str(r.get('dimensionality',''))
        if '3D' not in dim:
            continue
        if str(r.get('func','')) != 'OptB88vdW':
            continue
        gap=num(r.get('optb88vdw_bandgap'))
        if gap is None or gap <= 0.05:
            continue
        nat=int(r.get('nat',0) or 0)
        if nat < 2 or nat > 40:
            continue
        atoms=r.get('atoms')
        if not isinstance(atoms,dict):
            continue
        els=atoms.get('elements')
        if not isinstance(els,list) or len(els)!=nat:
            continue
        formula=str(r.get('formula','')).strip()
        if not formula or formula.lower() in ('nan','na'):
            continue
        try:
            comp=Composition(formula)
            red=comp.reduced_formula
            n_el=len(comp.elements)
        except Exception:
            continue
        icsd=str(r.get('icsd','na'))
        ref=str(r.get('reference',''))
        rows.append(dict(
            external_id=str(r.get('jid')),
            source_database='JARVIS-DFT',
            source_version='jdft_3d-9-24-2025.json; Figshare DOI 10.6084/m9.figshare.6815699.v11; published 2026-05-09',
            formula=formula,
            formula_reduced=red,
            n_elements=n_el,
            number_of_atoms=nat,
            band_gap_eV=gap,
            band_gap_method='OptB88vdW (JARVIS-DFT, func=OptB88vdW)',
            band_gap_task_id=str(r.get('jid')),
            mbj_bandgap_eV=num(r.get('mbj_bandgap')),
            hse_gap_eV=num(r.get('hse_gap')),
            magnetic_moment_oszicar=num(r.get('magmom_oszicar')),
            magnetic_moment_outcar=num(r.get('magmom_outcar')),
            soc_status='not explicitly tagged in JARVIS OptB88vdW record; treat as non-SOC/scalar-relativistic workflow, verify before SOC claims',
            spacegroup_number=r.get('spg_number'),
            spacegroup_symbol=r.get('spg_symbol'),
            dimensionality=dim,
            icsd=str(icsd),
            has_icsd=(icsd not in ('na','None','')),
            reference=ref,
            xml_data_link=str(r.get('xml_data_link','')),
            ehull_eV_atom=num(r.get('ehull')),
            formation_energy_eV_atom=num(r.get('formation_energy_peratom')),
            spillage=num(r.get('spillage')),
            encut_eV=num(r.get('encut')),
            kpoint_length_unit=num(r.get('kpoint_length_unit')),
            atoms_json=json.dumps(atoms,separators=(',',':'))
        ))
    except Exception:
        continue

df=pd.DataFrame(rows)
# exact JID duplicate check
df['jid_duplicate_count']=df.groupby('external_id').external_id.transform('count')
# formula duplicates are allowed (different phases), record count
df['formula_duplicate_count']=df.groupby('formula_reduced').formula_reduced.transform('count')
df.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_RAW.csv'), index=False)
print('raw candidates', len(df))
print('unique jid', df.external_id.nunique(), 'unique formulas', df.formula_reduced.nunique())
print('gap bins', pd.cut(df.band_gap_eV,[0.05,0.5,1,2,3,5,10]).value_counts().sort_index().to_dict())
print('nat', df.number_of_atoms.describe().to_dict())
print('3D dims', df.dimensionality.value_counts().to_dict())
print('icsd', df.has_icsd.value_counts().to_dict())
print('top formulas', df.formula_reduced.value_counts().head(10).to_dict())
