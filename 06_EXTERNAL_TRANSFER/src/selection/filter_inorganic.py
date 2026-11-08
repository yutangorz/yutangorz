# -*- coding: utf-8 -*-
"""Filter JARVIS raw candidates to inorganic perovskite-relevant chemistries and add lineage flags."""

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
import re
import pandas as pd
from pymatgen.core import Composition

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
OQ=_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv')
RUN14=_repo_path('05_CRISP_X/data/perturbation_design/new_parent_manifest.csv')
RUN19=_repo_path('05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv')

d=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_RAW.csv'),dtype={'external_id':str})
oq=pd.read_csv(OQ,dtype={'ID':str})
oq_formulas=set(oq.formula_group)
crisp_ids=set(pd.read_csv(RUN19,dtype={'ID':str}).ID)|set(pd.read_csv(RUN14,dtype={'ID':str}).ID)
crisp_formulas=set(oq[oq.ID.isin(crisp_ids)].formula_group)

AN={'O','F','Cl','Br','I','S','Se','N','Te'}
BAD={'C','H','He','Ne','Ar','Kr','Xe','Rn','Tc','Th','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'}
F_ELEM=set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr'.split())
MAG=set('V Cr Mn Fe Co Ni Cu'.split())
HEAVY=set('Sn Sb Te I Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn'.split())|set('Cs Ba La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split())

def formula_class(comp):
    # integer counts sorted; ABX3 and A2BB'X6 patterns
    counts=sorted([int(round(v)) for v in comp.get_el_amt_dict().values()])
    if len(counts)==3 and counts==[1,1,3]: return 'ABX3'
    if len(counts)==4 and counts==[1,1,2,6]: return 'A2BBX6'
    if len(counts)==3 and counts==[1,2,4]: return 'A2BX4'
    if len(counts)==2 and counts==[1,2]: return 'AX2'
    if len(counts)==2 and counts==[1,3]: return 'A2X3'
    return 'other'

rows=[]; rejected=[]
for r in d.itertuples(index=False):
    try:
        comp=Composition(r.formula)
        els=set(str(e.symbol) for e in comp.elements)
    except Exception:
        rejected.append((r.external_id,'formula_parse')); continue
    if els & BAD:
        rejected.append((r.external_id,'excluded_element_C_H_noble_radioactive')); continue
    if not (els & AN):
        rejected.append((r.external_id,'no_inorganic_anion')); continue
    if not (2 <= len(els) <= 5):
        rejected.append((r.external_id,'element_count')); continue
    ref=str(r.reference)
    if re.match(r'^mp-',ref,re.I):
        lineage='Materials-Project-derived'
    elif re.match(r'^(auid-|JVASP-)',ref,re.I):
        lineage='JARVIS-native/AFLOW'
    else:
        lineage='unknown'
    rows.append(dict(
        external_id=r.external_id, source_database=r.source_database, source_version=r.source_version,
        formula=r.formula, formula_reduced=r.formula_reduced, formula_class=formula_class(comp),
        n_elements=r.n_elements, number_of_atoms=r.number_of_atoms, band_gap_eV=r.band_gap_eV,
        band_gap_method=r.band_gap_method, band_gap_task_id=r.band_gap_task_id,
        mbj_bandgap_eV=r.mbj_bandgap_eV, hse_gap_eV=r.hse_gap_eV,
        magnetic_moment_oszicar=r.magnetic_moment_oszicar, magnetic_moment_outcar=r.magnetic_moment_outcar,
        soc_status=r.soc_status, spacegroup_number=r.spacegroup_number, spacegroup_symbol=r.spacegroup_symbol,
        dimensionality=r.dimensionality, icsd=r.icsd, has_icsd=r.has_icsd, reference=r.reference,
        xml_data_link=r.xml_data_link, ehull_eV_atom=r.ehull_eV_atom,
        formation_energy_eV_atom=r.formation_energy_eV_atom, spillage=r.spillage,
        encut_eV=r.encut_eV, kpoint_length_unit=r.kpoint_length_unit, atoms_json=r.atoms_json,
        source_lineage=lineage,
        source_independence_tier=('native-ish' if lineage=='JARVIS-native/AFLOW' else 'MP/ICSD-derived'),
        oqmd_formula_overlap=(r.formula_reduced in oq_formulas),
        crisp_formula_overlap=(r.formula_reduced in crisp_formulas),
        f_element=int(bool(els & F_ELEM)), magnetic_3d=int(bool(els & MAG)), heavy_soc=int(bool(els & HEAVY)),
        pilot_cost_proxy=float(r.number_of_atoms/5 + len(els)/2 + 2*bool(els & F_ELEM) + 0.7*bool(els & MAG) + 0.8*bool(els & HEAVY)),
    ))
out=pd.DataFrame(rows)
out.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_INORGANIC.csv'), index=False)
pd.DataFrame(rejected,columns=['external_id','reject_reason']).to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_INORGANIC_REJECTIONS.csv'),index=False)
print('inorganic',len(out),'unique formulas',out.formula_reduced.nunique())
print('formula classes',out.formula_class.value_counts().to_dict())
print('lineage',out.source_lineage.value_counts().to_dict())
print('oqmd formula overlap',out.oqmd_formula_overlap.sum(),'crisp formula overlap',out.crisp_formula_overlap.sum())
print('pilot cost',out.pilot_cost_proxy.describe().to_dict())
print('rejections',len(rejected),pd.Series([x[1] for x in rejected]).value_counts().to_dict())
