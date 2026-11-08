# -*- coding: utf-8 -*-
"""Fast multi-level deduplication with volume/size prefilter.
StructureMatcher is only run on plausible same-formula pairs; no supercell attempt.
"""

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
import json, time
import pandas as pd
from jarvis.core.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

ROOT=_repo_path('.')
RUN19=_repo_path('05_CRISP_X/workflows/local_reconstruction')
RUN14=_repo_path('05_CRISP_X/workflows/perturbation_design')
OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
d=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_WITH_DOMAIN.csv'),dtype={'external_id':str})
oq=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str})
crisp=pd.read_csv(_repo_path('05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv'),dtype={'ID':str})
dev=pd.read_csv(_repo_path('05_CRISP_X/data/perturbation_design/new_parent_manifest.csv'),dtype={'ID':str})
crisp_ids=set(crisp.ID)|set(dev.ID)
crisp_formulas=set(oq[oq.ID.isin(crisp_ids)].formula_group)
matcher=StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=False)

def ext_struct(row):
    return Atoms.from_dict(json.loads(row.atoms_json)).pymatgen_converter()

def plausible(a,b):
    if a is None or b is None: return False
    va,vb=a.volume,b.volume
    if va<=0 or vb<=0: return False
    if abs(va-vb)/max(va,vb) > 0.15: return False
    na,nb=len(a),len(b)
    if min(na,nb)/max(na,nb) < 0.80: return False
    return True

# OQMD structures only for overlapping formulas
oq_by_formula={}
for f in set(d.formula_reduced)&set(oq.formula_group):
    arr=[]
    for r in oq[oq.formula_group==f].itertuples(index=False):
        try: arr.append((r.ID,Structure.from_file(r.path)))
        except Exception: pass
    if arr: oq_by_formula[f]=arr
# CRISP structures for overlapping formulas (expected empty)
crisp_by_formula={}
for f in set(d.formula_reduced)&crisp_formulas:
    arr=[]
    for r in oq[oq.formula_group==f].itertuples(index=False):
        try: arr.append((r.ID,Structure.from_file(r.path)))
        except Exception: pass
    if arr: crisp_by_formula[f]=arr
print('OQMD formula groups',len(oq_by_formula),'CRISP groups',len(crisp_by_formula),flush=True)

rows=[]; t0=time.time()
for gi,(formula,group) in enumerate(d.groupby('formula_reduced',sort=False)):
    # representative preference: strict compatible, native-ish, low cost
    group=group.assign(_rank=(~group.strict_feature_compatible).astype(int)*10 +
                             (group.source_lineage!='JARVIS-native/AFLOW').astype(int)*2 +
                             group.pilot_cost_proxy/1000).sort_values('_rank')
    structs={}; errors={}
    for r in group.itertuples(index=False):
        try: structs[r.external_id]=ext_struct(r)
        except Exception as e: errors[r.external_id]=str(e)[:120]
    reps=[]
    for r in group.itertuples(index=False):
        status='keep'; matched=''; note=''
        s=structs.get(r.external_id)
        if s is not None:
            for rep_id,rep_s in reps:
                if plausible(s,rep_s):
                    try:
                        if matcher.fit(s,rep_s):
                            status='duplicate_external'; matched=rep_id; note='StructureMatcher ltol0.2 stol0.3 angle5 primitive scale; no supercell'; break
                    except Exception: pass
        if status=='keep' and formula in oq_by_formula and s is not None:
            for oid,os in oq_by_formula[formula]:
                if plausible(s,os):
                    try:
                        if matcher.fit(s,os):
                            status='duplicate_oqmd'; matched=oid; note='structure match to frozen 7176 OQMD'; break
                    except Exception: pass
        if status=='keep' and formula in crisp_by_formula and s is not None:
            for cid,cs in crisp_by_formula[formula]:
                if plausible(s,cs):
                    try:
                        if matcher.fit(s,cs):
                            status='duplicate_crisp'; matched=cid; note='structure match to CRISP parent'; break
                    except Exception: pass
        if status=='keep' and s is not None:
            reps.append((r.external_id,s))
        rows.append(dict(external_id=r.external_id, formula_reduced=formula, formula_class=r.formula_class,
            source_database=r.source_database, source_version=r.source_version, source_lineage=r.source_lineage,
            reference=r.reference, icsd=r.icsd, has_icsd=bool(r.has_icsd),
            oqmd_formula_overlap=bool(r.oqmd_formula_overlap), crisp_formula_overlap=bool(r.crisp_formula_overlap),
            duplicate_status=status, matched_id=matched, duplicate_note=note,
            structure_parse_error=errors.get(r.external_id,''),
            internal_group_size=len(group), strict_feature_compatible=bool(r.strict_feature_compatible),
            domain_group=r.domain_group, band_gap_eV=r.band_gap_eV, number_of_atoms=r.number_of_atoms,
            source_lineage_uncertain=bool(r.has_icsd or r.source_lineage!='JARVIS-native/AFLOW'),
            pilot_cost_proxy=r.pilot_cost_proxy))
    if (gi+1)%500==0:
        print('groups',gi+1,'rows',len(rows),'seconds',round(time.time()-t0,1),flush=True)
out=pd.DataFrame(rows)
out.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DEDUPLICATION_REPORT.csv'),index=False)
keep=out[out.duplicate_status=='keep'].copy(); keep.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DEDUP_KEEP.csv'),index=False)
print('dedupe rows',len(out),'seconds',round(time.time()-t0,1))
print(out.duplicate_status.value_counts().to_dict())
print('keep',len(keep),'strict',keep.strict_feature_compatible.sum())
print(pd.crosstab(keep.domain_group,keep.source_lineage).to_string())
