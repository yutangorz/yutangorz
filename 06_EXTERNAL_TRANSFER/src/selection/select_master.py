# -*- coding: utf-8 -*-
"""Select the 300-material master set, 40 pilot, 20 challenge; write CIFs and tables."""

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
import json, hashlib
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

OUT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
STRUCT=_repo_path('06_EXTERNAL_TRANSFER/data/external_source/structures'); STRUCT.mkdir(exist_ok=True)
SEED=20261025
FEATURES=['period','electronegativity','covalent_radius','valence_electrons',
          'ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']

keep=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DEDUP_KEEP.csv'),dtype={'external_id':str})
feat=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D_WITH_DOMAIN.csv'),dtype={'external_id':str})
feature_cols=['external_id','atoms_json']+FEATURES+['max_missing_fraction','angle_center_coverage','min_shell_distance',
                                        'mean_coordination','n_sites_extracted','volume_extracted',
                                        'mahalanobis_train','mahal_pct_vs_oqmd_heldout','nn_train_z']
d=keep.merge(feat[feature_cols],on='external_id',validate='one_to_one')
meta=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CANDIDATES_INORGANIC.csv'),dtype={'external_id':str})
meta_cols=['external_id','formula','band_gap_method','band_gap_task_id','mbj_bandgap_eV','hse_gap_eV',
           'magnetic_moment_oszicar','magnetic_moment_outcar','soc_status','spacegroup_number','spacegroup_symbol']
d=d.merge(meta[meta_cols],on='external_id',validate='one_to_one')
d=d[d.strict_feature_compatible].copy()
d['domain_group']=d.domain_group.replace({'A_near':'A','B_moderate':'B','C_challenge':'C'})
d['nat_bin']=pd.cut(d.number_of_atoms,[0,6,10,16,24,40],labels=['2-6','7-10','11-16','17-24','25-40']).astype(str)
d['gap_bin']=pd.cut(d.band_gap_eV,[0,0.5,1,2,3,5,10],labels=['0-.5','.5-1','1-2','2-3','3-5','5+']).astype(str)
d['cost_q']=pd.qcut(d.pilot_cost_proxy,4,labels=['Q1','Q2','Q3','Q4']).astype(str)
d['stratum']=d.formula_class+'|'+d.nat_bin+'|'+d.gap_bin+'|'+d.cost_q
d['hash']=[hashlib.sha256(f"{SEED}|{x}".encode()).hexdigest() for x in d.external_id]

def pick(sub, n, used_formulas):
    if n<=0 or len(sub)==0: return []
    sub=sub[~sub.formula_reduced.isin(used_formulas)].copy()
    if len(sub)==0: return []
    queues={}
    for k,g in sub.groupby('stratum'):
        queues[k]=g.sort_values('hash').to_dict('records')
    keys=sorted(queues); picks=[]
    # round-robin over strata, one per formula
    while len(picks)<n and any(queues.values()):
        progress=False
        for k in keys:
            if len(picks)>=n: break
            q=queues[k]
            while q:
                r=q.pop(0)
                if r['formula_reduced'] in used_formulas: continue
                picks.append(r); used_formulas.add(r['formula_reduced']); progress=True; break
        if not progress: break
    # if still short, allow second phase with formula reuse (max 2 phases)
    if len(picks)<n:
        rest=sub[~sub.external_id.isin([x['external_id'] for x in picks])].sort_values('hash')
        for _,r in rest.iterrows():
            if len(picks)>=n: break
            picks.append(r.to_dict())
    return picks

lineage_targets={
 'A':{'JARVIS-native/AFLOW':33,'unknown':44,'Materials-Project-derived':23},
 'B':{'JARVIS-native/AFLOW':54,'unknown':30,'Materials-Project-derived':16},
 'C':{'JARVIS-native/AFLOW':60,'unknown':20,'Materials-Project-derived':20},
}
selected=[]; used=set()
for grp in ['A','B','C']:
    g=d[d.domain_group==grp].copy()
    for lin,n in lineage_targets[grp].items():
        sub=g[g.source_lineage==lin]
        picks=pick(sub,n,used)
        selected+=picks
    # fill any shortfall from remaining same group
    if len([x for x in selected if x['domain_group']==grp])<100:
        need=100-len([x for x in selected if x['domain_group']==grp])
        rest=g[~g.external_id.isin(used)]
        selected+=pick(rest,need,used)
# dedupe by external_id, trim to 300 with group balance 100/100/100 if possible
sel=pd.DataFrame(selected).drop_duplicates('external_id')
sel['domain_group']=sel.domain_group.astype(str)
# enforce 100/100/100 if over
parts=[]
for grp in ['A','B','C']:
    parts.append(sel[sel.domain_group==grp].head(100))
sel=pd.concat(parts,ignore_index=True)
sel['selection_rank']=[i+1 for i in range(len(sel))]
sel['selection_seed']=SEED
sel['selection_rule']='strict 9D; no OQMD/CRISP duplicates; 100 per A/B/C; source-lineage quotas 33/44/23 (A), 54/30/16 (B), 60/20/20 (C); one formula per stratum round-robin'
print('selected',len(sel),sel.domain_group.value_counts().to_dict())
print(pd.crosstab(sel.domain_group,sel.source_lineage).to_string())

# Write CIFs and build master
master=[]; failures=[]
for r in sel.itertuples(index=False):
    try:
        s=Atoms.from_dict(json.loads(r.atoms_json)).pymatgen_converter()
        rel=f"structures/{r.external_id}.cif"
        p=OUT/rel
        CifWriter(s).write_file(str(p))
        # readback
        s2=Structure.from_file(p)
        master.append(dict(
            external_id=r.external_id, source_database=r.source_database, source_version=r.source_version,
            formula=r.formula, formula_reduced=r.formula_reduced, formula_class=r.formula_class,
            structure_path=rel, number_of_atoms=len(s2),
            band_gap_eV=r.band_gap_eV, band_gap_method=r.band_gap_method, band_gap_task_id=r.band_gap_task_id,
            mbj_bandgap_eV=r.mbj_bandgap_eV, hse_gap_eV=r.hse_gap_eV,
            magnetic_moment_oszicar=r.magnetic_moment_oszicar, magnetic_moment_outcar=r.magnetic_moment_outcar,
            soc_status=r.soc_status, spacegroup_number=r.spacegroup_number, spacegroup_symbol=r.spacegroup_symbol,
            icsd=r.icsd, has_icsd=bool(r.has_icsd), reference=r.reference,
            source_lineage=r.source_lineage, source_lineage_uncertain=bool(r.source_lineage_uncertain),
            duplicate_status='keep_after_external_and_oqmd_dedup',
            feature_compatibility_status='strict_9d_extracted_frozen_run019',
            domain_group=r.domain_group, D_distribution=r.mahal_pct_vs_oqmd_heldout,
            D_mahalanobis_train=r.mahalanobis_train, nn_train_z=r.nn_train_z,
            pilot_cost_proxy=r.pilot_cost_proxy, inclusion_reason=(
                f"{r.domain_group}; strict 9D; source={r.source_lineage}; class={r.formula_class}; "
                f"nat={len(s2)}; gap={r.band_gap_eV:.3f} eV"),
            cif_readback_nsites=len(s2), cif_readback_volume=float(s2.volume)
        ))
    except Exception as e:
        failures.append(dict(external_id=r.external_id,error=str(e)[:200]))
master=pd.DataFrame(master)
master.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_MASTER.csv'),index=False)
print('master',len(master),'failures',len(failures))
if failures: pd.DataFrame(failures).to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_CIF_FAILURES.csv'),index=False)
print(pd.crosstab(master.domain_group,master.source_lineage).to_string())
