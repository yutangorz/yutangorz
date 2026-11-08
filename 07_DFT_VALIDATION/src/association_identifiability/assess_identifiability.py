"""Offline run004 synthesis. All responses retain their source and evidence limits."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/association_identifiability')))
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT=Path(str(_repo_path('07_DFT_VALIDATION/workflows/association_identifiability')))
R2=Path(str(_repo_path('07_DFT_VALIDATION/workflows/verified_descriptors')))
R3=Path(str(_repo_path('07_DFT_VALIDATION/workflows/association_screening')))
ROOT.mkdir(exist_ok=True)
S=Path(str(_repo_path('07_DFT_VALIDATION/workflows/geometry_qualification/UNIFIED_440_RESULT_SET.csv')))
E=Path(str(_repo_path('07_DFT_VALIDATION/data/material_selection/FINAL_ELEMENT_SUBSTITUTION_MATRIX_DESIGN_ONLY.csv')))
T=Path(str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv')))
X=Path(str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features/extraction.py')))
SH=Path(str(_repo_path('outputs/05_CRISP_X/local_reconstruction/shap_baseline/TABLE_nine_importance.csv')))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(data,name):
    data.to_csv(ROOT/name,index=False,encoding='utf-8-sig')
def stat(x,y,method):
    x=np.asarray(x,float);y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan
    return float(method(x,y).statistic)

s=pd.read_csv(S);e=pd.read_csv(E);r=pd.read_csv(R3/'ELEMENT_RESPONSE_INPUT.csv')
site=pd.read_csv(R2/'ELEMENT_SITE_MAPPING_AUDIT.csv')
base=pd.read_csv(R2/'STRUCTURE_BASELINE_JOB_RECOVERY_v2.csv')
proto=pd.read_csv(R2/'STRUCTURE_PROTOCOL_RECOVERY.csv')
pair_protocol=pd.read_csv(ROOT/'STRUCTURE_PAIR_PROTOCOL_AUDIT.csv')
assert len(pair_protocol)==440 and set(pair_protocol.protocol_status)=={'KEY_INPUTS_EXACT_MATCH'}
cross=pd.read_csv(R3/'CROSS_SCALE_PROTOCOL_COMPATIBILITY.csv')
g=pd.read_csv(R3/'MATERIAL_GLOBAL_SENSITIVITY.csv')
l=pd.read_csv(R3/'LOCAL_GEOMETRY_RESPONSE_SUMMARY.csv')

freeze=[]
for label,path,count,role in [
    ('structure_440',S,440,'frozen_response'),('element_20',E,20,'frozen_response'),
    ('atom_table',T,100,'feature_definition'),('extractor',X,1,'feature_definition'),
    ('run002_baseline',R2/'STRUCTURE_BASELINE_JOB_RECOVERY_v2.csv',440,'identity'),
    ('run002_protocol',R2/'STRUCTURE_PROTOCOL_RECOVERY.csv',440,'protocol'),
    ('run003_element_input',R3/'ELEMENT_RESPONSE_INPUT.csv',16,'analysis_input'),
    ('run003_global',R3/'GLOBAL_GEOMETRY_RESPONSE_INPUT.csv',371,'analysis_input'),
    ('run019_shap',SH,9,'reference_only')]:
    freeze.append(dict(dataset=label,path=str(path),sha256=sha(path),expected_rows=count,role=role,eligibility='SOURCE_ONLY_PROVISIONAL_ANALYSIS'))
save(pd.DataFrame(freeze),'ANALYSIS_DATA_FREEZE.csv')

# Protocol, spectrum and numerical-resolution evidence are independent gates.
num=[]
for _,z in e.iterrows():
    fixed='LEGACY' not in z.protocol_class
    num.append(dict(data_family='ELEMENT',material_id=z.current_material_id,point_id=z.pair_id,
        parent_job=z.parent_job,perturbed_job=z.sub_job,parent_Eg_eV=z.Eg_parent_eV,perturbed_Eg_eV=z.Eg_sub_eV,
        delta_Eg_eV=z.delta_Eg_eV,protocol_match_status=z.pair_status,
        band_edge_tracking_status='SAMPLED_SPECTRUM_ACCEPTED_WITHOUT_CROSS_MESH_EDGE_TRACKING',
        numerical_error_evidence='NO_PAIR_SPECIFIC_TRANSFERABLE_ERROR_BOUND',delta_Eg_uncertainty_eV=np.nan,
        uncertainty_type='UNKNOWN',resolution_status='UNRESOLVED',geometry_scope='FIXED' if fixed else 'SEPARATELY_RELAXED',
        source_path=str(E),source_sha256=sha(E)))
for _,z in s.iterrows():
    br=base[base.point_id==z.point_id].iloc[0]
    pa=pair_protocol[pair_protocol.point_id==z.point_id].iloc[0]
    num.append(dict(data_family='STRUCTURE',material_id=z.material_id,point_id=z.point_id,
        parent_job=br.baseline_job,perturbed_job=z.job_id,parent_Eg_eV=float(z.Eg_eV-z.delta_Eg_eV),perturbed_Eg_eV=z.Eg_eV,
        delta_Eg_eV=z.delta_Eg_eV,protocol_match_status=pa.protocol_status if br.baseline_structure_match else 'UNRESOLVED',
        band_edge_tracking_status='SAMPLED_SPECTRUM_ACCEPTED_WITHOUT_CROSS_MESH_EDGE_TRACKING',
        numerical_error_evidence='NO_POINT_SPECIFIC_TRANSFERABLE_ERROR_BOUND',delta_Eg_uncertainty_eV=np.nan,
        uncertainty_type='UNKNOWN',resolution_status='UNRESOLVED',geometry_scope=z.scope,
        source_path=str(S),source_sha256=sha(S)))
num=pd.DataFrame(num);save(num,'NUMERICAL_RESOLUTION_MATRIX.csv')

# Seven fixed-element-table responses: per-protocol and all-fixed descriptive associations.
features=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume']
diagnostics={9136236,8158832,6235374}
formal_r=r[~r.material_id.isin(diagnostics)].copy()
items=[];stability=[]
for label,h in [('FORMAL_FIXED_13',formal_r),('ALL_FIXED_WITH_DIAGNOSTIC_16',r),*[(str(k),v) for k,v in formal_r.groupby('protocol_class')]]:
    for f in features:
        x=h[f].to_numpy(float);y=h.delta_Eg_eV.to_numpy(float)
        sr=stat(x,y,spearmanr);pr=stat(x,y,pearsonr);kr=stat(x,y,kendalltau)
        loo=[]
        for i in range(len(h)):
            h2=h.drop(h.index[i]);loo.append(stat(h2[f],h2.delta_Eg_eV,spearmanr))
            if label=='FORMAL_FIXED_13':stability.append(dict(descriptor=f,excluded_material_id=h.iloc[i].material_id,spearman_rho=loo[-1],n_materials=len(h2),status='PROVISIONAL'))
        items.append(dict(protocol_group=label,descriptor=f,n_materials=len(h),spearman_rho=sr,kendall_tau=kr,pearson_r=pr,
            abs_change_abs_response_spearman=stat(abs(x),abs(y),spearmanr),leave_one_min=np.nanmin(loo) if np.isfinite(loo).any() else np.nan,
            leave_one_max=np.nanmax(loo) if np.isfinite(loo).any() else np.nan,
            sign_stable=bool(np.isfinite(sr) and all(np.sign(v)==np.sign(sr) for v in loo if np.isfinite(v))),
            resolution_status='UNRESOLVED',scientific_grade='EXPLORATORY',status='PROVISIONAL'))
save(pd.DataFrame(items),'ELEMENT_IMPORTANCE_FINAL.csv')
save(pd.DataFrame(stability),'ELEMENT_IMPORTANCE_STABILITY.csv')
corr=formal_r[features].corr(method='spearman').reset_index().rename(columns={'index':'descriptor'})
save(corr,'ELEMENT_DESCRIPTOR_COLLINEARITY.csv')

# Material is the statistical unit; no 371-point pseudoreplication.
glob=g.copy();glob['resolution_status']='UNRESOLVED';glob['scientific_grade']='EXPLORATORY'
save(glob,'GLOBAL_L_THETA_MATERIAL_RESULTS.csv')
aggreg=[]
for f,h in glob.groupby('descriptor'):
    vals=h.slope.to_numpy(float);good=vals[np.isfinite(vals)]
    aggreg.append(dict(descriptor=f,n_materials=len(h),fitted_materials=len(good),median_slope=float(np.median(good)) if len(good) else np.nan,
        median_absolute_slope=float(np.median(abs(good))) if len(good) else np.nan,
        slope_unit=h.slope_unit.iloc[0],independent_importance_status='NOT_IDENTIFIABLE_FROM_COUPLED_GEOMETRY',
        numerical_resolution_status='UNRESOLVED',scientific_grade='EXPLORATORY'))
save(pd.DataFrame(aggreg),'GLOBAL_L_THETA_IMPORTANCE.csv')

loc=l.copy();loc['scientific_grade']='EXPLORATORY';loc['resolution_status']='UNRESOLVED'
save(loc[loc.perturbation_type=='LENGTH'],'LOCAL_BOND_LENGTH_IMPORTANCE.csv')
save(loc[loc.perturbation_type=='ANGLE'],'LOCAL_BOND_ANGLE_IMPORTANCE.csv')
save(loc,'LOCAL_GEOMETRY_MATERIAL_SUMMARY.csv')

# Same-point local/global response. A near-zero global denominator is explicitly undefined.
formal=s[(s.scope=='FORMAL') & (~s.first_shell_changed.astype(bool))].copy()
comp=[]
for _,z in formal.iterrows():
    if z.perturbation_type=='LENGTH':lx,gx=z.actual_delta_bond_A,z.delta_L_A;unit='angstrom'
    else:lx,gx=z.actual_delta_angle_degree,z.delta_Theta_degree;unit='degree'
    comp.append(dict(material_id=z.material_id,point_id=z.point_id,path=z.perturbation_type,
        local_delta=lx,global_delta=gx,unit=unit,global_to_local_ratio=gx/lx if abs(lx)>1e-12 else np.nan,
        direction_agrees=bool(np.sign(lx)==np.sign(gx)) if abs(lx)>1e-12 and abs(gx)>1e-12 else np.nan,
        delta_Eg_eV=z.delta_Eg_eV,resolution_status='UNRESOLVED',analysis_status='PROVISIONAL',source_path=str(S),source_sha256=sha(S)))
save(pd.DataFrame(comp),'LOCAL_GLOBAL_GEOMETRY_COMPARISON.csv')

# Cross-scale eligibility retains protocol and numerical limitations.
merged=r.merge(cross,on='material_id',how='left').merge(g.pivot(index='material_id',columns='descriptor',values='slope').reset_index(),on='material_id',how='left')
finite=l.pivot(index='material_id',columns='perturbation_type',values='mean_absolute_bidirectional_response').reset_index().rename(columns={'LENGTH':'finite_length_response_eV','ANGLE':'finite_angle_response_eV'})
merged=merged.merge(finite,on='material_id',how='left')
merged['cross_scale_qualification']=np.where(merged.protocol_compatible,'MATCHED_INPUTS_EXPLORATORY','PROTOCOL_LIMITED_OR_UNRESOLVED')
merged['resolution_status']='UNRESOLVED';merged['scientific_grade']='EXPLORATORY'
merged=merged[~merged.material_id.isin(diagnostics)].copy()
ld=pd.read_csv(R3/'LEGACY_ELEMENT_DESCRIPTOR_DELTA.csv').pivot(index='material_id',columns='descriptor',values='delta').reset_index()
legacy=e[e.protocol_class.str.contains('LEGACY')].merge(ld,left_on='current_material_id',right_on='material_id',how='left')
legacy=legacy.merge(g.pivot(index='material_id',columns='descriptor',values='slope').reset_index(),on='material_id',how='left').merge(finite,on='material_id',how='left')
legacy['protocol_compatible']=False;legacy['same_parent_structure']=False
legacy['cross_scale_qualification']='LEGACY_SEPARATELY_RELAXED_NOT_DIRECTLY_COMPARABLE'
legacy['resolution_status']='UNRESOLVED';legacy['scientific_grade']='EXPLORATORY'
merged=pd.concat([merged,legacy],ignore_index=True,sort=False)
assert len(merged)==17 and merged.material_id.nunique()==17
save(merged,'CROSS_SCALE_IMPORTANCE_EVIDENCE.csv')

matrix=s[['material_id','point_id','scope','first_shell_changed','result_status','formal_importance_status','delta_Eg_eV']].copy()
matrix['numerical_resolution_status']='UNRESOLVED'
matrix['analysis_eligibility']=np.where(matrix.scope=='DIAGNOSTIC','DIAGNOSTIC_ONLY',np.where(matrix.first_shell_changed,'HOLD_SHELL_SWITCH','PROVISIONAL_TREND_ELIGIBLE'))
matrix['scientific_grade']=np.where(matrix.scope=='DIAGNOSTIC','SUPPORTED_QUALITATIVE','EXPLORATORY')
erows=pd.DataFrame(dict(material_id=e.current_material_id,point_id=e.pair_id,scope=np.where(e.current_material_id.isin(diagnostics),'DIAGNOSTIC','FORMAL'),
    first_shell_changed=False,result_status=e.pair_status,formal_importance_status='PENDING_NUMERICAL_RESOLUTION',delta_Eg_eV=e.delta_Eg_eV,
    numerical_resolution_status='UNRESOLVED',analysis_eligibility=np.where(e.current_material_id.isin(diagnostics),'DIAGNOSTIC_ONLY','PROVISIONAL_TREND_ELIGIBLE'),
    scientific_grade=np.where(e.current_material_id.isin(diagnostics),'SUPPORTED_QUALITATIVE','EXPLORATORY')))
matrix['data_family']='STRUCTURE';erows['data_family']='ELEMENT'
matrix=pd.concat([matrix,erows],ignore_index=True)
save(matrix,'NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv')

# Candidate calculations are conditional: parent+point controls on one frozen branch.
plan=[]
small=formal.assign(abs_response=formal.delta_Eg_eV.abs()).sort_values('abs_response')
for _,z in small.groupby('material_id',sort=False).head(1).head(4).iterrows():
    plan.append(dict(priority=len(plan)+1,material_id=z.material_id,point_id=z.point_id,existing_job=z.job_id,
        observed_delta_Eg_eV=z.delta_Eg_eV,evidence_gap='PAIR_SPECIFIC_BAND_EDGE_AND_MESH_RESPONSE_UNCERTAINTY',
        proposed_check='MATCHED_PARENT_AND_POINT_DENSER_K_MESH_WITH_FROZEN_SCIENTIFIC_SETTINGS',
        minimum_new_DFT_calls_if_selected=2,estimated_core_hours='UNKNOWN_NO_MATCHED_RUNTIME_ESTIMATE',
        maximum_reserved_core_hours_if_48MPI_24h_each=2304,
        pilot_selection='FIRST_PILOT' if len(plan)==0 else 'CONDITIONAL_AFTER_FIRST_PILOT',
        stop_condition='STOP_IF_SCF_OR_SPECTRUM_FAILS; DO_NOT_PROMOTE_IF_EDGE_SWITCH_UNRESOLVED',
        authorization='NOT_AUTHORIZED_NOT_SUBMITTED'))
save(pd.DataFrame(plan),'MINIMAL_DFT_VALIDATION_PLAN.csv')

env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,seed='none_deterministic')
(ROOT/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
print('element',len(e),'structure',len(s),'formal_unshifted',len(formal),'cross',len(merged),'plan',len(plan))
