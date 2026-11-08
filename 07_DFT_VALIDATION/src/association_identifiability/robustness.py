"""Deterministic material-level resampling and coupled-geometry diagnostics."""

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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P=Path(str(_repo_path('07_DFT_VALIDATION/workflows/association_identifiability')))
R3=Path(str(_repo_path('07_DFT_VALIDATION/workflows/association_screening')))
e=pd.read_csv(R3/'ELEMENT_RESPONSE_INPUT.csv')
e=e[~e.material_id.isin([9136236,8158832,6235374])].copy()
s=pd.read_csv(R3/'GLOBAL_GEOMETRY_RESPONSE_INPUT.csv')
l=pd.read_csv(P/'LOCAL_GEOMETRY_MATERIAL_SUMMARY.csv')
rng=np.random.default_rng(20261030)
features=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume']

def rho(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return np.nan
    return float(spearmanr(x,y).statistic)

boot=[]
for f in features:
    x=e[f].to_numpy(float);y=e.delta_Eg_eV.to_numpy(float)
    vals=[]
    for _ in range(2000):
        ix=rng.integers(0,len(e),len(e));vals.append(rho(x[ix],y[ix]))
    v=np.asarray(vals);v=v[np.isfinite(v)]
    boot.append(dict(descriptor=f,n_materials=len(e),n_bootstrap=2000,valid_resamples=len(v),rho_q025=np.quantile(v,.025),rho_q50=np.quantile(v,.5),rho_q975=np.quantile(v,.975),
        note='material_resampling_sampling_variability_only;not_DFT_numerical_error',status='PROVISIONAL'))
pd.DataFrame(boot).to_csv(P/'ELEMENT_MATERIAL_BOOTSTRAP.csv',index=False,encoding='utf-8-sig')

# Robustness of local slopes against removal of one observed perturbation.
rows=[]
for (mid,typ),h in s.groupby(['material_id','perturbation_type']):
    col='actual_delta_bond_A' if typ=='LENGTH' else 'actual_delta_angle_degree'
    xx=h[col].to_numpy(float);yy=h.delta_Eg_eV.to_numpy(float)
    vals=[]
    for i in range(len(h)):
        mask=np.arange(len(h))!=i
        vals.append(float(np.polyfit(xx[mask],yy[mask],1)[0]) if mask.sum()>=3 and np.ptp(xx[mask])>1e-12 else np.nan)
    vals=np.asarray(vals);good=vals[np.isfinite(vals)]
    full=l[(l.material_id==mid)&(l.perturbation_type==typ)].iloc[0]
    rows.append(dict(material_id=mid,path=typ,n_points=len(h),full_slope=full.slope,leave_one_min=np.min(good) if len(good) else np.nan,leave_one_max=np.max(good) if len(good) else np.nan,
        slope_sign_stable=bool(len(good) and all(np.sign(good)==np.sign(full.slope))),status='PROVISIONAL'))
pd.DataFrame(rows).to_csv(P/'LOCAL_POINT_REMOVAL_SENSITIVITY.csv',index=False,encoding='utf-8-sig')

# Joint within-material linear terms. Report rank and conditioning before coefficients.
joint=[]
for mid,h in s.groupby('material_id'):
    x=h[['delta_L_A','delta_Theta_degree']].to_numpy(float);y=h.delta_Eg_eV.to_numpy(float)
    scale=np.std(x,axis=0)
    safe=np.where(scale>0,scale,1)
    z=(x-x.mean(axis=0))/safe
    rank=int(np.linalg.matrix_rank(z));cond=float(np.linalg.cond(z)) if rank==2 else np.inf
    if rank==2 and cond<100:
        beta=np.linalg.lstsq(z,y-y.mean(),rcond=None)[0]
        bL,bT=beta/safe
    else:bL=bT=np.nan
    joint.append(dict(material_id=mid,n_points=len(h),rank=rank,standardized_condition_number=cond,b_L_eV_per_A=bL,b_Theta_eV_per_degree=bT,
        independent_terms_status='DESCRIPTIVE_ONLY_NUMERICAL_RESOLUTION_UNKNOWN' if rank==2 and cond<100 else 'NOT_IDENTIFIABLE',
        status='PROVISIONAL'))
pd.DataFrame(joint).to_csv(P/'GLOBAL_JOINT_MODEL_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
print('bootstrap',len(boot),'local',len(rows),'joint',len(joint))
