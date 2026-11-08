"""Exploratory run003 statistics from frozen run019/run097 and audited run002."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/association_screening')))
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from pymatgen.io.vasp import Poscar
import sys
sys.path.insert(0,str(_repo_path('05_CRISP_X/src/local_reconstruction')))
from shared_features.extraction import ConceptExtractor, ATOMIC

o=Path(str(_repo_path('07_DFT_VALIDATION/workflows/association_screening')));o.mkdir(exist_ok=True)
r2root=Path(str(_repo_path('07_DFT_VALIDATION/workflows/verified_descriptors')))
e=pd.read_csv(r2root/'ELEMENT_DESCRIPTOR_DELTA_VERIFIED.csv')
p=pd.read_csv(str(_repo_path('07_DFT_VALIDATION/data/material_selection/FINAL_ELEMENT_SUBSTITUTION_MATRIX_DESIGN_ONLY.csv')))
s=pd.read_csv(str(_repo_path('07_DFT_VALIDATION/workflows/geometry_qualification/UNIFIED_440_RESULT_SET.csv')))
site=pd.read_csv(r2root/'ELEMENT_SITE_MAPPING_AUDIT.csv')
b=pd.read_csv(r2root/'STRUCTURE_BASELINE_JOB_RECOVERY_v2.csv')

def save(df,name): df.to_csv(o/name,index=False,encoding='utf-8-sig')
wide=e.pivot(index='material_id',columns='descriptor',values='delta').reset_index()
ep=wide.merge(p[['current_material_id','pair_id','protocol_class','delta_Eg_eV','grade']],left_on='material_id',right_on='current_material_id').merge(site[['material_id','replaced_fraction','mapping_status']],on='material_id')
save(ep,'ELEMENT_RESPONSE_INPUT.csv')
legacy=pd.read_csv(str(_repo_path('07_DFT_VALIDATION/workflows/relaxed_substitution/LEGACY_RELAXED_PAIR_MATRIX.csv')))
extractor=ConceptExtractor(str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv')))
legacyrows=[];legacydelta=[]
legacyroot=Path(str(_repo_path('07_DFT_VALIDATION/workflows/relaxed_substitution')))
for _,z in legacy.iterrows():
    ap=legacyroot/z.parent_structure;bp=legacyroot/z.sub_structure
    sa=Poscar.from_file(ap).structure;sb=Poscar.from_file(bp).structure
    va,_=extractor.extract(sa);vb,_=extractor.extract(sb)
    src,dst=z.operation.split('->');ca=sa.composition.get(src,0);cb=sb.composition.get(src,0)
    status='COMPOSITION_CHANGE_VERIFIED' if ca-cb==sb.composition.get(dst,0)-sa.composition.get(dst,0) else 'HOLD_MAPPING_AMBIGUOUS'
    legacyrows.append(dict(material_id=z.material_id,pair_id=z.pair_id,composition_status=status,changed_count=ca-cb,source_element=src,target_element=dst,recorded_site_indices=z.changed_sites_zero_based,site_mapping_status='HOLD_MAPPING_AMBIGUOUS',geometry_status='GEOMETRY_RELAXATION_CONFIRMED',parent_sha256=z.parent_input_SHA,sub_sha256=z.sub_input_SHA,parent_path=str(ap),sub_path=str(bp)))
    for j,f in enumerate(ATOMIC):legacydelta.append(dict(material_id=z.material_id,pair_id=z.pair_id,descriptor=f,parent_mean=va[j],sub_mean=vb[j],delta=vb[j]-va[j],analysis_scope='LEGACY_SEPARATELY_RELAXED',composition_status=status))
save(pd.DataFrame(legacyrows),'LEGACY_SITE_MAPPING_AUDIT.csv');save(pd.DataFrame(legacydelta),'LEGACY_ELEMENT_DESCRIPTOR_DELTA.csv')
imp=[];stability=[]
for group,g in [('ALL_FIXED',ep),*[(x,y) for x,y in ep.groupby('protocol_class')]]:
    for f in e.descriptor.unique():
        x=g[f].to_numpy(float);y=g.delta_Eg_eV.to_numpy(float)
        if len(g)<3 or np.std(x)==0:rho=tau=np.nan
        else:rho=spearmanr(x,y).statistic;tau=kendalltau(x,y).statistic
        imp.append(dict(protocol_group=group,descriptor=f,n_materials=len(g),spearman_rho=rho,kendall_tau=tau,absolute_response_spearman=spearmanr(abs(x),abs(y)).statistic if len(g)>=3 and np.std(abs(x)) else np.nan,numerical_resolution_status='UNRESOLVED',analysis_scope='PROVISIONAL'))
        if group=='ALL_FIXED' and len(g)>=4:
            for mid in g.material_id:
                h=g[g.material_id!=mid];v=spearmanr(h[f],h.delta_Eg_eV).statistic if h[f].nunique()>1 else np.nan
                stability.append(dict(descriptor=f,excluded_material_id=mid,spearman_rho=v,analysis_scope='PROVISIONAL'))
save(pd.DataFrame(imp),'ELEMENT_IMPORTANCE_PROVISIONAL.csv')
save(pd.DataFrame(stability),'ELEMENT_IMPORTANCE_STABILITY.csv')
save(ep[e.descriptor.unique()].corr(method='spearman').reset_index().rename(columns={'index':'descriptor'}),'ELEMENT_DESCRIPTOR_CORRELATION.csv')

formal=s[(s.scope=='FORMAL') & (s.first_shell_changed.astype(str)=='False')].copy()
formal['protocol_status']='BASELINE_STRUCTURE_VERIFIED_INPUT_FINGERPRINT_AVAILABLE'
formal['numerical_resolution_status']='UNRESOLVED'
formal['analysis_scope']='PROVISIONAL'
save(formal[['material_id','point_id','perturbation_type','delta_Eg_eV','delta_L_A','delta_Theta_degree','actual_delta_bond_A','actual_delta_angle_degree','first_shell_changed','scope','analysis_pool','protocol_status','numerical_resolution_status','analysis_scope']],'GLOBAL_GEOMETRY_RESPONSE_INPUT.csv')

def fit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)<=1e-12:return (np.nan,np.nan,np.nan)
    b,a=np.polyfit(x,y,1);pred=a+b*x
    r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum() if np.ptp(y)>0 else np.nan
    return float(b),float(a),float(r2)

glob=[];loc=[]
for mid,g in formal.groupby('material_id'):
    for typ,xcol,unit in [('L','delta_L_A','eV/angstrom'),('Theta','delta_Theta_degree','eV/degree')]:
        gg=g[np.isfinite(g[xcol]) & np.isfinite(g.delta_Eg_eV)]
        slope,intercept,r2=fit(gg[xcol],gg.delta_Eg_eV)
        glob.append(dict(material_id=mid,descriptor=typ,n_points=len(gg),slope=slope,intercept=intercept,r2=r2,slope_unit=unit,analysis_scope='PROVISIONAL',numerical_resolution_status='UNRESOLVED'))
    for typ,xcol,unit in [('LENGTH','actual_delta_bond_A','eV/angstrom'),('ANGLE','actual_delta_angle_degree','eV/degree')]:
        gg=g[g.perturbation_type==typ];x=gg[xcol].to_numpy(float);y=gg.delta_Eg_eV.to_numpy(float)
        slope,intercept,r2=fit(x,y)
        pos=x>0;neg=x<0
        sp=fit(x[pos],y[pos])[0];sn=fit(x[neg],y[neg])[0]
        design=np.column_stack([x,x*x])
        coef=np.linalg.lstsq(design,y,rcond=None)[0] if len(x)>=4 and np.linalg.matrix_rank(design)==2 else [np.nan,np.nan]
        linear0,curv=coef
        support=min(float(max(x[pos])),float(abs(min(x[neg])))) if pos.any() and neg.any() else np.nan
        predpos=float(linear0*support+curv*support**2) if np.isfinite(support) and np.isfinite(linear0) else np.nan
        predneg=float(-linear0*support+curv*support**2) if np.isfinite(support) and np.isfinite(linear0) else np.nan
        loc.append(dict(material_id=mid,perturbation_type=typ,n_points=len(gg),slope=slope,positive_slope=sp,negative_slope=sn,quadratic_coefficient=curv,r2=r2,slope_unit=unit,positive_support=float(max(x[pos])) if pos.any() else np.nan,negative_support=float(abs(min(x[neg]))) if neg.any() else np.nan,common_support_h=support,predicted_positive_response=predpos,predicted_negative_response=predneg,central_finite_response=abs(predpos-predneg)/2 if np.isfinite(predpos) and np.isfinite(predneg) else np.nan,mean_absolute_bidirectional_response=(abs(predpos)+abs(predneg))/2 if np.isfinite(predpos) and np.isfinite(predneg) else np.nan,analysis_scope='PROVISIONAL',numerical_resolution_status='UNRESOLVED'))
glob=pd.DataFrame(glob);loc=pd.DataFrame(loc)
save(glob,'MATERIAL_GLOBAL_SENSITIVITY.csv');save(glob.groupby('descriptor',as_index=False).agg(materials=('material_id','nunique'),median_abs_slope=('slope',lambda z:float(np.nanmedian(abs(z))))).assign(analysis_scope='PROVISIONAL'),'GLOBAL_STRUCTURE_IMPORTANCE.csv')
save(loc[loc.perturbation_type=='LENGTH'],'LOCAL_BOND_LENGTH_SENSITIVITY.csv');save(loc[loc.perturbation_type=='ANGLE'],'LOCAL_BOND_ANGLE_SENSITIVITY.csv')
save(loc,'LOCAL_GEOMETRY_RESPONSE_SUMMARY.csv')
cross=ep.merge(glob.pivot(index='material_id',columns='descriptor',values='slope').reset_index(),on='material_id',how='left').merge(loc.pivot(index='material_id',columns='perturbation_type',values='slope').reset_index(),on='material_id',how='left')
cross['same_parent_structure']=cross.material_id.map(lambda m: bool((b[b.material_id==m].baseline_structure_sha256==site[site.material_id==m].parent_sha256.iloc[0]).all()) if m in set(site.material_id) else False)
cross['cross_scale_status']='PROTOCOL_AND_NUMERICAL_REVIEW_REQUIRED'
save(cross,'CROSS_SCALE_MATERIAL_IMPORTANCE.csv')
raw=json.loads((r2root/'INPUT_EVIDENCE.json').read_text(encoding='utf-8'))['records']
base_raw=json.loads((r2root/'BASELINE_INPUT_EVIDENCE.json').read_text(encoding='utf-8'))
compat=[]
for _,z in ep.iterrows():
    mid=int(z.material_id);bj=str(b[b.material_id==mid].baseline_job.iloc[0]);ej=str(int(p[p.current_material_id==mid].parent_job.iloc[0]))
    er=next((v for v in raw.get(ej,[]) if v.get('poscar_text')),None)
    if ej=='12015':er=None
    sr=base_raw.get(bj,[None])[0]
    same_structure=bool(b[b.material_id==mid].baseline_structure_sha256.iloc[0]==site[site.material_id==mid].parent_sha256.iloc[0])
    same_paw=bool(er and sr and er.get('POTCAR_sha256')==sr.get('POTCAR_sha256'))
    same_k=bool(er and sr and er.get('KPOINTS_sha256')==sr.get('KPOINTS_sha256'))
    epv=er.get('actual_parameters',{}) if er else {};spi=sr.get('incar_text','') if sr else ''
    spv=dict((a.strip(),v.strip()) for line in spi.splitlines() if '=' in line for a,v in [line.split('=',1)])
    keys=['ENCUT','NBANDS','NELECT','LSORBIT','ISPIN','ISMEAR','SIGMA','EDIFF','IVDW']
    same_keys=bool(er and sr and all(str(epv.get(k,'')).lower()==str(spv.get(k,'')).lower() for k in keys))
    compat.append(dict(material_id=mid,element_parent_job=ej,structure_parent_job=bj,same_parent_structure=same_structure,same_PAW=same_paw,same_KPOINTS=same_k,same_key_INCAR=same_keys,protocol_compatible=bool(same_paw and same_k and same_keys),direct_quantitative_comparison=False,numerical_resolution_status='UNRESOLVED',element_input_source=er['directory'] if er else '07_DFT_VALIDATION/inputs/reference_geometry',structure_input_source=sr['directory'] if sr else ''))
save(pd.DataFrame(compat),'CROSS_SCALE_PROTOCOL_COMPATIBILITY.csv')
q=s[['material_id','point_id','scope','first_shell_changed','formal_importance_status','result_status']].copy();q['analysis_eligibility']=np.where(q.scope=='DIAGNOSTIC','DIAGNOSTIC_ONLY',np.where(q.first_shell_changed.astype(str)=='True','HOLD_SHELL_CHANGE','PROVISIONAL_TREND_ELIGIBLE'));q['numerical_resolution_status']='UNRESOLVED';save(q,'NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv')
print('element',len(ep),'structure formal',len(formal),'materials',formal.material_id.nunique(),'diagnostic',sum(s.scope=='DIAGNOSTIC'),'shell switches',sum(s.first_shell_changed.astype(str)=='True'))
