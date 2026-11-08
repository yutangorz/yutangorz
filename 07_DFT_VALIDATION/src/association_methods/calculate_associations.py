"""Run nine-descriptor exploratory/strict analysis on frozen or user-supplied CSVs."""
from __future__ import annotations

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/association_methods')))
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from importance.core import (FEATURES,DIAGNOSTIC,digest,validate_element,validate_structure,eligible_element,eligible_structure,elemental_results,geometry_results,corr)

def load_config(path):
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    root=(path.parent/cfg['project_root']).resolve()
    return cfg,root,{k:(root/v).resolve() for k,v in cfg['inputs'].items()}

def load_inputs(paths,custom=False):
    a=pd.read_csv(paths['element_fixed']);legacy_delta=pd.read_csv(paths['element_legacy_delta']);matrix=pd.read_csv(paths['element_matrix']);long=pd.read_csv(paths['element_long']);s=pd.read_csv(paths['structure']);cross=pd.read_csv(paths['cross_scale'])
    main=matrix[matrix.protocol_class.str.contains('LEGACY')==False].copy()
    if len(a)!=16 or len(main)!=16:raise ValueError('expected 16 mapped fixed pairs')
    if set(a.material_id)!=set(main.current_material_id):raise ValueError('fixed pair ID mismatch')
    a['analysis_group']=np.where(a.material_id.isin(DIAGNOSTIC),'DIAGNOSTIC_FIXED','FORMAL_FIXED')
    a['numerical_resolution_status']='UNRESOLVED';a['descriptor_unit']='run019_element_table'
    legacy=matrix[matrix.protocol_class.str.contains('LEGACY')].copy()
    wide=legacy_delta.pivot(index='material_id',columns='descriptor',values='delta').reset_index()
    if len(wide)!=4 or set(wide.material_id)!=set(legacy.current_material_id):raise ValueError('legacy descriptor ID mismatch')
    lv=legacy.merge(wide,left_on='current_material_id',right_on='material_id',validate='one_to_one',suffixes=('','_delta'))
    lv=lv.rename(columns={'current_material_id':'material_id_main'})
    lv['material_id']=lv.material_id_main;lv['mapping_status']='SITE_MAPPING_VERIFIED';lv['analysis_group']='LEGACY_RELAXED';lv['numerical_resolution_status']='UNRESOLVED';lv['descriptor_unit']='run019_element_table'
    supplementary=long[long.is_main_display_pair.astype(str).str.lower()=='false'].copy()
    if len(supplementary)!=2:raise ValueError('supplementary pair count mismatch')
    suppl=supplementary[['current_material_id','pair_id','protocol','delta_Eg_eV']].rename(columns={'current_material_id':'material_id','protocol':'protocol_class'})
    for f in FEATURES:suppl[f]=np.nan
    suppl['mapping_status']='UNVERIFIED';suppl['analysis_group']='SUPPLEMENTARY_UNMAPPED';suppl['numerical_resolution_status']='UNRESOLVED';suppl['descriptor_unit']='run019_element_table'
    element=pd.concat([a,lv,suppl],ignore_index=True,sort=False)
    validate_element(element);validate_structure(s,440)
    if len(element)!=22 or element.pair_id.duplicated().any():raise ValueError('22 pair uniqueness failed')
    if len(cross)!=17 or cross.material_id.nunique()!=17:raise ValueError('17 common-material identity failed')
    return element,s,cross

def save(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def cross_results(element,structure,cross,mode):
    # Existing audited run004 cross table already resolves parent and protocol identity.
    z=cross.copy()
    z['analysis_mode']=mode
    z['quantitative_cross_scale_eligible']=False
    z['eligibility_reason']=np.where(z.protocol_compatible.astype(bool),'MATCHED_INPUTS_BUT_NUMERICAL_RESOLUTION_UNRESOLVED','PROTOCOL_OR_PARENT_INCOMPATIBLE')
    if mode=='strict':z['eligibility_reason']='NO_CERTIFIED_PAIR_NUMERICAL_RESOLUTION'
    return z

def evidence_matrix(element,structure,mode):
    er=element[['material_id','pair_id','analysis_group','protocol_class','numerical_resolution_status']].copy().rename(columns={'pair_id':'point_id','analysis_group':'scope'})
    er['data_family']='ELEMENT';er['algorithm_status']='ALGORITHM_VALIDATED';er['data_identity_status']=np.where(er.scope=='SUPPLEMENTARY_UNMAPPED','DESCRIPTOR_MAPPING_UNVERIFIED','DATA_IDENTITY_VALIDATED')
    sr=structure[['material_id','point_id','scope','first_shell_changed']].copy();sr['data_family']='STRUCTURE';sr['numerical_resolution_status']='UNRESOLVED';sr['algorithm_status']='ALGORITHM_VALIDATED';sr['data_identity_status']='DATA_IDENTITY_VALIDATED'
    sr['scope']=np.where(sr.scope=='DIAGNOSTIC','DIAGNOSTIC',np.where(sr.first_shell_changed,'FORMAL_SHELL_SWITCH','FORMAL'))
    er['first_shell_changed']=False;sr['protocol_class']='MATCHED_INPUTS_RUN004'
    m=pd.concat([er,sr],ignore_index=True,sort=False)
    m['analysis_mode']=mode
    m['scientific_importance_status']=np.where(m.scope.isin(['FORMAL','FORMAL_FIXED']),'SCIENTIFIC_IMPORTANCE_PROVISIONAL','EXCLUDED_FROM_FORMAL_IMPORTANCE')
    m['strict_eligible']=False
    return m

def error_sensitivity(element,structure,config):
    """Shared parent errors within material; hypotheses, not empirical DFT error bounds."""
    e=eligible_element(element);s=eligible_structure(structure)
    rng=np.random.default_rng(int(config['seed']));out=[];draws=int(config['bootstrap_draws'])
    ex=e[list(FEATURES)].to_numpy(float);ey=e.delta_Eg_eV.to_numpy(float)
    for sigma in config['error_sigma_eV']:
        for rho in config['error_correlation']:
            element_sign=[];point_flips=[];slope_sign=[];global_sign=[]
            for _ in range(draws):
                yp=ey+(rho-1)*rng.normal(0,sigma,len(e))+np.sqrt(1-rho*rho)*rng.normal(0,sigma,len(e))
                element_sign.append(sum(np.sign(corr(ex[:,j],yp,'spearman'))==np.sign(corr(ex[:,j],ey,'spearman')) for j in range(7)))
                flips=0;stable=0;global_stable=0
                for _,g in s.groupby('material_id'):
                    parent=rng.normal(0,sigma);noise=(rho-1)*parent+np.sqrt(1-rho*rho)*rng.normal(0,sigma,len(g));yy=g.delta_Eg_eV.to_numpy(float);yp2=yy+noise
                    flips+=int(np.sum(np.sign(yp2)!=np.sign(yy)))
                    for col in ['delta_L_A','delta_Theta_degree']:
                        xx=g[col].to_numpy(float)
                        global_stable+=int(np.sign(np.polyfit(xx,yp2,1)[0])==np.sign(np.polyfit(xx,yy,1)[0]))
                    for typ,h in g.groupby('perturbation_type'):
                        col='actual_delta_bond_A' if typ=='LENGTH' else 'actual_delta_angle_degree';x=h[col].to_numpy(float);inds=g.index.get_indexer(h.index)
                        stable+=int(np.sign(np.polyfit(x,yp2[inds],1)[0])==np.sign(np.polyfit(x,yy[inds],1)[0]))
                point_flips.append(flips);slope_sign.append(stable);global_sign.append(global_stable)
            out.append(dict(assumed_single_gap_sigma_eV=sigma,parent_point_error_correlation=rho,paired_error_sd_eV=sigma*np.sqrt(2*(1-rho)),draws=draws,element_feature_signs_preserved_median=np.median(element_sign),structure_point_sign_flips_median=np.median(point_flips),local_path_slope_signs_preserved_median=np.median(slope_sign),global_L_Theta_slopes_preserved_median=np.median(global_sign),analysis_unit='13_MATERIAL_ELEMENT_17_MATERIAL_STRUCTURE',status='HYPOTHETICAL_NOT_NUMERICAL_CERTIFICATION'))
    return pd.DataFrame(out)

def regression_against_run004(paths,eres,gmat,local):
    olde=pd.read_csv(paths['run004_element']);oldg=pd.read_csv(paths['run004_global']);oldl=pd.concat([pd.read_csv(paths['run004_local_length']),pd.read_csv(paths['run004_local_angle'])])
    comparisons=[]
    for _,r in eres[eres.protocol_group=='FORMAL_FIXED_13'].iterrows():
        x=olde[(olde.protocol_group=='FORMAL_FIXED_13')&(olde.descriptor==r.descriptor)].iloc[0]
        for col in ['pearson_r','spearman_rho','kendall_tau']:comparisons.append(dict(family='ELEMENT',key=r.descriptor,metric=col,run006=r[col],run004=x[col],difference=abs(r[col]-x[col]),tolerance=1e-10))
    for _,r in gmat.iterrows():
        x=oldg[(oldg.material_id==r.material_id)&(oldg.descriptor==r.descriptor)].iloc[0]
        for col in ['slope','intercept','r2']:comparisons.append(dict(family='GLOBAL',key=f'{r.material_id}_{r.descriptor}',metric=col,run006=r[col],run004=x[col],difference=abs(r[col]-x[col]),tolerance=1e-8))
    for _,r in local.iterrows():
        x=oldl[(oldl.material_id==r.material_id)&(oldl.perturbation_type==r.perturbation_type)].iloc[0]
        for col in ['slope','quadratic_coefficient','mean_absolute_bidirectional_response','central_finite_response']:comparisons.append(dict(family='LOCAL',key=f'{r.material_id}_{r.perturbation_type}',metric=col,run006=r[col],run004=x[col],difference=abs(r[col]-x[col]),tolerance=1e-8))
    q=pd.DataFrame(comparisons);q['pass']=q.difference<=q.tolerance
    return q

def run(config_path,mode,output_override=None):
    cfg,root,paths=load_config(config_path);out=Path(output_override).resolve() if output_override else (root/cfg['output_dir']/(mode+'_results')).resolve();out.mkdir(parents=True,exist_ok=True)
    if cfg.get('verify_frozen_inputs',False):
        freeze=pd.read_csv(root/cfg['frozen_manifest'])
        expected={str((root/row.path).resolve()):row.sha256 for row in freeze.itertuples()}
        for key in ['element_fixed','element_matrix','structure']:
            actual=digest(paths[key]);target=expected.get(str(paths[key]))
            if target is None or actual!=target:raise ValueError(f'frozen input SHA mismatch: {key}')
    element,structure,cross=load_inputs(paths)
    inputs=pd.DataFrame([dict(input=k,path=str(v),sha256=digest(v),bytes=v.stat().st_size) for k,v in paths.items()])
    save(inputs,out/'INPUT_SHA256.csv')
    e=eligible_element(element,mode);s=eligible_structure(structure,mode)
    save(element,out/'ELEMENT_ALL_GROUPS.csv');save(structure[['material_id','point_id','scope','first_shell_changed','delta_Eg_eV']],out/'STRUCTURE_STATUS.csv')
    save(evidence_matrix(element,structure,mode),out/'NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv')
    save(cross_results(element,structure,cross,mode),out/'CROSS_SCALE_MATERIAL_RESULTS.csv')
    if e.empty or s.empty:
        pd.DataFrame(columns=['protocol_group','descriptor','n_materials','scientific_grade']).to_csv(out/'ELEMENT_IMPORTANCE_RESULTS.csv',index=False)
        pd.DataFrame(columns=['descriptor','n_materials','status']).to_csv(out/'GLOBAL_IMPORTANCE_RESULTS.csv',index=False)
        for name in ['LOCAL_LENGTH_IMPORTANCE.csv','LOCAL_ANGLE_IMPORTANCE.csv']:
            pd.DataFrame(columns=['material_id','perturbation_type','status']).to_csv(out/name,index=False)
        (out/'STRICT_MODE_RESULTS_SUMMARY.md').write_text(f'# STRICT_SCIENTIFIC_ELIGIBILITY\n\n正式元素{len(e)}对、结构{s.shape[0]}点；逐对DFT数值误差均未认证，结果表按科学门留空。算法和数据身份状态仍独立保留。\n',encoding='utf-8')
        return out
    er,loo,reg,corrmat=elemental_results(element,int(cfg['bootstrap_draws']),int(cfg['seed']),cfg['ridge_alphas'])
    gs,gm,joint,local,finite,comp,fe=geometry_results(structure,int(cfg['bootstrap_draws']),int(cfg['seed']))
    for name,data in [('ELEMENT_IMPORTANCE_RESULTS.csv',er),('ELEMENT_CORRELATION_MATRIX.csv',corrmat),('ELEMENT_REGRESSION_RESULTS.csv',reg),('ELEMENT_IMPORTANCE_STABILITY.csv',loo),('GLOBAL_IMPORTANCE_RESULTS.csv',gs),('GLOBAL_MATERIAL_SENSITIVITY.csv',gm),('GLOBAL_COUPLING_DIAGNOSTIC.csv',joint),('GLOBAL_FIXED_EFFECTS.csv',fe),('LOCAL_LENGTH_IMPORTANCE.csv',local[local.perturbation_type=='LENGTH']),('LOCAL_ANGLE_IMPORTANCE.csv',local[local.perturbation_type=='ANGLE']),('LOCAL_FINITE_RESPONSE.csv',finite),('LOCAL_GLOBAL_COMPARISON.csv',comp),('NUMERICAL_ERROR_SENSITIVITY.csv',error_sensitivity(element,structure,cfg)),('REGRESSION_AGAINST_RUN004.csv',regression_against_run004(paths,er,gm,local))]:save(data,out/name)
    rg=regression_against_run004(paths,er,gm,local)
    if not rg['pass'].all():raise AssertionError('run004 deterministic regression mismatch; see CSV')
    (out/'EXPLORATORY_RESULTS_SUMMARY.md').write_text(f'# EXPLORATORY_ALL_VALID_INPUTS\n\n13正式固定元素材料、3诊断固定、4历史独立弛豫、2补充未映射；17正式结构材料371非壳层切换点，3正式切换点与66诊断点单列。七属性7行主统计、全局2行汇总、局部34材料路径均完成。run004确定性回归{len(rg)}/{len(rg)}通过。算法已验证不等于数值误差认证；SCIENTIFIC_IMPORTANCE_PROVISIONAL。\n',encoding='utf-8')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(_repo_path('07_DFT_VALIDATION/src/association_methods/calculate_associations.py').with_name('config.yaml')));p.add_argument('--mode',choices=['exploratory','strict'],required=True);p.add_argument('--output-dir');a=p.parse_args()
    print(run(a.config,a.mode,a.output_dir))
