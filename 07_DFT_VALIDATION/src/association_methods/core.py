"""Pure statistical functions and explicit data qualification for run006.

All response fits are descriptive; sampled-spectrum QC is not numerical resolution.
"""
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
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURES=('period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume')
DIAGNOSTIC={9136236,8158832,6235374}
UNIT={'LENGTH':'angstrom','ANGLE':'degree'}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_structure(df:pd.DataFrame, expected_count:int|None=None):
    req={'material_id','point_id','perturbation_type','actual_delta_bond_A','actual_delta_angle_degree','delta_L_A','delta_Theta_degree','delta_Eg_eV','scope','first_shell_changed','job_id'}
    missing=req-set(df.columns)
    if missing:raise ValueError(f'missing structure fields: {sorted(missing)}')
    if expected_count is not None and len(df)!=expected_count:raise ValueError('structure row count mismatch')
    if df.point_id.isna().any() or df.point_id.duplicated().any():raise ValueError('duplicate or missing point_id')
    if df.material_id.isna().any() or df.delta_Eg_eV.isna().any():raise ValueError('missing material ID or response')
    if not all(str(p).startswith(str(m)+'_') for p,m in zip(df.point_id,df.material_id)):raise ValueError('point/material identity mismatch')
    if not set(df.perturbation_type).issubset(UNIT):raise ValueError('invalid perturbation type')
    if not set(df.scope).issubset({'FORMAL','DIAGNOSTIC'}):raise ValueError('invalid scope')
    if 'local_unit' in df and any(u!=UNIT[t] for t,u in zip(df.perturbation_type,df.local_unit)):raise ValueError('invalid local unit')
    if 'protocol_status' in df and df.protocol_status.isna().any():raise ValueError('missing protocol status')
    if not np.isfinite(df[['actual_delta_bond_A','actual_delta_angle_degree','delta_L_A','delta_Theta_degree','delta_Eg_eV']].to_numpy(float)).all():raise ValueError('nonfinite geometry or response')
    for r in df.itertuples():
        if (int(r.material_id) in DIAGNOSTIC)!=(r.scope=='DIAGNOSTIC'):raise ValueError('scope/material identity mismatch')
    return df

def validate_element(df:pd.DataFrame):
    req=set(FEATURES)|{'material_id','pair_id','delta_Eg_eV','protocol_class','mapping_status','analysis_group'}
    if req-set(df.columns):raise ValueError(f'missing element fields: {sorted(req-set(df.columns))}')
    if df.pair_id.isna().any() or df.pair_id.duplicated().any():raise ValueError('duplicate or missing pair ID')
    if df.material_id.isna().any() or df.delta_Eg_eV.isna().any():raise ValueError('missing ID or response')
    if not set(df.analysis_group).issubset({'FORMAL_FIXED','DIAGNOSTIC_FIXED','LEGACY_RELAXED','SUPPLEMENTARY_UNMAPPED'}):raise ValueError('invalid analysis group')
    if not set(df.mapping_status.dropna()).issubset({'VERIFIED_SITE_MAPPING','SITE_MAPPING_VERIFIED','UNVERIFIED'}):raise ValueError('invalid mapping')
    if 'descriptor_unit' in df and not set(df.descriptor_unit.dropna()).issubset({'run019_element_table'}):raise ValueError('invalid descriptor unit')
    # Formal pool may mix declared protocols only for marginal descriptive reporting, never for pooled conditional causality.
    if any(pd.isna(df.loc[df.analysis_group!='SUPPLEMENTARY_UNMAPPED',list(FEATURES)]).to_numpy().ravel()):raise ValueError('missing qualified descriptors')
    return df

def eligible_structure(df,mode='exploratory'):
    validate_structure(df)
    mask=(df.scope=='FORMAL')&(~df.first_shell_changed.astype(bool))
    if mode=='strict':
        if 'numerical_resolution_status' not in df: return df.iloc[0:0].copy()
        mask &= df.numerical_resolution_status.eq('CERTIFIED')
    elif mode!='exploratory':raise ValueError('unknown mode')
    return df.loc[mask].copy()

def eligible_element(df,mode='exploratory'):
    validate_element(df)
    mask=df.analysis_group.eq('FORMAL_FIXED')&df.mapping_status.eq('VERIFIED_SITE_MAPPING')
    if mode=='strict':
        if 'numerical_resolution_status' not in df:return df.iloc[0:0].copy()
        mask &= df.numerical_resolution_status.eq('CERTIFIED')
    elif mode!='exploratory':raise ValueError('unknown mode')
    return df.loc[mask].copy()

def fit_line(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or np.ptp(x)<=1e-12:return {'slope':np.nan,'intercept':np.nan,'r2':np.nan,'n':len(x),'status':'NOT_IDENTIFIABLE'}
    slope,intercept=np.polyfit(x,y,1); pred=slope*x+intercept
    ss=np.sum((y-y.mean())**2);r2=1-np.sum((y-pred)**2)/ss if ss>0 else np.nan
    return {'slope':float(slope),'intercept':float(intercept),'r2':float(r2),'n':len(x),'status':'DESCRIPTIVE_ONLY'}

def fit_zero_quadratic(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<4 or np.linalg.matrix_rank(np.column_stack([x,x*x]))<2:return {'linear':np.nan,'curvature':np.nan,'status':'NOT_IDENTIFIABLE'}
    a,b=np.linalg.lstsq(np.column_stack([x,x*x]),y,rcond=None)[0]
    return {'linear':float(a),'curvature':float(b),'status':'DESCRIPTIVE_ONLY'}

def paired_finite(x,y,ids=None):
    """Closest actual +/- pair; exact nonuniform 3-point derivative through observed zero baseline."""
    x=np.asarray(x,float);y=np.asarray(y,float);pos=np.where(x>0)[0];neg=np.where(x<0)[0]
    if not len(pos) or not len(neg):return {'status':'NO_BIDIRECTIONAL_PAIR','central_derivative':np.nan,'mean_absolute_response':np.nan}
    i,j=min(((i,j) for i in neg for j in pos),key=lambda ij:(abs(abs(x[ij[0]])-x[ij[1]]),max(abs(x[ij[0]]),x[ij[1]])))
    hm=-x[i];hp=x[j]
    derivative=(hm*hm*y[j]-hp*hp*y[i])/(hm*hp*(hm+hp))
    curvature=(hm*y[j]+hp*y[i])/(hm*hp*(hm+hp))
    return {'status':'OBSERVED_NONUNIFORM_PAIR','negative_point_id':str(ids[i]) if ids is not None else str(i),'positive_point_id':str(ids[j]) if ids is not None else str(j),'h_negative':hm,'h_positive':hp,'central_derivative':float(derivative),'pair_curvature':float(curvature),'mean_absolute_response':float((abs(y[i])+abs(y[j]))/2),'central_half_difference':float(abs(y[j]-y[i])/2)}

def quadratic_finite(x,y):
    """run003/run004 zero-reference quadratic, within common observed support only."""
    x=np.asarray(x,float);y=np.asarray(y,float);p=x[x>0];n=x[x<0]
    q=fit_zero_quadratic(x,y)
    if not len(p) or not len(n) or q['status']!='DESCRIPTIVE_ONLY':return {'finite_response':np.nan,'common_support':np.nan,'central_half_difference':np.nan}
    h=min(max(p),abs(min(n)));a=q['linear'];b=q['curvature'];yp=a*h+b*h*h;ym=-a*h+b*h*h
    return {'finite_response':float((abs(yp)+abs(ym))/2),'common_support':float(h),'central_half_difference':float(abs(yp-ym)/2)}

def corr(x,y,kind):
    x=np.asarray(x,float);y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    return float({'pearson':pearsonr,'spearman':spearmanr,'kendall':kendalltau}[kind](x,y).statistic)

def bootstrap_corr(x,y,kind='spearman',draws=500,seed=20261030):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x);rng=np.random.default_rng(seed);vals=[]
    for _ in range(draws):
        ix=rng.integers(0,n,n);z=corr(x[ix],y[ix],kind)
        if np.isfinite(z):vals.append(z)
    return {'q025':float(np.quantile(vals,.025)) if vals else np.nan,'q975':float(np.quantile(vals,.975)) if vals else np.nan,'valid_draws':len(vals),'sampling_unit':'MATERIAL'}

def rank_design(x):
    x=np.asarray(x,float);sd=x.std(axis=0);z=(x-x.mean(axis=0))/np.where(sd>0,sd,1)
    return {'rank':int(np.linalg.matrix_rank(z)),'condition':float(np.linalg.cond(z)) if np.linalg.matrix_rank(z)==z.shape[1] else np.inf,'standardized':z,'scale':sd}

def fit_joint(x,y,max_condition=100):
    x=np.asarray(x,float);y=np.asarray(y,float);d=rank_design(x)
    if d['rank']!=x.shape[1] or d['condition']>=max_condition:
        return {'rank':d['rank'],'condition':d['condition'],'standardized_beta':np.full(x.shape[1],np.nan),'raw_beta':np.full(x.shape[1],np.nan),'status':'NOT_IDENTIFIABLE'}
    beta=np.linalg.lstsq(d['standardized'],y-y.mean(),rcond=None)[0]
    return {'rank':d['rank'],'condition':d['condition'],'standardized_beta':beta,'raw_beta':beta/d['scale'],'status':'DESCRIPTIVE_COUPLED'}

def ridge_train_only_cv(x,y,alphas=(.01,.1,1,10,100)):
    """LOO by material; scaler fit on training fold only."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<4: return {'status':'INSUFFICIENT_MATERIALS'}
    scores=[]
    for alpha in alphas:
        err=[]
        for i in range(len(x)):
            ix=np.arange(len(x))!=i;sc=StandardScaler().fit(x[ix]);model=Ridge(alpha=alpha).fit(sc.transform(x[ix]),y[ix]);err.append((model.predict(sc.transform(x[i:i+1]))[0]-y[i])**2)
        scores.append(float(np.mean(err)))
    best=int(np.argmin(scores));sc=StandardScaler().fit(x);model=Ridge(alpha=alphas[best]).fit(sc.transform(x),y)
    return {'alpha':float(alphas[best]),'loo_mse':scores[best],'coefficient':model.coef_.tolist(),'status':'PREDICTIVE_REGULARIZED_NOT_CAUSAL'}

def elemental_results(df,draws=500,seed=20261030,alphas=(.01,.1,1,10,100)):
    validate_element(df);out=[];loo=[];reg=[]
    groups={'FORMAL_FIXED_13':df[df.analysis_group=='FORMAL_FIXED'],'ALL_FIXED_DIAGNOSTIC_16':df[df.analysis_group.isin(['FORMAL_FIXED','DIAGNOSTIC_FIXED'])]}
    groups.update({f'PROTOCOL_{p}':g for p,g in groups['FORMAL_FIXED_13'].groupby('protocol_class')})
    for label,g in groups.items():
        y=g.delta_Eg_eV.to_numpy(float)
        for f in FEATURES:
            x=g[f].to_numpy(float);b=bootstrap_corr(x,y,'spearman',draws,seed) if len(g)>=4 else {'q025':np.nan,'q975':np.nan,'valid_draws':0}
            pr=bootstrap_corr(x,y,'pearson',draws,seed) if len(g)>=4 else {'q025':np.nan,'q975':np.nan,'valid_draws':0}
            out.append(dict(protocol_group=label,descriptor=f,n_materials=len(g),pearson_r=corr(x,y,'pearson'),pearson_ci_low=pr['q025'],pearson_ci_high=pr['q975'],spearman_rho=corr(x,y,'spearman'),spearman_ci_low=b['q025'],spearman_ci_high=b['q975'],kendall_tau=corr(x,y,'kendall'),same_sign_fraction=float(np.mean(np.sign(x)==np.sign(y))),n_boot_valid=b['valid_draws'],numerical_resolution_status='UNRESOLVED',scientific_grade='EXPLORATORY'))
            if label=='FORMAL_FIXED_13':
                for i,m in enumerate(g.material_id):
                    ix=np.arange(len(g))!=i;loo.append(dict(descriptor=f,excluded_material_id=m,spearman_rho=corr(x[ix],y[ix],'spearman'),n_materials=sum(ix)))
    formal=groups['FORMAL_FIXED_13'];x=formal[list(FEATURES)].to_numpy(float);y=formal.delta_Eg_eV.to_numpy(float)
    d=rank_design(x);rr=ridge_train_only_cv(x,y,alphas)
    if len(y)>len(FEATURES)+1 and d['rank']==len(FEATURES) and d['condition']<100:
        ols=np.linalg.lstsq(np.column_stack([np.ones(len(y)),d['standardized']]),y,rcond=None)[0][1:]
        ols_status='CONDITIONAL_ASSOCIATION_ONLY'
    else:ols=np.full(len(FEATURES),np.nan);ols_status='NOT_IDENTIFIABLE'
    # Coefficient bootstrap samples whole materials, holding selected alpha fixed.
    rng=np.random.default_rng(seed);coef=[]
    for _ in range(draws):
        ix=rng.integers(0,len(y),len(y));sc=StandardScaler().fit(x[ix]);
        coef.append(Ridge(alpha=rr['alpha']).fit(sc.transform(x[ix]),y[ix]).coef_)
    coef=np.asarray(coef)
    mixed=formal.protocol_class.nunique()>1
    for j,f in enumerate(FEATURES):reg.append(dict(descriptor=f,n_materials=len(y),design_rank=d['rank'],design_condition=d['condition'],standardized_ols_coefficient_eV=ols[j],ols_status='MIXED_PROTOCOL_'+ols_status if mixed else ols_status,ridge_alpha=rr.get('alpha',np.nan),ridge_standardized_coefficient_eV=rr.get('coefficient',[np.nan]*7)[j],ridge_bootstrap_q025_eV=np.quantile(coef[:,j],.025),ridge_bootstrap_q975_eV=np.quantile(coef[:,j],.975),ridge_sign_stability_fraction=np.mean(np.sign(coef[:,j])==np.sign(rr['coefficient'][j])),ridge_loo_mse=rr.get('loo_mse',np.nan),ridge_status='MIXED_PROTOCOL_PREDICTIVE_ONLY' if mixed else rr['status'],scientific_grade='EXPLORATORY_NOT_CAUSAL'))
    corrmat=formal[list(FEATURES)].corr(method='spearman').reset_index().rename(columns={'index':'descriptor'})
    return pd.DataFrame(out),pd.DataFrame(loo),pd.DataFrame(reg),corrmat

def geometry_results(df,draws=500,seed=20261030):
    validate_structure(df);formal=eligible_structure(df);grows=[];jrows=[];lrows=[];frows=[];comp=[]
    for mid,g in formal.groupby('material_id'):
        y=g.delta_Eg_eV.to_numpy(float)
        for desc,col,unit in [('L','delta_L_A','eV/angstrom'),('Theta','delta_Theta_degree','eV/degree')]:
            x=g[col].to_numpy(float);lin=fit_line(x,y);quad=fit_zero_quadratic(x,y);fin=quadratic_finite(x,y);sd=float(np.std(x));
            grows.append(dict(material_id=mid,descriptor=desc,n_points=len(g),slope=lin['slope'],intercept=lin['intercept'],r2=lin['r2'],slope_unit=unit,standardized_coefficient_eV=lin['slope']*sd if np.isfinite(lin['slope']) else np.nan,quadratic_curvature=quad['curvature'],positive_slope=fit_line(x[x>0],y[x>0])['slope'],negative_slope=fit_line(x[x<0],y[x<0])['slope'],finite_response_eV=fin['finite_response'],excursion=float(np.ptp(x)),status=lin['status'],numerical_resolution_status='UNRESOLVED'))
        x=g[['delta_L_A','delta_Theta_degree']].to_numpy(float);joint=fit_joint(x,y);beta=joint['standardized_beta'];raw=joint['raw_beta']
        jrows.append(dict(material_id=mid,n_points=len(g),rank=joint['rank'],condition_number=joint['condition'],standardized_beta_L_eV=beta[0],standardized_beta_Theta_eV=beta[1],raw_beta_L_eV_per_A=raw[0],raw_beta_Theta_eV_per_degree=raw[1],status=joint['status']))
        for typ,h in g.groupby('perturbation_type'):
            col='actual_delta_bond_A' if typ=='LENGTH' else 'actual_delta_angle_degree';xx=h[col].to_numpy(float);yy=h.delta_Eg_eV.to_numpy(float)
            lin=fit_line(xx,yy);quad=fit_zero_quadratic(xx,yy);fin=quadratic_finite(xx,yy);pair=paired_finite(xx,yy,h.point_id.to_numpy())
            loo=[fit_line(np.delete(xx,i),np.delete(yy,i))['slope'] for i in range(len(h))]
            lrows.append(dict(material_id=mid,perturbation_type=typ,n_points=len(h),slope=lin['slope'],intercept=lin['intercept'],r2=lin['r2'],slope_unit='eV/angstrom' if typ=='LENGTH' else 'eV/degree',positive_slope=fit_line(xx[xx>0],yy[xx>0])['slope'],negative_slope=fit_line(xx[xx<0],yy[xx<0])['slope'],quadratic_coefficient=quad['curvature'],common_support_h=fin['common_support'],mean_absolute_bidirectional_response=fin['finite_response'],central_finite_response=fin['central_half_difference'],paired_nonuniform_derivative=pair['central_derivative'],observed_pair_mean_abs_response=pair['mean_absolute_response'],negative_point_id=pair.get('negative_point_id',''),positive_point_id=pair.get('positive_point_id',''),leave_one_min=np.nanmin(loo),leave_one_max=np.nanmax(loo),slope_sign_stable=bool(all(np.sign(z)==np.sign(lin['slope']) for z in loo)),numerical_resolution_status='UNRESOLVED'))
            frows.append(dict(material_id=mid,path=typ,negative_point_id=pair.get('negative_point_id',''),positive_point_id=pair.get('positive_point_id',''),h_negative=pair.get('h_negative',np.nan),h_positive=pair.get('h_positive',np.nan),paired_derivative=pair['central_derivative'],paired_curvature=pair.get('pair_curvature',np.nan),observed_mean_absolute_response=pair['mean_absolute_response'],quadratic_finite_response=fin['finite_response'],common_support_h=fin['common_support'],status=pair['status']))
    for r in formal.itertuples():
        loc=r.actual_delta_bond_A if r.perturbation_type=='LENGTH' else r.actual_delta_angle_degree
        glob=r.delta_L_A if r.perturbation_type=='LENGTH' else r.delta_Theta_degree
        comp.append(dict(material_id=r.material_id,point_id=r.point_id,path=r.perturbation_type,local_delta=loc,global_delta=glob,unit=UNIT[r.perturbation_type],global_to_local_ratio=glob/loc if abs(loc)>1e-12 else np.nan,direction_agrees=np.sign(loc)==np.sign(glob) if abs(loc)>1e-12 and abs(glob)>1e-12 else np.nan,delta_Eg_eV=r.delta_Eg_eV,status='DESCRIPTIVE_GEOMETRY_ONLY'))
    gm=pd.DataFrame(grows);joint=pd.DataFrame(jrows);local=pd.DataFrame(lrows);finite=pd.DataFrame(frows);comp=pd.DataFrame(comp)
    summary=[]
    for desc,h in gm.groupby('descriptor'):
        vals=h.slope.to_numpy(float)
        # Material-level bootstrap, never resample 22 points as independent materials.
        rng=np.random.default_rng(seed);med=[np.median(np.abs(vals[rng.integers(0,len(vals),len(vals))])) for _ in range(draws)]
        leave=[np.median(np.abs(np.delete(vals,i))) for i in range(len(vals))]
        summary.append(dict(descriptor=desc,n_materials=len(h),median_slope=np.median(vals),median_absolute_slope=np.median(abs(vals)),bootstrap_q025=np.quantile(med,.025),bootstrap_q975=np.quantile(med,.975),leave_one_material_min=np.min(leave),leave_one_material_max=np.max(leave),slope_unit=h.slope_unit.iloc[0],independent_importance_status='NOT_IDENTIFIABLE_FROM_COUPLED_GEOMETRY',numerical_resolution_status='UNRESOLVED'))
    # Fixed-effects within transformation; cluster-robust inference is not claimed.
    z=formal[['material_id','delta_L_A','delta_Theta_degree','delta_Eg_eV']].copy()
    for col in ['delta_L_A','delta_Theta_degree','delta_Eg_eV']:z[col]=z[col]-z.groupby('material_id')[col].transform('mean')
    design=z[['delta_L_A','delta_Theta_degree']].to_numpy(float);dd=rank_design(design)
    fe=np.linalg.lstsq(design,z.delta_Eg_eV.to_numpy(float),rcond=None)[0] if dd['rank']==2 else [np.nan,np.nan]
    return pd.DataFrame(summary),gm,joint,local,finite,comp,pd.DataFrame([dict(beta_L_eV_per_A=fe[0],beta_Theta_eV_per_degree=fe[1],n_materials=formal.material_id.nunique(),n_points=len(formal),status='DESCRIPTIVE_FIXED_EFFECTS_NOT_CAUSAL')])
