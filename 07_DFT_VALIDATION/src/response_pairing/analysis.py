"""Figure 6: frozen-source feasibility/statistics only. No fitting, inference or plotting."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/response_pairing')))
from pathlib import Path
import json, hashlib, shutil, warnings, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from pymatgen.core import Structure

ROOT = _repo_path('.')
RUN = _repo_path('07_DFT_VALIDATION/workflows/response_pairing')
OUT = _repo_path('outputs/07_DFT_VALIDATION/response_pairing')
SNAP = _repo_path('07_DFT_VALIDATION/data/response_pairing')
SEED, B = 20261101, 2000
FEATURES = ['valence_electrons','electron_affinity','ionization_energy','atomic_volume',
            'electronegativity','covalent_radius','period','bond_length','bond_angle']
ATOMIC = FEATURES[:7]
DIAGNOSTIC = {9136236,8158832,6235374}  # frozen run004/run006 scope, not response selection
SOURCE = {
 'matrix':str(_repo_path('07_DFT_VALIDATION/data/material_selection/FINAL_ELEMENT_SUBSTITUTION_MATRIX_DESIGN_ONLY.csv')),
 'all_pairs':str(_repo_path('07_DFT_VALIDATION/data/material_selection/ALL_ACCEPTED_PAIRS_LONG_DESIGN_ONLY.csv')),
 'atomic':str(_repo_path('07_DFT_VALIDATION/workflows/verified_descriptors/ELEMENT_DESCRIPTOR_DELTA_VERIFIED.csv')),
 'legacy':str(_repo_path('07_DFT_VALIDATION/workflows/association_screening/LEGACY_ELEMENT_DESCRIPTOR_DELTA.csv')),
 'fixed':str(_repo_path('07_DFT_VALIDATION/workflows/association_screening/ELEMENT_RESPONSE_INPUT.csv')),
 'old_stats':str(_repo_path('07_DFT_VALIDATION/workflows/association_methods/ELEMENT_IMPORTANCE_RESULTS.csv')),
 'freeze':str(_repo_path('07_DFT_VALIDATION/workflows/association_identifiability/ANALYSIS_DATA_FREEZE.csv')),
 'structure':str(_repo_path('07_DFT_VALIDATION/workflows/geometry_qualification/UNIFIED_440_RESULT_SET.csv')),
 'manifest':str(_repo_path('07_DFT_VALIDATION/data/geometry_design/geometry_node_manifest.csv')),
 'parents':str(_repo_path('07_DFT_VALIDATION/data/geometry_design/CURRENT_20_MATERIALS.csv')),
 'protocol':str(_repo_path('07_DFT_VALIDATION/workflows/association_identifiability/STRUCTURE_PAIR_PROTOCOL_AUDIT.csv')),
 'recovery':str(_repo_path('07_DFT_VALIDATION/workflows/verified_descriptors/STRUCTURE_BASELINE_JOB_RECOVERY_v2.csv')),
 'local':str(_repo_path('07_DFT_VALIDATION/workflows/descriptor_evidence/DFT_LOCAL_PATH_RESULTS.csv')),
 'global':str(_repo_path('07_DFT_VALIDATION/workflows/association_methods/GLOBAL_MATERIAL_SENSITIVITY.csv')),
 'et':str(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ET_TREESHAP_FINAL.csv')),
 'lime':str(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/LIME_FINAL.csv')),
 'crisp':str(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_CONDITIONED_TREESHAP.csv')),
 'crisp_parents':str(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_PARENT_POINTWISE.csv')),
 'crisp_phi':str(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_DELTA_SHAP.csv')),
 'xgb':str(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/TABLE_nine_single_feature_ranking.csv')),
 'responses':str(_repo_path('outputs/05_CRISP_X/local_reconstruction/response_predictions.csv')),
 'atom_table':str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv')),
 'extractor':str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features/extraction.py')),
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(k): return pd.read_csv(SNAP / SOURCE[k])
def write(d, name): d.to_csv(OUT / name, index=False, encoding='utf-8-sig', na_rep='NA')
def corr(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return np.nan
    return float(spearmanr(x,y).statistic)
def localpath(s): return Path(str(s).replace('\\\\','/').replace('\\','/'))

def freeze():
    if OUT.exists(): raise RuntimeError('Executed run is immutable; use a new run number')
    SNAP.mkdir(parents=True,exist_ok=True)
    cfg={'seed':SEED,'bootstrap_replicates':B,'atomic_scope':'13 formal fixed; 3 diagnostic and 4 relaxed kept separate',
         'bootstrap_unit':'material; geometry paths resampled as whole material summaries',
         'b_primary':'signed Spearman; pooled protocol result explicitly exploratory; separate protocol estimates',
         'b_interval_display_gate':'valid bootstrap fraction >=0.80; otherwise suppress display CI, retain conditional interval',
         'c_primary':'formal non-shell-switch 371 points/17 parents; all440 retained in ledger',
         'geometry_nonzero_tolerance':1e-10,'geometry_tolerance_meaning':'serialization precision only, not DFT accuracy',
         'e_identity_tolerances':{'lattice_A':1e-10,'fractional_modulo_PBC':1e-10},
         'e_identity':'same species in site order, same lattice, same periodic fractional positions; no relaxation/alignment/rescaling',
         'd_crisp_primary':'mean absolute change of ET TreeSHAP phi over all 2304 points/24 parents',
         'd_dft':'atomic7 marginal abs Spearman diagnostic only; independent L/Theta importance NA',
         'prohibited':'no DFT/SLURM/LOBSTER; no model fitting/inference; no formal plots'}
    (_repo_path('07_DFT_VALIDATION/workflows/response_pairing/PROTOCOL_FROZEN.json')).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    checks=[]
    for r in pd.read_csv(ROOT/SOURCE['freeze']).itertuples():
        p=ROOT/localpath(r.path); checks.append({'path':str(p),'expected':r.sha256,'actual':sha(p),'pass':sha(p)==r.sha256})
    assert all(c['pass'] for c in checks),'Historical data freeze changed: stop'
    extra=[]
    matrix=pd.read_csv(ROOT/SOURCE['matrix']); man=pd.read_csv(ROOT/SOURCE['manifest'])
    cp=pd.read_csv(ROOT/SOURCE['crisp_parents']); ids=set(matrix.current_material_id)&set(cp.ID)
    responses=pd.read_csv(ROOT/SOURCE['responses'])
    rr=responses[(responses.model=='frozen_ET')&(responses.method=='block_b64_g40')&(responses.kind=='geometry')&responses.ID.isin(ids)]
    for r in man[man.material_id.isin(ids)].itertuples(): extra.append(localpath(r.structure_path))
    parents=pd.read_csv(ROOT/SOURCE['parents'])
    for r in parents[parents.material_id.isin(ids)].itertuples(): extra.append(localpath(r.baseline_structure))
    for r in rr.itertuples():
        p=ROOT/f'outputs/05_CRISP_X/local_reconstruction/physical_pools/{r.ID}_{r.seed}'
        extra.extend([p/f'structure_{r.candidate}.json',p/'parent.json',p/'pool.csv'])
    rows=[]
    for p in [ROOT/s for s in SOURCE.values()] + extra:
        p=p.resolve()
        if any(r['source_path']==str(p) for r in rows):continue
        rel=p.relative_to(ROOT); dest=SNAP/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        rows.append({'source_path':str(p),'snapshot_path':str(dest),'sha256':sha(p),'bytes':p.stat().st_size})
    pd.DataFrame(rows).to_csv(_repo_path('07_DFT_VALIDATION/workflows/response_pairing/INPUT_SOURCE_MAPPING.csv'),index=False,encoding='utf-8-sig')
    (_repo_path('07_DFT_VALIDATION/workflows/response_pairing/EXECUTION_FREEZE.json')).write_text(json.dumps({'script_sha256':sha(__file__),
       'protocol_sha256':sha(_repo_path('07_DFT_VALIDATION/workflows/response_pairing/PROTOCOL_FROZEN.json')),'historical_freeze_checks':checks},indent=2),encoding='utf-8')
    OUT.mkdir()

def b_analysis():
    matrix=read('matrix').rename(columns={'current_material_id':'material_id'})
    atom=pd.concat([read('atomic'),read('legacy')],ignore_index=True)
    assert len(matrix)==20 and matrix.material_id.nunique()==20 and len(atom)==140
    atom=atom.merge(matrix,on=['material_id','pair_id'],validate='many_to_one',suffixes=('','_matrix'))
    fixed=set(read('fixed').material_id)
    atom['analysis_scope']=atom.material_id.map(lambda v:'DIAGNOSTIC_FIXED' if v in DIAGNOSTIC else 'FORMAL_FIXED_13' if v in fixed else 'LEGACY_RELAXED_4')
    atom['descriptor_unit']='frozen CGCNN table coordinate; scale not independently re-certified'
    atom.loc[atom.descriptor=='covalent_radius','descriptor_unit']='pm (source elemental table)'
    atom.loc[atom.descriptor.isin(['period','valence_electrons']),'descriptor_unit']='dimensionless table coordinate'
    atom['numerical_response_resolution']='UNRESOLVED; technical acceptance is not pairwise numerical error bound'
    assert np.max(abs(atom.sub_mean-atom.parent_mean-atom.delta))<1e-12
    assert np.max(abs(atom.Eg_sub_eV-atom.Eg_parent_eV-atom.delta_Eg_eV))<1e-12
    write(atom,'FIG6B_ELEMENT_PAIRED_LONG.csv')
    wide=atom.pivot(index='material_id',columns='descriptor',values='delta').reset_index()
    wide=wide.merge(matrix,on='material_id',validate='one_to_one')
    wide['analysis_scope']=wide.material_id.map(lambda v:'DIAGNOSTIC_FIXED' if v in DIAGNOSTIC else 'FORMAL_FIXED_13' if v in fixed else 'LEGACY_RELAXED_4')
    write(wide,'FIG6B_ELEMENT_PAIRED_WIDE.csv')
    main=wide[wide.analysis_scope=='FORMAL_FIXED_13'].copy();assert len(main)==13
    rng=np.random.default_rng(SEED); ix=rng.integers(0,len(main),(B,len(main)))
    rows=[];boots=[];loo=[]
    groups=[('FORMAL_FIXED_13',main),('DIAGNOSTIC_FIXED',wide[wide.analysis_scope=='DIAGNOSTIC_FIXED']),('LEGACY_RELAXED_4',wide[wide.analysis_scope=='LEGACY_RELAXED_4'])]
    groups += [('PROTOCOL_'+str(k),v) for k,v in main.groupby('protocol_class')]
    for scope,g in groups:
      for f in ATOMIC:
        x=g[f].to_numpy(); y=g.delta_Eg_eV.to_numpy(); rho=corr(x,y)
        row={'scope':scope,'descriptor':f,'n_materials':len(g),'n_unique_changes':len(np.unique(np.round(x,10))),
             'n_nonzero_changes':int((abs(x)>1e-10).sum()),'spearman_rho':rho,'numerical_resolution':'UNRESOLVED'}
        if scope=='FORMAL_FIXED_13':
            vals=np.array([corr(x[a],y[a]) for a in ix]); good=vals[np.isfinite(vals)]
            lo,hi=np.quantile(good,[.025,.975]) if len(good) else [np.nan,np.nan]
            row.update(valid_bootstrap_n=len(good),total_bootstrap_n=B,valid_fraction=len(good)/B,
                conditional_ci_low=lo,conditional_ci_high=hi,display_ci_low=lo if len(good)/B>=.8 else np.nan,
                display_ci_high=hi if len(good)/B>=.8 else np.nan,
                interval_status='POINTWISE_PERCENTILE' if len(good)/B>=.8 else 'SUPPRESSED_DEGENERATE_RESAMPLING')
            boots.extend({'descriptor':f,'replicate':i,'rho':v,'valid':bool(np.isfinite(v))} for i,v in enumerate(vals))
            for i,mid in enumerate(g.material_id):
                take=np.arange(len(g))!=i;loo.append({'descriptor':f,'omitted_material_id':mid,'rho':corr(x[take],y[take])})
        rows.append(row)
    stats=pd.DataFrame(rows);write(stats,'FIG6B_ASSOCIATION_STATISTICS.csv')
    write(pd.DataFrame(boots),'FIG6B_BOOTSTRAP_DRAWS.csv');write(pd.DataFrame(loo),'FIG6B_LEAVE_ONE_MATERIAL_OUT.csv')
    cm=main[ATOMIC].corr(method='spearman');write(cm.rename_axis('descriptor').reset_index(),'FIG6B_DESCRIPTOR_COLLINEARITY.csv')
    old=read('old_stats').query("protocol_group=='FORMAL_FIXED_13'").set_index('descriptor')
    replay=max(abs(stats.query("scope=='FORMAL_FIXED_13'").set_index('descriptor').spearman_rho-old.spearman_rho))
    assert replay<1e-12
    return stats,main,{'old_spearman_max_difference':replay,'n20':20,'formal_fixed':13,'diagnostic_fixed':3,'legacy_relaxed':4}

def c_analysis():
    s=read('structure');p=read('protocol');base=read('recovery')
    s=s.drop(columns=['baseline_job','protocol_id']).merge(p[['point_id','baseline_job','protocol_status','structure_SHA_match']],on='point_id',validate='one_to_one')
    s=s.merge(base[['point_id','baseline_poscar_sha256','actual_poscar_sha256']],on='point_id',validate='one_to_one')
    assert (s.baseline_structure_sha256==s.baseline_poscar_sha256).all()
    assert (s.structure_sha256==s.actual_poscar_sha256).all()
    assert (s.protocol_status=='KEY_INPUTS_EXACT_MATCH').all()
    s['Eg_parent_recovered_eV']=s.Eg_eV-s.delta_Eg_eV
    s['baseline_gap_origin']='algebraic recovery Eg(point)-saved DeltaEg; not a fresh independent baseline read'
    s['baseline_global_L_A']=s.global_L_A-s.delta_L_A
    s['baseline_global_Theta_degree']=s.global_Theta_degree-s.delta_Theta_degree
    s['target_operation']=s.perturbation_type.map({'LENGTH':'local-distance path','ANGLE':'local-angle path'})
    s['observed_global_change_class']=np.select([
       (abs(s.delta_L_A)>1e-10)&(abs(s.delta_Theta_degree)>1e-10),
       abs(s.delta_L_A)>1e-10,abs(s.delta_Theta_degree)>1e-10],['mixed L and Theta','L only at serialization tolerance','Theta only at serialization tolerance'],default='no resolved global change at serialization tolerance')
    s['primary_distribution']=(s.scope=='FORMAL')&~s.first_shell_changed
    s['numerical_response_resolution']='UNRESOLVED'
    write(s,'FIG6C_GEOMETRY_PAIRED_ALL_440.csv')
    main=s[s.primary_distribution];assert len(main)==371 and main.material_id.nunique()==17
    rows=[]; parent=[];rng=np.random.default_rng(SEED+1)
    for (scope,kind),g in s.groupby(['scope','perturbation_type']):
        rows.append({'scope':scope,'target_operation':kind,'n_points':len(g),'n_materials':g.material_id.nunique(),
                     'n_shell_switch':int(g.first_shell_changed.sum()),'summary_role':'all-in inventory'})
    for kind,g in main.groupby('perturbation_type'):
        vals=g.delta_Eg_eV.to_numpy()*1000
        per=g.groupby('material_id').delta_Eg_eV.agg(['median',lambda v: np.median(abs(v))]);per.columns=['median_eV','median_abs_eV']
        for mid,r in per.iterrows():parent.append({'material_id':mid,'target_operation':kind,**r.to_dict()})
        sam=rng.integers(0,len(per),(B,len(per)))
        med=np.median(per.median_eV.to_numpy()[sam],axis=1)*1000
        ab=np.median(per.median_abs_eV.to_numpy()[sam],axis=1)*1000
        rows.append({'scope':'FORMAL_NON_SWITCH_PRIMARY','target_operation':kind,'n_points':len(g),'n_materials':g.material_id.nunique(),
          'n_shell_switch':0,'summary_role':'point distribution plus equal-parent center; intervals resample parents',
          'point_median_meV':np.median(vals),'point_Q1_meV':np.quantile(vals,.25),'point_Q3_meV':np.quantile(vals,.75),
          'point_min_meV':min(vals),'point_max_meV':max(vals),'positive_points':sum(vals>0),'negative_points':sum(vals<0),'zero_points':sum(vals==0),
          'median_of_parent_medians_meV':np.median(per.median_eV)*1000,'parent_center_ci_low_meV':np.quantile(med,.025),'parent_center_ci_high_meV':np.quantile(med,.975),
          'median_parent_median_abs_meV':np.median(per.median_abs_eV)*1000,'parent_abs_ci_low_meV':np.quantile(ab,.025),'parent_abs_ci_high_meV':np.quantile(ab,.975)})
    summary=pd.DataFrame(rows);write(summary,'FIG6C_RESPONSE_DISTRIBUTION_SUMMARY.csv');write(pd.DataFrame(parent),'FIG6C_PARENT_BALANCED_SUMMARIES.csv')
    write(main.groupby(['perturbation_type','observed_global_change_class']).size().rename('n_points').reset_index(),'FIG6C_GLOBAL_COCHANGE_COUNTS.csv')
    loc=read('local');glob=read('global')
    loc['qualification']='local operation slope; not isolated aggregate L/Theta derivative; numerical resolution unresolved'
    glob['qualification']='marginal coupled-path fit times observed within-parent SD; not independent partial derivative'
    write(loc,'FIG6C_LOCAL_PATH_SENSITIVITY_REUSED.csv');write(glob,'FIG6C_GLOBAL_STANDARDIZED_SENSITIVITY_REUSED.csv')
    return s,summary,{'all_points':len(s),'all_parents':s.material_id.nunique(),'main_points':len(main),'main_parents':17,
        'local_sign_unstable_paths':int((~loc.slope_sign_stable).sum()),'local_paths':len(loc),
        'main_mixed_global_points':int((main.observed_global_change_class=='mixed L and Theta').sum())}

def d_analysis(stats):
    et=read('et').set_index('feature');li=read('lime').set_index('feature');x=read('xgb').set_index('feature')
    cr=read('crisp').query("scope=='all'").set_index('descriptor')
    bs=stats.query("scope=='FORMAL_FIXED_13'").set_index('descriptor');rows=[]
    specifications=[('ET',et,'mean_abs_SHAP_eV','eV',1462,'global TreeSHAP; fixed fold0 ET'),
       ('LIME',li,'mean_abs_LIME_weight_eV_per_SD','eV/training SD',1462,'mean absolute Gaussian LIME Ridge slope; high-fidelity gate only 1/1462'),
       ('XGBoost',x,'paired_MAE_increase_eV','eV',7176,'signed 5-fold OOF single-feature retraining DeltaMAE'),
       ('CRISP-X-conditioned ET',cr,'mean_abs_delta_phi_eV','eV',24,'mean absolute ET TreeSHAP change on CRISP-X pools; 2304 points, 24 parents; not independent learner')]
    for method,df,col,unit,n,definition in specifications:
        ranks=rankdata(-df.loc[FEATURES,col].to_numpy(),method='average')
        for f,rk in zip(FEATURES,ranks):
            rows.append({'method':method,'descriptor':f,'score':float(df.loc[f,col]),'score_unit':unit,'rank_within_method_9':rk,'rank_atomic7':np.nan,
              'n_materials':n,'definition':definition,'qualification':'ESTIMATOR_AND_SUPPORT_SPECIFIC; not equal physical importance'})
    for f in FEATURES:
        rows.append({'method':'DFT','descriptor':f,'score':abs(bs.loc[f,'spearman_rho']) if f in ATOMIC else np.nan,
            'score_unit':'absolute marginal Spearman' if f in ATOMIC else 'NA','rank_within_method_9':np.nan,
            'rank_atomic7':rankdata(-abs(bs.loc[ATOMIC,'spearman_rho']).to_numpy())[ATOMIC.index(f)] if f in ATOMIC else np.nan,
            'n_materials':13 if f in ATOMIC else 17,'definition':'joint substitution marginal association; geometry independent importance not identified',
            'qualification':'EXPLORATORY_ATOMIC7_ONLY' if f in ATOMIC else 'NOT_IDENTIFIABLE_FROM_COUPLED_PATHS'})
    d=pd.DataFrame(rows);write(d,'FIG6D_FIVE_METHOD_LONG.csv')
    write(d.pivot(index='descriptor',columns='method',values='rank_within_method_9').reindex(FEATURES).reset_index(),'FIG6D_NINE_RANK_MATRIX_WITH_NA.csv')
    write(cr.reset_index(),'FIG6D_CRISP_ATTRIBUTION_DEFINITION_SENSITIVITY.csv')
    agreements=[]
    for a,b in itertools.combinations(d.method.unique(),2):
        aa=d[d.method==a].set_index('descriptor');bb=d[d.method==b].set_index('descriptor')
        features=ATOMIC if 'DFT' in [a,b] else FEATURES
        av=aa.loc[features,'score'].to_numpy();bv=bb.loc[features,'score'].to_numpy()
        ra=rankdata(-av); rb=rankdata(-bv)
        agreements.append({'method_A':a,'method_B':b,'n_descriptors':len(features),'spearman':corr(av,bv),
           'kendall_tau_b':kendalltau(av,bv).statistic,'top3_overlap':len(set(np.array(features)[np.argsort(ra)[:3]])&set(np.array(features)[np.argsort(rb)[:3]])),
           'interpretation':'UNMATCHED_SUPPORT_DESCRIPTIVE; estimator-specific; no physical/causal validation'})
    write(pd.DataFrame(agreements),'FIG6D_RANK_AGREEMENT_EXPLORATORY.csv')
    return d

def identity(a,b):
    if len(a)!=len(b) or [str(s.species) for s in a]!=[str(s.species) for s in b]:return False,np.nan,np.nan,'SPECIES_OR_SITE_ORDER_MISMATCH'
    lat=float(np.max(abs(a.lattice.matrix-b.lattice.matrix)))
    delta=a.frac_coords-b.frac_coords;delta-=np.rint(delta);frac=float(np.max(abs(delta)))
    return lat<=1e-10 and frac<=1e-10,lat,frac,'EXACT_ORDERED_PERIODIC_COMPARISON'

def e_analysis(s):
    man=read('manifest');pa=read('parents');cp=read('crisp_parents');res=read('responses')
    overlap=set(s.material_id)&set(cp.ID); assert overlap=={1230247}
    cand=res[(res.model=='frozen_ET')&(res.method=='block_b64_g40')&(res.kind=='geometry')].copy()
    assert len(cand)==1152 and cand.ID.nunique()==24
    rows=[];parent_rows=[]
    def sp(p):return SNAP/Path(p).resolve().relative_to(ROOT)
    for mid in overlap:
      dft=man[man.material_id==mid];rr=cand[cand.ID==mid]
      base=Structure.from_file(sp(localpath(pa.set_index('material_id').loc[mid,'baseline_structure'])))
      for seed in rr.seed.unique():
        p=ROOT/f'outputs/05_CRISP_X/local_reconstruction/physical_pools/{mid}_{seed}/parent.json'
        same,lat,fc,note=identity(base,Structure.from_dict(json.loads(sp(p).read_text(encoding='utf-8'))))
        parent_rows.append({'material_id':mid,'seed':seed,'same_parent_structure':same,'max_lattice_difference_A':lat,'max_fractional_difference_PBC':fc,'test':note})
      candidates={}
      for r in rr.itertuples():
        p=ROOT/f'outputs/05_CRISP_X/local_reconstruction/physical_pools/{mid}_{r.seed}/structure_{r.candidate}.json'
        candidates[(r.seed,r.candidate)]=Structure.from_dict(json.loads(sp(p).read_text(encoding='utf-8')))
      for r in dft.itertuples():
        a=Structure.from_file(sp(localpath(r.structure_path)))
        assert sha(sp(localpath(r.structure_path)))==r.perturbed_structure_hash
        for c in rr.itertuples():
          same,lat,fc,note=identity(a,candidates[(c.seed,c.candidate)])
          rows.append({'material_id':mid,'DFT_point_id':r.task_id,'seed':c.seed,'candidate':c.candidate,'same_perturbed_structure':same,
            'max_lattice_difference_A':lat,'max_fractional_difference_PBC':fc,'test':note,
            'CRISP_local_surrogate_predicted_delta_eV':c.prediction,'ET_direct_delta_eV':c.truth,
            'DFT_delta_eV':float(s.set_index('point_id').loc[r.task_id,'delta_Eg_eV'])})
    tests=pd.DataFrame(rows);write(tests,'FIG6E_EXACT_STRUCTURE_CANDIDATE_TESTS.csv');write(pd.DataFrame(parent_rows),'FIG6E_PARENT_STRUCTURE_IDENTITY.csv')
    assert not tests.same_perturbed_structure.any(),'Strict matching points found: amend e analysis before conclusion'
    eligibility=s[['material_id','point_id','scope','primary_distribution','baseline_job','job_id','baseline_structure_sha256','structure_sha256','delta_Eg_eV','protocol_status']].copy()
    eligibility['parent_ID_in_frozen_CRISP24']=eligibility.material_id.isin(cp.ID)
    eligibility['strict_same_perturbation_match']=False
    eligibility['CRISP_predicted_delta_eV']=np.nan
    eligibility['reason']=np.where(eligibility.parent_ID_in_frozen_CRISP24,'SHARED_PARENT_ID_BUT_NO_EXACT_PERTURBATION_MATCH','PARENT_NOT_IN_FROZEN_CRISP24')
    eligibility['metrics_eligible']=False;write(eligibility,'FIG6E_PAIR_ELIGIBILITY_ALL_440.csv')
    metrics={'eligible_pairs':0,'MAE_eV':None,'RMSE_eV':None,'R2':None,'direction_agreement':None,
        'reason':'No same-parent/same-perturbed-structure pairs; do not pair by ID, target step or nearest descriptor',
        'CRISP_estimator':'block_b64_g40 frozen_ET local surrogate predicted response, not direct ET query or SHAP attribution',
        'overlap_parent_ids':sorted(overlap),'candidate_structure_tests':len(tests),'shared_parent_candidates':len(rr)}
    (_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6E_METRICS_NOT_ESTIMABLE.json')).write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics

def main():
    freeze()
    stats,bmain,bqa=b_analysis();s,cs,cqa=c_analysis();d=d_analysis(stats);em=e_analysis(s)
    source=pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/response_pairing/INPUT_SOURCE_MAPPING.csv'))
    assert all(sha(r.source_path)==r.sha256 and sha(r.snapshot_path)==r.sha256 for r in source.itertuples())
    qa={'source_sha_unchanged':True,'source_files':len(source),'b':bqa,'c':cqa,'e':em,
        'full_20_set_preserved':set(read('matrix').current_material_id)==set(s.material_id),
        'no_new_model_or_DFT_execution':True,'formal_plotting_performed':False,
        'python':__import__('sys').version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__}
    (_repo_path('outputs/07_DFT_VALIDATION/response_pairing/ANALYSIS_QA.json')).write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    main()
