"""Rebuild run010 from sealed run009 and the replayed historical Ward labels."""
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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/descriptor_evidence')))

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = _repo_path('.')
OUT = _repo_path('07_DFT_VALIDATION/workflows/descriptor_evidence')
R9 = _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap')
SHAP = _repo_path('outputs/05_CRISP_X/local_reconstruction/shap_baseline/per_material_nine_attributions.csv')
WARD = _repo_path('unavailable_inputs/source_2ded8002d935/historical_profile15_zero_partition_7176.csv')
FEATURES = ['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']
ATOMIC = FEATURES[:7]
SRC = [SHAP, WARD, _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_FEATURE_IMPORTANCE_FINAL.csv'), _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_GLOBAL_MATERIAL_DETAIL.csv'),
       _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_LOCAL_PATH_SUMMARY.csv'), _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ET_TREESHAP_FINAL.csv'), _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/LIME_FINAL.csv'),
       _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_CONDITIONED_TREESHAP.csv'), _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/FEATURE_RANK_AGREEMENT.csv'),
       _repo_path('07_DFT_VALIDATION/workflows/association_methods/NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv'),
       _repo_path('07_DFT_VALIDATION/workflows/association_identifiability/STRUCTURE_PAIR_PROTOCOL_AUDIT.csv')]

def write_csv(name, data):
    data.to_csv(OUT/name, index=False, float_format='%.15g')

def main():
    OUT.mkdir(exist_ok=False) if not OUT.exists() else None
    ward = pd.read_csv(WARD, dtype={'material_id':str})
    shap = pd.read_csv(SHAP, dtype={'ID':str})
    assert len(shap)==1462 and shap.ID.is_unique and ward.material_id.is_unique and len(ward)==7176
    assert set(ward.cluster_id)==set(range(1,11))
    df = shap.merge(ward[['material_id','cluster_id']], left_on='ID',right_on='material_id',how='left',validate='one_to_one')
    assert len(df)==1462 and df.cluster_id.notna().all()
    assert df.formula_group.notna().all()
    expected = pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ET_TREESHAP_FINAL.csv')).set_index('feature')
    for f in FEATURES:
        assert abs(df['SHAP_'+f].abs().mean()-expected.loc[f,'mean_abs_SHAP_eV'])<1e-12, f
    identity = []
    heat=[]; stats=[]; ranks=[]; normalized=[]
    rng=np.random.default_rng(20261030)
    for cid,g in df.groupby('cluster_id',sort=True):
        n=len(g); ng=g.formula_group.nunique()
        # The 7176 partition is mutually exclusive; resample whole formula groups.
        grouped=g.groupby('formula_group',sort=True)
        means=np.stack([x[['SHAP_'+f for f in FEATURES]].abs().sum().to_numpy(float) for _,x in grouped])
        counts=np.array([len(x) for _,x in grouped],dtype=float)
        draws=rng.integers(0,ng,size=(2000,ng))
        bootstrap=means[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
        vals=g[['SHAP_'+f for f in FEATURES]].abs().to_numpy(float)
        m=vals.mean(axis=0); total=m.sum()
        qualifies=n>=30 and ng>=20
        identity.append({'cluster_id':cid,'definition':'historical_profile15_zero; replayed distance-profile Ward, not nine-feature direct Ward',
                         'materials_in_ET_holdout':n,'formula_groups':ng,'mutually_exclusive':True,
                         'unassigned_in_ET_holdout':0,'interval_display':'YES' if qualifies else 'WITHHELD_SMALL_CLUSTER',
                         'source':str(WARD.relative_to(ROOT))})
        order=pd.Series(-m,index=FEATURES).rank(method='min').astype(int)
        for k,f in enumerate(FEATURES):
            lo,hi=np.percentile(bootstrap[:,k],[2.5,97.5]); base=dict(cluster_id=cid,descriptor=f,n_materials=n,n_formula_groups=ng)
            heat.append({**base,'mean_abs_SHAP_eV':m[k],'rank':int(order[f])})
            stats.append({**base,'mean_abs_SHAP_eV':m[k],'median_abs_SHAP_eV':np.median(vals[:,k]),
                          'std_abs_SHAP_eV':np.std(vals[:,k],ddof=1),'bootstrap_low95_eV':lo if qualifies else np.nan,
                          'bootstrap_high95_eV':hi if qualifies else np.nan,'bootstrap_draws':2000,
                          'bootstrap_unit':'formula_group','interval_status':'DISPLAY' if qualifies else 'WITHHELD_SMALL_CLUSTER'})
            ranks.append({**base,'rank':int(order[f]),'mean_abs_SHAP_eV':m[k]})
            normalized.append({**base,'within_cluster_percent':100*m[k]/total,'denominator_sum_mean_abs_SHAP_eV':total,
                               'cross_cluster_absolute_comparison_allowed':False})
    for name,rows in [('CLUSTER_IDENTITY_AUDIT.csv',identity),('CLUSTER_IMPORTANCE_HEATMAP_DATA.csv',heat),
                      ('CLUSTER_IMPORTANCE_STATISTICS.csv',stats),('CLUSTER_IMPORTANCE_RANKS.csv',ranks),
                      ('CLUSTER_WITHIN_NORMALIZED_PERCENT.csv',normalized)]: write_csv(name,pd.DataFrame(rows))
    write_csv('CLUSTER_MEMBER_IDENTITY.csv',df[['ID','formula_group','cluster_id']].rename(columns={'ID':'material_id'}))

    dft=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_FEATURE_IMPORTANCE_FINAL.csv'))
    atom=dft[dft.descriptor.isin(ATOMIC)].copy()
    atom['scientific_qualification']='EXPLORATORY_MARGINAL_ASSOCIATION; 13 paired substitutions; numerical resolution unresolved; correlated attributes'
    write_csv('DFT_SEVEN_FEATURE_RESULTS.csv',atom)
    global_=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_GLOBAL_MATERIAL_DETAIL.csv'))
    local=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/DFT_LOCAL_PATH_SUMMARY.csv'))
    global_['response_scope']='GLOBAL_COUPLED_L_THETA'
    local['response_scope']='LOCAL_GEOMETRY_PATH'
    write_csv('DFT_GEOMETRY_RESPONSE_RESULTS.csv',global_)
    write_csv('DFT_LOCAL_PATH_RESULTS.csv',local)
    assert len(global_)==34 and len(local)==34 and (~local.slope_sign_stable.astype(bool)).sum()==15

    et=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ET_TREESHAP_FINAL.csv')).set_index('feature')
    atom=atom.set_index('descriptor')
    agreement=[]
    for f in ATOMIC:
        agreement.append({'descriptor':f,'DFT_marginal_rho':atom.loc[f,'marginal_spearman_rho'],
                          'DFT_abs_marginal_rho':atom.loc[f,'absolute_marginal_rho'],
                          'DFT_rank':int(atom.loc[f,'exploratory_marginal_rank_atomic7']),
                          'ET_mean_abs_TreeSHAP_eV':et.loc[f,'mean_abs_SHAP_eV'],
                          'ET_rank':int(et.loc[f,'rank']),
                          'rank_difference_DFT_minus_ET':int(atom.loc[f,'exploratory_marginal_rank_atomic7']-et.loc[f,'rank']),
                          'DFT_n_pairs':13,'ET_n_holdout':1462,'qualification':'EXPLORATORY_RANK_AGREEMENT_ONLY'})
    agree=pd.DataFrame(agreement)
    write_csv('DFT_ET_RANK_AGREEMENT.csv',agree)
    sp=float(spearmanr(agree.DFT_rank,agree.ET_rank).statistic)
    kt=float(kendalltau(agree.DFT_rank,agree.ET_rank).statistic)
    assert abs(sp-0.8928571428571429)<1e-12 and abs(kt-0.8095238095238096)<1e-12
    summary=pd.DataFrame([{'n_features':7,'spearman_rank':sp,'kendall_tau':kt,
                           'top3_overlap':len(set(agree.nsmallest(3,'DFT_rank').descriptor)&set(agree.nsmallest(3,'ET_rank').descriptor)),
                           'qualification':'EXPLORATORY; different estimands, populations and correlated DFT substitutions'}])
    write_csv('DFT_ET_RANK_AGREEMENT_SUMMARY.csv',summary)

    lime=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/LIME_FINAL.csv')).set_index('feature')
    crisp=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_CONDITIONED_TREESHAP.csv'))
    evidence=[]
    for f in FEATURES:
        dc=dft.set_index('descriptor').loc[f]
        cc=crisp[(crisp.scope=='all')&(crisp.descriptor==f)].iloc[0]
        for method,n,estimand,value,rank,qualification,compare in [
            ('ET TreeSHAP',1462,'mean absolute TreeSHAP eV',et.loc[f,'mean_abs_SHAP_eV'],et.loc[f,'rank'],'MODEL_ATTRIBUTION','MODEL_ONLY'),
            ('LIME',1462,'mean absolute local Ridge weight eV per training SD',lime.loc[f,'mean_abs_LIME_weight_eV_per_SD'],lime.loc[f,'rank'],'LOW_FIDELITY_1_OF_1462','NO_NUMERIC_CROSS_METHOD'),
            ('CRISP-X conditioned TreeSHAP',24,'mean absolute parent-to-perturbed delta phi eV',cc.mean_abs_delta_phi_eV,cc.rank_abs_delta_within_scope,'SAME_FROZEN_ET_MODEL; 2304 perturbation points','MODEL_ONLY'),
            ('DFT element substitution',13,'signed marginal Spearman rho',dc.marginal_spearman_rho if f in ATOMIC else np.nan,dc.exploratory_marginal_rank_atomic7 if f in ATOMIC else np.nan,'EXPLORATORY_MARGINAL_ASSOCIATION' if f in ATOMIC else 'NOT_APPLICABLE','ATOMIC_RANK_EXPLORATORY' if f in ATOMIC else 'NO'),
            ('DFT structural perturbation',17,'median absolute within-material standardized coefficient eV',dc.median_abs_standardized_coefficient_eV if f not in ATOMIC else np.nan,np.nan,'COUPLED_L_THETA; 371 formal points' if f not in ATOMIC else 'NOT_APPLICABLE','NO_INDEPENDENT_NINE_RANK')]:
            evidence.append({'descriptor':f,'method':method,'source':str((R9/('DFT_FEATURE_IMPORTANCE_FINAL.csv' if method.startswith('DFT') else 'ET_TREESHAP_FINAL.csv' if method=='ET TreeSHAP' else 'LIME_FINAL.csv' if method=='LIME' else 'CRISP_X_CONDITIONED_TREESHAP.csv')).relative_to(ROOT)),
                             'n_independent_units':n,'estimand':estimand,'value':value,'rank':rank,'scientific_qualification':qualification,
                             'cross_method_comparison':compare})
    write_csv('NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv',pd.DataFrame(evidence))

    # A parent ID match is necessary but cannot establish evaluated-structure identity.
    # All 460 paired DFT responses lack a certified pairwise numerical-error bound.
    formal=pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/association_methods/NINE_DESCRIPTOR_EVIDENCE_MATRIX.csv'),dtype={'material_id':str})
    formal=formal[(formal.data_family=='STRUCTURE')&(formal.scope=='FORMAL')]
    mapping=pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/association_identifiability/STRUCTURE_PAIR_PROTOCOL_AUDIT.csv'),dtype={'material_id':str})
    formal=formal.merge(mapping[['material_id','point_id','structure_SHA_match','protocol_status']],on=['material_id','point_id'],how='left',validate='one_to_one')
    assert len(formal)==371 and formal.structure_SHA_match.notna().all()
    audit=[];holdout=set(df.ID)
    for row in formal.itertuples(index=False):
        audit.append({'material_id':row.material_id,'point_id':row.point_id,
                      'DFT_internal_structure_SHA_match':row.structure_SHA_match,
                      'DFT_protocol_status':row.protocol_status,
                      'parent_in_1462_frozen_ET_holdout':row.material_id in holdout,
                      'perturbed_structure_exact_ET_input_match':'NOT_VERIFIED',
                      'DFT_pairwise_numerical_resolution':row.numerical_resolution_status,
                      'official_DFT_ET_response_eligibility':'NA',
                      'reason':'No certified pairwise DFT numerical resolution; exact parent/perturbed ET structure and model domain/protocol mapping not jointly established'})
    write_csv('DFT_ET_PAIRED_RESPONSE_AUDIT.csv',pd.DataFrame(audit))

    hashes=[]
    for p in SRC:
        hashes.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    write_csv('INPUT_SOURCE_SHA256.csv',pd.DataFrame(hashes))
    (_repo_path('07_DFT_VALIDATION/configs/descriptor_evidence.json')).write_text(json.dumps({'run':'descriptor_evidence','date':'2026-10-30','seed':20261030,
        'bootstrap_draws':2000,'bootstrap_unit':'formula_group within historical Ward cluster',
        'source_ET':'frozen fold0 ExtraTrees heldout TreeSHAP','cluster_definition':'replayed historical profile15 zero-target Ward',
        'small_cluster_interval_rule':'n>=30 and formula_groups>=20',
        'no_cross_cluster_weighting':True,'no_new_DFT_or_model_inference':True},indent=2),encoding='utf-8')
    print(json.dumps({'clusters':len(identity),'holdout':len(df),'small_clusters':[x['cluster_id'] for x in identity if x['interval_display']!='YES'],
                      'DFT_ET_parent_holdout_overlap':len({x['material_id'] for x in audit if x['parent_in_1462_frozen_ET_holdout']}),
                      'DFT_ET_formal_points_audited':len(audit),
                      'rank_spearman':sp,'rank_kendall':kt,'local_unstable':15},ensure_ascii=False))

if __name__=='__main__': main()
