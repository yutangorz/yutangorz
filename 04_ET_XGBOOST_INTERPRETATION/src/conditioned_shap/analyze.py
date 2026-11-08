"""Reassemble archived ET/CRISP-X contrasts; no new model explanation or fit."""
from __future__ import annotations

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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/conditioned_shap')))

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = _repo_path('.')
OUT = _repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap')
CFG = json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/conditioned_shap.json')).read_text(encoding='utf-8'))
F = CFG['features']
BASE = _repo_path('05_CRISP_X/workflows')
R19 = _repo_path('05_CRISP_X/workflows/local_reconstruction')
R21 = _repo_path('05_CRISP_X/workflows/attribution_validation')

def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def csvout(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT/name, index=False, float_format='%.17g')

def main() -> None:
    manifest = read(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/INPUT_SOURCE_SHA256.csv'))
    for row in manifest.itertuples(index=False):
        p = ROOT/row.path
        if not p.is_file() or digest(p) != row.sha256 or p.stat().st_size != row.bytes:
            raise SystemExit(f'Source changed: {row.path}')
    source = read(_repo_path('05_CRISP_X/workflows/attribution_validation/results/aligned_comparison/per_response_nine_contrasts.csv'), dtype={'ID':str})
    delta = source[source.method == 'SHAP_contrast'].copy()
    parent = read(_repo_path('outputs/05_CRISP_X/local_reconstruction/shap_baseline/per_material_nine_attributions.csv'), dtype={'ID':str})
    pools = json.loads((_repo_path('05_CRISP_X/workflows/attribution_validation/inputs/pool_manifest.json')).read_text(encoding='utf-8'))
    assert len(delta) == 2304 and len(pools) == 48 and delta[['ID','seed','candidate']].duplicated().sum() == 0
    assert delta.kind.value_counts().to_dict() == {'geometry':1152,'chemical':1152}
    assert delta.ID.nunique() == 24 and delta[['ID','seed']].drop_duplicates().shape[0] == 48
    assert parent.ID.is_unique and set(delta.ID) <= set(parent.ID)
    poolrows = []
    for entry in pools:
        ident, seed = str(entry['ID']), int(entry['seed'])
        directory = _repo_path('05_CRISP_X/workflows/attribution_validation/inputs/pools')/f'{ident}_{seed}'
        pool = read(directory/'pool.csv')
        with np.load(directory/'indices.npz') as index:
            selected = index['test']
        assert len(selected) == 48 and len(set(selected)) == 48
        for candidate in selected:
            row = pool.loc[int(candidate)]
            record = {'ID':ident,'seed':seed,'candidate':int(candidate),
                      'pool_delta_ET_eV':float(row['delta_frozen_ET']),
                      'role':row['role'],'amplitude':float(row['amplitude'])}
            record.update({f'input_{f}':float(row[f]) for f in F})
            poolrows.append(record)
    physical = pd.DataFrame(poolrows)
    assert not physical[['ID','seed','candidate']].duplicated().any()
    d = delta.merge(physical, on=['ID','seed','candidate'], validate='one_to_one')
    parent_cols = ['ID','formula_group','prediction_eV','base_eV']+F+['SHAP_'+f for f in F]
    p = parent[parent_cols].rename(columns={'prediction_eV':'parent_prediction_eV','base_eV':'shap_reference_eV',**{f:'parent_input_'+f for f in F},**{'SHAP_'+f:'parent_phi_'+f for f in F}})
    d = d.merge(p, on='ID', validate='many_to_one')
    response = read(_repo_path('outputs/05_CRISP_X/local_reconstruction/response_predictions.csv'), dtype={'ID':str})
    response = response[(response.model=='frozen_ET')&(response.method=='block_b64_g40')]
    d = d.merge(response[['ID','seed','candidate','reason','prediction']], on=['ID','seed','candidate'], validate='one_to_one')
    assert len(d) == 2304 and set(d.reason) == {'supported','joint_path_mean_only_no_geometry_slope','unknown_path'}
    assert d[d.kind=='chemical'].reason.value_counts().to_dict() == {'joint_path_mean_only_no_geometry_slope':1058,'supported':86,'unknown_path':8}
    input_mismatch = max(float(np.max(abs(d['input_'+f]-d['parent_input_'+f]))) for f in F)
    # The physical vector is traced to the source pool; parent/perturbed differences are retained below.
    max_source_delta = float(np.max(abs(d.truth_delta_eV-d.pool_delta_ET_eV)))
    max_contrast = float(np.max(abs(d.truth_delta_eV-d.reconstructed_delta_eV)))
    assert max_source_delta < 1e-9 and max_contrast < 1e-9
    for f in F:
        d['delta_input_'+f] = d['input_'+f]-d['parent_input_'+f]
        d['parent_phi_'+f] = d['parent_phi_'+f].astype(float)
        d['delta_phi_'+f] = d[f].astype(float)
        d['perturbed_phi_'+f] = d['parent_phi_'+f]+d['delta_phi_'+f]
    d['phi_delta_sum_eV'] = d[['delta_phi_'+f for f in F]].sum(axis=1)
    d['additivity_residual_eV'] = d['phi_delta_sum_eV']-d.truth_delta_eV
    d['perturbed_prediction_eV'] = d.parent_prediction_eV+d.truth_delta_eV
    d['perturbed_reconstruction_eV'] = d.shap_reference_eV+d[['perturbed_phi_'+f for f in F]].sum(axis=1)
    max_perturbed = float(np.max(abs(d.perturbed_prediction_eV-d.perturbed_reconstruction_eV)))
    assert max_perturbed < 1e-9
    q = read(_repo_path('outputs/05_CRISP_X/local_reconstruction/query_and_path_design.csv'), dtype={'ID':str})
    q = q[(q.model=='frozen_ET')&(q.path=='geometry_only')]
    assert len(q) == 48 and q.separate_geometry_slopes_qualified.astype(bool).all()
    d['geometry_separate_LTheta_qualified'] = d.kind.eq('geometry') & d[['ID','seed']].apply(tuple,axis=1).isin(set(q[['ID','seed']].itertuples(index=False,name=None)))
    d['chemical_status'] = np.select([d.kind.eq('chemical')&d.reason.eq('supported'),d.kind.eq('chemical')&d.reason.eq('joint_path_mean_only_no_geometry_slope'),d.kind.eq('chemical')&d.reason.eq('unknown_path')],['AFFINE_SUPPORT','JOINT_PATH_MEAN_ONLY','ABSTAIN'],default='NOT_CHEMICAL')
    assert d[d.kind=='geometry'].geometry_separate_LTheta_qualified.all()
    # Long pointwise output includes both input vectors and model allocations.
    common = ['ID','seed','candidate','formula_group','kind','path','reason','chemical_status','geometry_separate_LTheta_qualified','amplitude','parent_prediction_eV','perturbed_prediction_eV','truth_delta_eV','pool_delta_ET_eV','additivity_residual_eV']
    pointwise = []
    for f in F:
        frame = d[common+['parent_input_'+f,'input_'+f,'delta_input_'+f,'parent_phi_'+f,'perturbed_phi_'+f,'delta_phi_'+f]].copy()
        frame.columns = common+['parent_input','perturbed_input','delta_input','phi_parent_eV','phi_perturbed_eV','delta_phi_eV']
        frame.insert(3,'descriptor',f)
        pointwise.append(frame)
    csvout(pd.concat(pointwise,ignore_index=True),'CRISP_X_DELTA_SHAP.csv')

    # Equal-parent descriptive summaries and 2000 parent-cluster bootstrap draws.
    scopes = {
        'all':np.ones(len(d),dtype=bool),
        'geometry':d.kind.eq('geometry').to_numpy(),
        'chemical':d.kind.eq('chemical').to_numpy(),
        'chemical_nonabstain':(d.kind.eq('chemical')&~d.chemical_status.eq('ABSTAIN')).to_numpy(),
        'geometry_L_positive':(d.kind.eq('geometry')&(d.delta_input_bond_length>0)).to_numpy(),
        'geometry_L_negative':(d.kind.eq('geometry')&(d.delta_input_bond_length<0)).to_numpy(),
        'geometry_Theta_positive':(d.kind.eq('geometry')&(d.delta_input_bond_angle>0)).to_numpy(),
        'geometry_Theta_negative':(d.kind.eq('geometry')&(d.delta_input_bond_angle<0)).to_numpy(),
        'geometry_separate_slopes_qualified':d.geometry_separate_LTheta_qualified.to_numpy(),
        'chemical_affine_support':d.chemical_status.eq('AFFINE_SUPPORT').to_numpy(),
        'chemical_joint_path_mean':d.chemical_status.eq('JOINT_PATH_MEAN_ONLY').to_numpy(),
        'chemical_abstain_descriptive':d.chemical_status.eq('ABSTAIN').to_numpy(),
    }
    rng = np.random.default_rng(CFG['bootstrap_seed'])
    parent_ids = sorted(d.ID.unique())
    bootstrap_indices = rng.integers(0,len(parent_ids),size=(CFG['bootstrap_draws'],len(parent_ids)))
    summary, boot, parentrows = [],[],[]
    for scope,mask in scopes.items():
        subset = d.loc[mask]
        if subset.empty: continue
        for f in F:
            vals = {'abs_phi_perturbed_eV':subset['perturbed_phi_'+f].abs(),
                    'abs_delta_phi_eV':subset['delta_phi_'+f].abs(),
                    'signed_delta_phi_eV':subset['delta_phi_'+f]}
            per_parent = {}
            for metric,series in vals.items():
                tmp = subset[['ID']].copy();tmp['value']=series.to_numpy()
                group = tmp.groupby('ID').value
                per_parent[metric] = group.mean().reindex(parent_ids).to_numpy(float)
                for ident,group_mean in group.mean().items():
                    parentrows.append({'scope':scope,'descriptor':f,'material_id':ident,'metric':metric,'mean_eV':group_mean,'n_points':int((subset.ID==ident).sum())})
            smed = subset.groupby('ID')['delta_phi_'+f].median().reindex(parent_ids).to_numpy(float)
            record = {'scope':scope,'descriptor':f,'n_points':len(subset),'n_parents':subset.ID.nunique(),
                      'mean_abs_phi_perturbed_eV':float(np.nanmean(per_parent['abs_phi_perturbed_eV'])),
                      'mean_abs_delta_phi_eV':float(np.nanmean(per_parent['abs_delta_phi_eV'])),
                      'mean_signed_delta_phi_eV':float(np.nanmean(per_parent['signed_delta_phi_eV'])),
                      'median_signed_delta_phi_eV':float(np.nanmedian(smed)),
                      'median_definition':'median of parent-level point medians'}
            for metric,arr in {**per_parent,'median_signed_delta_phi_eV':smed}.items():
                sampled = arr[bootstrap_indices]
                if metric=='median_signed_delta_phi_eV':
                    values = np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in sampled])
                else:
                    count = np.isfinite(sampled).sum(axis=1)
                    values = np.divide(np.nansum(sampled,axis=1),count,out=np.full(len(count),np.nan),where=count>0)
                valid = values[np.isfinite(values)]
                low,high = np.quantile(valid,[0.025,0.975]) if len(valid) else (np.nan,np.nan)
                record[metric+'_low95'] = low;record[metric+'_high95'] = high;record[metric+'_bootstrap_valid_n']=len(valid)
                boot.extend({'scope':scope,'descriptor':f,'metric':metric,'draw':i,'value_eV':v} for i,v in enumerate(values))
            summary.append(record)
    summary = pd.DataFrame(summary)
    summary['rank_abs_delta_within_scope'] = summary.groupby('scope').mean_abs_delta_phi_eV.rank(ascending=False,method='min').astype(int)
    summary['rank_abs_perturbed_within_scope'] = summary.groupby('scope').mean_abs_phi_perturbed_eV.rank(ascending=False,method='min').astype(int)
    csvout(summary,'CRISP_X_CONDITIONED_TREESHAP.csv')
    parent24 = parent[parent.ID.isin(parent_ids)].set_index('ID').loc[parent_ids].reset_index()
    parent24[['ID','formula_group','prediction_eV','base_eV']+F+['SHAP_'+f for f in F]].to_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/CRISP_X_PARENT_POINTWISE.csv'),index=False,float_format='%.17g')
    parent_summary=[]
    for f in F:
        v=parent24['SHAP_'+f].to_numpy(float); draws=np.abs(v[bootstrap_indices]).mean(axis=1)
        parent_summary.append({'descriptor':f,'n_parents':24,'mean_abs_parent_phi_eV':np.abs(v).mean(),
                               'mean_signed_parent_phi_eV':v.mean(),'median_parent_phi_eV':np.median(v),
                               'mean_abs_low95':np.quantile(draws,.025),'mean_abs_high95':np.quantile(draws,.975),
                               'positive_fraction':(v>0).mean(),'reference':'same tree_path_dependent frozen ET'})
    csvout(pd.DataFrame(parent_summary),'CRISP_X_PARENT_TREESHAP.csv')
    csvout(pd.DataFrame(parentrows),'PARENT_CONTRIBUTIONS.csv')
    csvout(pd.DataFrame(boot),'BOOTSTRAP_DISTRIBUTIONS.csv.gz')
    pathrows=[]
    for (ident,seed,kind,path),g in d.groupby(['ID','seed','kind','path'],sort=True):
        for f in F:
            pathrows.append({'material_id':ident,'seed':seed,'kind':kind,'path':path,'descriptor':f,'n_points':len(g),'mean_abs_delta_phi_eV':g['delta_phi_'+f].abs().mean(),'mean_signed_delta_phi_eV':g['delta_phi_'+f].mean(),'chemical_status':','.join(sorted(g.chemical_status.unique())),'geometry_separate_slopes_qualified':bool(g.geometry_separate_LTheta_qualified.all())})
    csvout(pd.DataFrame(pathrows),'PATH_CONTRIBUTIONS.csv')

    element = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/ELEMENT_IMPORTANCE_RESULTS.csv'))
    element = element[element.protocol_group=='FORMAL_FIXED_13'].set_index('descriptor')
    stability = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/ELEMENT_IMPORTANCE_STABILITY.csv'))
    corr = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/ELEMENT_CORRELATION_MATRIX.csv')).set_index('descriptor')
    global_result = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/GLOBAL_IMPORTANCE_RESULTS.csv')).set_index('descriptor')
    global_detail = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/GLOBAL_MATERIAL_SENSITIVITY.csv'))
    csvout(global_detail,'DFT_GLOBAL_MATERIAL_DETAIL.csv')
    local_length = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/LOCAL_LENGTH_IMPORTANCE.csv'))
    local_angle = read(_repo_path('07_DFT_VALIDATION/workflows/association_methods/LOCAL_ANGLE_IMPORTANCE.csv'))
    csvout(pd.concat([local_length,local_angle],ignore_index=True),'DFT_LOCAL_PATH_SUMMARY.csv')
    csvout(corr.reset_index(),'DFT_ATOMIC_CORRELATION_MATRIX.csv')
    et = read(_repo_path('feature_interpretability_audit/tables/Table_A_TreeSHAP.csv')).set_index('feature')
    lime = read(_repo_path('feature_interpretability_audit/tables/Table_B_LIME.csv')).set_index('feature')
    dft_rows=[]
    element_rank = element.spearman_rho.abs().rank(ascending=False,method='min')
    for f in F:
        if f in element.index:
            e=element.loc[f]; loo=stability[stability.descriptor==f].spearman_rho
            correlations=corr.loc[f].drop(f).abs()
            row={'descriptor':f,'evidence_type':'13_fixed_geometry_element_substitutions','n_materials':13,'n_points':13,
                 'marginal_spearman_rho':e.spearman_rho,'direction':np.sign(e.spearman_rho),
                 'absolute_marginal_rho':abs(e.spearman_rho),'exploratory_marginal_rank_atomic7':int(element_rank[f]),
                 'bootstrap_low95':e.spearman_ci_low,'bootstrap_high95':e.spearman_ci_high,
                 'bootstrap_valid_draws':int(e.n_boot_valid),'LOMO_min':loo.min(),'LOMO_max':loo.max(),
                 'largest_other_attribute_abs_rho':correlations.max(),'most_correlated_attribute':correlations.idxmax(),
                 'protocol':'5 nonSOC + 7 SOC + 1 supplementary SOC; fixed geometry',
                 'numerical_resolution':'UNRESOLVED','independent_importance':'NOT_IDENTIFIABLE; correlated attribute substitutions'}
        else:
            name={'bond_length':'L','bond_angle':'Theta'}[f];g=global_result.loc[name]
            row={'descriptor':f,'evidence_type':'17_material_371_point_global_geometry','n_materials':17,'n_points':371,
                 'marginal_spearman_rho':np.nan,'direction':np.nan,'absolute_marginal_rho':np.nan,
                 'exploratory_marginal_rank_atomic7':np.nan,'bootstrap_low95':g.bootstrap_q025,'bootstrap_high95':g.bootstrap_q975,
                 'bootstrap_valid_draws':np.nan,'LOMO_min':g.leave_one_material_min,'LOMO_max':g.leave_one_material_max,
                 'largest_other_attribute_abs_rho':np.nan,'most_correlated_attribute':'L–Theta co-vary',
                 'protocol':'17 matched parent–perturbation protocols; SOC strata retained',
                 'numerical_resolution':'UNRESOLVED','independent_importance':g.independent_importance_status,
                 'median_absolute_slope':g.median_absolute_slope,'slope_unit':g.slope_unit,
                 'median_abs_standardized_coefficient_eV':global_detail[global_detail.descriptor==name].standardized_coefficient_eV.abs().median(),
                 'finite_response_valid_materials':global_detail[(global_detail.descriptor==name)&global_detail.finite_response_eV.notna()].material_id.nunique()}
        dft_rows.append(row)
    dft=pd.DataFrame(dft_rows);dft['independent_importance_rank']='NA';csvout(dft,'DFT_FEATURE_IMPORTANCE_FINAL.csv')
    etout=et.loc[F].reset_index();etout['positive_phi_fraction_1462']=[float((parent['SHAP_'+f]>0).mean()) for f in F]
    etout['negative_phi_fraction_1462']=[float((parent['SHAP_'+f]<0).mean()) for f in F]
    etout['median_phi_eV_1462']=[float(parent['SHAP_'+f].median()) for f in F]
    csvout(etout,'ET_TREESHAP_FINAL.csv')
    csvout(lime.loc[F].reset_index(),'LIME_FINAL.csv')
    m=dft.merge(etout[['feature','mean_abs_SHAP_eV','rank','low95_eV','high95_eV']].rename(columns={'feature':'descriptor','rank':'ET_rank','low95_eV':'ET_low95','high95_eV':'ET_high95'}),on='descriptor')
    m=m.merge(lime[['mean_abs_LIME_weight_eV_per_SD','rank','high_fidelity_n']].rename(columns={'rank':'LIME_rank'}),left_on='descriptor',right_index=True)
    for scope in ['all','geometry','chemical','chemical_nonabstain']:
        s=summary[summary.scope==scope][['descriptor','mean_abs_delta_phi_eV','mean_signed_delta_phi_eV','rank_abs_delta_within_scope']].rename(columns={c:scope+'_'+c for c in ['mean_abs_delta_phi_eV','mean_signed_delta_phi_eV','rank_abs_delta_within_scope']})
        m=m.merge(s,on='descriptor')
    m['cross_method_numeric_comparability']='NO; distinct estimands and populations'
    csvout(m,'NINE_DESCRIPTOR_CROSS_METHOD_MATRIX.csv')
    ranks=[]
    def agreement(label,a,b,scope,qualification):
        sr=spearmanr(a,b).statistic; kt=kendalltau(a,b).statistic
        ranks.append({'comparison':label,'n_features':len(a),'scope':scope,'spearman_rank':sr,'kendall_tau':kt,'qualification':qualification})
    agreement('ET_vs_native_LIME',et.loc[F,'rank'],lime.loc[F,'rank'],'same 1462 heldout materials','MODEL_RANK_ONLY; LIME dual-fidelity 1/1462')
    agreement('ET_vs_DFT_abs_marginal',et.loc[F[:7],'rank'],element_rank.loc[F[:7]],'7 atomic features; 13 DFT pairs vs 1462 ET','EXPLORATORY; correlations not attribution')
    agreement('ET_vs_CRISP_delta_all',et.loc[F,'rank'],summary[summary.scope=='all'].set_index('descriptor').loc[F,'rank_abs_delta_within_scope'],'24 CRISP parents vs 1462 ET','SAME_MODEL_DIFFERENT_INPUT_SUPPORT; not independent validation')
    csvout(pd.DataFrame(ranks),'FEATURE_RANK_AGREEMENT.csv')
    grades=[]
    for f in F:
        if f in ['valence_electrons','electron_affinity','ionization_energy']:
            grade='PARTIALLY_SUPPORTED';reason='formal DFT marginal association and high ET model rank; joint chemistry and numerical resolution prevent independent rank validation'
        elif f in ['bond_length','bond_angle']:
            grade='PARTIALLY_SUPPORTED';reason='actual local intervention changes global descriptor and model allocation; L/Theta co-vary and DFT response numerical resolution is unresolved'
        else:
            grade='UNRESOLVED';reason='ET model allocation exists but formal DFT marginal interval overlaps zero or attribution is confounded; no independent importance certification'
        grades.append({'descriptor':f,'scientific_grade':grade,'reason':reason,'ET_rank':int(et.loc[f,'rank']),'LIME_rank':int(lime.loc[f,'rank']),
                       'DFT_marginal_rank_atomic7':int(element_rank[f]) if f in element_rank else 'NA',
                       'CRISP_delta_rank_all':int(summary[(summary.scope=='all')&(summary.descriptor==f)].rank_abs_delta_within_scope.iloc[0]),
                       'eligible_main_text':'bounded descriptive association only' if grade=='PARTIALLY_SUPPORTED' else 'model ranking only with limitation',
                       'independent_DFT_validation':'NO'})
    csvout(pd.DataFrame(grades),'FEATURE_BY_FEATURE_VALIDATION.csv')
    # Original attribution allocation correlations are a diagnostic for shared feature support.
    phicorr=parent[['SHAP_'+f for f in F]].corr(method='spearman')
    phicorr.index=F;phicorr.columns=F;phicorr.to_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ET_ATTRIBUTION_CORRELATIONS.csv'))
    qa={'source_files_verified':len(manifest),'model_sha_equal':digest(_repo_path('05_CRISP_X/models/property_model.joblib'))==digest(_repo_path('05_CRISP_X/models/property_model.joblib')),
        'n_physical_points':len(d),'n_parent_materials':d.ID.nunique(),'n_parent_seed_pools':d[['ID','seed']].drop_duplicates().shape[0],
        'geometry_points':int((d.kind=='geometry').sum()),'chemical_points':int((d.kind=='chemical').sum()),
        'qualified_geometry_paths':len(q),'chemical_affine_points':int((d.chemical_status=='AFFINE_SUPPORT').sum()),
        'chemical_joint_points':int((d.chemical_status=='JOINT_PATH_MEAN_ONLY').sum()),'chemical_abstain_points':int((d.chemical_status=='ABSTAIN').sum()),
        'max_source_prediction_difference_eV':max_source_delta,'max_contrast_additivity_residual_eV':max_contrast,
        'max_perturbed_reconstruction_residual_eV':max_perturbed,'max_parent_to_perturbed_descriptor_change':input_mismatch,
        'bootstrap_draws':CFG['bootstrap_draws'],'bootstrap_unit':'parent material ID','new_tree_explanations':0,'new_model_fits':0,'new_DFT':0}
    (_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/conditioned_shap/ANALYSIS_QA.json')).write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(qa,ensure_ascii=False))

if __name__=='__main__':main()
