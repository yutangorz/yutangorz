"""Reuse SHAP/CRISP; aggregate new LIME and paired grouped statistics. No fitting."""

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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/native_lime')))
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr,kendalltau
from sklearn.metrics import r2_score

R=_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/native_lime'); S=_repo_path('unavailable_inputs/source_f4b5c8e23ecb/results'); OUT=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison'); OUT.mkdir(exist_ok=False)
F=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']
DISPLAY=['valence_electrons','electron_affinity','ionization_energy','atomic_volume','electronegativity','covalent_radius','period','bond_length','bond_angle']
NAME=dict(zip(DISPLAY,['价电子数','电子亲和能','电离能','元素Atomic Volume','电负性','共价半径','周期','周期壳层距离L','周期壳层角度Theta']))
cfg=json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/native_lime.json')).read_text(encoding='utf-8')); B=cfg['bootstrap']['repeats']
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(df,name):df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig');return df
def boot_means(matrix,groups,seed=20261022):
 labels,ix=np.unique(np.asarray(groups),return_inverse=True);n=np.bincount(ix)
 sums=np.zeros((len(labels),matrix.shape[1]));np.add.at(sums,ix,matrix)
 rng=np.random.default_rng(seed);v=[];indices=[]
 for b in range(B):
  draw=rng.integers(0,len(labels),len(labels));v.append(sums[draw].sum(axis=0)/n[draw].sum());indices.append(draw)
 return np.array(v),np.array(indices),labels
lime=pd.read_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/per_material_lime_1462.csv'),dtype={'ID':str},float_precision='round_trip')
shap=pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/shap_baseline/per_material_nine_attributions.csv'),dtype={'ID':str},float_precision='round_trip')
assert len(lime)==len(shap)==1462 and lime.ID.tolist()==shap.ID.tolist()
assert np.array_equal(lime[F].to_numpy(),shap[F].to_numpy())
assert abs(lime.ET_prediction_eV-shap.prediction_eV).max()<=1e-12
weights=lime[['LIME_weight_'+f for f in F]].to_numpy();sv=shap[['SHAP_'+f for f in F]].to_numpy()
both,draws,groups=boot_means(np.c_[abs(sv),abs(weights)],shap.formula_group.to_numpy())
np.savez_compressed(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/paired_formula_bootstrap.npz'),draw_indices=draws,formula_groups=groups,SHAP_mean_abs=both[:,:9],LIME_mean_abs=both[:,9:])
lb=both[:,9:];lr=rankdata(-lb,axis=1,method='average');lp=abs(weights).mean(axis=0);lrank=rankdata(-lp,method='average')
base=pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/shap_baseline/TABLE_nine_importance.csv')).set_index('feature').loc[DISPLAY].reset_index()
base['n_materials']=1462;base['formula_groups']=933;base['status']='reused verified existing TreeSHAP';base['feature_name']=base.feature.map(NAME)
A=save(base,'Table_A_TreeSHAP')
rows=[]
for j,f in enumerate(F):
 rows.append({'feature':f,'feature_name':NAME[f],'mean_abs_LIME_weight_eV_per_SD':lp[j],'rank':lrank[j],
  'low95':np.quantile(lb[:,j],.025),'high95':np.quantile(lb[:,j],.975),'rank_low95':np.quantile(lr[:,j],.025),'rank_high95':np.quantile(lr[:,j],.975),
  'positive_fraction':(weights[:,j]>0).mean(),'negative_fraction':(weights[:,j]<0).mean(),'zero_fraction':(weights[:,j]==0).mean(),
  'signed_p025':np.quantile(weights[:,j],.025),'signed_median':np.median(weights[:,j]),'signed_p975':np.quantile(weights[:,j],.975),
  'n_materials':1462,'high_fidelity_n':int(lime.high_fidelity.sum()),'low_fidelity_fraction':float(1-lime.high_fidelity.mean()),
  'definition':'Ridge slope eV per training SD; Gaussian LIME; not SHAP point contribution'})
Btab=save(pd.DataFrame(rows).set_index('feature').loc[DISPLAY].reset_index(),'Table_B_LIME')
sp=abs(sv).mean(axis=0);srank=rankdata(-sp,method='average')
c=[]
for f in DISPLAY:
 j=F.index(f); a=base.set_index('feature').loc[f];b=Btab.set_index('feature').loc[f]
 c.append({'feature':f,'feature_name':NAME[f],'TreeSHAP_mean_abs_eV':sp[j],'SHAP_rank':srank[j],
  'LIME_mean_abs_weight_eV_per_SD':lp[j],'LIME_rank':lrank[j],'rank_difference_LIME_minus_SHAP':lrank[j]-srank[j],
  'SHAP_rank_low95_existing':a.bootstrap_rank_low,'SHAP_rank_high95_existing':a.bootstrap_rank_high,
  'LIME_rank_low95':b.rank_low95,'LIME_rank_high95':b.rank_high95})
C=save(pd.DataFrame(c),'Table_C_Rank_comparison')
corrs=np.array([[spearmanr(x[:9],x[9:]).statistic,kendalltau(x[:9],x[9:]).statistic] for x in both])
save(pd.DataFrame(corrs,columns=['spearman','kendall_tau']),'Table_C_paired_bootstrap_correlations')
topS=set(np.array(F)[np.argsort(-sp)[:3]]);topL=set(np.array(F)[np.argsort(-lp)[:3]])
summary={'n_materials':1462,'formula_groups':933,'spearman':float(spearmanr(sp,lp).statistic),
 'kendall_tau':float(kendalltau(sp,lp).statistic),'spearman_low95':float(np.quantile(corrs[:,0],.025)),
 'spearman_high95':float(np.quantile(corrs[:,0],.975)),'kendall_low95':float(np.quantile(corrs[:,1],.025)),
 'kendall_high95':float(np.quantile(corrs[:,1],.975)), 'SHAP_top3':sorted(topS),'LIME_top3':sorted(topL),
 'top3_overlap_n':len(topS&topL),'top3_overlap_features':sorted(topS&topL),
 'rank_stability_definition':'formula-group sampling only, one frozen model and one fixed LIME seed per ID; not repeated-seed stability'}
save(pd.DataFrame([summary]),'Table_C_Rank_statistics')
fid_cols=['anchor_error_eV','fit_weighted_r2','fit_weighted_rmse_eV','validation_weighted_r2','validation_weighted_rmse_eV','validation_weighted_mae_eV']
fid=lime[fid_cols].describe(percentiles=[.025,.05,.25,.5,.75,.95,.975]).reset_index().rename(columns={'index':'statistic'})
save(fid,'LIME_Fidelity_distribution');save(lime[['ID','formula_group','high_fidelity']+fid_cols],'LIME_Per_material_fidelity')
hf=lime.high_fidelity.astype(bool).to_numpy();hfinfo={'threshold_fixed_before_LIME':cfg['high_fidelity_predefined'],'high_fidelity_n':int(hf.sum()),'low_fidelity_n':int((~hf).sum()),'fraction_high':float(hf.mean())}
if hf.any():
 sub=abs(weights[hf]).mean(axis=0);boot,subdraw,subgroups=boot_means(abs(weights[hf]),lime.formula_group[hf]);rank=rankdata(-sub,method='average');br=rankdata(-boot,axis=1)
 rows=[{'feature':f,'n':int(hf.sum()),'mean_abs_weight_eV_per_SD':sub[j],'rank':rank[j],'rank_change_vs_all':rank[j]-lrank[j],
  'low95':np.quantile(boot[:,j],.025),'high95':np.quantile(boot[:,j],.975),'rank_low95':np.quantile(br[:,j],.025),'rank_high95':np.quantile(br[:,j],.975)} for j,f in enumerate(F)]
 save(pd.DataFrame(rows),'Table_B_High_fidelity_subset');np.savez_compressed(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/high_fidelity_bootstrap.npz'),draw_indices=subdraw,formula_groups=subgroups,mean_abs_weights=boot)
 hfinfo.update({'subset_formula_groups':len(subgroups),'spearman_subset_vs_all':float(spearmanr(sub,lp).statistic),'note':'selection changes population; small subset not representative'})
else:
 save(pd.DataFrame([{'status':'not_available_no_sample_passes_predefined_threshold','n':0}]),'Table_B_High_fidelity_subset')
dump(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/LIME_FIDELITY_AUDIT.json'),hfinfo)

# D: Preserve different estimands, never fabricate CRISP independent9 rank.
rel=[]
for f in DISPLAY:
 j=F.index(f);rel.append({'feature':f,'SHAP_rank_1462':srank[j],'LIME_rank_1462':lrank[j],'CRISP_independent_nine_rank':'NA',
 'CRISP_available_quantity':'qualified path-local geometric slope' if j>=7 else 'joint chemical path finite response',
 'same_model':True,'same_descriptor_definition':True,'same_population':False,'CRISP_mothers_n':24,'attribution_materials_n':1462,
 'direct_nine_rank_comparison':'not_defined','reason':'different path/scales and slope qualification; cannot convert to SHAP allocation'})
D=save(pd.DataFrame(rel),'Table_D_Attribution_response_relation')
design=pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/query_and_path_design.csv'),dtype={'ID':str});design=design[(design.model=='frozen_ET')&(design.configuration=='block_b64_g40')].copy()
save(design,'Table_D_All_path_parameters')
path=design.groupby('path').agg(parent_n=('ID','nunique'),parent_seed_n=('ID','size'),mean_abs_joint_intercept_eV=('intercept_at_origin',lambda x:abs(x).mean()),
 affine_fraction=('prediction_mode',lambda x:(x=='affine').mean()),separate_slopes_qualified_fraction=('separate_geometry_slopes_qualified','mean')).reset_index()
path['path_response_rank_descriptive']=rankdata(-path.mean_abs_joint_intercept_eV,method='average')
path['comparability']='different finite substitution events and mother support; descriptive only, not9descriptor ranking'
save(path,'Table_D_Joint_path_sensitivity')
for fname in ['TABLE_aligned_rankings.csv','TABLE_rank_agreement.csv','per_response_nine_contrasts.csv','PLOT_rank_correlation_bootstrap.csv','PLOT_importance_bootstrap.csv']:
 source=_repo_path('unavailable_inputs/source_6652757ff62d/aligned_comparison')/fname
 if source.is_file():save(pd.read_csv(source,dtype={'ID':str}),'Reused_response_'+fname[:-4])

# E: re-evaluate saved predictions on explicit pairwise common support only.
resp=pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/response_predictions.csv'),dtype={'ID':str});resp=resp[resp.model=='frozen_ET']
keys=['ID','seed','candidate'];cr=resp[resp.method=='block_b64_g40'].copy();assert not cr.duplicated(keys).any()
methods=['block_b64_g40_strong_baseline','standard_LIME512','constrained_LIME9_matched','constrained_LIME_geometry_matched']
ER=[];paired=[];boots=[]
for method in methods:
 other=resp[resp.method==method];assert not other.duplicated(keys).any()
 merged=cr.merge(other[keys+['truth','prediction','queries']],on=keys,suffixes=('_CRISP','_other'),validate='one_to_one')
 assert abs(merged.truth_CRISP-merged.truth_other).max()<1e-12
 for scope in ['all','geometry','chemical']:
  a=merged if scope=='all' else merged[merged.kind==scope]
  good=np.isfinite(a.prediction_CRISP)&np.isfinite(a.prediction_other);v=a[good].copy()
  if not len(v):continue
  y=v.truth_CRISP.to_numpy();cp=v.prediction_CRISP.to_numpy();op=v.prediction_other.to_numpy()
  gain=abs(y-op)-abs(y-cp);boot,bi,bgroups=boot_means(gain[:,None],v.ID.to_numpy())
  lo,hi=np.quantile(boot[:,0],[.025,.975]);
  for b,value in enumerate(boot[:,0]):boots.append({'other_method':method,'scope':scope,'bootstrap':b,'MAE_gain_eV':value})
  for who,p,qn in [('CRISP-X',cp,64),(method,op,int(v.queries_other.iloc[0]))]:
   ER.append({'comparison_other':method,'scope':scope,'method':who,'planned_n':len(a),'common_n':len(v),
    'common_coverage':len(v)/len(a),'native_coverage':float(np.isfinite(a.prediction_CRISP if who=='CRISP-X' else a.prediction_other).mean()),
    'MAE_eV':float(abs(y-p).mean()),'RMSE_eV':float(np.sqrt(np.mean((y-p)**2))), 'R2':float(r2_score(y,p)) if len(y)>1 else np.nan,
    'queries_fit':qn,'query_note':'CRISP64+parent and192 structure extraction; LIME geometry40; LIME standard512; evaluation queries extra',
    'paired_gain_other_minus_CRISP_eV':float(gain.mean()),'gain_low95':lo,'gain_high95':hi,'resampling_unit':'24 mother IDs, fixed seeds/predictions; no model retraining'})
  for _,row in v.iterrows():paired.append({'other_method':method,'scope':scope,'ID':row.ID,'seed':row.seed,'candidate':row.candidate,
   'truth_delta_eV':row.truth_CRISP,'CRISP_prediction':row.prediction_CRISP,'other_prediction':row.prediction_other,
   'paired_abs_error_gain':abs(row.truth_CRISP-row.prediction_other)-abs(row.truth_CRISP-row.prediction_CRISP)})
E=save(pd.DataFrame(ER),'Table_E_Common_support_fidelity');save(pd.DataFrame(paired),'Table_E_Per_response_paired_errors');save(pd.DataFrame(boots),'Table_E_Bootstrap_improvement')
save(pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/TABLE_response_increment.csv')),'Reused_CRISP_frozen_increment_gates')
save(pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/TABLE_response_metrics.csv')).query("model=='frozen_ET'"),'Reused_Native_response_metrics')
dump(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/SUMMARY.json'),{'rank_comparison':summary,'fidelity':hfinfo,'SHAP_rank_order':np.array(F)[np.argsort(-sp)].tolist(),
 'LIME_rank_order':np.array(F)[np.argsort(-lp)].tolist(),'input_identity_exact':True,'new_SHAP_evaluations':0,'new_CRISP_evaluations':0,'new_LIME_materials':1462})
sheet_tables={'Table A SHAP':A,'Table B LIME':Btab,'Table C ranks':C,'Rank statistics':pd.DataFrame([summary]),
 'Table D relation':D,'Table E response':E,'High fidelity subset':pd.read_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/Table_B_High_fidelity_subset.csv')),
 'LIME fidelity':fid,'Per material fidelity':lime[['ID','formula_group','high_fidelity']+fid_cols],
 'Joint path sensitivity':path,'Response rank comparison':pd.read_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/comparison/Reused_response_TABLE_rank_agreement.csv'))}
bundle=[]
for name,df in sheet_tables.items():
 clean=df.copy()
 for col in clean:
  clean[col]=clean[col].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(list,dict)) else x)
 data=json.loads(clean.to_json(orient='split',index=False,force_ascii=False))
 bundle.append({'sheet':name,'columns':data['columns'],'rows':data['data']})
dump(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/WORKBOOK_TABLES.json'),bundle)
print(json.dumps({'ranking':summary,'fidelity':hfinfo,'new_tables':len(sheet_tables),'response_comparisons_rows':len(E)},ensure_ascii=False,indent=2))
