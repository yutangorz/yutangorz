"""Additional conservative baseline and no-change-response diagnostics; no model use."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/direction_magnitude')))
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=_repo_path('07_DFT_VALIDATION/workflows/direction_magnitude');O=_repo_path('outputs/07_DFT_VALIDATION/direction_magnitude');B=5000;SEED=20261105
if (_repo_path('outputs/07_DFT_VALIDATION/direction_magnitude/FIG6E_RESAMPLED_ORACLE_BASELINE_INTERVALS.csv')).exists():raise RuntimeError('No overwrite')
(_repo_path('07_DFT_VALIDATION/workflows/direction_magnitude/SUPPLEMENT_PREEXEC.json')).write_text(json.dumps({'script_sha256':hashlib.sha256(_repo_path('07_DFT_VALIDATION/src/direction_magnitude/supplement.py').read_bytes()).hexdigest(),
 'purpose':'In-sample best-constant chosen anew inside each bootstrap is conservative oracle sensitivity, not a deployable predictor; zero-response amplitude baseline; threshold parent jackknife',
 'no_threshold_selection':'retain all four thresholds; no modifications to analyze.py or its sealed outputs'},indent=2),encoding='utf-8')
d=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/direction_magnitude/FIG6E_NODE_DIAGNOSTICS_ALL_440.csv'));m=d[d.primary_distribution]
groups={'ALL_FORMAL_371':m,'IN_BOX_124':m[~m.outside_local_box_any],'OUT_BOX_247':m[m.outside_local_box_any],
 'LOCAL_DISTANCE':m[m.perturbation_type=='LENGTH'],'LOCAL_ANGLE':m[m.perturbation_type=='ANGLE']}
for base in ['ALL_FORMAL_371','IN_BOX_124','OUT_BOX_247']:
 for t in [.5,1,2,5]:groups[f'THRESHOLD_{base}_GE_{t:g}meV']=groups[base][abs(groups[base].delta_Eg_eV)>=t/1000]
rows=[];raw=[];macros=[];zero=[];jack=[];parentamps=[]
for name,g in groups.items():
 if not len(g):continue
 ids=np.sort(g.material_id.unique());n=len(ids);counts=np.random.default_rng(SEED).multinomial(n,[1/n]*n,size=B)
 ix=np.searchsorted(ids,g.material_id.to_numpy());w=counts[:,ix];den=w.sum(axis=1)
 y=g.delta_Eg_eV.to_numpy();p=g.CRISP_X_predicted_delta_eV.to_numpy();sy=np.sign(y);sp=np.sign(p)
 pos=w@(sy>0);neg=w@(sy<0);sgn=np.where(pos>neg,1,-1)
 agree=w@(sy==sp)/den;oracle=np.maximum(pos,neg)/den;adv=agree-oracle
 parent_crisp=g.assign(v=sy==sp).groupby('material_id').v.mean().reindex(ids).to_numpy()
 parent_pos=g.assign(v=sy>0).groupby('material_id').v.mean().reindex(ids).to_numpy()
 parent_neg=g.assign(v=sy<0).groupby('material_id').v.mean().reindex(ids).to_numpy()
 macro=counts@parent_crisp/n
 macrobase=np.where(sgn>0,counts@parent_pos/n,counts@parent_neg/n)
 for metric,v in [('advantage_vs_resample_best_constant',adv),('equal_parent_advantage_vs_resample_best_constant',macro-macrobase)]:
  rows.append({'scope':name,'metric':metric,'n_nodes':len(g),'n_parents':n,'low95':np.quantile(v,.025) if n>=3 else np.nan,
    'high95':np.quantile(v,.975) if n>=3 else np.nan,'B':B,'interpretation':'exploratory in-sample oracle baseline sensitivity; not test-set-trained CRISP'})
  raw.append(pd.DataFrame({'scope':name,'metric':metric,'replicate':np.arange(B),'value':v}))
 sign=1 if sum(sy>0)>sum(sy<0) else -1
 macros.append({'scope':name,'n_nodes':len(g),'n_parents':n,'equal_parent_direction_agreement':np.mean(parent_crisp),
  'equal_parent_best_constant_accuracy':np.mean(parent_pos if sign>0 else parent_neg),'majority_constant_sign':sign,
  'equal_parent_best_constant_advantage':np.mean(parent_crisp)-np.mean(parent_pos if sign>0 else parent_neg)})
 denr=np.sum((y-y.mean())**2)
 zero.append({'scope':name,'n_nodes':len(g),'n_parents':n,'zero_response_MAE_eV':np.mean(abs(y)),'zero_response_RMSE_eV':np.sqrt(np.mean(y*y)),
  'zero_response_R2':1-np.sum(y*y)/denr if denr else np.nan,'CRISP_minus_zero_MAE_eV':np.mean(abs(p-y)-abs(y)),
  'ET_minus_zero_MAE_eV':np.mean(abs(g.ET_direct_delta_eV-y)-abs(y))})
 if name.startswith('THRESHOLD_'):
  for mid in ids:
   h=g[g.material_id!=mid]
   if not len(h):continue
   yy=h.delta_Eg_eV.to_numpy();pp=h.CRISP_X_predicted_delta_eV.to_numpy();rat=abs(pp)/abs(yy)
   jack.append({'scope':name,'omitted_parent':mid,'remaining_nodes':len(h),'remaining_parents':h.material_id.nunique(),
    'direction_agreement':np.mean(np.sign(yy)==np.sign(pp)),'advantage_vs_best_constant':np.mean(np.sign(yy)==np.sign(pp))-max(np.mean(yy>0),np.mean(yy<0)),
    'median_absolute_response_ratio':np.median(rat),'mean_absolute_response_difference_eV':np.mean(abs(pp)-abs(yy)),
    'magnitude_slope_through_origin':np.sum(abs(yy)*abs(pp))/np.sum(yy*yy)})
 for mid,h in g.groupby('material_id'):
  for method,col in [('CRISP-X','CRISP_X_predicted_delta_eV'),('ET','ET_direct_delta_eV')]:
   yy=h.delta_Eg_eV.to_numpy();pp=h[col].to_numpy();r={'scope':name,'material_id':mid,'method':method,'n_nodes':len(h),
     'mean_signed_error_eV':np.mean(pp-yy),'mean_absolute_response_difference_eV':np.mean(abs(pp)-abs(yy)),
     'median_abs_pred_eV':np.median(abs(pp)),'median_abs_DFT_eV':np.median(abs(yy))}
   if name.startswith('THRESHOLD_'):r['median_absolute_response_ratio']=np.median(abs(pp)/abs(yy))
   parentamps.append(r)
for df,name in [(rows,'FIG6E_RESAMPLED_ORACLE_BASELINE_INTERVALS.csv'),(pd.concat(raw,ignore_index=True),'FIG6E_RESAMPLED_ORACLE_BASELINE_DRAWS.csv'),
 (macros,'FIG6E_EQUAL_PARENT_DIRECTION_SUMMARY.csv'),(zero,'FIG6E_ZERO_RESPONSE_BASELINE.csv'),
 (jack,'FIG6E_THRESHOLD_LEAVE_ONE_PARENT_OUT.csv'),(parentamps,'FIG6E_AMPLITUDE_BY_PARENT.csv')]:
 pd.DataFrame(df).to_csv(O/name,index=False,encoding='utf-8-sig',na_rep='NA')
print(pd.DataFrame(rows).query("scope=='IN_BOX_124' or scope=='THRESHOLD_IN_BOX_124_GE_1meV'").to_string(index=False))
