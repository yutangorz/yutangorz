"""Frozen Figure6 response/direction/amplitude diagnosis; read saved predictions only."""

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
import json,hashlib,shutil,itertools,warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr,rankdata
R=_repo_path('07_DFT_VALIDATION/workflows/direction_magnitude');ROOT=_repo_path('.');SRC=_repo_path('07_DFT_VALIDATION/workflows/node_prediction');I=_repo_path('07_DFT_VALIDATION/data/direction_magnitude');O=_repo_path('outputs/07_DFT_VALIDATION/direction_magnitude')
B=5000;SEED=20261105;THRESHOLDS=[0.5,1.,2.,5.]
COL={'ET':'ET_direct_delta_eV','CRISP-X':'CRISP_X_predicted_delta_eV','DFT':'delta_Eg_eV'}
PAIRS=[('A','CRISP-X','ET'),('B','ET','DFT'),('C','CRISP-X','DFT')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d,name):pd.DataFrame(d).to_csv(O/name,index=False,encoding='utf-8-sig',na_rep='NA')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
def freeze():
 if O.exists():raise RuntimeError('No overwrite: executed run immutable')
 I.mkdir(parents=True,exist_ok=True)
 names=['results/FIG6E_NODE_PREDICTIONS_ALL_440.csv','results/FIG6E_LOCAL_MODEL_DESIGN.csv','results/FIG6E_EXACT_NODE_MODEL_INPUTS.csv',
  'results/FIG6E_RESPONSE_AND_ABSOLUTE_METRICS.csv','results/INDEPENDENT_QA.json','PROTOCOL_FROZEN.json',
  'Documentation_v002/FIG6_DATA_FEASIBILITY_AND_DESIGN_REPORT_CN.md','Documentation_v002/FIG6_PANEL_FEASIBILITY.csv',
  'statistical_results_v002/FIG6D_PRIMARY_DISPLAY_QUALIFICATION.csv','statistical_results_v002/FIG6D_RANK_AGREEMENT_EXPLORATORY.csv']
 rows=[]
 for n in names:
  s=SRC/n;t=I/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,t)
  rows.append({'source':str(s),'snapshot':str(t),'sha256':sha(s),'bytes':s.stat().st_size})
 pd.DataFrame(rows).to_csv(_repo_path('07_DFT_VALIDATION/workflows/direction_magnitude/INPUT_SOURCE_MAPPING.csv'),index=False,encoding='utf-8-sig')
 cfg={'task':'saved-response decomposition, direction first then amplitude; no prediction/training/DFT/plotting',
 'primary_DFT_scope':'original371/17; common local box124/10 and outside247 kept unchanged; no selection by outcomes',
 'bootstrap':{'B':B,'seed':SEED,'unit':'whole parent within each stated stratum; point weights and equal-parent weights reported separately',
  'display_gate':'at least3 contributing parents; undefined draws stayNA; finite-draw counts reported; no bootstrap p-values'},
 'zero_policy':'exact saved float0 is separate ternary state; never recode near-zero as0; binary truth excludes exact truezero only; zero prediction on nonzero truth counted as directional miss',
 'baselines':'full371 majority sign fixed for all strata; subset in-sample best constant reported with optimism caveat; LOPO majority learned from other parents inside same stratum, ties prefer negative, one-parent prediction NA',
 'thresholds_meV':THRESHOLDS,'threshold_policy':'all four cumulative >= thresholds and disjoint bins; primary amplitude-ratio support>=1meV; thresholds are sensitivity choices not DFT error bounds',
 'amplitude_scopes':'all, direction-concordant, and discordant; distance/angle; box in/out; exact same original node qualification',
 'slope_definition':'predicted response on observedDFT response, with intercept and through origin; magnitude slope similarly; diagnostic only, never applied calibration',
 'ratio_policy':'abs(pred)/abs(DFT) only on saved absDFT>=threshold; no unthresholded nearzero ratio',
 'claims':'overall and strict-domain claims separated; no50percent chance test; compare paired majority/LOPO advantage and BA-.5 with parentCI; threshold and leave-parent sensitivity all disclosed; no threshold selection for claims',
 'multiplicity':'one primary in-box direction contrast plus whole371 descriptive; other strata/thresholds exploratory pointwise CIs, no selected-significance claim',
 'script_sha256':sha(__file__)}
 dump(_repo_path('07_DFT_VALIDATION/configs/direction_magnitude_protocol.json'),cfg);O.mkdir()

def directional(y,p):
 sy=np.sign(y);sp=np.sign(p);pos=sy>0;neg=sy<0;nz=pos|neg
 tp=int(sum(pos&(sp>0)));tn=int(sum(neg&(sp<0)));fp=int(sum(neg&(sp>0)));fn=int(sum(pos&(sp<0)))
 pr=tp/sum(pos) if pos.any() else np.nan;nr=tn/sum(neg) if neg.any() else np.nan
 return dict(n=len(y),true_positive_n=int(sum(pos)),true_negative_n=int(sum(neg)),true_zero_n=int(sum(sy==0)),
  predicted_positive_n=int(sum(sp>0)),predicted_negative_n=int(sum(sp<0)),predicted_zero_n=int(sum(sp==0)),
  TP=tp,TN=tn,FP=fp,FN=fn,zero_prediction_on_positive=int(sum(pos&(sp==0))),zero_prediction_on_negative=int(sum(neg&(sp==0))),
  true_zero_predicted_nonzero=int(sum((sy==0)&(sp!=0))),true_zero_predicted_zero=int(sum((sy==0)&(sp==0))),
  direction_agreement=float(np.mean(sy==sp)) if len(y) else np.nan,positive_recall=pr,negative_recall=nr,
  balanced_accuracy=(pr+nr)/2 if pos.any() and neg.any() else np.nan,
  binary_true_nonzero_n=int(sum(nz)),binary_true_nonzero_accuracy=float(np.mean(sy[nz]==sp[nz])) if nz.any() else np.nan)

def comparison(g,pm,tm):
 p=g[COL[pm]].to_numpy();y=g[COL[tm]].to_numpy();ok=np.isfinite(y)&np.isfinite(p);p=p[ok];y=y[ok]
 out=dict(n_total=len(g),n_finite=int(sum(ok)),n_parents=g.loc[ok,'material_id'].nunique(),
  MAE_eV=float(np.mean(abs(p-y))),RMSE_eV=float(np.sqrt(np.mean((p-y)**2))),
  R2=1-float(np.sum((p-y)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else np.nan,
  pearson_r=float(pearsonr(y,p).statistic) if len(y)>2 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan,
  spearman_rho=float(spearmanr(y,p).statistic) if len(y)>2 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan)
 out.update(directional(y,p));return out

def majority(y):
 s=np.sign(y);return 1 if sum(s>0)>sum(s<0) else -1
def baselines(g,full_sign):
 y=g.delta_Eg_eV.to_numpy();best=majority(y);lopo=np.full(len(g),np.nan)
 for mid in g.material_id.unique():
  tr=g[g.material_id!=mid]
  if len(tr):lopo[g.material_id.to_numpy()==mid]=majority(tr.delta_Eg_eV.to_numpy())
 return {'GLOBAL_CONSTANT':np.full(len(g),full_sign),'SUBSET_BEST_CONSTANT':np.full(len(g),best),'LOPO_MAJORITY':lopo}

def weighted_median(values,weights):
 order=np.argsort(values);v=values[order];w=weights[:,order];cs=np.cumsum(w,axis=1);n=cs[:,-1]
 # Exact weighted replication median, including average of middle two at even N.
 lower=(np.maximum(1,np.ceil(n/2)));upper=np.floor(n/2)+1
 il=(cs>=lower[:,None]).argmax(axis=1);iu=(cs>=upper[:,None]).argmax(axis=1)
 return np.where(n>0,(v[il]+v[iu])/2,np.nan)

def bootstrap(g,scope,full_sign):
 if len(g)==0:return [],[],[]
 ids=np.sort(g.material_id.unique());n=len(ids);rng=np.random.default_rng(SEED)
 counts=rng.multinomial(n,[1/n]*n,size=B);ix=np.searchsorted(ids,g.material_id.to_numpy());w=counts[:,ix].astype(float);den=w.sum(axis=1)
 y=g.delta_Eg_eV.to_numpy();p=g.CRISP_X_predicted_delta_eV.to_numpy();e=g.ET_direct_delta_eV.to_numpy();sy=np.sign(y);sp=np.sign(p)
 def avg(v):return (w@np.asarray(v,float))/den
 def ba(pred):
  return .5*((w@((sy>0)&(pred>0)))/(w@(sy>0))+(w@((sy<0)&(pred<0)))/(w@(sy<0)))
 base=baselines(g,full_sign);acc=avg(sy==sp);bal=ba(sp)
 parent_acc=g.assign(a=sy==sp).groupby('material_id').a.mean().reindex(ids).to_numpy()
 vals={'direction_agreement':acc,'equal_parent_direction_agreement':counts@parent_acc/n,'balanced_accuracy':bal,
  'balanced_accuracy_minus_half':bal-.5,'CRISP_MAE_eV':avg(abs(p-y)),'ET_MAE_eV':avg(abs(e-y)),
  'MAE_increase_CRISP_over_ET_eV':avg(abs(p-y)-abs(e-y)),
  'mean_signed_bias_eV':avg(p-y),'mean_absolute_response_difference_eV':avg(abs(p)-abs(y)),
  'median_absolute_response_difference_eV':weighted_median(abs(p)-abs(y),w)}
 for name,b in base.items():
  if np.isfinite(b).all():
   vals['advantage_vs_'+name]=acc-avg(sy==b)
   vals['BA_advantage_vs_'+name]=bal-ba(b)
   parent_diff=g.assign(z=(sy==sp).astype(float)-(sy==b).astype(float)).groupby('material_id').z.mean().reindex(ids).to_numpy()
   vals['equal_parent_advantage_vs_'+name]=counts@parent_diff/n
 # Uncorrected diagnostic regressions: predicted CRISP on observed DFT, never used to change predictions.
 for prefix,x,z in [('signed',y,p),('magnitude',abs(y),abs(p)),('ET_magnitude',abs(y),abs(e))]:
  mx=avg(x);mz=avg(z);xx=avg(x*x);xz=avg(x*z)
  vals[prefix+'_slope_with_intercept']=(xz-mx*mz)/(xx-mx*mx)
  vals[prefix+'_slope_through_origin']=xz/xx
  vals[prefix+'_intercept_eV']=mz-vals[prefix+'_slope_with_intercept']*mx
 if scope.startswith('THRESHOLD_') or scope.startswith('CONCORDANT_THRESHOLD_'):
  vals['median_absolute_response_ratio']=weighted_median(abs(p)/abs(y),w)
 raw=pd.DataFrame({'scope':scope,'replicate':np.arange(B),'sampled_node_n':den,'n_source_parents':n,**vals})
 ci=[]
 for name,v in vals.items():
  good=np.asarray(v)[np.isfinite(v)];display=n>=3 and len(good)>=.8*B
  ci.append({'scope':scope,'metric':name,'n_nodes':len(g),'n_parents':n,'B':B,'valid_draws':len(good),
   'low95':np.quantile(good,.025) if display else np.nan,'high95':np.quantile(good,.975) if display else np.nan,
   'conditional_low95':np.quantile(good,.025) if len(good) else np.nan,'conditional_high95':np.quantile(good,.975) if len(good) else np.nan,
   'qualification':'POINTWISE_EXPLORATORY' if display else 'SUPPRESSED_INSUFFICIENT_PARENT_OR_VALID_DRAWS'})
 return raw,ci,counts

def amplitude(g,scope,method='CRISP-X',ratio_threshold=None):
 y=g.delta_Eg_eV.to_numpy();p=g[COL[method]].to_numpy();err=p-y;amp=abs(p)-abs(y)
 result={'scope':scope,'method':method,'n_nodes':len(g),'n_parents':g.material_id.nunique(),
  'mean_signed_error_eV':np.mean(err) if len(g) else np.nan,'median_signed_error_eV':np.median(err) if len(g) else np.nan,
  'mean_absolute_response_difference_eV':np.mean(amp) if len(g) else np.nan,'median_absolute_response_difference_eV':np.median(amp) if len(g) else np.nan,
  'overestimated_amplitude_n':int(sum(amp>0)),'underestimated_amplitude_n':int(sum(amp<0)),'equal_amplitude_n':int(sum(amp==0))}
 for prefix,x,z in [('signed',y,p),('magnitude',abs(y),abs(p))]:
  den=np.sum((x-x.mean())**2) if len(x) else 0
  slope=np.sum((x-x.mean())*(z-z.mean()))/den if den>0 else np.nan
  result[prefix+'_slope_with_intercept']=slope;result[prefix+'_intercept_eV']=z.mean()-slope*x.mean() if len(x) else np.nan
  result[prefix+'_slope_through_origin']=float(x@z/(x@x)) if len(x) and x@x>0 else np.nan
 if ratio_threshold is not None:
  assert (abs(y)>=ratio_threshold/1000-1e-14).all()
  rat=abs(p)/abs(y)
  for name,q in [('q1',.25),('median',.5),('q3',.75)]:result['absolute_response_ratio_'+name]=np.quantile(rat,q) if len(rat) else np.nan
  result['ratio_threshold_meV']=ratio_threshold
 return result

def main():
 freeze();warnings.filterwarnings('ignore',category=RuntimeWarning)
 d=pd.read_csv(_repo_path('07_DFT_VALIDATION/data/direction_magnitude/results/FIG6E_NODE_PREDICTIONS_ALL_440.csv'));m=d[d.primary_distribution].copy()
 assert len(d)==440 and d.material_id.nunique()==20 and len(m)==371 and m.material_id.nunique()==17
 assert (~m.outside_local_box_any).sum()==124 and m.outside_local_box_any.sum()==247
 for v in COL.values():assert np.isfinite(d[v]).all()
 full_sign=majority(m.delta_Eg_eV.to_numpy())
 d['signed_error_CRISP_minus_DFT_eV']=d[COL['CRISP-X']]-d[COL['DFT']]
 d['signed_error_ET_minus_DFT_eV']=d[COL['ET']]-d[COL['DFT']]
 d['signed_error_CRISP_minus_ET_eV']=d[COL['CRISP-X']]-d[COL['ET']]
 d['absolute_response_difference_CRISP_minus_DFT_eV']=abs(d[COL['CRISP-X']])-abs(d[COL['DFT']])
 d['absolute_response_difference_ET_minus_DFT_eV']=abs(d[COL['ET']])-abs(d[COL['DFT']])
 d['direction_concordant_CRISP_DFT']=np.sign(d[COL['CRISP-X']])==np.sign(d[COL['DFT']])
 d['absolute_DFT_response_meV']=abs(d[COL['DFT']])*1000
 for t in THRESHOLDS:
  d[f'absolute_response_ratio_CRISP_DFT_ge_{t:g}meV']=np.where(d.absolute_DFT_response_meV>=t,abs(d[COL['CRISP-X']])/abs(d[COL['DFT']]),np.nan)
 save(d,'FIG6E_NODE_DIAGNOSTICS_ALL_440.csv');m=d[d.primary_distribution].copy()
 scopes={'ALL_FORMAL_371':m,'IN_BOX_124':m[~m.outside_local_box_any],'OUT_BOX_247':m[m.outside_local_box_any],
  'LOCAL_DISTANCE':m[m.perturbation_type=='LENGTH'],'LOCAL_ANGLE':m[m.perturbation_type=='ANGLE']}
 scopes.update({'PROTOCOL_'+str(k):v for k,v in m.groupby('task_mode')})
 for box,mask in [('IN_BOX',~m.outside_local_box_any),('OUT_BOX',m.outside_local_box_any)]:
  for kind in ['LENGTH','ANGLE']:scopes[box+'_'+kind]=m[mask&(m.perturbation_type==kind)]
 comparisons=[];conf=[];perparent=[];perstrata=[];distribution=[];amps=[];decomp=[];concentr=[]
 for scope,g in scopes.items():
  for label,p,t in PAIRS:
   row={'scope':scope,'comparison':label,'prediction':p,'reference':t,**comparison(g,p,t)};comparisons.append(row)
   sy=np.sign(g[COL[t]].to_numpy());sp=np.sign(g[COL[p]].to_numpy())
   for a,b in itertools.product([-1,0,1],repeat=2):conf.append({'scope':scope,'comparison':label,'true_direction':a,'predicted_direction':b,'n':int(sum((sy==a)&(sp==b)))})
  row={'scope':scope,'n_parents':g.material_id.nunique(),**directional(g.delta_Eg_eV.to_numpy(),g.CRISP_X_predicted_delta_eV.to_numpy())}
  for name,b in baselines(g,full_sign).items():
   ok=np.isfinite(b);acc=np.mean(np.sign(g.delta_Eg_eV.to_numpy()[ok])==b[ok]) if ok.any() else np.nan
   row[name+'_accuracy']=acc;row[name+'_accuracy_advantage']=row['direction_agreement']-acc
  row['subset_best_constant_direction']=majority(g.delta_Eg_eV.to_numpy());perstrata.append(row)
  for method in COL:
   v=g[COL[method]].to_numpy();a=abs(v)
   distribution.append({'scope':scope,'method':method,'n_nodes':len(g),'n_parents':g.material_id.nunique(),'mean_eV':np.mean(v),'SD_eV_ddof1':np.std(v,ddof=1),
    'median_absolute_response_eV':np.median(a),**{f'q{q:g}_eV':np.quantile(v,q) for q in [.025,.25,.5,.75,.975]},
    **{f'absolute_q{q:g}_eV':np.quantile(a,q) for q in [.025,.25,.5,.75,.975]}})
  for method in ['CRISP-X','ET']:amps.append(amplitude(g,scope,method))
  et=g.ET_direct_delta_eV.to_numpy()-g.delta_Eg_eV.to_numpy();local=g.CRISP_X_predicted_delta_eV.to_numpy()-g.ET_direct_delta_eV.to_numpy();total=et+local
  decomp.append({'scope':scope,'n_nodes':len(g),'n_parents':g.material_id.nunique(),'MSE_CRISP_DFT_eV2':np.mean(total**2),'MSE_ET_DFT_eV2':np.mean(et**2),
   'MSE_CRISP_ET_eV2':np.mean(local**2),'cross_term_eV2':2*np.mean(et*local),'MAE_increase_local_over_ET_eV':np.mean(abs(total)-abs(et)),
   'ET_exact_zero_n':int(sum(g.ET_direct_delta_eV==0))})
 for mid,g in m.groupby('material_id'):
  row={'material_id':mid,**directional(g.delta_Eg_eV.to_numpy(),g.CRISP_X_predicted_delta_eV.to_numpy()),
   'DFT_positive_fraction':np.mean(g.delta_Eg_eV>0),'CRISP_positive_fraction':np.mean(g.CRISP_X_predicted_delta_eV>0),
   'positive_fraction_prediction_bias':np.mean(g.CRISP_X_predicted_delta_eV>0)-np.mean(g.delta_Eg_eV>0),
   'in_box_n':int(sum(~g.outside_local_box_any)),'ET_exact_zero_n':int(sum(g.ET_direct_delta_eV==0))}
  for name,b in baselines(m,full_sign).items():
   bb=b[m.material_id.to_numpy()==mid];row[name+'_accuracy']=np.mean(np.sign(g.delta_Eg_eV)==bb)
  perparent.append(row)
 save(comparisons,'FIG6E_ET_CRISPX_DFT_COMPARISON.csv');save(conf,'FIG6E_DIRECTION_CONFUSION_MATRIX.csv')
 save(perparent,'FIG6E_DIRECTION_BY_PARENT.csv');save(perstrata,'FIG6E_DIRECTION_BY_PERTURBATION.csv')
 save(distribution,'FIG6E_RESPONSE_DISTRIBUTIONS.csv');save(decomp,'FIG6E_MSE_DECOMPOSITION.csv')
 # Stage 2 directional thresholds fixed before calculating amplitude diagnostics.
 threshold=[];threshold_scopes={}
 for base in ['ALL_FORMAL_371','IN_BOX_124','OUT_BOX_247']:
  g=scopes[base]
  for t in THRESHOLDS:
   h=g[abs(g.delta_Eg_eV)>=t/1000];name=f'THRESHOLD_{base}_GE_{t:g}meV';threshold_scopes[name]=h
   row={'scope':base,'selection':'cumulative_ge','threshold_meV':t,'n_parents':h.material_id.nunique(),**directional(h.delta_Eg_eV.to_numpy(),h.CRISP_X_predicted_delta_eV.to_numpy())}
   if len(h):
    for bn,bp in baselines(h,full_sign).items():row[bn+'_accuracy']=np.mean(np.sign(h.delta_Eg_eV)==bp) if np.isfinite(bp).all() else np.nan
    cc=h.groupby('material_id').size().sort_values(ascending=False);row.update(largest_parent_share=cc.iloc[0]/len(h),top3_parent_share=cc.head(3).sum()/len(h))
   threshold.append(row)
   for mid,k in h.groupby('material_id'):concentr.append({'scope':base,'threshold_meV':t,'material_id':mid,'n':len(k),'share':len(k)/len(h)})
  edges=[0,.5,1,2,5,np.inf]
  for lo,hi in zip(edges[:-1],edges[1:]):
   h=g[(abs(g.delta_Eg_eV)*1000>=lo)&(abs(g.delta_Eg_eV)*1000<hi)]
   threshold.append({'scope':base,'selection':'disjoint_bin','lower_meV':lo,'upper_meV':hi,'n_parents':h.material_id.nunique(),**directional(h.delta_Eg_eV.to_numpy(),h.CRISP_X_predicted_delta_eV.to_numpy())})
 save(threshold,'FIG6E_DIRECTION_THRESHOLD_SENSITIVITY.csv');save(concentr,'FIG6E_THRESHOLD_MATERIAL_SUPPORT.csv')
 boots=[];cis=[]
 for name,g in {**scopes,**threshold_scopes}.items():
  if not len(g):continue
  raw,ci,_=bootstrap(g,name,full_sign);boots.append(raw);cis+=ci
 save(pd.concat(boots,ignore_index=True),'FIG6E_DIRECTION_BOOTSTRAP.csv')
 save(cis,'FIG6E_BOOTSTRAP_INTERVALS.csv')
 # Stage3 amplitude: whole sample always retained; sign-concordance is diagnostic, not validation subset.
 for base in ['ALL_FORMAL_371','IN_BOX_124','OUT_BOX_247','LOCAL_DISTANCE','LOCAL_ANGLE']:
  g=scopes[base];h=g[g.direction_concordant_CRISP_DFT]
  for method in ['CRISP-X','ET']:amps.append(amplitude(h,'CONCORDANT_'+base,method))
 for name,g in threshold_scopes.items():
  t=float(name.rsplit('_',1)[1][:-3])
  for subset,h in [(name,g),('CONCORDANT_'+name,g[g.direction_concordant_CRISP_DFT])]:
   for method in ['CRISP-X','ET']:amps.append(amplitude(h,subset,method,t))
 save(amps,'FIG6E_AMPLITUDE_BIAS_AND_RATIOS.csv')
 aboot=[];aci=[]
 for base in ['ALL_FORMAL_371','IN_BOX_124','OUT_BOX_247']:
  g=scopes[base];h=g[g.direction_concordant_CRISP_DFT]
  raw,ci,_=bootstrap(h,'CONCORDANT_'+base,full_sign);aboot.append(raw);aci+=ci
  for t in THRESHOLDS:
   h=g[g.direction_concordant_CRISP_DFT&(abs(g.delta_Eg_eV)>=t/1000)]
   if len(h):
    raw,ci,_=bootstrap(h,f'CONCORDANT_THRESHOLD_{base}_GE_{t:g}meV',full_sign);aboot.append(raw);aci+=ci
 save(pd.concat(aboot,ignore_index=True),'FIG6E_AMPLITUDE_BOOTSTRAP.csv');save(aci,'FIG6E_AMPLITUDE_BOOTSTRAP_INTERVALS.csv')
 # Leave-one-parent checks and contribution ledgers; never remove these parents from primary results.
 jack=[];contributions=[]
 for base in ['ALL_FORMAL_371','IN_BOX_124']:
  g=scopes[base];err=g.CRISP_X_predicted_delta_eV-g.delta_Eg_eV
  for mid,h in g.groupby('material_id'):
   er=h.CRISP_X_predicted_delta_eV-h.delta_Eg_eV
   contributions.append({'scope':base,'material_id':mid,'n_nodes':len(h),'absolute_error_share':sum(abs(er))/sum(abs(err)),
    'squared_error_share':sum(er**2)/sum(err**2),'median_DFT_abs_meV':np.median(abs(h.delta_Eg_eV))*1000,'median_CRISP_abs_meV':np.median(abs(h.CRISP_X_predicted_delta_eV))*1000})
   k=g[g.material_id!=mid];b=baselines(k,full_sign);r={'scope':base,'omitted_parent':mid,**amplitude(k,'LEAVE_ONE_OUT')}
   ds=directional(k.delta_Eg_eV.to_numpy(),k.CRISP_X_predicted_delta_eV.to_numpy());r.update(ds)
   r['advantage_vs_subset_best_constant']=ds['direction_agreement']-np.mean(np.sign(k.delta_Eg_eV)==b['SUBSET_BEST_CONSTANT'])
   jack.append(r)
 save(jack,'FIG6E_LEAVE_ONE_PARENT_OUT.csv');save(contributions,'FIG6E_PARENT_ERROR_CONTRIBUTIONS.csv')
 diagnostics=[]
 for mid,g in m.groupby('material_id'):
  diagnostics.append({'material_id':mid,'n':len(g),'all_global_LTheta_cochange':bool(((abs(g.delta_L_A)>1e-10)&(abs(g.delta_Theta_degree)>1e-10)).all()),
   'local_length_excursion_A':g.actual_delta_bond_A.max()-g.actual_delta_bond_A.min(),'global_L_excursion_A':g.delta_L_A.max()-g.delta_L_A.min(),
   'local_angle_excursion_deg':g.actual_delta_angle_degree.max()-g.actual_delta_angle_degree.min(),'global_Theta_excursion_deg':g.delta_Theta_degree.max()-g.delta_Theta_degree.min(),
   'ET_zero_fraction':np.mean(g.ET_direct_delta_eV==0),'outside_box_fraction':np.mean(g.outside_local_box_any),
   'spearman_DFT_global_L':spearmanr(g.delta_L_A,g.delta_Eg_eV).statistic,'spearman_DFT_global_Theta':spearmanr(g.delta_Theta_degree,g.delta_Eg_eV).statistic})
 save(diagnostics,'FIG6E_AGGREGATION_PATH_DIAGNOSTICS.csv')
 source=pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/direction_magnitude/INPUT_SOURCE_MAPPING.csv'));assert all(sha(v.source)==sha(v.snapshot)==v.sha256 for v in source.itertuples())
 old=pd.read_csv(_repo_path('07_DFT_VALIDATION/data/direction_magnitude/results/FIG6E_RESPONSE_AND_ABSOLUTE_METRICS.csv')).query("scope=='PRIMARY_FORMAL_NON_SWITCH' and variable=='RESPONSE_DELTA'").iloc[0]
 cur=pd.DataFrame(comparisons).query("scope=='ALL_FORMAL_371' and comparison=='C'").iloc[0]
 assert max(abs(cur.MAE_eV-old.MAE_eV),abs(cur.RMSE_eV-old.RMSE_eV),abs(cur.R2-old.R2),abs(cur.direction_agreement-old.numeric_direction_agreement))<1e-12
 dump(_repo_path('outputs/07_DFT_VALIDATION/direction_magnitude/EXECUTION_QA.json'),{'source_SHA_unchanged':True,'n_all':len(d),'n_main':len(m),'n_main_parents':17,'n_inbox':124,'n_outbox':247,
  'reference_metrics_reproduced':True,'baseline_global_majority_direction':full_sign,'bootstrap_B':B,'bootstrap_seed':SEED,
  'python':__import__('sys').version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__,
  'new_model_fit_or_inference':0,'DFT_tasks':0,'formal_plots':0})
 print(pd.DataFrame(perstrata)[['scope','n','n_parents','direction_agreement','balanced_accuracy','SUBSET_BEST_CONSTANT_accuracy','LOPO_MAJORITY_accuracy']].to_string(index=False))
 print(pd.DataFrame(comparisons).query("scope=='ALL_FORMAL_371'")[['comparison','MAE_eV','RMSE_eV','R2','pearson_r','spearman_rho','direction_agreement']].to_string(index=False))

if __name__=='__main__':main()
