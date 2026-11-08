
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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/grouped_ablation')))
from pathlib import Path
import sys,json,hashlib,time
import pandas as pd,numpy as np,joblib
from scipy.stats import spearmanr
R=_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/grouped_ablation');I=_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation');O=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation');C=json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/grouped_ablation.json')).read_text());F=C['features'];ORDER=C['display_order'];sys.path.insert(0,str(I));from explain_tree import explain_tree
assert json.loads((_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_REPRODUCIBILITY.json')).read_text())['status']=='F_DATA_VALIDATED_NO_DRAWING_YET'
T=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA');T.mkdir(exist_ok=False)
d=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/data.csv'),dtype={'ID':str});fold=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/folds.csv'),dtype={'ID':str});d=d.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one');old=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/oof.csv'),dtype={'ID':str}).set_index('ID').loc[d.ID].reset_index();et=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/ET_SHAP.csv'),dtype={'ID':str});et=et.sort_values('ID');assert len(et)==1462 and et.ID.is_unique and set(et.ID)==set(d[d.outer_fold==0].ID)
canon=d.set_index('ID').loc[et.ID];assert np.allclose(et[F+['band_gap_eV']],canon[F+['band_gap_eV']],rtol=0,atol=1e-12)
cached=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/secondary_SHAP_24.csv'),dtype={'ID':str});cached=cached[cached.model=='XGBoost'];assert len(cached)==216 and cached.ID.nunique()==24 and not cached[['ID','feature']].duplicated().any()
model=joblib.load(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/XGBoost.joblib'));params=model.get_params();(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/XGB_SHAP_MODEL_PARAMETERS.json')).write_text(json.dumps(params,indent=2),encoding='utf-8')
xgb=et[['ID','formula_group','band_gap_eV']+F].copy();xgb['prediction_eV']=model.predict(xgb[F].to_numpy()).astype(float);base_ids=set(cached.ID);missing=xgb[~xgb.ID.isin(base_ids)].copy();assert len(missing)==1438
vv,base,pred,res=explain_tree(model,missing[F].to_numpy(),atol=1e-5);assert np.max(abs(res))<1e-5
for k,f in enumerate(F):xgb['SHAP_'+f]=np.nan;xgb.loc[~xgb.ID.isin(base_ids),'SHAP_'+f]=vv[:,k]
cachewide=cached.pivot(index='ID',columns='feature',values='SHAP_eV');indices=xgb.ID.isin(base_ids)
for f in F:xgb.loc[indices,'SHAP_'+f]=cachewide.loc[xgb.loc[indices,'ID'],f].to_numpy()
cached_pred=cached.groupby('ID').prediction_eV.first();assert np.max(abs(xgb.loc[indices,'prediction_eV'].to_numpy()-cached_pred.loc[xgb.loc[indices,'ID']].to_numpy()))<1e-5
cachebase=cached.groupby('ID').base_eV.first();assert np.max(abs(cachebase-base))<1e-5
xgb['base_eV']=base;xgb['additivity_residual_eV']=base+xgb[['SHAP_'+f for f in F]].sum(axis=1)-xgb.prediction_eV;assert xgb.additivity_residual_eV.abs().max()<1e-5
xgb['attribution_source']=np.where(indices,'reused_saved_24','new_missing_1438_same_frozen_model');xgb.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/XGB_SHAP_MATCHED_1462.csv'),index=False)
em=np.abs(et[['SHAP_'+f for f in ORDER]].to_numpy()).mean(axis=0);xm=np.abs(xgb[['SHAP_'+f for f in ORDER]].to_numpy()).mean(axis=0)
imp=pd.DataFrame(dict(feature=ORDER,ET_mean_abs_SHAP_eV=em,XGB_mean_abs_SHAP_eV=xm,ET_rank=pd.Series(em).rank(ascending=False).to_numpy(),XGB_rank=pd.Series(xm).rank(ascending=False).to_numpy()));imp.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5D_MATCHED_SHAP_IMPORTANCE.csv'),index=False)
rho=float(spearmanr(em,xm).statistic);top={str(k):len(set(np.argsort(-em)[:k])&set(np.argsort(-xm)[:k]))/k for k in [3,5]};(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5D_RANK_AGREEMENT.json')).write_text(json.dumps(dict(spearman=rho,top_k_overlap_fraction=top,n=1462,ET_model='frozen fold0 ExtraTrees',XGB_model='frozen fold0 XGBoost',aggregation='mean absolute tree_path_dependent SHAP on exactly same heldout rows',new_explained_rows=1438,reused_XGB_rows=24,SHAP_XGB_max_residual_eV=float(xgb.additivity_residual_eV.abs().max())),indent=2),encoding='utf-8')
# b/c reuse old complete nested OOF, recompute metrics from per-ID data.
rows=[]
for set_name in C['sets']:
 for model_name in ['extra_trees','xgboost']:
  truth=old.band_gap_eV.to_numpy();pr=old[f'prediction__{set_name}__{model_name}'].to_numpy();err=truth-pr
  rows.append(dict(feature_set=set_name,model=model_name,n=len(old),MAE_eV=float(abs(err).mean()),RMSE_eV=float(np.sqrt(np.mean(err**2))),R2=float(1-np.sum(err**2)/np.sum((truth-truth.mean())**2))))
metrics=pd.DataFrame(rows);historic=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/historical_metrics.csv'))
# Marginal metric intervals derived from saved predictions, not new fitting.
labels,inverse=np.unique(old.formula_group.to_numpy(),return_inverse=True);counts=np.bincount(inverse);rng=np.random.default_rng(C['bootstrap_seed']);draws=rng.integers(0,len(labels),(C['bootstrap_repeats'],len(labels)));den=counts[draws].sum(axis=1);intervals=[]
for z in metrics.itertuples():
 err=old[f'prediction__{z.feature_set}__{z.model}'].to_numpy()-old.band_gap_eV.to_numpy();ae=np.bincount(inverse,weights=abs(err));se=np.bincount(inverse,weights=err**2);bm=ae[draws].sum(axis=1)/den;br=np.sqrt(se[draws].sum(axis=1)/den);intervals.append(dict(feature_set=z.feature_set,model=z.model,MAE_low95_eV=float(np.quantile(bm,.025)),MAE_high95_eV=float(np.quantile(bm,.975)),RMSE_low95_eV=float(np.quantile(br,.025)),RMSE_high95_eV=float(np.quantile(br,.975))))
pd.DataFrame(intervals).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5BC_FIXED_OOF_INTERVALS.csv'),index=False)
for z in metrics.itertuples():
 ref=historic[(historic.feature_set==z.feature_set)&(historic.estimator==z.model)].iloc[0];assert abs(ref.mae_eV-z.MAE_eV)<1e-12 and abs(ref.rmse_eV-z.RMSE_eV)<1e-12 and abs(ref.r2-z.R2)<1e-12
metrics.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5BC_METRICS_ALL.csv'),index=False);metrics[metrics.feature_set=='combined9'].to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5B_PREDICTIVE_PERFORMANCE.csv'),index=False)
mp={'atomic7':'M0','atomic7_distance':'M1','atomic7_angle':'M2','combined9':'M3'};metrics['configuration']=metrics.feature_set.map(mp);metrics.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5C_M0_M3_COMPARISON.csv'),index=False)
old[['ID','formula_group','band_gap_eV']+[f'prediction__{s}__{m}' for s in C['sets'] for m in ['extra_trees','xgboost']]].merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one').to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/FIG5BC_ALL_PREDICTIONS.csv'),index=False)
# Recheck old XGB full9 against run011 OOF; do not retrain successful XGB tasks.
xhist=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/XGB_oof_history.csv'),dtype={'ID':str});xf=xhist[xhist.variant=='full9'].set_index('ID').loc[old.ID];assert np.max(abs(xf.prediction_eV.to_numpy()-old.prediction__combined9__xgboost.to_numpy()))<1e-12
for n in ['FIG5E_SPEARMAN_7176.csv','FIG5E_STRONG_PAIRS_FULL_DATA_DESCRIPTIVE.csv','FIG5F_DESCRIPTOR_CORRELATION_GROUPS.csv']:
 (T/n).write_bytes((_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/grouped_ablation/audit')/n).read_bytes())
for n in ['FIG5F_PRIMARY_WITH_INTERVALS.csv','FIG5F_NONADDITIVITY.csv']:(T/n).write_bytes((O/n).read_bytes())
definitions=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/Feature_Definitions.csv'));definitions['display_order']=definitions.feature.map({f:k+1 for k,f in enumerate(ORDER)});definitions.sort_values('display_order').to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/PANEL_DATA/DESCRIPTOR_DEFINITIONS.csv'),index=False)
qa=dict(status='ALL_B_TO_F_DATA_CONFIRMED_ALLOW_DRAWING',F_validation='PASS_before_D_completion_and_before_any_plotting',historical_metrics_replay='PASS',ET_full_and_L_Theta_deletions_replay='PASS',XGB_full_replay='PASS',matched_SHAP_rows=1462,new_XGB_SHAP_rows=1438,reused_XGB_SHAP_rows=24,new_ET_fits=json.loads((_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_REPRODUCIBILITY.json')).read_text())['model_fits'],no_new_ALIGNN_XGB_training=True,no_new_extraction=True,no_DFT_LOBSTER_SLURM=True,no_plot_before_gate=True,correlation_groups_train_only=True,SHAP_rank_agreement=rho)
(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/ALL_PANEL_DATA_CONFIRMED.json')).write_text(json.dumps(qa,indent=2),encoding='utf-8')
pd.DataFrame([dict(path=str(p.relative_to(R)),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(R.rglob('*')) if p.is_file() and 'drawing' not in p.parts]).to_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/grouped_ablation/PRE_DRAW_DATA_SHA256.csv'),index=False)
print(json.dumps(qa,indent=2),flush=True)
