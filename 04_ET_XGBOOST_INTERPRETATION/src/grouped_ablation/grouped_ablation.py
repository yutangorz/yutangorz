
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
import sys,json,time,hashlib,importlib.util
import numpy as np,pandas as pd
from sklearn.metrics import r2_score
R=_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/grouped_ablation');I=_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation');C=json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/grouped_ablation.json')).read_text());F=C['features'];sys.path.insert(0,str(I))
from original_ablation import build
from ablation_core import clustered_interval
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for x in pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/grouped_ablation/SOURCE_SHA256.csv')).itertuples():assert sha(I/x.name)==x.SHA256
O=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation');O.mkdir(exist_ok=False);(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/predictions')).mkdir();(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/actual_parameters')).mkdir()
d=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/data.csv'),dtype={'ID':str}).merge(pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/folds.csv'),dtype={'ID':str})[['ID','outer_fold']],on='ID',validate='one_to_one');old=pd.read_csv(_repo_path('04_ET_XGBOOST_INTERPRETATION/data/grouped_ablation/oof.csv'),dtype={'ID':str}).set_index('ID').loc[d.ID]
X=d[F].to_numpy();Y=d.band_gap_eV.to_numpy();fold=d.outer_fold.to_numpy();assert np.isfinite(X).all() and np.isfinite(Y).all()
unique_groups=sorted(set(tuple(z['indices']) for z in C['groups']));assert all(set(tuple(z['indices']) for z in C['groups'] if z['fold']==k)==set(unique_groups) for k in range(5))
variants=[()]+[(j,) for j in range(9)]+unique_groups
def name(cols):return 'full9' if not cols else 'drop_'+'_'.join(map(str,cols))
fold_metrics=[];replay=[];all_summary=[];frames={};started=time.perf_counter()
for seed in C['model_seeds']:
 for cols in variants:
  label=name(cols);keep=[j for j in range(9) if j not in cols];p=np.full(len(d),np.nan)
  for k in range(5):
   tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);assert not set(d.iloc[tr].formula_group)&set(d.iloc[te].formula_group)
   m=build(dict(id='et1',family='extra_trees',params={'min_samples_leaf':1}),seed+k);t=time.perf_counter();m.fit(X[tr][:,keep],Y[tr]);p[te]=m.predict(X[te][:,keep]);e=Y[te]-p[te]
   fold_metrics.append(dict(seed=seed,fold=k,variant=label,n=len(te),training_n=len(tr),mae_eV=float(abs(e).mean()),rmse_eV=float(np.sqrt(np.mean(e**2))),r2=float(r2_score(Y[te],p[te])),seconds=time.perf_counter()-t))
   (_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/actual_parameters')/f'{seed}_{label}_fold{k}.json').write_text(json.dumps(dict(parameters=m.get_params(),retained_features=[F[j] for j in keep],removed_features=[F[j] for j in cols],train_ID_SHA256=hashlib.sha256('\n'.join(d.ID.iloc[tr]).encode()).hexdigest(),test_ID_SHA256=hashlib.sha256('\n'.join(d.ID.iloc[te]).encode()).hexdigest()),indent=2),encoding='utf-8')
  assert np.isfinite(p).all();z=d[['ID','formula_group','outer_fold','band_gap_eV']].copy();z['seed']=seed;z['variant']=label;z['prediction_eV']=p;z['abs_error_eV']=abs(p-Y);frames[(seed,label)]=z
  z.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/predictions')/f'{seed}_{label}.csv',index=False)
  all_summary.append(dict(seed=seed,variant=label,removed_features=json.dumps([F[j] for j in cols]),n=len(z),mae_eV=float(abs(p-Y).mean()),rmse_eV=float(np.sqrt(np.mean((p-Y)**2))),r2=float(r2_score(Y,p))))
  if seed==C['primary_seed'] and cols in [(),(7,),(8,)]:
   histset={(): 'combined9',(7,):'atomic7_angle',(8,):'atomic7_distance'}[cols];diff=float(abs(p-old[f'prediction__{histset}__extra_trees'].to_numpy()).max());replay.append(dict(variant=label,historical_feature_set=histset,max_abs_prediction_difference_eV=diff));assert diff<1e-12,'Historical ET baseline/single reconstruction mismatch'
  pd.DataFrame(fold_metrics).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FOLD_METRICS.csv'),index=False)
  print('ET',seed,label,'MAE',all_summary[-1]['mae_eV'],'elapsed_s',round(time.perf_counter()-started,1),flush=True)
summary=pd.DataFrame(all_summary);summary['delta_MAE_eV']=[row.mae_eV-float(summary[(summary.seed==row.seed)&(summary.variant=='full9')].mae_eV.iloc[0]) for row in summary.itertuples()]
summary.to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_GROUP_ABLATION_RESULTS.csv'),index=False)
primary=C['primary_seed'];out=[];boot=[]
for cols in variants:
 label=name(cols);z=frames[(primary,label)];effect=z.abs_error_eV.to_numpy()-frames[(primary,'full9')].abs_error_eV.to_numpy();bs=clustered_interval(effect,z.formula_group.to_numpy(),seed=C['bootstrap_seed'],repeats=C['bootstrap_repeats']);low,high=np.quantile(bs,[.025,.975]);r=summary[(summary.seed==primary)&(summary.variant==label)].iloc[0].to_dict();r.update(low95_eV=float(low),high95_eV=float(high),seed_min_eV=float(summary[summary.variant==label].delta_MAE_eV.min()),seed_max_eV=float(summary[summary.variant==label].delta_MAE_eV.max()));out.append(r)
 for j,v in enumerate(bs):boot.append(dict(variant=label,replicate=j,delta_MAE_eV=v))
pd.DataFrame(out).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_PRIMARY_WITH_INTERVALS.csv'),index=False);pd.DataFrame(boot).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_PAIRED_FORMULA_BOOTSTRAP.csv'),index=False)
nonadd=[]
for cols in unique_groups:
 label=name(cols)
 for seed in C['model_seeds']:
  base=frames[(seed,'full9')].abs_error_eV.to_numpy();joint=frames[(seed,label)].abs_error_eV.to_numpy()-base;single=sum(frames[(seed,name((j,)))].abs_error_eV.to_numpy()-base for j in cols);interaction=joint-single
  row=dict(seed=seed,variant=label,group_size=len(cols),removed_features=json.dumps([F[j] for j in cols]),joint_delta_MAE_eV=float(joint.mean()),sum_single_delta_MAE_eV=float(single.mean()),nonadditivity_eV=float(interaction.mean()))
  if seed==primary:
   bs=clustered_interval(interaction,d.formula_group.to_numpy(),seed=C['bootstrap_seed'],repeats=C['bootstrap_repeats']);row.update(low95_eV=float(np.quantile(bs,.025)),high95_eV=float(np.quantile(bs,.975)))
  nonadd.append(row)
pd.DataFrame(nonadd).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_NONADDITIVITY.csv'),index=False)
repro=dict(status='F_DATA_VALIDATED_NO_DRAWING_YET',n=7176,formula_groups=4665,folds=5,model_fits=len(fold_metrics),seeds=C['model_seeds'],groups=unique_groups,baseline_and_available_single_replay=replay,hyperparameters='Frozen ET et1 for all removals; no new tuning',no_test_fold_group_selection=True,no_group_leakage=True,new_extraction=0,new_DFT=0,uncertainty=C['uncertainty'],elapsed_s=time.perf_counter()-started)
(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_REPRODUCIBILITY.json')).write_text(json.dumps(repro,indent=2),encoding='utf-8')
report=['# Figure5(f)实际计算与验证','',f'完成{len(fold_metrics)}次ET重拟合，3种子×5固定公式分组外折×12规格。完整9、9单删、2训练集相关连通组。完整及L/Theta单删原逐ID预测差均<1e-12。其他ET单删此前无结果，当前首次计算，不冒称历史核对通过。未重做旧XGB消融。','', '组间结果（主种子；点值eV）：']
for z in nonadd:
 if z['seed']==primary:report.append(f"- {z['removed_features']}: joint ΔMAE={z['joint_delta_MAE_eV']:.9f}; sum singles={z['sum_single_delta_MAE_eV']:.9f}; I={z['nonadditivity_eV']:.9f} [{z['low95_eV']:.9f},{z['high95_eV']:.9f}].")
report+=['','ΔMAE=删除后MAE−完整MAE；正值代表误差增大，负值照实保留。联合减单删之和仅为重训练非加性，不是物理相互作用或因果强度。区间为固定OOF预测的公式簇重采样，非全管线重训区间，未作多重校正。两组由各折训练输入独立筛出且均相同，才合并完整OOF；full7176相关矩阵不参与该折分组。源表中median元素填充值可能影响相关，未消除或重新提取。','', '后续必须先完成d相同1462材料XGB SHAP补齐及b–f数据总验收，再开始正式绘图。']
(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/grouped_ablation/FIG5F_GROUP_ABLATION_SUMMARY.md')).write_text('\n'.join(report),encoding='utf-8');print('F_DATA_VALIDATED',flush=True)
