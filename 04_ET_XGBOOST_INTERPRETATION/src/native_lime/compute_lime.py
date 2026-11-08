"""Only explain immutable ET. Checkpoint every ID; never fit or modify ET."""

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
import hashlib,json,sys,time,os
from pathlib import Path
import numpy as np,pandas as pd,joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import r2_score

R=_repo_path('04_ET_XGBOOST_INTERPRETATION/workflows/native_lime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
cfg=json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/native_lime.json')).read_text(encoding='utf-8'))
assert json.loads((_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/IDENTITY_AUDIT.json')).read_text(encoding='utf-8'))['passed']
for x in json.loads((_repo_path('04_ET_XGBOOST_INTERPRETATION/configs/native_lime_source_identity_frozen_input_code.json')).read_text(encoding='utf-8')):assert sha(R/x['path'])==x['sha256'],x['path']
F=json.loads((_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/IDENTITY_AUDIT.json')).read_text(encoding='utf-8'))['features_computational_order']
d=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str})
fold=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv'),dtype={'ID':str})
d=d.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one')
train=d[d.outer_fold!=0][F].to_numpy()
hold=d[d.outer_fold==0].sort_values('ID').reset_index(drop=True)
model=joblib.load(_repo_path('05_CRISP_X/models/property_model.joblib'))
model_params=model.get_params().copy(); original_sha=sha(_repo_path('05_CRISP_X/models/property_model.joblib'))
out=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline');out.mkdir(exist_ok=True); detail=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/per_material');detail.mkdir(exist_ok=True); neigh=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/neighborhoods');neigh.mkdir(exist_ok=True)
start=time.perf_counter()
for i,row in hold.iterrows():
 ID=row.ID; file=detail/(ID+'.json')
 if file.exists():continue
 seed=int(hashlib.sha256(('run022|'+ID).encode()).hexdigest()[:8],16)%(2**31-1)
 x=row[F].to_numpy(dtype=float); t=time.perf_counter()
 ex=LimeTabularExplainer(train,feature_names=F,mode='regression',discretize_continuous=False,
  sample_around_instance=True,feature_selection='none',kernel_width=cfg['kernel_width'],random_state=seed)
 captured=[]
 def predict(z):
  p=model.predict(z);captured.append((np.array(z,copy=True),np.array(p,copy=True)));return p
 result=ex.explain_instance(x,predict,num_features=9,num_samples=cfg['fit_samples'])
 coef=np.zeros(9);mapping=result.as_map()[1]
 assert len(mapping)==9 and set(j for j,v in mapping)==set(range(9))
 for j,v in mapping:coef[j]=v
 raw,pred=captured[0];assert raw.shape==(512,9)
 z=(raw-ex.scaler.mean_)/ex.scaler.scale_;z0=(x-ex.scaler.mean_)/ex.scaler.scale_
 dist=np.linalg.norm((raw-x)/ex.scaler.scale_,axis=1)
 weights=np.exp(-dist**2/(2*cfg['kernel_width']**2))
 intercept=float(result.intercept[1]); proxy=intercept+z@coef
 assert abs(float(result.score)-r2_score(pred,proxy,sample_weight=weights))<1e-10
 anchor=intercept+z0@coef
 assert abs(anchor-float(result.local_pred[0]))<1e-10
 # Independent Gaussian neighborhood, same distribution/scale/kernel, no refit.
 rng=np.random.RandomState(seed+1)
 val_raw=x+rng.normal(size=(cfg['validation_samples'],9))*ex.scaler.scale_
 val_pred=model.predict(val_raw); vz=(val_raw-ex.scaler.mean_)/ex.scaler.scale_
 val_proxy=intercept+vz@coef;vw=np.exp(-np.sum(((val_raw-x)/ex.scaler.scale_)**2,axis=1)/(2*cfg['kernel_width']**2))
 vr=float(r2_score(val_pred,val_proxy,sample_weight=vw))
 wmse=lambda a,b,w:float(np.average((a-b)**2,weights=w))
 rec={'ID':ID,'formula_group':row.formula_group,'ET_prediction_eV':float(pred[0]),'LIME_intercept_eV':intercept,
  'LIME_anchor_prediction_eV':anchor,'anchor_error_eV':float(anchor-pred[0]),'fit_weighted_r2':float(result.score),
  'fit_weighted_rmse_eV':wmse(pred,proxy,weights)**.5,'validation_weighted_r2':vr,
  'validation_weighted_rmse_eV':wmse(val_pred,val_proxy,vw)**.5,
  'validation_weighted_mae_eV':float(np.average(abs(val_pred-val_proxy),weights=vw)),
  'high_fidelity':bool(np.isfinite(vr) and vr>=.8 and abs(anchor-pred[0])<=.05),
  'num_samples_fit':512,'num_samples_validation':128,'seed':seed,'validation_seed':seed+1,
  'kernel_width':cfg['kernel_width'],'scaling':'LIME frozen training mean and population SD; no discretization',
  'distribution':'Gaussian independently per descriptor, centered at instance; may be chemically unrealizable',
  'discretize_continuous':False,'feature_selection':'none','surrogate':'Ridge alpha1',
  'seconds':time.perf_counter()-t}
 for j,f in enumerate(F):rec[f]=float(x[j]);rec['LIME_weight_'+f]=float(coef[j]);rec['LIME_point_component_'+f]=float(z0[j]*coef[j])
 # Complete data to reconstruct fits and independent local fidelity, not just ranks.
 np.savez_compressed(neigh/(ID+'.npz'),fit_features=raw,fit_ET_prediction=pred,fit_proxy=proxy,fit_kernel_weights=weights,
  validation_features=val_raw,validation_ET_prediction=val_pred,validation_proxy=val_proxy,validation_kernel_weights=vw,
  training_mean=ex.scaler.mean_,training_scale=ex.scaler.scale_,coefficients=coef,intercept=intercept)
 dump(file,rec)
 if (i+1)%25==0:print(f'{i+1}/1462 LIME saved; elapsed {time.perf_counter()-start:.1f}s',flush=True)
 assert model.get_params()==model_params
rows=[json.loads((detail/(ID+'.json')).read_text(encoding='utf-8')) for ID in hold.ID]
pd.DataFrame(rows).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/per_material_lime_1462.csv'),index=False)
pd.DataFrame({'feature':F,'training_mean':train.mean(axis=0),'training_SD':train.std(axis=0)}).to_csv(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/training_scaling.csv'),index=False)
assert sha(_repo_path('05_CRISP_X/models/property_model.joblib'))==original_sha
dump(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/native_lime/lime_baseline/COMPLETION.json'),{'n':len(rows),'new_model_training':0,'ET_model_sha256_unchanged':True,'no_ET_parameters_changed':True,
 'fit_prediction_rows':1462*512,'validation_prediction_rows':1462*128,'whole_seconds_this_invocation':time.perf_counter()-start})
print('ALL1462 LIME explanations and independent validation saved',flush=True)
