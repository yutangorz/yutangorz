"""Author-directed exact DFT-node CRISP-X evaluation; no DFT and no global training."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/node_prediction')))
from pathlib import Path
import sys,json,hashlib,shutil,time,traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from pymatgen.core import Structure

R=_repo_path('07_DFT_VALIDATION/workflows/node_prediction');ROOT=_repo_path('.');I=_repo_path('07_DFT_VALIDATION/data/node_prediction');O=_repo_path('outputs/07_DFT_VALIDATION/node_prediction')
SOURCES={
 'nodes':str(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6C_GEOMETRY_PAIRED_PROTOCOL_ANNOTATED.csv')),
 'atomic':str(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6B_ELEMENT_PAIRED_LONG.csv')),
 'manifest':str(_repo_path('07_DFT_VALIDATION/data/geometry_design/geometry_node_manifest.csv')),
 'parents':str(_repo_path('07_DFT_VALIDATION/data/geometry_design/CURRENT_20_MATERIALS.csv')),
 'model':str(_repo_path('05_CRISP_X/models/property_model.joblib')),
 'table':str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv')),
 'data':str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv')),
 'fold':str(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv')),
 'old_response':str(_repo_path('outputs/05_CRISP_X/local_reconstruction/response_predictions.csv')),
}
SEEDS=[307,419]
REUSE=_repo_path('unavailable_inputs/source_9ea7440e8e3e/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(p):return Path(str(p).replace('\\\\','/').replace('\\','/'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf-8')
def csv(d,p):d.to_csv(p,index=False,encoding='utf-8-sig',na_rep='NA')
def read(k):return pd.read_csv(I/SOURCES[k])
def snapshot(p):return I/Path(p).resolve().relative_to(ROOT)

def freeze():
 if O.exists():raise RuntimeError('Executed run cannot be overwritten')
 files=[ROOT/v for v in SOURCES.values()] + [p for p in REUSE.rglob('*') if p.is_file()]
 files+=list((_repo_path('05_CRISP_X/src/local_reconstruction')).rglob('*.py'))
 m=pd.read_csv(ROOT/SOURCES['manifest']);pa=pd.read_csv(ROOT/SOURCES['parents'])
 files += [norm(p) for p in m.structure_path]+[norm(p) for p in pa.baseline_structure]
 # Reuse the successful Ba parent pools; reconstruct only the small local fit.
 for seed in SEEDS:
  pp=ROOT/f'outputs/05_CRISP_X/local_reconstruction/physical_pools/1230247_{seed}'
  files +=[pp/'pool.csv',pp/'events.json',pp/'indices.npz',pp/'parent.json']
 rows=[]
 for p in dict.fromkeys(p.resolve() for p in files):
  dest=snapshot(p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  rows.append({'source_path':str(p),'snapshot_path':str(dest),'sha256':sha(p),'bytes':p.stat().st_size})
 csv(pd.DataFrame(rows),_repo_path('07_DFT_VALIDATION/workflows/node_prediction/INPUT_SOURCE_MAPPING.csv'))
 cfg={'authorization':'User: directly predict actual existing DFT-perturbed node structures with CRISP-X, compare to corresponding DFT. Historical candidate overlap is not a requirement.',
  'method':'run019 block_b64_g40; ridge .001; budget64 geometryquota40; identical generator and predictor',
  'seeds':SEEDS,'seed_aggregation':'arithmetic mean of both fixed seeds if both finite; never select better seed; seed-specific coverage retained',
  'inputs':'all440 actual DFT POSCARs, exact SHA; existing verified nine-descriptor coordinates reused; composition and feature provenance checked',
  'local_training':'only ET queries on independent frozen-generator local pools; no DFT labels; no global ET/ALIGNN/XGB training; no tuning',
  'baseline_absolute':'frozen ET(parent) + CRISP-X predicted DeltaEg; not DFT-anchored',
  'primary_response':'CRISP-X predicted DeltaEg versus DFT saved DeltaEg; also independent absolute-gap comparison',
  'main_scope':'371 formal non-shell-switch nodes /17 parents; all440 retained incl diagnostics/switches',
  'out_of_box':'retain finite output and flag; report in-box sensitivity separately; no target-based exclusion',
  'uncertainty':'2000 parent-cluster bootstrap seed20261104; numerical DFT error bounds unresolved',
  'resume_from':'12 completed parent response sets reused unchanged; finite-mask agreement retains missing values',
  'script_sha256':sha(__file__),'prohibited':'no VASP/LOBSTER/SLURM; no final plotting; no DFT labels in local model fitting'}
 dump(_repo_path('07_DFT_VALIDATION/configs/node_prediction_protocol.json'),cfg);O.mkdir();shutil.copytree(REUSE,O,dirs_exist_ok=True)

def metrics(df,pred,target):
 y=df[target].to_numpy(float);p=df[pred].to_numpy(float);ok=np.isfinite(y)&np.isfinite(p)
 y,p=y[ok],p[ok]
 return {'total_n':len(df),'covered_n':int(ok.sum()),'coverage':float(ok.mean()) if len(df) else None,
  'MAE_eV':float(np.mean(abs(y-p))) if len(y) else None,'RMSE_eV':float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,
  'R2':float(r2_score(y,p)) if len(y)>1 and np.var(y)>1e-20 else None,
  'numeric_direction_agreement':float(np.mean(np.sign(y)==np.sign(p))) if len(y) and target=='delta_Eg_eV' else None,
  'direction_qualification':'numeric signs only; no pair-specific DFT response-resolution threshold available'}

def main():
 freeze();sys.path.insert(0,str(_repo_path('unavailable_inputs/source_f3f04892732e/code')))
 from shared_features.extraction import ConceptExtractor,FEATURES
 from study import make_pool
 from crispx_closure.paths import block_subset,fit_closed,predict_closed
 from crispx.path_model import path_key
 from crispx.perturb import structure_key
 model=joblib.load(I/SOURCES['model']);model.n_jobs=1 # runtime only; frozen estimators unchanged
 extractor=ConceptExtractor(I/SOURCES['table']);data=read('data');fold=read('fold')
 data=data.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one');train=data[data.outer_fold!=0][FEATURES].to_numpy()
 scale=train.std(axis=0);scale[scale==0]=1
 csv(pd.DataFrame({'feature':FEATURES,'training_SD':scale,'train_min':train.min(axis=0),'train_max':train.max(axis=0)}),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/TRAINING_SCALE_AND_DOMAIN.csv'))
 nodes=read('nodes');man=read('manifest');parents=read('parents');atoms=read('atomic')
 assert len(nodes)==440 and nodes.point_id.nunique()==440
 assert set(nodes.material_id)==set(parents.material_id)==set(atoms.material_id)
 rows=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_NODE_PREDICTIONS_ALL_440.csv')).to_dict('records');design=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_LOCAL_MODEL_DESIGN.csv')).to_dict('records');features=[];statuses=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/node_prediction/PARENT_PROGRESS.csv')).to_dict('records');replay=[]
 completed={int(v['material_id']) for v in statuses}
 start=time.time()
 for parent in parents.itertuples():
  mid=int(parent.material_id);t=time.time();pp=_repo_path('outputs/07_DFT_VALIDATION/node_prediction/local_models')/str(mid);pp.mkdir(parents=True,exist_ok=True)
  sub=nodes[nodes.material_id==mid].copy();mm=man[man.material_id==mid].set_index('task_id')
  bas=snapshot(norm(parent.baseline_structure));assert sha(bas)==parent.baseline_SHA256
  structure=Structure.from_file(bas)
  ad=atoms[atoms.material_id==mid].set_index('descriptor')
  x0=np.r_[ad.loc[FEATURES[:7],'parent_mean'].to_numpy(float),parent.baseline_L_A,parent.baseline_Theta_degree]
  # Frozen definitions reused; do not re-extract geometry for the 440 verified nodes.
  av=extractor.numeric.loc[[s.specie.symbol for s in structure]].to_numpy()
  av=np.where(np.isfinite(av),av,extractor.medians).mean(axis=0)
  assert np.max(abs(av-x0[:7]))<1e-10
  xx=[];node_keys=[]
  for n in sub.itertuples():
   path=snapshot(norm(mm.loc[n.point_id,'structure_path']));assert sha(path)==n.structure_sha256
   node=Structure.from_file(path)
   assert [s.specie.symbol for s in node]==[s.specie.symbol for s in structure]
   assert np.max(abs(node.lattice.matrix-structure.lattice.matrix))<1e-10
   assert n.baseline_structure_sha256==parent.baseline_SHA256
   assert abs((n.global_L_A-n.delta_L_A)-x0[7])<1e-9 and abs((n.global_Theta_degree-n.delta_Theta_degree)-x0[8])<1e-8
   v=np.r_[x0[:7],n.global_L_A,n.global_Theta_degree];xx.append(v);node_keys.append(structure_key(node))
   features.append({'material_id':mid,'point_id':n.point_id,'structure_sha256':n.structure_sha256,'source_structure':str(norm(mm.loc[n.point_id,'structure_path'])),**dict(zip(FEATURES,v))})
  xx=np.array(xx);zz=(xx-x0)/scale
  if mid in completed:continue
  base_pred=float(model.predict(x0[None,:])[0]);direct=model.predict(xx)-base_pred
  outcomes={};outs={};reasons={}
  for seed in SEEDS:
   pooldir=pp/f'seed_{seed}';pooldir.mkdir(exist_ok=True)
   if mid==1230247:
    old=I/f'outputs/05_CRISP_X/local_reconstruction/physical_pools/{mid}_{seed}'
    pool=pd.read_csv(old/'pool.csv');xp=pool[FEATURES].to_numpy();events=json.loads((old/'events.json').read_text())
    ix=np.load(old/'indices.npz');fitidx=ix['fit'];testidx=ix['test'];origin='REUSED_SAVED_SUCCESSFUL_POOL'
    # No generation or repeated teacher queries for saved successful local pool.
    yp=pool.delta_frozen_ET.to_numpy()
    assert np.max(abs(read('old_response').query("ID==@mid and seed==@seed and model=='frozen_ET' and method=='block_b64_g40' and kind=='geometry'").truth.to_numpy()-pool.loc[testidx[::2],'delta_frozen_ET'].to_numpy()))<1e-10
   else:
    ss,xp,events,fitidx,testidx,amps,rejected=make_pool(structure,extractor,seed)
    assert not set(node_keys)&{structure_key(st) for st in ss},'Target node leaked into local pool'
    yp=model.predict(xp)-base_pred;pool=pd.DataFrame(xp,columns=FEATURES);pool['delta_frozen_ET']=yp;pool['amplitude']=amps
    pool['role']='fit';pool.loc[testidx,'role']='heldout';origin='NEW_LOCAL_POOL_FROZEN_GENERATOR_NO_DFT'
    np.savez_compressed(pooldir/'indices.npz',fit=fitidx,test=testidx)
    dump(pooldir/'rejected.json',rejected)
    # Save actual new neighborhood structures as reproducible model-only inputs, not DFT jobs.
    for j,st in enumerate(ss):dump(pooldir/f'structure_{j}.json',st.as_dict())
   csv(pool,pooldir/'pool.csv');dump(pooldir/'events.json',events)
   zp=(xp-x0)/scale
   selected=fitidx[block_subset(zp[fitidx],[events[j] for j in fitidx],64,40)]
   local=fit_closed(zp[selected],yp[selected],[events[j] for j in selected])
   joblib.dump(local,pooldir/'CRISP_X_LOCAL.joblib')
   p,_,why,outside=predict_closed(local,zz,[{'kind':'strain_and_internal_displacement'}]*len(sub))
   assert predict_closed(local,np.zeros((1,9)),[{'kind':'strain_and_internal_displacement'}])[0][0]==0
   outcomes[seed]=p;outs[seed]=outside;reasons[seed]=why
   testpred,_,_,_=predict_closed(local,zp[testidx],[events[j] for j in testidx])
   gmask=np.array([path_key(events[j])=='geometry_only' for j in testidx]);good=gmask&np.isfinite(testpred)
   gm=local['geometry_only']
   design.append({'material_id':mid,'seed':seed,'pool_source':origin,'n_local_queries':len(selected),'n_geometry_queries':int(sum(path_key(events[j])=='geometry_only' for j in selected)),
     'selected_indices':json.dumps(selected.tolist()),'numerical_geometry_rank':gm['numerical_geometry_rank'],'condition_number':gm['condition_number'],
     'response_qualified':gm['response_qualified'],'separate_geometry_slopes_qualified':gm['separate_geometry_slopes_qualified'],
     'distance_coefficient_eV_per_training_SD':gm['coefficient'][0],'angle_coefficient_eV_per_training_SD':gm['coefficient'][1],
     'heldout_geometry_n':int(gmask.sum()),'heldout_geometry_covered_n':int(good.sum()),
     'heldout_geometry_MAE_vs_ET_eV':float(np.mean(abs(testpred[good]-yp[testidx][good]))) if good.any() else np.nan,
     'DFT_labels_used_in_fit':0,'runtime_seconds':time.time()-t})
   if mid==1230247:
    old=read('old_response').query("ID==@mid and seed==@seed and model=='frozen_ET' and method=='block_b64_g40'").set_index('candidate')
    expected=old.loc[testidx,'prediction'].to_numpy()
    assert np.array_equal(np.isfinite(testpred),np.isfinite(expected))
    dif=np.nanmax(abs(testpred-expected))
    assert dif<1e-10
    replay.append({'material_id':mid,'seed':seed,'replayed_n':len(testidx),'prediction_max_abs_difference_eV':dif})
  mat=np.array([outcomes[seed] for seed in SEEDS]);both=np.isfinite(mat).all(axis=0)
  combined=np.where(both,np.mean(mat,axis=0),np.nan)
  for j,n in enumerate(sub.itertuples()):
   row=n._asdict();row.pop('Index',None)
   row.update(ET_parent_prediction_eV=base_pred,ET_direct_delta_eV=direct[j],CRISP_X_predicted_delta_eV=combined[j],
     CRISP_X_predicted_final_gap_eV=base_pred+combined[j],prediction_finite_both_seeds=bool(both[j]),
     seed_min_delta_eV=float(np.min(mat[:,j])) if both[j] else np.nan,seed_max_delta_eV=float(np.max(mat[:,j])) if both[j] else np.nan,
     outside_local_box_any=bool(any(outs[seed][j] for seed in SEEDS)),
     outside_global_training_feature_range=bool(((xx[j]<train.min(axis=0)-1e-10)|(xx[j]>train.max(axis=0)+1e-10)).any()),
     parent_in_ET_training=bool(mid in set(data.loc[data.outer_fold!=0,'ID'])),
     estimator='block_b64_g40; 2seed mean; frozen ET(parent)+local response for final gap',
     prediction_provenance='existing DFT input structure SHA -> frozen nine coordinates -> local CRISP-X model; DFT target unused in fit')
   for seed in SEEDS:row.update({f'CRISP_delta_seed_{seed}_eV':outcomes[seed][j],f'reason_seed_{seed}':reasons[seed][j],f'outside_box_seed_{seed}':bool(outs[seed][j])})
   rows.append(row)
  statuses.append({'material_id':mid,'status':'COMPLETED','n_target_nodes':len(sub),'finite_both_seeds':int(both.sum()),'seconds':time.time()-t})
  csv(pd.DataFrame(rows),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_NODE_PREDICTIONS_ALL_440.csv'));csv(pd.DataFrame(design),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_LOCAL_MODEL_DESIGN.csv'));csv(pd.DataFrame(statuses),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/PARENT_PROGRESS.csv'))
  print('DFT-node CRISP-X evaluation',len(statuses),'/20',mid,'nodes',len(sub),'elapsed',round(time.time()-start,1),flush=True)
 csv(pd.DataFrame(features),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_EXACT_NODE_MODEL_INPUTS.csv'));csv(pd.DataFrame(replay),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_REPLAY_QA.csv'))
 df=pd.DataFrame(rows);assert len(df)==440 and df.point_id.nunique()==440
 met=[]
 scopes={'PRIMARY_FORMAL_NON_SWITCH':df[df.primary_distribution],'ALL_440_INVENTORY':df,
  'DIAGNOSTIC':df[df.scope=='DIAGNOSTIC'],'FORMAL_SHELL_SWITCH':df[(df.scope=='FORMAL')&df.first_shell_changed],
  'PRIMARY_IN_LOCAL_BOX':df[df.primary_distribution&~df.outside_local_box_any]}
 for scope,g in scopes.items():
  for target,pred,variable in [('delta_Eg_eV','CRISP_X_predicted_delta_eV','RESPONSE_DELTA'),('Eg_eV','CRISP_X_predicted_final_gap_eV','ABSOLUTE_FINAL_GAP_ET_ANCHORED')]:
   met.append({'scope':scope,'variable':variable,'n_parents':g.material_id.nunique(),**metrics(g,pred,target)})
 csv(pd.DataFrame(met),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_RESPONSE_AND_ABSOLUTE_METRICS.csv'))
 main=df[df.primary_distribution&df.prediction_finite_both_seeds].copy();parentmetrics=[]
 for mid,g in main.groupby('material_id'):
  parentmetrics.append({'material_id':mid,**metrics(g,'CRISP_X_predicted_delta_eV','delta_Eg_eV')})
 pm=pd.DataFrame(parentmetrics);csv(pm,_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_PER_PARENT_RESPONSE_METRICS.csv'))
 rng=np.random.default_rng(20261104);ids=main.material_id.unique();boot=[]
 for b in range(2000):
  sampled=rng.choice(ids,size=len(ids),replace=True);v=pd.concat([main[main.material_id==mid] for mid in sampled],ignore_index=True)
  m=metrics(v,'CRISP_X_predicted_delta_eV','delta_Eg_eV');boot.append({'replicate':b,**m})
 bs=pd.DataFrame(boot);csv(bs,_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_MATERIAL_CLUSTER_BOOTSTRAP.csv'))
 intervals=[]
 for col in ['MAE_eV','RMSE_eV','R2','numeric_direction_agreement']:
  valid=bs[col].dropna();intervals.append({'metric':col,'low95':valid.quantile(.025),'high95':valid.quantile(.975),'valid_n':len(valid),'B':2000,
   'unit':'eV' if col in ['MAE_eV','RMSE_eV'] else 'dimensionless','scope':'primary, parent-cluster percentile; not DFT numerical uncertainty'})
 csv(pd.DataFrame(intervals),_repo_path('outputs/07_DFT_VALIDATION/node_prediction/FIG6E_CLUSTER_INTERVALS.csv'))
 dump(_repo_path('outputs/07_DFT_VALIDATION/node_prediction/EXECUTION_QA.json'),{'nodes':len(df),'parents':df.material_id.nunique(),'main_nodes':len(main),'main_parents':main.material_id.nunique(),
  'all_sources_unchanged':all(sha(r.source_path)==r.sha256 and sha(r.snapshot_path)==r.sha256 for r in pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/node_prediction/INPUT_SOURCE_MAPPING.csv')).itertuples()),
  'frozen_model_sha256':sha(I/SOURCES['model']),'global_training_runs':0,'DFT_tasks':0,'local_models':len(design),
  'no_DFT_labels_used_in_local_fit':True,'formal_plots_created':False,'seconds':time.time()-start})
 print(pd.DataFrame(met).to_string(index=False),flush=True)

if __name__=='__main__':main()
