"""Nested grouped property prediction; never use ALIGNN predictions as labels."""

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
import argparse, hashlib, importlib.metadata, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ATOMIC=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume']
SETS={'atomic7':ATOMIC,'atomic7_distance':ATOMIC+['bond_length'],'atomic7_angle':ATOMIC+['bond_angle'],'combined9':ATOMIC+['bond_length','bond_angle']}
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
    return dict(n=len(y),mae_eV=float(mean_absolute_error(y,p)),rmse_eV=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))
def specs():
    return [dict(id='mean',family='mean',params={}),dict(id='ridge1',family='ridge',params={'alpha':1.}),dict(id='ridge100',family='ridge',params={'alpha':100.}),
            dict(id='et1',family='extra_trees',params={'min_samples_leaf':1}),dict(id='et3',family='extra_trees',params={'min_samples_leaf':3}),
            dict(id='xgb3',family='xgboost',params={'max_depth':3}),dict(id='xgb6',family='xgboost',params={'max_depth':6})]
def build(s,seed):
    f,p=s['family'],s['params']
    if f=='mean':return DummyRegressor()
    if f=='ridge':return make_pipeline(StandardScaler(),Ridge(**p))
    if f=='extra_trees':return ExtraTreesRegressor(n_estimators=250,max_features=1.,n_jobs=2,random_state=seed,**p)
    if f=='xgboost':return XGBRegressor(n_estimators=400,learning_rate=.04,reg_lambda=5.,min_child_weight=3,subsample=.9,colsample_bytree=1.,tree_method='hist',objective='reg:squarederror',n_jobs=2,random_state=seed,**p)
    raise ValueError(f)
def folds(g,n,seed):
    z=list(GroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros((len(g),1)),groups=g))
    for tr,va in z:assert not set(g[tr])&set(g[va])
    assert sorted(np.concatenate([va for tr,va in z]))==list(range(len(g)))
    return z
def choose(rows,choices,family=None):
    ids=[i for i,s in enumerate(choices) if family is None or s['family']==family]
    return min(ids,key=lambda j:(rows[j]['mae_eV'],rows[j]['rmse_eV'],j))
def bootstrap(y,base,new,g,repeats,seed):
    # Resample complete formula clusters, retain all rows and both predictions.
    u,inv=np.unique(g,return_inverse=True);ng=len(u)
    cnt=np.bincount(inv);ae0=np.bincount(inv,weights=abs(y-base));ae1=np.bincount(inv,weights=abs(y-new))
    se0=np.bincount(inv,weights=(y-base)**2);se1=np.bincount(inv,weights=(y-new)**2)
    rng=np.random.default_rng(seed);rows=[]
    for b in range(repeats):
        ix=rng.integers(0,ng,ng);n=cnt[ix].sum()
        rows.append((b,(ae0[ix].sum()-ae1[ix].sum())/n,np.sqrt(se0[ix].sum()/n)-np.sqrt(se1[ix].sum()/n)))
    return pd.DataFrame(rows,columns=['replicate','mae_improvement_eV','rmse_improvement_eV'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    out=run/'results';out.mkdir(exist_ok=False);(out/'models').mkdir();started=time.perf_counter()
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'));choices=specs();assert choices==cfg['candidates']
    for entry in json.loads((run/'INPUT_CODE_SHA256.json').read_text(encoding='utf-8')):
        assert sha(run/entry['path'])==entry['sha256'],entry['path']
    d=pd.read_csv(run/'inputs/model_input_7176.csv',dtype={'ID':str})
    y=d.band_gap_eV.to_numpy();g=d.formula_group.to_numpy();outer=folds(g,5,cfg['seed'])
    mapping=d[['ID','formula_group']].copy();mapping['outer_fold']=-1;innerrows=[]
    for k,(tr,te) in enumerate(outer):
        mapping.loc[te,'outer_fold']=k
        for j,(it,iv) in enumerate(folds(g[tr],3,cfg['seed']+k+1)):
            innerrows.extend(dict(ID=d.ID.iloc[i],outer_fold=k,inner_fold=j,role=role) for role,idx in [('inner_train',tr[it]),('inner_validation',tr[iv])] for i in idx)
    mapping.to_csv(out/'outer_fold_manifest.csv',index=False);pd.DataFrame(innerrows).to_csv(out/'inner_fold_manifest.csv',index=False)
    families=['selected','extra_trees','xgboost','ridge','mean'];allpred={};score_rows=[];selections=[];searchrows=[]
    for name,features in SETS.items():
        x=d[features].to_numpy();pred={f:np.full(len(y),np.nan) for f in families}
        for k,(tr,te) in enumerate(outer):
            local=[];inside=folds(g[tr],3,cfg['seed']+k+1)
            for s in choices:
                v=np.empty(len(tr));t=time.perf_counter()
                for it,iv in inside:
                    m=build(s,cfg['seed']+k);m.fit(x[tr[it]],y[tr[it]]);v[iv]=m.predict(x[tr[iv]])
                row=dict(feature_set=name,outer_fold=k,candidate=s['id'],family=s['family'],seconds=time.perf_counter()-t,**metrics(y[tr],v));local.append(row);searchrows.append(row)
                # Every validation prediction is retained for later table/plot rebuilding.
                pd.DataFrame({'ID':d.ID.iloc[tr].to_numpy(),'target_eV':y[tr],'prediction_eV':v}).to_csv(out/f'inner_oof_{name}_fold{k}_{s["id"]}.csv',index=False)
            for family in families:
                ix=choose(local,choices,None if family=='selected' else family);s=choices[ix]
                m=build(s,cfg['seed']+k);m.fit(x[tr],y[tr]);p=m.predict(x[te]);pred[family][te]=p
                score_rows.append(dict(feature_set=name,estimator=family,outer_fold=k,chosen_candidate=s['id'],**metrics(y[te],p)))
                selections.append(dict(feature_set=name,estimator=family,outer_fold=k,training_n=len(tr),evaluation_n=len(te),spec=s,actual_parameters=m.get_params(deep=False) if family=='mean' else str(m.get_params(deep=True))))
            pd.DataFrame(searchrows).to_csv(out/'inner_candidate_scores.csv',index=False);pd.DataFrame(score_rows).to_csv(out/'outer_fold_metrics.csv',index=False)
            dump(out/'selected_parameters.json',selections)
            print(name,'outer',k+1,'/5',next(r for r in reversed(score_rows) if r['estimator']=='selected'),flush=True)
        for f in families:assert np.isfinite(pred[f]).all();allpred[(name,f)]=pred[f]
        for f in families:d[f'prediction__{name}__{f}']=pred[f]
        d.to_csv(out/'nested_oof_predictions.csv',index=False)
    totals=[dict(feature_set=n,estimator=f,**metrics(y,p)) for (n,f),p in allpred.items()]
    pd.DataFrame(totals).to_csv(out/'pooled_oof_metrics.csv',index=False)
    evidence=[]
    for f in families:
        b=allpred[('atomic7',f)];p=allpred[('combined9',f)];bs=bootstrap(y,b,p,g,cfg['bootstrap_repeats'],cfg['seed']+10)
        bs.to_csv(out/f'paired_bootstrap_{f}.csv',index=False);ci=bs.mae_improvement_eV.quantile([.025,.975]).to_list()
        fold_deltas=[]
        for k,(tr,te) in enumerate(outer):fold_deltas.append(float(np.mean(abs(y[te]-b[te]))-np.mean(abs(y[te]-p[te]))))
        evidence.append(dict(estimator=f,mae_improvement_eV=float(np.mean(abs(y-b))-np.mean(abs(y-p))),rmse_improvement_eV=float(np.sqrt(np.mean((y-b)**2))-np.sqrt(np.mean((y-p)**2))),mae_improvement_cluster_bootstrap95=ci,outer_fold_mae_improvements_eV=fold_deltas,positive_fold_count=sum(t>0 for t in fold_deltas)))
    ev={r['estimator']:r for r in evidence}
    passed=all(ev[f]['mae_improvement_cluster_bootstrap95'][0]>0 and ev[f]['positive_fold_count']>=4 for f in ['selected','extra_trees','xgboost'])
    # Deployment candidates: select on all-data grouped CV, no new performance claim.
    for name,features in SETS.items():
        relevant=[r for r in searchrows if r['feature_set']==name]
        best=min(choices,key=lambda s:np.mean([r['mae_eV'] for r in relevant if r['candidate']==s['id']]))
        # Selection uses inner results across outer fits; all-data refit is for reuse only.
        model=build(best,cfg['seed']);model.fit(d[features].to_numpy(),y)
        joblib.dump(model,out/'models'/f'{name}_refit.joblib',compress=3)
        dump(out/'models'/f'{name}_refit_spec.json',dict(features=features,spec=best,parameters=str(model.get_params(deep=True)),scope='full-data refit for subsequent development, not a separately evaluated model'))
    pd.DataFrame(evidence).to_json(out/'paired_improvements.json',orient='records',indent=2)
    report=dict(status='completed',target='original OQMD database band gap; not ALIGNN output',n=len(d),formula_groups=int(d.formula_group.nunique()),design='5 outer x3 inner formula-grouped nested CV; all four sets same split and seven candidates',metrics=totals,paired_evidence=evidence,operational_criterion=cfg['success_criterion'],structural_improvement_supported=passed,elapsed_seconds=time.perf_counter()-started,limitations=['exploratory same-dataset internal validation, no new untouched external test','cluster bootstrap conditions on fitted OOF predictions; not full pipeline retraining uncertainty','not an ALIGNN mechanism proof; seven features are not full CGCNN','mean-shell geometry, not certified B-X-B or chemical bonds','units and median-filled element properties inherited; no causal importance claim','all-data refit models have no independent score'])
    dump(out/'ABLATION_REPORT.json',report);print(json.dumps(dict(completed=True,structural_improvement_supported=passed,evidence=evidence)),flush=True)
if __name__=='__main__':main()
