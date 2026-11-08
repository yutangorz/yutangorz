
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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/xgboost_ablation')))
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from ablation_core import correlation_groups,make_variants,paired_errors,clustered_interval

FEATURES=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(cols):return 'full9' if not cols else 'drop_'+'_'.join(map(str,cols))
def met(y,p):return dict(n=len(y),mae_eV=float(np.mean(abs(y-p))),rmse_eV=float(np.sqrt(np.mean((y-p)**2))),r2=float(r2_score(y,p)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    for e in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/e['path'])==e['sha256']
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'));out=run/'results';out.mkdir(exist_ok=False)
    for name in ['models','training_correlations','fold_manifests']:(out/name).mkdir()
    d=pd.read_csv(run/'inputs/model_input_7176.csv',dtype={'ID':str});f=pd.read_csv(run/'inputs/outer_fold_manifest.csv',dtype={'ID':str});d=d.merge(f[['ID','outer_fold']],on='ID',validate='one_to_one')
    assert len(d)==7176 and d.ID.is_unique and d.groupby('formula_group').outer_fold.nunique().max()==1
    x=d[FEATURES].to_numpy();y=d.band_gap_eV.to_numpy();assert np.isfinite(x).all() and np.isfinite(y).all()
    predictions=[];design=[];metrics=[];timing=[];edges=[];replays=[]
    for fold in range(5):
        tr=np.flatnonzero(d.outer_fold.to_numpy()!=fold);te=np.flatnonzero(d.outer_fold.to_numpy()==fold)
        assert not set(d.iloc[tr].formula_group)&set(d.iloc[te].formula_group)
        groups,corr=correlation_groups(x[tr],cfg['correlation_threshold'])
        pd.DataFrame(corr,index=FEATURES,columns=FEATURES).to_csv(out/'training_correlations'/f'fold{fold}.csv')
        d.iloc[tr][['ID','formula_group','outer_fold','band_gap_eV']].to_csv(out/'fold_manifests'/f'fold{fold}_training.csv',index=False)
        d.iloc[te][['ID','formula_group','outer_fold','band_gap_eV']].to_csv(out/'fold_manifests'/f'fold{fold}_test.csv',index=False)
        for i in range(9):
            for j in range(i+1,9):edges.append(dict(fold=fold,a=FEATURES[i],b=FEATURES[j],pearson=corr[i,j],connected=bool(abs(corr[i,j])>=cfg['correlation_threshold'])))
        variants=make_variants(9,groups);full=None
        fold_parameters=dict(cfg['model_parameters']);fold_parameters['random_state']+=fold
        for cols,tags in variants.items():
            retained=[j for j in range(9) if j not in cols];name=key(cols);start=time.perf_counter()
            if retained:
                m=XGBRegressor(**fold_parameters).fit(x[tr][:,retained],y[tr]);p=m.predict(x[te][:,retained]).astype(float)
            else:
                m=DummyRegressor().fit(np.zeros((len(tr),1)),y[tr]);p=m.predict(np.zeros((len(te),1)))
            joblib.dump(m,out/'models'/f'fold{fold}_{name}.joblib',compress=3)
            if not cols:
                full=p.copy()
                if fold==0:
                    ref=joblib.load(run/'inputs/reference_fold0_XGBoost.joblib').predict(x[te]);diff=float(abs(p-ref).max())
                    replays.append(dict(fold=fold,max_abs_prediction_difference_eV=diff,tolerance_eV=1e-5));assert diff<1e-5,'baseline model replay'
            effect=paired_errors(y[te],full,p)
            for k,j in enumerate(te):predictions.append(dict(ID=d.ID.iloc[j],formula_group=d.formula_group.iloc[j],fold=fold,variant=name,
                truth_eV=y[j],full_prediction_eV=full[k],prediction_eV=p[k],paired_MAE_increase_eV=effect[k]))
            design.append(dict(fold=fold,variant=name,removed_features=json.dumps([FEATURES[j] for j in cols]),retained_features=json.dumps([FEATURES[j] for j in retained]),
                tags=json.dumps(tags),removed_count=len(cols),training_n=len(tr),test_n=len(te),actual_random_seed=fold_parameters['random_state']))
            metrics.append(dict(fold=fold,variant=name,**met(y[te],p),paired_MAE_increase_eV=float(effect.mean())))
            timing.append(dict(fold=fold,variant=name,seconds=time.perf_counter()-start,training_n=len(tr)))
            print('fold',fold,name,'deltaMAE',float(effect.mean()),flush=True)
        pd.DataFrame(predictions).to_csv(out/'PLOT_oof_material_predictions_and_effects.csv',index=False)
        pd.DataFrame(design).to_csv(out/'TABLE_fold_variant_design.csv',index=False)
        pd.DataFrame(metrics).to_csv(out/'TABLE_fold_metrics.csv',index=False)
        pd.DataFrame(timing).to_csv(out/'training_times.csv',index=False)
        if fold==0:
            # Fixed secondary negative control, not a tuning candidate.
            permutation=np.random.default_rng(cfg['seed']).permutation(len(tr))
            neg=XGBRegressor(**cfg['model_parameters']).fit(x[tr],y[tr][permutation])
            npred=neg.predict(x[te]).astype(float)
            joblib.dump(neg,out/'models/fold0_label_permuted_control.joblib',compress=3)
            pd.DataFrame(dict(ID=d.ID.iloc[tr].to_numpy(),original_target_eV=y[tr],permuted_target_eV=y[tr][permutation])).to_csv(out/'fold_manifests/fold0_label_permutation.csv',index=False)
            pd.DataFrame(dict(ID=d.ID.iloc[te].to_numpy(),truth_eV=y[te],prediction_eV=npred)).to_csv(out/'PLOT_negative_control_predictions.csv',index=False)
            dump(out/'NEGATIVE_CONTROL.json',dict(**met(y[te],npred),baseline=met(y[te],full)))
    frame=pd.DataFrame(predictions);design=pd.DataFrame(design);pooled=[];bootrows=[]
    for name,a in frame.groupby('variant'):
        folds=sorted(a.fold.unique().tolist());complete=(len(a)==7176 and len(folds)==5)
        if complete:
            vals=clustered_interval(a.paired_MAE_increase_eV.to_numpy(),a.formula_group.to_numpy())
            for b,v in enumerate(vals):bootrows.append(dict(variant=name,replicate=b,paired_MAE_increase_eV=v))
            lo,hi=np.quantile(vals,[.025,.975])
        else:lo,hi=None,None
        pooled.append(dict(variant=name,folds=json.dumps(folds),complete5fold=complete,**met(a.truth_eV.to_numpy(),a.prediction_eV.to_numpy()),
            paired_MAE_increase_eV=float(a.paired_MAE_increase_eV.mean()),low95_eV=float(lo) if lo is not None else None,high95_eV=float(hi) if hi is not None else None,
            interpretation='full OOF paired retraining effect' if complete else 'fold-specific correlation group; not full-dataset OOF'))
    pooled=pd.DataFrame(pooled);pooled.to_csv(out/'TABLE_pooled_ablation.csv',index=False)
    pd.DataFrame(bootrows).to_csv(out/'PLOT_formula_cluster_bootstrap.csv',index=False)
    pd.DataFrame(edges).to_csv(out/'TABLE_training_only_correlation_edges.csv',index=False)
    single=pooled[pooled.variant.isin(['drop_'+str(j) for j in range(9)])].copy()
    mapping={'drop_'+str(j):f for j,f in enumerate(FEATURES)};single['feature']=single.variant.map(mapping)
    single['rank']=rankdata(-single.paired_MAE_increase_eV.to_numpy());single.sort_values('rank').to_csv(out/'TABLE_nine_single_feature_ranking.csv',index=False)
    # Joint removal effects are not expected to sum: refitting and redundancy change the task.
    nonadd=[]
    fm=pd.DataFrame(metrics).set_index(['fold','variant'])
    for row in design[design.removed_count>1].itertuples():
        removed=json.loads(row.removed_features);individual=sum(float(fm.loc[(row.fold,'drop_'+str(FEATURES.index(f))),'paired_MAE_increase_eV']) for f in removed)
        joint=float(fm.loc[(row.fold,row.variant),'paired_MAE_increase_eV'])
        nonadd.append(dict(fold=row.fold,variant=row.variant,removed_features=row.removed_features,joint_effect_eV=joint,sum_individual_effects_eV=individual,
                           difference_eV=joint-individual,meaning='retraining nonadditivity, not a causal or SHAP interaction coefficient'))
    pd.DataFrame(nonadd).to_csv(out/'TABLE_group_retraining_nonadditivity.csv',index=False)
    dump(out/'BASELINE_REPLAY.json',replays)
    dump(out/'EXECUTION_RESULT.json',dict(status='completed_pending_QA',target='original database band_gap_eV; no ALIGNN teacher',
        rows=7176,formula_groups=int(d.formula_group.nunique()),folds=5,total_fits=len(design),
        grouped_split=True,groups_learned_from_training_features_only=True,threshold=.8,hyperparameter_tuning=False,
        no_physical_causality_claim=True,DFT_not_executed=True))
    print('XGBOOST ABLATION COMPLETE',flush=True)


if __name__=='__main__':main()
