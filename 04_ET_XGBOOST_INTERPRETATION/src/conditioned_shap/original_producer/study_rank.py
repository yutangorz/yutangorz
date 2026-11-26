import argparse,hashlib,json,time,itertools
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr,kendalltau
from sklearn.metrics import r2_score
from rank_core import lime_contrast,shap_contrast,paired_bootstrap,row_spearman,recover_legacy_coefficient_blocks
from shap_baseline.importance import explain_tree

FEATURES=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']
METHODS=['SHAP_contrast','LIME_standard512_contrast','LIME_constrained9_contrast']
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    for e in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/e['path'])==e['sha256']
    out=run/'results';out.mkdir(exist_ok=False);(out/'shap_baseline').mkdir();(out/'lime_baseline').mkdir();(out/'aligned_comparison').mkdir()
    model=joblib.load(run/'inputs/property_model.joblib');d=pd.read_csv(run/'inputs/model_input_7176.csv',dtype={'ID':str})
    folds=pd.read_csv(run/'inputs/outer_fold_manifest.csv',dtype={'ID':str});d=d.merge(folds[['ID','outer_fold']],on='ID',validate='one_to_one')
    scale=d[d.outer_fold!=0][FEATURES].to_numpy().std(axis=0);assert (scale>0).all()
    native=pd.read_csv(run/'inputs/per_material_nine_attributions.csv',dtype={'ID':str}).set_index('ID')
    weights=pd.read_csv(run/'inputs/lime_weights_and_origin.csv',dtype={'ID':str})
    source_predictions=pd.read_csv(run/'inputs/response_predictions.csv',dtype={'ID':str})
    parents=pd.read_csv(run/'inputs/new_parent_manifest.csv',dtype={'ID':str}).ID.tolist()
    manifest=json.loads((run/'inputs/pool_manifest.json').read_text(encoding='utf-8'))
    raw=[];queries=[];replay=[];start=time.perf_counter()
    for n,r in enumerate(manifest):
        ID,seed=r['ID'],r['seed'];folder=run/'inputs/pools'/f'{ID}_{seed}';pool=pd.read_csv(folder/'pool.csv')
        idx=np.load(folder/'indices.npz')['test'];x=pool.loc[idx,FEATURES].to_numpy();x0=native.loc[ID,FEATURES].to_numpy(float)
        events=json.loads((folder/'events.json').read_text(encoding='utf-8'));base=native.loc[ID,['SHAP_'+f for f in FEATURES]].to_numpy(float)
        t=time.perf_counter();v,b,p,res=explain_tree(model,x);sc=shap_contrast(v,base)
        actual=p-float(model.predict(x0[None,:])[0]);saved=pool.loc[idx,'delta_frozen_ET'].to_numpy()
        assert np.max(abs(actual-saved))<1e-9,'frozen physical response replay'
        assert np.max(abs(sc.sum(axis=1)-actual))<1e-9,'contrast completeness'
        contributions={'SHAP_contrast':sc}
        for source,target in [('standard_LIME512','LIME_standard512_contrast'),('constrained_LIME9_matched','LIME_constrained9_contrast')]:
            a=weights[(weights.ID==ID)&(weights.seed==seed)&(weights.method==source)]
            blocks=recover_legacy_coefficient_blocks(a,FEATURES)
            assert len(blocks)==(1 if source=='standard_LIME512' else 2)
            for bi,beta in enumerate(blocks):
                model_name=['frozen_ET','XGBoost'][bi]
                truth=source_predictions[(source_predictions.ID==ID)&(source_predictions.seed==seed)&(source_predictions.model==model_name)&(source_predictions.method==source)].set_index('candidate').loc[idx,'prediction'].to_numpy()
                reconstructed=lime_contrast(x-x0,scale,beta).sum(axis=1)
                error=float(abs(truth-reconstructed).max());assert error<1e-9,'legacy model block replay'
                replay.append(dict(ID=ID,seed=seed,method=source,decoded_model=model_name,max_replay_error_eV=error))
            contributions[target]=lime_contrast(x-x0,scale,blocks[0])
        for method,c in contributions.items():
            for k,j in enumerate(idx):
                row=dict(ID=ID,seed=seed,candidate=int(j),method=method,kind='geometry' if events[j]['kind']=='strain_and_internal_displacement' else 'chemical',
                    path='geometry_only' if events[j]['kind']=='strain_and_internal_displacement' else events[j]['old']+'->'+events[j]['new'],
                    truth_delta_eV=float(actual[k]),reconstructed_delta_eV=float(c[k].sum()))
                for f,value in zip(FEATURES,c[k]):row[f]=float(value)
                raw.append(row)
        queries.append(dict(ID=ID,seed=seed,SHAP_queries=len(x),seconds=time.perf_counter()-t,max_additivity_residual_eV=float(abs(res).max()),
                            max_contrast_residual_eV=float(abs(sc.sum(axis=1)-actual).max())))
        pd.DataFrame(raw).to_csv(out/'aligned_comparison/per_response_nine_contrasts.csv',index=False)
        pd.DataFrame(queries).to_csv(out/'shap_baseline/query_cost_and_residual.csv',index=False)
        pd.DataFrame(replay).to_csv(out/'lime_baseline/legacy_model_block_replay.csv',index=False)
        print('aligned pool',n+1,'/',len(manifest),ID,seed,flush=True)
    frame=pd.DataFrame(raw);saved=pd.read_csv(run/'inputs/response_predictions.csv',dtype={'ID':str})
    for ours,theirs in [('LIME_standard512_contrast','standard_LIME512'),('LIME_constrained9_contrast','constrained_LIME9_matched')]:
        a=frame[frame.method==ours].merge(saved[(saved.model=='frozen_ET')&(saved.method==theirs)][['ID','seed','candidate','prediction']],on=['ID','seed','candidate'],validate='one_to_one')
        error=float(abs(a.reconstructed_delta_eV-a.prediction).max());assert error<1e-9
        replay.append(dict(method=ours,max_replay_error_eV=error))
    dump(out/'aligned_comparison/REPLAY_CHECKS.json',replay)
    records=[];rankrows=[];corr=[];quality=[];bootrows=[];corrboot=[];parentrows=[]
    for scope in ['all','geometry','chemical']:
        a=frame if scope=='all' else frame[frame.kind==scope]
        parent_scores=np.zeros((len(parents),len(METHODS),9))
        for mi,method in enumerate(METHODS):
            b=a[a.method==method]
            # absolute per-response -> mean per seed -> mean per parent -> equal parent mean
            scores=b.assign(**{f:abs(b[f]) for f in FEATURES}).groupby(['ID','seed'])[FEATURES].mean().groupby('ID').mean().loc[parents]
            parent_scores[:,mi]=scores.to_numpy()
            for ID,row in scores.iterrows():
                for f,value in row.items():parentrows.append(dict(scope=scope,ID=ID,method=method,feature=f,mean_abs_contrast_eV=value))
            truth=b.truth_delta_eV.to_numpy();pred=b.reconstructed_delta_eV.to_numpy()
            m=dict(scope=scope,method=method,n=len(b),mae_eV=float(np.mean(abs(truth-pred))),r2=float(r2_score(truth,pred)))
            m['interpretation_status']='exact model-allocation contrast; not independent physical truth' if method=='SHAP_contrast' else ('limited_response_fidelity' if m['r2']>=.8 and m['mae_eV']<=.05 else 'diagnostic_only_fidelity_gate_failed')
            quality.append(m)
        point=parent_scores.mean(axis=0);boot=paired_bootstrap(parent_scores);brank=rankdata(-boot,axis=2);rank=rankdata(-point,axis=1)
        for mi,method in enumerate(METHODS):
            for j,f in enumerate(FEATURES):
                rankrows.append(dict(scope=scope,method=method,feature=f,score_eV=point[mi,j],share=float(point[mi,j]/point[mi].sum()),rank=float(rank[mi,j]),
                    low95_eV=float(np.quantile(boot[:,mi,j],.025)),high95_eV=float(np.quantile(boot[:,mi,j],.975)),
                    rank_low95=float(np.quantile(brank[:,mi,j],.025)),rank_high95=float(np.quantile(brank[:,mi,j],.975))))
            for rep in range(len(boot)):
                row=dict(scope=scope,method=method,replicate=rep)
                row.update(dict(zip(FEATURES,boot[rep,mi])));bootrows.append(row)
        for i,j in itertools.combinations(range(len(METHODS)),2):
            for sub,cols in [('nine',np.arange(9)),('atomic7',np.arange(7))]:
                rho=float(spearmanr(point[i,cols],point[j,cols]).statistic);tau=float(kendalltau(point[i,cols],point[j,cols]).statistic)
                rb=row_spearman(boot[:,i,cols],boot[:,j,cols]);finite=rb[np.isfinite(rb)]
                corr.append(dict(scope=scope,feature_scope=sub,method_a=METHODS[i],method_b=METHODS[j],n_features=len(cols),spearman=rho,kendall_tau=tau,
                    spearman_low95=float(np.quantile(finite,.025)) if len(finite) else None,spearman_high95=float(np.quantile(finite,.975)) if len(finite) else None,
                    bootstrap_valid_n=len(finite)))
                for rep,v in enumerate(rb):corrboot.append(dict(scope=scope,feature_scope=sub,method_a=METHODS[i],method_b=METHODS[j],replicate=rep,spearman=v))
    for name,rows in [('TABLE_aligned_rankings.csv',rankrows),('TABLE_rank_agreement.csv',corr),('TABLE_response_fidelity.csv',quality),('PLOT_parent_scores.csv',parentrows),('PLOT_importance_bootstrap.csv',bootrows),('PLOT_rank_correlation_bootstrap.csv',corrboot)]:
        pd.DataFrame(rows).to_csv(out/'aligned_comparison'/name,index=False)
    # Preserve the original population-level SHAP ranking as a DIFFERENT estimand.
    pd.read_csv(run/'inputs/TABLE_nine_importance.csv').to_csv(out/'shap_baseline/original_global_SHAP_1462.csv',index=False)
    dump(out/'EXECUTION_RESULT.json',dict(status='completed_pending_QA',parents=len(parents),physical_heldout_responses=len(frame)//3,
        estimand='equal-parent mean absolute per-feature contribution to fixed parent-to-physical-perturbation prediction contrasts',
        shap='delta of TreeSHAP attributions with same tree-path reference; interaction redistribution can involve unchanged features',
        lime='local coefficient per trainingSD times actual descriptor displacement/trainingSD',
        comparison_not_same_causal_game=True,bootstrap='paired parent bootstrap; seeds nested in parent; fixed model',
        structural_two_feature_rank_correlation_not_used=True,DFT_not_executed=True,seconds=time.perf_counter()-start))
    print('RANK ALIGNMENT COMPLETE',flush=True)


if __name__=='__main__':main()
