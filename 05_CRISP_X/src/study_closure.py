"""Reproducible development / frozen confirmation of the CRISP-X closure."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/shap_baseline')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/lime_baseline')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx_closure')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr
from shap_baseline.importance import explain_tree, aggregate
from crispx_closure.paths import block_subset, fit_closed, predict_closed
from audited_paths import fit_audited, predict_audited
from crispx.path_model import path_key, path_balanced_subset
from shared_features.extraction import FEATURES, ConceptExtractor
from lime_baseline.compare import explain as lime_explain
from refined import weighted_local


def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_parents(manifest, data):
    """Join by ID only: manifest metadata must not shadow canonical columns."""
    m=pd.read_csv(manifest,dtype={'ID':str})
    if m.ID.duplicated().any() or not set(m.ID).issubset(set(data.ID)):
        raise ValueError('Invalid parent IDs')
    return m[['ID']].merge(data,on='ID',validate='one_to_one')


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float);ok=np.isfinite(p)&np.isfinite(y)
    a,b=y[ok],p[ok]
    return dict(n=len(y),covered_n=int(ok.sum()),coverage=float(ok.mean()) if len(ok) else 0.,
        mae_eV=float(np.mean(abs(a-b))) if len(a) else None,
        rmse_eV=float(np.sqrt(np.mean((a-b)**2))) if len(a) else None,
        r2=float(r2_score(a,b)) if len(a)>1 and np.var(a)>1e-20 else None)


def summaries(df):
    rows=[]
    for (model,method),a in df.groupby(['model','method']):
        for scope,v in [('all',a),('geometry',a[a.kind=='geometry']),('chemical',a[a.kind=='chemical'])]:
            if len(v):rows.append(dict(model=model,method=method,scope=scope,**metrics(v.truth,v.prediction)))
    return pd.DataFrame(rows)


def response_case(ID,seed,x,x0,events,fitidx,testidx,scale,model_name,model,candidates,with_lime=False,training=None):
    z=(x-x0)/scale;y=model.predict(x)-float(model.predict(x0[None,:])[0]);te=[events[j] for j in testidx]
    out=[];design=[];weights=[];t=time.perf_counter()
    def add(label,p,reference=None,reasons=None,query_n=0):
        for k,j in enumerate(testidx):
            out.append(dict(ID=ID,seed=seed,model=model_name,method=label,candidate=int(j),
                path=path_key(events[j]),kind='geometry' if path_key(events[j])=='geometry_only' else 'chemical',
                truth=float(y[j]),prediction=float(p[k]),strong_baseline=float(reference[k]) if reference is not None else np.nan,
                reason=reasons[k] if reasons is not None else 'baseline',queries=query_n))
    for c in candidates:
        ii=fitidx[block_subset(z[fitidx],[events[j] for j in fitidx],c['budget'],c['geometry_quota'])]
        f=fit_closed(z[ii],y[ii],[events[j] for j in ii]);p,baseline,reasons,box=predict_closed(f,z[testidx],te)
        add(c['name'],p,baseline,reasons,len(ii))
        add(c['name']+'_strong_baseline',baseline,query_n=len(ii))
        for key,m in f.items():
            design.append(dict(ID=ID,seed=seed,model=model_name,configuration=c['name'],path=key,
                selected_candidate_indices=json.dumps([int(j) for j in ii if path_key(events[j])==key]),
                n=m['n'],numerical_rank=m['numerical_geometry_rank'],condition=m['condition_number'],
                prediction_mode=m['prediction_mode'],
                response_qualified=m['response_qualified'],separate_geometry_slopes_qualified=m['separate_geometry_slopes_qualified'],
                intercept_at_origin=m['joint_jump_at_unperturbed_geometry'],
                distance_coefficient=m['coefficient'][0],angle_coefficient=m['coefficient'][1]))
        if with_lime:
            # Same query rows and output budget for these local baselines.
            fit9=weighted_local(z[ii],y[ii],.0001,1.5,True)
            add('constrained_LIME9_matched',z[testidx]@fit9['coefficient'],query_n=len(ii))
            gi=ii[[path_key(events[j])=='geometry_only' for j in ii]]
            geom=weighted_local(z[gi],y[gi],.0001,.5,True,columns=[7,8])
            pp=z[testidx]@geom['coefficient'];pp[[path_key(e)!='geometry_only' for e in te]]=np.nan
            add('constrained_LIME_geometry_matched',pp,query_n=len(gi))
            for j,name in enumerate(FEATURES):
                weights.append(dict(ID=ID,seed=seed,method='constrained_LIME9_matched',feature=name,
                    coefficient_per_training_SD=fit9['coefficient'][j]))
            # Confirm exact origin identity and unknown-path abstention.
            origin=predict_closed(f,np.zeros((1,9)),[dict(kind='strain_and_internal_displacement')])[0][0]
            assert origin==0.,'geometry origin must be exactly zero'
            unknown=dict(kind='bounded_same_block_species_substitution',old='UNKNOWN',new='UNSEEN')
            assert np.isnan(predict_closed(f,np.zeros((1,9)),[unknown])[0][0])
            rng=np.random.default_rng(seed+900)
            shuffled=y[ii].copy();keys=np.array([path_key(events[j]) for j in ii])
            for key in sorted(set(keys)):
                jj=np.flatnonzero(keys==key);shuffled[jj]=shuffled[rng.permutation(jj)]
            ff=fit_closed(z[ii],shuffled,[events[j] for j in ii]);sp,*_=predict_closed(ff,z[testidx],te)
            add('within_path_label_shuffle',sp,baseline,query_n=len(ii))
            ff=fit_closed(z[ii],np.zeros(len(ii)),[events[j] for j in ii]);cp,*_=predict_closed(ff,z[testidx],te)
            assert np.max(abs(cp[np.isfinite(cp)]),initial=0.)==0
    oldidx=fitidx[path_balanced_subset(z[fitidx],[events[j] for j in fitidx],64)]
    old=fit_audited(z[oldidx],y[oldidx],[events[j] for j in oldidx]);p,*_=predict_audited(old,z[testidx],te)
    add('legacy_v05_64',p,query_n=64)
    if with_lime and model_name=='frozen_ET':
        t_lime=time.perf_counter();a=lime_explain(model,training,x0,x[testidx],seed,512)
        add('standard_LIME512',a['delta'],query_n=512)
        for j,name in enumerate(FEATURES):
            weights.append(dict(ID=ID,seed=seed,method='standard_LIME512',feature=name,
                coefficient_per_training_SD=a['coefficient'][j],origin_error_eV=a['anchor_prediction']-float(model.predict(x0[None,:])[0]),
                own_neighborhood_weighted_r2=a['weighted_r2']))
    return out,design,weights,dict(ID=ID,seed=seed,model=model_name,seconds=time.perf_counter()-t,
                                 driver_candidate_n=len(x),heldout_n=len(testidx))


def synthetic_benchmark(out):
    # Compare TreeSHAP against an independently enumerated 2^9-coalition
    # interventional game. Coalition rows are synthetic numbers, not materials.
    rows=[]
    for seed in [71,83,97]:
        rng=np.random.default_rng(seed);train=rng.normal(size=(80,9));target=train[:,0]+train[:,1]*train[:,2]+.2*train[:,8]
        tree=DecisionTreeRegressor(max_depth=4,random_state=seed).fit(train,target)
        bg=train[:16];query=train[40:43];values,base,p,res=explain_tree(tree,query,bg)
        for k,x in enumerate(query):
            game=np.zeros(512)
            for mask in range(512):
                a=bg.copy();cols=[j for j in range(9) if mask&(1<<j)];a[:,cols]=x[cols]
                game[mask]=tree.predict(a).mean()
            exact=np.zeros(9)
            for j in range(9):
                for mask in range(512):
                    if mask&(1<<j):continue
                    n=mask.bit_count();w=math.factorial(n)*math.factorial(8-n)/math.factorial(9)
                    exact[j]+=w*(game[mask|(1<<j)]-game[mask])
            for j,name in enumerate(FEATURES):rows.append(dict(seed=seed,case=k,feature=name,
                exact_shapley=exact[j],library_shap=values[k,j],abs_error=abs(exact[j]-values[k,j])))
            np.savez_compressed(out/f'coalition_game_{seed}_{k}.npz',game=game,background=bg,query=x)
    df=pd.DataFrame(rows);df.to_csv(out/'known_game_shapley.csv',index=False)
    assert df.abs_error.max()<1e-6
    return dict(exact_game_max_error=float(df.abs_error.max()),coalition_n=512,synthetic_queries=9)


def attribution(run,df,fold,model,others,cfg):
    out=run/'results/shap_baseline';out.mkdir()
    hold=df[df.outer_fold==0].sort_values('ID');xx=hold[FEATURES].to_numpy()
    vv=[];pred=[];res=[];start=time.perf_counter()
    for a in range(0,len(xx),64):
        v,base,p,r=explain_tree(model,xx[a:a+64]);vv.append(v);pred.extend(p);res.extend(r)
        print('SHAP heldout',min(a+64,len(xx)),'/',len(xx),flush=True)
    values=np.vstack(vv)
    detail=hold[['ID','formula_group','band_gap_eV']+FEATURES].copy();detail['prediction_eV']=pred
    detail['base_eV']=base;detail['additivity_residual_eV']=res
    for j,name in enumerate(FEATURES):detail['SHAP_'+name]=values[:,j]
    detail.to_csv(out/'per_material_nine_attributions.csv',index=False)
    point,share,boot,ranks=aggregate(values,hold.formula_group.to_numpy(),seed=cfg['seed'],repeats=2000)
    table=[]
    for j,name in enumerate(FEATURES):
        table.append(dict(feature=name,category='atomic' if j<7 else 'structural',mean_abs_SHAP_eV=point[j],
            share=share[j],rank=int(np.argsort(np.argsort(-point))[j]+1),
            low95_eV=float(np.quantile(boot[:,j],.025)),high95_eV=float(np.quantile(boot[:,j],.975)),
            bootstrap_rank_low=float(np.quantile(ranks[:,j],.025)),bootstrap_rank_high=float(np.quantile(ranks[:,j],.975)),
            convention='TreeSHAP tree_path_dependent; fixed fold0 ExtraTrees'))
    table=pd.DataFrame(table).sort_values('rank');table.to_csv(out/'TABLE_nine_importance.csv',index=False)
    pd.DataFrame(boot,columns=FEATURES).to_csv(out/'PLOT_cluster_bootstrap_importance.csv',index=False)
    pd.DataFrame(ranks,columns=FEATURES).to_csv(out/'PLOT_bootstrap_ranks.csv',index=False)
    # Rank differences across conventions are disclosed, never renamed noise stability.
    ids=pd.read_csv(run/'inputs/new_parent_manifest.csv',dtype={'ID':str}).ID.tolist()
    sample=hold.set_index('ID').loc[ids].reset_index();sx=sample[FEATURES].to_numpy()
    primary=detail.set_index('ID').loc[ids][['SHAP_'+f for f in FEATURES]].to_numpy()
    train=df[df.outer_fold!=0];sensitivity=[];control=[]
    for seed in [809,811]:
        ii=np.random.default_rng(seed).choice(len(train),32,replace=False);bg=train.iloc[ii]
        bg[['ID']+FEATURES].to_csv(out/f'background_{seed}.csv',index=False)
        v,b,p,r=explain_tree(model,sx,bg[FEATURES].to_numpy())
        for k,ID in enumerate(ids):
            for j,f in enumerate(FEATURES):sensitivity.append(dict(ID=ID,seed=seed,feature=f,SHAP_eV=v[k,j],
                primary_SHAP_eV=primary[k,j],base_eV=b,residual_eV=r[k]))
    pd.DataFrame(sensitivity).to_csv(out/'reference_convention_sensitivity.csv',index=False)
    for name,m in others.items():
        v,b,p,r=explain_tree(m,sx,atol=1e-5 if name=='XGBoost' else 1e-6)
        for k,ID in enumerate(ids):
            for j,f in enumerate(FEATURES):control.append(dict(ID=ID,model=name,feature=f,SHAP_eV=v[k,j],
                primary_SHAP_eV=primary[k,j],prediction_eV=p[k],base_eV=b,residual_eV=r[k]))
    pd.DataFrame(control).to_csv(out/'other_model_and_label_permutation.csv',index=False)
    random=pd.DataFrame(control);random=random[random.model=='label_permuted_ET']
    change=float(np.sqrt(np.mean((random.SHAP_eV-random.primary_SHAP_eV)**2)))
    assert change>1e-5,'attribution must respond to model randomization'
    pd.DataFrame(np.corrcoef(train[FEATURES].to_numpy(),rowvar=False),index=FEATURES,columns=FEATURES).to_csv(out/'training_feature_correlations.csv')
    result=dict(n=len(hold),formula_groups=int(hold.formula_group.nunique()),
        max_additivity_residual_eV=float(np.max(abs(res))),model_permuted_attribution_rms_change_eV=change,
        property_prediction=metrics(hold.band_gap_eV,pred),seconds=time.perf_counter()-start,
        reference='tree_path_dependent uses training tree counts; not true conditional/causal',
        bootstrap='resample heldout formula groups with fixed model; no model-fitting uncertainty')
    dump(out/'ATTRIBUTION_VALIDATION.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    for e in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):
        assert sha(run/e['path'])==e['sha256'],e['path']
    out=run/'results';out.mkdir(exist_ok=False)
    df=pd.read_csv(run/'inputs/model_input_7176.csv',dtype={'ID':str})
    fold=pd.read_csv(run/'inputs/outer_fold_manifest.csv',dtype={'ID':str});df=df.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one')
    training=df[df.outer_fold!=0][FEATURES].to_numpy();scale=training.std(axis=0);scale[scale==0]=1
    pd.DataFrame({'feature':FEATURES,'training_SD':scale}).to_csv(out/'training_scaling.csv',index=False)
    model=joblib.load(run/'inputs/property_model.joblib')
    responses=[];design=[];weights=[];cost=[];source=Path(cfg['development_source'])
    models={'frozen_ET':model}
    if cfg['phase']=='confirmation':
        models['XGBoost']=joblib.load(run/'inputs/XGBoost.joblib')
        (out/'physical_pools').mkdir()
        from study import make_pool
        from pymatgen.core import Structure
        extractor=ConceptExtractor(run/'inputs/atom_importance.csv')
        parents=load_parents(run/'inputs/new_parent_manifest.csv',df)
    else:
        parents=pd.read_csv(source/'inputs/new_parent_manifest.csv',dtype={'ID':str})
    for n,parent in enumerate(parents.itertuples()):
        ID=parent.ID;x0=df.set_index('ID').loc[ID,FEATURES].to_numpy(float)
        for seed in cfg['seeds']:
            if cfg['phase']=='development':
                folder=source/'results/new_parents'/ID
                pool=pd.read_csv(folder/f'pool_{seed}.csv');x=pool[FEATURES].to_numpy()
                events=json.loads((folder/f'events_{seed}.json').read_text(encoding='utf-8'))
                idx=np.load(folder/f'indices_{seed}.npz');fitidx,testidx=idx['fit'],idx['test']
            else:
                s=Structure.from_file(parent.path);assert sha(parent.path)==parent.structure_sha256
                extracted,_=extractor.extract(s);assert np.allclose(extracted,x0,atol=1e-10)
                t=time.perf_counter();ss,x,events,fitidx,testidx,amps,rejected=make_pool(s,extractor,seed)
                folder=out/'physical_pools'/f'{ID}_{seed}';folder.mkdir()
                dump(folder/'parent.json',s.as_dict());dump(folder/'events.json',events);dump(folder/'rejected.json',rejected)
                pool=pd.DataFrame(x,columns=FEATURES);pool['role']='fit';pool.loc[testidx,'role']='heldout'
                pool['amplitude']=amps
                for name,m in models.items():pool['delta_'+name]=m.predict(x)-float(m.predict(x0[None,:])[0])
                pool.to_csv(folder/'pool.csv',index=False);np.savez_compressed(folder/'indices.npz',fit=fitidx,test=testidx)
                for j,st in enumerate(ss):dump(folder/f'structure_{j}.json',st.as_dict())
                cost.append(dict(ID=ID,seed=seed,model='shared_features',seconds=time.perf_counter()-t,driver_candidate_n=len(x),heldout_n=len(testidx)))
            for name,m in models.items():
                rr,dd,ww,cc=response_case(ID,seed,x,x0,events,fitidx,testidx,scale,name,m,cfg['candidates'],
                    with_lime=cfg['phase']=='confirmation',training=training)
                responses.extend(rr);design.extend(dd);weights.extend(ww);cost.append(cc)
        pd.DataFrame(responses).to_csv(out/'response_predictions.csv',index=False)
        pd.DataFrame(design).to_csv(out/'query_and_path_design.csv',index=False)
        pd.DataFrame(cost).to_csv(out/'query_cost_and_walltime.csv',index=False)
        if weights:pd.DataFrame(weights).to_csv(out/'lime_weights_and_origin.csv',index=False)
        print(cfg['phase'],'parent',n+1,'/',len(parents),ID,flush=True)
    frame=pd.DataFrame(responses);summary=summaries(frame);summary.to_csv(out/'TABLE_response_metrics.csv',index=False)
    if cfg['phase']=='development':
        table=summary[(summary.model=='frozen_ET')&summary.method.isin([c['name'] for c in cfg['candidates']])]
        viable=[]
        for c in cfg['candidates']:
            a=table[(table.method==c['name'])&(table.scope=='all')].iloc[0]
            g=table[(table.method==c['name'])&(table.scope=='geometry')].iloc[0]
            if a.coverage>=.95 and g.coverage>=.95:viable.append((c,float(g.mae_eV)))
        if not viable:raise RuntimeError('No configuration meets development coverage gate')
        best=min(v for c,v in viable)
        near=[(c,v) for c,v in viable if v<=1.05*best]
        selected=min(near,key=lambda cv:(cv[0]['budget'],cv[1]))[0]
        dump(out/'SELECTED_CONFIGURATION.json',dict(selected=selected,selection_rule='coverage>=.95, minimize geometry MAE; choose lower budget if within5% of best',
             diagnostic_only=True,independent_confirmation_required=True))
    else:
        synthetic=out/'synthetic';synthetic.mkdir();bench=synthetic_benchmark(synthetic)
        other={'XGBoost':models['XGBoost'],'label_permuted_ET':joblib.load(run/'inputs/label_permuted_ET.joblib')}
        att=attribution(run,df,fold,model,other,cfg)
        dump(out/'CLOSURE_EXECUTION.json',dict(status='execution_completed_pending_final_QA',synthetic=bench,attribution=att))
    print('STUDY COMPLETE',cfg['phase'],flush=True)


if __name__=='__main__':main()
