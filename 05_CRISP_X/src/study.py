"""Frozen audit of CRISP-X: baselines, negative controls, generalization, scope."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from pymatgen.core import Structure,Lattice
from shared_features.extraction import ConceptExtractor,FEATURES
from crispx.perturb import generate_candidates
from crispx.path_model import path_balanced_subset,fit_paths,predict_paths,path_key
from refined import weighted_local
from audited_paths import fit_audited,predict_audited,svd_audit

def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float);mask=np.isfinite(y)&np.isfinite(p);n=len(y);y,p=y[mask],p[mask];eligible=abs(y)>=.05
    return dict(total_n=n,n=len(y),coverage=float(mask.mean()) if n else 0.,mae_eV=float(np.mean(abs(y-p))) if len(y) else None,rmse_eV=float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,r2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>1e-20 else None,direction_n=int(eligible.sum()),direction_accuracy=float(np.mean(np.sign(y[eligible])==np.sign(p[eligible]))) if eligible.any() else None)

def make_pool(s,extractor,seed):
    structures=[];x=[];events=[];train=[];test=[];amplitude=[];rejected=[]
    for j,a in enumerate([.25,.5,1.]):
        ss,xx,ee,rr=generate_candidates(s,extractor,seed+1000*j,64,geometry_scale=a);assert len(xx)==64
        start=len(x);structures+=ss;x.extend(xx);events.extend(ee);train.extend(range(start,start+48));test.extend(range(start+48,start+64));amplitude.extend([a]*64);rejected+=rr
    return structures,np.array(x),events,np.array(train),np.array(test),np.array(amplitude),rejected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run;cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    for e in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/e['path'])==e['sha256'],e['path']
    out=run/'results';out.mkdir(exist_ok=False)
    for f in ['common','new_parents','legacy_audit','ood','invariance','synthetic','models']:(out/f).mkdir()
    d=pd.read_csv(run/'inputs/model_input_7176.csv',dtype={'ID':str});fold=pd.read_csv(run/'inputs/outer_fold_manifest.csv',dtype={'ID':str});d=d.merge(fold[['ID','outer_fold']],on='ID',validate='one_to_one');train=d[d.outer_fold!=0];xx=train[FEATURES].to_numpy();yy=train.band_gap_eV.to_numpy();scale=xx.std(axis=0);scale[scale==0]=1
    model=joblib.load(run/'inputs/property_model.joblib');models={'frozen_ET':model};rng=np.random.default_rng(cfg['seed']);shuffled=yy[rng.permutation(len(yy))]
    random_model=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=1,max_features=1.,random_state=cfg['seed'],n_jobs=2).fit(xx,shuffled);models['label_permuted_ET']=random_model
    other=XGBRegressor(n_estimators=400,learning_rate=.04,max_depth=6,reg_lambda=5.,min_child_weight=3,subsample=.9,colsample_bytree=1.,tree_method='hist',objective='reg:squarederror',random_state=20261019,n_jobs=2).fit(xx,yy);models['XGBoost']=other
    for name,m in models.items():
        if name!='frozen_ET':joblib.dump(m,out/'models'/f'{name}.joblib',compress=3)
    dump(out/'models/parameters.json',{name:str(m.get_params()) for name,m in models.items()});pd.DataFrame({'training_ID':train.ID.to_numpy(),'original_target_eV':yy,'permuted_target_eV':shuffled}).to_csv(out/'models/label_permutation_manifest.csv',index=False)
    # Exact replay of prior fits, then evaluate omitted baselines and origin.
    legacy_rows=[];origin=[];legacy_manifest=json.loads((run/'inputs/legacy_pool_manifest.json').read_text(encoding='utf-8'))
    for r in legacy_manifest:
        ID,seed=r['ID'],r['seed'];pool=pd.read_csv(run/'inputs/legacy_pools'/f'{ID}_{seed}.csv');x0=d.set_index('ID').loc[ID,FEATURES].to_numpy(float);z=(pool[FEATURES].to_numpy()-x0)/scale;y=pool.true_model_delta_eV.to_numpy();events=json.loads((run/'inputs/legacy_pools'/f'{ID}_{seed}_events.json').read_text(encoding='utf-8'));idx=np.load(run/'inputs/legacy_pools'/f'{ID}_{seed}_contrast.npz')['chosen_indices'];test=np.array(r['test_indices']);fit=fit_paths(z[idx],y[idx],[events[j] for j in idx]);v,ok=predict_paths(fit,z[test],[events[j] for j in test]);aud=fit_audited(z[idx],y[idx],[events[j] for j in idx]);v5,_,mean,_=predict_audited(aud,z[test],[events[j] for j in test]);p0,_=predict_paths(fit,np.zeros((1,9)),[dict(kind='strain_and_internal_displacement')]);p05,*_=predict_audited(aud,np.zeros((1,9)),[dict(kind='strain_and_internal_displacement')]);origin.append(dict(ID=ID,seed=seed,phase=r['phase'],legacy_origin_delta_eV=float(p0[0]),audited_origin_delta_eV=float(p05[0])))
        for k,j in enumerate(test):legacy_rows.append(dict(ID=ID,seed=seed,candidate=int(j),phase=r['phase'],path=path_key(events[j]),kind='geometry' if path_key(events[j])=='geometry_only' else 'chemical',truth=y[j],legacy_prediction=v[k],audited_prediction=v5[k],path_mean=mean[k],supported=bool(ok[k])))
    pd.DataFrame(legacy_rows).to_csv(out/'legacy_audit/replayed_and_baselines.csv',index=False);pd.DataFrame(origin).to_csv(out/'legacy_audit/origin_identity.csv',index=False)
    extractor=ConceptExtractor(run/'inputs/atom_importance.csv');parents=pd.read_csv(run/'inputs/new_parent_manifest.csv',dtype={'ID':str}).merge(d,on='ID',validate='one_to_one');assert len(parents)==24 and (parents.outer_fold==0).all()
    responses=[];counts=[];origins=[];transfer=[];invariance=[];ood=[];cost=[];starttime=time.perf_counter()
    def record(ID,seed,model_name,method,ztest,ytest,events,indices,p,means,outside,regime='same_generator_holdout',amp=None):
        for k,j in enumerate(indices):
            key=path_key(events[k]);responses.append(dict(ID=ID,seed=seed,model=model_name,method=method,regime=regime,candidate=int(j),path=key,kind='geometry' if key=='geometry_only' else 'chemical',truth_eV=float(ytest[k]),predicted_eV=float(p[k]) if np.isfinite(p[k]) else np.nan,training_path_mean_eV=float(means[k]) if np.isfinite(means[k]) else np.nan,outside_training_path_geometry_box=bool(outside[k]),physical_amplitude=float(amp[k]) if amp is not None else np.nan))
    for position,parent in enumerate(parents.itertuples()):
        structure=Structure.from_file(parent.path);assert sha(parent.path)==parent.structure_sha256
        x0,_=extractor.extract(structure);assert np.allclose(x0,np.array([getattr(parent,f) for f in FEATURES]),atol=1e-10);folder=out/'new_parents'/parent.ID;folder.mkdir();dump(folder/'parent.json',structure.as_dict())
        # Symmetries act on raw structure, with all concepts recomputed.
        if position<8:
            q,_=np.linalg.qr(np.random.default_rng(cfg['seed']+position).normal(size=(3,3)));rot=Structure(Lattice(structure.lattice.matrix@q),[s.specie.symbol for s in structure],structure.frac_coords);translation=structure.copy();translation.translate_sites(range(len(structure)),[.137,.219,.331],frac_coords=True,to_unit_cell=True);permuted=Structure(structure.lattice,[s.specie.symbol for s in structure][::-1],structure.frac_coords[::-1]);supercell=structure.copy();supercell.make_supercell([2,1,1])
            for name,s in [('rotation',rot),('translation',translation),('site_permutation',permuted),('supercell_2x1x1',supercell)]:
                v,_=extractor.extract(s);invariance.append(dict(ID=parent.ID,transform=name,max_concept_absolute_difference=float(np.max(abs(v-x0))),prediction_difference_eV=float(model.predict(v[None,:])[0]-model.predict(x0[None,:])[0])));dump(out/'invariance'/f'{parent.ID}_{name}.json',s.as_dict())
        for seed in cfg['seeds']:
            t=time.perf_counter();structures,x,events,fitidx,testidx,amps,rejected=make_pool(structure,extractor,seed);featuretime=time.perf_counter()-t;z=(x-x0)/scale;chosen=fitidx[path_balanced_subset(z[fitidx],[events[j] for j in fitidx],64)];tevents=[events[j] for j in testidx]
            pool=pd.DataFrame(x,columns=FEATURES);pool['candidate']=np.arange(len(x));pool['role']='fit';pool.loc[testidx,'role']='heldout';pool['physical_amplitude']=amps;pool['path']=[path_key(e) for e in events]
            for name,m in models.items():pool['delta_'+name]=m.predict(x)-float(m.predict(x0[None,:])[0])
            pool.to_csv(folder/f'pool_{seed}.csv',index=False);dump(folder/f'events_{seed}.json',events);dump(folder/f'rejected_{seed}.json',rejected);np.savez_compressed(folder/f'indices_{seed}.npz',fit=fitidx,test=testidx,chosen64=chosen)
            for j,s in enumerate(structures):dump(folder/f'structure_{seed}_{j}.json',s.as_dict())
            for name,m in models.items():
                y=pool['delta_'+name].to_numpy();fite=[events[j] for j in chosen];t=time.perf_counter();old=fit_paths(z[chosen],y[chosen],fite);oldp,_=predict_paths(old,z[testidx],tevents);fit=fit_audited(z[chosen],y[chosen],fite);v,ok,means,outside=predict_audited(fit,z[testidx],tevents)
                for label,p in [('legacy_v04',oldp),('audited_v05',v),('path_mean',means),('zero_response',np.zeros(len(v)))]:record(parent.ID,seed,name,label,z[testidx],y[testidx],tevents,testidx,p,means,outside,amp=amps[testidx])
                f=weighted_local(z[chosen],y[chosen],.0001,1.5,True);p=z[testidx]@f['coefficient'];record(parent.ID,seed,name,'matched_affine64',z[testidx],y[testidx],tevents,testidx,p,means,outside,amp=amps[testidx])
                cost.append(dict(ID=parent.ID,seed=seed,model=name,base_queries=1,fitting_queries=64,evaluation_queries=48,feature_generation_seconds_shared=featuretime,path_and_baseline_fitting_seconds=time.perf_counter()-t,actual_driver_cache_rows=192))
                if name!='frozen_ET':continue
                p0,*_=predict_audited(fit,np.zeros((1,9)),[dict(kind='strain_and_internal_displacement')]);origins.append(dict(ID=parent.ID,seed=seed,audited_origin_delta_eV=float(p0[0])))
                # Within-path shuffling preserves lookup means, destroys geometry association.
                targets={'global_shuffle':np.random.default_rng(seed+40000).permutation(y[chosen])}
                perm=y[chosen].copy();keys=np.array([path_key(e) for e in fite]);rng=np.random.default_rng(seed+50000)
                for key in sorted(set(keys)):
                    ii=np.flatnonzero(keys==key);perm[ii]=perm[rng.permutation(ii)]
                targets['within_path_shuffle']=perm
                targets['constant_model']=np.zeros(len(chosen))
                for label,target in targets.items():
                    f=fit_audited(z[chosen],target,fite);p,*_=predict_audited(f,z[testidx],tevents);truth=np.zeros(len(testidx)) if label=='constant_model' else y[testidx];reference=np.zeros(len(testidx)) if label=='constant_model' else means;record(parent.ID,seed,name,label,z[testidx],truth,tevents,testidx,p,reference,outside,amp=amps[testidx])
                # Budgets and regularization are diagnostics, not post-hoc selection.
                for budget,ridge in [(16,.001),(32,.001),(96,.001),(64,.00001),(64,.1)]:
                    ii=fitidx[path_balanced_subset(z[fitidx],[events[j] for j in fitidx],budget)];f=fit_audited(z[ii],y[ii],[events[j] for j in ii],ridge=ridge);p,*_=predict_audited(f,z[testidx],tevents);record(parent.ID,seed,name,f'budget{budget}_ridge{ridge}',z[testidx],y[testidx],tevents,testidx,p,means,outside,amp=amps[testidx])
                for replicate in range(3):
                    ii=np.random.default_rng(seed+60000+replicate).choice(fitidx,64,replace=False);f=fit_audited(z[ii],y[ii],[events[j] for j in ii]);p,*_=predict_audited(f,z[testidx],tevents);record(parent.ID,seed,name,'random64_'+str(replicate),z[testidx],y[testidx],tevents,testidx,p,means,outside,amp=amps[testidx])
                # Entire chemical-path removal must cause abstention, never a false high R2.
                for key in sorted(set(keys)-{'geometry_only'}):
                    target_idx=np.array([j for j in testidx if path_key(events[j])==key]);keep=chosen[keys!=key]
                    if not len(target_idx):continue
                    f=fit_audited(z[keep],y[keep],[events[j] for j in keep]);p,ok2,*_=predict_audited(f,z[target_idx],[events[j] for j in target_idx]);transfer.append(dict(ID=parent.ID,seed=seed,path=key,heldout_n=len(target_idx),predicted_n=int(ok2.sum()),coverage=float(ok2.mean())))
                # Hold out the largest generator amplitude entirely.
                small=fitidx[amps[fitidx]<1.];large=np.flatnonzero(amps==1.);ii=small[path_balanced_subset(z[small],[events[j] for j in small],64)];f=fit_audited(z[ii],y[ii],[events[j] for j in ii]);p,_,mu,ext=predict_audited(f,z[large],[events[j] for j in large]);record(parent.ID,seed,name,'audited_v05',z[large],y[large],[events[j] for j in large],large,p,mu,ext,regime='largest_amplitude_heldout',amp=amps[large])
                counts.append(dict(ID=parent.ID,seed=seed,fit_n=len(fitidx),test_n=len(testidx),known_fit_paths=len(fit),covered_test=int(ok.sum()),box_extrapolated_test=int(outside[ok].sum())))
                # Direction-specific raw-structure stress probes, fixed in advance.
                if position<12 and seed==cfg['seeds'][0]:
                    for mode in ['isotropic','shear','internal']:
                        for amplitude in [.5,1.,1.5]:
                            for sign in [-1.,1.]:
                                mat=np.eye(3)
                                if mode=='isotropic':mat*=1+sign*.02*amplitude
                                if mode=='shear':mat[0,1]=mat[1,0]=sign*.02*amplitude
                                lattice=Lattice(structure.lattice.matrix@mat);coords=structure.frac_coords.copy()
                                if mode=='internal':
                                    shift=np.random.default_rng(cfg['seed']+position).normal(size=(len(structure),3));shift-=shift.mean(axis=0);shift*=sign*.02*amplitude/max(np.sqrt(np.mean(shift**2)),1e-12);coords=(coords+shift@np.linalg.inv(lattice.matrix))%1
                                s=Structure(lattice,[site.specie.symbol for site in structure],coords);v,q=extractor.extract(s);valid=q['min_shell_distance']>=max(.65,.75*parent.min_shell_distance) and .85<s.volume/structure.volume<1.15
                                key=f'{parent.ID}_{mode}_{amplitude}_{sign}'
                                if valid:
                                    zz=(v-x0)/scale;truth=float(model.predict(v[None,:])[0]-model.predict(x0[None,:])[0]);pp,*_=predict_audited(fit,zz[None,:],[dict(kind='strain_and_internal_displacement')]);legacy,*_=predict_paths(old,zz[None,:],[dict(kind='strain_and_internal_displacement')]);baseline=float(means[[k for k,j in enumerate(testidx) if path_key(events[j])=='geometry_only'][0]]);ood.append(dict(ID=parent.ID,mode=mode,amplitude=amplitude,sign=sign,valid=True,truth_eV=truth,audited_prediction_eV=float(pp[0]),legacy_prediction_eV=float(legacy[0]),path_mean_prediction_eV=baseline));dump(out/'ood'/f'{key}.json',s.as_dict())
                                else:ood.append(dict(ID=parent.ID,mode=mode,amplitude=amplitude,sign=sign,valid=False,truth_eV=np.nan,audited_prediction_eV=np.nan,legacy_prediction_eV=np.nan,path_mean_prediction_eV=np.nan))
            pd.DataFrame(responses).to_csv(out/'response_predictions.csv',index=False);pd.DataFrame(transfer).to_csv(out/'whole_path_holdout.csv',index=False);pd.DataFrame(cost).to_csv(out/'query_and_time.csv',index=False)
        pd.DataFrame(invariance).to_csv(out/'invariance_checks.csv',index=False);pd.DataFrame(ood).to_csv(out/'ood_direction_probes.csv',index=False);pd.DataFrame(counts).to_csv(out/'case_design.csv',index=False);pd.DataFrame(origins).to_csv(out/'origin_identity.csv',index=False)
        print('new parent',position+1,'/24',parent.ID,'complete',flush=True)
    frame=pd.DataFrame(responses);summary=[]
    for (model_name,method,regime),z in frame.groupby(['model','method','regime']):
        for scope,v in [('all',z),('geometry',z[z.kind=='geometry']),('chemical',z[z.kind=='chemical']),('inside_geometry_box',z[~z.outside_training_path_geometry_box]),('outside_geometry_box',z[z.outside_training_path_geometry_box])]:
            if len(v):summary.append(dict(model=model_name,method=method,regime=regime,scope=scope,**metrics(v.truth_eV,v.predicted_eV)))
        summary.append(dict(model=model_name,method=method,regime=regime,scope='within_path_centered_using_fit_means',**metrics(z.truth_eV-z.training_path_mean_eV,z.predicted_eV-z.training_path_mean_eV)))
    pd.DataFrame(summary).to_csv(out/'TABLE_metrics.csv',index=False)
    # Primary paired test, using only same-supported cases and parent clusters.
    primary=frame[(frame.model=='frozen_ET')&(frame.regime=='same_generator_holdout')];a=primary[primary.method=='audited_v05'];b=primary[primary.method=='path_mean'][['ID','seed','candidate','predicted_eV']].rename(columns={'predicted_eV':'mean_prediction'});pair=a.merge(b,on=['ID','seed','candidate'],validate='one_to_one');pair=pair[pair.predicted_eV.notna()&pair.mean_prediction.notna()];pair['ae_improvement']=abs(pair.truth_eV-pair.mean_prediction)-abs(pair.truth_eV-pair.predicted_eV);pair.to_csv(out/'PLOT_primary_vs_path_mean.csv',index=False)
    bootstrap=[];intervals=[];rng=np.random.default_rng(cfg['seed']+99)
    for scope,z in [('all',pair),('geometry',pair[pair.kind=='geometry']),('chemical',pair[pair.kind=='chemical'])]:
        grouped=z.groupby('ID').ae_improvement.agg(['sum','count']);values=[]
        for j in range(2000):
            ii=rng.integers(0,len(grouped),len(grouped));v=float(grouped['sum'].to_numpy()[ii].sum()/grouped['count'].to_numpy()[ii].sum());values.append(v);bootstrap.append(dict(scope=scope,replicate=j,mae_improvement_eV=v))
        intervals.append(dict(scope=scope,n=len(z),mae_improvement_eV=float(z.ae_improvement.mean()),low95=float(np.quantile(values,.025)),high95=float(np.quantile(values,.975))))
    pd.DataFrame(bootstrap).to_csv(out/'PLOT_parent_bootstrap.csv',index=False)
    macro=[]
    for (name,method,ID),z in frame[frame.regime=='same_generator_holdout'].groupby(['model','method','ID']):macro.append(dict(model=name,method=method,ID=ID,**metrics(z.truth_eV,z.predicted_eV)))
    pd.DataFrame(macro).to_csv(out/'TABLE_parent_macro_metrics.csv',index=False)
    result=dict(status='completed',new_parents=24,seeds=cfg['seeds'],new_structures=24*len(cfg['seeds'])*192,models=list(models),primary_intervals=intervals,whole_path_holdout_prediction_count=int(pd.DataFrame(transfer).predicted_n.sum()),invariance_max_feature_error=float(pd.DataFrame(invariance).max_concept_absolute_difference.max()),invariance_max_prediction_error_eV=float(abs(pd.DataFrame(invariance).prediction_difference_eV).max()),legacy_origin_max_abs_eV=float(abs(pd.DataFrame(origin).legacy_origin_delta_eV).max()),audited_origin_max_abs_eV=float(abs(pd.DataFrame(origins).audited_origin_delta_eV).max()),elapsed_seconds=time.perf_counter()-starttime,scope='auxiliary-model response approximation, not DFT, not nine independent causal contributions')
    dump(out/'VALIDATION_REPORT.json',result);print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
