
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from common.evaluation import dump, metrics
from common.data import explanation_parents
from common.oracle import ProxyOracle, ExternalCrystalOracle
from shared_features.extraction import ConceptExtractor, FEATURES
from crispx.perturb import generate_candidates
from crispx.core import informative_subset, fit_response, bootstrap_coefficients
from crispx.path_model import path_balanced_subset,fit_paths,predict_paths,path_key


def run(config, shared, out):
    start = time.perf_counter()
    shared, out = Path(shared), Path(out); out.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(shared/'proxy_dataset_1291.csv', dtype={'ID':str})
    parents = explanation_parents(data,config['explain_parents'],config['seed'])
    extractor = ConceptExtractor(config['atomic_table'])
    oracle = (ExternalCrystalOracle(config['direct_oracle_plugin']) if config.get('direct_oracle_plugin')
              else ProxyOracle(shared/'proxy.joblib', extractor))
    scale = np.load(shared/'training_scale.npy')
    rows, responses, manifests, importance = [], [], [], []
    support_rows, directions, ablation_rows, path_rows = [], [], [], []
    for _, parent in parents.iterrows():
        s = Structure.from_file(parent.path)
        x0, _ = extractor.extract(s)
        f0 = float(oracle.predict_structures([s])[0])
        (out/'structures').mkdir(exist_ok=True)
        dump(out/'structures'/f'{parent.ID}_parent.json', s.as_dict())
        for seed_number,seed in enumerate(config['explanation_seeds']):
            crystals, x, events, rejected = generate_candidates(s, extractor, seed, config['candidate_pool'])
            z = (x-x0)/scale
            # Split independent candidates into a fitting pool and a held-out response set.
            val_n = config['response_holdout']
            fit_n = len(x)-val_n
            budget = min(config['query_budget'], fit_n)
            chosen = path_balanced_subset(z[:fit_n], events[:fit_n], budget)
            yfit = oracle.predict_structures([crystals[i] for i in chosen])-f0
            fit = fit_response(z[chosen], yfit)
            # Explicit held-out teacher calls are counted separately from explanation queries.
            eval_idx = np.arange(fit_n,len(x))
            ytest = oracle.predict_structures([crystals[i] for i in eval_idx])-f0
            linear_pred = z[eval_idx]@fit['coefficient']
            path_fit=fit_paths(z[chosen],yfit,[events[j] for j in chosen])
            pred,supported=predict_paths(path_fit,z[eval_idx],[events[j] for j in eval_idx])
            report = metrics(ytest[supported],pred[supported]) if supported.any() else dict(n=0,mae_eV=None,rmse_eV=None,r2=None)
            for key,path_result in path_fit.items():
                path_rows.append(dict(ID=parent.ID,seed=seed,path=key,
                    joint_jump_eV=float(path_result['joint_jump_at_unperturbed_geometry']),
                    length_response_standardized=float(path_result['coefficient'][0]),
                    angle_response_standardized=float(path_result['coefficient'][1]),n_queried=path_result['n'],
                    within_path_geometry_rank=path_result['geometry_rank']))
            # Equal fitting budgets for a random-query ablation; heldout responses
            # never enter selection. Only newly required calls are issued.
            random_idx=np.random.default_rng(seed+10000).choice(fit_n,budget,replace=False)
            known_y={int(j):float(y) for j,y in zip(chosen,yfit)}
            new_idx=[int(j) for j in random_idx if int(j) not in known_y]
            if new_idx:
                additional=oracle.predict_structures([crystals[j] for j in new_idx])-f0
                known_y.update({j:float(y) for j,y in zip(new_idx,additional)})
            random_y=np.array([known_y[int(j)] for j in random_idx])
            random_fit=fit_response(z[random_idx],random_y)
            untruncated=fit_response(z[chosen],yfit,relative_singular_cutoff=1e-10)
            for label,other in [('random_queries',random_fit),('no_singular_threshold',untruncated)]:
                ablation_rows.append(dict(ID=parent.ID,seed=seed,variant=label,
                    rank=other['rank'],fitting_budget=budget,
                    **metrics(ytest,z[eval_idx]@other['coefficient'])))
            random_paths=fit_paths(z[random_idx],random_y,[events[j] for j in random_idx])
            random_pred,random_supported=predict_paths(random_paths,z[eval_idx],[events[j] for j in eval_idx])
            if random_supported.any():
                ablation_rows.append(dict(ID=parent.ID,seed=seed,variant='random_queries_with_path_model',
                    rank=random_fit['rank'],fitting_budget=budget,coverage=float(random_supported.mean()),
                    **metrics(ytest[random_supported],random_pred[random_supported])))
            ablation_rows.append(dict(ID=parent.ID,seed=seed,variant='pooled_linear_with_same_balanced_queries',
                    rank=fit['rank'],fitting_budget=budget,coverage=1.,**metrics(ytest,linear_pred)))
            lo, hi = bootstrap_coefficients(z[chosen], yfit, seed, repeats=config['bootstrap_repeats'])
            scores = np.mean(np.abs(z[chosen]*fit['coefficient']), axis=0)
            importance.append(dict(ID=parent.ID,seed=seed,**dict(zip(FEATURES,scores))))
            support_rows.append(dict(ID=parent.ID,seed=seed,**dict(zip(FEATURES,fit['support']))))
            rows.append(dict(ID=parent.ID,seed=seed,rank=fit['rank'],fit_queries=budget+int(seed_number==0),
                             ablation_additional_queries=len(new_idx),
                             heldout_queries=val_n,pool_n=len(x),rejected=len(rejected),
                             path_coverage=float(supported.mean()),
                             pooled_linear_rmse_eV=metrics(ytest,linear_pred)['rmse_eV'],**report))
            for j,f in enumerate(FEATURES):
                directions.append(dict(ID=parent.ID,seed=seed,concept=f,coefficient_standardized=fit['coefficient'][j],
                                       ci_low=lo[j],ci_high=hi[j],support=fit['support'][j],
                                       individual_identifiable=bool(fit['individual_identifiable'][j])))
            selected = set(chosen.tolist())
            for j,event in enumerate(events):
                role = 'queried_fit' if j in selected else 'heldout_response' if j>=fit_n else 'unqueried_pool'
                manifests.append(dict(ID=parent.ID,seed=seed,role=role,**event))
                # Save every generated structure, making all accepted perturbations replayable.
                dump(out/'structures'/f'{parent.ID}_{seed}_{j}.json',crystals[j].as_dict())
            for j,t,p,ok,lp in zip(eval_idx,ytest,pred,supported,linear_pred):
                responses.append(dict(ID=parent.ID,seed=seed,candidate=int(j),true_delta_eV=t,pred_delta_eV=p,
                                      path_supported=bool(ok),pooled_linear_delta_eV=lp,path=path_key(events[j]),
                                      **{'delta_'+f:float(x[j,k]-x0[k]) for k,f in enumerate(FEATURES)}))
            # Full candidate coordinates and rejected reasons preserve auditability.
            pd.DataFrame(x,columns=FEATURES).to_csv(out/f'concept_pool_{parent.ID}_{seed}.csv',index=False)
            dump(out/f'rejected_{parent.ID}_{seed}.json',rejected)
        print('CRISP-X parent',parent.ID,'done',flush=True)
    pd.DataFrame(rows).to_csv(out/'local_fidelity.csv',index=False)
    pd.DataFrame(responses).to_csv(out/'heldout_responses.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'local_scores.csv',index=False)
    pd.DataFrame(support_rows).to_csv(out/'identifiability_support.csv',index=False)
    pd.DataFrame(directions).to_csv(out/'coefficients_with_uncertainty.csv',index=False)
    pd.DataFrame(ablation_rows).to_csv(out/'component_ablation.csv',index=False)
    pd.DataFrame(path_rows).to_csv(out/'joint_chemical_path_responses.csv',index=False)
    dump(out/'perturbation_manifest.json',manifests)
    arr = pd.DataFrame(importance)[FEATURES].to_numpy()
    identified=pd.DataFrame(directions).groupby('concept')['individual_identifiable'].mean().reindex(FEATURES).to_numpy()
    pd.DataFrame(dict(concept=FEATURES,score=arr.mean(axis=0),individual_identifiable_fraction=identified,
                     has_high_projection_support=identified>=.8,
                     validated_as_independent_physical_importance=False,
                     score_scope='pooled projected diagnostic; path-model fidelity does not validate these scalar scores')).sort_values('score',ascending=False).to_csv(out/'global_ranking.csv',index=False)
    resp = pd.DataFrame(responses)
    eligible_resp=resp[resp.path_supported]
    summary = {'method':'CRISP-X v0.2', 'oracle':oracle.kind, 'parents_n':len(parents),
               'seeds':config['explanation_seeds'],'teacher_calls_total':oracle.calls,
               'explanation_calls':int(sum(r['fit_queries'] for r in rows)),
               'evaluation_calls':int(sum(r['heldout_queries'] for r in rows)),
               'ablation_additional_calls':int(sum(r['ablation_additional_queries'] for r in rows)),
               'aggregate_heldout_response':metrics(eligible_resp.true_delta_eV,eligible_resp.pred_delta_eV),
               'response_coverage':float(resp.path_supported.mean()),'unsupported_response_n':int((~resp.path_supported).sum()),
               'pooled_linear_response_diagnostic':metrics(resp.true_delta_eV,resp.pooled_linear_delta_eV),
               'seconds':time.perf_counter()-start,
               'interpretation':'predictive model separates finite chemical paths and within-path geometry. Nine scalar scores remain pooled projected diagnostics and are not validated by path-response fidelity. Use joint path table for chemical conclusions; no Shapley or causal claim',
               'uncertainty':'within-neighborhood bootstrap intervals, not population-level confidence intervals',
               'missing':'full direct ALIGNN replay and independent DFT validation remain pending'}
    dump(out/'summary.json',summary)
    print('CRISP-X',summary['aggregate_heldout_response'],flush=True)
