"""External driver only: calls byte-identical original response/generator functions."""

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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/constrained_lime')))
from pathlib import Path
import sys,json,time,hashlib,traceback,warnings
import numpy as np,pandas as pd,joblib
from pymatgen.core import Structure
RUN=_repo_path('06_EXTERNAL_TRANSFER/workflows/constrained_lime')
SOURCE=_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response')
sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/external_response')))
from shared_features.extraction import ConceptExtractor,FEATURES
from study import make_pool
from study_closure import response_case
from crispx.path_model import path_key
from crispx_closure.paths import block_subset,fit_closed

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert json.loads((_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response/qualification/QUALIFICATION_QA.json')).read_text(encoding='utf-8'))['passed']
    for r in json.loads((_repo_path('06_EXTERNAL_TRANSFER/configs/external_response_source_identity_frozen_input_code.json')).read_text(encoding='utf-8')):assert sha(SOURCE/r['path'])==r['sha256'],r['path']
    cfg=json.loads((_repo_path('06_EXTERNAL_TRANSFER/configs/external_response.json')).read_text(encoding='utf-8'))
    model=joblib.load(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/property_model.joblib'))
    ex=ConceptExtractor(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv'))
    d=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str})
    f=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv'),dtype={'ID':str})
    d=d.merge(f[['ID','outer_fold']],on='ID',validate='one_to_one')
    training=d[d.outer_fold!=0][FEATURES].to_numpy();scale=training.std(axis=0);scale[scale==0]=1
    out=_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime');out.mkdir(exist_ok=False);(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/physical_pools')).mkdir()
    pd.DataFrame({'feature':FEATURES,'training_SD':scale}).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/training_scaling.csv'),index=False)
    assert np.allclose(scale,pd.read_csv(_repo_path('outputs/05_CRISP_X/local_reconstruction/training_scaling.csv')).training_SD,atol=1e-14)
    responses=[];design=[];weights=[];cost=[];cases=[];acceptance=[];ident=[]
    start=time.perf_counter()
    parents=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/PILOT38.csv'))
    for n,r in enumerate(parents.itertuples(),1):
        ID=r.external_id
        s=Structure.from_file(SOURCE/r.local_cif)
        x0,q=ex.extract(s);assert np.allclose(x0,np.array([getattr(r,f) for f in FEATURES]),atol=1e-10)
        for seed in cfg['seeds']:
            tick=time.perf_counter();folder=_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/physical_pools')/f'{ID}_{seed}';folder.mkdir()
            try:
                ss,x,events,fitidx,testidx,amps,rejected=make_pool(s,ex,seed)
                assert len(x)==192 and len(testidx)==48 and not set(fitidx)&set(testidx)
                dump(folder/'parent.json',s.as_dict());dump(folder/'events.json',events);dump(folder/'rejected.json',rejected)
                for j,st in enumerate(ss):dump(folder/f'structure_{j}.json',st.as_dict())
                z=(x-x0)/scale
                ii=fitidx[block_subset(z[fitidx],[events[j] for j in fitidx],64,40)]
                fit=fit_closed(z[ii],model.predict(x[ii])-float(model.predict(x0[None,:])[0]),[events[j] for j in ii])
                np.savez_compressed(folder/'indices.npz',fit=fitidx,test=testidx,selected64=ii)
                pool=pd.DataFrame(x,columns=FEATURES);pool['role']='fit';pool.loc[testidx,'role']='heldout';pool['amplitude']=amps;pool['path']=[path_key(e) for e in events];pool['delta_frozen_ET']=model.predict(x)-float(model.predict(x0[None,:])[0]);pool.to_csv(folder/'pool.csv',index=False)
                rr,dd,ww,cc=response_case(ID,seed,x,x0,events,fitidx,testidx,scale,'frozen_ET',model,cfg['candidates'],with_lime=True,training=training)
                responses.extend(rr);design.extend(dd);weights.extend(ww);cost.append(cc)
                for j in testidx:
                    m=fit.get(path_key(events[j]));ident.append({'ID':ID,'seed':seed,'candidate':int(j),'path':path_key(events[j]),'two_geometry_slopes_identifiable':bool(m is not None and m['separate_geometry_slopes_qualified'])})
                ng=sum(path_key(e)=='geometry_only' for e in events)
                acceptance.append({'ID':ID,'seed':seed,'accepted_total':len(x),'accepted_geometry':ng,'accepted_chemical':len(x)-ng,'recorded_rejections':len(rejected),'accepted_geometry_fraction_of_pool':ng/len(x),'accepted_geometric_constraint_pass_fraction':1.,'exact_geometry_attempt_acceptance_rate':None,'attempt_rate_note':'original generator does not log rejected path type or silent duplicate attempts; not inferable'})
                cases.append({'ID':ID,'seed':seed,'status':'COMPLETE','seconds':time.perf_counter()-tick,'planned_test_n':48,'actual_test_n':48})
            except Exception as e:
                dump(folder/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_thresholds_changed':False})
                cases.append({'ID':ID,'seed':seed,'status':'FAILED','seconds':time.perf_counter()-tick,'planned_test_n':48,'actual_test_n':0,'error':repr(e)})
                # Failed cases remain in all-candidate/case and planned-response denominators.
            pd.DataFrame(cases).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/CASE_EXECUTION.csv'),index=False)
            pd.DataFrame(responses).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/response_predictions.csv'),index=False)
            pd.DataFrame(design).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/query_and_path_design.csv'),index=False)
        print(f'PILOT {n}/38 {ID}; {time.perf_counter()-start:.1f}s',flush=True)
    pd.DataFrame(weights).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/lime_weights_and_origin.csv'),index=False)
    pd.DataFrame(cost).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/query_cost_and_walltime.csv'),index=False)
    pd.DataFrame(acceptance).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/PERTURBATION_ACCEPTANCE.csv'),index=False)
    pd.DataFrame(ident).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/RESPONSE_IDENTIFIABILITY.csv'),index=False)
    dump(_repo_path('outputs/06_EXTERNAL_TRANSFER/constrained_lime/EXECUTION_COMPLETE.json'),{'status':'COMPLETE_WITH_FAILURES' if any(r['status']!='COMPLETE' for r in cases) else 'COMPLETE','candidate_materials':38,'planned_cases':76,'planned_holdout_responses':3648,'actual_complete_cases':sum(r['status']=='COMPLETE' for r in cases),'seconds':time.perf_counter()-start,'frozen_model_sha256':sha(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/property_model.joblib')),'target':'fixed ET perturbation response','new_property_model_training':False,'challenge_executed':False})
if __name__=='__main__':main()
