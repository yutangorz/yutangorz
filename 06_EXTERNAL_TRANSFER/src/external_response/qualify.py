
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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/external_response')))
from pathlib import Path
import sys,json,hashlib,warnings,time
import numpy as np,pandas as pd,joblib
from pymatgen.core import Structure
RUN=_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response')
sys.path.insert(0,str(_repo_path('06_EXTERNAL_TRANSFER/src/external_response')))
from shared_features.extraction import ConceptExtractor,FEATURES
from crispx.perturb import generate_candidates,structure_key
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    out=_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response/qualification');out.mkdir(exist_ok=False)
    for r in json.loads((_repo_path('06_EXTERNAL_TRANSFER/configs/external_response_source_identity_frozen_input_code.json')).read_text(encoding='utf-8')):assert sha(RUN/r['path'])==r['sha256'],r['path']
    cfg=json.loads((_repo_path('06_EXTERNAL_TRANSFER/configs/external_response.json')).read_text(encoding='utf-8'))
    model=joblib.load(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/property_model.joblib'))
    assert model.n_features_in_==9
    ex=ConceptExtractor(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv'))
    rows=[];start=time.perf_counter()
    for r in pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/PILOT38.csv')).itertuples():
        row={'external_id':r.external_id,'passed':False}
        try:
            with warnings.catch_warnings(record=True) as ws:
                s=Structure.from_file(RUN/r.local_cif)
            assert sha(RUN/r.local_cif)==r.structure_sha256
            x,q=ex.extract(s);reference=np.array([getattr(r,f) for f in FEATURES])
            feature_match=bool(np.allclose(x,reference,atol=1e-10)) # original confirmation rule, including its rtol
            assert feature_match,'frozen feature mapping mismatch'
            assert s.is_ordered and s.volume>0 and len(s)==r.n_sites_extracted and np.isfinite(x).all()
            assert all(q['missing_fraction_'+f]==0 for f in FEATURES[:7])
            prediction=float(model.predict(x[None,:])[0]);assert np.isfinite(prediction)
            # Bounded preflight only, original generator and first original seed; no explanation fitted here.
            ss,xx,events,rejected=generate_candidates(s,ex,cfg['seeds'][0],size=8,geometry_scale=.25)
            assert len(xx)==8 and len({structure_key(a) for a in ss})==8
            geom=sum(e['kind']=='strain_and_internal_displacement' for e in events)
            assert geom>0
            row.update(passed=True,structure_ordered=True,nsites=len(s),volume=s.volume,min_shell_distance=q['min_shell_distance'],max_feature_absolute_difference=float(abs(x-reference).max()),feature_match=feature_match,base_ET_prediction_eV=prediction,preflight_valid_candidates=8,preflight_geometry_candidates=geom,rejected_attempts=len(rejected),warnings=' | '.join(str(w.message) for w in ws))
        except Exception as e:row['error']=repr(e)
        rows.append(row)
    pd.DataFrame(rows).to_csv(_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response/qualification/PILOT38_QUALIFICATION.csv'),index=False)
    ok=all(r['passed'] for r in rows)
    dump(_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response/qualification/QUALIFICATION_QA.json'),{'passed':ok,'candidate_master_pilot':40,'preexcluded_overlap':2,'approved_candidates':38,'qualified':sum(r['passed'] for r in rows),'explanation_runs':0,'preflight_only':True,'seconds':time.perf_counter()-start,'cache_policy':'new stage-owned cache only; identity includes original model/table/code/config/CIF SHA; no old OQMD response cache substituted','original_frozen_config_and_source_passed':True,'challenge_materials_executed':0})
    print('QUALIFICATION',ok,sum(r['passed'] for r in rows),'/38',flush=True)
    if not ok:raise SystemExit('STOP formal pilot: qualification failed; no threshold/parser/algorithm changes')
if __name__=='__main__':main()
