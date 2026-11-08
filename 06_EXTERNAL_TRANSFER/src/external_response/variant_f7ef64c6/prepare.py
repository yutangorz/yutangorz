
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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/external_response/variant_f7ef64c6')))
from pathlib import Path
import hashlib,json,shutil,sys,importlib.metadata as md
import pandas as pd

ROOT=_repo_path('.')
RUN=_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response')
OLD=_repo_path('05_CRISP_X/workflows/local_reconstruction')
PKG=_repo_path('02_ALIGNN_TRAINING/data/external_selection_reference')
LOB=_repo_path('06_EXTERNAL_TRANSFER/data/external_source')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    for n in ['inputs','frozen_code','logs','reports']:(RUN/n).mkdir(exist_ok=False)
    # Copy verified original code, without editing any scientific function.
    manifest={x['path'].replace('\\','/'):x['sha256'] for x in json.loads((_repo_path('05_CRISP_X/configs/local_reconstruction_source_identity_frozen_input_code.json')).read_text(encoding='utf-8'))}
    records=[]
    for p in (_repo_path('05_CRISP_X/src/local_reconstruction')).rglob('*.py'):
        if '__pycache__' in p.parts:continue
        rel=p.relative_to(OLD).as_posix()
        assert sha(p)==manifest[rel],rel
        dst=_repo_path('06_EXTERNAL_TRANSFER/src/external_response')/p.relative_to(_repo_path('05_CRISP_X/src/local_reconstruction'));dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        records.append({'source':str(p),'copy':str(dst.relative_to(RUN)),'sha256':sha(dst)})
    for n in ['property_model.joblib','atom_importance.csv','model_input_7176.csv','outer_fold_manifest.csv']:
        p=_repo_path('05_CRISP_X/data/local_reconstruction')/n;assert sha(p)==manifest['inputs/'+n]
        shutil.copyfile(p,_repo_path('06_EXTERNAL_TRANSFER/data/external_response')/n)
    assert sha(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/property_model.joblib'))=='1e3c7c3370b5fcc41b25db1eb20ff483c877c190d23d3c83b2035e3fd21784c6'
    cfg=json.loads((_repo_path('05_CRISP_X/configs/local_reconstruction.json')).read_text(encoding='utf-8'))
    assert cfg['seeds']==[307,419] and cfg['candidates']==[{'name':'block_b64_g40','budget':64,'geometry_quota':40}]
    cfg.update(external_pilot=True,parent_candidate_count=40,approved_parent_count=38,output_response='frozen ET structural perturbation response; not JARVIS label or ALIGNN',challenge_locked=True,no_tuning=True,bootstrap_parent_seed=20261729)
    # Parent-bootstrap seed is exactly original seed+700.
    cfg['bootstrap_parent_seed']=cfg['seed']+700
    dump(_repo_path('06_EXTERNAL_TRANSFER/configs/external_response.json'),cfg)
    t=pd.read_csv(_repo_path('02_ALIGNN_TRAINING/data/external_selection_reference/01_DATA/ALIGNN_DATA_SPLIT_MANIFEST.csv'));test=t[t.split=='test']
    pilot=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_PILOT.csv'))
    challenge=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_DATASET_CHALLENGE.csv'))
    feat=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_source/EXTERNAL_FEATURES_9D.csv'))
    pilot['included']=pilot.external_id.isin(test.external_id)
    pilot.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/ALL40_PILOT_ROLE.csv'),index=False)
    approved=pilot[pilot.included].copy()
    assert len(approved)==38 and len(challenge)==20
    assert not set(approved.external_id)&set(challenge.external_id)
    assert set(challenge.external_id)<=set(test.external_id)
    approved=approved[['external_id']].merge(test,on='external_id',validate='one_to_one').merge(feat,on='external_id',validate='one_to_one')
    (_repo_path('06_EXTERNAL_TRANSFER/data/external_response/cif')).mkdir()
    for r in approved.itertuples():
        p=PKG/r.cif_path;assert sha(p)==r.structure_sha256
        shutil.copyfile(p,_repo_path('06_EXTERNAL_TRANSFER/data/external_response/cif')/f'{r.external_id}.cif')
    approved['local_cif']=['inputs/cif/'+x+'.cif' for x in approved.external_id]
    approved.to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/PILOT38.csv'),index=False)
    challenge[['external_id']].to_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/CHALLENGE20_LOCKED_IDS.csv'),index=False)
    shutil.copyfile(_repo_path('05_CRISP_X/configs/local_reconstruction.json'),_repo_path('06_EXTERNAL_TRANSFER/data/external_response/ORIGINAL_CONFIRMATION_CONFIG.json'))
    dump(_repo_path('06_EXTERNAL_TRANSFER/configs/external_response_source_identity_frozen_scientific_source_identity.json'),records)
    dump(_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response/ENVIRONMENT.json'),{'python':sys.version,'executable':sys.executable,'packages':{k:md.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib','pymatgen','shap','lime','xgboost']}})
    dump(_repo_path('06_EXTERNAL_TRANSFER/configs/external_response_source_identity_frozen_input_code.json'),[{'path':str(p.relative_to(RUN)),'sha256':sha(p)} for p in sorted(RUN.rglob('*')) if p.is_file() and '__pycache__' not in p.parts])
    print('PREPARED 38 pilot; original algorithm/model/config SHA checked; challenge untouched',flush=True)
if __name__=='__main__':main()
