"""Finish only the secondary model checks from immutable run018 output caches."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/shap_baseline')))
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/external_response')))
from pathlib import Path
import hashlib,json,time
import joblib,numpy as np,pandas as pd
from shap_baseline.importance import explain_tree
from shared_features.extraction import FEATURES
from study_closure import dump,metrics
run=_repo_path('06_EXTERNAL_TRANSFER/workflows/external_response')
for file in ['FROZEN_INPUT_CODE.json','inputs/CACHE_REUSE_MANIFEST.json']:
    for r in json.loads((run/file).read_text(encoding='utf-8')):
        assert hashlib.sha256((run/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
out=_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/shap_baseline');detail=pd.read_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/shap_baseline/per_material_nine_attributions.csv'),dtype={'ID':str})
ids=pd.read_csv(_repo_path('06_EXTERNAL_TRANSFER/data/external_response/new_parent_manifest.csv'),dtype={'ID':str}).ID.tolist()
s=detail.set_index('ID').loc[ids];sx=s[FEATURES].to_numpy();primary=s[['SHAP_'+f for f in FEATURES]].to_numpy()
rows=[];t=time.perf_counter()
for name,tolerance in [('XGBoost',1e-5),('label_permuted_ET',1e-6)]:
    m=joblib.load(_repo_path('06_EXTERNAL_TRANSFER/data/external_response')/f'{name}.joblib');v,base,p,res=explain_tree(m,sx,atol=tolerance)
    for k,ID in enumerate(ids):
        for j,f in enumerate(FEATURES):rows.append(dict(ID=ID,model=name,feature=f,SHAP_eV=v[k,j],primary_SHAP_eV=primary[k,j],
            prediction_eV=float(p[k]),base_eV=base,residual_eV=float(res[k]),numerical_tolerance_eV=tolerance))
    print(name,'max residual',float(abs(res).max()),'tolerance',tolerance,flush=True)
controls=pd.DataFrame(rows);controls.to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/shap_baseline/other_model_and_label_permutation.csv'),index=False)
a=controls[controls.model=='label_permuted_ET'];change=float(np.sqrt(np.mean((a.SHAP_eV-a.primary_SHAP_eV)**2)));assert change>1e-5
d=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv'),dtype={'ID':str});f=pd.read_csv(_repo_path('03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv'),dtype={'ID':str});d=d.merge(f[['ID','outer_fold']],on='ID')
pd.DataFrame(np.corrcoef(d[d.outer_fold!=0][FEATURES].to_numpy(),rowvar=False),index=FEATURES,columns=FEATURES).to_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/shap_baseline/training_feature_correlations.csv'))
result=dict(n=len(detail),formula_groups=int(detail.formula_group.nunique()),max_additivity_residual_eV=float(abs(detail.additivity_residual_eV).max()),
    model_permuted_attribution_rms_change_eV=change,property_prediction=metrics(detail.band_gap_eV,detail.prediction_eV),
    seconds=time.perf_counter()-t,seconds_scope='secondary-resume only; primary computation recorded in run018 execution logs',
    reference='tree_path_dependent uses training tree counts; not true conditional/causal',bootstrap='resample heldout formula groups with fixed model; no model-fitting uncertainty')
dump(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/shap_baseline/ATTRIBUTION_VALIDATION.json'),result)
syn=pd.read_csv(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/synthetic/known_game_shapley.csv'))
dump(_repo_path('outputs/06_EXTERNAL_TRANSFER/external_response/CLOSURE_EXECUTION.json'),dict(status='execution_completed_pending_final_QA',synthetic=dict(exact_game_max_error=float(syn.abs_error.max()),coalition_n=512,synthetic_queries=9),attribution=result))
print('RESUMED COMPLETE',flush=True)
