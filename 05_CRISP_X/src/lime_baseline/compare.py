
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/lime_baseline')))
import time,numpy as np
from lime.lime_tabular import LimeTabularExplainer
from shared_features.extraction import FEATURES
def explain(model,training,x0,trial,seed,samples):
    start=time.perf_counter()
    ex=LimeTabularExplainer(training,feature_names=FEATURES,mode='regression',discretize_continuous=False,sample_around_instance=True,feature_selection='none',kernel_width=.75*np.sqrt(9),random_state=seed)
    result=ex.explain_instance(x0,model.predict,num_features=9,num_samples=samples)
    coeff=np.zeros(9)
    for j,v in result.as_map()[1]:coeff[j]=v
    # Centered differences cancel the fitted intercept; report anchor error separately.
    delta=(trial-x0)@(coeff/ex.scaler.scale_)
    return dict(delta=delta,coefficient=coeff,weighted_r2=float(result.score),intercept=float(result.intercept[1]),anchor_prediction=float(np.ravel(result.local_pred)[0]),seconds=time.perf_counter()-start)
