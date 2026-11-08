
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/common')))
import importlib.util
import json
from pathlib import Path
import numpy as np
import joblib


class ProxyOracle:
    """Explicit nine-concept surrogate of saved ALIGNN outputs, not direct ALIGNN."""
    kind = 'nine_concept_proxy_of_saved_ALIGNN_predictions'

    def __init__(self, model, extractor):
        self.model = joblib.load(model) if isinstance(model, (str, Path)) else model
        self.extractor = extractor
        self.calls = 0

    def predict_concepts(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
            raise ValueError('Oracle expects finite n by 9 concepts')
        self.calls += len(x)
        return np.asarray(self.model.predict(x), dtype=float).reshape(-1)

    def predict_structures(self, structures):
        return self.predict_concepts(np.array([self.extractor.extract(s)[0] for s in structures]))


class ExternalCrystalOracle:
    """Plug-in adapter for a functioning ALIGNN environment.

    Module must define predict_structure_dicts(list_of_pymatgen_structure_dicts)
    and ORACLE_METADATA containing checkpoint_sha256 and model_name. No fallback
    to proxy, labels, or synthetic values is allowed if a plug-in fails.
    """
    kind = 'external_crystal_oracle'

    def __init__(self, module_path):
        spec = importlib.util.spec_from_file_location('crystal_teacher_plugin', module_path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.metadata = self.module.ORACLE_METADATA
        if not self.metadata.get('checkpoint_sha256'):
            raise ValueError('Missing teacher checkpoint hash')
        self.calls = 0

    def predict_structures(self, structures):
        self.calls += len(structures)
        y = np.asarray(self.module.predict_structure_dicts([s.as_dict() for s in structures]), dtype=float).reshape(-1)
        if len(y) != len(structures) or not np.isfinite(y).all():
            raise ValueError('Malformed direct crystal predictions')
        return y
