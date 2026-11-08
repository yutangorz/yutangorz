"""Explicit TreeSHAP attribution, not a newly invented CRISP-X estimator.

Primary convention: tree_path_dependent, raw regression prediction. Training
tree counts define the reference; this is not the true conditional distribution,
causal attribution, or a guaranteed physically realizable coalition game.
"""

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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/shap_baseline')))
import numpy as np
import shap


def explain_tree(model, x, background=None, atol=1e-6):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != model.n_features_in_ or not np.isfinite(x).all():
        raise ValueError('Malformed feature matrix')
    mode = 'tree_path_dependent' if background is None else 'interventional'
    explainer = shap.TreeExplainer(model, data=background,
        feature_perturbation=mode, model_output='raw')
    values = np.asarray(explainer.shap_values(x, check_additivity=True))
    base = float(np.ravel(explainer.expected_value)[0])
    pred = np.asarray(model.predict(x))
    residual = base + values.sum(axis=1) - pred
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError('Malformed SHAP output')
    if not np.isfinite(atol) or atol<=0:
        raise ValueError('Positive finite numerical tolerance required')
    if np.max(abs(residual), initial=0.) > (atol if background is None else max(atol,1e-3)):
        raise ValueError('SHAP additivity outside declared numerical tolerance')
    return values, base, pred, residual


def aggregate(values, groups, seed=20261019, repeats=2000):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    if values.ndim != 2 or len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Bad importance matrix/groups')
    labels, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(labels), values.shape[1]))
    np.add.at(sums, inverse, abs(values))
    count = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((repeats, values.shape[1]))
    for b in range(repeats):
        ii = rng.integers(0, len(labels), len(labels))
        bootstrap[b] = sums[ii].sum(axis=0) / count[ii].sum()
    point = abs(values).mean(axis=0)
    share = point / point.sum() if point.sum() else np.zeros_like(point)
    ranks = np.argsort(np.argsort(-bootstrap, axis=1), axis=1) + 1
    return point, share, bootstrap, ranks
