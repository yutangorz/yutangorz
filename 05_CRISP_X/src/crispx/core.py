"""Projected response fitting and query selection. No model-specific code."""

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
import numpy as np


def informative_subset(z, budget, ridge=0.1):
    """Greedy D-optimal selection by rank-one updates, using concepts only."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all() or budget < 1 or len(z) < budget:
        raise ValueError('Invalid candidate matrix or query budget')
    inv = np.eye(z.shape[1]) / ridge
    selected = []
    for _ in range(budget):
        scores = np.einsum('ij,jk,ik->i', z, inv, z)
        scores[selected] = -np.inf
        j = int(np.argmax(scores))
        selected.append(j)
        v = inv @ z[j]
        inv -= np.outer(v, v) / (1 + z[j] @ v)
    return np.asarray(selected)


def fit_response(z, delta_y, relative_singular_cutoff=.03, ridge=1e-5):
    """Zero-intercept local response in supported singular directions.

    Ridge coefficients are a representative, not unique physical contributions.
    Return the data-supported projector to flag individual non-identifiability.
    """
    z, y = np.asarray(z, dtype=float), np.asarray(delta_y, dtype=float)
    if z.ndim != 2 or len(z) != len(y) or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('Malformed response observations')
    weights = np.exp(-np.sum(z ** 2, axis=1) / (2 * z.shape[1]))
    a, b = z * np.sqrt(weights)[:, None], y * np.sqrt(weights)
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    threshold = max(1e-10, singular[0] * relative_singular_cutoff) if len(singular) else 1e-10
    keep = singular > threshold
    v = vt[keep].T
    coeff = v @ ((singular[keep] / (singular[keep] ** 2 + ridge)) * (u[:, keep].T @ b))
    projector = v @ v.T
    return {'coefficient': coeff, 'projector': projector, 'support': np.diag(projector),
            'individual_identifiable': np.diag(projector) > .95,
            'rank': int(keep.sum()), 'singular_values': singular,
            'weights': weights}


def bootstrap_coefficients(z, y, seed, repeats=100):
    rng = np.random.default_rng(seed)
    coeffs = []
    for _ in range(repeats):
        idx = rng.integers(0, len(z), len(z))
        coeffs.append(fit_response(z[idx], y[idx])['coefficient'])
    return np.quantile(coeffs, [.025, .975], axis=0)
