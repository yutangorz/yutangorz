"""CRISP-X closure: block-design queries and qualified local responses.

No independent chemical attribution is inferred from the path intercepts.
The nine-importance module explicitly delegates to TreeSHAP.
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx_closure')))
import numpy as np
from audited_paths import validate, fit_audited
from crispx.path_model import path_key


def block_subset(z, events, budget=64, geometry_quota=32, ridge=.001):
    z, _ = validate(z, None, events)
    if not 1 <= budget <= len(z) or not 0 <= geometry_quota <= budget:
        raise ValueError('Invalid budget/quota')
    keys = np.array([path_key(e) for e in events])
    groups = {k: np.flatnonzero(keys == k) for k in sorted(set(keys))}
    # This H is the actual response-estimator design, not global atomic Z.
    h = {k: (z[ii, 7:] if k == 'geometry_only' else
              np.ones((len(ii),1)) if len(ii)<3 else
              np.column_stack([np.ones(len(ii)), z[ii, 7:]])) for k, ii in groups.items()}
    selected = []
    local = {k: [] for k in groups}

    def take(k):
        ii, a, used = groups[k], h[k], local[k]
        if len(used) == len(ii):
            return False
        penalty = np.eye(a.shape[1]) * ridge
        if k != 'geometry_only':
            penalty[0, 0] = 1e-8  # numerical design prior, intercept not penalized in fit
        information = penalty + a[used].T @ a[used]
        inv = np.linalg.inv(information)
        score = np.einsum('ij,jk,ik->i', a, inv, a)
        score[used] = -np.inf
        j = int(np.argmax(score))
        used.append(j)
        selected.append(int(ii[j]))
        return True

    if 'geometry_only' in groups:
        for _ in range(min(geometry_quota, len(groups['geometry_only']))):
            take('geometry_only')
    chemical = sorted((k for k in groups if k != 'geometry_only'),
                      key=lambda k: (-len(groups[k]), k))
    # First cover each observed chemical path. Sparse paths get a mean-only
    # estimator, NOT an underdetermined geometry slope. Unknown keys still abstain.
    for k in chemical:
        if len(selected)<budget:take(k)
    # Allocate the remaining budget by balanced replication; D-optimal within block.
    active = [k for k in groups if local[k]]
    while len(selected) < budget:
        available = [k for k in active if len(local[k]) < len(groups[k])]
        if not available:
            break
        k = min(available, key=lambda k: (len(local[k]), k))
        take(k)
    # Unused slots can arise when all qualified blocks are exhausted.
    return np.asarray(selected, dtype=int)


def fit_closed(z, y, events, ridge=.001):
    z, y = validate(z, y, events)
    model = fit_audited(z, y, events, ridge=ridge)
    for k, m in model.items():
        ii = [j for j, e in enumerate(events) if path_key(e) == k]
        a = z[ii, 7:] - m['center']
        _, s, vt = np.linalg.svd(a, full_matrices=False)
        keep = s > max(1e-10, (s[0] if len(s) else 0) * 1e-6)
        v = vt[keep].T
        m['projector'] = v @ v.T
        m['numerical_geometry_rank'] = int(keep.sum())
        m['condition_number'] = float(s[0] / s[-1]) if len(s) == 2 and s[-1] > 1e-12 else None
        m['response_qualified'] = len(ii) >= 3 and bool(keep.any())
        m['prediction_mode'] = 'affine' if m['response_qualified'] else 'joint_path_mean_only'
        if k != 'geometry_only' and not m['response_qualified']:
            m['coefficient'] = np.zeros(2)
            m['joint_jump_at_unperturbed_geometry'] = m['mean']
        m['separate_geometry_slopes_qualified'] = (len(ii) >= 6 and keep.sum() == 2
            and m['condition_number'] is not None and m['condition_number'] < 1000)
    return model


def predict_closed(model, z, events):
    z, _ = validate(z, None, events)
    p = np.full(len(z), np.nan)
    baseline = np.full(len(z), np.nan)
    reasons, outside = [], []
    for j, e in enumerate(events):
        k = path_key(e)
        m = model.get(k)
        reason = 'supported'
        box = False
        if m is None:
            reason = 'unknown_path'
        elif not m['response_qualified']:
            if k == 'geometry_only':
                reason = 'insufficient_path_information'
            else:
                p[j] = m['mean']
                baseline[j] = m['train_path_mean']
                reason = 'joint_path_mean_only_no_geometry_slope'
        else:
            x = z[j, 7:] - m['center']
            residual = np.linalg.norm(x - x @ m['projector'])
            if residual > 1e-7 * (1 + np.linalg.norm(x)):
                reason = 'outside_identified_subspace'
            else:
                p[j] = m['mean'] + x @ m['coefficient']
                baseline[j] = 0. if k == 'geometry_only' else m['train_path_mean']
                box = bool(np.any(z[j, 7:] < m['geometry_bounds_min'] - 1e-12)
                           or np.any(z[j, 7:] > m['geometry_bounds_max'] + 1e-12))
        reasons.append(reason)
        outside.append(box)
    return p, baseline, reasons, np.asarray(outside)
