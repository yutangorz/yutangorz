
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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/xgboost_ablation')))
import numpy as np
import pytest
from ablation_core import correlation_groups,make_variants,paired_errors,clustered_interval


def test_training_correlation_groups_and_sign():
    rng=np.random.default_rng(5);a=rng.normal(size=200);b=rng.normal(size=200)
    x=np.column_stack([a,-a+.001*b,b])
    groups,c=correlation_groups(x)
    assert groups==[(0,1)] and c[0,1]<-.99


def test_group_members_are_fixed_by_training_only():
    rng=np.random.default_rng(7);train=rng.normal(size=(200,9));train[:,1]=train[:,0]
    groups,_=correlation_groups(train)
    test=rng.normal(size=(20,9));test[:,1]=-test[:,0]
    assert (0,1) in groups
    variants=make_variants(9,groups)
    assert () in variants and all((j,) in variants for j in range(9))
    assert (7,8) in variants and tuple(range(7)) in variants
    assert 'training_correlation_component' in variants[(0,1)]


def test_duplicate_semantic_and_correlated_groups_not_fit_twice():
    v=make_variants(9,[(7,8)])
    assert len(v[(7,8)])==2


def test_signed_effects_allow_negative_ablation():
    d=paired_errors([1,2],[0,2],[1,3])
    assert np.array_equal(d,[-1,1])


def test_formula_bootstrap_reproducibility_and_constant_effect():
    a=clustered_interval(np.ones(4),['a','a','b','c'],repeats=30)
    assert np.all(a==1)
    assert np.array_equal(a,clustered_interval(np.ones(4),['a','a','b','c'],repeats=30))


@pytest.mark.parametrize('threshold',[0,1,-.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):correlation_groups(np.eye(4),threshold)


def test_invalid_effect_shape():
    with pytest.raises(ValueError):paired_errors([1],[1,2],[1])
