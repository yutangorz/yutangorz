
# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/material_selection')))
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,fcluster
from cluster_core import DiskDistanceBackend,normalize_id,select_local_two,candidate_partition_metrics


def test_disk_replay_exact_native_arithmetic(tmp_path):
    x=np.random.default_rng(23).normal(size=(35,15));x[:,-1]=0
    with pytest.warns(UserWarning):expected=linkage(squareform(pdist(x)),method='ward')
    disk=DiskDistanceBackend(tmp_path/'cache');d=disk.squareform(disk.pdist(x));actual=disk.linkage(d)
    assert np.array_equal(d,squareform(pdist(x)))
    assert np.array_equal(actual,expected)
    assert np.array_equal(disk.linkage(d),expected)
    with pytest.raises(ValueError):disk.pdist(x+1)


def test_profile_is_not_direct_feature_geometry():
    x=np.random.default_rng(9).normal(size=(12,4))
    with pytest.warns(UserWarning):profile=linkage(squareform(pdist(x)),method='ward')
    direct=linkage(x,method='ward')
    assert not np.allclose(profile[:,2],direct[:,2])


def test_id_suffix_and_invalid_ids():
    assert normalize_id('1229318.vasp')=='1229318'
    assert normalize_id(' 1229318 ')=='1229318'
    with pytest.raises(ValueError):normalize_id('demo')


def test_selection_cluster_relabel_and_row_order_invariance():
    x=np.array([[0.],[1.],[2.],[10.],[11.],[12.]])
    ids=np.array(['a','b','c','d','e','f']);labels=np.array([1,1,1,2,2,2])
    chosen=set(ids[select_local_two(x,labels,ids)])
    ii=np.array([5,1,3,0,2,4]);other=10-labels[ii]
    assert set(ids[ii][select_local_two(x[ii],other,ids[ii])])==chosen


def test_selection_small_cluster_and_duplicate_ID_guard():
    with pytest.raises(ValueError):select_local_two(np.eye(2),[1,2],['a','b'])
    with pytest.raises(ValueError):select_local_two(np.eye(2),[1,1],['a','a'])


def test_candidate_partition_ignores_numeric_label_names():
    exp=pd.DataFrame(dict(material_id=['a','b','c','d'],cluster_id=[1,1,2,2]))
    m=candidate_partition_metrics(['a','b','c','d'],[9,9,7,7],exp)
    assert m['same_pair_recovered']==2 and m['different_pair_separated']==4 and not m['numeric_labels_equal']


def test_physical_size_features_are_not_supercell_invariant():
    # A deliberate audit of dimensional meaning, not a claim to fix historical input.
    from pymatgen.core import Structure,Lattice
    s=Structure(Lattice.cubic(4),['Cs','Cl'],[[0,0,0],[.5,.5,.5]])
    t=s.copy();t.make_supercell([2,1,1])
    assert t.volume==pytest.approx(2*s.volume)
    assert t.composition.weight==pytest.approx(2*s.composition.weight)
    assert t.volume/len(t)==pytest.approx(s.volume/len(s))
