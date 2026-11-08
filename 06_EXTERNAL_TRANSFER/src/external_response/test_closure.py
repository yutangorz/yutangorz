
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
import numpy as np
import pytest
from crispx_closure.paths import block_subset, fit_closed, predict_closed
from shap_baseline.importance import aggregate, explain_tree
from sklearn.tree import DecisionTreeRegressor


def test_parent_manifest_metadata_cannot_shadow_canonical_path(tmp_path):
    import pandas as pd
    from study_closure import load_parents
    p=tmp_path/'parents.csv'
    pd.DataFrame({'ID':['1'],'path':['metadata'],'formula_group':['A']}).to_csv(p,index=False)
    data=pd.DataFrame({'ID':['1'],'path':['canonical'],'formula_group':['A']})
    out=load_parents(p,data)
    assert out.path.iloc[0]=='canonical'
    assert 'path_x' not in out and len(out)==1


def event(k='geometry_only'):
    return ({'kind':'strain_and_internal_displacement'} if k == 'geometry_only'
            else {'kind':'bounded_same_block_species_substitution','old':k,'new':'Y'})


def fixture():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(120,9))
    events = [event()] * 60 + [event('A')] * 30 + [event('B')] * 30
    return z, events


def test_actual_block_selection_and_minimum_replication():
    z, e = fixture()
    ii = block_subset(z, e, 48, 24)
    assert len(ii) == len(set(ii)) == 48
    assert sum(j < 60 for j in ii) >= 24
    assert sum(60 <= j < 90 for j in ii) >= 3
    assert sum(j >= 90 for j in ii) >= 3


def test_atomic_coordinates_do_not_change_query_design():
    z, e = fixture()
    a = block_subset(z, e)
    z[:,:7] *= 10000
    assert np.array_equal(a, block_subset(z, e))


def test_linear_response_recovery_and_origin():
    z, e = fixture()
    y = z[:,7:] @ np.array([2.,-1.]) + np.r_[np.zeros(60),np.ones(30),np.ones(30)*3]
    m = fit_closed(z, y, e, 1e-9)
    p, _, why, _ = predict_closed(m, z, e)
    assert np.max(abs(p-y)) < 1e-8
    origin = predict_closed(m, np.zeros((1,9)), [event()])[0][0]
    assert origin == 0
    assert set(why) == {'supported'}


def test_unknown_path_abstains():
    z, e = fixture()
    m = fit_closed(z, z[:,7], e)
    p, _, why, _ = predict_closed(m, z[:1], [event('NEW')])
    assert np.isnan(p[0]) and why == ['unknown_path']


def test_insufficient_and_nullspace_abstention():
    z = np.zeros((5,9)); z[:,7] = np.arange(5)
    m = fit_closed(z, np.arange(5.), [event()]*5)
    q = z[:1].copy(); q[0,8] = 1
    assert predict_closed(m,q,[event()])[2] == ['outside_identified_subspace']
    m2 = fit_closed(z[:2],np.arange(2.),[event()]*2)
    assert predict_closed(m2,z[:1],[event()])[2] == ['insufficient_path_information']


def test_sparse_chemical_path_mean_only_not_fabricated_slope():
    z=np.zeros((2,9));z[:,7]=[1.,2.]
    m=fit_closed(z,np.array([3.,5.]),[event('A')]*2)
    q=np.zeros((1,9));q[0,8]=100
    p,b,reason,_=predict_closed(m,q,[event('A')])
    assert p[0]==b[0]==4.
    assert reason==['joint_path_mean_only_no_geometry_slope']
    assert not m['A->Y']['separate_geometry_slopes_qualified']
    assert np.max(abs(m['A->Y']['coefficient']))==0


@pytest.mark.parametrize('budget,quota',[(0,0),(130,0),(48,49),(-1,0)])
def test_bad_budgets(budget,quota):
    z,e=fixture()
    with pytest.raises(ValueError):block_subset(z,e,budget,quota)


def test_invalid_importance_and_shape():
    with pytest.raises(ValueError):aggregate(np.ones((2,9)),['a'])
    with pytest.raises(ValueError):aggregate(np.ones((0,9)),[])
    z,e=fixture()
    with pytest.raises(ValueError):fit_closed(z,np.ones(119),e)


def test_shap_dummy_constant_and_known_stump():
    rng=np.random.default_rng(5);x=rng.normal(size=(100,9))
    tree=DecisionTreeRegressor(max_depth=1).fit(x,x[:,0])
    v,base,p,res=explain_tree(tree,x[:10])
    assert np.max(abs(v[:,1:]))==0
    assert np.allclose(v[:,0],p-base)
    assert np.max(abs(res))<1e-12
    tree.fit(x,np.ones(len(x))*2)
    v,base,p,res=explain_tree(tree,x[:10])
    assert np.max(abs(v))==0 and base==2


def test_cluster_bootstrap_reproducibility():
    v=np.arange(36.).reshape(4,9);g=['a','a','b','c']
    a=aggregate(v,g,repeats=20);b=aggregate(v,g,repeats=20)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert abs(a[1].sum()-1)<1e-12


def test_secondary_float32_model_has_explicit_small_tolerance():
    from xgboost import XGBRegressor
    rng=np.random.default_rng(41);x=rng.normal(size=(40,9))
    m=XGBRegressor(n_estimators=20,max_depth=2,n_jobs=2).fit(x,x[:,0]+x[:,1])
    _,_,_,r=explain_tree(m,x[:8],atol=1e-5)
    assert np.max(abs(r))<1e-5
    with pytest.raises(ValueError):explain_tree(m,x[:2],atol=-1)
