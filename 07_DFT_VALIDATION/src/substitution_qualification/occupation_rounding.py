"""Bound EIGENVAL occupation printing error; never silently drop electron checks."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
from decimal import Decimal
from weight_rounding import normalized_electron_gate as original_gate
def half_unit(token):
    x=Decimal(token)
    if not x.is_finite():raise ValueError('Nonfinite occupation')
    # Exact printed integers 0 and 1 are not uncertain by half an electron.
    if x in [0,1] and x.as_tuple().exponent>=0:return 0.0
    return float(Decimal(10)**x.as_tuple().exponent)/2
def normalized_electron_gate(nelect,raw_electrons,weights,weight_tokens,occupation_bound=0.0):
    passed,info=original_gate(nelect,raw_electrons,weights,weight_tokens)
    total=info['printed_weight_sum']
    bound=occupation_bound/total if total>0 else 0.
    tol=1e-5+bound
    passed=total>0 and abs(total-1)<=info['weight_rounding_bound'] and abs(info['normalized_electrons']-nelect)<tol
    info.update(occupation_printing_bound=bound,normalized_electron_tolerance=tol,policy='original1e-5 plus explicit weighted occupation half-last-digit bound; physical EDIFF unchanged')
    return passed,info
