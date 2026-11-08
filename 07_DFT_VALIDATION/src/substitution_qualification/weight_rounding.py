
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
import math

def normalized_electron_gate(nelect,raw_electrons,weights,weight_tokens):
    total=sum(weights)
    rounding_bound=sum(float(Decimal(10)**Decimal(t).as_tuple().exponent)/2 for t in weight_tokens)+1e-12
    finite=math.isfinite(total) and total>0 and all(math.isfinite(w) and w>=0 for w in weights)
    normalized=raw_electrons/total if finite else float('nan')
    passed=finite and abs(total-1)<=rounding_bound and abs(normalized-nelect)<1e-5
    return passed,{'raw_electrons':raw_electrons,'normalized_electrons':normalized,'printed_weight_sum':total,'weight_rounding_bound':rounding_bound,'normalized_electron_tolerance':1e-5}
