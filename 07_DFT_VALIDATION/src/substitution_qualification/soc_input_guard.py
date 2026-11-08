"""Strict writer/guard for the approved zero-moment two-host SOC protocol."""

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
import re,math
def flat_zero_magmom(text,nsites):
 line='MAGMOM = '+' '.join(['0.0']*(3*nsites))
 hits=re.findall(r'(?im)^\s*MAGMOM\s*=.*$',text)
 if len(hits)!=1:raise ValueError('Exactly one MAGMOM required')
 return re.sub(r'(?im)^\s*MAGMOM\s*=.*$',line,text)
def guard(text,nsites):
 hits=re.findall(r'(?im)^\s*MAGMOM\s*=\s*([^\n#;]+)',text)
 if len(hits)!=1:raise ValueError('Exactly one MAGMOM required')
 tokens=hits[0].split()
 if len(tokens)!=3*nsites:raise ValueError('SOC MAGMOM requires explicit3N scalars')
 try:values=[float(x) for x in tokens]
 except ValueError:raise ValueError('Compressed/nested MAGMOM rejected by explicit-scalar policy')
 if not all(math.isfinite(x) and x==0 for x in values):raise ValueError('Approved protocol requires finite zero initial moments')
 return {'flat_components':len(values),'nsites':nsites,'all_zero':True}
