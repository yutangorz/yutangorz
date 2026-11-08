"""Do not infer a metal or fill missing targets with zero from a single mesh."""

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
import math
def classify(separation, occupied_min, empty_max):
    if not all(math.isfinite(x) for x in [separation,occupied_min,empty_max]):
        raise ValueError('Nonfinite spectral evidence')
    if not 0<=occupied_min<=1.000001 or not 0<=empty_max<=1.000001:
        raise ValueError('Invalid occupations')
    # EIGENVAL/vasprun precision and the historical resolved-signal floor.
    resolved=separation>1e-5 and occupied_min>.999 and empty_max<.001
    return {'spectral_class':'resolved_semiconductor_on_sampled_mesh' if resolved else 'unresolved_requires_mesh_smearing_review',
            'gap_eV':separation if resolved else None,
            'metal_verified':False,
            'metal_candidate':bool(separation< -1e-5 or occupied_min<=.999 or empty_max>=.001),
            'negative_separation_is_not_a_negative_gap':True,
            'feature_calculability':'separate_geometry_composition_check',
            'model_domain':'requires_separate_assessment'}
