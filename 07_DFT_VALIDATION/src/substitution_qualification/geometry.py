"""Small reference implementation of periodic, baseline-relative local paths.
These are frozen-ion geometry paths, not independent interventions on all nine
descriptors and not symmetry-adapted phonon or rigid-octahedron modes.
"""

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
import numpy as np

def vector(s,i,j):
    d,image=s.lattice.get_distance_and_image(s[i].frac_coords,s[j].frac_coords)
    return s.lattice.get_cartesian_coords(s[j].frac_coords+image-s[i].frac_coords)

def angle(s,a,b,c):
    u,v=vector(s,b,a),vector(s,b,c)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))

def change_length(base,i,j,delta):
    if i==j:raise ValueError('Distinct sites required; image-only movement is not a periodic displacement')
    v=vector(base,i,j);d=np.linalg.norm(v)
    if d+delta<=0:raise ValueError('Nonpositive length')
    out=base.copy();out.translate_sites([i],-v/d*delta/2,frac_coords=False,to_unit_cell=True);out.translate_sites([j],v/d*delta/2,frac_coords=False,to_unit_cell=True)
    if abs(np.linalg.norm(vector(out,i,j))-(d+delta))>1e-7:raise ValueError('Nearest image changed; reduce amplitude')
    return out

def change_angle(base,a,b,c,delta):
    if len({a,b,c})!=3:raise ValueError('Three distinct sites required')
    u,v=vector(base,b,a),vector(base,b,c);theta=angle(base,a,b,c)
    if not 0<theta+delta<180:raise ValueError('Angle boundary requires a defined one-sided bending mode')
    axis=np.cross(u,v);norm=np.linalg.norm(axis)
    if norm<1e-8:raise ValueError('Collinear angle needs explicit physical bending mode')
    axis/=norm;t=np.radians(delta)
    rotated=v*np.cos(t)+np.cross(axis,v)*np.sin(t)+axis*np.dot(axis,v)*(1-np.cos(t))
    out=base.copy();out.translate_sites([c],rotated-v,frac_coords=False,to_unit_cell=True)
    if abs(angle(out,a,b,c)-(theta+delta))>1e-6:raise ValueError('Angle target not achieved')
    return out
