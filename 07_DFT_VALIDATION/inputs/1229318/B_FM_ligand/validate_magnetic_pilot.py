"""Preserve production technical/spectral gates, add frozen seed and magnetic output evidence."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/inputs/1229318/B_FM_ligand')))
from pathlib import Path
import json,math,re,sys,xml.etree.ElementTree as ET
from validate_substitution import check as base_check
from soc_input_guard import guard

def initial_vectors(path):
    vectors=[];axes=[]
    for event,elem in ET.iterparse(path,events=('end',)):
        if elem.tag=='v' and elem.get('name')=='MAGMOM':vectors.append([float(v) for v in elem.text.split()])
        if elem.tag=='v' and elem.get('name')=='SAXIS':axes.append([float(v) for v in elem.text.split()])
        elem.clear()
    return vectors,axes

def magnetic_observables(text,nsites):
    output={}
    for axis in 'xyz':
        starts=list(re.finditer(r'magnetization \('+axis+r'\)',text))
        if not starts:raise ValueError('Missing final projected magnetization axis '+axis)
        chunk=text[starts[-1].end():]
        rows=[]
        for line in chunk.splitlines():
            bits=line.split()
            if bits and bits[0].isdigit():
                if int(bits[0])!=len(rows)+1:raise ValueError('Magnetization ion sequence mismatch')
                row=[float(v) for v in bits[1:]]
                if not row or not all(math.isfinite(v) for v in row):raise ValueError('Invalid projected moment')
                rows.append(row[-1])
                if len(rows)==nsites:break
        if len(rows)!=nsites:raise ValueError('Incomplete magnetization table '+axis)
        output[axis]=rows
    energy=re.findall(r'energy\(sigma->0\)\s*=\s*([-+\d.Ee]+)',text)
    if not energy:raise ValueError('Missing final sigma->0 energy')
    e=float(energy[-1]);assert math.isfinite(e)
    return dict(final_sigma0_energy_eV=e,site_projected_spin_muB=output,
                sum_projected_spin_muB={axis:sum(values) for axis,values in output.items()},
                projected_moments_not_total_orbital_magnetization=True,printed_moment_precision_muB=.001)

def check(folder,config):
    p=Path(folder);guard((p/'INCAR').read_text(),config['nsites'])
    expected=config['initial_MAGMOM_flat'];vectors,axes=initial_vectors(p/'vasprun.xml')
    assert vectors and all(len(v)==len(expected) and max(abs(x-y) for x,y in zip(v,expected))<1e-10 for v in vectors),'Actual initial MAGMOM mismatch'
    assert axes and all(len(v)==3 and max(abs(x-y) for x,y in zip(v,[0,0,1]))<1e-10 for v in axes),'Actual SAXIS mismatch'
    verdict=base_check(p,config)
    assert verdict['passed'] and verdict['technical_validity']=='passed'
    verdict.update(magnetic_observables((p/'OUTCAR').read_text(errors='replace'),config['nsites']))
    verdict.update(initial_MAGMOM_verified=True,finite_seed=config['pilot_seed'],pilot_scope='one of three fixed initial conditions; no magnetic ground-state or stability certification')
    return verdict

if __name__=='__main__':
    p=Path(sys.argv[1]);config=json.loads((p/'TASK.json').read_text());value=check(p,config)
    with (p/'validation.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
