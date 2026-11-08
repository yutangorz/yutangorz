
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/pair_windows')))
from pathlib import Path
import numpy as np,json,hashlib,re,base64,zlib
R=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_A10_run105_20261030')
result=[]
for t in json.loads((_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_A10_run105_20261030/BATCH.json')).read_text()):
 d=R/t['state'];o=d/'OUTPUT';q=json.loads((d/'TECHNICAL_SCIENTIFIC_QC.json').read_text());p=o/'COHPCAR.lobster';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==q['output_SHA'][p.name]
 with p.open() as f:
  next(f);hd=next(f).split();nc,n=int(hd[0]),int(hd[2]);labels=[next(f).strip() for _ in range(nc)];a=np.loadtxt(f)
 assert a.shape==(n,1+2*nc)
 pairs=[]
 for i,label in enumerate(labels):
  if '->' not in label or '[' in label:continue
  js=[j for j,s in enumerate(labels) if s.startswith(label.split(':')[0]+':') and re.search(r'Rh\d+\[4d',s) and re.search(r'Cl\d+\[3p',s)]
  y=a[:,1+2*i];dy=a[:,[1+2*j for j in js]].sum(axis=1);E=a[:,0];m=E<=0
  pairs.append(dict(label=label,orbital_channels=len(js),ICOHP=float(np.interp(0,E,a[:,2+2*i])),Rh4d_Cl3p_ICOHP=float(np.interp(0,E,a[:,[2+2*j for j in js]].sum(axis=1))),occupied_COHP_integral=float(np.trapz(y[m],E[m])),occupied_d_p_integral=float(np.trapz(dy[m],E[m])),window_samples=[dict(E=float(e),COHP=float(v),Rh4d_Cl3p=float(w)) for e,v,w in zip(E[(E>=-0.6)&(E<=1)],y[(E>=-0.6)&(E<=1)],dy[(E>=-0.6)&(E<=1)])]))
 result.append(dict(state=t['state'],cohp_SHA=h,own_EF=float(hd[5]),pairs=pairs,internal_structure=(o/'POSCAR.lobster.vasp').read_text()))
 print('A13_PROGRESS '+t['state'],flush=True)
diag=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_a09_length06_diag_run103_20261030')
extra=dict(diagnostic_directory=str(diag),files=[dict(path=str(p.relative_to(diag)),size=p.stat().st_size) for p in diag.rglob('*') if p.is_file() and (p.suffix not in ['.xml'] and p.name not in ['WAVECAR','CHGCAR'])],qc_files={str(p.relative_to(diag)):p.read_text() for p in diag.glob('*.json') if p.stat().st_size<20000})
b=base64.b64encode(zlib.compress(json.dumps(dict(results=result,diagnostic=extra)).encode())).decode();print('A13_BEGIN')
for i in range(0,len(b),64):print(b[i:i+64])
print('A13_END')
