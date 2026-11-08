
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/response_matrix')))
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json,re,hashlib,base64,zlib
P=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_a09_length06_diag_run103_20261030/OUTPUT')
def parse(name):
 blocks={};keys={};part=None;k=None;labels=[];data=[]
 def flush():
  if data:
   ar=np.array(data);assert ar.shape[0]==192 and np.isfinite(ar).all();blocks[(k,part)]=ar;keys[(k,part)]=list(labels)
 with (P/name).open() as f:
  for line in f:
   if 'matrix for ' in line:
    flush();data=[];labels=[];k=int(re.search(r'kpoint\s+(\d+)',line)[1]);part=None
   elif 'Real parts' in line:part='real'
   elif 'Imag parts' in line:part='imag'
   elif line.strip() and 'basisfunction' not in line:
    s=line.split()
    if re.match(r'^[A-Z][a-z]?\d+_',s[0]):labels.append(s[0]);data.append([float(x) for x in s[1:]])
  flush()
 assert len(blocks)==128,(name,len(blocks),list(blocks)[:4]);out={}
 for k in range(1,65):
  assert keys[k,'real']==keys[k,'imag'];out[k]=blocks[k,'real']+1j*blocks[k,'imag']
 return out,keys[1,'real']
names=['coefficientMatrices.lobster','coefficientMatricesLSO1.lobster','coefficientMatricesLSO2.lobster','overlapMatrices.lobster']
mat={};hashes={}
for n in names:
 mat[n],labels=parse(n);hashes[n]=hashlib.sha256((P/n).read_bytes()).hexdigest();print('MATRIX_PARSED '+n,flush=True)
C=mat[names[0]];C1=mat[names[1]];C2=mat[names[2]];Ss=mat[names[3]];result=[];sub=[]
for k in range(1,65):
 s=Ss[k];herm=float(np.max(abs(s-s.conj().T)));g=C[k].conj().T@s@C[k];g1=C1[k].conj().T@s@C1[k];g2=C2[k].conj().T@C2[k];I=np.eye(192)
 result.append(dict(k=k,S_hermitian_residual=herm,raw_Snorm_max=float(abs(g-I).max()),LSO1_Snorm_max=float(abs(g1-I).max()),LSO2_Euclidean_max=float(abs(g2-I).max()),LSO2_occupied144_max=float(abs((g2-I)[:144,:144]).max()),LSO2_row144_max=float(abs((g2-I)[143]).max()),LSO2_row145_max=float(abs((g2-I)[144]).max()),LSO2_row146_max=float(abs((g2-I)[145]).max())))
 if k!=1:continue
 eig,U=np.linalg.eigh((s+s.conj().T)/2);sqrtS=(U*np.sqrt(np.maximum(eig,0)))@U.conj().T
 transformed={'raw_Shalf_C':sqrtS@C[k],'LSO1_Shalf_C':sqrtS@C1[k],'LSO2_C':C2[k]}
 groups={'Rh13_4d':[i for i,x in enumerate(labels) if x.startswith('Rh13_4d')],'Cl18_3p':[i for i,x in enumerate(labels) if x.startswith('Cl18_3p')],'Cl19_3p':[i for i,x in enumerate(labels) if x.startswith('Cl19_3p')]}
 for bands in [[144],[145],[146],[145,146],[139,140,141,142,143,144]]:
  ix=np.array(bands)-1
  for stage,c in transformed.items():
   b=c[:,ix];gram=b.conj().T@b;rot=np.linalg.qr(np.arange(1,len(ix)**2+1).reshape(len(ix),len(ix))+np.eye(len(ix))*.231)[0];br=b@rot
   for group,inds in groups.items():
    w=float(np.sum(abs(b[inds])**2));wr=float(np.sum(abs(br[inds])**2));sub.append(dict(k=1,bands=bands,stage=stage,group=group,trace_weight=w,unitary_rotation_trace_residual=abs(w-wr),subspace_gram_max=float(abs(gram-np.eye(len(ix))).max()),subspace_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),qualification='coordinate-basis trace; not signed pair COHP; stage comparison not independent radial-basis accuracy'))
  for j in ix:assert j<192
result=dict(hash=hashes,labels=labels,band_count=192,k_count=64,spin=1,S_eigenvalue_Gamma_min=float(eig.min()),all_k_gram=result,subspaces=sub,projection_binary_header=(_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_a09_length06_diag_run103_20261030/OUTPUT/projectionData.lobster')).open('rb').read(64).hex(),projection_binary_size=(_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_a09_length06_diag_run103_20261030/OUTPUT/projectionData.lobster')).stat().st_size)
b=base64.b64encode(zlib.compress(json.dumps(result).encode())).decode();print('A14_BEGIN')
for i in range(0,len(b),64):print(b[i:i+64])
print('A14_END')
