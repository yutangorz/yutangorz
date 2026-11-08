
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/original_spectra')))
from pathlib import Path
import numpy as np,re,hashlib,csv,json
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geom(path):
 s=Path(path).read_text().splitlines();scale=float(s[1]);l=np.array([list(map(float,t.split())) for t in s[2:5]])*scale;elements=s[5].split();counts=list(map(int,s[6].split()));n=sum(counts);offset=8 if not s[7].lower().startswith('s') else 9;direct=s[offset-1].lower().startswith('d');a=np.array([list(map(float,t.split()[:3])) for t in s[offset:offset+n]]);a=a if direct else a@np.linalg.inv(l)
 return l,a,[el for el,c in zip(elements,counts) for _ in range(c)]
def read_charge(path):
 with Path(path).open() as f:
  header=[next(f) for _ in range(8)];n=sum(map(int,header[6].split()));selective=header[7].lower().startswith('s');header += [next(f) for _ in range(n+(1 if selective else 0))]
  line=next(f)
  while not line.strip():line=next(f)
  dims=tuple(map(int,line.split()));assert len(dims)==3
  parts=[];size=0;N=int(np.prod(dims))
  while size<N:
   a=np.fromstring(next(f),sep=' ');parts.append(a);size+=len(a)
  assert size==N
  grid=np.concatenate(parts).reshape(dims,order='F')
  # CHGCAR augmentation occupancies are intentionally not interpreted as a density grid.
 l=np.array([list(map(float,t.split())) for t in header[2:5]])*float(header[1]);vol=abs(np.linalg.det(l));idx=8+(1 if selective else 0);coords=np.array([list(map(float,t.split()[:3])) for t in header[idx:idx+n]])
 return dict(header=header,dims=dims,raw=grid,rho=grid/vol,volume=vol,lattice=l,coords=coords,integral=float(grid.mean()),comment=header[0].strip())
def read_cohp(path):
 with Path(path).open() as f:
  first=next(f);h=next(f).split();nc=int(h[0]);n=int(h[2]);labels=[next(f).strip() for _ in range(nc)];a=np.array([np.fromstring(x,sep=' ') for x in f if x.strip()])
 assert a.shape==(n,1+2*nc) and np.isfinite(a).all() and np.all(np.diff(a[:,0])>0)
 return dict(data=a,labels=labels,ef=float(h[5]),n=n,curves=nc)
def pair_cols(c):return [i for i,s in enumerate(c['labels']) if '->' in s and '[' not in s]
def orbital_cols(c,idx,rh='4d',cl='3p'):
 prefix=c['labels'][idx].split(':')[0]+':'
 return [i for i,s in enumerate(c['labels']) if s.startswith(prefix) and '[' in s and re.search(r'Rh\d+\['+rh,s) and re.search(r'Cl\d+\['+cl,s)]
def write_grid(path,header,grid):
 with Path(path).open('w') as f:
  f.writelines(header);f.write('\n'+' '.join(map(str,grid.shape))+'\n');a=grid.ravel(order='F')
  for i in range(0,a.size,5):f.write(' '.join(format(x,'.16e') for x in a[i:i+5])+'\n')
