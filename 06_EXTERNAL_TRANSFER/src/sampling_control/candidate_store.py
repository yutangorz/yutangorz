"""Read the preflight-passed candidate bytes; contains NO sampler/RNG calls."""

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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/sampling_control')))
from pathlib import Path
import csv,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FrozenCandidateStore:
 def __init__(self,manifest,expected_sha,expected_IDs):
  self.path=Path(manifest).resolve();self.expected_sha=expected_sha
  if sha(self.path)!=expected_sha:raise RuntimeError('Candidate manifest SHA mismatch')
  with self.path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  self.rows={r['ID']:r for r in rows}
  if len(rows)!=38 or len(self.rows)!=38 or set(self.rows)!=set(expected_IDs):raise RuntimeError('Candidate parent identity mismatch')
  for r in rows:
   if r['dtype']!='<f8' or r['shape']!='(512, 9)' or r['index_definition']!='original rows0..511; no reorder; anchor0':raise RuntimeError('Candidate index/dtype/shape declaration changed')
   p=Path(r['path']).resolve()
   if p.name!=r['ID']+'.npy' or sha(p)!=r['sha256']:raise RuntimeError('Candidate path/SHA mismatch')
 def load(self,ID,x0):
  import numpy as np
  if sha(self.path)!=self.expected_sha:raise RuntimeError('Candidate manifest changed')
  row=self.rows[ID];p=Path(row['path']).resolve()
  if sha(p)!=row['sha256']:raise RuntimeError('Candidate changed before read')
  a=np.load(p,allow_pickle=False)
  if sha(p)!=row['sha256']:raise RuntimeError('Candidate changed during read')
  if a.shape!=(512,9) or a.dtype.str!='<f8' or not a.flags.c_contiguous or not np.isfinite(a).all():raise RuntimeError('Candidate numerical/dtype/shape mismatch')
  if not np.array_equal(a[0],x0) or a[0].tobytes()!=x0.tobytes():raise RuntimeError('Candidate anchor changed')
  return a
