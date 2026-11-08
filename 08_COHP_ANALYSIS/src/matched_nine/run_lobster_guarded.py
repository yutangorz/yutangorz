"""Fail-closed analysis runner. This file does not submit jobs."""

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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/matched_nine')))
import os,json,hashlib,subprocess,fcntl
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 p=_repo_path('08_COHP_ANALYSIS/workflows/matched_nine');t=json.loads((_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/TASK.json')).read_text())
 # Approval must be a separate, author-reviewed artifact. No inherited DFT quota.
 approval=Path(os.environ['COHP_AUTHOR_APPROVAL_FILE']);a=json.loads(approval.read_text())
 assert a.get('author_approved') is True and t['task_id'] in a['approved_task_ids']
 assert t['status']=='READY_FOR_AUTHOR_REVIEW','AUDIT_HOLD_NO_EXECUTION'
 assert a['TASK_SHA256'][t['task_id']]==digest(_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/TASK.json'))
 binary=Path(t['binary']);assert digest(binary)==t['binary_sha256']
 assert len(os.sched_getaffinity(0))>=t['threads'],'CPUSET_TOO_SMALL'
 work=Path(t['output_directory']);work.mkdir(parents=True,exist_ok=False)
 with (work/'analysis.lock').open('x') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,r in t['frozen_files'].items():
   src=Path(r['path']);assert digest(src)==r['sha256'],name
   (work/name).symlink_to(src)
  src=_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/lobsterin');assert digest(src)==t['lobsterin_sha256'];(work/'lobsterin').write_bytes(src.read_bytes())
  env=dict(os.environ,OMP_NUM_THREADS=str(t['threads']),OMP_PLACES='cores',OMP_PROC_BIND='close',MKL_NUM_THREADS='1')
  with (work/'lobster.stdout').open('xb') as out,(work/'lobster.stderr').open('xb') as err:
   code=subprocess.call([str(binary)],cwd=str(work),env=env,stdout=out,stderr=err)
  assert code==0,'LOBSTER_PROCESS_FAILED_NO_AUTO_RETRY'
  # Successful exit is not scientific acceptance; warnings require offline QA.
  (work/'POSTPROCESS_QA_REQUIRED').write_text('Check projection, overlap warnings, bands, electron reconstruction and edge linkage.\n')
if __name__=='__main__':main()
