
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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/inputs/8366012/L_minus_2h')))
from pathlib import Path
from datetime import datetime
import json,hashlib,os,socket,subprocess,sys

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def event(name,**more):
 r=dict(time=datetime.now().astimezone().isoformat(),event=name,job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname(),**more)
 with open('events.jsonl','a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(r),flush=True)
if __name__=='__main__':
 rc=1
 try:
  c=json.loads(Path('TASK.json').read_text());event('start',task=c['id'])
  if socket.gethostname().split('.')[0]!='node01' or os.environ.get('SLURM_NTASKS')!='48':raise ValueError('node01/48 only')
  hosts=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],universal_newlines=True).strip()
  if hosts!='node01':raise ValueError('Unexpected allocation')
  subprocess.run(['sha256sum','-c','INPUT_SHA256SUMS'],check=True)
  if sha(c['binary'])!=c['binary_sha256']:raise ValueError('Binary hash changed')
  if any(Path(n).exists() for n in ['vasp.stdout','vasp.stderr','vasprun.xml','OUTCAR','validation.json']):raise ValueError('Refusing to overwrite previous run')
  event('vasp_begin',mode=c['mode'],binary=c['binary'])
  with open('vasp.stdout','x') as out,open('vasp.stderr','x') as err:
   r=subprocess.run(['mpirun','-np','48','-hosts','node01',c['binary']],stdout=out,stderr=err)
  event('vasp_process_exit',exit_code=r.returncode)
  if r.returncode:raise RuntimeError('VASP failed '+str(r.returncode))
  subprocess.run([sys.executable,'validate_production.py','.'],check=True)
  event('validated');rc=0
 except BaseException as exc:event('failure',error=str(exc));raise
 finally:event('job_script_exit',exit_code=rc)
