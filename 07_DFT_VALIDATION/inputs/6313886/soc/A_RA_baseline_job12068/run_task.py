
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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/inputs/6313886/soc/A_RA_baseline_job12068')))
from pathlib import Path
from datetime import datetime
import hashlib,json,os,socket,subprocess,sys
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
  c=json.loads(Path('TASK.json').read_text());node=c['assigned_node'];event('start',task=c['id'])
  assert node in ['node01','node02','node03']
  assert socket.gethostname().split('.')[0]==node and os.environ.get('SLURM_NTASKS')=='48'
  hosts=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],universal_newlines=True).strip();assert hosts==node
  subprocess.run(['sha256sum','-c','INPUT_SHA256SUMS'],check=True)
  assert sha(c['binary'])==c['binary_sha256']
  assert not any(Path(n).exists() for n in ['vasp.stdout','vasp.stderr','vasprun.xml','OUTCAR','validation.json'])
  if c.get('requires_technical_success_of'):
   batch=Path.cwd().parents[1];ref=(batch/c['requires_technical_success_of']).resolve();ref.relative_to(batch.resolve())
   receipt=json.loads((ref/'SUBMISSION.json').read_text());assert receipt['task']==c['requires_submitted_task_id']
   prior=[json.loads(x) for x in (ref/'events.jsonl').read_text().splitlines() if x.strip()]
   assert all(str(x['job_id'])==str(receipt['job_id']) for x in prior)
   assert any(x['event']=='validated' for x in prior) and any(x['event']=='job_script_exit' and x['exit_code']==0 for x in prior)
   value=json.loads((ref/'validation.json').read_text());assert value['passed'] and value['technical_validity']=='passed'
  from paw_record_guard import reader_layout
  reader_layout(Path('POTCAR').read_bytes(),Path('POSCAR').read_text().splitlines()[5].split())
  event('vasp_begin',mode=c['mode'],binary=c['binary'])
  with open('vasp.stdout','x') as out,open('vasp.stderr','x') as err:
   run=subprocess.run(['mpirun','-np','48','-hosts',node,c['binary']],stdout=out,stderr=err)
  event('vasp_process_exit',exit_code=run.returncode)
  if run.returncode:raise RuntimeError('VASP failed '+str(run.returncode))
  subprocess.run([sys.executable,'validate_substitution.py','.'],check=True)
  event('validated');rc=0
 except BaseException as exc:event('failure',error=str(exc));raise
 finally:event('job_script_exit',exit_code=rc)
