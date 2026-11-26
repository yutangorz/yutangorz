"""Materialize minimum stage inputs; never import or execute scientific scripts."""
from pathlib import Path
import sys,json,csv,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from repository_paths import repo_path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):
 if dst.exists():
  if sha(src)!=sha(dst):raise RuntimeError('Existing input differs: '+str(dst))
  return
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def prepare(stage):
 specs={'et':('04_ET_XGBOOST_INTERPRETATION','grouped_cv','grouped_cv.json'),
 'xgboost':('04_ET_XGBOOST_INTERPRETATION','xgboost_ablation','xgboost_ablation.json'),
 'crispx':('05_CRISP_X','local_reconstruction','local_reconstruction.json'),
 'lime':('04_ET_XGBOOST_INTERPRETATION','native_lime','native_lime.json'),
 'external':('06_EXTERNAL_TRANSFER','external_response','external_response.json')}
 module,flow,cfg=specs[stage];run=repo_path(module+'/workflows/'+flow)
 if run.exists():raise RuntimeError('Choose a new JCTC_WORKSPACE_ROOT; existing workspace is immutable')
 run.mkdir(parents=True,exist_ok=False)
 copy(ROOT/module/'configs'/cfg,run/'config.json')
 # Preserve source code in the new integrity receipt, separate from historical receipts.
 for p in (ROOT/module/'src').rglob('*.py'):copy(p,run/'code'/p.relative_to(ROOT/module/'src'))
 common={'model_input_7176.csv':ROOT/'03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv','outer_fold_manifest.csv':ROOT/'03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv','atom_importance.csv':ROOT/'03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv'}
 for name,src in common.items():copy(src,run/'inputs'/name)
 for name in ['property_model.joblib','XGBoost.joblib','label_permuted_ET.joblib']:copy(ROOT/'05_CRISP_X/models'/name,run/'inputs'/name)
 if stage=='xgboost':copy(ROOT/'05_CRISP_X/models/XGBoost.joblib',run/'inputs/reference_fold0_XGBoost.joblib')
 if stage=='crispx':copy(ROOT/'05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv',run/'inputs/new_parent_manifest.csv')
 if stage=='external':
  for p in (ROOT/'06_EXTERNAL_TRANSFER/data/external_response/cif').glob('*.cif'):copy(p,run/'inputs/cif'/p.name)
 # The input receipt describes this workspace; archived provenance remains separate.
 entries=[dict(path=p.relative_to(run).as_posix(),sha256=sha(p),scope='prepared_workspace_input_identity') for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['INPUT_CODE_SHA256.json','FROZEN_INPUT_CODE.json','WORKSPACE_ORIGIN.json'] and 'results' not in p.relative_to(run).parts]
 for name in ['INPUT_CODE_SHA256.json','FROZEN_INPUT_CODE.json']:
  target=run/name
  if target.exists():
   old=json.loads(target.read_text())
   # Once scientific results are present, prepare must not silently re-freeze them.
   if (run/'results').exists():raise RuntimeError('Results exist; use a new JCTC_WORKSPACE_ROOT')
  target.write_text(json.dumps(entries,indent=2),encoding='utf-8')
 (run/'WORKSPACE_ORIGIN.json').write_text(json.dumps(dict(scope='prepared input layout; not scientific execution authorization',scientific_code_executed=False,historical_manifests='preserved in module configs and provenance',stage=stage),indent=2),encoding='utf-8')
 # LIME identity receipt is produced only by verify_input_identity.py after real checks.
 return str(run)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','et','xgboost','crispx','lime','external'],required=True);a=ap.parse_args()
 stages=['et','xgboost','crispx','lime','external'] if a.stage=='all' else [a.stage]
 print(json.dumps({'workspaces':[prepare(s) for s in stages],'scientific_operations':0},indent=2))
if __name__=='__main__':main()
