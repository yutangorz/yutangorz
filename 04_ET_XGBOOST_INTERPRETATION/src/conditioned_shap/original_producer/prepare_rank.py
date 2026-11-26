from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.metadata,json,shutil,sys
parser=argparse.ArgumentParser(description='Prepare a contrast workspace from a declared source workspace')
parser.add_argument('--source-workspace',type=Path,required=True)
parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args();old=args.source_workspace.resolve();run=args.run.resolve();src=Path(__file__).parent
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not run.exists()
for p in src.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
run.mkdir()
for n in ['code','inputs','logs','reports']:(run/n).mkdir()
for n in ['rank_core.py','study_rank.py','test_rank.py','finalize_rank.py','prepare_rank.py']:
    source=src/n if (src/n).is_file() else old/'code'/n
    if not source.is_file():raise FileNotFoundError('Declared source workspace dependency missing: '+str(source))
    shutil.copy2(source,run/'code'/n)
for n in ['shap_baseline','lime_baseline','shared_features']:
    shutil.copytree(old/'code'/n,run/'code'/n,ignore=shutil.ignore_patterns('__pycache__'))
import pandas as pd
for n in ['property_model.joblib','model_input_7176.csv','outer_fold_manifest.csv','new_parent_manifest.csv']:shutil.copy2(old/'inputs'/n,run/'inputs'/n)
for n in ['per_material_nine_attributions.csv','TABLE_nine_importance.csv','training_feature_correlations.csv']:shutil.copy2(old/'results/shap_baseline'/n,run/'inputs'/n)
for n in ['lime_weights_and_origin.csv','response_predictions.csv']:shutil.copy2(old/'results'/n,run/'inputs'/n)
(run/'inputs/pools').mkdir();manifest=[];sourcehash=[]
for ID in pd.read_csv(run/'inputs/new_parent_manifest.csv',dtype={'ID':str}).ID:
    for seed in [307,419]:
        folder=run/'inputs/pools'/f'{ID}_{seed}';folder.mkdir()
        for n in ['pool.csv','indices.npz','events.json']:
            source=old/'results/physical_pools'/f'{ID}_{seed}'/n;shutil.copy2(source,folder/n)
            sourcehash.append(dict(source=str(source),path=str((folder/n).relative_to(run)),sha256=sha(source)))
        manifest.append(dict(ID=ID,seed=seed))
dump(run/'inputs/pool_manifest.json',manifest);dump(run/'inputs/SOURCE_HASHES.json',sourcehash)
cfg=dict(parent_run='run_019',model='frozen fold0 ExtraTrees, original database gap target',
    comparison='same24 parents x2 seeds x48 physical holdouts; contrast parent reference, nine eV contributions',
    local_to_global='mean absolute across responses withinseed; average seeds withinparent; equal average parents',
    shap_contrast='phi_j(x_perturbed)-phi_j(x_parent), same tree_path_dependent reference',
    lime_contrast='beta_j per trainingSD * actual (xpert-xparent)/trainingSD',
    importance_scope='local-response allocation contrast, not population global prediction attribution or causal effect',
    bootstrap=2000,seed=20261030,LIME_qualification=dict(min_r2=.8,max_mae_eV=.05),
    no_selective_drop_of_low_fidelity_parents=True,no_tuning_or_new_model_training=True,
    no_CRISP_independent_SHAP_rank=True,DFT_deferred=True,next_stage='03_xgboost_validation/run_001 only after feature stage sealed')
dump(run/'config.json',cfg);dump(run/'PROTOCOL_FROZEN.json',dict(time=datetime.datetime.now().isoformat(),before_new_contrast_SHAP=True))
dump(run/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','shap','joblib','pytest']}))
delivery=run.parent/(run.name+'_delivery')
(run/'RUN_COMMANDS.md').write_text(f'python -m pytest code/test_rank.py -q\npython code/study_rank.py --run "{run}"\npython code/finalize_rank.py --run "{run}" --delivery "{delivery}"\nSet PYTHONPATH to code for pytest. Never overwrite executed results.\n',encoding='utf-8')
dump(run/'FROZEN_INPUT_CODE.json',[dict(path=str(p.relative_to(run)),sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file()])
print('Contrast workspace prepared with source hashes; scientific execution is separate',flush=True)
