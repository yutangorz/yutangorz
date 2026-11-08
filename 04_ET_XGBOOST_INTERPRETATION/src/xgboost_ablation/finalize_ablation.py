
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
_repository_sys.path.insert(0, str(_repo_path('04_ET_XGBOOST_INTERPRETATION/src/xgboost_ablation')))
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run;root=run.parents[2];stage=run.parent
out=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation');out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (run/'SEAL_MANIFEST.json').exists()
for e in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/e['path'])==e['sha256']
assert '9 passed' in (run/'logs/tests.txt').read_text(encoding='utf-8')
d=pd.read_csv(run/'results/PLOT_oof_material_predictions_and_effects.csv',dtype={'ID':str})
assert not d.duplicated(['ID','variant']).any()
pooled=pd.read_csv(run/'results/TABLE_pooled_ablation.csv');single=pd.read_csv(run/'results/TABLE_nine_single_feature_ranking.csv')
assert len(single)==9 and single.complete5fold.all() and (single.n==7176).all()
assert (d[d.variant=='full9'].groupby('formula_group').fold.nunique()==1).all()
full=pooled[pooled.variant=='full9'].iloc[0];negative=json.loads((run/'results/NEGATIVE_CONTROL.json').read_text(encoding='utf-8'))
assert negative['r2']<negative['baseline']['r2'],'negative control should not match actual label training'
hist=pd.read_csv(run/'inputs/historical_model_performance_reference.csv')
histfull=hist[(hist.feature_set=='combined9')&(hist.estimator=='xgboost')].iloc[0]
replay=float(abs(full.mae_eV-histfull.mae_eV));assert replay<1e-5,'same historical XGB ablation full baseline replay'
design=pd.read_csv(run/'results/TABLE_fold_variant_design.csv');groups=design[design.tags.str.contains('training_correlation_component')]
lines=['**XGBoost逐特征及相关组验证完成**','',
 '目标为原数据库带隙，不拟合ALIGNN输出；沿用既有5折化学式分组，固定400树/depth6/learning rate0.04等超参数，不按删除后测试分数调参。删除变量后重新训练，与同折全九特征模型在同一材料上配对比较。',
 f'全九特征OOF：MAE={full.mae_eV:.6f}eV，RMSE={full.rmse_eV:.6f}eV，R2={full.r2:.6f}；与历史同配置MAE差值{replay:.3g}eV。',
 f'标签置乱负对照fold0 R2={negative["r2"]:.6f}，真实目标训练fold0 R2={negative["baseline"]["r2"]:.6f}。',
 '', '**九个单特征删除结果**','', '|删除特征|MAE增加/eV|95%区间|排名|','|---|---:|---|---:|']
for r in single.sort_values('rank').itertuples():lines.append(f'|{r.feature}|{r.paired_MAE_increase_eV:.6f}|[{r.low95_eV:.6f}, {r.high95_eV:.6f}]|{r.rank:g}|')
lines+=['','正值表示删除后预测变差；负值表示固定模型条件下删除后略有改善，不能删除或截断这些结果。置信区间是固定折模型下化学式聚类bootstrap的逐项区间，没有多重校正，不据此宣布九个独立物理发现。',
 '', '**相关组与组别消融**',
 '除七原子/两结构语义组外，每折仅用该折训练特征构造|Pearson|>=0.8相关图，删除大小至少2的连通分量后重训。连通分量包含间接相关，不要求每对变量均超过阈值。组定义保留每折训练ID、相关矩阵与边。',
 '相同删除集合若覆盖全部5折才报告完整7176的OOF置信区间；仅部分折出现的相关组不能伪装成全数据OOF排名。组删除与成员单删之和不同，源于冗余、互补及重训，不直接命名为物理协同或SHAP交互。',
 '', '**与特征提取的关系及自我审查**',
 '本阶段检验描述符信息在另一预测模型中的条件预测价值，是模型层面交叉检查，不能代替DFT物理验证。XGB重训消融、ET模型SHAP全局归因、局部LIME/SHAP响应分配不是相同统计量或相同模型，不能仅拼接排名就宣称强一致。',
 '使用既定超参数而非每次删除后重新网格搜索，以控制成本和模型规格；结论是此固定模型家族下的删除重训效应，不能据此宣称被删变量在所有模型中无用。',
 '单特征消融受相关性影响，弱单删效应不等于物理不重要。组删帮助识别冗余，但不能唯一分配七元素属性的独立物理作用。分组依据不读取验证折特征或标签，避免测试集选组。',
 '数据集限定于当前7176候选，不覆盖外部材料族；没有新增DFT、结构弛豫或因果干预。所有负结果和区间跨零的项目保留。',
 '', '**论文落点与下一步**',
 '正文如需支撑结构增益，可简述两结构联合删除效应；SI放九项单删、相关组、区间、数据划分、负对照和全部参数。只有核对真实DFT证据后才能讨论其与模型排序的物理对应。当前不执行DFT，不宣称四方法强一致。']
(run/'reports/XGBOOST_VALIDATION_REPORT_CN.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
dump(run/'reports/FIGURE_TABLE_DATA_INDEX.json',dict(input_and_parameters=['config.json','ENVIRONMENT.json','inputs/model_input_7176.csv','inputs/outer_fold_manifest.csv'],
    plot_and_table_data=[str(p.relative_to(run)) for p in sorted((run/'results').rglob('*.csv'))]))
result=dict(status='paired_original_target_ablation_completed',full_model=dict(mae_eV=float(full.mae_eV),rmse_eV=float(full.rmse_eV),r2=float(full.r2)),
    rows=7176,single_feature_count=9,model_fits=int(len(design)+1),negative_control=negative,
    correlation_component_definitions=groups[['fold','removed_features']].to_dict(orient='records'),DFT_not_executed=True,
    no_strong_four_method_agreement_claim=True)
dump(run/'results/XGBOOST_VERDICT.json',result);dump(run/'EXECUTION_COMPLETED.json',result)
dump(run/'SEAL_MANIFEST.json',[dict(path=str(p.relative_to(run)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(run.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='SEAL_MANIFEST.json'])
b=root/'docs/development_state/xgboost_run001_final';b.mkdir(exist_ok=False)
for p in [root/'PROJECT_PROGRESS.md',stage/'RUN_INDEX.json',stage/'STAGE_STATUS_CN.md']:shutil.copy2(p,b/p.name)
msg=f'特征提取run021已封存。03_xgboost_validation/run001完成：9单特征、七原子/两结构语义组、折内相关组配对删除重训；7176材料/5折原带隙目标，完整模型MAE{full.mae_eV:.6f} R2{full.r2:.6f}。逐ID、参数、全部分组、模型、bootstrap及负结果保留。DFT暂缓，不能宣称四方法物理强一致。下一本阶段run002，特征阶段若改则run022。'
p=root/'PROJECT_PROGRESS.md';p.write_text('# XGBoost验证run001已完成\n\n'+msg+'\n\n运行环境/命令/配置与图表数据索引见本版ENVIRONMENT.json、RUN_COMMANDS.md、config.json和reports/FIGURE_TABLE_DATA_INDEX.json；无Git仓库，旧版本不改。模型目标为原数据库带隙，不能冒称ALIGNN独立物理验证。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
idx=json.loads((stage/'RUN_INDEX.json').read_text(encoding='utf-8'));idx['runs'][0]['status']='sealed_paired_ablation_completed';dump(stage/'RUN_INDEX.json',idx)
(stage/'STAGE_STATUS_CN.md').write_text(msg,encoding='utf-8')
delivery=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary');delivery.mkdir(exist_ok=False);shutil.copytree(run/'reports',_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary/reports'))
for p in (run/'results').rglob('*.csv'):
    dest=_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary/results')/p.relative_to(run/'results');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for n in ['config.json','ENVIRONMENT.json','RUN_COMMANDS.md','SEAL_MANIFEST.json']:shutil.copy2(run/n,delivery/n)
shutil.copy2(run/'results/XGBOOST_VERDICT.json',_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary/XGBOOST_VERDICT.json'))
with zipfile.ZipFile(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary_code_inputs_results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in run.rglob('*'):
        if p.is_file() and p.suffix!='.joblib' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:z.write(p,str(p.relative_to(run)))
dump(_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/xgboost_summary/MODEL_LOCATIONS.json'),[dict(path=str(p),sha256=sha(p)) for p in (run/'results/models').glob('*.joblib')])
shutil.copy2(root/'PROJECT_PROGRESS.md',_repo_path('outputs/04_ET_XGBOOST_INTERPRETATION/xgboost_ablation/PROJECT_PROGRESS_LATEST.md'))
print('XGBOOST SEALED',json.dumps(result,ensure_ascii=False),flush=True)
