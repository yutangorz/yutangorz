import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--delivery',type=Path,required=True);args=ap.parse_args();run=args.run.resolve();out=args.delivery.resolve();out.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (run/'SEAL_MANIFEST.json').exists()
for r in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/r['path'])==r['sha256']
assert '10 passed' in (run/'logs/tests.txt').read_text(encoding='utf-8')
r=pd.read_csv(run/'results/aligned_comparison/per_response_nine_contrasts.csv',dtype={'ID':str})
assert len(r)==2304*3 and not r.duplicated(['ID','seed','candidate','method']).any() and r.ID.nunique()==24
q=pd.read_csv(run/'results/aligned_comparison/TABLE_response_fidelity.csv');rank=pd.read_csv(run/'results/aligned_comparison/TABLE_aligned_rankings.csv');corr=pd.read_csv(run/'results/aligned_comparison/TABLE_rank_agreement.csv')
assert len(rank)==81 and rank[['score_eV','share','rank']].notna().all().all()
q.to_csv(run/'reports/TABLE_fidelity.csv',index=False)
allcorr=corr[(corr.scope=='all')&(corr.feature_scope=='nine')]
assert np.isfinite(allcorr.spearman).all()
result=dict(status='feature_extraction_closed_with_explicit_LIME_limits',
    completed=True,rank_comparison='same-model same-parent same-response contrasts; different attribution mechanisms remain',
    methods=['SHAP_contrast','LIME_standard512_contrast','LIME_constrained9_contrast'],
    native_global_SHAP_preserved=True,CRISP_not_an_independent_nine_rank=True,
    no_DFT_or_four_method_strong_agreement_claim=True,
    next_stage='03_xgboost_validation/run_001',new_XGBoost_training_target='original database band gap; not ALIGNN outputs')
dump(run/'results/FEATURE_STAGE_VERDICT.json',result)
lines=['**LIME与SHAP排序口径统一完成，特征提取阶段封存**','',
 '同一冻结ExtraTrees、同24母体/2种子/2304物理探测响应，统一以母体到扰动的九维贡献变化/eV比较。LIME为beta乘实际标准化位移，SHAP为同一参考下phi扰动减phi母体。先各响应绝对值平均，再种子内/母体内聚合，最后等权平均母体。2000次母体配对bootstrap。',
 '这是局部响应分配比较，不是把全局SHAP与局部LIME改成同一因果估计；原1462全局SHAP保留，不因本次局部排序变化而替换。',
 '', '**全九特征排序相关，仅对应本次局部响应任务**']
for a in allcorr.itertuples():lines.append(f'{a.method_a} vs {a.method_b}，Spearman={a.spearman:.4f}，母体bootstrap95%[{a.spearman_low95:.4f},{a.spearman_high95:.4f}]，Kendall={a.kendall_tau:.4f}。')
lines+=['','**保真度与采用边界**']
for a in q[q.scope=='all'].itertuples():lines.append(f'{a.method}：MAE={a.mae_eV:.6f}eV，R2={a.r2:.6f}，状态{a.interpretation_status}。')
lines+=['', 'LIME低保真结果可以用于说明方法差异和局限，不能作为物理九排名的独立确认。SHAP对模型变化严格加和是其定义检查，不是DFT物理真值证明。',
 '几何子集的原子坐标属性没有变，SHAP原子贡献仍可能因交互分配而改变，不能读作原子性质实际变化。相关原子特征的局部LIME系数也未被本次排序统一消除不可辨识性。',
 '未对仅两个结构特征计算可误导的秩相关一致性；九特征和七原子分别报告，包含重复/零分数的并列排名。未剔除保真度低的母体来美化相关性。CI只涵盖母体采样，固定模型和探测生成器。',
 '正文建议保留原全局SHAP九重要性、模型角色和必要限制。SI放本次统一任务定义、排名/相关区间、完整保真度、相关矩阵和失败案例。不能将CRISP内的同一SHAP排名再当作独立方法；不宣称四方法强一致。',
 '代码/分析部分已完成并停止继续调LIME。论文正文、SI图表和回稿信尚需正式编辑排版。下一阶段开始原数据库带隙目标的XGBoost逐特征及训练折内相关组消融。DFT仍暂缓。']
(run/'reports/RANK_ALIGNMENT_REPORT_CN.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
dump(run/'reports/FIGURE_TABLE_DATA_INDEX.json',dict(input_parameters=['config.json','inputs/model_input_7176.csv','inputs/pool_manifest.json','inputs/lime_weights_and_origin.csv'],
    figure_and_table_data=[str(p.relative_to(run)) for p in sorted((run/'results').rglob('*.csv'))]))
dump(run/'EXECUTION_COMPLETED.json',result)
dump(run/'SEAL_MANIFEST.json',[dict(path=str(p.relative_to(run)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(run.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='SEAL_MANIFEST.json'])
delivery=out/'feature_run021';delivery.mkdir(exist_ok=False);shutil.copytree(run/'reports',delivery/'reports');shutil.copytree(run/'results',delivery/'results')
for n in ['config.json','ENVIRONMENT.json','RUN_COMMANDS.md','SEAL_MANIFEST.json']:shutil.copy2(run/n,delivery/n)
with zipfile.ZipFile(out/'feature_run021_code_inputs_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in run.rglob('*'):
        if p.is_file() and p.suffix!='.joblib' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:z.write(p,str(p.relative_to(run)))
dump(delivery/'MODEL_LOCATION.json',dict(path=str(run/'inputs/property_model.joblib'),sha256=sha(run/'inputs/property_model.joblib')))
print('FEATURE STAGE SEALED',allcorr.to_json(orient='records'),flush=True)
